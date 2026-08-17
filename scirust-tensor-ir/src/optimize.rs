use alloc::{vec, vec::Vec};
use core::fmt;

use crate::{Graph, GraphError, NodeId, Operation, SemanticError, TensorType, validate_semantics};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizationConfig {
    pub dead_code_elimination: bool,
    pub common_subexpression_elimination: bool,
    pub algebraic_simplification: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            dead_code_elimination: true,
            common_subexpression_elimination: true,
            algebraic_simplification: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptimizationStats {
    pub original_nodes: usize,
    pub retained_nodes: usize,
    pub eliminated_dead_nodes: usize,
    pub common_subexpressions: usize,
    pub simplified_nodes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizedGraph {
    pub graph: Graph,
    pub old_to_new: Vec<Option<NodeId>>,
    pub stats: OptimizationStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationError {
    InvalidGraph(GraphError),
    InvalidSemantics(SemanticError),
    MissingMappedInput(NodeId),
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidGraph(error) => write!(f, "cannot optimize invalid graph: {error}"),
            Self::InvalidSemantics(error) =>
            {
                write!(f, "cannot optimize semantically invalid graph: {error}")
            },
            Self::MissingMappedInput(node) =>
            {
                write!(f, "optimizer lost mapping for node {}", node.get())
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OptimizationError {}

impl From<GraphError> for OptimizationError {
    fn from(value: GraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

/// Deterministic backend-neutral canonical graph optimizer.
///
/// Constant tensor payloads intentionally live outside `Graph`, so payload
/// constant folding belongs to the runtime/compiler layer. This pass performs
/// every optimization that can be proven from IR structure alone: DCE, exact
/// CSE, identity/view simplification and scalar-attribute simplification.
pub fn optimize_graph(
    source: &Graph,
    config: OptimizationConfig,
) -> Result<OptimizedGraph, OptimizationError> {
    source.validate()?;
    validate_semantics(source).map_err(OptimizationError::InvalidSemantics)?;

    let reachable = if config.dead_code_elimination
    {
        reachable_nodes(source)
    }
    else
    {
        vec![true; source.nodes().len()]
    };

    let mut graph = Graph::new();
    let mut mapping = vec![None; source.nodes().len()];
    let mut stats = OptimizationStats {
        original_nodes: source.nodes().len(),
        ..OptimizationStats::default()
    };

    for (index, node) in source.nodes().iter().enumerate()
    {
        if !reachable[index]
        {
            stats.eliminated_dead_nodes += 1;
            continue;
        }

        let mapped_inputs = node
            .inputs
            .iter()
            .map(|&input| {
                mapping[input.get() as usize].ok_or(OptimizationError::MissingMappedInput(input))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if config.algebraic_simplification
        {
            if let Some(alias) =
                simplify_alias(&graph, &node.operation, &mapped_inputs, &node.output)
            {
                mapping[index] = Some(alias);
                stats.simplified_nodes += 1;
                continue;
            }
        }

        let mut operation = node.operation.clone();
        if config.algebraic_simplification
        {
            if let Some(folded) = simplify_operation(&operation)
            {
                operation = folded;
                stats.simplified_nodes += 1;
            }
        }

        if config.common_subexpression_elimination && cse_eligible(&operation)
        {
            if let Some(existing) =
                find_equivalent(&graph, &operation, &mapped_inputs, &node.output)
            {
                mapping[index] = Some(existing);
                stats.common_subexpressions += 1;
                continue;
            }
        }

        let new_id = graph.add_node(operation, mapped_inputs, node.output.clone())?;
        mapping[index] = Some(new_id);
    }

    let outputs = source
        .outputs()
        .iter()
        .map(|&output| {
            mapping[output.get() as usize].ok_or(OptimizationError::MissingMappedInput(output))
        })
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_outputs(outputs)?;
    stats.retained_nodes = graph.nodes().len();

    Ok(OptimizedGraph {
        graph,
        old_to_new: mapping,
        stats,
    })
}

fn reachable_nodes(graph: &Graph) -> Vec<bool> {
    let mut reachable = vec![false; graph.nodes().len()];
    let mut pending = graph.outputs().to_vec();
    while let Some(node) = pending.pop()
    {
        let index = node.get() as usize;
        if reachable[index]
        {
            continue;
        }
        reachable[index] = true;
        pending.extend(graph.nodes()[index].inputs.iter().copied());
    }
    reachable
}

fn simplify_alias(
    graph: &Graph,
    operation: &Operation,
    inputs: &[NodeId],
    output: &TensorType,
) -> Option<NodeId> {
    let &input = inputs.first()?;
    let input_type = &graph.nodes()[input.get() as usize].output;
    match operation
    {
        Operation::Scale { factor } if factor.as_f32() == Some(1.0) => Some(input),
        Operation::Reshape { shape }
            if input_type.dtype == output.dtype && input_type.shape == *shape =>
        {
            Some(input)
        },
        Operation::Transpose { permutation }
            if permutation.iter().copied().eq(0..permutation.len()) =>
        {
            Some(input)
        },
        Operation::BroadcastTo { shape }
            if input_type.dtype == output.dtype && input_type.shape == *shape =>
        {
            Some(input)
        },
        Operation::ReduceSumTo { shape }
            if input_type.dtype == output.dtype && input_type.shape == *shape =>
        {
            Some(input)
        },
        _ => None,
    }
}

/// Only folds transformations that do not require changing the input edge.
/// Nested-scale folding deliberately stays out of this helper: replacing
/// `scale(scale(x, a), b)` by `scale(scale(x, a), a*b)` would apply `a` twice
/// unless the parent edge is rewritten at the same time.
fn simplify_operation(operation: &Operation) -> Option<Operation> {
    match operation
    {
        Operation::Scale { factor } if factor.as_f32() == Some(0.0) => Some(Operation::ZerosLike),
        _ => None,
    }
}

fn cse_eligible(operation: &Operation) -> bool {
    !matches!(
        operation,
        Operation::Input { .. }
            | Operation::Constant { .. }
            | Operation::StopGradient
            | Operation::Checkpoint
    )
}

fn find_equivalent(
    graph: &Graph,
    operation: &Operation,
    inputs: &[NodeId],
    output: &TensorType,
) -> Option<NodeId> {
    graph
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| {
            &node.operation == operation && node.inputs == inputs && &node.output == output
        })
        .map(|(index, _)| NodeId::new(index as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    fn ty() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[test]
    fn removes_dead_code_and_exact_common_subexpressions() {
        let mut graph = Graph::new();
        let a = graph.add_input("a", ty()).unwrap();
        let b = graph.add_input("b", ty()).unwrap();
        let first = graph.add_node(Operation::Add, vec![a, b], ty()).unwrap();
        let second = graph.add_node(Operation::Add, vec![a, b], ty()).unwrap();
        let live = graph
            .add_node(Operation::Mul, vec![first, second], ty())
            .unwrap();
        let _dead = graph.add_node(Operation::Relu, vec![a], ty()).unwrap();
        graph.set_outputs(vec![live]).unwrap();

        let optimized = optimize_graph(&graph, OptimizationConfig::default()).unwrap();
        assert_eq!(optimized.stats.eliminated_dead_nodes, 1);
        assert_eq!(optimized.stats.common_subexpressions, 1);
        assert!(optimized.graph.nodes().len() < graph.nodes().len());
    }

    #[test]
    fn simplifies_identity_scale_and_reshape() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", ty()).unwrap();
        let scale = graph
            .add_node(
                Operation::Scale {
                    factor: crate::Scalar::f32(1.0),
                },
                vec![x],
                ty(),
            )
            .unwrap();
        let reshape = graph
            .add_node(
                Operation::Reshape {
                    shape: Shape::new(vec![4]),
                },
                vec![scale],
                ty(),
            )
            .unwrap();
        graph.set_outputs(vec![reshape]).unwrap();

        let optimized = optimize_graph(&graph, OptimizationConfig::default()).unwrap();
        assert_eq!(optimized.stats.simplified_nodes, 2);
        assert_eq!(optimized.graph.nodes().len(), 1);
        assert_eq!(optimized.graph.outputs(), &[NodeId::new(0)]);
    }

    #[test]
    fn scale_zero_becomes_zeros_like_without_changing_edge() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", ty()).unwrap();
        let zero = graph
            .add_node(
                Operation::Scale {
                    factor: crate::Scalar::f32(0.0),
                },
                vec![x],
                ty(),
            )
            .unwrap();
        graph.set_outputs(vec![zero]).unwrap();
        let optimized = optimize_graph(&graph, OptimizationConfig::default()).unwrap();
        assert!(matches!(
            optimized.graph.nodes()[1].operation,
            Operation::ZerosLike
        ));
    }
}
