//! Compilation du graphe tensoriel canonique.
//!
//! Cette première passe valide le graphe, élimine les nœuds inaccessibles
//! depuis les sorties et produit un plan topologique indépendant du backend.

use core::fmt;

use scirust_tensor_ir::{Graph, GraphError, NodeId, Operation, TensorType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    InvalidGraph(GraphError),
    NoOutputs,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidGraph(error) => write!(formatter, "invalid tensor graph: {error}"),
            Self::NoOutputs => formatter.write_str("tensor graph has no declared outputs"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<GraphError> for CompileError {
    fn from(error: GraphError) -> Self {
        Self::InvalidGraph(error)
    }
}

/// Une instruction conservant les identifiants du graphe canonique.
///
/// Elle ne contient ni stockage concret, ni kernel, ni état de backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub id: NodeId,
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub output: TensorType,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompileStats {
    pub original_nodes: usize,
    pub retained_nodes: usize,
    pub eliminated_nodes: usize,
}

/// Plan topologique immuable prêt pour les passes de lowering suivantes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    instructions: Vec<Instruction>,
    outputs: Vec<NodeId>,
    stats: CompileStats,
}

impl ExecutionPlan {
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub const fn stats(&self) -> CompileStats {
        self.stats
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CanonicalCompiler;

impl CanonicalCompiler {
    pub const fn new() -> Self {
        Self
    }

    pub fn compile(self, graph: &Graph) -> Result<ExecutionPlan, CompileError> {
        graph.validate()?;

        if graph.outputs().is_empty()
        {
            return Err(CompileError::NoOutputs);
        }

        let mut reachable = vec![false; graph.nodes().len()];
        let mut pending = graph.outputs().to_vec();

        while let Some(id) = pending.pop()
        {
            let index = id.get() as usize;

            if reachable[index]
            {
                continue;
            }

            reachable[index] = true;
            pending.extend(graph.nodes()[index].inputs.iter().copied());
        }

        let instructions = graph
            .nodes()
            .iter()
            .enumerate()
            .filter(|(index, _)| reachable[*index])
            .map(|(index, node)| Instruction {
                id: NodeId::new(index as u32),
                operation: node.operation.clone(),
                inputs: node.inputs.clone(),
                output: node.output.clone(),
            })
            .collect::<Vec<_>>();

        let original_nodes = graph.nodes().len();
        let retained_nodes = instructions.len();

        Ok(ExecutionPlan {
            instructions,
            outputs: graph.outputs().to_vec(),
            stats: CompileStats {
                original_nodes,
                retained_nodes,
                eliminated_nodes: original_nodes - retained_nodes,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{ConstantId, DType, Shape};

    fn matrix_type() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![2, 2]))
    }

    #[test]
    fn compilation_preserves_topological_identifiers() {
        let mut graph = Graph::new();

        let lhs = graph.add_input("lhs", matrix_type()).unwrap();
        let rhs = graph.add_input("rhs", matrix_type()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![lhs, rhs], matrix_type())
            .unwrap();

        graph.set_outputs(vec![sum]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();

        assert_eq!(
            plan.instructions()
                .iter()
                .map(|instruction| instruction.id)
                .collect::<Vec<_>>(),
            vec![lhs, rhs, sum]
        );
        assert_eq!(plan.outputs(), &[sum]);
        assert_eq!(plan.stats().eliminated_nodes, 0);
    }

    #[test]
    fn compilation_eliminates_unreachable_nodes() {
        let mut graph = Graph::new();

        let input = graph.add_input("input", matrix_type()).unwrap();
        let live = graph
            .add_node(Operation::Relu, vec![input], matrix_type())
            .unwrap();

        let unused_constant = graph
            .add_constant(ConstantId::new(7), matrix_type())
            .unwrap();
        let _unused = graph
            .add_node(Operation::Exp, vec![unused_constant], matrix_type())
            .unwrap();

        graph.set_outputs(vec![live]).unwrap();

        let plan = CanonicalCompiler::new().compile(&graph).unwrap();

        assert_eq!(plan.instructions().len(), 2);
        assert_eq!(
            plan.stats(),
            CompileStats {
                original_nodes: 4,
                retained_nodes: 2,
                eliminated_nodes: 2,
            }
        );
    }

    #[test]
    fn compilation_requires_declared_outputs() {
        let mut graph = Graph::new();
        graph.add_input("input", matrix_type()).unwrap();

        assert_eq!(
            CanonicalCompiler::new().compile(&graph),
            Err(CompileError::NoOutputs)
        );
    }
}
