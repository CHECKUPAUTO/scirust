use alloc::{format, vec, vec::Vec};
use core::fmt;

use scirust_compute::{DType, Shape};

use crate::{
    Graph, GraphError, Node, NodeId, Operation, SemanticError, TensorType, validate_semantics,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutodiffError {
    InvalidGraph(GraphError),
    InvalidSemantics(SemanticError),
    InvalidNode(NodeId),
    UnsupportedDType { node: NodeId, dtype: DType },
    UnsupportedOperation { node: NodeId, operation: Operation },
    MatMulRequiresRank2 { node: NodeId },
    BatchMatMulRequiresRankAtLeast3 { node: NodeId },
}

impl fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid graph during autodiff: {error}"),
            Self::InvalidSemantics(error) => write!(f, "invalid tensor semantics during autodiff: {error}"),
            Self::InvalidNode(node) => write!(f, "autodiff references unknown node {}", node.get()),
            Self::UnsupportedDType { node, dtype } => write!(
                f,
                "canonical autodiff node {} uses unsupported dtype {dtype:?}; Core2 scalar derivative attributes are currently F32",
                node.get()
            ),
            Self::UnsupportedOperation { node, operation } => write!(
                f,
                "autodiff does not define a rule for node {} operation {operation:?}",
                node.get()
            ),
            Self::MatMulRequiresRank2 { node } => write!(
                f,
                "autodiff requires rank-2 MatMul at node {}",
                node.get()
            ),
            Self::BatchMatMulRequiresRankAtLeast3 { node } => write!(
                f,
                "autodiff requires BatchMatMul rank >= 3 at node {}",
                node.get()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AutodiffError {}

impl From<GraphError> for AutodiffError {
    fn from(value: GraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

/// Result of reverse-mode transformation with an explicit cotangent input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VjpGraph {
    pub graph: Graph,
    pub primal_output: NodeId,
    pub cotangent_input: NodeId,
    pub gradients: Vec<NodeId>,
}

/// Result of `grad`; an all-one cotangent is generated in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradGraph {
    pub graph: Graph,
    pub primal_output: NodeId,
    pub gradients: Vec<NodeId>,
}

/// Result of forward-mode transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JvpGraph {
    pub graph: Graph,
    pub primal_output: NodeId,
    pub tangent_output: NodeId,
    pub tangent_inputs: Vec<NodeId>,
}

/// Reverse-mode vector-Jacobian product.
pub fn vjp(source: &Graph, output: NodeId, wrt: &[NodeId]) -> Result<VjpGraph, AutodiffError> {
    validate_for_autodiff(source, output, wrt)?;
    let (mut graph, primal) = copy_primal(source)?;
    let primal_output = mapped(&primal, output)?;
    let cotangent_input = graph.add_input(
        "__scirust_cotangent",
        node_type(source, output)?.clone(),
    )?;
    let gradients = reverse_accumulate(source, &mut graph, &primal, output, cotangent_input, wrt)?;
    graph.set_outputs(gradients.clone())?;
    Ok(VjpGraph {
        graph,
        primal_output,
        cotangent_input,
        gradients,
    })
}

/// Reverse-mode gradient using an all-one seed.
///
/// For a non-scalar output this is the gradient of the sum of all output
/// elements, equivalently a VJP with an all-one cotangent.
pub fn grad(source: &Graph, output: NodeId, wrt: &[NodeId]) -> Result<GradGraph, AutodiffError> {
    validate_for_autodiff(source, output, wrt)?;
    let (mut graph, primal) = copy_primal(source)?;
    let primal_output = mapped(&primal, output)?;
    let seed = graph.add_node(
        Operation::OnesLike,
        vec![primal_output],
        node_type(source, output)?.clone(),
    )?;
    let gradients = reverse_accumulate(source, &mut graph, &primal, output, seed, wrt)?;
    graph.set_outputs(gradients.clone())?;
    Ok(GradGraph {
        graph,
        primal_output,
        gradients,
    })
}

/// Return the primal value followed by its reverse-mode gradients.
pub fn value_and_grad(
    source: &Graph,
    output: NodeId,
    wrt: &[NodeId],
) -> Result<GradGraph, AutodiffError> {
    let mut transformed = grad(source, output, wrt)?;
    let mut outputs = Vec::with_capacity(1 + transformed.gradients.len());
    outputs.push(transformed.primal_output);
    outputs.extend(transformed.gradients.iter().copied());
    transformed.graph.set_outputs(outputs)?;
    Ok(transformed)
}

/// Forward-mode Jacobian-vector product.
///
/// One tangent input is created for every node in `wrt`. The transformed graph
/// returns `[primal_output, tangent_output]`.
pub fn jvp(source: &Graph, output: NodeId, wrt: &[NodeId]) -> Result<JvpGraph, AutodiffError> {
    validate_for_autodiff(source, output, wrt)?;
    let (mut graph, primal) = copy_primal(source)?;
    let mut tangents = vec![None; source.nodes().len()];
    let mut tangent_inputs = Vec::with_capacity(wrt.len());

    for &node in wrt {
        let tangent = graph.add_input(
            format!("__scirust_tangent_{}", node.get()),
            node_type(source, node)?.clone(),
        )?;
        tangents[node.get() as usize] = Some(tangent);
        tangent_inputs.push(tangent);
    }

    for (index, node) in source.nodes().iter().enumerate() {
        if tangents[index].is_some() {
            continue;
        }
        let id = NodeId::new(index as u32);
        tangents[index] = forward_rule(source, &mut graph, &primal, &tangents, id, node)?;
    }

    let primal_output = mapped(&primal, output)?;
    let tangent_output = match tangents[output.get() as usize] {
        Some(id) => id,
        None => graph.add_node(
            Operation::ZerosLike,
            vec![primal_output],
            node_type(source, output)?.clone(),
        )?,
    };
    graph.set_outputs(vec![primal_output, tangent_output])?;
    Ok(JvpGraph {
        graph,
        primal_output,
        tangent_output,
        tangent_inputs,
    })
}

fn validate_for_autodiff(
    source: &Graph,
    output: NodeId,
    wrt: &[NodeId],
) -> Result<(), AutodiffError> {
    source.validate()?;
    validate_semantics(source).map_err(AutodiffError::InvalidSemantics)?;
    ensure_supported_dtype(source, output)?;
    for &node in wrt {
        ensure_supported_dtype(source, node)?;
    }
    Ok(())
}

fn reverse_accumulate(
    source: &Graph,
    graph: &mut Graph,
    primal: &[NodeId],
    output: NodeId,
    seed: NodeId,
    wrt: &[NodeId],
) -> Result<Vec<NodeId>, AutodiffError> {
    let mut adjoints = vec![None; source.nodes().len()];
    adjoints[output.get() as usize] = Some(seed);

    for index in (0..source.nodes().len()).rev() {
        let Some(cotangent) = adjoints[index] else {
            continue;
        };
        let id = NodeId::new(index as u32);
        reverse_rule(
            source,
            graph,
            primal,
            &mut adjoints,
            id,
            &source.nodes()[index],
            cotangent,
        )?;
    }

    let mut gradients = Vec::with_capacity(wrt.len());
    for &node in wrt {
        let gradient = match adjoints[node.get() as usize] {
            Some(id) => id,
            None => graph.add_node(
                Operation::ZerosLike,
                vec![mapped(primal, node)?],
                node_type(source, node)?.clone(),
            )?,
        };
        gradients.push(gradient);
    }
    Ok(gradients)
}

fn reverse_rule(
    source: &Graph,
    graph: &mut Graph,
    primal: &[NodeId],
    adjoints: &mut [Option<NodeId>],
    id: NodeId,
    node: &Node,
    cotangent: NodeId,
) -> Result<(), AutodiffError> {
    let p = |node_id| mapped(primal, node_id);
    let ty = |node_id| node_type(source, node_id).cloned();

    match &node.operation {
        Operation::Input { .. } | Operation::Constant { .. } => {}
        Operation::Add => {
            accumulate(graph, adjoints, node.inputs[0], cotangent, ty(node.inputs[0])?)?;
            accumulate(graph, adjoints, node.inputs[1], cotangent, ty(node.inputs[1])?)?;
        }
        Operation::Sub => {
            accumulate(graph, adjoints, node.inputs[0], cotangent, ty(node.inputs[0])?)?;
            let neg = graph.add_node(
                Operation::Scale {
                    factor: crate::Scalar::f32(-1.0),
                },
                vec![cotangent],
                ty(node.inputs[1])?,
            )?;
            accumulate(graph, adjoints, node.inputs[1], neg, ty(node.inputs[1])?)?;
        }
        Operation::Mul => {
            let left = graph.add_node(
                Operation::Mul,
                vec![cotangent, p(node.inputs[1])?],
                ty(node.inputs[0])?,
            )?;
            let right = graph.add_node(
                Operation::Mul,
                vec![cotangent, p(node.inputs[0])?],
                ty(node.inputs[1])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], left, ty(node.inputs[0])?)?;
            accumulate(graph, adjoints, node.inputs[1], right, ty(node.inputs[1])?)?;
        }
        Operation::Div => {
            let left = graph.add_node(
                Operation::Div,
                vec![cotangent, p(node.inputs[1])?],
                ty(node.inputs[0])?,
            )?;
            let denominator = graph.add_node(
                Operation::Mul,
                vec![p(node.inputs[1])?, p(node.inputs[1])?],
                ty(node.inputs[1])?,
            )?;
            let numerator = graph.add_node(
                Operation::Mul,
                vec![cotangent, p(node.inputs[0])?],
                ty(node.inputs[1])?,
            )?;
            let quotient = graph.add_node(
                Operation::Div,
                vec![numerator, denominator],
                ty(node.inputs[1])?,
            )?;
            let right = graph.add_node(
                Operation::Scale {
                    factor: crate::Scalar::f32(-1.0),
                },
                vec![quotient],
                ty(node.inputs[1])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], left, ty(node.inputs[0])?)?;
            accumulate(graph, adjoints, node.inputs[1], right, ty(node.inputs[1])?)?;
        }
        Operation::Scale { factor } => {
            let contribution = graph.add_node(
                Operation::Scale { factor: *factor },
                vec![cotangent],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::Relu => {
            let contribution = graph.add_node(
                Operation::ReluGrad,
                vec![p(node.inputs[0])?, cotangent],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::Exp => {
            let contribution = graph.add_node(
                Operation::Mul,
                vec![cotangent, p(id)?],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::Log => {
            let contribution = graph.add_node(
                Operation::Div,
                vec![cotangent, p(node.inputs[0])?],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::ReluGrad => {
            let contribution = graph.add_node(
                Operation::ReluGrad,
                vec![p(node.inputs[0])?, cotangent],
                ty(node.inputs[1])?,
            )?;
            accumulate(graph, adjoints, node.inputs[1], contribution, ty(node.inputs[1])?)?;
        }
        Operation::ZerosLike | Operation::OnesLike | Operation::StopGradient => {}
        Operation::Checkpoint => {
            accumulate(graph, adjoints, node.inputs[0], cotangent, ty(node.inputs[0])?)?;
        }
        Operation::Reshape { .. } => {
            let contribution = graph.add_node(
                Operation::Reshape {
                    shape: ty(node.inputs[0])?.shape.clone(),
                },
                vec![cotangent],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::Transpose { permutation } => {
            let inverse = inverse_permutation(permutation).ok_or_else(|| {
                AutodiffError::UnsupportedOperation {
                    node: id,
                    operation: node.operation.clone(),
                }
            })?;
            let contribution = graph.add_node(
                Operation::Transpose {
                    permutation: inverse,
                },
                vec![cotangent],
                ty(node.inputs[0])?,
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, ty(node.inputs[0])?)?;
        }
        Operation::BroadcastTo { .. } => {
            let input_type = ty(node.inputs[0])?;
            let contribution = graph.add_node(
                Operation::ReduceSumTo {
                    shape: input_type.shape.clone(),
                },
                vec![cotangent],
                input_type.clone(),
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, input_type)?;
        }
        Operation::ReduceSumTo { .. } => {
            let input_type = ty(node.inputs[0])?;
            let contribution = graph.add_node(
                Operation::BroadcastTo {
                    shape: input_type.shape.clone(),
                },
                vec![cotangent],
                input_type.clone(),
            )?;
            accumulate(graph, adjoints, node.inputs[0], contribution, input_type)?;
        }
        Operation::MatMul => {
            require_rank2(source, id, node.inputs[0], node.inputs[1])?;
            reverse_matmul(
                graph,
                primal,
                source,
                adjoints,
                node,
                cotangent,
                Operation::MatMul,
            )?;
        }
        Operation::BatchMatMul => {
            require_batch_rank(source, id, node.inputs[0], node.inputs[1])?;
            reverse_matmul(
                graph,
                primal,
                source,
                adjoints,
                node,
                cotangent,
                Operation::BatchMatMul,
            )?;
        }
    }
    Ok(())
}

fn reverse_matmul(
    graph: &mut Graph,
    primal: &[NodeId],
    source: &Graph,
    adjoints: &mut [Option<NodeId>],
    node: &Node,
    cotangent: NodeId,
    product: Operation,
) -> Result<(), AutodiffError> {
    let lhs_type = node_type(source, node.inputs[0])?.clone();
    let rhs_type = node_type(source, node.inputs[1])?.clone();
    let (lhs_perm, lhs_t_type) = transpose_last_two(&lhs_type);
    let (rhs_perm, rhs_t_type) = transpose_last_two(&rhs_type);
    let rhs_t = graph.add_node(
        Operation::Transpose {
            permutation: rhs_perm,
        },
        vec![mapped(primal, node.inputs[1])?],
        rhs_t_type,
    )?;
    let lhs_grad = graph.add_node(
        product.clone(),
        vec![cotangent, rhs_t],
        lhs_type.clone(),
    )?;
    let lhs_t = graph.add_node(
        Operation::Transpose {
            permutation: lhs_perm,
        },
        vec![mapped(primal, node.inputs[0])?],
        lhs_t_type,
    )?;
    let rhs_grad = graph.add_node(
        product,
        vec![lhs_t, cotangent],
        rhs_type.clone(),
    )?;
    accumulate(graph, adjoints, node.inputs[0], lhs_grad, lhs_type)?;
    accumulate(graph, adjoints, node.inputs[1], rhs_grad, rhs_type)?;
    Ok(())
}

fn forward_rule(
    source: &Graph,
    graph: &mut Graph,
    primal: &[NodeId],
    tangents: &[Option<NodeId>],
    id: NodeId,
    node: &Node,
) -> Result<Option<NodeId>, AutodiffError> {
    let p = |node_id| mapped(primal, node_id);
    let t = |node_id: NodeId| tangents[node_id.get() as usize];
    let ty = |node_id| node_type(source, node_id).cloned();

    let result = match &node.operation {
        Operation::Input { .. } | Operation::Constant { .. } => None,
        Operation::Add => add_optional(graph, t(node.inputs[0]), t(node.inputs[1]), node.output.clone())?,
        Operation::Sub => {
            let right = match t(node.inputs[1]) {
                Some(value) => Some(graph.add_node(
                    Operation::Scale {
                        factor: crate::Scalar::f32(-1.0),
                    },
                    vec![value],
                    node.output.clone(),
                )?),
                None => None,
            };
            add_optional(graph, t(node.inputs[0]), right, node.output.clone())?
        }
        Operation::Mul => {
            let left = optional_product(
                graph,
                t(node.inputs[0]),
                p(node.inputs[1])?,
                node.output.clone(),
                Operation::Mul,
            )?;
            let right = optional_product(
                graph,
                t(node.inputs[1]),
                p(node.inputs[0])?,
                node.output.clone(),
                Operation::Mul,
            )?;
            add_optional(graph, left, right, node.output.clone())?
        }
        Operation::Div => {
            let left = match t(node.inputs[0]) {
                Some(value) => Some(graph.add_node(
                    Operation::Div,
                    vec![value, p(node.inputs[1])?],
                    node.output.clone(),
                )?),
                None => None,
            };
            let right = match t(node.inputs[1]) {
                Some(value) => {
                    let numerator = graph.add_node(
                        Operation::Mul,
                        vec![p(node.inputs[0])?, value],
                        node.output.clone(),
                    )?;
                    let denominator = graph.add_node(
                        Operation::Mul,
                        vec![p(node.inputs[1])?, p(node.inputs[1])?],
                        ty(node.inputs[1])?,
                    )?;
                    let quotient = graph.add_node(
                        Operation::Div,
                        vec![numerator, denominator],
                        node.output.clone(),
                    )?;
                    Some(graph.add_node(
                        Operation::Scale {
                            factor: crate::Scalar::f32(-1.0),
                        },
                        vec![quotient],
                        node.output.clone(),
                    )?)
                }
                None => None,
            };
            add_optional(graph, left, right, node.output.clone())?
        }
        Operation::Scale { factor } => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Scale { factor: *factor },
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::Relu => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::ReluGrad,
                vec![p(node.inputs[0])?, value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::Exp => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Mul,
                vec![p(id)?, value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::Log => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Div,
                vec![value, p(node.inputs[0])?],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::ReluGrad => match t(node.inputs[1]) {
            Some(value) => Some(graph.add_node(
                Operation::ReluGrad,
                vec![p(node.inputs[0])?, value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::ZerosLike | Operation::OnesLike | Operation::StopGradient => None,
        Operation::Checkpoint => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Checkpoint,
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::Reshape { shape } => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Reshape {
                    shape: shape.clone(),
                },
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::Transpose { permutation } => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::Transpose {
                    permutation: permutation.clone(),
                },
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::BroadcastTo { shape } => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::BroadcastTo {
                    shape: shape.clone(),
                },
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::ReduceSumTo { shape } => match t(node.inputs[0]) {
            Some(value) => Some(graph.add_node(
                Operation::ReduceSumTo {
                    shape: shape.clone(),
                },
                vec![value],
                node.output.clone(),
            )?),
            None => None,
        },
        Operation::MatMul => {
            require_rank2(source, id, node.inputs[0], node.inputs[1])?;
            forward_matmul(graph, primal, tangents, node, Operation::MatMul)?
        }
        Operation::BatchMatMul => {
            require_batch_rank(source, id, node.inputs[0], node.inputs[1])?;
            forward_matmul(graph, primal, tangents, node, Operation::BatchMatMul)?
        }
    };
    Ok(result)
}

fn forward_matmul(
    graph: &mut Graph,
    primal: &[NodeId],
    tangents: &[Option<NodeId>],
    node: &Node,
    product: Operation,
) -> Result<Option<NodeId>, AutodiffError> {
    let lhs_tangent = tangents[node.inputs[0].get() as usize];
    let rhs_tangent = tangents[node.inputs[1].get() as usize];
    let left = match lhs_tangent {
        Some(value) => Some(graph.add_node(
            product.clone(),
            vec![value, mapped(primal, node.inputs[1])?],
            node.output.clone(),
        )?),
        None => None,
    };
    let right = match rhs_tangent {
        Some(value) => Some(graph.add_node(
            product,
            vec![mapped(primal, node.inputs[0])?, value],
            node.output.clone(),
        )?),
        None => None,
    };
    add_optional(graph, left, right, node.output.clone())
}

fn optional_product(
    graph: &mut Graph,
    tangent: Option<NodeId>,
    primal: NodeId,
    output: TensorType,
    operation: Operation,
) -> Result<Option<NodeId>, AutodiffError> {
    match tangent {
        Some(value) => Ok(Some(graph.add_node(operation, vec![value, primal], output)?)),
        None => Ok(None),
    }
}

fn copy_primal(source: &Graph) -> Result<(Graph, Vec<NodeId>), AutodiffError> {
    let mut graph = Graph::new();
    let mut mapped_nodes = Vec::with_capacity(source.nodes().len());
    for node in source.nodes() {
        let inputs = node
            .inputs
            .iter()
            .map(|&id| mapped(&mapped_nodes, id))
            .collect::<Result<Vec<_>, _>>()?;
        let copied = graph.add_node(node.operation.clone(), inputs, node.output.clone())?;
        mapped_nodes.push(copied);
    }
    Ok((graph, mapped_nodes))
}

fn accumulate(
    graph: &mut Graph,
    adjoints: &mut [Option<NodeId>],
    target: NodeId,
    contribution: NodeId,
    ty: TensorType,
) -> Result<(), AutodiffError> {
    let slot = &mut adjoints[target.get() as usize];
    *slot = Some(match *slot {
        Some(existing) => graph.add_node(Operation::Add, vec![existing, contribution], ty)?,
        None => contribution,
    });
    Ok(())
}

fn add_optional(
    graph: &mut Graph,
    lhs: Option<NodeId>,
    rhs: Option<NodeId>,
    ty: TensorType,
) -> Result<Option<NodeId>, AutodiffError> {
    Ok(match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(graph.add_node(Operation::Add, vec![lhs, rhs], ty)?),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    })
}

fn mapped(mapping: &[NodeId], id: NodeId) -> Result<NodeId, AutodiffError> {
    mapping
        .get(id.get() as usize)
        .copied()
        .ok_or(AutodiffError::InvalidNode(id))
}

fn node_type(graph: &Graph, id: NodeId) -> Result<&TensorType, AutodiffError> {
    graph
        .nodes()
        .get(id.get() as usize)
        .map(|node| &node.output)
        .ok_or(AutodiffError::InvalidNode(id))
}

fn ensure_supported_dtype(graph: &Graph, id: NodeId) -> Result<(), AutodiffError> {
    let dtype = node_type(graph, id)?.dtype;
    if dtype == DType::F32 {
        Ok(())
    } else {
        Err(AutodiffError::UnsupportedDType { node: id, dtype })
    }
}

fn inverse_permutation(permutation: &[usize]) -> Option<Vec<usize>> {
    let mut inverse = vec![usize::MAX; permutation.len()];
    for (output_axis, &input_axis) in permutation.iter().enumerate() {
        if input_axis >= permutation.len() || inverse[input_axis] != usize::MAX {
            return None;
        }
        inverse[input_axis] = output_axis;
    }
    Some(inverse)
}

fn require_rank2(
    graph: &Graph,
    node: NodeId,
    lhs: NodeId,
    rhs: NodeId,
) -> Result<(), AutodiffError> {
    if node_type(graph, lhs)?.shape.rank() == 2 && node_type(graph, rhs)?.shape.rank() == 2 {
        Ok(())
    } else {
        Err(AutodiffError::MatMulRequiresRank2 { node })
    }
}

fn require_batch_rank(
    graph: &Graph,
    node: NodeId,
    lhs: NodeId,
    rhs: NodeId,
) -> Result<(), AutodiffError> {
    if node_type(graph, lhs)?.shape.rank() >= 3 && node_type(graph, rhs)?.shape.rank() >= 3 {
        Ok(())
    } else {
        Err(AutodiffError::BatchMatMulRequiresRankAtLeast3 { node })
    }
}

fn transpose_last_two(ty: &TensorType) -> (Vec<usize>, TensorType) {
    let rank = ty.shape.rank();
    let mut permutation: Vec<usize> = (0..rank).collect();
    permutation.swap(rank - 2, rank - 1);
    let mut dims = ty.shape.dims().to_vec();
    dims.swap(rank - 2, rank - 1);
    (
        permutation,
        TensorType::new(ty.dtype, Shape::new(dims)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scalar;

    fn vector_type() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![4]))
    }

    #[test]
    fn reverse_mode_builds_product_rule_and_accumulates_paths() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", vector_type()).unwrap();
        let square = graph
            .add_node(Operation::Mul, vec![x, x], vector_type())
            .unwrap();
        graph.set_outputs(vec![square]).unwrap();

        let transformed = grad(&graph, square, &[x]).unwrap();
        assert_eq!(transformed.gradients.len(), 1);
        assert_eq!(transformed.graph.outputs(), transformed.gradients.as_slice());
        assert!(transformed
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.operation, Operation::OnesLike)));
        assert!(transformed
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.operation, Operation::Add)));
    }

    #[test]
    fn stop_gradient_produces_an_explicit_zero() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", vector_type()).unwrap();
        let stopped = graph
            .add_node(Operation::StopGradient, vec![x], vector_type())
            .unwrap();
        graph.set_outputs(vec![stopped]).unwrap();

        let transformed = grad(&graph, stopped, &[x]).unwrap();
        let gradient = &transformed.graph.nodes()[transformed.gradients[0].get() as usize];
        assert!(matches!(gradient.operation, Operation::ZerosLike));
    }

    #[test]
    fn jvp_adds_explicit_tangent_inputs() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", vector_type()).unwrap();
        let scaled = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(3.0),
                },
                vec![x],
                vector_type(),
            )
            .unwrap();
        graph.set_outputs(vec![scaled]).unwrap();

        let transformed = jvp(&graph, scaled, &[x]).unwrap();
        assert_eq!(transformed.tangent_inputs.len(), 1);
        assert_eq!(transformed.graph.outputs().len(), 2);
        assert!(matches!(
            transformed.graph.nodes()[transformed.tangent_output.get() as usize].operation,
            Operation::Scale { .. }
        ));
    }

    #[test]
    fn broadcast_reverse_uses_reduce_sum_to() {
        let mut graph = Graph::new();
        let x_ty = TensorType::new(DType::F32, Shape::new(vec![1, 3]));
        let y_ty = TensorType::new(DType::F32, Shape::new(vec![5, 3]));
        let x = graph.add_input("x", x_ty).unwrap();
        let y = graph
            .add_node(
                Operation::BroadcastTo {
                    shape: y_ty.shape.clone(),
                },
                vec![x],
                y_ty,
            )
            .unwrap();
        graph.set_outputs(vec![y]).unwrap();
        let transformed = grad(&graph, y, &[x]).unwrap();
        assert!(transformed.graph.nodes().iter().any(|node| {
            matches!(node.operation, Operation::ReduceSumTo { .. })
        }));
    }

    #[test]
    fn batched_matmul_is_differentiable() {
        let mut graph = Graph::new();
        let a_ty = TensorType::new(DType::F32, Shape::new(vec![4, 2, 3]));
        let b_ty = TensorType::new(DType::F32, Shape::new(vec![4, 3, 5]));
        let out_ty = TensorType::new(DType::F32, Shape::new(vec![4, 2, 5]));
        let a = graph.add_input("a", a_ty).unwrap();
        let b = graph.add_input("b", b_ty).unwrap();
        let out = graph
            .add_node(Operation::BatchMatMul, vec![a, b], out_ty)
            .unwrap();
        graph.set_outputs(vec![out]).unwrap();
        let transformed = value_and_grad(&graph, out, &[a, b]).unwrap();
        assert_eq!(transformed.gradients.len(), 2);
        assert!(transformed.graph.nodes().iter().any(|node| {
            matches!(node.operation, Operation::BatchMatMul)
        }));
    }
}
