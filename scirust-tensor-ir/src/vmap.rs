use alloc::{vec, vec::Vec};
use core::fmt;

use crate::{
    Graph, GraphError, NodeId, Operation, SemanticError, Shape, TensorType, validate_semantics,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmapError {
    InvalidGraph(GraphError),
    InvalidSemantics(SemanticError),
    InvalidMappedInput(NodeId),
    MissingMappedNode(NodeId),
    UnsupportedOperation { node: NodeId, operation: Operation },
}

impl fmt::Display for VmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidGraph(error) => write!(f, "invalid graph during vmap: {error}"),
            Self::InvalidSemantics(error) =>
            {
                write!(f, "invalid tensor semantics during vmap: {error}")
            },
            Self::InvalidMappedInput(node) => write!(
                f,
                "vmap node {} was requested as a mapped input but is not an Input node",
                node.get()
            ),
            Self::MissingMappedNode(node) => write!(f, "vmap lost mapping for node {}", node.get()),
            Self::UnsupportedOperation { node, operation } => write!(
                f,
                "vmap does not define a batching rule for node {} operation {operation:?}",
                node.get()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VmapError {}

impl From<GraphError> for VmapError {
    fn from(value: GraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

/// Result of a leading-axis vectorization transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmapGraph {
    pub graph: Graph,
    /// Old node -> transformed node, preserving source-node order.
    pub mapping: Vec<NodeId>,
    /// Whether each source node carries the newly introduced batch axis.
    pub mapped: Vec<bool>,
    pub outputs: Vec<NodeId>,
    pub batch_size: usize,
}

/// Vectorize a canonical graph by introducing one leading batch axis.
///
/// `mapped_inputs` must name source `Input` nodes. Unmapped operands are
/// automatically lifted with [`Operation::BroadcastTo`] when they participate
/// in a mapped operation. Rank-2 `MatMul` is lowered to `BatchMatMul`; applying
/// `vmap` again to an already batched graph adds another batch-prefix axis.
pub fn vmap(
    source: &Graph,
    batch_size: usize,
    mapped_inputs: &[NodeId],
) -> Result<VmapGraph, VmapError> {
    source.validate()?;
    validate_semantics(source).map_err(VmapError::InvalidSemantics)?;

    let mut requested = vec![false; source.nodes().len()];
    for &input in mapped_inputs
    {
        let Some(node) = source.nodes().get(input.get() as usize)
        else
        {
            return Err(VmapError::InvalidMappedInput(input));
        };
        if !matches!(&node.operation, Operation::Input { .. })
        {
            return Err(VmapError::InvalidMappedInput(input));
        }
        requested[input.get() as usize] = true;
    }

    let mut graph = Graph::new();
    let mut mapping = Vec::with_capacity(source.nodes().len());
    let mut mapped = Vec::with_capacity(source.nodes().len());

    for (index, node) in source.nodes().iter().enumerate()
    {
        let input_ids = node
            .inputs
            .iter()
            .map(|&input| {
                mapping
                    .get(input.get() as usize)
                    .copied()
                    .ok_or(VmapError::MissingMappedNode(input))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_mapped = node
            .inputs
            .iter()
            .map(|&input| {
                mapped
                    .get(input.get() as usize)
                    .copied()
                    .ok_or(VmapError::MissingMappedNode(input))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let (new_id, is_mapped) = match &node.operation
        {
            Operation::Input { name } =>
            {
                let is_mapped = requested[index];
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (graph.add_input(name.clone(), output)?, is_mapped)
            },
            Operation::Constant { id } => (graph.add_constant(*id, node.output.clone())?, false),
            Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::ReluGrad =>
            {
                let is_mapped = input_mapped.iter().copied().any(|value| value);
                if !is_mapped
                {
                    (
                        graph.add_node(node.operation.clone(), input_ids, node.output.clone())?,
                        false,
                    )
                }
                else
                {
                    let output = batched_type(&node.output, batch_size);
                    let operands = batch_operands(
                        &mut graph,
                        source,
                        node,
                        &input_ids,
                        &input_mapped,
                        batch_size,
                    )?;
                    (
                        graph.add_node(node.operation.clone(), operands, output)?,
                        true,
                    )
                }
            },
            Operation::Relu
            | Operation::Exp
            | Operation::Log
            | Operation::Scale { .. }
            | Operation::ZerosLike
            | Operation::OnesLike
            | Operation::StopGradient
            | Operation::Checkpoint =>
            {
                let is_mapped = input_mapped[0];
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (
                    graph.add_node(node.operation.clone(), input_ids, output)?,
                    is_mapped,
                )
            },
            Operation::Reshape { shape } =>
            {
                let is_mapped = input_mapped[0];
                let operation = if is_mapped
                {
                    Operation::Reshape {
                        shape: batched_shape(shape, batch_size),
                    }
                }
                else
                {
                    node.operation.clone()
                };
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (graph.add_node(operation, input_ids, output)?, is_mapped)
            },
            Operation::Transpose { permutation } =>
            {
                let is_mapped = input_mapped[0];
                let operation = if is_mapped
                {
                    let mut lifted = Vec::with_capacity(permutation.len() + 1);
                    lifted.push(0);
                    lifted.extend(permutation.iter().map(|axis| axis + 1));
                    Operation::Transpose {
                        permutation: lifted,
                    }
                }
                else
                {
                    node.operation.clone()
                };
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (graph.add_node(operation, input_ids, output)?, is_mapped)
            },
            Operation::BroadcastTo { shape } =>
            {
                let is_mapped = input_mapped[0];
                let operation = if is_mapped
                {
                    Operation::BroadcastTo {
                        shape: batched_shape(shape, batch_size),
                    }
                }
                else
                {
                    node.operation.clone()
                };
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (graph.add_node(operation, input_ids, output)?, is_mapped)
            },
            Operation::ReduceSumTo { shape } =>
            {
                let is_mapped = input_mapped[0];
                let operation = if is_mapped
                {
                    Operation::ReduceSumTo {
                        shape: batched_shape(shape, batch_size),
                    }
                }
                else
                {
                    node.operation.clone()
                };
                let output = if is_mapped
                {
                    batched_type(&node.output, batch_size)
                }
                else
                {
                    node.output.clone()
                };
                (graph.add_node(operation, input_ids, output)?, is_mapped)
            },
            Operation::MatMul | Operation::BatchMatMul =>
            {
                let is_mapped = input_mapped.iter().copied().any(|value| value);
                if !is_mapped
                {
                    (
                        graph.add_node(node.operation.clone(), input_ids, node.output.clone())?,
                        false,
                    )
                }
                else
                {
                    let operands = batch_operands(
                        &mut graph,
                        source,
                        node,
                        &input_ids,
                        &input_mapped,
                        batch_size,
                    )?;
                    let output = batched_type(&node.output, batch_size);
                    (
                        graph.add_node(Operation::BatchMatMul, operands, output)?,
                        true,
                    )
                }
            },
        };

        mapping.push(new_id);
        mapped.push(is_mapped);
    }

    let outputs = source
        .outputs()
        .iter()
        .map(|&output| {
            mapping
                .get(output.get() as usize)
                .copied()
                .ok_or(VmapError::MissingMappedNode(output))
        })
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_outputs(outputs.clone())?;
    validate_semantics(&graph).map_err(VmapError::InvalidSemantics)?;

    Ok(VmapGraph {
        graph,
        mapping,
        mapped,
        outputs,
        batch_size,
    })
}

fn batch_operands(
    graph: &mut Graph,
    source: &Graph,
    node: &crate::Node,
    input_ids: &[NodeId],
    input_mapped: &[bool],
    batch_size: usize,
) -> Result<Vec<NodeId>, VmapError> {
    let mut operands = Vec::with_capacity(input_ids.len());
    for (position, &value) in input_ids.iter().enumerate()
    {
        if input_mapped[position]
        {
            operands.push(value);
            continue;
        }
        let source_input = node.inputs[position];
        let input_type = source
            .nodes()
            .get(source_input.get() as usize)
            .map(|node| &node.output)
            .ok_or(VmapError::MissingMappedNode(source_input))?;
        let lifted = batched_type(input_type, batch_size);
        let broadcast = graph.add_node(
            Operation::BroadcastTo {
                shape: lifted.shape.clone(),
            },
            vec![value],
            lifted,
        )?;
        operands.push(broadcast);
    }
    Ok(operands)
}

fn batched_type(ty: &TensorType, batch_size: usize) -> TensorType {
    TensorType::new(ty.dtype, batched_shape(&ty.shape, batch_size))
}

fn batched_shape(shape: &Shape, batch_size: usize) -> Shape {
    let mut dims = Vec::with_capacity(shape.rank() + 1);
    dims.push(batch_size);
    dims.extend_from_slice(shape.dims());
    Shape::new(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Scalar};

    fn vector() -> TensorType {
        TensorType::new(DType::F32, Shape::new(vec![3]))
    }

    #[test]
    fn vectorizes_elementwise_graph_and_broadcasts_unmapped_operand() {
        let mut graph = Graph::new();
        let x = graph.add_input("x", vector()).unwrap();
        let bias = graph.add_input("bias", vector()).unwrap();
        let sum = graph
            .add_node(Operation::Add, vec![x, bias], vector())
            .unwrap();
        let out = graph
            .add_node(
                Operation::Scale {
                    factor: Scalar::f32(2.0),
                },
                vec![sum],
                vector(),
            )
            .unwrap();
        graph.set_outputs(vec![out]).unwrap();

        let transformed = vmap(&graph, 8, &[x]).unwrap();
        let output = &transformed.graph.nodes()[transformed.outputs[0].get() as usize].output;
        assert_eq!(output.shape.dims(), &[8, 3]);
        assert!(
            transformed
                .graph
                .nodes()
                .iter()
                .any(|node| matches!(&node.operation, Operation::BroadcastTo { .. }))
        );
    }

    #[test]
    fn vectorizes_matmul_to_batch_matmul() {
        let mut graph = Graph::new();
        let lhs_ty = TensorType::new(DType::F32, Shape::new(vec![2, 3]));
        let rhs_ty = TensorType::new(DType::F32, Shape::new(vec![3, 4]));
        let out_ty = TensorType::new(DType::F32, Shape::new(vec![2, 4]));
        let lhs = graph.add_input("lhs", lhs_ty).unwrap();
        let rhs = graph.add_input("rhs", rhs_ty).unwrap();
        let out = graph
            .add_node(Operation::MatMul, vec![lhs, rhs], out_ty)
            .unwrap();
        graph.set_outputs(vec![out]).unwrap();

        let transformed = vmap(&graph, 5, &[lhs, rhs]).unwrap();
        let output_node = &transformed.graph.nodes()[transformed.outputs[0].get() as usize];
        assert!(matches!(&output_node.operation, Operation::BatchMatMul));
        assert_eq!(output_node.output.shape.dims(), &[5, 2, 4]);
    }

    #[test]
    fn nested_vmap_adds_batch_prefix() {
        let mut graph = Graph::new();
        let lhs_ty = TensorType::new(DType::F32, Shape::new(vec![2, 3]));
        let rhs_ty = TensorType::new(DType::F32, Shape::new(vec![3, 4]));
        let out_ty = TensorType::new(DType::F32, Shape::new(vec![2, 4]));
        let lhs = graph.add_input("lhs", lhs_ty).unwrap();
        let rhs = graph.add_input("rhs", rhs_ty).unwrap();
        let out = graph
            .add_node(Operation::MatMul, vec![lhs, rhs], out_ty)
            .unwrap();
        graph.set_outputs(vec![out]).unwrap();

        let first = vmap(&graph, 5, &[lhs, rhs]).unwrap();
        let second = vmap(
            &first.graph,
            7,
            &[
                first.mapping[lhs.get() as usize],
                first.mapping[rhs.get() as usize],
            ],
        )
        .unwrap();
        let output = &second.graph.nodes()[second.outputs[0].get() as usize].output;
        assert_eq!(output.shape.dims(), &[7, 5, 2, 4]);
    }
}
