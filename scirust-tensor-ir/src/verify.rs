use alloc::vec::Vec;
use core::fmt;

use crate::{Graph, NodeId, Operation, TensorType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticError {
    InvalidNode(NodeId),
    TypeMismatch { node: NodeId, expected: TensorType, actual: TensorType },
    BinaryTypeMismatch { node: NodeId },
    InvalidReshape { node: NodeId },
    InvalidTranspose { node: NodeId },
    InvalidBroadcast { node: NodeId },
    InvalidReduction { node: NodeId },
    InvalidMatMul { node: NodeId },
    InvalidBatchMatMul { node: NodeId },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNode(node) => write!(f, "semantic verifier cannot resolve node {}", node.get()),
            Self::TypeMismatch { node, expected, actual } => write!(
                f,
                "node {} declares type {actual:?}, expected {expected:?}",
                node.get()
            ),
            Self::BinaryTypeMismatch { node } => write!(
                f,
                "node {} requires identical operand and result tensor types",
                node.get()
            ),
            Self::InvalidReshape { node } => write!(f, "node {} has an invalid reshape", node.get()),
            Self::InvalidTranspose { node } => write!(f, "node {} has an invalid transpose", node.get()),
            Self::InvalidBroadcast { node } => write!(f, "node {} has an invalid broadcast", node.get()),
            Self::InvalidReduction { node } => write!(f, "node {} has an invalid reduce-sum-to", node.get()),
            Self::InvalidMatMul { node } => write!(f, "node {} has an invalid rank-2 matmul", node.get()),
            Self::InvalidBatchMatMul { node } => write!(f, "node {} has an invalid batched matmul", node.get()),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SemanticError {}

/// Validate shape and dtype semantics in addition to [`Graph::validate`]'s
/// structural checks.
pub fn validate_semantics(graph: &Graph) -> Result<(), SemanticError> {
    for (index, node) in graph.nodes().iter().enumerate() {
        let id = NodeId::new(index as u32);
        let inputs = node
            .inputs
            .iter()
            .map(|&input| {
                graph
                    .nodes()
                    .get(input.get() as usize)
                    .map(|node| &node.output)
                    .ok_or(SemanticError::InvalidNode(input))
            })
            .collect::<Result<Vec<_>, _>>()?;

        match &node.operation {
            Operation::Input { .. } | Operation::Constant { .. } => {}
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
                if inputs.len() != 2 || *inputs[0] != node.output || *inputs[1] != node.output {
                    return Err(SemanticError::BinaryTypeMismatch { node: id });
                }
            }
            Operation::Scale { factor } => {
                require_same(id, inputs[0], &node.output)?;
                if factor.dtype() != node.output.dtype {
                    return Err(SemanticError::TypeMismatch {
                        node: id,
                        expected: TensorType::new(factor.dtype(), node.output.shape.clone()),
                        actual: node.output.clone(),
                    });
                }
            }
            Operation::Relu
            | Operation::Exp
            | Operation::Log
            | Operation::ZerosLike
            | Operation::OnesLike
            | Operation::StopGradient
            | Operation::Checkpoint => {
                require_same(id, inputs[0], &node.output)?;
            }
            Operation::ReluGrad => {
                if *inputs[0] != node.output || *inputs[1] != node.output {
                    return Err(SemanticError::BinaryTypeMismatch { node: id });
                }
            }
            Operation::Reshape { shape } => {
                let input = inputs[0];
                let source_elements = input
                    .shape
                    .checked_num_elements()
                    .map_err(|_| SemanticError::InvalidReshape { node: id })?;
                let target_elements = shape
                    .checked_num_elements()
                    .map_err(|_| SemanticError::InvalidReshape { node: id })?;
                if input.dtype != node.output.dtype
                    || node.output.shape != *shape
                    || source_elements != target_elements
                {
                    return Err(SemanticError::InvalidReshape { node: id });
                }
            }
            Operation::Transpose { permutation } => {
                let input = inputs[0];
                if permutation.len() != input.shape.rank()
                    || node.output.dtype != input.dtype
                    || node.output.shape.rank() != input.shape.rank()
                {
                    return Err(SemanticError::InvalidTranspose { node: id });
                }
                let mut seen = alloc::vec![false; permutation.len()];
                for (output_axis, &input_axis) in permutation.iter().enumerate() {
                    if input_axis >= permutation.len()
                        || seen[input_axis]
                        || node.output.shape.dims()[output_axis] != input.shape.dims()[input_axis]
                    {
                        return Err(SemanticError::InvalidTranspose { node: id });
                    }
                    seen[input_axis] = true;
                }
            }
            Operation::BroadcastTo { shape } => {
                let input = inputs[0];
                if input.dtype != node.output.dtype
                    || node.output.shape != *shape
                    || !can_broadcast_to(input.shape.dims(), shape.dims())
                {
                    return Err(SemanticError::InvalidBroadcast { node: id });
                }
            }
            Operation::ReduceSumTo { shape } => {
                let input = inputs[0];
                if input.dtype != node.output.dtype
                    || node.output.shape != *shape
                    || !can_reduce_sum_to(input.shape.dims(), shape.dims())
                {
                    return Err(SemanticError::InvalidReduction { node: id });
                }
            }
            Operation::MatMul => {
                let lhs = inputs[0];
                let rhs = inputs[1];
                if !valid_rank2_matmul(lhs, rhs, &node.output) {
                    return Err(SemanticError::InvalidMatMul { node: id });
                }
            }
            Operation::BatchMatMul => {
                let lhs = inputs[0];
                let rhs = inputs[1];
                if !valid_batch_matmul(lhs, rhs, &node.output) {
                    return Err(SemanticError::InvalidBatchMatMul { node: id });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn can_broadcast_to(source: &[usize], target: &[usize]) -> bool {
    if source.len() > target.len() {
        return false;
    }
    let offset = target.len() - source.len();
    source
        .iter()
        .zip(&target[offset..])
        .all(|(&source_dim, &target_dim)| source_dim == target_dim || source_dim == 1)
}

fn can_reduce_sum_to(source: &[usize], target: &[usize]) -> bool {
    if target.len() > source.len() {
        return false;
    }
    let offset = source.len() - target.len();
    target
        .iter()
        .zip(&source[offset..])
        .all(|(&target_dim, &source_dim)| target_dim == source_dim || target_dim == 1)
}

fn valid_rank2_matmul(lhs: &TensorType, rhs: &TensorType, output: &TensorType) -> bool {
    lhs.dtype == rhs.dtype
        && lhs.dtype == output.dtype
        && lhs.shape.rank() == 2
        && rhs.shape.rank() == 2
        && output.shape.rank() == 2
        && lhs.shape.dims()[1] == rhs.shape.dims()[0]
        && output.shape.dims()[0] == lhs.shape.dims()[0]
        && output.shape.dims()[1] == rhs.shape.dims()[1]
}

fn valid_batch_matmul(lhs: &TensorType, rhs: &TensorType, output: &TensorType) -> bool {
    if lhs.dtype != rhs.dtype
        || lhs.dtype != output.dtype
        || lhs.shape.rank() < 3
        || lhs.shape.rank() != rhs.shape.rank()
        || lhs.shape.rank() != output.shape.rank()
    {
        return false;
    }

    let rank = lhs.shape.rank();
    let lhs_dims = lhs.shape.dims();
    let rhs_dims = rhs.shape.dims();
    let out_dims = output.shape.dims();
    lhs_dims[..rank - 2] == rhs_dims[..rank - 2]
        && lhs_dims[..rank - 2] == out_dims[..rank - 2]
        && lhs_dims[rank - 1] == rhs_dims[rank - 2]
        && out_dims[rank - 2] == lhs_dims[rank - 2]
        && out_dims[rank - 1] == rhs_dims[rank - 1]
}

fn require_same(node: NodeId, expected: &TensorType, actual: &TensorType) -> Result<(), SemanticError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SemanticError::TypeMismatch {
            node,
            expected: expected.clone(),
            actual: actual.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph, Operation, Shape, TensorType};

    #[test]
    fn rejects_declared_binary_shape_lies() {
        let mut graph = Graph::new();
        let a_ty = TensorType::new(DType::F32, Shape::new(vec![2, 2]));
        let wrong = TensorType::new(DType::F32, Shape::new(vec![4]));
        let a = graph.add_input("a", a_ty.clone()).unwrap();
        let b = graph.add_input("b", a_ty).unwrap();
        graph.add_node(Operation::Add, vec![a, b], wrong).unwrap();
        assert!(matches!(
            validate_semantics(&graph),
            Err(SemanticError::BinaryTypeMismatch { .. })
        ));
    }

    #[test]
    fn validates_rank2_matmul_contract() {
        let mut graph = Graph::new();
        let a = graph
            .add_input("a", TensorType::new(DType::F32, Shape::new(vec![2, 3])))
            .unwrap();
        let b = graph
            .add_input("b", TensorType::new(DType::F32, Shape::new(vec![3, 4])))
            .unwrap();
        graph
            .add_node(
                Operation::MatMul,
                vec![a, b],
                TensorType::new(DType::F32, Shape::new(vec![2, 4])),
            )
            .unwrap();
        assert_eq!(validate_semantics(&graph), Ok(()));
    }

    #[test]
    fn validates_broadcast_and_reduce_inverse_shapes() {
        let mut graph = Graph::new();
        let x = graph
            .add_input("x", TensorType::new(DType::F32, Shape::new(vec![1, 3])))
            .unwrap();
        let broadcast_ty = TensorType::new(DType::F32, Shape::new(vec![4, 3]));
        let broadcast = graph
            .add_node(
                Operation::BroadcastTo {
                    shape: broadcast_ty.shape.clone(),
                },
                vec![x],
                broadcast_ty,
            )
            .unwrap();
        graph
            .add_node(
                Operation::ReduceSumTo {
                    shape: Shape::new(vec![1, 3]),
                },
                vec![broadcast],
                TensorType::new(DType::F32, Shape::new(vec![1, 3])),
            )
            .unwrap();
        assert_eq!(validate_semantics(&graph), Ok(()));
    }

    #[test]
    fn validates_batched_matmul_contract() {
        let mut graph = Graph::new();
        let a = graph
            .add_input("a", TensorType::new(DType::F32, Shape::new(vec![5, 2, 3])))
            .unwrap();
        let b = graph
            .add_input("b", TensorType::new(DType::F32, Shape::new(vec![5, 3, 4])))
            .unwrap();
        graph
            .add_node(
                Operation::BatchMatMul,
                vec![a, b],
                TensorType::new(DType::F32, Shape::new(vec![5, 2, 4])),
            )
            .unwrap();
        assert_eq!(validate_semantics(&graph), Ok(()));
    }
}
