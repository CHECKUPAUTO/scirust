use std::fmt;

use scirust_tensor_core::{Tensor, TensorDevice, TensorError};
use scirust_tensor_ir::{
    ConstantId, DType, Graph, GraphError, Node, NodeId, Operation, SemanticError, Shape,
    TensorType, validate_semantics,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Core2RuntimeError {
    InvalidGraph(GraphError),
    InvalidSemantics(SemanticError),
    MissingInput(NodeId),
    DuplicateInput(NodeId),
    MissingConstant(ConstantId),
    DuplicateConstant(ConstantId),
    TypeMismatch { node: NodeId },
    HostReferenceRequired { node: NodeId },
    UnsupportedDType { node: NodeId, dtype: DType },
    InvalidScale { node: NodeId },
    UnsupportedOperation { node: NodeId, operation: Operation },
    Tensor(TensorError),
}

impl fmt::Display for Core2RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid Core2 graph: {error}"),
            Self::InvalidSemantics(error) => write!(f, "invalid Core2 tensor semantics: {error}"),
            Self::MissingInput(node) => write!(f, "missing Core2 input node {}", node.get()),
            Self::DuplicateInput(node) => write!(f, "duplicate Core2 input node {}", node.get()),
            Self::MissingConstant(id) => write!(f, "missing Core2 constant {}", id.get()),
            Self::DuplicateConstant(id) => write!(f, "duplicate Core2 constant {}", id.get()),
            Self::TypeMismatch { node } => write!(f, "Core2 value type mismatch at node {}", node.get()),
            Self::HostReferenceRequired { node } => write!(f, "Core2 reference runtime requires Host placement at node {}", node.get()),
            Self::UnsupportedDType { node, dtype } => write!(f, "Core2 reference runtime does not yet execute dtype {dtype:?} at node {}", node.get()),
            Self::InvalidScale { node } => write!(f, "Core2 scale attribute is not an F32 scalar at node {}", node.get()),
            Self::UnsupportedOperation { node, operation } => write!(f, "Core2 reference runtime has no executor for node {} operation {operation:?}", node.get()),
            Self::Tensor(error) => write!(f, "Core2 tensor error: {error}"),
        }
    }
}

impl std::error::Error for Core2RuntimeError {}

impl From<GraphError> for Core2RuntimeError {
    fn from(value: GraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

impl From<TensorError> for Core2RuntimeError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Core2Inputs {
    values: Vec<(NodeId, Tensor)>,
}

impl Core2Inputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: NodeId, tensor: Tensor) -> Result<(), Core2RuntimeError> {
        if self.values.iter().any(|(existing, _)| *existing == node) {
            return Err(Core2RuntimeError::DuplicateInput(node));
        }
        self.values.push((node, tensor));
        Ok(())
    }

    fn get(&self, node: NodeId) -> Option<&Tensor> {
        self.values
            .iter()
            .find(|(candidate, _)| *candidate == node)
            .map(|(_, tensor)| tensor)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Core2Constants {
    values: Vec<(ConstantId, Tensor)>,
}

impl Core2Constants {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: ConstantId, tensor: Tensor) -> Result<(), Core2RuntimeError> {
        if self.values.iter().any(|(existing, _)| *existing == id) {
            return Err(Core2RuntimeError::DuplicateConstant(id));
        }
        self.values.push((id, tensor));
        Ok(())
    }

    fn get(&self, id: ConstantId) -> Option<&Tensor> {
        self.values
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, tensor)| tensor)
    }
}

#[derive(Debug, Clone)]
pub struct Core2Outputs {
    values: Vec<(NodeId, Tensor)>,
}

impl Core2Outputs {
    pub fn values(&self) -> &[(NodeId, Tensor)] {
        &self.values
    }

    pub fn get(&self, node: NodeId) -> Option<&Tensor> {
        self.values
            .iter()
            .find(|(candidate, _)| *candidate == node)
            .map(|(_, tensor)| tensor)
    }
}

/// Prepared deterministic CPU oracle for the canonical Core2 IR.
///
/// This path intentionally favors semantic coverage and auditability. Compiled
/// backends can be tested against it without sharing their lowering code.
#[derive(Debug, Clone)]
pub struct Core2ReferenceSession {
    graph: Graph,
    constants: Core2Constants,
}

impl Core2ReferenceSession {
    pub fn prepare(graph: Graph, constants: Core2Constants) -> Result<Self, Core2RuntimeError> {
        graph.validate()?;
        validate_semantics(&graph).map_err(Core2RuntimeError::InvalidSemantics)?;

        for (index, node) in graph.nodes().iter().enumerate() {
            let node_id = NodeId::new(index as u32);
            require_f32(node_id, &node.output)?;
            if let Operation::Constant { id } = &node.operation {
                let tensor = constants
                    .get(*id)
                    .ok_or(Core2RuntimeError::MissingConstant(*id))?;
                validate_value(node_id, tensor, &node.output)?;
            }
        }

        Ok(Self { graph, constants })
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn execute(&self, inputs: &Core2Inputs) -> Result<Core2Outputs, Core2RuntimeError> {
        let mut values: Vec<Option<Tensor>> = vec![None; self.graph.nodes().len()];

        for (index, node) in self.graph.nodes().iter().enumerate() {
            let node_id = NodeId::new(index as u32);
            let value = self.execute_node(node_id, node, &values, inputs)?;
            validate_value(node_id, &value, &node.output)?;
            values[index] = Some(value);
        }

        let mut outputs = Vec::with_capacity(self.graph.outputs().len());
        for &output in self.graph.outputs() {
            outputs.push((output, value_at(&values, output)?.clone()));
        }
        Ok(Core2Outputs { values: outputs })
    }

    fn execute_node(
        &self,
        node_id: NodeId,
        node: &Node,
        values: &[Option<Tensor>],
        inputs: &Core2Inputs,
    ) -> Result<Tensor, Core2RuntimeError> {
        match &node.operation {
            Operation::Input { .. } => {
                let tensor = inputs
                    .get(node_id)
                    .ok_or(Core2RuntimeError::MissingInput(node_id))?;
                validate_value(node_id, tensor, &node.output)?;
                Ok(tensor.clone())
            }
            Operation::Constant { id } => Ok(self
                .constants
                .get(*id)
                .ok_or(Core2RuntimeError::MissingConstant(*id))?
                .clone()),
            Operation::Add => binary_f32(values, node, |a, b| a + b),
            Operation::Sub => binary_f32(values, node, |a, b| a - b),
            Operation::Mul => binary_f32(values, node, |a, b| a * b),
            Operation::Div => binary_f32(values, node, |a, b| a / b),
            Operation::Scale { factor } => {
                let factor = factor
                    .as_f32()
                    .ok_or(Core2RuntimeError::InvalidScale { node: node_id })?;
                unary_f32(values, node, |value| value * factor)
            }
            Operation::Relu => unary_f32(values, node, |value| value.max(0.0)),
            Operation::Exp => unary_f32(values, node, f32::exp),
            Operation::Log => unary_f32(values, node, f32::ln),
            Operation::ReluGrad => {
                let primal = value_at(values, node.inputs[0])?.to_f32_vec()?;
                let tangent = value_at(values, node.inputs[1])?.to_f32_vec()?;
                let data = primal
                    .into_iter()
                    .zip(tangent)
                    .map(|(primal, tangent)| if primal > 0.0 { tangent } else { 0.0 })
                    .collect();
                Ok(Tensor::from_f32(data, node.output.shape.dims().to_vec())?)
            }
            Operation::ZerosLike => Ok(Tensor::zeros(
                DType::F32,
                node.output.shape.clone(),
                TensorDevice::Host,
            )?),
            Operation::OnesLike => Ok(Tensor::from_f32(
                vec![1.0; checked_elements(&node.output)],
                node.output.shape.dims().to_vec(),
            )?),
            Operation::MatMul => matmul_f32(values, node, false),
            Operation::BatchMatMul => matmul_f32(values, node, true),
            Operation::Reshape { shape } => {
                let input = value_at(values, node.inputs[0])?;
                if input.is_contiguous() {
                    Ok(input.reshape(shape.clone())?)
                } else {
                    Ok(input.contiguous()?.reshape(shape.clone())?)
                }
            }
            Operation::Transpose { permutation } => {
                Ok(value_at(values, node.inputs[0])?.permute(permutation)?)
            }
            Operation::BroadcastTo { shape } => {
                Ok(value_at(values, node.inputs[0])?.broadcast_to(shape.clone())?)
            }
            Operation::ReduceSumTo { shape } => {
                reduce_sum_to_f32(value_at(values, node.inputs[0])?, shape)
            }
            Operation::StopGradient | Operation::Checkpoint => {
                Ok(value_at(values, node.inputs[0])?.clone())
            }
            operation => Err(Core2RuntimeError::UnsupportedOperation {
                node: node_id,
                operation: operation.clone(),
            }),
        }
    }
}

fn value_at(values: &[Option<Tensor>], id: NodeId) -> Result<&Tensor, Core2RuntimeError> {
    values
        .get(id.get() as usize)
        .and_then(Option::as_ref)
        .ok_or(Core2RuntimeError::MissingInput(id))
}

fn require_f32(node: NodeId, ty: &TensorType) -> Result<(), Core2RuntimeError> {
    if ty.dtype == DType::F32 {
        Ok(())
    } else {
        Err(Core2RuntimeError::UnsupportedDType {
            node,
            dtype: ty.dtype,
        })
    }
}

fn validate_value(node: NodeId, tensor: &Tensor, ty: &TensorType) -> Result<(), Core2RuntimeError> {
    if tensor.dtype() != ty.dtype || tensor.shape() != &ty.shape {
        return Err(Core2RuntimeError::TypeMismatch { node });
    }
    if tensor.device() != &TensorDevice::Host {
        return Err(Core2RuntimeError::HostReferenceRequired { node });
    }
    Ok(())
}

fn checked_elements(ty: &TensorType) -> usize {
    ty.shape
        .checked_num_elements()
        .expect("validated graph shape must fit usize")
}

fn unary_f32(
    values: &[Option<Tensor>],
    node: &Node,
    function: impl Fn(f32) -> f32,
) -> Result<Tensor, Core2RuntimeError> {
    let data = value_at(values, node.inputs[0])?
        .to_f32_vec()?
        .into_iter()
        .map(function)
        .collect();
    Ok(Tensor::from_f32(data, node.output.shape.dims().to_vec())?)
}

fn binary_f32(
    values: &[Option<Tensor>],
    node: &Node,
    function: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, Core2RuntimeError> {
    let lhs = value_at(values, node.inputs[0])?.to_f32_vec()?;
    let rhs = value_at(values, node.inputs[1])?.to_f32_vec()?;
    let data = lhs
        .into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| function(lhs, rhs))
        .collect();
    Ok(Tensor::from_f32(data, node.output.shape.dims().to_vec())?)
}

fn matmul_f32(
    values: &[Option<Tensor>],
    node: &Node,
    batched: bool,
) -> Result<Tensor, Core2RuntimeError> {
    let lhs_tensor = value_at(values, node.inputs[0])?;
    let rhs_tensor = value_at(values, node.inputs[1])?;
    let lhs = lhs_tensor.to_f32_vec()?;
    let rhs = rhs_tensor.to_f32_vec()?;
    let lhs_dims = lhs_tensor.shape().dims();
    let rhs_dims = rhs_tensor.shape().dims();
    let rank = lhs_dims.len();
    let m = lhs_dims[rank - 2];
    let k = lhs_dims[rank - 1];
    let n = rhs_dims[rank - 1];
    let batches = if batched {
        lhs_dims[..rank - 2].iter().copied().product()
    } else {
        1
    };
    let mut output = vec![0.0f32; batches * m * n];
    for batch in 0..batches {
        let lhs_base = batch * m * k;
        let rhs_base = batch * k * n;
        let out_base = batch * m * n;
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    sum += lhs[lhs_base + row * k + inner] * rhs[rhs_base + inner * n + col];
                }
                output[out_base + row * n + col] = sum;
            }
        }
    }
    Ok(Tensor::from_f32(output, node.output.shape.dims().to_vec())?)
}

fn reduce_sum_to_f32(input: &Tensor, target: &Shape) -> Result<Tensor, Core2RuntimeError> {
    let source_dims = input.shape().dims();
    let target_dims = target.dims();
    let source = input.to_f32_vec()?;
    let target_elements: usize = target_dims.iter().copied().product();
    let mut output = vec![0.0f32; target_elements];
    let target_strides = contiguous_strides(target_dims);
    let rank_delta = source_dims.len() - target_dims.len();

    for (linear, value) in source.into_iter().enumerate() {
        let mut remainder = linear;
        let mut target_offset = 0usize;
        for source_axis in (0..source_dims.len()).rev() {
            let dimension = source_dims[source_axis];
            let coordinate = remainder % dimension;
            remainder /= dimension;
            if source_axis >= rank_delta {
                let target_axis = source_axis - rank_delta;
                let target_coordinate = if target_dims[target_axis] == 1 {
                    0
                } else {
                    coordinate
                };
                target_offset += target_coordinate * target_strides[target_axis];
            }
        }
        output[target_offset] += value;
    }

    Ok(Tensor::from_f32(output, target_dims.to_vec())?)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    if shape.len() > 1 {
        for axis in (0..shape.len() - 1).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{Operation, TensorType, grad, vmap};

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(DType::F32, Shape::new(shape.to_vec()))
    }

    #[test]
    fn executes_vmap_and_autodiff_composition() {
        let mut source = Graph::new();
        let x = source.add_input("x", ty(&[3])).unwrap();
        let square = source
            .add_node(Operation::Mul, vec![x, x], ty(&[3]))
            .unwrap();
        source.set_outputs(vec![square]).unwrap();

        let batched = vmap(&source, 2, &[x]).unwrap();
        let batched_x = batched.mapping[x.get() as usize];
        let differentiated = grad(&batched.graph, batched.outputs[0], &[batched_x]).unwrap();
        let session = Core2ReferenceSession::prepare(
            differentiated.graph,
            Core2Constants::new(),
        )
        .unwrap();
        let mut inputs = Core2Inputs::new();
        inputs
            .insert(
                batched_x,
                Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap(),
            )
            .unwrap();
        let outputs = session.execute(&inputs).unwrap();
        assert_eq!(
            outputs.values()[0].1.to_f32_vec().unwrap(),
            vec![2., 4., 6., 8., 10., 12.]
        );
    }

    #[test]
    fn executes_batch_matmul() {
        let mut graph = Graph::new();
        let a = graph.add_input("a", ty(&[2, 1, 2])).unwrap();
        let b = graph.add_input("b", ty(&[2, 2, 1])).unwrap();
        let out = graph
            .add_node(Operation::BatchMatMul, vec![a, b], ty(&[2, 1, 1]))
            .unwrap();
        graph.set_outputs(vec![out]).unwrap();
        let session = Core2ReferenceSession::prepare(graph, Core2Constants::new()).unwrap();
        let mut inputs = Core2Inputs::new();
        inputs
            .insert(
                a,
                Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 1, 2]).unwrap(),
            )
            .unwrap();
        inputs
            .insert(
                b,
                Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2, 1]).unwrap(),
            )
            .unwrap();
        let outputs = session.execute(&inputs).unwrap();
        assert_eq!(outputs.get(out).unwrap().to_f32_vec().unwrap(), vec![17., 53.]);
    }
}
