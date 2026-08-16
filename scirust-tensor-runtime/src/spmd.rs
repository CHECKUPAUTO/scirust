use std::fmt;

use scirust_tensor_core::{Tensor, TensorDevice, TensorError};
use scirust_tensor_ir::{
    Graph, NodeId, Operation, ShardError, ShardMapGraph, TensorType, shard_map,
};

use crate::{Core2Constants, Core2Inputs, Core2ReferenceSession, Core2RuntimeError};

#[derive(Debug, Clone, Default)]
pub struct SpmdInputs {
    values: Vec<(NodeId, Tensor)>,
}

impl SpmdInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: NodeId, tensor: Tensor) -> Result<(), SpmdError> {
        if self.values.iter().any(|(existing, _)| *existing == node) {
            return Err(SpmdError::DuplicateInput(node));
        }
        self.values.push((node, tensor));
        Ok(())
    }

    pub fn get(&self, node: NodeId) -> Option<&Tensor> {
        self.values
            .iter()
            .find(|(candidate, _)| *candidate == node)
            .map(|(_, tensor)| tensor)
    }

    pub fn entries(&self) -> &[(NodeId, Tensor)] {
        &self.values
    }
}

#[derive(Debug, Clone)]
pub struct SpmdOutputs {
    values: Vec<(NodeId, Tensor)>,
}

impl SpmdOutputs {
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

#[derive(Debug)]
pub enum SpmdError {
    Shard(ShardError),
    Runtime(Core2RuntimeError),
    Tensor(TensorError),
    MissingInput(NodeId),
    DuplicateInput(NodeId),
    UnexpectedInput(NodeId),
    InputTypeMismatch { node: NodeId },
    InputMustBeHost { node: NodeId },
    OutputMappingLost(NodeId),
    ReplicatedOutputDiverged { node: NodeId, rank: usize },
}

impl fmt::Display for SpmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shard(error) => write!(f, "SPMD sharding failed: {error}"),
            Self::Runtime(error) => write!(f, "SPMD local execution failed: {error}"),
            Self::Tensor(error) => write!(f, "SPMD tensor operation failed: {error}"),
            Self::MissingInput(node) => write!(f, "missing SPMD input node {}", node.get()),
            Self::DuplicateInput(node) => write!(f, "duplicate SPMD input node {}", node.get()),
            Self::UnexpectedInput(node) => write!(f, "unexpected SPMD input node {}", node.get()),
            Self::InputTypeMismatch { node } => {
                write!(f, "SPMD input type mismatch at node {}", node.get())
            }
            Self::InputMustBeHost { node } => write!(
                f,
                "reference SPMD input node {} must use Host placement",
                node.get()
            ),
            Self::OutputMappingLost(node) => {
                write!(f, "SPMD output mapping lost node {}", node.get())
            }
            Self::ReplicatedOutputDiverged { node, rank } => write!(
                f,
                "replicated SPMD output node {} diverged at rank {rank}",
                node.get()
            ),
        }
    }
}

impl std::error::Error for SpmdError {}

impl From<ShardError> for SpmdError {
    fn from(value: ShardError) -> Self {
        Self::Shard(value)
    }
}

impl From<Core2RuntimeError> for SpmdError {
    fn from(value: Core2RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<TensorError> for SpmdError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

/// Deterministic single-process SPMD reference executor.
///
/// The same local graph is executed once per logical rank, strictly in ascending
/// rank order. Sharded inputs are zero-copy slices of the global input whenever
/// the leading-axis slice remains contiguous. Sharded outputs are concatenated
/// in rank order; replicated outputs must be bit-identical across all ranks.
///
/// This type deliberately contains no networking or device transport. Its role
/// is to make `shard_map` semantics executable and testable before a physical
/// multi-device/multi-host transport is selected.
#[derive(Debug, Clone)]
pub struct ReferenceSpmdSession {
    source: Graph,
    transform: ShardMapGraph,
    local: Core2ReferenceSession,
}

impl ReferenceSpmdSession {
    pub fn prepare(
        source: Graph,
        constants: Core2Constants,
        world_size: usize,
        sharded_inputs: &[NodeId],
    ) -> Result<Self, SpmdError> {
        let transform = shard_map(&source, world_size, sharded_inputs)?;
        let local = Core2ReferenceSession::prepare(transform.graph.clone(), constants)?;
        Ok(Self {
            source,
            transform,
            local,
        })
    }

    pub fn world_size(&self) -> usize {
        self.transform.world_size
    }

    pub fn local_batch(&self) -> usize {
        self.transform.local_batch
    }

    pub fn local_graph(&self) -> &Graph {
        &self.transform.graph
    }

    pub fn transform(&self) -> &ShardMapGraph {
        &self.transform
    }

    pub fn execute(&self, inputs: &SpmdInputs) -> Result<SpmdOutputs, SpmdError> {
        self.validate_global_inputs(inputs)?;

        let mut rank_outputs = Vec::with_capacity(self.transform.world_size);
        for rank in 0..self.transform.world_size {
            let mut local_inputs = Core2Inputs::new();
            for (index, node) in self.source.nodes().iter().enumerate() {
                if !matches!(&node.operation, Operation::Input { .. }) {
                    continue;
                }
                let source_id = NodeId::new(index as u32);
                let global = inputs
                    .get(source_id)
                    .ok_or(SpmdError::MissingInput(source_id))?;
                let local_value = if self.transform.sharded[index] {
                    let start = rank
                        .checked_mul(self.transform.local_batch)
                        .ok_or(TensorError::ShapeOverflow)?;
                    let end = start
                        .checked_add(self.transform.local_batch)
                        .ok_or(TensorError::ShapeOverflow)?;
                    global.slice(0, start, end, 1)?
                } else {
                    global.clone()
                };
                local_inputs.insert(self.transform.mapping[index], local_value)?;
            }
            rank_outputs.push(self.local.execute(&local_inputs)?);
        }

        let mut outputs = Vec::with_capacity(self.source.outputs().len());
        for (position, &source_output) in self.source.outputs().iter().enumerate() {
            let transformed_output = *self
                .transform
                .outputs
                .get(position)
                .ok_or(SpmdError::OutputMappingLost(source_output))?;
            let source_type = &self.source.nodes()[source_output.get() as usize].output;
            let value = if self.transform.sharded[source_output.get() as usize] {
                reassemble_sharded_output(
                    source_output,
                    transformed_output,
                    source_type,
                    &rank_outputs,
                )?
            } else {
                verify_replicated_output(
                    source_output,
                    transformed_output,
                    &rank_outputs,
                )?
            };
            outputs.push((source_output, value));
        }
        Ok(SpmdOutputs { values: outputs })
    }

    fn validate_global_inputs(&self, inputs: &SpmdInputs) -> Result<(), SpmdError> {
        for (node, _) in inputs.entries() {
            let Some(source) = self.source.nodes().get(node.get() as usize) else {
                return Err(SpmdError::UnexpectedInput(*node));
            };
            if !matches!(&source.operation, Operation::Input { .. }) {
                return Err(SpmdError::UnexpectedInput(*node));
            }
        }

        for (index, node) in self.source.nodes().iter().enumerate() {
            if !matches!(&node.operation, Operation::Input { .. }) {
                continue;
            }
            let id = NodeId::new(index as u32);
            let tensor = inputs.get(id).ok_or(SpmdError::MissingInput(id))?;
            validate_input_type(id, tensor, &node.output)?;
        }
        Ok(())
    }
}

fn validate_input_type(node: NodeId, tensor: &Tensor, expected: &TensorType) -> Result<(), SpmdError> {
    if tensor.dtype() != expected.dtype || tensor.shape() != &expected.shape {
        return Err(SpmdError::InputTypeMismatch { node });
    }
    if tensor.device() != &TensorDevice::Host {
        return Err(SpmdError::InputMustBeHost { node });
    }
    Ok(())
}

fn reassemble_sharded_output(
    source_node: NodeId,
    transformed_node: NodeId,
    global_type: &TensorType,
    rank_outputs: &[crate::Core2Outputs],
) -> Result<Tensor, SpmdError> {
    let element_width = global_type.dtype.size_bytes();
    let elements = global_type
        .shape
        .checked_num_elements()
        .map_err(|_| TensorError::ShapeOverflow)?;
    let capacity = elements
        .checked_mul(element_width)
        .ok_or(TensorError::ShapeOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for outputs in rank_outputs {
        let local = outputs
            .get(transformed_node)
            .ok_or(SpmdError::OutputMappingLost(source_node))?;
        bytes.extend_from_slice(&local.to_contiguous_bytes());
    }
    Ok(Tensor::from_bytes(
        bytes,
        global_type.dtype,
        global_type.shape.clone(),
        TensorDevice::Host,
    )?)
}

fn verify_replicated_output(
    source_node: NodeId,
    transformed_node: NodeId,
    rank_outputs: &[crate::Core2Outputs],
) -> Result<Tensor, SpmdError> {
    let first = rank_outputs
        .first()
        .and_then(|outputs| outputs.get(transformed_node))
        .ok_or(SpmdError::OutputMappingLost(source_node))?;
    let first_bytes = first.to_contiguous_bytes();
    for (rank, outputs) in rank_outputs.iter().enumerate().skip(1) {
        let candidate = outputs
            .get(transformed_node)
            .ok_or(SpmdError::OutputMappingLost(source_node))?;
        if candidate.dtype() != first.dtype()
            || candidate.shape() != first.shape()
            || candidate.to_contiguous_bytes() != first_bytes
        {
            return Err(SpmdError::ReplicatedOutputDiverged {
                node: source_node,
                rank,
            });
        }
    }
    Ok(first.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_ir::{DType, Shape, TensorType, grad, vmap};

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(DType::F32, Shape::new(shape.to_vec()))
    }

    #[test]
    fn executes_grad_vmap_through_four_spmd_ranks() {
        let mut scalar = Graph::new();
        let x = scalar.add_input("x", ty(&[3])).unwrap();
        let square = scalar
            .add_node(Operation::Mul, vec![x, x], ty(&[3]))
            .unwrap();
        scalar.set_outputs(vec![square]).unwrap();

        let batched = vmap(&scalar, 8, &[x]).unwrap();
        let batched_x = batched.mapping[x.get() as usize];
        let differentiated = grad(&batched.graph, batched.outputs[0], &[batched_x]).unwrap();
        let session = ReferenceSpmdSession::prepare(
            differentiated.graph,
            Core2Constants::new(),
            4,
            &[batched_x],
        )
        .unwrap();

        assert_eq!(session.local_batch(), 2);
        let mut inputs = SpmdInputs::new();
        inputs
            .insert(
                batched_x,
                Tensor::from_f32(
                    vec![
                        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
                        16., 17., 18., 19., 20., 21., 22., 23., 24.,
                    ],
                    vec![8, 3],
                )
                .unwrap(),
            )
            .unwrap();

        let outputs = session.execute(&inputs).unwrap();
        let gradient = outputs.values()[0].1.to_f32_vec().unwrap();
        assert_eq!(
            gradient,
            vec![
                2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32.,
                34., 36., 38., 40., 42., 44., 46., 48.,
            ]
        );
    }

    #[test]
    fn replicated_bias_is_broadcast_locally_after_vmap() {
        let mut scalar = Graph::new();
        let x = scalar.add_input("x", ty(&[3])).unwrap();
        let bias = scalar.add_input("bias", ty(&[3])).unwrap();
        let sum = scalar
            .add_node(Operation::Add, vec![x, bias], ty(&[3]))
            .unwrap();
        scalar.set_outputs(vec![sum]).unwrap();

        let batched = vmap(&scalar, 4, &[x]).unwrap();
        let batched_x = batched.mapping[x.get() as usize];
        let batched_bias = batched.mapping[bias.get() as usize];
        let session = ReferenceSpmdSession::prepare(
            batched.graph,
            Core2Constants::new(),
            2,
            &[batched_x],
        )
        .unwrap();

        let mut inputs = SpmdInputs::new();
        inputs
            .insert(
                batched_x,
                Tensor::from_f32(
                    vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                    vec![4, 3],
                )
                .unwrap(),
            )
            .unwrap();
        inputs
            .insert(
                batched_bias,
                Tensor::from_f32(vec![10., 20., 30.], vec![3]).unwrap(),
            )
            .unwrap();

        let outputs = session.execute(&inputs).unwrap();
        assert_eq!(
            outputs.values()[0].1.to_f32_vec().unwrap(),
            vec![11., 22., 33., 14., 25., 36., 17., 28., 39., 20., 31., 42.]
        );
    }
}
