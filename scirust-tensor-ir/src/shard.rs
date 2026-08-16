use alloc::{string::String, vec, vec::Vec};
use core::fmt;

use crate::{Graph, GraphError, NodeId, Operation, SemanticError, Shape, TensorType, validate_semantics};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    name: String,
    size: usize,
}

impl MeshAxis {
    pub fn new(name: impl Into<String>, size: usize) -> Result<Self, ShardError> {
        if size == 0 {
            return Err(ShardError::EmptyMeshAxis);
        }
        Ok(Self {
            name: name.into(),
            size,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

/// Logical device mesh. It contains topology coordinates, not physical backend
/// handles; binding logical ranks to concrete devices belongs to the runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceMesh {
    axes: Vec<MeshAxis>,
    world_size: usize,
}

impl DeviceMesh {
    pub fn new(axes: Vec<MeshAxis>) -> Result<Self, ShardError> {
        if axes.is_empty() {
            return Err(ShardError::EmptyMesh);
        }
        let mut world_size = 1usize;
        for (index, axis) in axes.iter().enumerate() {
            if axes[..index].iter().any(|other| other.name == axis.name) {
                return Err(ShardError::DuplicateMeshAxis(axis.name.clone()));
            }
            world_size = world_size
                .checked_mul(axis.size)
                .ok_or(ShardError::MeshSizeOverflow)?;
        }
        Ok(Self { axes, world_size })
    }

    pub fn one_dimensional(name: impl Into<String>, size: usize) -> Result<Self, ShardError> {
        Self::new(vec![MeshAxis::new(name, size)?])
    }

    pub fn axes(&self) -> &[MeshAxis] {
        &self.axes
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn coordinates(&self, rank: usize) -> Result<Vec<usize>, ShardError> {
        if rank >= self.world_size {
            return Err(ShardError::InvalidRank {
                rank,
                world_size: self.world_size,
            });
        }
        let mut remainder = rank;
        let mut coordinates = vec![0usize; self.axes.len()];
        for axis in (0..self.axes.len()).rev() {
            let size = self.axes[axis].size;
            coordinates[axis] = remainder % size;
            remainder /= size;
        }
        Ok(coordinates)
    }
}

/// Tensor-axis to mesh-axis mapping. `None` means replicated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionSpec {
    axes: Vec<Option<usize>>,
}

impl PartitionSpec {
    pub fn replicated(rank: usize) -> Self {
        Self {
            axes: vec![None; rank],
        }
    }

    pub fn new(axes: Vec<Option<usize>>) -> Self {
        Self { axes }
    }

    pub fn axes(&self) -> &[Option<usize>] {
        &self.axes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardPolicy {
    /// Every participating tensor dimension must be exactly divisible by its
    /// mesh axis. This is the policy required by uniform SPMD compilation.
    Even,
    /// Distribute the remainder over the lowest mesh coordinates. Useful for
    /// data movement/reference execution, but local shapes may differ by rank.
    Balanced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisShard {
    pub start: usize,
    pub end: usize,
}

impl AxisShard {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankShard {
    pub rank: usize,
    pub coordinates: Vec<usize>,
    pub local_type: TensorType,
    /// One range per tensor axis. Replicated axes cover their whole dimension.
    pub ranges: Vec<AxisShard>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardPlan {
    pub global_type: TensorType,
    pub mesh: DeviceMesh,
    pub partition: PartitionSpec,
    pub policy: ShardPolicy,
    pub ranks: Vec<RankShard>,
}

impl ShardPlan {
    pub fn uniform_local_type(&self) -> Option<&TensorType> {
        let first = self.ranks.first()?.local_type.clone();
        if self.ranks.iter().all(|rank| rank.local_type == first) {
            self.ranks.first().map(|rank| &rank.local_type)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardError {
    InvalidGraph(GraphError),
    InvalidSemantics(SemanticError),
    EmptyMesh,
    EmptyMeshAxis,
    DuplicateMeshAxis(String),
    MeshSizeOverflow,
    PartitionRankMismatch { tensor_rank: usize, spec_rank: usize },
    InvalidMeshAxis { tensor_axis: usize, mesh_axis: usize },
    MeshAxisReused { mesh_axis: usize },
    UnevenPartition { tensor_axis: usize, dimension: usize, parts: usize },
    InvalidRank { rank: usize, world_size: usize },
    InvalidShardedInput(NodeId),
    BatchAxisRequired { node: NodeId },
    UnsupportedOperation { node: NodeId, operation: Operation },
    MixedShardedBinary { node: NodeId },
    BatchAxisMoved { node: NodeId },
    BatchReductionRequiresCollective { node: NodeId },
    MissingMappedNode(NodeId),
}

impl fmt::Display for ShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid graph during sharding: {error}"),
            Self::InvalidSemantics(error) => write!(f, "invalid tensor semantics during sharding: {error}"),
            Self::EmptyMesh => f.write_str("device mesh must contain at least one axis"),
            Self::EmptyMeshAxis => f.write_str("device mesh axis size must be non-zero"),
            Self::DuplicateMeshAxis(name) => write!(f, "duplicate device mesh axis {name:?}"),
            Self::MeshSizeOverflow => f.write_str("device mesh world size overflows usize"),
            Self::PartitionRankMismatch { tensor_rank, spec_rank } => write!(
                f,
                "partition rank {spec_rank} does not match tensor rank {tensor_rank}"
            ),
            Self::InvalidMeshAxis { tensor_axis, mesh_axis } => write!(
                f,
                "tensor axis {tensor_axis} references unknown mesh axis {mesh_axis}"
            ),
            Self::MeshAxisReused { mesh_axis } => write!(
                f,
                "mesh axis {mesh_axis} is assigned to more than one tensor axis"
            ),
            Self::UnevenPartition { tensor_axis, dimension, parts } => write!(
                f,
                "tensor axis {tensor_axis} of size {dimension} is not divisible by {parts} shards"
            ),
            Self::InvalidRank { rank, world_size } => {
                write!(f, "rank {rank} is outside world size {world_size}")
            }
            Self::InvalidShardedInput(node) => write!(
                f,
                "node {} was requested as a sharded input but is not an Input node",
                node.get()
            ),
            Self::BatchAxisRequired { node } => write!(
                f,
                "node {} has no leading batch axis to shard",
                node.get()
            ),
            Self::UnsupportedOperation { node, operation } => write!(
                f,
                "shard_map has no communication-free rule for node {} operation {operation:?}",
                node.get()
            ),
            Self::MixedShardedBinary { node } => write!(
                f,
                "node {} mixes a sharded tensor with an equally shaped replicated tensor; shard that input too or broadcast from an unbatched value",
                node.get()
            ),
            Self::BatchAxisMoved { node } => write!(
                f,
                "node {} moves the sharded leading batch axis; this requires a reshard operation",
                node.get()
            ),
            Self::BatchReductionRequiresCollective { node } => write!(
                f,
                "node {} reduces the sharded batch axis and therefore requires a collective",
                node.get()
            ),
            Self::MissingMappedNode(node) => write!(f, "shard_map lost mapping for node {}", node.get()),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ShardError {}

impl From<GraphError> for ShardError {
    fn from(value: GraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

/// Build a rank-by-rank partition plan for an arbitrary tensor type.
pub fn plan_sharding(
    tensor_type: TensorType,
    mesh: DeviceMesh,
    partition: PartitionSpec,
    policy: ShardPolicy,
) -> Result<ShardPlan, ShardError> {
    if partition.axes.len() != tensor_type.shape.rank() {
        return Err(ShardError::PartitionRankMismatch {
            tensor_rank: tensor_type.shape.rank(),
            spec_rank: partition.axes.len(),
        });
    }

    let mut used_mesh_axes = vec![false; mesh.axes.len()];
    for (tensor_axis, assignment) in partition.axes.iter().enumerate() {
        if let Some(mesh_axis) = assignment {
            if *mesh_axis >= mesh.axes.len() {
                return Err(ShardError::InvalidMeshAxis {
                    tensor_axis,
                    mesh_axis: *mesh_axis,
                });
            }
            if used_mesh_axes[*mesh_axis] {
                return Err(ShardError::MeshAxisReused {
                    mesh_axis: *mesh_axis,
                });
            }
            used_mesh_axes[*mesh_axis] = true;
            if policy == ShardPolicy::Even {
                let dimension = tensor_type.shape.dims()[tensor_axis];
                let parts = mesh.axes[*mesh_axis].size;
                if dimension % parts != 0 {
                    return Err(ShardError::UnevenPartition {
                        tensor_axis,
                        dimension,
                        parts,
                    });
                }
            }
        }
    }

    let mut ranks = Vec::with_capacity(mesh.world_size);
    for rank in 0..mesh.world_size {
        let coordinates = mesh.coordinates(rank)?;
        let mut local_dims = tensor_type.shape.dims().to_vec();
        let mut ranges = Vec::with_capacity(local_dims.len());
        for tensor_axis in 0..local_dims.len() {
            let dimension = tensor_type.shape.dims()[tensor_axis];
            if let Some(mesh_axis) = partition.axes[tensor_axis] {
                let parts = mesh.axes[mesh_axis].size;
                let coordinate = coordinates[mesh_axis];
                let range = partition_range(dimension, parts, coordinate, policy, tensor_axis)?;
                local_dims[tensor_axis] = range.len();
                ranges.push(range);
            } else {
                ranges.push(AxisShard {
                    start: 0,
                    end: dimension,
                });
            }
        }
        ranks.push(RankShard {
            rank,
            coordinates,
            local_type: TensorType::new(tensor_type.dtype, Shape::new(local_dims)),
            ranges,
        });
    }

    Ok(ShardPlan {
        global_type: tensor_type,
        mesh,
        partition,
        policy,
        ranks,
    })
}

fn partition_range(
    dimension: usize,
    parts: usize,
    coordinate: usize,
    policy: ShardPolicy,
    tensor_axis: usize,
) -> Result<AxisShard, ShardError> {
    match policy {
        ShardPolicy::Even => {
            if dimension % parts != 0 {
                return Err(ShardError::UnevenPartition {
                    tensor_axis,
                    dimension,
                    parts,
                });
            }
            let width = dimension / parts;
            Ok(AxisShard {
                start: coordinate * width,
                end: (coordinate + 1) * width,
            })
        }
        ShardPolicy::Balanced => {
            let base = dimension / parts;
            let remainder = dimension % parts;
            let width = base + usize::from(coordinate < remainder);
            let start = coordinate * base + coordinate.min(remainder);
            Ok(AxisShard {
                start,
                end: start + width,
            })
        }
    }
}

/// Communication-free SPMD transform for a leading batch dimension.
///
/// This transform deliberately handles only map-like graphs. Reducing or moving
/// the sharded batch axis is rejected because doing so correctly requires an
/// explicit collective/reshard operation. Use [`crate::vmap`] first to turn an
/// unbatched function into a batch program, then shard that leading axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardMapGraph {
    pub graph: Graph,
    pub mapping: Vec<NodeId>,
    pub sharded: Vec<bool>,
    pub outputs: Vec<NodeId>,
    pub world_size: usize,
    pub local_batch: usize,
}

pub fn shard_map(
    source: &Graph,
    world_size: usize,
    sharded_inputs: &[NodeId],
) -> Result<ShardMapGraph, ShardError> {
    source.validate()?;
    validate_semantics(source).map_err(ShardError::InvalidSemantics)?;
    if world_size == 0 {
        return Err(ShardError::EmptyMeshAxis);
    }

    let mut requested = vec![false; source.nodes().len()];
    let mut global_batch = None;
    for &input in sharded_inputs {
        let Some(node) = source.nodes().get(input.get() as usize) else {
            return Err(ShardError::InvalidShardedInput(input));
        };
        if !matches!(&node.operation, Operation::Input { .. }) {
            return Err(ShardError::InvalidShardedInput(input));
        }
        let Some(&batch) = node.output.shape.dims().first() else {
            return Err(ShardError::BatchAxisRequired { node: input });
        };
        if batch % world_size != 0 {
            return Err(ShardError::UnevenPartition {
                tensor_axis: 0,
                dimension: batch,
                parts: world_size,
            });
        }
        match global_batch {
            Some(existing) if existing != batch => {
                return Err(ShardError::MixedShardedBinary { node: input });
            }
            None => global_batch = Some(batch),
            _ => {}
        }
        requested[input.get() as usize] = true;
    }
    let global_batch = global_batch.unwrap_or(world_size);
    let local_batch = global_batch / world_size;

    let mut graph = Graph::new();
    let mut mapping = Vec::with_capacity(source.nodes().len());
    let mut sharded = Vec::with_capacity(source.nodes().len());

    for (index, node) in source.nodes().iter().enumerate() {
        let old_id = NodeId::new(index as u32);
        let inputs = node
            .inputs
            .iter()
            .map(|&input| {
                mapping
                    .get(input.get() as usize)
                    .copied()
                    .ok_or(ShardError::MissingMappedNode(input))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_sharded = node
            .inputs
            .iter()
            .map(|&input| {
                sharded
                    .get(input.get() as usize)
                    .copied()
                    .ok_or(ShardError::MissingMappedNode(input))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let (new_id, is_sharded) = match &node.operation {
            Operation::Input { name } => {
                let is_sharded = requested[index];
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_input(name.clone(), output)?, is_sharded)
            }
            Operation::Constant { id } => {
                (graph.add_constant(*id, node.output.clone())?, false)
            }
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::ReluGrad => {
                let any = input_sharded.iter().any(|value| *value);
                if any && input_sharded.iter().any(|value| !*value) {
                    return Err(ShardError::MixedShardedBinary { node: old_id });
                }
                let output = if any {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(node.operation.clone(), inputs, output)?, any)
            }
            Operation::Relu
            | Operation::Exp
            | Operation::Log
            | Operation::Scale { .. }
            | Operation::ZerosLike
            | Operation::OnesLike
            | Operation::StopGradient
            | Operation::Checkpoint => {
                let is_sharded = input_sharded[0];
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(node.operation.clone(), inputs, output)?, is_sharded)
            }
            Operation::BroadcastTo { shape } => {
                let is_sharded = input_sharded[0] || shape.dims().first() == Some(&global_batch);
                let operation = if is_sharded {
                    Operation::BroadcastTo {
                        shape: local_batch_shape(shape, world_size, old_id)?,
                    }
                } else {
                    node.operation.clone()
                };
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(operation, inputs, output)?, is_sharded)
            }
            Operation::Reshape { shape } => {
                let is_sharded = input_sharded[0];
                if is_sharded
                    && (node.output.shape.dims().first() != Some(&global_batch)
                        || shape.dims().first() != Some(&global_batch))
                {
                    return Err(ShardError::BatchAxisMoved { node: old_id });
                }
                let operation = if is_sharded {
                    Operation::Reshape {
                        shape: local_batch_shape(shape, world_size, old_id)?,
                    }
                } else {
                    node.operation.clone()
                };
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(operation, inputs, output)?, is_sharded)
            }
            Operation::Transpose { permutation } => {
                let is_sharded = input_sharded[0];
                if is_sharded && permutation.first() != Some(&0) {
                    return Err(ShardError::BatchAxisMoved { node: old_id });
                }
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(node.operation.clone(), inputs, output)?, is_sharded)
            }
            Operation::ReduceSumTo { shape } => {
                let is_sharded = input_sharded[0];
                if is_sharded && shape.dims().first() != Some(&global_batch) {
                    return Err(ShardError::BatchReductionRequiresCollective { node: old_id });
                }
                let operation = if is_sharded {
                    Operation::ReduceSumTo {
                        shape: local_batch_shape(shape, world_size, old_id)?,
                    }
                } else {
                    node.operation.clone()
                };
                let output = if is_sharded {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(operation, inputs, output)?, is_sharded)
            }
            Operation::BatchMatMul => {
                let any = input_sharded.iter().any(|value| *value);
                if any && input_sharded.iter().any(|value| !*value) {
                    return Err(ShardError::MixedShardedBinary { node: old_id });
                }
                let output = if any {
                    local_batch_type(&node.output, world_size, old_id)?
                } else {
                    node.output.clone()
                };
                (graph.add_node(Operation::BatchMatMul, inputs, output)?, any)
            }
            Operation::MatMul => {
                if input_sharded.iter().any(|value| *value) {
                    return Err(ShardError::UnsupportedOperation {
                        node: old_id,
                        operation: node.operation.clone(),
                    });
                }
                (
                    graph.add_node(Operation::MatMul, inputs, node.output.clone())?,
                    false,
                )
            }
            operation => {
                return Err(ShardError::UnsupportedOperation {
                    node: old_id,
                    operation: operation.clone(),
                });
            }
        };
        mapping.push(new_id);
        sharded.push(is_sharded);
    }

    let outputs = source
        .outputs()
        .iter()
        .map(|&output| {
            mapping
                .get(output.get() as usize)
                .copied()
                .ok_or(ShardError::MissingMappedNode(output))
        })
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_outputs(outputs.clone())?;
    validate_semantics(&graph).map_err(ShardError::InvalidSemantics)?;

    Ok(ShardMapGraph {
        graph,
        mapping,
        sharded,
        outputs,
        world_size,
        local_batch,
    })
}

fn local_batch_type(
    tensor_type: &TensorType,
    world_size: usize,
    node: NodeId,
) -> Result<TensorType, ShardError> {
    Ok(TensorType::new(
        tensor_type.dtype,
        local_batch_shape(&tensor_type.shape, world_size, node)?,
    ))
}

fn local_batch_shape(
    shape: &Shape,
    world_size: usize,
    node: NodeId,
) -> Result<Shape, ShardError> {
    let Some(&batch) = shape.dims().first() else {
        return Err(ShardError::BatchAxisRequired { node });
    };
    if batch % world_size != 0 {
        return Err(ShardError::UnevenPartition {
            tensor_axis: 0,
            dimension: batch,
            parts: world_size,
        });
    }
    let mut dims = shape.dims().to_vec();
    dims[0] = batch / world_size;
    Ok(Shape::new(dims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, vmap};

    fn ty(shape: &[usize]) -> TensorType {
        TensorType::new(DType::F32, Shape::new(shape.to_vec()))
    }

    #[test]
    fn balanced_partition_covers_every_element_once() {
        let mesh = DeviceMesh::one_dimensional("data", 3).unwrap();
        let plan = plan_sharding(
            ty(&[10, 4]),
            mesh,
            PartitionSpec::new(vec![Some(0), None]),
            ShardPolicy::Balanced,
        )
        .unwrap();
        assert_eq!(plan.ranks[0].ranges[0], AxisShard { start: 0, end: 4 });
        assert_eq!(plan.ranks[1].ranges[0], AxisShard { start: 4, end: 7 });
        assert_eq!(plan.ranks[2].ranges[0], AxisShard { start: 7, end: 10 });
    }

    #[test]
    fn even_policy_rejects_non_divisible_dimension() {
        let mesh = DeviceMesh::one_dimensional("data", 3).unwrap();
        assert!(matches!(
            plan_sharding(
                ty(&[10]),
                mesh,
                PartitionSpec::new(vec![Some(0)]),
                ShardPolicy::Even,
            ),
            Err(ShardError::UnevenPartition { .. })
        ));
    }

    #[test]
    fn shard_map_composes_after_vmap() {
        let mut scalar_graph = Graph::new();
        let x = scalar_graph.add_input("x", ty(&[3])).unwrap();
        let y = scalar_graph
            .add_node(Operation::Mul, vec![x, x], ty(&[3]))
            .unwrap();
        scalar_graph.set_outputs(vec![y]).unwrap();

        let batched = vmap(&scalar_graph, 8, &[x]).unwrap();
        let batched_x = batched.mapping[x.get() as usize];
        let sharded = shard_map(&batched.graph, 4, &[batched_x]).unwrap();
        let local_x = sharded.mapping[batched_x.get() as usize];
        assert_eq!(sharded.graph.nodes()[local_x.get() as usize].output.shape.dims(), &[2, 3]);
        let local_output = sharded.outputs[0];
        assert_eq!(sharded.graph.nodes()[local_output.get() as usize].output.shape.dims(), &[2, 3]);
    }

    #[test]
    fn shard_map_rejects_batch_reduction_without_collective() {
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty(&[8, 3])).unwrap();
        let reduced = graph
            .add_node(
                Operation::ReduceSumTo {
                    shape: Shape::new(vec![1, 3]),
                },
                vec![input],
                ty(&[1, 3]),
            )
            .unwrap();
        graph.set_outputs(vec![reduced]).unwrap();
        assert!(matches!(
            shard_map(&graph, 4, &[input]),
            Err(ShardError::BatchReductionRequiresCollective { .. })
        ));
    }
}
