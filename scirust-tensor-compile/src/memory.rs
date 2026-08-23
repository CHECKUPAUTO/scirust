//! Deterministic logical-buffer planning for canonical tensor values.
//!
//! Inputs and constants remain externally supplied. Computed values receive
//! logical buffer slots, reused only after their last use and only when the
//! complete tensor type matches.
//!
//! The legacy builder still accepts canonical [`crate::Instruction`]s while the
//! compiler pipeline is being migrated. [`MemoryPlan::from_compiler_ir`] is the
//! new compiler-IR-driven path and uses linear SSA liveness directly.

use std::collections::BTreeMap;

use scirust_tensor_ir::{NodeId, Operation, TensorType};

use crate::{
    compiler_ir::{
        verify_compiler_ir, CompilerAnalysis, CompilerIr, CompilerIrError, IrValueId,
        LinearLivenessAnalysis,
    },
    Instruction,
};

/// Stable identifier of one logical internal buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSlot(usize);

impl BufferSlot {
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// Storage class assigned to one canonical value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueStorage {
    ExternalInput,
    ExternalConstant,
    Buffer(BufferSlot),
}

/// Storage assignment for one retained graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueAllocation {
    pub node: NodeId,
    pub storage: ValueStorage,
}

/// Type carried by one reusable logical buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSlotSpec {
    pub slot: BufferSlot,
    pub tensor_type: TensorType,
}

/// Backend-neutral memory plan.
///
/// This plan describes logical storage only. Physical allocation, memory space,
/// alignment and device ownership remain runtime/backend responsibilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    allocations: Vec<ValueAllocation>,
    slots: Vec<BufferSlotSpec>,
    peak_live_buffers: usize,
}

impl MemoryPlan {
    pub fn allocations(&self) -> &[ValueAllocation] {
        &self.allocations
    }

    pub fn slots(&self) -> &[BufferSlotSpec] {
        &self.slots
    }

    pub const fn peak_live_buffers(&self) -> usize {
        self.peak_live_buffers
    }

    pub fn storage_for(&self, node: NodeId) -> Option<ValueStorage> {
        self.allocations
            .iter()
            .find(|allocation| allocation.node == node)
            .map(|allocation| allocation.storage)
    }

    /// Build logical storage from verified compiler SSA and its linear liveness.
    ///
    /// This is the migration target for the canonical pipeline. Allocation is
    /// still reported by canonical [`NodeId`] so the existing logical lowerer
    /// can consume the result unchanged; the lifetime decisions themselves are
    /// made from compiler [`IrValueId`]s.
    pub fn from_compiler_ir(ir: &CompilerIr) -> Result<Self, CompilerIrError> {
        verify_compiler_ir(ir)?;
        let liveness = <LinearLivenessAnalysis as CompilerAnalysis>::run(ir);

        let mut allocations = Vec::with_capacity(ir.operations().len());
        let mut slots = Vec::<BufferSlotSpec>::new();
        let mut free_slots = Vec::<usize>::new();
        let mut active = BTreeMap::<IrValueId, (BufferSlot, u32, bool)>::new();
        let mut peak_live_buffers = 0usize;

        for operation in ir.operations()
        {
            let step = operation.id().get();
            let retired = active
                .iter()
                .filter_map(|(&value, &(_, last_use, live_out))| {
                    (!live_out && last_use < step).then_some(value)
                })
                .collect::<Vec<_>>();

            for value in retired
            {
                if let Some((slot, _, _)) = active.remove(&value)
                {
                    free_slots.push(slot.get());
                }
            }
            free_slots.sort_unstable();

            let result_id = operation.result();
            let result = ir.value(result_id).ok_or(CompilerIrError::InvalidValue {
                operation: operation.id(),
                value: result_id,
            })?;

            let storage = match operation.operation()
            {
                Operation::Input { .. } => ValueStorage::ExternalInput,
                Operation::Constant { .. } => ValueStorage::ExternalConstant,
                _ =>
                {
                    let compatible = free_slots
                        .iter()
                        .position(|&slot| slots[slot].tensor_type == *result.tensor_type());

                    let slot = if let Some(position) = compatible
                    {
                        BufferSlot::new(free_slots.remove(position))
                    }
                    else
                    {
                        let slot = BufferSlot::new(slots.len());
                        slots.push(BufferSlotSpec {
                            slot,
                            tensor_type: result.tensor_type().clone(),
                        });
                        slot
                    };

                    let range =
                        liveness
                            .get(result_id)
                            .ok_or(CompilerIrError::InvalidValue {
                                operation: operation.id(),
                                value: result_id,
                            })?;
                    let last_use = range
                        .last_operation_use()
                        .unwrap_or(range.definition())
                        .get();

                    active.insert(result_id, (slot, last_use, range.is_live_out()));
                    peak_live_buffers = peak_live_buffers.max(active.len());
                    ValueStorage::Buffer(slot)
                },
            };

            allocations.push(ValueAllocation {
                node: result.canonical_node(),
                storage,
            });
        }

        Ok(Self {
            allocations,
            slots,
            peak_live_buffers,
        })
    }

    pub(crate) fn build(instructions: &[Instruction], outputs: &[NodeId]) -> Self {
        let final_step = instructions.len().saturating_sub(1);
        let mut last_use = BTreeMap::<NodeId, usize>::new();

        for (step, instruction) in instructions.iter().enumerate()
        {
            last_use.entry(instruction.id).or_insert(step);

            for &input in &instruction.inputs
            {
                last_use
                    .entry(input)
                    .and_modify(|use_step| *use_step = (*use_step).max(step))
                    .or_insert(step);
            }
        }

        for &output in outputs
        {
            last_use
                .entry(output)
                .and_modify(|use_step| *use_step = (*use_step).max(final_step))
                .or_insert(final_step);
        }

        let mut allocations = Vec::with_capacity(instructions.len());
        let mut slots = Vec::<BufferSlotSpec>::new();
        let mut free_slots = Vec::<usize>::new();
        let mut active = BTreeMap::<NodeId, BufferSlot>::new();
        let mut peak_live_buffers = 0usize;

        for (step, instruction) in instructions.iter().enumerate()
        {
            let retired = active
                .keys()
                .copied()
                .filter(|node| last_use.get(node).is_some_and(|last| *last < step))
                .collect::<Vec<_>>();

            for node in retired
            {
                if let Some(slot) = active.remove(&node)
                {
                    free_slots.push(slot.get());
                }
            }
            free_slots.sort_unstable();

            let storage = match &instruction.operation
            {
                Operation::Input { .. } => ValueStorage::ExternalInput,
                Operation::Constant { .. } => ValueStorage::ExternalConstant,
                _ =>
                {
                    let compatible = free_slots
                        .iter()
                        .position(|&slot| slots[slot].tensor_type == instruction.output);

                    let slot = if let Some(position) = compatible
                    {
                        BufferSlot::new(free_slots.remove(position))
                    }
                    else
                    {
                        let slot = BufferSlot::new(slots.len());
                        slots.push(BufferSlotSpec {
                            slot,
                            tensor_type: instruction.output.clone(),
                        });
                        slot
                    };

                    active.insert(instruction.id, slot);
                    peak_live_buffers = peak_live_buffers.max(active.len());
                    ValueStorage::Buffer(slot)
                },
            };

            allocations.push(ValueAllocation {
                node: instruction.id,
                storage,
            });
        }

        Self {
            allocations,
            slots,
            peak_live_buffers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalCompiler, CompilerIr};
    use scirust_tensor_ir::{DType, Graph, Shape};

    #[test]
    fn reuses_retired_exact_type_slot() {
        let ty = TensorType::new(DType::F32, Shape::new(vec![2]));
        let node = |id, op, inputs| Instruction {
            id: NodeId::new(id),
            operation: op,
            inputs,
            output: ty.clone(),
        };

        let plan = MemoryPlan::build(
            &[
                node(0, Operation::Input { name: "x".into() }, vec![]),
                node(1, Operation::Relu, vec![NodeId::new(0)]),
                node(2, Operation::Exp, vec![NodeId::new(1)]),
                node(3, Operation::Log, vec![NodeId::new(2)]),
            ],
            &[NodeId::new(3)],
        );

        assert_eq!(
            plan.storage_for(NodeId::new(0)),
            Some(ValueStorage::ExternalInput)
        );
        assert_eq!(
            plan.storage_for(NodeId::new(1)),
            Some(ValueStorage::Buffer(BufferSlot::new(0)))
        );
        assert_eq!(
            plan.storage_for(NodeId::new(2)),
            Some(ValueStorage::Buffer(BufferSlot::new(1)))
        );
        assert_eq!(
            plan.storage_for(NodeId::new(3)),
            Some(ValueStorage::Buffer(BufferSlot::new(0)))
        );
        assert_eq!(plan.peak_live_buffers(), 2);
    }

    #[test]
    fn compiler_ir_memory_matches_legacy_linear_plan() {
        let ty = TensorType::new(DType::F32, Shape::new(vec![2]));
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).unwrap();
        let relu = graph
            .add_node(Operation::Relu, vec![input], ty.clone())
            .unwrap();
        let exp = graph
            .add_node(Operation::Exp, vec![relu], ty.clone())
            .unwrap();
        let output = graph.add_node(Operation::Log, vec![exp], ty).unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let execution = CanonicalCompiler::new().compile(&graph).unwrap();
        let ir = CompilerIr::from_execution_plan(&execution).unwrap();
        let memory = MemoryPlan::from_compiler_ir(&ir).unwrap();

        assert_eq!(&memory, execution.memory_plan());
    }

    #[test]
    fn compiler_ir_memory_keeps_early_output_live() {
        let ty = TensorType::new(DType::F32, Shape::new(vec![2]));
        let mut graph = Graph::new();
        let input = graph.add_input("x", ty.clone()).unwrap();
        let held_output = graph
            .add_node(Operation::Relu, vec![input], ty.clone())
            .unwrap();
        let temporary = graph
            .add_node(Operation::Exp, vec![input], ty.clone())
            .unwrap();
        let final_output = graph
            .add_node(Operation::Log, vec![temporary], ty)
            .unwrap();
        graph
            .set_outputs(vec![held_output, final_output])
            .unwrap();

        let execution = CanonicalCompiler::new().compile(&graph).unwrap();
        let ir = CompilerIr::from_execution_plan(&execution).unwrap();
        let memory = MemoryPlan::from_compiler_ir(&ir).unwrap();

        assert_eq!(&memory, execution.memory_plan());
        assert_eq!(memory.slots().len(), 3);
        assert_eq!(memory.peak_live_buffers(), 3);
    }
}
