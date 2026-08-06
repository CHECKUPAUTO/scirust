//! Deterministic logical-buffer planning for canonical tensor graphs.
//!
//! Inputs and constants remain externally supplied. Computed values receive
//! logical buffer slots, reused only after their last use and only when the
//! complete tensor type matches.

use std::collections::BTreeMap;

use scirust_tensor_ir::{NodeId, Operation, TensorType};

use crate::Instruction;

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
    use scirust_tensor_ir::{DType, Shape};

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
}
