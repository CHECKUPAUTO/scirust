use scirust_tensor_ir::{NodeId, Operation};

use super::ids::{IrOperationId, IrValueId};

/// One operation in the compiler IR.
///
/// `operation` is still the canonical tensor operation in this first phase.
/// Compiler-specific dialect operations are intentionally not invented yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrOperation {
    id: IrOperationId,
    canonical_node: NodeId,
    operation: Operation,
    operands: Vec<IrValueId>,
    result: IrValueId,
}

impl IrOperation {
    pub(crate) fn new(
        id: IrOperationId,
        canonical_node: NodeId,
        operation: Operation,
        operands: Vec<IrValueId>,
        result: IrValueId,
    ) -> Self {
        Self {
            id,
            canonical_node,
            operation,
            operands,
            result,
        }
    }

    pub const fn id(&self) -> IrOperationId {
        self.id
    }

    pub const fn canonical_node(&self) -> NodeId {
        self.canonical_node
    }

    pub const fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn operands(&self) -> &[IrValueId] {
        &self.operands
    }

    pub const fn result(&self) -> IrValueId {
        self.result
    }
}
