use super::ids::{IrBlockId, IrOperationId};

/// Ordered basic block of compiler-IR operations.
///
/// The initial bridge emits one block only.  Making the block explicit now
/// leaves room for branches, structured control flow and block arguments
/// without contaminating the canonical tensor DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrBlock {
    id: IrBlockId,
    operations: Vec<IrOperationId>,
}

impl IrBlock {
    pub(crate) fn new(id: IrBlockId, operations: Vec<IrOperationId>) -> Self {
        Self { id, operations }
    }

    pub const fn id(&self) -> IrBlockId {
        self.id
    }

    pub fn operations(&self) -> &[IrOperationId] {
        &self.operations
    }
}
