use scirust_tensor_ir::{NodeId, TensorType};

use super::ids::{IrOperationId, IrValueId};

/// One SSA value in the compiler IR.
///
/// The first compiler-IR phase has exactly one result per operation.  The
/// explicit value identity is nevertheless introduced now so future
/// multi-result operations, block arguments and rewrites do not have to use
/// canonical `NodeId`s as SSA identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrValue {
    id: IrValueId,
    canonical_node: NodeId,
    tensor_type: TensorType,
    defining_operation: IrOperationId,
}

impl IrValue {
    pub(crate) fn new(
        id: IrValueId,
        canonical_node: NodeId,
        tensor_type: TensorType,
        defining_operation: IrOperationId,
    ) -> Self {
        Self {
            id,
            canonical_node,
            tensor_type,
            defining_operation,
        }
    }

    pub const fn id(&self) -> IrValueId {
        self.id
    }

    pub const fn canonical_node(&self) -> NodeId {
        self.canonical_node
    }

    pub const fn tensor_type(&self) -> &TensorType {
        &self.tensor_type
    }

    pub const fn defining_operation(&self) -> IrOperationId {
        self.defining_operation
    }
}
