use super::ids::{IrBlockId, IrRegionId};

/// Ordered collection of basic blocks.
///
/// Regions are structural compiler entities and are deliberately absent from
/// the canonical tensor graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrRegion {
    id: IrRegionId,
    blocks: Vec<IrBlockId>,
}

impl IrRegion {
    pub(crate) fn new(id: IrRegionId, blocks: Vec<IrBlockId>) -> Self {
        Self { id, blocks }
    }

    pub const fn id(&self) -> IrRegionId {
        self.id
    }

    pub fn blocks(&self) -> &[IrBlockId] {
        &self.blocks
    }
}
