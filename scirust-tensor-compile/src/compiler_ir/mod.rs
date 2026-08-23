//! Backend-neutral compiler IR.
//!
//! This layer is intentionally distinct from the canonical tensor graph:
//!
//! * `scirust_tensor_ir::Graph` describes tensor mathematics;
//! * [`CompilerIr`] introduces explicit SSA values and structural containers;
//! * physical buffers, devices, streams and target code remain outside this IR.
//!
//! The first version is deliberately conservative: one region, one block and
//! one SSA result per canonical instruction.  The structural model is already
//! general enough to grow blocks, regions, rewrites and dialects later without
//! changing the canonical tensor IR.

mod block;
mod ids;
mod operation;
mod program;
mod region;
mod value;
mod verify;

pub use block::IrBlock;
pub use ids::{IrBlockId, IrOperationId, IrRegionId, IrValueId};
pub use operation::IrOperation;
pub use program::{
    CompilerIr, CompilerIrError, CompilerIrIdentifierSpace, CompilerPass, OperationRewrite,
    PassManager, PassManagerStats, PassResult, RewriteStats, Rewriter,
    ScaleZeroCanonicalizationPass,
};
pub use region::IrRegion;
pub use value::IrValue;
pub use verify::verify_compiler_ir;
