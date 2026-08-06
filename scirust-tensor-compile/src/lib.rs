//! Tensor graph compilation for SciRust.
//!
//! # Canonical compilation pipeline
//!
//! Four successive descriptions of one computation, each answering a different
//! question and none of them containing backend state:
//!
//! 1. **Canonical IR** — [`scirust_tensor_ir::Graph`]: *what* is computed. Graph
//!    structure and tensor metadata, no payload, no device, no kernel.
//! 2. **Execution plan** — [`ExecutionPlan`], from [`CanonicalCompiler`]: *in
//!    which order*. Validated graph, dead nodes eliminated, topological order,
//!    canonical [`scirust_tensor_ir::NodeId`] preserved.
//! 3. **Memory plan** — [`MemoryPlan`]: *in which logical buffer* each value
//!    lives. Logical [`BufferSlot`]s with deterministic reuse after last use,
//!    external inputs and constants kept distinct from internal buffers. No
//!    physical allocation, no memory space, no alignment, no device ownership.
//! 4. **Lowered plan** — [`LoweredPlan`], from [`KernelLowerer`]: *by which
//!    logical kernel*, fed by which logical arguments. Detailed below.
//!
//! # Logical binding versus physical buffer binding
//!
//! A [`KernelArgument`] is a *logical* binding. It names a role — "the second
//! read operand of this dispatch" — and the value that fills it, as one of an
//! external input, an external constant, or an internal logical buffer slot. It
//! carries no pointer, no byte offset, no length, no device, no lifetime.
//!
//! The physical counterpart, `scirust_compute::BufferBinding`, is a different
//! object entirely: it is generic over a backend buffer type, borrows that
//! buffer for a lifetime, and carries byte offsets and lengths that only a
//! runtime that has actually allocated memory can know. The mapping between the
//! two is a *runtime* responsibility, and deliberately not expressed here.
//!
//! The same separation applies to launch geometry. [`LogicalDispatch`] states an
//! iteration space; it is not `scirust_compute::LaunchConfig`, which carries
//! grid, block and shared-memory sizes that are tuned per backend (WGPU, for
//! instance, encodes workgroup size in the shader itself and requires a unit
//! block, while CUDA does not).
//!
//! # Why the lowered plan holds no compiled kernel
//!
//! A [`LogicalKernel`] is a *signature to be generated*, never generated code.
//! `scirust_compute::KernelModule` holds a byte payload in one concrete format
//! (`Reference`, `Wgsl`, `SpirV`, `Ptx`); producing one requires committing to a
//! target. WGSL and PTX emission is therefore necessarily target-specific and
//! stays out of this phase: a single logical plan is meant to be consumed by
//! several independent generators.
//!
//! ```text
//! LoweredPlan
//!   → ReferenceGenerator → KernelModule { format: Reference, .. }
//!   → WgslGenerator      → KernelModule { format: Wgsl, .. }
//!   → PtxGenerator       → KernelModule { format: Ptx, .. }
//! ```
//!
//! None of those generators exists yet. **A [`LoweredPlan`] is a description,
//! not an executable artefact**, and nothing in this crate can run a kernel.
//!
//! # Deterministic invariants of the lowered plan
//!
//! * Lowered instructions follow the canonical order of
//!   [`ExecutionPlan::instructions`].
//! * Arguments are ordered read operands first, in the exact order of
//!   [`Instruction::inputs`], then the single write result. The logical index of
//!   an argument equals its position.
//! * [`LogicalKernelId`] is the position of the kernel in
//!   [`LoweredPlan::kernels`], assigned on first use in canonical order.
//! * [`LogicalBindingId`] is the position of the entry in the external binding
//!   table supplied to [`KernelLowerer::lower`].
//! * Outputs keep the order declared by [`ExecutionPlan::outputs`].
//! * No identifier derives from an address, a randomly seeded hash, a hash-map
//!   iteration order, a device, or an execution context. Internal lookups use
//!   ordered maps and stably built vectors only.
//!
//! # Operations supported and rejected by lowering
//!
//! Supported in this phase:
//!
//! * `Add`, `Sub`, `Mul`, `Div` — [`KernelFamily::ElementwiseBinary`];
//! * `Relu`, `Exp`, `Log`, `Scale` — [`KernelFamily::ElementwiseUnary`];
//! * `Reshape` — [`KernelFamily::ShapeCopy`];
//! * `Transpose` — [`KernelFamily::Permute`].
//!
//! `Input` and `Constant` never become kernels; they feed the external binding
//! table ([`ExternalBindings`]) instead.
//!
//! `MatMul` is **rejected** with [`LoweringError::UnsupportedOperation`]. An
//! honest lowering of it needs rank validation, `M`/`K`/`N` extraction, a batch
//! policy, memory order, an accumulation policy with a stated numeric
//! determinism guarantee, a kernel strategy and possibly tiling constraints.
//! None of that is modelled here, and inventing an incomplete `MatMul` family
//! would advertise a capability that does not exist. It gets its own phase.
//!
//! Any operation added to the non-exhaustive [`scirust_tensor_ir::Operation`]
//! enum in the future is likewise rejected with a typed error. There is no
//! silent fallback, no no-op substitution and no partial lowering: a plan either
//! lowers completely or fails.
//!
//! # Semantic validation performed while lowering
//!
//! [`KernelLowerer::lower`] validates operand types, reshape consistency and
//! transpose permutations. These checks are *invariants of the canonical plan*
//! that no earlier pass enforces today — `Graph::validate` checks arity and
//! reference direction, not shapes. They live there because a code generator
//! must be able to trust them. A later semantic-validation pass of this crate
//! may own them instead, or share them; moving them would not change the
//! contract.
//!
//! # Legacy element-wise fusion compiler
//!
//! The original, still supported entry point is unrelated to the pipeline above:
//! a chain of element-wise operations (scale, bias, ReLU, …) is compiled into a
//! single [`FusedKernel`] that is evaluated in **one pass** over the data,
//! instead of materialising an intermediate tensor per operation. This is the
//! same memory-bandwidth win described for the tensor stack: fewer passes, fewer
//! temporaries.

#![forbid(unsafe_code)]

use scirust_tensor_core::TensorND;

mod canonical;
mod lowering;
mod memory;

pub use canonical::{CanonicalCompiler, CompileError, CompileStats, ExecutionPlan, Instruction};
pub use lowering::{
    BinaryKernel, ExternalBinding, ExternalBindings, ExternalValueKind, IdentifierSpace,
    KernelArgument, KernelArgumentAccess, KernelArgumentSource, KernelFamily, KernelLowerer,
    LogicalBindingId, LogicalDispatch, LogicalKernel, LogicalKernelId, LoweredInstruction,
    LoweredOutput, LoweredPlan, LoweringError, UnaryKernel,
};
pub use memory::{BufferSlot, BufferSlotSpec, MemoryPlan, ValueAllocation, ValueStorage};

// Re-export the contraction planner so multi-operand contractions and
// element-wise fusion share one entry point.
pub use scirust_tensor_contraction::ContractionPlan;

/// A single element-wise operation `f(x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementwiseOp {
    AddScalar(f32),
    MulScalar(f32),
    Relu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
}

impl ElementwiseOp {
    #[inline]
    fn eval(&self, x: f32) -> f32 {
        match self
        {
            ElementwiseOp::AddScalar(c) => x + c,
            ElementwiseOp::MulScalar(c) => x * c,
            ElementwiseOp::Relu => x.max(0.0),
            ElementwiseOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ElementwiseOp::Tanh => x.tanh(),
            ElementwiseOp::Exp => x.exp(),
            ElementwiseOp::Log => x.ln(),
        }
    }
}

/// Builds a fused element-wise pipeline.
#[derive(Default)]
pub struct GraphCompiler {
    ops: Vec<ElementwiseOp>,
}

impl GraphCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an element-wise op (builder style).
    pub fn op(mut self, op: ElementwiseOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Number of ops that will be fused into one pass.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Compile the chain into a single fused kernel.
    pub fn compile(self) -> FusedKernel {
        FusedKernel { ops: self.ops }
    }
}

/// A compiled, fused element-wise kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedKernel {
    ops: Vec<ElementwiseOp>,
}

impl FusedKernel {
    /// Apply all fused ops in a single pass over the input, allocating exactly
    /// one output buffer regardless of how many ops were fused.
    pub fn apply(&self, t: &TensorND) -> TensorND {
        let data = t
            .data
            .iter()
            .map(|&x| self.ops.iter().fold(x, |acc, op| op.eval(acc)))
            .collect();
        TensorND::new(data, t.shape.clone())
    }

    pub fn num_fused(&self) -> usize {
        self.ops.len()
    }
}

/// Node of an optimized operation graph, as described in `DESIGN_SCIRUST_TENSOR.md`.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    MatMul(usize, usize),
    Add(usize, usize),
    ReLU(usize),
    Fused(FusedOp),
}

/// A high-level fused operation.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOp {
    /// MatMul + Add Bias + ReLU fusion (or other element-wise ops).
    Linear {
        input_idx: usize,
        weight_idx: usize,
        bias_idx: Option<usize>,
        activation: Option<FusedKernel>,
    },
    /// Multi-operand contraction plan.
    OptimizedContraction(ContractionPlan),
}

/// A graph of optimized tensor operations.
pub struct TensorGraph {
    pub ops: Vec<TensorOp>,
    pub buffers: Vec<TensorND>,
}
