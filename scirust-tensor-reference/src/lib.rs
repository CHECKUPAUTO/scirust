//! Deterministic Reference-format kernel generation and serial CPU
//! interpretation for SciRust.
//!
//! # Where this sits
//!
//! ```text
//! scirust-tensor-ir
//!         |
//! scirust-tensor-compile   (Graph -> ExecutionPlan -> MemoryPlan -> LoweredPlan)
//!         |
//! scirust-tensor-reference (this crate: LoweredPlan -> artefacts -> CPU execution)
//!         |
//! scirust-compute          (KernelModule, KernelFormat::Reference)
//! ```
//!
//! # The four objects this crate owns
//!
//! | Type | Role |
//! |---|---|
//! | [`ReferenceKernelArtifact`] | one logical kernel in the canonical binary format: encode, decode, wrap as a [`scirust_compute::KernelModule`] |
//! | [`PreparedReferenceKernel`] | that artefact decoded, validated and converted into the form the CPU executes |
//! | [`ReferenceInterpreter`] | serial execution of **one individual** kernel over `f32` slices |
//! | `CpuComputeAdapter` (in `scirust-gpu`) | the physical integration, **still absent** — see below |
//!
//! ```text
//! LoweredPlan
//!   -> ReferenceKernelGenerator -> ReferenceKernelArtifact
//!   -> ReferenceKernelArtifact::to_kernel_module -> KernelModule{ format: Reference }
//!   -> PreparedReferenceKernel::from_kernel_module
//!   -> ReferenceInterpreter::execute(&prepared, &[&[f32]], &mut [f32])
//! ```
//!
//! # What this crate does not do
//!
//! * **No plan execution.** [`ReferenceInterpreter`] runs one kernel. It does
//!   not walk a `scirust_tensor_compile::LoweredPlan`, resolve inter-kernel
//!   dependencies, consult a memory plan, allocate slots, bind external inputs
//!   or constants, or order instructions. That is a later plan runtime.
//! * **No backend buffer, no device, no physical allocation.** Operands and
//!   results are plain `&[f32]` / `&mut [f32]` the caller owns. Nothing here
//!   touches `scirust_compute::BufferBinding`, a `MemorySpace`, or a device.
//! * **No `ComputeBackend` integration.** `CpuComputeAdapter::compile` still
//!   stores a Reference module without parsing it, and
//!   `CpuComputeAdapter::launch` still returns
//!   `Unsupported("CPU reference kernel launch ABI is not implemented")`.
//!   Nothing in this crate changes either. Wiring the two together — mapping
//!   `BufferBinding`s onto `f32` slices and calling this interpreter — is a
//!   separate phase.
//! * **No WGSL and no PTX.** Those are separate target-specific generators.
//!
//! # `Exp` and `Log` are not executable
//!
//! [`PreparedReferenceKernel`] **rejects** the `Exp` and `Log` opcodes with
//! [`ReferenceExecutionError::DeterministicMathUnavailable`], at preparation
//! time. `f32::exp` and `f32::ln` delegate to the platform's math library;
//! neither Rust nor IEEE 754 requires a correctly-rounded result for
//! transcendental functions, and implementations differ in practice across
//! operating systems, architectures and versions. Using them would contradict
//! the determinism this format exists to provide, so the artefact format keeps
//! both opcodes — a generator may emit them — while this interpreter refuses
//! them outright rather than returning a number it cannot reproduce.
//!
//! # Numeric guarantees, and their limits
//!
//! The element-wise opcodes run serially, in increasing index order, one scalar
//! `f32` operation per element, with no reassociation, no `mul_add`, no explicit
//! FMA, no explicit SIMD, no Rayon, no fast-math and no widening to `f64`. That
//! is the contract this crate implements and tests.
//!
//! This crate does **not** claim a compiled `f32` operator is bit-identical on
//! every conforming architecture; that depends on target and toolchain, not on
//! this code. The verified guarantee is narrower: golden-bit tests over the
//! platforms this project's CI covers.
//!
//! NaN payloads are preserved bit-for-bit **only** where a value is returned or
//! copied without arithmetic — the NaN branch of `Relu`, [`ReferenceOpcode::ShapeCopy`]
//! and [`ReferenceOpcode::Permute`]. For `Add`, `Sub`, `Mul`, `Div` and `Scale`,
//! a NaN operand produces a NaN result but the propagated payload is
//! unspecified, exactly as IEEE 754 leaves it.
//!
//! `Scale`'s factor, by contrast, is carried bit-exact end to end: it travels as
//! a raw `u32` bit pattern, so `-0.0`, `+0.0`, `+inf`, `-inf` and every NaN
//! payload reach the multiplication unchanged, with no canonicalisation.
//!
//! # Deterministic invariants of generation
//!
//! * [`ReferenceKernelGenerator::generate`] preserves the exact order of
//!   `LoweredPlan::kernels`, and independently re-verifies that each
//!   [`scirust_tensor_compile::LogicalKernelId`] equals its position.
//! * An artefact's entry-point name is derived *only* from its
//!   `LogicalKernelId` (`scirust_reference_kernel_<id>`) — it is not stored in
//!   the encoded bytes, so there is exactly one name for a given kernel and no
//!   second source of truth that could disagree with it.
//!   [`PreparedReferenceKernel::from_kernel_module`] re-derives it and rejects a
//!   module whose `entry_point` disagrees.
//! * Two generations of the same `LoweredPlan` produce identical artefacts,
//!   identical entry-point names, identical encoded bytes and identical
//!   `KernelModule`s, in identical order — independent of memory addresses,
//!   thread count, or any hash-map iteration order (this crate uses none).
//!
//! # Supported and rejected operations
//!
//! The artefact format covers every kernel family
//! `scirust_tensor_compile::KernelLowerer` produces: element-wise unary
//! (`Relu`, `Exp`, `Log`, `Scale`), element-wise binary (`Add`, `Sub`, `Mul`,
//! `Div`), `ShapeCopy` (the lowering of `Reshape`) and `Permute` (the lowering
//! of `Transpose`). The CPU interpreter executes all of them **except `Exp` and
//! `Log`**, as explained above.
//!
//! `MatMul` never reaches this crate: `scirust_tensor_compile`'s lowering phase
//! rejects it before a `LoweredPlan` can exist. Any kernel family this crate
//! does not recognise is rejected with
//! [`ReferenceGenerationError::UnsupportedKernelFamily`]. There is no silent
//! fallback, no no-op substitution and no partial artefact anywhere in this
//! crate.
//!
//! Reference v1.0 supports exactly `DType::F32`. Every other `DType` SciRust
//! declares has a *reserved, stable wire tag*, so a decoder can always
//! distinguish a tag it has never heard of from a `DType` it recognises but does
//! not support — but generating, decoding or executing a kernel of any other
//! type is rejected. No type is ever silently narrowed to `F32`.
//!
//! # Safety and failure discipline
//!
//! The crate forbids `unsafe`. No preparation or execution path contains
//! `panic!`, `unwrap()`, `expect()` or unchecked indexing: every buffer, shape
//! and stride access goes through `get`/`get_mut`, every index arithmetic step
//! is checked, and a broken internal invariant surfaces as a typed error rather
//! than an abort.
//!
//! [`ReferenceInterpreter::execute`] validates an invocation completely before
//! writing anything, so a rejected call leaves the caller's output buffer
//! bit-for-bit unchanged.

#![forbid(unsafe_code)]

mod artifact;
mod codec;
mod error;
mod format;
mod generator;
mod interpreter;
mod invocation;

pub use artifact::{ReferenceAttributes, ReferenceKernelArtifact, ReferenceTensorLayout};
pub use error::{ReferenceDecodeError, ReferenceExecutionError, ReferenceGenerationError};
pub use format::{
    REFERENCE_FORMAT_VERSION, REFERENCE_MAGIC, REFERENCE_MAX_RANK, ReferenceFormatVersion,
    ReferenceOpcode,
};
pub use generator::{ReferenceKernelGenerator, ReferenceKernelSet};
pub use interpreter::{PreparedReferenceKernel, ReferenceInterpreter};

#[cfg(test)]
mod tests;
