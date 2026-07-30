//! Deterministic Reference-format kernel generation for SciRust.
//!
//! # Where this sits
//!
//! ```text
//! scirust-tensor-ir
//!         |
//! scirust-tensor-compile   (Graph -> ExecutionPlan -> MemoryPlan -> LoweredPlan)
//!         |
//! scirust-tensor-reference (this crate: LoweredPlan -> Reference artefacts)
//!         |
//! scirust-compute          (KernelModule, KernelFormat::Reference)
//! ```
//!
//! `scirust_tensor_compile::LoweredPlan` is a backend-neutral *description* of
//! a computation: which [`scirust_tensor_compile::LogicalKernel`] each
//! canonical node lowers to, and which logical arguments feed each dispatch.
//! It contains no compiled kernel, no physical buffer, no device and no
//! runtime.
//!
//! This crate is the first generator that turns those logical kernels into a
//! concrete, target-specific artefact: a deterministic Reference-format byte
//! encoding, wrapped in a [`scirust_compute::KernelModule`] tagged
//! [`scirust_compute::KernelFormat::Reference`].
//!
//! **This is not an interpreter.** [`ReferenceKernelArtifact::decode`] parses,
//! validates and reconstructs the typed description above — it never
//! evaluates an operation, reads a buffer or performs arithmetic. Producing
//! and round-tripping a Reference [`scirust_compute::KernelModule`] does not
//! make a plan executable: `CpuComputeAdapter::compile` in `scirust-gpu`
//! accepts a `KernelFormat::Reference` module today by storing it verbatim,
//! and `CpuComputeAdapter::launch` still returns
//! `Unsupported("CPU reference kernel launch ABI is not implemented")` for
//! every kernel. Nothing in this crate changes that. WGSL and PTX generation
//! are separate, later, target-specific phases and are out of scope here.
//!
//! # Format versions and the four planes
//!
//! | Plane | Type | Contains |
//! |---|---|---|
//! | Canonical IR | `scirust_tensor_ir::Graph` | graph structure, no payload |
//! | Execution plan | `scirust_tensor_compile::ExecutionPlan` | topological order |
//! | Memory plan | `scirust_tensor_compile::MemoryPlan` | logical buffer slots |
//! | Lowered plan | `scirust_tensor_compile::LoweredPlan` | logical kernels and arguments |
//! | **Reference artefact** | [`ReferenceKernelArtifact`] | one logical kernel's target-specific encoding |
//!
//! # Logical binding versus `KernelModule`
//!
//! A Reference artefact's operands and result are described by
//! [`ReferenceTensorLayout`] — dtype (implied by the artefact's single
//! [`ReferenceKernelArtifact::dtype`]), rank, dimensions and element count.
//! This is not a physical buffer binding: there is no pointer, no byte offset,
//! no device and no lifetime anywhere in this crate. The eventual mapping from
//! a logical argument to a physical `scirust_compute::BufferBinding` is a
//! future CPU-interpreter and runtime responsibility, not this generator's.
//!
//! # Deterministic invariants
//!
//! * [`ReferenceKernelGenerator::generate`] preserves the exact order of
//!   `LoweredPlan::kernels`, and independently re-verifies that each
//!   [`scirust_tensor_compile::LogicalKernelId`] equals its position.
//! * An artefact's entry-point name is derived *only* from its
//!   `LogicalKernelId` (`scirust_reference_kernel_<id>`) — it is not stored in
//!   the encoded bytes, so there is exactly one name for a given kernel and no
//!   second source of truth that could disagree with it.
//! * Two generations of the same `LoweredPlan` produce identical artefacts,
//!   identical entry-point names, identical encoded bytes and identical
//!   `KernelModule`s, in identical order — independent of memory addresses,
//!   thread count, or any hash-map iteration order (this crate uses none).
//! * `Scale`'s factor is encoded bit-exact from `Scalar::bits()`; `-0.0`,
//!   `+0.0`, `+inf`, `-inf` and every NaN payload round-trip exactly, and no
//!   NaN is ever canonicalised.
//!
//! # Supported and rejected operations
//!
//! Every kernel family `scirust_tensor_compile::KernelLowerer` produces today:
//! elementwise unary (`Relu`, `Exp`, `Log`, `Scale`), elementwise binary
//! (`Add`, `Sub`, `Mul`, `Div`), `ShapeCopy` (the lowering of `Reshape`) and
//! `Permute` (the lowering of `Transpose`). `MatMul` never reaches this crate:
//! `scirust_tensor_compile`'s lowering phase already rejects it before a
//! `LoweredPlan` can exist. Any kernel family this crate does not recognise —
//! including a future, currently non-exhaustive addition — is rejected with
//! [`ReferenceGenerationError::UnsupportedKernelFamily`]; there is no silent
//! fallback, no-op, or partial artefact.
//!
//! Reference v1.0 supports exactly `DType::F32`. Every other `DType` SciRust
//! declares has a *reserved, stable wire tag* (see the format module), so a
//! decoder can always distinguish a tag it has never heard of from a `DType`
//! it recognises but does not support — but generating or decoding a kernel of
//! any other type is rejected with `UnsupportedDType`. No type is ever
//! silently narrowed to `F32`.
//!
//! # Format documentation
//!
//! See the `format` and `codec` modules (re-exported flat from here) for the
//! exact wire layout, the opcode and `DType` tag tables, and the versioning
//! policy.

#![forbid(unsafe_code)]

mod artifact;
mod codec;
mod error;
mod format;
mod generator;

pub use artifact::{ReferenceAttributes, ReferenceKernelArtifact, ReferenceTensorLayout};
pub use error::{ReferenceDecodeError, ReferenceGenerationError};
pub use format::{
    REFERENCE_FORMAT_VERSION, REFERENCE_MAGIC, REFERENCE_MAX_RANK, ReferenceFormatVersion,
    ReferenceOpcode,
};
pub use generator::{ReferenceKernelGenerator, ReferenceKernelSet};

#[cfg(test)]
mod tests;
