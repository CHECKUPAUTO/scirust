//! Serial CPU interpretation of one individual Reference kernel.
//!
//! # Numeric contract
//!
//! For every element-wise opcode (`Relu`, `Scale`, `Add`, `Sub`, `Mul`, `Div`)
//! this interpreter guarantees:
//!
//! * serial execution, one thread, no Rayon and no explicit SIMD;
//! * increasing element order, from index `0` upward;
//! * exactly one scalar operation per output element;
//! * no reassociation — no output element is the result of more than one
//!   arithmetic operation, so there is no accumulation order to specify;
//! * no `mul_add`, no explicit FMA;
//! * no fast-math, no reordering hints;
//! * no conversion to `f64`;
//! * plain Rust `f32` operators (`+`, `-`, `*`, `/`).
//!
//! That contract fixes *what this crate does*. It deliberately does **not**
//! claim that a compiled `f32` operator is bit-identical on every conforming
//! architecture — that is a property of the target and toolchain, not something
//! this crate can promise on their behalf. What is actually verified is
//! narrower and checked: golden-bit tests over the platforms this project's CI
//! covers.
//!
//! # NaN handling
//!
//! Bit-exact NaN preservation is guaranteed **only** where a value is returned
//! or copied without arithmetic:
//!
//! * the NaN branch of `Relu`, which returns its input untouched;
//! * `ShapeCopy`, which copies elements verbatim;
//! * `Permute`, which copies elements verbatim to permuted positions.
//!
//! For `Add`, `Sub`, `Mul`, `Div` and `Scale`, a NaN operand yields a NaN
//! result, but **the payload of that NaN is not specified** by this crate: IEEE
//! 754 leaves the propagated payload to the implementation. Tests assert
//! `is_nan()` for those cases and never a specific bit pattern.
//!
//! # Scope
//!
//! One kernel at a time. This module does not execute a
//! `scirust_tensor_compile::LoweredPlan`, does not resolve inter-kernel
//! dependencies, does not consult a memory plan, does not allocate slots, does
//! not bind external inputs or constants, and does not order instructions — all
//! of that belongs to a later plan runtime. It also touches no backend buffer,
//! no device and no `scirust_compute::ComputeBackend`; see the crate
//! documentation for the still-missing `CpuComputeAdapter` bridge.

use scirust_compute::{DType, KernelFormat, KernelModule};
use scirust_tensor_compile::LogicalKernelId;

use crate::artifact::{ReferenceAttributes, ReferenceKernelArtifact};
use crate::error::ReferenceExecutionError;
use crate::format::{REFERENCE_MAX_RANK, ReferenceOpcode};
use crate::invocation;

/// Upper bound on rank, as a `usize`, matching [`REFERENCE_MAX_RANK`].
const MAX_RANK: usize = REFERENCE_MAX_RANK as usize;

/// Attribute payload converted once, at preparation, into the exact form the
/// executor needs.
///
/// For `Permute` this includes the row-major strides of both the operand and the
/// result, computed once here with `checked_mul` rather than recomputed on every
/// invocation.
#[derive(Debug, Clone)]
pub(crate) enum PreparedAttributes {
    None,
    Scale {
        factor: f32,
    },
    Permute {
        input_dims: Vec<usize>,
        output_dims: Vec<usize>,
        input_strides: Vec<usize>,
        output_strides: Vec<usize>,
        permutation: Vec<usize>,
    },
}

/// A Reference artefact decoded, validated and converted into the form the CPU
/// interpreter executes.
///
/// Holds no pointer, no borrowed data, no device handle and no backend buffer:
/// it is `'static` and independent of any invocation. All fields are private,
/// and the only ways to build one are [`Self::from_artifact`] and
/// [`Self::from_kernel_module`], both of which reject anything they cannot
/// execute.
///
/// `Exp` and `Log` are rejected **here**, at preparation, rather than at
/// execution: a caller learns that a kernel is not executable before it ever
/// assembles operand buffers for it.
#[derive(Debug, Clone)]
pub struct PreparedReferenceKernel {
    kernel_id: LogicalKernelId,
    opcode: ReferenceOpcode,
    dtype: DType,
    operand_lengths: Vec<usize>,
    result_length: usize,
    prepared_attributes: PreparedAttributes,
}

impl PreparedReferenceKernel {
    /// Prepares a decoded artefact for CPU execution.
    ///
    /// Rejects a non-`F32` element type, `Exp`, `Log`, and any attribute payload
    /// inconsistent with the opcode. Converts every `u64` dimension and element
    /// count to `usize` through a checked conversion, and precomputes the
    /// `Permute` strides.
    pub fn from_artifact(
        artifact: &ReferenceKernelArtifact,
    ) -> Result<Self, ReferenceExecutionError> {
        let opcode = artifact.opcode();

        // Reject what this interpreter cannot evaluate, before any other work.
        match opcode
        {
            ReferenceOpcode::Exp | ReferenceOpcode::Log =>
            {
                return Err(ReferenceExecutionError::DeterministicMathUnavailable { opcode });
            },
            ReferenceOpcode::Relu
            | ReferenceOpcode::Scale
            | ReferenceOpcode::Add
            | ReferenceOpcode::Sub
            | ReferenceOpcode::Mul
            | ReferenceOpcode::Div
            | ReferenceOpcode::ShapeCopy
            | ReferenceOpcode::Permute =>
            {},
        }

        let dtype = artifact.dtype();
        if dtype != DType::F32
        {
            return Err(ReferenceExecutionError::UnsupportedDType { dtype });
        }

        let mut operand_lengths = Vec::with_capacity(artifact.operands().len());
        for layout in artifact.operands()
        {
            operand_lengths.push(checked_usize(layout.elements())?);
        }
        let result_length = checked_usize(artifact.result().elements())?;

        let prepared_attributes = match (opcode, artifact.attributes())
        {
            (ReferenceOpcode::Scale, ReferenceAttributes::Scale { factor_bits }) =>
            {
                // Bit pattern in, value out: `from_bits` is exact, so `-0.0`,
                // the infinities and every NaN payload survive unchanged.
                PreparedAttributes::Scale {
                    factor: f32::from_bits(*factor_bits),
                }
            },
            (ReferenceOpcode::Permute, ReferenceAttributes::Permute { axes }) =>
            {
                let operand = artifact
                    .operands()
                    .first()
                    .ok_or(ReferenceExecutionError::AttributeMismatch { opcode })?;

                let input_dims = checked_dims(operand.dims())?;
                let output_dims = checked_dims(artifact.result().dims())?;

                if axes.len() > MAX_RANK
                    || input_dims.len() > MAX_RANK
                    || output_dims.len() > MAX_RANK
                {
                    // Defensive: the encoder and decoder both cap rank at
                    // `REFERENCE_MAX_RANK`, so a valid artefact cannot get here.
                    return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
                }

                let mut permutation = Vec::with_capacity(axes.len());
                for &axis in axes
                {
                    permutation.push(usize::from(axis));
                }

                let input_strides = row_major_strides(&input_dims)?;
                let output_strides = row_major_strides(&output_dims)?;

                PreparedAttributes::Permute {
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                }
            },
            (
                ReferenceOpcode::Relu
                | ReferenceOpcode::Add
                | ReferenceOpcode::Sub
                | ReferenceOpcode::Mul
                | ReferenceOpcode::Div
                | ReferenceOpcode::ShapeCopy,
                ReferenceAttributes::None,
            ) => PreparedAttributes::None,
            _ =>
            {
                // Defensive: `codec` and `generator` both establish
                // opcode/attribute coherence before an artefact can exist.
                return Err(ReferenceExecutionError::AttributeMismatch { opcode });
            },
        };

        Ok(Self {
            kernel_id: artifact.kernel_id(),
            opcode,
            dtype,
            operand_lengths,
            result_length,
            prepared_attributes,
        })
    }

    /// Prepares a [`KernelModule`] for CPU execution.
    ///
    /// Performs, strictly in this order:
    ///
    /// 1. verify `module.format == KernelFormat::Reference` — WGSL, PTX, SPIR-V
    ///    and any future format are rejected with
    ///    [`ReferenceExecutionError::WrongKernelFormat`];
    /// 2. decode `module.code` as a Reference stream;
    /// 3. derive the canonical entry-point name from the decoded artefact;
    /// 4. compare it exactly with `module.entry_point`;
    /// 5. prepare the artefact ([`Self::from_artifact`]);
    /// 6. which in turn rejects `Exp` and `Log`.
    pub fn from_kernel_module(module: &KernelModule) -> Result<Self, ReferenceExecutionError> {
        if module.format != KernelFormat::Reference
        {
            return Err(ReferenceExecutionError::WrongKernelFormat {
                found: module.format,
            });
        }

        let artifact = ReferenceKernelArtifact::decode(&module.code)
            .map_err(ReferenceExecutionError::InvalidEncoding)?;

        let expected = artifact.entry_point();
        if module.entry_point != expected
        {
            return Err(ReferenceExecutionError::EntryPointMismatch {
                expected,
                found: module.entry_point.clone(),
            });
        }

        Self::from_artifact(&artifact)
    }

    /// The canonical [`LogicalKernelId`] this kernel was generated from.
    pub const fn kernel_id(&self) -> LogicalKernelId {
        self.kernel_id
    }

    pub const fn opcode(&self) -> ReferenceOpcode {
        self.opcode
    }

    /// Element type of every operand and of the result. Always `F32` for a
    /// kernel this crate agreed to prepare.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Element count each operand slice must hold, in operand order.
    pub fn operand_lengths(&self) -> &[usize] {
        &self.operand_lengths
    }

    /// Element count the output slice must hold.
    pub const fn result_length(&self) -> usize {
        self.result_length
    }

    pub(crate) const fn attributes(&self) -> &PreparedAttributes {
        &self.prepared_attributes
    }
}

/// Serial CPU interpreter for one prepared Reference kernel.
///
/// Stateless: it holds nothing between calls and performs no I/O, no
/// allocation proportional to the element count, and no device interaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferenceInterpreter;

impl ReferenceInterpreter {
    pub const fn new() -> Self {
        Self
    }

    /// Executes `prepared` over `operands`, writing into `output`.
    ///
    /// # Aliasing
    ///
    /// `operands: &[&[f32]]` and `output: &mut [f32]` cannot legally overlap:
    /// holding a shared borrow and an exclusive borrow of the same data at the
    /// same call site is rejected by the borrow checker. No runtime aliasing
    /// check is therefore performed — or needed — and no in-place execution is
    /// offered. The result is always written to a buffer the caller owns
    /// separately from every operand.
    ///
    /// # Failure atomicity
    ///
    /// Every check runs before the first write, so a rejected invocation leaves
    /// `output` bit-for-bit unchanged.
    pub fn execute(
        &self,
        prepared: &PreparedReferenceKernel,
        operands: &[&[f32]],
        output: &mut [f32],
    ) -> Result<(), ReferenceExecutionError> {
        invocation::validate(prepared, operands, output)?;

        let opcode = prepared.opcode();

        match opcode
        {
            ReferenceOpcode::Relu =>
            {
                let operand = first_operand(operands)?;
                for (target, &value) in output.iter_mut().zip(operand.iter())
                {
                    *target = relu(value);
                }
            },
            ReferenceOpcode::Scale =>
            {
                let operand = first_operand(operands)?;
                let PreparedAttributes::Scale { factor } = prepared.attributes()
                else
                {
                    return Err(ReferenceExecutionError::AttributeMismatch { opcode });
                };
                let factor = *factor;
                for (target, &value) in output.iter_mut().zip(operand.iter())
                {
                    *target = value * factor;
                }
            },
            ReferenceOpcode::Add =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs + rhs;
                }
            },
            ReferenceOpcode::Sub =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs - rhs;
                }
            },
            ReferenceOpcode::Mul =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs * rhs;
                }
            },
            ReferenceOpcode::Div =>
            {
                let (left, right) = binary_operands(operands)?;
                for (target, (&lhs, &rhs)) in output.iter_mut().zip(left.iter().zip(right.iter()))
                {
                    *target = lhs / rhs;
                }
            },
            ReferenceOpcode::ShapeCopy =>
            {
                let operand = first_operand(operands)?;
                // Validation already established both lengths; re-checked so
                // that `copy_from_slice`, which panics on a length mismatch,
                // provably cannot.
                if operand.len() != output.len()
                {
                    return Err(ReferenceExecutionError::OutputLengthMismatch {
                        expected: operand.len(),
                        actual: output.len(),
                    });
                }
                output.copy_from_slice(operand);
            },
            ReferenceOpcode::Permute =>
            {
                let operand = first_operand(operands)?;
                let PreparedAttributes::Permute {
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                } = prepared.attributes()
                else
                {
                    return Err(ReferenceExecutionError::AttributeMismatch { opcode });
                };

                execute_permute(
                    input_dims,
                    output_dims,
                    input_strides,
                    output_strides,
                    permutation,
                    operand,
                    output,
                )?;
            },
            ReferenceOpcode::Exp | ReferenceOpcode::Log =>
            {
                // Unreachable: preparation rejects both. Re-stated rather than
                // reached, so no opcode is ever silently treated as a no-op.
                return Err(ReferenceExecutionError::DeterministicMathUnavailable { opcode });
            },
        }

        Ok(())
    }
}

/// Rectified linear unit, with fully specified behaviour on every `f32` class.
///
/// * NaN: returned untouched, so its payload is preserved bit-for-bit.
/// * strictly positive, including `+inf`: returned unchanged.
/// * negative, `-0.0`, `+0.0` and `-inf`: `+0.0`.
///
/// `0.0 > 0.0` is false in IEEE 754, so `+0.0` takes the second branch and
/// becomes `+0.0` — the same value it started as. `f32::max` is deliberately not
/// used: its treatment of signed zeros and NaN does not match this
/// specification.
///
/// The two "return the input" cases (NaN, strictly positive) are one branch
/// rather than two identical ones purely so the condition reads as a single
/// predicate; the semantics are exactly those spelled out above.
#[inline]
fn relu(x: f32) -> f32 {
    if x.is_nan() || x > 0.0 { x } else { 0.0 }
}

/// The single operand of a unary opcode.
///
/// Defensive: [`invocation::validate`] has already checked the operand count.
fn first_operand<'a>(operands: &[&'a [f32]]) -> Result<&'a [f32], ReferenceExecutionError> {
    operands
        .first()
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)
}

/// The two operands of a binary opcode, in operand order.
///
/// Defensive, for the same reason as [`first_operand`].
fn binary_operands<'a>(
    operands: &[&'a [f32]],
) -> Result<(&'a [f32], &'a [f32]), ReferenceExecutionError> {
    let left = operands
        .first()
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
    let right = operands
        .get(1)
        .copied()
        .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;

    Ok((left, right))
}

/// Row-major strides of `dims`: `strides[rank - 1] == 1` and
/// `strides[i] == product(dims[i + 1 ..])`.
///
/// Built by multiplication only — never by division — with `checked_mul` at
/// every step. A shape containing a zero dimension yields zero strides for the
/// axes outside it; those strides are never used, because such a shape has zero
/// elements and [`execute_permute`] returns before touching them.
fn row_major_strides(dims: &[usize]) -> Result<Vec<usize>, ReferenceExecutionError> {
    let mut strides = vec![0usize; dims.len()];
    let mut accumulator = 1usize;

    for axis in (0..dims.len()).rev()
    {
        let stride = strides
            .get_mut(axis)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        *stride = accumulator;

        let dim = *dims
            .get(axis)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        accumulator = accumulator
            .checked_mul(dim)
            .ok_or(ReferenceExecutionError::IndexOverflow)?;
    }

    Ok(strides)
}

/// Copies each element to its permuted position, under the convention
/// `output.shape[i] == input.shape[permutation[i]]`.
///
/// For every output element, in increasing linear order:
///
/// 1. the output coordinates are reconstructed from the precomputed output
///    strides and dimensions;
/// 2. each coordinate is mapped to its input axis through `permutation`;
/// 3. the source linear index is accumulated from the precomputed input
///    strides, with `checked_mul` and `checked_add`;
/// 4. the source `f32` is copied verbatim — no arithmetic, so a NaN payload is
///    preserved bit-for-bit.
///
/// Coordinates live in a fixed `[usize; 32]` array, so nothing proportional to
/// the element count is allocated; the caller owns the only such buffer, the
/// output. Rank `0` is a one-element scalar: the coordinate loops are empty, the
/// source index stays `0`, and the single element is copied. A shape with a zero
/// dimension has zero elements, and the early return below means no division is
/// ever attempted — which is precisely why a zero dimension cannot divide by
/// zero here.
fn execute_permute(
    input_dims: &[usize],
    output_dims: &[usize],
    input_strides: &[usize],
    output_strides: &[usize],
    permutation: &[usize],
    operand: &[f32],
    output: &mut [f32],
) -> Result<(), ReferenceExecutionError> {
    if output.is_empty()
    {
        // Zero elements: nothing to permute, and no stride is ever divided by.
        return Ok(());
    }

    if permutation.len() != output_dims.len()
        || input_dims.len() != output_dims.len()
        || input_strides.len() != input_dims.len()
        || output_strides.len() != output_dims.len()
        || output_dims.len() > MAX_RANK
    {
        // Defensive: preparation built all five from one validated artefact.
        return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
    }

    let mut output_coordinates = [0usize; MAX_RANK];

    for (output_index, target) in output.iter_mut().enumerate()
    {
        // 1. Reconstruct the output coordinates.
        for (axis, (&stride, &dim)) in output_strides.iter().zip(output_dims.iter()).enumerate()
        {
            if stride == 0 || dim == 0
            {
                // Defensive: `output` is non-empty, so every output dimension
                // is non-zero and so is every stride.
                return Err(ReferenceExecutionError::InternalIndexOutOfBounds);
            }

            let coordinate = (output_index / stride) % dim;
            let slot = output_coordinates
                .get_mut(axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
            *slot = coordinate;
        }

        // 2 and 3. Map through the permutation and accumulate the source index.
        let mut source_index = 0usize;
        for (axis, &input_axis) in permutation.iter().enumerate()
        {
            let coordinate = *output_coordinates
                .get(axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
            let input_stride = *input_strides
                .get(input_axis)
                .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;

            let term = coordinate
                .checked_mul(input_stride)
                .ok_or(ReferenceExecutionError::IndexOverflow)?;
            source_index = source_index
                .checked_add(term)
                .ok_or(ReferenceExecutionError::IndexOverflow)?;
        }

        // 4. Copy the source element verbatim.
        let value = *operand
            .get(source_index)
            .ok_or(ReferenceExecutionError::InternalIndexOutOfBounds)?;
        *target = value;
    }

    Ok(())
}

/// Checked `u64 -> usize` conversion for one dimension or element count.
fn checked_usize(value: u64) -> Result<usize, ReferenceExecutionError> {
    usize::try_from(value)
        .map_err(|_| ReferenceExecutionError::DimensionConversionOverflow { dimension: value })
}

/// Checked `u64 -> usize` conversion for a whole dimension list.
fn checked_dims(dims: &[u64]) -> Result<Vec<usize>, ReferenceExecutionError> {
    let mut converted = Vec::with_capacity(dims.len());
    for &dim in dims
    {
        converted.push(checked_usize(dim)?);
    }

    Ok(converted)
}
