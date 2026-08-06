//! Canonical little-endian byte encoding and validating decoding of a
//! [`ReferenceKernelArtifact`].
//!
//! # Layout
//!
//! ```text
//! MAGIC              25 bytes, b"SCIRUST-REFERENCE-KERNEL\0"
//! version_major      u16
//! version_minor      u16
//! kernel_id          u32
//! opcode             u16
//! dtype_tag          u16
//! operand_count      u16
//! result_count       u16   (always 1 in this format version)
//! operands           operand_count x LAYOUT
//! result             LAYOUT
//! attribute_tag      u16
//! attribute_payload  size determined by opcode / attribute_tag
//! <end of stream — any remaining byte is rejected>
//!
//! LAYOUT:
//!   rank             u16
//!   dimensions       rank x u64
//!   elements         u64
//!
//! attribute_payload:
//!   0x0000 None      (empty)
//!   0x0001 Scale     factor_bits: u32
//!   0x0002 Permute   permutation_rank: u16, axes: permutation_rank x u16
//! ```
//!
//! # Checked conversions
//!
//! Every `usize -> u64` and `usize -> u16` conversion performed while encoding
//! is checked (`try_from`), even though none can fail on any platform Rust
//! actually targets today (`usize` never exceeds 64 bits there). This crate
//! deliberately keeps [`ReferenceTensorLayout`](crate::ReferenceTensorLayout)
//! and every wire integer in `u16`/`u32`/`u64` end to end, so no
//! `u64 -> usize` narrowing conversion is needed while decoding a field's
//! *value* — dimensions, element counts and factor bits are never converted
//! back to `usize`. The only widening conversions performed while decoding
//! (`u16 -> usize` for a rank or an axis, to size a `Vec` or index one) are
//! lossless on every platform Rust supports, since `usize` is always at least
//! 16 bits wide; they are not wrapped in a `Result` because there is no
//! failure they could report.

use scirust_compute::DType;

use crate::artifact::{ReferenceAttributes, ReferenceKernelArtifact, ReferenceTensorLayout};
use crate::error::ReferenceDecodeError;
use crate::format::{
    self, ATTRIBUTE_TAG_NONE, ATTRIBUTE_TAG_PERMUTE, ATTRIBUTE_TAG_SCALE, REFERENCE_FORMAT_VERSION,
    REFERENCE_MAGIC, REFERENCE_MAX_RANK, ReferenceFormatVersion, ReferenceOpcode,
};

/// Appends fixed-width little-endian fields to a growing byte buffer.
struct ByteWriter {
    bytes: Vec<u8>,
}

impl ByteWriter {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn raw(&mut self, data: &[u8]) {
        self.bytes.extend_from_slice(data);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Encodes `artifact` as its canonical Reference-format byte stream.
///
/// Infallible: an artefact's invariants were already checked when it was built
/// (by the generator or by [`decode`]), so writing it out cannot fail.
pub(crate) fn encode(artifact: &ReferenceKernelArtifact) -> Vec<u8> {
    let mut writer = ByteWriter::new();

    writer.raw(REFERENCE_MAGIC.as_slice());
    writer.u16(artifact.version().major());
    writer.u16(artifact.version().minor());
    writer.u32(artifact.kernel_id().get());
    writer.u16(artifact.opcode().code());
    writer.u16(format::dtype_to_tag(artifact.dtype()));

    // `operands().len()` and `1` (the result count) both fit `u16`: every
    // opcode's arity in this format version is 1 or 2, and there is always
    // exactly one result.
    writer.u16(artifact.operands().len() as u16);
    writer.u16(1);

    for operand in artifact.operands()
    {
        write_layout(&mut writer, operand);
    }
    write_layout(&mut writer, artifact.result());

    match artifact.attributes()
    {
        ReferenceAttributes::None => writer.u16(ATTRIBUTE_TAG_NONE),
        ReferenceAttributes::Scale { factor_bits } =>
        {
            writer.u16(ATTRIBUTE_TAG_SCALE);
            writer.u32(*factor_bits);
        },
        ReferenceAttributes::Permute { axes } =>
        {
            writer.u16(ATTRIBUTE_TAG_PERMUTE);
            // Bounded by `REFERENCE_MAX_RANK` at construction time.
            writer.u16(axes.len() as u16);
            for &axis in axes
            {
                writer.u16(axis);
            }
        },
    }

    writer.into_bytes()
}

fn write_layout(writer: &mut ByteWriter, layout: &ReferenceTensorLayout) {
    // Bounded by `REFERENCE_MAX_RANK` at construction time.
    writer.u16(layout.dims().len() as u16);
    for &dim in layout.dims()
    {
        writer.u64(dim);
    }
    writer.u64(layout.elements());
}

/// A cursor over an untrusted byte slice, with bounds-checked reads.
struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ReferenceDecodeError> {
        let available = self.remaining();
        if available < len
        {
            return Err(ReferenceDecodeError::TruncatedInput {
                needed: len,
                available,
            });
        }

        let slice = &self.bytes[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    fn u16(&mut self) -> Result<u16, ReferenceDecodeError> {
        let slice = self.take(2)?;
        let mut buffer = [0u8; 2];
        buffer.copy_from_slice(slice);
        Ok(u16::from_le_bytes(buffer))
    }

    fn u32(&mut self) -> Result<u32, ReferenceDecodeError> {
        let slice = self.take(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(slice);
        Ok(u32::from_le_bytes(buffer))
    }

    fn u64(&mut self) -> Result<u64, ReferenceDecodeError> {
        let slice = self.take(8)?;
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(slice);
        Ok(u64::from_le_bytes(buffer))
    }
}

/// Parses, validates and reconstructs an artefact. Never executes anything.
pub(crate) fn decode(bytes: &[u8]) -> Result<ReferenceKernelArtifact, ReferenceDecodeError> {
    let mut reader = ByteReader::new(bytes);

    let magic = reader.take(REFERENCE_MAGIC.len())?;
    if magic != REFERENCE_MAGIC.as_slice()
    {
        return Err(ReferenceDecodeError::InvalidMagic);
    }

    let major = reader.u16()?;
    let minor = reader.u16()?;
    if major != REFERENCE_FORMAT_VERSION.major()
    {
        return Err(ReferenceDecodeError::IncompatibleMajorVersion {
            found: major,
            supported: REFERENCE_FORMAT_VERSION.major(),
        });
    }
    if minor != REFERENCE_FORMAT_VERSION.minor()
    {
        return Err(ReferenceDecodeError::UnsupportedMinorVersion {
            found: minor,
            supported: REFERENCE_FORMAT_VERSION.minor(),
        });
    }

    let kernel_id = reader.u32()?;

    let opcode_code = reader.u16()?;
    let opcode = ReferenceOpcode::from_code(opcode_code)
        .ok_or(ReferenceDecodeError::UnknownOpcode { code: opcode_code })?;

    let dtype_tag = reader.u16()?;
    let dtype = format::dtype_from_tag(dtype_tag)
        .ok_or(ReferenceDecodeError::UnknownDType { tag: dtype_tag })?;
    if dtype != DType::F32
    {
        return Err(ReferenceDecodeError::UnsupportedDType { dtype });
    }

    let operand_count = reader.u16()?;
    let result_count = reader.u16()?;
    if result_count != 1
    {
        return Err(ReferenceDecodeError::InvalidResultCount {
            found: result_count,
        });
    }

    let expected_operands = opcode.operand_count();
    if operand_count != expected_operands
    {
        return Err(ReferenceDecodeError::UnexpectedOperandCount {
            opcode,
            expected: expected_operands as usize,
            actual: operand_count as usize,
        });
    }

    let mut operands = Vec::with_capacity(operand_count as usize);
    for _ in 0..operand_count
    {
        operands.push(read_layout(&mut reader)?);
    }
    let result = read_layout(&mut reader)?;

    let attribute_tag = reader.u16()?;
    let attributes = read_attributes(&mut reader, opcode, attribute_tag, result.rank())?;

    if reader.remaining() != 0
    {
        return Err(ReferenceDecodeError::TrailingBytes {
            remaining: reader.remaining(),
        });
    }

    validate_opcode_shapes(opcode, &operands, &result, &attributes)?;

    Ok(ReferenceKernelArtifact::from_validated_parts(
        scirust_tensor_compile::LogicalKernelId::new(kernel_id),
        ReferenceFormatVersion::new(major, minor),
        opcode,
        dtype,
        operands,
        result,
        attributes,
    ))
}

fn read_layout(reader: &mut ByteReader<'_>) -> Result<ReferenceTensorLayout, ReferenceDecodeError> {
    let rank = reader.u16()?;
    if rank > REFERENCE_MAX_RANK
    {
        return Err(ReferenceDecodeError::RankTooLarge {
            rank,
            limit: REFERENCE_MAX_RANK,
        });
    }

    // Check the stream actually has enough bytes left before allocating a
    // `Vec` sized by an attacker-controlled `rank`.
    let needed = rank as usize * 8 + 8;
    if reader.remaining() < needed
    {
        return Err(ReferenceDecodeError::TruncatedInput {
            needed,
            available: reader.remaining(),
        });
    }

    let mut dims = Vec::with_capacity(rank as usize);
    for _ in 0..rank
    {
        dims.push(reader.u64()?);
    }
    let elements = reader.u64()?;

    let computed = dims
        .iter()
        .try_fold(1u64, |accumulator, &dim| accumulator.checked_mul(dim))
        .ok_or(ReferenceDecodeError::ElementCountOverflow)?;
    if computed != elements
    {
        return Err(ReferenceDecodeError::ElementCountMismatch);
    }

    Ok(ReferenceTensorLayout::from_validated_parts(dims, elements))
}

fn read_attributes(
    reader: &mut ByteReader<'_>,
    opcode: ReferenceOpcode,
    attribute_tag: u16,
    result_rank: usize,
) -> Result<ReferenceAttributes, ReferenceDecodeError> {
    match (opcode, attribute_tag)
    {
        (ReferenceOpcode::Scale, ATTRIBUTE_TAG_SCALE) =>
        {
            let factor_bits = reader.u32()?;
            Ok(ReferenceAttributes::Scale { factor_bits })
        },
        (ReferenceOpcode::Permute, ATTRIBUTE_TAG_PERMUTE) =>
        {
            let permutation_rank = reader.u16()?;
            if permutation_rank as usize != result_rank
            {
                return Err(ReferenceDecodeError::InvalidPermutation);
            }

            let needed = permutation_rank as usize * 2;
            if reader.remaining() < needed
            {
                return Err(ReferenceDecodeError::TruncatedInput {
                    needed,
                    available: reader.remaining(),
                });
            }

            let mut axes = Vec::with_capacity(permutation_rank as usize);
            for _ in 0..permutation_rank
            {
                axes.push(reader.u16()?);
            }
            Ok(ReferenceAttributes::Permute { axes })
        },
        (ReferenceOpcode::Scale | ReferenceOpcode::Permute, _) =>
        {
            Err(ReferenceDecodeError::AttributeOpcodeMismatch { opcode })
        },
        (_, ATTRIBUTE_TAG_NONE) => Ok(ReferenceAttributes::None),
        (_, ATTRIBUTE_TAG_SCALE | ATTRIBUTE_TAG_PERMUTE) =>
        {
            Err(ReferenceDecodeError::AttributeOpcodeMismatch { opcode })
        },
        (_, tag) => Err(ReferenceDecodeError::UnknownAttributeTag { tag }),
    }
}

/// Cross-checks operand/result shapes against the opcode's semantics. Each
/// individual layout's own element count was already checked in
/// [`read_layout`]; this checks *between* layouts.
fn validate_opcode_shapes(
    opcode: ReferenceOpcode,
    operands: &[ReferenceTensorLayout],
    result: &ReferenceTensorLayout,
    attributes: &ReferenceAttributes,
) -> Result<(), ReferenceDecodeError> {
    match opcode
    {
        ReferenceOpcode::Relu
        | ReferenceOpcode::Exp
        | ReferenceOpcode::Log
        | ReferenceOpcode::Scale =>
        {
            if operands[0].dims() != result.dims()
            {
                return Err(ReferenceDecodeError::ShapeMismatch);
            }
        },
        ReferenceOpcode::Add
        | ReferenceOpcode::Sub
        | ReferenceOpcode::Mul
        | ReferenceOpcode::Div =>
        {
            if operands[0].dims() != result.dims() || operands[1].dims() != result.dims()
            {
                return Err(ReferenceDecodeError::ShapeMismatch);
            }
        },
        ReferenceOpcode::ShapeCopy =>
        {
            if operands[0].elements() != result.elements()
            {
                return Err(ReferenceDecodeError::ElementCountMismatch);
            }
        },
        ReferenceOpcode::Permute =>
        {
            let ReferenceAttributes::Permute { axes } = attributes
            else
            {
                return Err(ReferenceDecodeError::AttributeOpcodeMismatch { opcode });
            };
            validate_permutation(&operands[0], result, axes)?;
        },
    }

    Ok(())
}

/// Verifies `axes` is a valid rank permutation and that
/// `output.shape[i] == input.shape[axes[i]]` for every `i`.
fn validate_permutation(
    input: &ReferenceTensorLayout,
    output: &ReferenceTensorLayout,
    axes: &[u16],
) -> Result<(), ReferenceDecodeError> {
    let rank = input.rank();
    if axes.len() != rank || output.rank() != rank
    {
        return Err(ReferenceDecodeError::InvalidPermutation);
    }

    let mut seen = vec![false; rank];
    for &axis in axes
    {
        let index = axis as usize;
        if index >= rank || seen[index]
        {
            return Err(ReferenceDecodeError::InvalidPermutation);
        }
        seen[index] = true;
    }

    for (position, &axis) in axes.iter().enumerate()
    {
        if output.dims()[position] != input.dims()[axis as usize]
        {
            return Err(ReferenceDecodeError::ShapeMismatch);
        }
    }

    Ok(())
}
