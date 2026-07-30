//! Typed representation of one generated Reference-format kernel.

use scirust_compute::{DType, KernelFormat, KernelModule, Shape};
use scirust_tensor_compile::LogicalKernelId;

use crate::codec;
use crate::error::{ReferenceDecodeError, ReferenceGenerationError};
use crate::format::{REFERENCE_MAX_RANK, ReferenceFormatVersion, ReferenceOpcode};

/// Shape of one operand or result, as it appears on the wire.
///
/// Dimensions and element count are `u64`, independent of the host
/// architecture's `usize` width. `elements` is redundant with the product of
/// `dims` — both the generator and the decoder verify the two agree, so it
/// doubles as an integrity check rather than trusted, unchecked bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceTensorLayout {
    dims: Vec<u64>,
    elements: u64,
}

impl ReferenceTensorLayout {
    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Dimensions, outermost first, matching the tensor's declared shape.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Total element count (the product of `dims`).
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    /// Builds a layout from a canonical `Shape`, checking every conversion.
    pub(crate) fn from_shape(
        shape: &Shape,
        kernel: LogicalKernelId,
    ) -> Result<Self, ReferenceGenerationError> {
        let rank = shape.rank();
        if rank > REFERENCE_MAX_RANK as usize
        {
            return Err(ReferenceGenerationError::RankTooLarge {
                kernel,
                rank,
                limit: REFERENCE_MAX_RANK,
            });
        }

        let mut dims = Vec::with_capacity(rank);
        for &dim in shape.dims()
        {
            dims.push(
                u64::try_from(dim)
                    .map_err(|_| ReferenceGenerationError::DimensionOverflow { kernel })?,
            );
        }

        let elements = shape
            .checked_num_elements()
            .map_err(|_| ReferenceGenerationError::ElementCountOverflow { kernel })?;
        let elements = u64::try_from(elements)
            .map_err(|_| ReferenceGenerationError::ElementCountOverflow { kernel })?;

        Ok(Self { dims, elements })
    }

    /// Reconstructs a layout already known to satisfy its invariants (used only
    /// by the decoder, after it has independently verified them).
    pub(crate) fn from_validated_parts(dims: Vec<u64>, elements: u64) -> Self {
        Self { dims, elements }
    }
}

/// Operation-specific attributes of one kernel, beyond its operand and result
/// shapes.
///
/// `Scale`'s factor is encoded bit-exact: it is the raw bit pattern of the
/// scalar (`Scalar::bits()`, checked to fit `u32` — Reference v1.0 supports
/// only `F32`, whose bits always do), never a textual or canonicalised form.
/// `-0.0`, `+0.0`, `+inf`, `-inf` and every NaN payload therefore round-trip
/// exactly; no NaN is ever canonicalised.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReferenceAttributes {
    /// No attribute payload: the elementwise binary kernels and `ShapeCopy`.
    None,
    /// `Scale`'s bit-exact factor.
    Scale { factor_bits: u32 },
    /// `Permute`'s axis list: `output.shape[i] == input.shape[axes[i]]`.
    Permute { axes: Vec<u16> },
}

/// One generated, encodable, decodable Reference-format kernel.
///
/// An artefact is a **description**: a signature to be interpreted, never
/// executed code. It contains no pointer, no closure, no trait object and
/// nothing that is not itself plain, comparable data.
///
/// Fields are private. The only ways to obtain an artefact are
/// [`ReferenceKernelGenerator`](crate::ReferenceKernelGenerator), which builds
/// one from a real `LogicalKernel`, and [`ReferenceKernelArtifact::decode`],
/// which builds one only after fully validating a byte stream. There is no
/// public unchecked constructor, so [`ReferenceKernelArtifact::encode`] can
/// stay infallible: every artefact that exists is already valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceKernelArtifact {
    kernel: LogicalKernelId,
    version: ReferenceFormatVersion,
    opcode: ReferenceOpcode,
    dtype: DType,
    operands: Vec<ReferenceTensorLayout>,
    result: ReferenceTensorLayout,
    attributes: ReferenceAttributes,
}

impl ReferenceKernelArtifact {
    /// Builds an artefact from parts already validated by the caller. Not
    /// exposed publicly: callers outside this crate can only reach an artefact
    /// through the generator or through [`Self::decode`].
    pub(crate) fn from_validated_parts(
        kernel: LogicalKernelId,
        version: ReferenceFormatVersion,
        opcode: ReferenceOpcode,
        dtype: DType,
        operands: Vec<ReferenceTensorLayout>,
        result: ReferenceTensorLayout,
        attributes: ReferenceAttributes,
    ) -> Self {
        Self {
            kernel,
            version,
            opcode,
            dtype,
            operands,
            result,
            attributes,
        }
    }

    /// The canonical [`LogicalKernelId`] this artefact was generated from.
    pub const fn kernel_id(&self) -> LogicalKernelId {
        self.kernel
    }

    /// The Reference format version this artefact was encoded (or decoded) as.
    pub const fn version(&self) -> ReferenceFormatVersion {
        self.version
    }

    pub const fn opcode(&self) -> ReferenceOpcode {
        self.opcode
    }

    /// Element type of every operand, the result, and (where applicable) the
    /// `Scale` factor.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn operands(&self) -> &[ReferenceTensorLayout] {
        &self.operands
    }

    pub const fn result(&self) -> &ReferenceTensorLayout {
        &self.result
    }

    pub const fn attributes(&self) -> &ReferenceAttributes {
        &self.attributes
    }

    /// Deterministic entry-point name, derived only from [`LogicalKernelId`]:
    /// `scirust_reference_kernel_<id>`.
    ///
    /// This name is *not* stored in the encoded bytes (see
    /// [`Self::encode`]) — it is always recomputed from the id, so there is
    /// exactly one name for a given kernel, never two sources of truth that
    /// could disagree.
    pub fn entry_point(&self) -> String {
        format!("scirust_reference_kernel_{}", self.kernel.get())
    }

    /// Canonical binary encoding of this artefact.
    ///
    /// Infallible: every [`ReferenceKernelArtifact`] that exists has already
    /// been validated, either by
    /// [`ReferenceKernelGenerator`](crate::ReferenceKernelGenerator) or by
    /// [`Self::decode`], so encoding it can never fail.
    pub fn encode(&self) -> Vec<u8> {
        codec::encode(self)
    }

    /// Parses, validates and reconstructs an artefact from its canonical
    /// encoding.
    ///
    /// This only decodes the description: it never executes the kernel it
    /// describes.
    pub fn decode(bytes: &[u8]) -> Result<Self, ReferenceDecodeError> {
        codec::decode(bytes)
    }

    /// Converts this artefact into a [`KernelModule`] of
    /// [`KernelFormat::Reference`].
    ///
    /// `entry_point` is [`Self::entry_point`] and `code` is exactly
    /// [`Self::encode`] — nothing else. The result is accepted today by
    /// `CpuComputeAdapter::compile` (which only stores the module), but no
    /// runtime yet knows how to launch it: `CpuComputeAdapter::launch` still
    /// returns `Unsupported` for every kernel. This module is generated,
    /// encoded and decodable — it is not, yet, executable.
    pub fn to_kernel_module(&self) -> Result<KernelModule, ReferenceGenerationError> {
        KernelModule::new(KernelFormat::Reference, self.entry_point(), self.encode()).map_err(
            |source| ReferenceGenerationError::InvalidKernelModule {
                kernel: self.kernel,
                source,
            },
        )
    }
}
