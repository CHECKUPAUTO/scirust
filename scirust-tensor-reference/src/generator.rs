//! Deterministic generation of Reference-format artefacts from a
//! [`LoweredPlan`].

use scirust_compute::{DType, KernelModule};
use scirust_tensor_compile::{
    BinaryKernel, KernelFamily, LogicalKernel, LogicalKernelId, LoweredPlan, UnaryKernel,
};

use crate::artifact::{ReferenceAttributes, ReferenceKernelArtifact, ReferenceTensorLayout};
use crate::error::ReferenceGenerationError;
use crate::format::{REFERENCE_FORMAT_VERSION, ReferenceOpcode};

/// Deterministic generator of Reference-format artefacts.
///
/// Generation performs no I/O, holds no state between calls, and always
/// produces the same artefacts, names and bytes for the same
/// [`LoweredPlan`] — see the crate documentation for the exact determinism
/// guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReferenceKernelGenerator;

impl ReferenceKernelGenerator {
    pub const fn new() -> Self {
        Self
    }

    /// Generates a Reference artefact for every kernel of `plan`, preserving
    /// the exact order of [`LoweredPlan::kernels`].
    ///
    /// Independently re-validates, per kernel, that
    /// [`LogicalKernelId`] equals its position in the plan — even though a
    /// plan produced by `KernelLowerer::lower` already guarantees this by
    /// construction. See [`ReferenceGenerationError`] for which rejections
    /// this makes possible today versus which remain purely defensive.
    pub fn generate(
        self,
        plan: &LoweredPlan,
    ) -> Result<ReferenceKernelSet, ReferenceGenerationError> {
        let kernels = plan.kernels();
        let count = kernels.len();
        let mut seen = vec![false; count];
        let mut artifacts = Vec::with_capacity(count);

        for (position, kernel) in kernels.iter().enumerate()
        {
            let expected = LogicalKernelId::new(
                u32::try_from(position)
                    .map_err(|_| ReferenceGenerationError::TooManyKernels { count })?,
            );

            let raw = kernel.id().get() as usize;
            if raw >= count
            {
                return Err(ReferenceGenerationError::OutOfOrderLogicalKernel {
                    expected,
                    found: kernel.id(),
                });
            }
            if seen[raw]
            {
                return Err(ReferenceGenerationError::DuplicateLogicalKernel {
                    kernel: kernel.id(),
                });
            }
            seen[raw] = true;

            if kernel.id() != expected
            {
                return Err(ReferenceGenerationError::OutOfOrderLogicalKernel {
                    expected,
                    found: kernel.id(),
                });
            }

            artifacts.push(self.generate_kernel(kernel)?);
        }

        // Unreachable through a `LoweredPlan` produced by `KernelLowerer::lower`:
        // by the time the loop above completes without error, every position
        // has matched its expected id, so `seen` already holds `true`
        // everywhere (pigeonhole — see `ReferenceGenerationError::MissingLogicalKernel`).
        if let Some(missing_position) = seen.iter().position(|&is_present| !is_present)
        {
            let expected = LogicalKernelId::new(
                u32::try_from(missing_position)
                    .map_err(|_| ReferenceGenerationError::TooManyKernels { count })?,
            );
            return Err(ReferenceGenerationError::MissingLogicalKernel { expected });
        }

        Ok(ReferenceKernelSet { artifacts })
    }

    /// Generates the Reference artefact for one [`LogicalKernel`] in isolation.
    pub fn generate_kernel(
        self,
        kernel: &LogicalKernel,
    ) -> Result<ReferenceKernelArtifact, ReferenceGenerationError> {
        let id = kernel.id();
        let (opcode, attribute_source) = opcode_for_family(kernel.family(), id)?;

        let dtype = kernel.result().dtype;
        if dtype != DType::F32
        {
            return Err(ReferenceGenerationError::UnsupportedDType { kernel: id, dtype });
        }
        for operand in kernel.operands()
        {
            if operand.dtype != dtype
            {
                return Err(ReferenceGenerationError::MixedDTypes { kernel: id });
            }
        }

        let expected_operands = opcode.operand_count() as usize;
        if kernel.operands().len() != expected_operands
        {
            return Err(ReferenceGenerationError::UnexpectedOperandCount {
                kernel: id,
                expected: expected_operands,
                actual: kernel.operands().len(),
            });
        }

        let mut operands = Vec::with_capacity(kernel.operands().len());
        for operand in kernel.operands()
        {
            operands.push(ReferenceTensorLayout::from_shape(&operand.shape, id)?);
        }
        let result = ReferenceTensorLayout::from_shape(&kernel.result().shape, id)?;

        let attributes = match attribute_source
        {
            AttributeSource::None => ReferenceAttributes::None,
            AttributeSource::Scale {
                factor_dtype,
                factor_bits,
            } =>
            {
                if factor_dtype != dtype
                {
                    return Err(ReferenceGenerationError::ScaleFactorTypeMismatch { kernel: id });
                }
                let factor_bits = u32::try_from(factor_bits).map_err(|_| {
                    ReferenceGenerationError::ScaleFactorBitsOverflow { kernel: id }
                })?;
                ReferenceAttributes::Scale { factor_bits }
            },
            AttributeSource::Permute { permutation } =>
            {
                let mut axes = Vec::with_capacity(permutation.len());
                for axis in permutation
                {
                    axes.push(
                        u16::try_from(axis).map_err(|_| {
                            ReferenceGenerationError::AxisIndexOverflow { kernel: id }
                        })?,
                    );
                }
                ReferenceAttributes::Permute { axes }
            },
        };

        Ok(ReferenceKernelArtifact::from_validated_parts(
            id,
            REFERENCE_FORMAT_VERSION,
            opcode,
            dtype,
            operands,
            result,
            attributes,
        ))
    }
}

/// Where a kernel's attribute payload comes from, extracted from
/// `KernelFamily` without naming `scirust_tensor_ir::Scalar` in this crate:
/// the factor's `dtype()`/`bits()` are read immediately at the match site.
enum AttributeSource {
    None,
    Scale {
        factor_dtype: DType,
        factor_bits: u64,
    },
    Permute {
        permutation: Vec<usize>,
    },
}

/// Maps a lowered kernel family to its Reference opcode and attribute source.
///
/// `MatMul` never reaches this function: `scirust_tensor_compile`'s lowering
/// phase rejects it before a `LoweredPlan` can exist. Any kernel family this
/// match does not recognise — including one a future, currently
/// non-exhaustive addition to `KernelFamily`, `UnaryKernel` or `BinaryKernel`
/// might introduce — is rejected explicitly; there is no wildcard fallback
/// that treats an unrecognised family as any existing opcode.
fn opcode_for_family(
    family: &KernelFamily,
    kernel: LogicalKernelId,
) -> Result<(ReferenceOpcode, AttributeSource), ReferenceGenerationError> {
    match family
    {
        KernelFamily::ElementwiseUnary(unary) => match unary
        {
            UnaryKernel::Relu => Ok((ReferenceOpcode::Relu, AttributeSource::None)),
            UnaryKernel::Exp => Ok((ReferenceOpcode::Exp, AttributeSource::None)),
            UnaryKernel::Log => Ok((ReferenceOpcode::Log, AttributeSource::None)),
            UnaryKernel::Scale { factor } => Ok((
                ReferenceOpcode::Scale,
                AttributeSource::Scale {
                    factor_dtype: factor.dtype(),
                    factor_bits: factor.bits(),
                },
            )),
            _ => Err(ReferenceGenerationError::UnsupportedKernelFamily { kernel }),
        },
        KernelFamily::ElementwiseBinary(binary) => match binary
        {
            BinaryKernel::Add => Ok((ReferenceOpcode::Add, AttributeSource::None)),
            BinaryKernel::Sub => Ok((ReferenceOpcode::Sub, AttributeSource::None)),
            BinaryKernel::Mul => Ok((ReferenceOpcode::Mul, AttributeSource::None)),
            BinaryKernel::Div => Ok((ReferenceOpcode::Div, AttributeSource::None)),
            _ => Err(ReferenceGenerationError::UnsupportedKernelFamily { kernel }),
        },
        KernelFamily::ShapeCopy => Ok((ReferenceOpcode::ShapeCopy, AttributeSource::None)),
        KernelFamily::Permute { permutation } => Ok((
            ReferenceOpcode::Permute,
            AttributeSource::Permute {
                permutation: permutation.clone(),
            },
        )),
        _ => Err(ReferenceGenerationError::UnsupportedKernelFamily { kernel }),
    }
}

/// Ordered, deterministic set of Reference artefacts produced from one
/// [`LoweredPlan`].
///
/// Preserves the plan's kernel order and its [`LogicalKernelId`] values: an
/// artefact's position in [`Self::artifacts`] equals its
/// [`ReferenceKernelArtifact::kernel_id`], so lookup by id never needs a hash
/// map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceKernelSet {
    artifacts: Vec<ReferenceKernelArtifact>,
}

impl ReferenceKernelSet {
    /// Artefacts in canonical [`LogicalKernelId`] order.
    pub fn artifacts(&self) -> &[ReferenceKernelArtifact] {
        &self.artifacts
    }

    /// The artefact generated for `id`, if the set contains one.
    pub fn artifact(&self, id: LogicalKernelId) -> Option<&ReferenceKernelArtifact> {
        self.artifacts.get(id.get() as usize)
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }

    /// Converts every artefact to a [`KernelModule`], preserving order.
    pub fn to_kernel_modules(&self) -> Result<Vec<KernelModule>, ReferenceGenerationError> {
        self.artifacts
            .iter()
            .map(ReferenceKernelArtifact::to_kernel_module)
            .collect()
    }
}
