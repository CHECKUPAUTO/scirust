use alloc::{string::String, vec::Vec};

use scirust_compute::{DType, Shape};

use crate::ConstantId;

/// Bit-exact scalar attribute.
///
/// Floating-point values are stored as raw bits so graph equality and hashing
/// never depend on floating-point comparison semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scalar {
    dtype: DType,
    bits: u64,
}

impl Scalar {
    pub const fn f32(value: f32) -> Self {
        Self {
            dtype: DType::F32,
            bits: value.to_bits() as u64,
        }
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn bits(self) -> u64 {
        self.bits
    }

    pub const fn as_f32(self) -> Option<f32> {
        if matches!(self.dtype, DType::F32)
        {
            Some(f32::from_bits(self.bits as u32))
        }
        else
        {
            None
        }
    }
}

/// One canonical tensor operation.
///
/// Constants are referenced externally through [`ConstantId`]; tensor payloads
/// are deliberately not embedded in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Operation {
    Input { name: String },
    Constant { id: ConstantId },

    Add,
    Sub,
    Mul,
    Div,
    Scale { factor: Scalar },

    Relu,
    Exp,
    Log,

    /// `cotangent * 1(primal > 0)`. This primitive makes reverse- and
    /// forward-mode differentiation of ReLU executable without embedding a
    /// comparison sub-language in the graph IR.
    ReluGrad,
    /// Produce an all-zero tensor with the input's type.
    ZerosLike,
    /// Produce an all-one tensor with the input's type.
    OnesLike,

    MatMul,

    Reshape { shape: Shape },
    Transpose { permutation: Vec<usize> },

    /// Identity in the primal program and a barrier to differentiation.
    StopGradient,
    /// Identity marker declaring that intermediates may be recomputed instead
    /// of retained by a future execution planner.
    Checkpoint,
}

impl Operation {
    pub const fn expected_arity(&self) -> usize {
        match self
        {
            Self::Input { .. } | Self::Constant { .. } => 0,

            Self::Relu
            | Self::Exp
            | Self::Log
            | Self::Scale { .. }
            | Self::ZerosLike
            | Self::OnesLike
            | Self::Reshape { .. }
            | Self::Transpose { .. }
            | Self::StopGradient
            | Self::Checkpoint => 1,

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::MatMul
            | Self::ReluGrad => 2,
        }
    }
}
