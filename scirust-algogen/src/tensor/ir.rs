//! Intermediate representation for generated tensor programs.

use serde::{Deserialize, Serialize};

/// One instruction in a linear, register-based tensor program.
///
/// Every source register must refer to an instruction positioned strictly
/// before the current instruction. This invariant is checked by the verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorInstruction {
    /// Borrow one of the tensors supplied to the program.
    Input { input: usize },

    /// Element-wise addition of two tensors with identical shapes.
    Add { lhs: usize, rhs: usize },

    /// Rank-two matrix multiplication.
    MatMul { lhs: usize, rhs: usize },

    /// Transpose a rank-two tensor.
    Transpose2d { src: usize },

    /// Element-wise rectified linear activation.
    Relu { src: usize },

    /// Element-wise multiplication by a scalar.
    Scale { src: usize, factor: f32 },
}

impl TensorInstruction {
    /// Visit every register read by this instruction.
    pub fn for_each_source(&self, mut visitor: impl FnMut(usize)) {
        match *self
        {
            Self::Input { .. } =>
            {},
            Self::Add { lhs, rhs } | Self::MatMul { lhs, rhs } =>
            {
                visitor(lhs);
                visitor(rhs);
            },
            Self::Transpose2d { src } | Self::Relu { src } | Self::Scale { src, .. } =>
            {
                visitor(src);
            },
        }
    }
}

/// A generated tensor algorithm.
///
/// `output` identifies the register that constitutes the observable result.
/// The output therefore does not have to be the final instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorProgram {
    pub instructions: Vec<TensorInstruction>,
    pub output: usize,
}

impl TensorProgram {
    pub fn new(instructions: Vec<TensorInstruction>, output: usize) -> Self {
        Self {
            instructions,
            output,
        }
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}
