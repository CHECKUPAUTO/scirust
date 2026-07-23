use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum CausalError {
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    NotSquare {
        rows: usize,
        cols: usize,
    },
    ZeroDimension,
    ZeroSamples,
    NonStrictLowerTriangular {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteWeight {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteInput {
        index: usize,
        value: f64,
    },
    NonFiniteComputation {
        operation: &'static str,
        index: usize,
        value: f64,
    },
    AllocationOverflow,
    InvalidConfiguration {
        name: &'static str,
        value: f64,
    },
    InvalidPermutation {
        detail: &'static str,
    },
    CyclicGraph,
    OptimizerFailure {
        reason: &'static str,
    },
    NonConvergence {
        iterations: usize,
        gradient_norm: f64,
    },
}

impl fmt::Display for CausalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::DimensionMismatch { expected, got } =>
            {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            },
            Self::NotSquare { rows, cols } =>
            {
                write!(f, "matrix must be square, got {rows}x{cols}")
            },
            Self::ZeroDimension =>
            {
                write!(f, "dimension must be at least 1")
            },
            Self::ZeroSamples =>
            {
                write!(f, "sample count must be at least 1")
            },
            Self::NonStrictLowerTriangular { row, col, value } =>
            {
                write!(
                    f,
                    "matrix is not strictly lower triangular: \
                     entry ({row}, {col}) is {value:.17e}"
                )
            },
            Self::NonFiniteWeight { row, col, value } =>
            {
                write!(f, "non-finite coefficient at ({row}, {col}): {value:.17e}")
            },
            Self::NonFiniteInput { index, value } =>
            {
                write!(f, "non-finite input at index {index}: {value:.17e}")
            },
            Self::NonFiniteComputation {
                operation,
                index,
                value,
            } =>
            {
                write!(
                    f,
                    "non-finite computation in {operation} at index {index}: {value:.17e}"
                )
            },
            Self::AllocationOverflow =>
            {
                write!(f, "allocation size overflow")
            },
            Self::InvalidConfiguration { name, value } =>
            {
                write!(f, "invalid configuration {name}: {value:.17e}")
            },
            Self::InvalidPermutation { detail } =>
            {
                write!(f, "invalid permutation: {detail}")
            },
            Self::CyclicGraph =>
            {
                write!(f, "extracted graph contains a cycle")
            },
            Self::OptimizerFailure { reason } =>
            {
                write!(f, "optimizer failure: {reason}")
            },
            Self::NonConvergence {
                iterations,
                gradient_norm,
            } =>
            {
                write!(
                    f,
                    "optimizer did not converge after {iterations} \
                     iterations, final gradient norm {gradient_norm:.17e}"
                )
            },
        }
    }
}

impl Error for CausalError {}
