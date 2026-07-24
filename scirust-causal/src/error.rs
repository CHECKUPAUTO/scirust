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
    InvalidContract {
        detail: &'static str,
    },
    UnknownVariableIndex {
        index: usize,
    },
    SameVariable {
        variable: usize,
    },
    ConditioningContainsEndpoint {
        variable: usize,
    },
    DuplicateConditioningVariable {
        variable: usize,
    },
    UnsupportedVariableKind {
        variable: usize,
    },
    InsufficientSamples {
        required: usize,
        actual: usize,
    },
    NonFiniteSample {
        row: usize,
        variable: usize,
        value: f64,
    },
    ZeroVariance {
        variable: usize,
    },
    RankDeficientConditioningSet {
        rank: usize,
        columns: usize,
    },
    ScatterFailure(scirust_multivariate::RobustGeometryError),
    SolverFailure {
        detail: String,
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
                write!(
                    f,
                    "graph extraction produced a cycle (a super-threshold entry \
                     on the diagonal or a thresholded pattern with a directed cycle)"
                )
            },
            Self::InvalidContract { detail } =>
            {
                write!(f, "invalid causal contract: {detail}")
            },
            Self::UnknownVariableIndex { index } =>
            {
                write!(f, "reference to unknown variable index {index}")
            },
            Self::SameVariable { variable } =>
            {
                write!(
                    f,
                    "variable {variable} cannot be tested against itself (x == y)"
                )
            },
            Self::ConditioningContainsEndpoint { variable } =>
            {
                write!(
                    f,
                    "conditioning set contains variable {variable}, which is also an endpoint (x or y)"
                )
            },
            Self::DuplicateConditioningVariable { variable } =>
            {
                write!(
                    f,
                    "variable {variable} appears more than once in the conditioning set"
                )
            },
            Self::UnsupportedVariableKind { variable } =>
            {
                write!(
                    f,
                    "variable {variable} has a kind this test does not support (expected Continuous)"
                )
            },
            Self::InsufficientSamples { required, actual } =>
            {
                write!(
                    f,
                    "at least {required} eligible samples are required, got {actual}"
                )
            },
            Self::NonFiniteSample {
                row,
                variable,
                value,
            } =>
            {
                write!(
                    f,
                    "non-finite value for variable {variable} at row {row}: {value:.17e}"
                )
            },
            Self::ZeroVariance { variable } =>
            {
                write!(
                    f,
                    "variable {variable} has zero variance (no association is defined)"
                )
            },
            Self::RankDeficientConditioningSet { rank, columns } =>
            {
                write!(
                    f,
                    "conditioning design has rank {rank} but {columns} columns \
                     (near-singular under the configured tolerance)"
                )
            },
            Self::ScatterFailure(source) =>
            {
                write!(f, "robust scatter estimation failed: {source}")
            },
            Self::SolverFailure { detail } =>
            {
                write!(f, "linear solver failed: {detail}")
            },
        }
    }
}

impl Error for CausalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self
        {
            Self::ScatterFailure(source) => Some(source),
            _ => None,
        }
    }
}
