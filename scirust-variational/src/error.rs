use std::fmt;

#[derive(Debug, Clone)]
pub enum VariationalError {
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    InvalidInterval {
        start: f32,
        end: f32,
    },
    InvalidBoundaryCondition {
        details: String,
    },
    SingularVelocityHessian {
        condition_number: f32,
        tolerance: f32,
    },
    IllConditionedSystem {
        condition_number: f32,
        tolerance: f32,
    },
    NonFiniteValue {
        component: &'static str,
        value: f32,
    },
    AutodiffFailure {
        details: String,
    },
    SymbolicFailure {
        details: String,
    },
    LinearSolveFailure {
        details: String,
    },
    NonlinearSolveFailure {
        details: String,
    },
    ConstraintViolation {
        constraint: String,
        residual: f32,
    },
    InfeasibleControlProblem {
        details: String,
    },
    TrainingFailure {
        details: String,
    },
    UnsupportedOperation {
        details: String,
    },
}

impl fmt::Display for VariationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::DimensionMismatch {
                expected,
                got,
                context,
            } =>
            {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, got {got}"
                )
            },
            Self::InvalidInterval { start, end } =>
            {
                write!(f, "invalid interval [{start}, {end}]: start must be < end")
            },
            Self::InvalidBoundaryCondition { details } =>
            {
                write!(f, "invalid boundary condition: {details}")
            },
            Self::SingularVelocityHessian {
                condition_number,
                tolerance,
            } =>
            {
                write!(
                    f,
                    "singular velocity Hessian: condition number {condition_number:.2e} exceeds tolerance {tolerance:.2e}"
                )
            },
            Self::IllConditionedSystem {
                condition_number,
                tolerance,
            } =>
            {
                write!(
                    f,
                    "ill-conditioned system: condition number {condition_number:.2e} exceeds tolerance {tolerance:.2e}"
                )
            },
            Self::NonFiniteValue { component, value } =>
            {
                write!(f, "non-finite value in {component}: {value}")
            },
            Self::AutodiffFailure { details } =>
            {
                write!(f, "autodiff failure: {details}")
            },
            Self::SymbolicFailure { details } =>
            {
                write!(f, "symbolic failure: {details}")
            },
            Self::LinearSolveFailure { details } =>
            {
                write!(f, "linear solve failure: {details}")
            },
            Self::NonlinearSolveFailure { details } =>
            {
                write!(f, "nonlinear solve failure: {details}")
            },
            Self::ConstraintViolation {
                constraint,
                residual,
            } =>
            {
                write!(
                    f,
                    "constraint violation '{constraint}': residual {residual:.2e}"
                )
            },
            Self::InfeasibleControlProblem { details } =>
            {
                write!(f, "infeasible control problem: {details}")
            },
            Self::TrainingFailure { details } =>
            {
                write!(f, "training failure: {details}")
            },
            Self::UnsupportedOperation { details } =>
            {
                write!(f, "unsupported operation: {details}")
            },
        }
    }
}

impl std::error::Error for VariationalError {}

pub type Result<T> = std::result::Result<T, VariationalError>;
