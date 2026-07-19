//! Structured errors for public quantum APIs.

use core::fmt;

/// Result type used by SciRust Quantum.
pub type QuantumResult<T> = Result<T, QuantumError>;

/// Failures reported by circuit construction, simulation, and differentiation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QuantumError {
    /// Qubit index is outside `0..num_qubits`.
    InvalidQubitIndex { qubit: usize, num_qubits: usize },
    /// An operation or observable names the same qubit twice.
    DuplicateQubit { qubit: usize },
    /// A controlled operation uses the same control and target.
    InvalidControlTarget { control: usize, target: usize },
    /// `2^num_qubits` or its byte size cannot be represented by `usize`.
    StateDimensionOverflow { num_qubits: usize },
    /// The state exceeds the dense backend's explicit allocation limit.
    AllocationTooLarge {
        requested_bytes: usize,
        limit_bytes: usize,
    },
    /// A gate is not available on the selected backend.
    UnsupportedGate { gate: &'static str },
    /// The requested parameter does not have a valid differentiation rule.
    UnsupportedGradientRule { parameter: u32 },
    /// A symbolic parameter has no supplied value.
    UnboundParameter { parameter: u32 },
    /// A parameter, amplitude, tolerance, or epsilon is not finite.
    NonFiniteParameter { what: &'static str },
    /// State norm differs from one by more than the specified tolerance.
    NonNormalizedState { norm_sqr: f32, tolerance: f32 },
    /// Observable structure or Hermiticity validation failed.
    InvalidObservable { reason: &'static str },
    /// The execution request exceeds a backend's advertised capability.
    BackendCapabilityMismatch { capability: &'static str },
    /// A numerical invariant failed during otherwise valid execution.
    NumericalFailure { operation: &'static str },
    /// An amplitude buffer has the wrong dimension.
    InvalidStateDimension { expected: usize, actual: usize },
    /// A parameter binding contains an ID not present in the circuit.
    UnknownParameter { parameter: u32 },
    /// Feature/parameter tensors do not map one-to-one onto circuit symbols.
    InvalidParameterMapping { reason: &'static str },
    /// A batched quantum layer requires at least `minimum` samples.
    InvalidBatchSize { minimum: usize, actual: usize },
    /// A quantum layer received an unsupported number of observables.
    InvalidObservableCount {
        minimum: usize,
        maximum: Option<usize>,
        actual: usize,
    },
    /// A real-valued autograd tensor does not satisfy the quantum-layer contract.
    /// `None` means that dimension is unconstrained by this particular check.
    InvalidTensorShape {
        tensor: &'static str,
        expected_rows: Option<usize>,
        expected_cols: Option<usize>,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// Quantum operands belong to different reverse-mode tapes.
    MismatchedAutodiffTapes,
}

impl fmt::Display for QuantumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidQubitIndex { qubit, num_qubits } =>
            {
                write!(formatter, "qubit {qubit} is outside 0..{num_qubits}")
            },
            Self::DuplicateQubit { qubit } => write!(formatter, "qubit {qubit} is duplicated"),
            Self::InvalidControlTarget { control, target } => write!(
                formatter,
                "control {control} and target {target} must be different"
            ),
            Self::StateDimensionOverflow { num_qubits } => write!(
                formatter,
                "the dense dimension 2^{num_qubits} cannot be represented"
            ),
            Self::AllocationTooLarge {
                requested_bytes,
                limit_bytes,
            } => write!(
                formatter,
                "dense state needs {requested_bytes} bytes, exceeding the {limit_bytes}-byte limit"
            ),
            Self::UnsupportedGate { gate } => write!(formatter, "unsupported gate {gate}"),
            Self::UnsupportedGradientRule { parameter } => write!(
                formatter,
                "parameter {parameter} has no supported parameter-shift rule"
            ),
            Self::UnboundParameter { parameter } =>
            {
                write!(formatter, "parameter {parameter} is unbound")
            },
            Self::NonFiniteParameter { what } => write!(formatter, "{what} must be finite"),
            Self::NonNormalizedState {
                norm_sqr,
                tolerance,
            } => write!(
                formatter,
                "state squared norm {norm_sqr} differs from one by more than {tolerance}"
            ),
            Self::InvalidObservable { reason } => write!(formatter, "invalid observable: {reason}"),
            Self::BackendCapabilityMismatch { capability } =>
            {
                write!(formatter, "backend does not support {capability}")
            },
            Self::NumericalFailure { operation } =>
            {
                write!(formatter, "numerical failure during {operation}")
            },
            Self::InvalidStateDimension { expected, actual } => write!(
                formatter,
                "state dimension mismatch: expected {expected} amplitudes, got {actual}"
            ),
            Self::UnknownParameter { parameter } =>
            {
                write!(
                    formatter,
                    "parameter {parameter} is not used by the circuit"
                )
            },
            Self::InvalidParameterMapping { reason } =>
            {
                write!(
                    formatter,
                    "invalid quantum-layer parameter mapping: {reason}"
                )
            },
            Self::InvalidBatchSize { minimum, actual } => write!(
                formatter,
                "invalid quantum batch size: expected at least {minimum} rows, got {actual}"
            ),
            Self::InvalidObservableCount {
                minimum,
                maximum,
                actual,
            } => match maximum
            {
                Some(maximum) if minimum == maximum => write!(
                    formatter,
                    "invalid observable count: expected exactly {minimum}, got {actual}"
                ),
                Some(maximum) => write!(
                    formatter,
                    "invalid observable count: expected {minimum}..={maximum}, got {actual}"
                ),
                None => write!(
                    formatter,
                    "invalid observable count: expected at least {minimum}, got {actual}"
                ),
            },
            Self::InvalidTensorShape {
                tensor,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } =>
            {
                let expected_rows = expected_rows
                    .map(|rows| rows.to_string())
                    .unwrap_or_else(|| "any".to_string());
                let expected_cols = expected_cols
                    .map(|cols| cols.to_string())
                    .unwrap_or_else(|| "any".to_string());
                write!(
                    formatter,
                    "invalid {tensor} shape: expected {expected_rows} x {expected_cols}, got {actual_rows} x {actual_cols}"
                )
            },
            Self::MismatchedAutodiffTapes => write!(
                formatter,
                "quantum-layer tensors belong to different autodiff tapes"
            ),
        }
    }
}

impl std::error::Error for QuantumError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_shape_display_includes_expected_and_actual_dimensions() {
        let error = QuantumError::InvalidTensorShape {
            tensor: "classical_features",
            expected_rows: None,
            expected_cols: Some(2),
            actual_rows: 3,
            actual_cols: 4,
        };
        assert_eq!(
            error.to_string(),
            "invalid classical_features shape: expected any x 2, got 3 x 4"
        );
    }
}
