//! Exact differentiable computational-basis probabilities from Pauli-Z moments.
//!
//! For an `n`-qubit state and basis index `b`,
//!
//! `p(b) = 2^-n * (1 + Σ_{S≠∅} (-1)^{popcount(S & b)} <Z_S>)`.
//!
//! SciRust already differentiates every measured Pauli-product expectation via
//! the dense adjoint path. This module therefore reconstructs probabilities by a
//! fixed Walsh projection over the complete non-empty Z-moment basis instead of
//! introducing a second simulator or quantum backward rule.

use super::VariationalCircuitBuilder;
use crate::autodiff::reverse::{Tensor, Var};
use crate::quantum::{Observable, Pauli, PauliTerm, QuantumError, QuantumResult};

/// Explicit limit for the dense Walsh projection used by this exact readout.
///
/// The coefficient matrix has `(2^n - 1) * 2^n` `f32` entries and the current
/// dense adjoint path also keeps one adjoint state per ordered observable. A hard
/// limit prevents accidental super-exponential memory use at framework level.
pub const MAX_EXACT_PROBABILITY_QUBITS: usize = 10;

/// Fixed differentiable Walsh projection from complete Pauli-Z moments to
/// computational-basis probabilities in little-endian state-index order.
#[derive(Debug, Clone)]
pub struct ComputationalBasisReadout {
    num_qubits: usize,
    moment_observables: Vec<Observable>,
    coefficients: Tensor,
    bias: Tensor,
}

impl ComputationalBasisReadout {
    /// Builds the readout for `num_qubits` from an exact ordered Z-moment basis.
    ///
    /// `observables` must contain exactly one entry for each non-empty Z product
    /// in ascending binary-mask order: mask bit `q` denotes `Z` on qubit `q`.
    pub fn from_observables(num_qubits: usize, observables: &[Observable]) -> QuantumResult<Self> {
        let dimension = checked_probability_dimension(num_qubits)?;
        let expected = complete_z_moment_basis(num_qubits, dimension)?;
        if observables != expected.as_slice()
        {
            return Err(QuantumError::InvalidObservable {
                reason: "computational-basis readout requires the complete ordered non-empty Z-moment basis",
            });
        }

        let scale = 1.0 / dimension as f32;
        let mut coefficients = Vec::with_capacity((dimension - 1) * dimension);
        for mask in 1..dimension
        {
            for basis in 0..dimension
            {
                let parity = (mask & basis).count_ones() & 1;
                coefficients.push(if parity == 0 { scale } else { -scale });
            }
        }

        Ok(Self {
            num_qubits,
            moment_observables: expected,
            coefficients: Tensor::from_vec(coefficients, dimension - 1, dimension),
            bias: Tensor::from_vec(vec![scale; dimension], 1, dimension),
        })
    }

    /// Builds a readout directly from a variational circuit whose observables are
    /// the complete ordered Z-moment basis produced by
    /// [`VariationalCircuitBuilder::measure_computational_basis_moments`].
    pub fn from_circuit(circuit: &super::VariationalCircuit) -> QuantumResult<Self> {
        Self::from_observables(circuit.circuit().num_qubits(), circuit.observables())
    }

    /// Projects `[batch, 2^n-1]` exact Z moments to `[batch, 2^n]` raw exact-model
    /// probabilities in little-endian state-index order.
    ///
    /// No clamp or renormalization is applied: tiny floating residuals remain
    /// visible and the transform stays exactly linear for reverse-mode gradients.
    pub fn apply<'t>(&self, moments: Var<'t>) -> QuantumResult<Var<'t>> {
        let (batch, columns) = moments.shape();
        if batch == 0
        {
            return Err(QuantumError::InvalidBatchSize {
                minimum: 1,
                actual: 0,
            });
        }
        if columns != self.moment_count()
        {
            return Err(QuantumError::InvalidTensorShape {
                tensor: "computational_probability_moments",
                expected_rows: None,
                expected_cols: Some(self.moment_count()),
                actual_rows: batch,
                actual_cols: columns,
            });
        }

        let coefficients = moments.tape.input(self.coefficients.clone());
        let bias = moments.tape.input(self.bias.clone());
        Ok(moments.matmul(coefficients).add_bias(bias))
    }

    /// Number of qubits represented by this probability readout.
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of required non-empty Pauli-Z moments (`2^n - 1`).
    #[must_use]
    pub fn moment_count(&self) -> usize {
        self.moment_observables.len()
    }

    /// Number of computational-basis probabilities (`2^n`).
    #[must_use]
    pub fn probability_count(&self) -> usize {
        self.coefficients.cols
    }

    /// Required Z-moment observables in ascending binary-mask order.
    #[must_use]
    pub fn moment_observables(&self) -> &[Observable] {
        &self.moment_observables
    }

    /// Fixed Walsh coefficient matrix with shape `[2^n-1, 2^n]`.
    #[must_use]
    pub const fn coefficients(&self) -> &Tensor {
        &self.coefficients
    }
}

impl VariationalCircuitBuilder {
    /// Appends the complete non-empty Pauli-Z moment basis in ascending binary
    /// mask order for an exact computational-basis probability readout.
    ///
    /// This helper requires an empty measurement list so the resulting circuit's
    /// output columns have an unambiguous, canonical probability-moment layout.
    pub fn measure_computational_basis_moments(&mut self) -> QuantumResult<&mut Self> {
        if !self.observables.is_empty()
        {
            return Err(QuantumError::InvalidObservable {
                reason: "computational-basis moments must be the circuit's first measurement basis",
            });
        }
        let dimension = checked_probability_dimension(self.num_qubits())?;
        let basis = complete_z_moment_basis(self.num_qubits(), dimension)?;
        for observable in basis
        {
            self.measure(observable)?;
        }
        Ok(self)
    }
}

fn checked_probability_dimension(num_qubits: usize) -> QuantumResult<usize> {
    if num_qubits == 0 || num_qubits > MAX_EXACT_PROBABILITY_QUBITS
    {
        return Err(QuantumError::StateDimensionOverflow { num_qubits });
    }
    1usize
        .checked_shl(num_qubits as u32)
        .ok_or(QuantumError::StateDimensionOverflow { num_qubits })
}

fn complete_z_moment_basis(num_qubits: usize, dimension: usize) -> QuantumResult<Vec<Observable>> {
    let mut observables = Vec::with_capacity(dimension - 1);
    for mask in 1..dimension
    {
        let mut terms = Vec::with_capacity(mask.count_ones() as usize);
        for qubit in 0..num_qubits
        {
            if mask & (1usize << qubit) != 0
            {
                terms.push(PauliTerm::new(qubit, Pauli::Z));
            }
        }
        observables.push(Observable::new(terms)?);
    }
    Ok(observables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::quantum::Operation;
    use crate::vqnet::{RotationAxis, VariationalCircuitBuilder};

    const TOLERANCE: f32 = 5.0e-5;

    #[test]
    fn canonical_moment_basis_is_binary_mask_order() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder.measure_computational_basis_moments().unwrap();
        let circuit = builder.build().unwrap();

        assert_eq!(circuit.observables().len(), 3);
        assert_eq!(circuit.observables()[0], Observable::z(0));
        assert_eq!(circuit.observables()[1], Observable::z(1));
        assert_eq!(
            circuit.observables()[2],
            Observable::new(vec![
                PauliTerm::new(0, Pauli::Z),
                PauliTerm::new(1, Pauli::Z),
            ])
            .unwrap()
        );
    }

    #[test]
    fn walsh_projection_reconstructs_two_qubit_probabilities_exactly() {
        let observables = complete_z_moment_basis(2, 4).unwrap();
        let readout = ComputationalBasisReadout::from_observables(2, &observables).unwrap();
        let tape = Tape::new();
        // |00> has <Z0>=<Z1>=<Z0Z1>=1.
        let moments = tape.input(Tensor::from_vec(vec![1.0, 1.0, 1.0], 1, 3));
        let probabilities = readout.apply(moments).unwrap();
        let values = tape.value(probabilities.idx()).data;
        assert_eq!(values, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn probability_gradient_is_the_fixed_walsh_transform() {
        let observables = complete_z_moment_basis(1, 2).unwrap();
        let readout = ComputationalBasisReadout::from_observables(1, &observables).unwrap();
        let tape = Tape::new();
        let moments = tape.input(Tensor::from_vec(vec![0.4], 1, 1));
        let probabilities = readout.apply(moments).unwrap();
        let weights = tape.input(Tensor::from_vec(vec![2.0, -3.0], 2, 1));
        let objective = probabilities.matmul(weights);
        objective.backward();

        let gradient = tape.grad(moments.idx()).data[0];
        // p0=(1+z)/2, p1=(1-z)/2 => d(2p0-3p1)/dz = 2.5.
        assert!((gradient - 2.5).abs() <= TOLERANCE);
    }

    #[test]
    fn quantum_adjoint_flows_through_probability_readout() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder.measure_computational_basis_moments().unwrap();
        let circuit = builder.build().unwrap();
        let readout = ComputationalBasisReadout::from_circuit(&circuit).unwrap();

        let tape = Tape::new();
        let x = 0.4f32;
        let features = tape.input(Tensor::from_vec(vec![x], 1, 1));
        let parameters = tape.input(Tensor::from_vec(Vec::new(), 1, 0));
        let moments = circuit.forward_batch(features, parameters).unwrap();
        let probabilities = readout.apply(moments).unwrap();
        let values = tape.value(probabilities.idx()).data;

        let expected_zero = 0.5 * (1.0 + x.cos());
        let expected_one = 0.5 * (1.0 - x.cos());
        assert!((values[0] - expected_zero).abs() <= TOLERANCE);
        assert!((values[1] - expected_one).abs() <= TOLERANCE);

        let select_one = tape.input(Tensor::from_vec(vec![0.0, 1.0], 2, 1));
        probabilities.matmul(select_one).backward();
        let gradient = tape.grad(features.idx()).data[0];
        assert!((gradient - 0.5 * x.sin()).abs() <= TOLERANCE);
    }

    #[test]
    fn measurement_helper_rejects_prior_measurements_without_mutation() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder.measure_z(&[0]).unwrap();
        let before = builder.observables.clone();
        assert_eq!(
            builder.measure_computational_basis_moments().unwrap_err(),
            QuantumError::InvalidObservable {
                reason: "computational-basis moments must be the circuit's first measurement basis",
            }
        );
        assert_eq!(builder.observables, before);
    }

    #[test]
    fn probability_readout_has_explicit_qubit_ceiling() {
        assert_eq!(
            ComputationalBasisReadout::from_observables(MAX_EXACT_PROBABILITY_QUBITS + 1, &[],)
                .unwrap_err(),
            QuantumError::StateDimensionOverflow {
                num_qubits: MAX_EXACT_PROBABILITY_QUBITS + 1,
            }
        );
    }

    #[test]
    fn probability_circuit_uses_only_measurements_not_extra_gates() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder.measure_computational_basis_moments().unwrap();
        let circuit = builder.build().unwrap();
        assert!(circuit.circuit().operations().is_empty());
        assert!(
            !circuit
                .circuit()
                .operations()
                .iter()
                .any(|operation| matches!(operation, Operation::H { .. }))
        );
    }
}
