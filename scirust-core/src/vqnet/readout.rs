//! Differentiable Hamiltonian readout over ordered quantum expectations.
//!
//! The quantum runtime already produces exact expectation values for ordered
//! Pauli-product observables. This module keeps that execution path authoritative
//! and represents a Hamiltonian as a fixed real linear combination of those
//! observables plus an optional identity offset. The linear projection itself is
//! ordinary SciRust reverse-mode `matmul` + `add_bias`, so gradients continue
//! through the existing quantum adjoint node without a second differentiation
//! implementation.

use super::VariationalCircuit;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::error::{Result as SciRustResult, SciRustError};
use crate::nn::module::Module;
use crate::quantum::{Observable, QuantumError, QuantumResult};

/// One real coefficient multiplying a Pauli-product observable.
#[derive(Debug, Clone, PartialEq)]
pub struct HamiltonianTerm {
    coefficient: f32,
    observable: Observable,
}

impl HamiltonianTerm {
    /// Creates one finite Hamiltonian term.
    pub fn new(coefficient: f32, observable: Observable) -> QuantumResult<Self> {
        if !coefficient.is_finite()
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "Hamiltonian coefficient",
            });
        }
        Ok(Self {
            coefficient,
            observable,
        })
    }

    /// Real coefficient multiplying this observable.
    #[must_use]
    pub const fn coefficient(&self) -> f32 {
        self.coefficient
    }

    /// Pauli-product observable multiplied by this coefficient.
    #[must_use]
    pub const fn observable(&self) -> &Observable {
        &self.observable
    }
}

/// A real Hamiltonian expressed in the measured Pauli-product basis.
///
/// `offset` is the coefficient of the identity operator. Repeated equivalent
/// observables are allowed; their coefficients are accumulated deterministically
/// when a [`HamiltonianReadout`] is built.
#[derive(Debug, Clone, PartialEq)]
pub struct Hamiltonian {
    terms: Vec<HamiltonianTerm>,
    offset: f32,
}

impl Hamiltonian {
    /// Creates a Hamiltonian with zero identity offset.
    pub fn new(terms: Vec<HamiltonianTerm>) -> QuantumResult<Self> {
        Self::with_offset(terms, 0.0)
    }

    /// Creates a Hamiltonian with an explicit finite identity offset.
    pub fn with_offset(terms: Vec<HamiltonianTerm>, offset: f32) -> QuantumResult<Self> {
        if !offset.is_finite()
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "Hamiltonian identity offset",
            });
        }
        Ok(Self { terms, offset })
    }

    /// Ordered Hamiltonian terms.
    #[must_use]
    pub fn terms(&self) -> &[HamiltonianTerm] {
        &self.terms
    }

    /// Identity-operator coefficient.
    #[must_use]
    pub const fn offset(&self) -> f32 {
        self.offset
    }
}

/// Fixed differentiable projection from measured expectations to Hamiltonian
/// expectation values.
///
/// Coefficients are stored as an `observable_count × hamiltonian_count` matrix,
/// so an input `[batch, observable_count]` expectation tensor projects directly
/// to `[batch, hamiltonian_count]` through the existing reverse-mode matrix
/// multiplication node. The readout is also a stateless [`Module`]: its fixed
/// coefficients are problem-definition data, not trainable parameters.
#[derive(Debug, Clone)]
pub struct HamiltonianReadout {
    observables: Vec<Observable>,
    coefficients: Tensor,
    offsets: Tensor,
}

impl HamiltonianReadout {
    /// Builds a fixed readout against an exact ordered measured-observable basis.
    ///
    /// Every non-identity Hamiltonian term must be semantically present in
    /// `observables`. Pauli factors are matched independent of factor ordering,
    /// because distinct-qubit Pauli factors commute inside a tensor product.
    pub fn from_observables(
        observables: &[Observable],
        hamiltonians: &[Hamiltonian],
    ) -> QuantumResult<Self> {
        if observables.is_empty()
        {
            return Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            });
        }
        if hamiltonians.is_empty()
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "Hamiltonian readout needs at least one Hamiltonian",
            });
        }

        let observable_count = observables.len();
        let hamiltonian_count = hamiltonians.len();
        let mut coefficients = vec![0.0f32; observable_count * hamiltonian_count];
        let mut offsets = Vec::with_capacity(hamiltonian_count);

        for (hamiltonian_index, hamiltonian) in hamiltonians.iter().enumerate()
        {
            offsets.push(hamiltonian.offset());
            for term in hamiltonian.terms()
            {
                let observable_index = observables
                    .iter()
                    .position(|observable| observables_equivalent(observable, term.observable()))
                    .ok_or(QuantumError::InvalidObservable {
                        reason: "Hamiltonian term is not present in the measured observable basis",
                    })?;
                let index = observable_index * hamiltonian_count + hamiltonian_index;
                coefficients[index] += term.coefficient();
                if !coefficients[index].is_finite()
                {
                    return Err(QuantumError::NonFiniteParameter {
                        what: "accumulated Hamiltonian coefficient",
                    });
                }
            }
        }

        Ok(Self {
            observables: observables.to_vec(),
            coefficients: Tensor::from_vec(coefficients, observable_count, hamiltonian_count),
            offsets: Tensor::from_vec(offsets, 1, hamiltonian_count),
        })
    }

    /// Applies this fixed Hamiltonian projection to a batch of ordered
    /// expectation values.
    pub fn apply<'t>(&self, expectations: Var<'t>) -> QuantumResult<Var<'t>> {
        let (batch, observable_count) = expectations.shape();
        if batch == 0
        {
            return Err(QuantumError::InvalidBatchSize {
                minimum: 1,
                actual: 0,
            });
        }
        if observable_count != self.observable_count()
        {
            return Err(QuantumError::InvalidTensorShape {
                tensor: "hamiltonian_expectations",
                expected_rows: None,
                expected_cols: Some(self.observable_count()),
                actual_rows: batch,
                actual_cols: observable_count,
            });
        }

        let coefficients = expectations.tape.input(self.coefficients.clone());
        let offsets = expectations.tape.input(self.offsets.clone());
        Ok(expectations.matmul(coefficients).add_bias(offsets))
    }

    /// Ordered measured-observable basis expected by this readout.
    #[must_use]
    pub fn observables(&self) -> &[Observable] {
        &self.observables
    }

    /// Number of measured expectation columns consumed by this readout.
    #[must_use]
    pub fn observable_count(&self) -> usize {
        self.coefficients.rows
    }

    /// Number of Hamiltonian expectation columns produced by this readout.
    #[must_use]
    pub fn hamiltonian_count(&self) -> usize {
        self.coefficients.cols
    }

    /// Fixed row-major `observable_count × hamiltonian_count` coefficient matrix.
    #[must_use]
    pub const fn coefficients(&self) -> &Tensor {
        &self.coefficients
    }

    /// Fixed `1 × hamiltonian_count` identity offsets.
    #[must_use]
    pub const fn offsets(&self) -> &Tensor {
        &self.offsets
    }

    pub(super) fn validate_circuit(&self, circuit: &VariationalCircuit) -> QuantumResult<()> {
        if circuit.observables().len() != self.observables.len()
            || !circuit
                .observables()
                .iter()
                .zip(&self.observables)
                .all(|(left, right)| observables_equivalent(left, right))
        {
            return Err(QuantumError::InvalidObservable {
                reason: "Hamiltonian readout observable basis does not match the variational circuit",
            });
        }
        Ok(())
    }
}

impl Module for HamiltonianReadout {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.try_forward(tape, input)
            .expect("HamiltonianReadout::forward received an invalid expectation tensor")
    }

    fn try_forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> SciRustResult<Var<'t>> {
        if !std::ptr::eq(input.tape, tape)
        {
            return Err(SciRustError::InvalidConfig(
                "VQNet Hamiltonian readout input belongs to a different autodiff tape".to_string(),
            ));
        }

        self.apply(input).map_err(|error| {
            SciRustError::InvalidConfig(format!("VQNet Hamiltonian readout: {error}"))
        })
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    fn sync(&mut self, _tape: &Tape) {}
}

fn observables_equivalent(left: &Observable, right: &Observable) -> bool {
    left.terms().len() == right.terms().len()
        && left.terms().iter().all(|left_term| {
            right.terms().iter().any(|right_term| {
                left_term.qubit == right_term.qubit && left_term.pauli == right_term.pauli
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::nn::init::Zeros;
    use crate::nn::linear::Linear;
    use crate::nn::rng::PcgEngine;
    use crate::nn::sequential::Sequential;
    use crate::quantum::{Pauli, PauliTerm};
    use crate::vqnet::{
        EntanglementTopology, EntanglingGate, ParameterInitializer, QuantumModule, RotationAxis,
        VariationalCircuitBuilder,
    };

    const TOLERANCE: f32 = 4.0e-5;

    #[test]
    fn fixed_readout_projects_multiple_hamiltonians_and_gradients() {
        let observables = vec![Observable::z(0), Observable::z(1)];
        let first = Hamiltonian::with_offset(
            vec![
                HamiltonianTerm::new(0.5, Observable::z(0)).unwrap(),
                HamiltonianTerm::new(-0.25, Observable::z(1)).unwrap(),
            ],
            0.1,
        )
        .unwrap();
        let second = Hamiltonian::new(vec![
            HamiltonianTerm::new(-1.0, Observable::z(0)).unwrap(),
            HamiltonianTerm::new(2.0, Observable::z(0)).unwrap(),
            HamiltonianTerm::new(0.75, Observable::z(1)).unwrap(),
        ])
        .unwrap();
        let readout = HamiltonianReadout::from_observables(&observables, &[first, second]).unwrap();

        assert_eq!(readout.coefficients().shape(), (2, 2));
        assert_eq!(readout.coefficients().data, vec![0.5, 1.0, -0.25, 0.75]);
        assert_eq!(readout.offsets().data, vec![0.1, 0.0]);

        let tape = Tape::new();
        let expectations = tape.input(Tensor::from_vec(vec![0.2, 0.4, -0.6, 0.8], 2, 2));
        let output = readout.apply(expectations).unwrap();
        let values = tape.value(output.idx()).data;
        let expected = [0.1, 0.5, -0.4, 0.0];
        for (actual, expected) in values.iter().zip(expected)
        {
            assert!((*actual - expected).abs() <= TOLERANCE);
        }

        output.sum().backward();
        let gradient = tape.grad(expectations.idx()).data;
        assert_eq!(gradient, vec![1.5, 0.5, 1.5, 0.5]);
    }

    #[test]
    fn pauli_factor_order_does_not_change_hamiltonian_matching() {
        let measured = Observable::new(vec![
            PauliTerm::new(0, Pauli::X),
            PauliTerm::new(1, Pauli::Z),
        ])
        .unwrap();
        let reordered = Observable::new(vec![
            PauliTerm::new(1, Pauli::Z),
            PauliTerm::new(0, Pauli::X),
        ])
        .unwrap();
        let hamiltonian =
            Hamiltonian::new(vec![HamiltonianTerm::new(2.0, reordered).unwrap()]).unwrap();

        let readout = HamiltonianReadout::from_observables(&[measured], &[hamiltonian]).unwrap();
        assert_eq!(readout.coefficients().data, vec![2.0]);
    }

    #[test]
    fn missing_observable_is_rejected() {
        let hamiltonian =
            Hamiltonian::new(vec![HamiltonianTerm::new(1.0, Observable::x(0)).unwrap()]).unwrap();
        assert_eq!(
            HamiltonianReadout::from_observables(&[Observable::z(0)], &[hamiltonian]).unwrap_err(),
            QuantumError::InvalidObservable {
                reason: "Hamiltonian term is not present in the measured observable basis",
            }
        );
    }

    #[test]
    fn hamiltonian_projection_preserves_quantum_adjoint_gradients() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let circuit = builder.build().unwrap();
        let hamiltonian = Hamiltonian::with_offset(
            vec![HamiltonianTerm::new(1.5, Observable::z(0)).unwrap()],
            0.25,
        )
        .unwrap();
        let readout = circuit.hamiltonian_readout(&[hamiltonian]).unwrap();

        let features_data = [0.2f32, -0.4];
        let theta = 0.3f32;
        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(features_data.to_vec(), 2, 1));
        let parameters = tape.input(Tensor::from_vec(vec![theta], 1, 1));
        let output = circuit
            .forward_hamiltonians(features, parameters, &readout)
            .unwrap();
        let values = tape.value(output.idx()).data;

        for (sample, feature) in features_data.into_iter().enumerate()
        {
            let phase = feature + theta;
            let expected = 1.5 * phase.cos() + 0.25;
            assert!((values[sample] - expected).abs() <= TOLERANCE);
        }

        output.sum().backward();
        let feature_gradient = tape.grad(features.idx()).data;
        let parameter_gradient = tape.grad(parameters.idx()).data;
        let mut expected_parameter_gradient = 0.0f32;
        for (sample, feature) in features_data.into_iter().enumerate()
        {
            let expected = -1.5 * (feature + theta).sin();
            expected_parameter_gradient += expected;
            assert!((feature_gradient[sample] - expected).abs() <= TOLERANCE);
        }
        assert!((parameter_gradient[0] - expected_parameter_gradient).abs() <= TOLERANCE);
    }

    #[test]
    fn readout_is_stateless_module_inside_hybrid_sequential() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0, 1]).unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let circuit = builder.build().unwrap();

        let first = Hamiltonian::new(vec![
            HamiltonianTerm::new(0.5, Observable::z(0)).unwrap(),
            HamiltonianTerm::new(0.25, Observable::z(1)).unwrap(),
        ])
        .unwrap();
        let second = Hamiltonian::with_offset(
            vec![
                HamiltonianTerm::new(-0.75, Observable::z(0)).unwrap(),
                HamiltonianTerm::new(0.4, Observable::z(1)).unwrap(),
            ],
            0.1,
        )
        .unwrap();
        let readout = circuit.hamiltonian_readout(&[first, second]).unwrap();
        let quantum = QuantumModule::new(circuit, ParameterInitializer::Constant(0.2)).unwrap();

        let mut rng = PcgEngine::new(41);
        let mut linear = Linear::new(2, 1, &Zeros, &Zeros, &mut rng);
        linear.weight = Tensor::from_vec(vec![1.0, -1.0], 2, 1);
        linear.bias = Tensor::from_vec(vec![0.0], 1, 1);

        let mut model = Sequential::new().add(quantum).add(readout).add(linear);
        let tape = Tape::new();
        let input = tape.input(Tensor::from_vec(vec![0.2, -0.4], 1, 2));
        let output = model.forward(&tape, input);

        assert_eq!(output.shape(), (1, 1));
        let parameter_indices = model.parameter_indices();
        assert_eq!(parameter_indices.len(), 3);

        output.sum().backward();
        let quantum_gradient = tape.grad(parameter_indices[0]);
        assert_eq!(quantum_gradient.shape(), (1, 2));
        assert!(quantum_gradient.data.iter().any(|value| value.abs() > 1.0e-5));

        let state = model.state_dict();
        assert!(state.contains_key("0.parameters"));
        assert!(!state.keys().any(|key| key.starts_with("1.")));
        assert!(state.contains_key("2.weight"));
        assert!(state.contains_key("2.bias"));
    }

    #[test]
    fn module_rejects_input_from_a_different_tape() {
        let hamiltonian = Hamiltonian::new(vec![
            HamiltonianTerm::new(1.0, Observable::z(0)).unwrap(),
        ])
        .unwrap();
        let mut readout =
            HamiltonianReadout::from_observables(&[Observable::z(0)], &[hamiltonian]).unwrap();
        let input_tape = Tape::new();
        let module_tape = Tape::new();
        let input = input_tape.input(Tensor::from_vec(vec![0.3], 1, 1));

        let error = Module::try_forward(&mut readout, &module_tape, input).unwrap_err();
        assert_eq!(error.code(), "E_CONFIG");
    }

    #[test]
    fn non_finite_coefficients_and_offsets_are_rejected() {
        assert_eq!(
            HamiltonianTerm::new(f32::NAN, Observable::z(0)).unwrap_err(),
            QuantumError::NonFiniteParameter {
                what: "Hamiltonian coefficient",
            }
        );
        assert_eq!(
            Hamiltonian::with_offset(Vec::new(), f32::INFINITY).unwrap_err(),
            QuantumError::NonFiniteParameter {
                what: "Hamiltonian identity offset",
            }
        );
    }
}
