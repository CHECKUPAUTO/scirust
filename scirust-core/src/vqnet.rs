//! High-level deterministic building blocks for VQNet-like hybrid models.
//!
//! This module deliberately composes the existing typed quantum circuit IR and
//! [`QuantumLayer`] autograd integration instead of introducing a second quantum
//! execution path. The builder owns symbolic parameter allocation so callers do
//! not need to coordinate [`ParameterId`] values by hand.

mod ansatz;
mod encoding;
mod module_state;
mod nn_module;
mod optimizer_state;
mod readout;
mod trainer;

pub use ansatz::{EntanglementTopology, EntanglingGate};
pub use encoding::AngleEncodingHandle;
pub use module_state::{
    ParameterInitializer, QuantumForward, QuantumModule, VariationalParameters,
};
pub use optimizer_state::{OptimizerSlot, PersistentParameterOptimizer};
pub use readout::{Hamiltonian, HamiltonianReadout, HamiltonianTerm};
pub use trainer::{TrainStepReport, TrainingSession};

use crate::autodiff::reverse::Var;
use crate::quantum::{
    Circuit, Observable, Operation, Parameter, ParameterId, QuantumError, QuantumLayer,
    QuantumResult,
};
use std::collections::BTreeSet;

/// Rotation axis used by high-level feature encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    /// Pauli-X rotation.
    X,
    /// Pauli-Y rotation.
    Y,
    /// Pauli-Z rotation.
    Z,
}

/// A reusable differentiable variational circuit.
///
/// The contained [`QuantumLayer`] remains the execution and reverse-mode
/// authority. This wrapper exposes the high-level parameter layout produced by
/// [`VariationalCircuitBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct VariationalCircuit {
    layer: QuantumLayer,
}

impl VariationalCircuit {
    /// Executes one single-sample, single-observable forward pass.
    pub fn forward<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
    ) -> QuantumResult<Var<'t>> {
        self.layer.forward(classical_features, quantum_parameters)
    }

    /// Executes a batched, ordered multi-observable forward pass.
    pub fn forward_batch<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
    ) -> QuantumResult<Var<'t>> {
        self.layer
            .forward_batch(classical_features, quantum_parameters)
    }

    /// Builds a fixed Hamiltonian projection against this circuit's exact
    /// measured-observable basis.
    pub fn hamiltonian_readout(
        &self,
        hamiltonians: &[Hamiltonian],
    ) -> QuantumResult<HamiltonianReadout> {
        HamiltonianReadout::from_observables(self.observables(), hamiltonians)
    }

    /// Executes the existing batched adjoint-backed expectation path and then
    /// projects those ordered expectations into one or more Hamiltonian values.
    ///
    /// The readout must have been built against the same semantic observable
    /// basis. Its linear projection is ordinary SciRust reverse-mode matrix
    /// multiplication, so gradients propagate unchanged through the quantum
    /// adjoint node to encoded features and trainable quantum parameters.
    pub fn forward_hamiltonians<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
        readout: &HamiltonianReadout,
    ) -> QuantumResult<Var<'t>> {
        readout.validate_circuit(self)?;
        let expectations = self.forward_batch(classical_features, quantum_parameters)?;
        readout.apply(expectations)
    }

    /// Underlying typed circuit template.
    #[must_use]
    pub fn circuit(&self) -> &Circuit {
        self.layer.circuit()
    }

    /// Ordered output observables.
    #[must_use]
    pub fn observables(&self) -> &[Observable] {
        self.layer.observables()
    }

    /// Feature-column parameter IDs in deterministic allocation order.
    #[must_use]
    pub fn input_parameters(&self) -> &[ParameterId] {
        self.layer.input_parameters()
    }

    /// Trainable parameter IDs in deterministic allocation order.
    #[must_use]
    pub fn trainable_parameters(&self) -> &[ParameterId] {
        self.layer.trainable_parameters()
    }

    /// Number of classical feature columns expected by [`Self::forward_batch`].
    #[must_use]
    pub fn input_parameter_count(&self) -> usize {
        self.input_parameters().len()
    }

    /// Number of shared trainable quantum parameters expected by
    /// [`Self::forward_batch`].
    #[must_use]
    pub fn trainable_parameter_count(&self) -> usize {
        self.trainable_parameters().len()
    }

    /// Access to the lower-level differentiable quantum layer.
    #[must_use]
    pub const fn layer(&self) -> &QuantumLayer {
        &self.layer
    }
}

/// Deterministic builder for a differentiable variational circuit.
///
/// Parameter IDs are allocated from zero in exact builder-call and qubit order.
/// Input and trainable parameters are recorded separately and then validated by
/// [`QuantumLayer::new_multi`] when [`Self::build`] is called.
#[derive(Debug, Clone, PartialEq)]
pub struct VariationalCircuitBuilder {
    circuit: Circuit,
    input_parameters: Vec<ParameterId>,
    trainable_parameters: Vec<ParameterId>,
    observables: Vec<Observable>,
    next_parameter: u32,
}

impl VariationalCircuitBuilder {
    /// Creates an empty high-level circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> QuantumResult<Self> {
        Ok(Self {
            circuit: Circuit::new(num_qubits)?,
            input_parameters: Vec::new(),
            trainable_parameters: Vec::new(),
            observables: Vec::new(),
            next_parameter: 0,
        })
    }

    /// Number of qubits in the circuit under construction.
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    /// Appends one symbolic angle-encoding rotation for every listed qubit.
    ///
    /// The feature tensor column order is exactly `qubits` order. Qubits must be
    /// distinct and inside the circuit domain.
    pub fn angle_encoding(
        &mut self,
        axis: RotationAxis,
        qubits: &[usize],
    ) -> QuantumResult<&mut Self> {
        validate_distinct_qubits(qubits, self.num_qubits())?;

        for &qubit in qubits
        {
            let parameter = self.allocate_input_parameter()?;
            self.circuit.push(rotation_operation(
                axis,
                qubit,
                Parameter::Symbol(parameter),
            ))?;
        }

        Ok(self)
    }

    /// Appends a deterministic hardware-efficient variational ansatz.
    ///
    /// Each layer applies one symbolic `Ry` and one symbolic `Rz` rotation to
    /// every qubit, followed by a nearest-neighbour CNOT chain
    /// `0→1→...→n-1`. Trainable parameter order is layer, rotation family, then
    /// ascending qubit index. At least one layer is required.
    pub fn hardware_efficient_ansatz(&mut self, layers: usize) -> QuantumResult<&mut Self> {
        if layers == 0
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "hardware-efficient ansatz needs at least one layer",
            });
        }

        for _ in 0..layers
        {
            for qubit in 0..self.num_qubits()
            {
                let parameter = self.allocate_trainable_parameter()?;
                self.circuit.push(Operation::Ry {
                    target: qubit,
                    parameter: Parameter::Symbol(parameter),
                })?;
            }

            for qubit in 0..self.num_qubits()
            {
                let parameter = self.allocate_trainable_parameter()?;
                self.circuit.push(Operation::Rz {
                    target: qubit,
                    parameter: Parameter::Symbol(parameter),
                })?;
            }

            for control in 0..self.num_qubits().saturating_sub(1)
            {
                self.circuit.push(Operation::Cnot {
                    control,
                    target: control + 1,
                })?;
            }
        }

        Ok(self)
    }

    /// Adds one output observable after validating every referenced qubit.
    pub fn measure(&mut self, observable: Observable) -> QuantumResult<&mut Self> {
        for term in observable.terms()
        {
            validate_qubit(term.qubit, self.num_qubits())?;
        }
        self.observables.push(observable);
        Ok(self)
    }

    /// Adds one Pauli-Z observable per listed qubit in exact list order.
    pub fn measure_z(&mut self, qubits: &[usize]) -> QuantumResult<&mut Self> {
        validate_distinct_qubits(qubits, self.num_qubits())?;
        for &qubit in qubits
        {
            self.observables.push(Observable::z(qubit));
        }
        Ok(self)
    }

    /// Adds one Pauli-Z observable for every qubit in ascending order.
    pub fn measure_all_z(&mut self) -> QuantumResult<&mut Self> {
        for qubit in 0..self.num_qubits()
        {
            self.observables.push(Observable::z(qubit));
        }
        Ok(self)
    }

    /// Finalizes the builder into the existing differentiable [`QuantumLayer`]
    /// execution path.
    pub fn build(self) -> QuantumResult<VariationalCircuit> {
        if self.observables.is_empty()
        {
            return Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            });
        }

        let layer = QuantumLayer::new_multi(
            self.circuit,
            self.observables,
            self.input_parameters,
            self.trainable_parameters,
        )?;
        Ok(VariationalCircuit { layer })
    }

    fn allocate_input_parameter(&mut self) -> QuantumResult<ParameterId> {
        let parameter = self.allocate_parameter()?;
        self.input_parameters.push(parameter);
        Ok(parameter)
    }

    fn allocate_trainable_parameter(&mut self) -> QuantumResult<ParameterId> {
        let parameter = self.allocate_parameter()?;
        self.trainable_parameters.push(parameter);
        Ok(parameter)
    }

    fn allocate_parameter(&mut self) -> QuantumResult<ParameterId> {
        let parameter = ParameterId(self.next_parameter);
        self.next_parameter =
            self.next_parameter
                .checked_add(1)
                .ok_or(QuantumError::NumericalFailure {
                    operation: "VQNet parameter ID allocation",
                })?;
        Ok(parameter)
    }
}

fn rotation_operation(axis: RotationAxis, target: usize, parameter: Parameter) -> Operation {
    match axis
    {
        RotationAxis::X => Operation::Rx { target, parameter },
        RotationAxis::Y => Operation::Ry { target, parameter },
        RotationAxis::Z => Operation::Rz { target, parameter },
    }
}

fn validate_distinct_qubits(qubits: &[usize], num_qubits: usize) -> QuantumResult<()> {
    let mut seen = BTreeSet::new();
    for &qubit in qubits
    {
        validate_qubit(qubit, num_qubits)?;
        if !seen.insert(qubit)
        {
            return Err(QuantumError::DuplicateQubit { qubit });
        }
    }
    Ok(())
}

fn validate_qubit(qubit: usize, num_qubits: usize) -> QuantumResult<()> {
    if qubit < num_qubits
    {
        Ok(())
    }
    else
    {
        Err(QuantumError::InvalidQubitIndex { qubit, num_qubits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};

    const TOLERANCE: f32 = 2.0e-5;

    #[test]
    fn builder_assigns_stable_parameter_ids_and_operation_order() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0, 1]).unwrap();
        builder.hardware_efficient_ansatz(2).unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        assert_eq!(model.input_parameters(), &[ParameterId(0), ParameterId(1)]);
        assert_eq!(
            model.trainable_parameters(),
            &[
                ParameterId(2),
                ParameterId(3),
                ParameterId(4),
                ParameterId(5),
                ParameterId(6),
                ParameterId(7),
                ParameterId(8),
                ParameterId(9),
            ]
        );
        assert_eq!(model.circuit().operations().len(), 12);
        assert_eq!(model.observables(), &[Observable::z(0), Observable::z(1)]);
        assert!(matches!(
            model.circuit().operations()[0],
            Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(ParameterId(0)),
            }
        ));
        assert!(matches!(
            model.circuit().operations()[6],
            Operation::Cnot {
                control: 0,
                target: 1,
            }
        ));
    }

    #[test]
    fn high_level_model_runs_through_existing_autograd_path() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder.hardware_efficient_ansatz(1).unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(vec![0.2, -0.4], 2, 1));
        let parameters = tape.input(Tensor::from_vec(vec![0.3, 0.5], 1, 2));
        let output = model.forward_batch(features, parameters).unwrap();
        let values = tape.value(output.idx()).data;

        let expected = [(0.2f32 + 0.3).cos(), (-0.4f32 + 0.3).cos()];
        for (actual, expected) in values.iter().zip(expected)
        {
            assert!((*actual - expected).abs() <= TOLERANCE);
        }

        output.sum().backward();
        let feature_gradient = tape.grad(features.idx()).data;
        let parameter_gradient = tape.grad(parameters.idx()).data;
        let expected_feature = [-(0.2f32 + 0.3).sin(), -(-0.4f32 + 0.3).sin()];

        for (actual, expected) in feature_gradient.iter().zip(expected_feature)
        {
            assert!((*actual - expected).abs() <= TOLERANCE);
        }
        assert!((parameter_gradient[0] - expected_feature.iter().sum::<f32>()).abs() <= TOLERANCE);
        assert!(parameter_gradient[1].abs() <= TOLERANCE);
    }

    #[test]
    fn duplicate_encoding_qubits_are_rejected_before_mutation() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        assert_eq!(
            builder
                .angle_encoding(RotationAxis::X, &[0, 0])
                .unwrap_err(),
            QuantumError::DuplicateQubit { qubit: 0 }
        );
        assert!(builder.circuit.operations().is_empty());
        assert!(builder.input_parameters.is_empty());
    }

    #[test]
    fn measurement_qubits_are_checked_against_circuit_domain() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        assert_eq!(
            builder.measure(Observable::z(2)).unwrap_err(),
            QuantumError::InvalidQubitIndex {
                qubit: 2,
                num_qubits: 2,
            }
        );
    }

    #[test]
    fn zero_layer_ansatz_is_rejected() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        assert_eq!(
            builder.hardware_efficient_ansatz(0).unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "hardware-efficient ansatz needs at least one layer",
            }
        );
    }

    #[test]
    fn build_requires_at_least_one_measurement() {
        let builder = VariationalCircuitBuilder::new(1).unwrap();
        assert_eq!(
            builder.build().unwrap_err(),
            QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            }
        );
    }
}
