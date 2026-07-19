//! Deterministic batched SciRust reverse-mode integration for exact expectations.

use super::error::{QuantumError, QuantumResult};
use super::ir::{Circuit, ParameterId};
use super::observable::Observable;
use crate::autodiff::reverse::Var;
use std::collections::BTreeSet;

/// A batched quantum layer backed by exact dense CPU execution.
///
/// `input_parameters[i]` binds feature tensor column `i`; likewise,
/// `trainable_parameters[i]` binds quantum-parameter tensor column `i`.
/// Every circuit symbol must occur exactly once across those two mappings.
/// [`Self::forward_batch`] accepts features shaped `batch x input_parameter_count`
/// and one shared quantum-parameter row shaped `1 x trainable_parameter_count`.
/// It returns `batch x observable_count` in row-major `output[sample, observable]`
/// order. Backward uses parameter shift for `Rx`, `Ry`, and `Rz`, including
/// encoded inputs, so gradients can reach classical layers before this
/// operation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumLayer {
    circuit: Circuit,
    observables: Vec<Observable>,
    input_parameters: Vec<ParameterId>,
    trainable_parameters: Vec<ParameterId>,
}

impl QuantumLayer {
    /// Constructs a backward-compatible layer containing one observable.
    pub fn new(
        circuit: Circuit,
        observable: Observable,
        input_parameters: Vec<ParameterId>,
        trainable_parameters: Vec<ParameterId>,
    ) -> QuantumResult<Self> {
        Self::new_multi(
            circuit,
            vec![observable],
            input_parameters,
            trainable_parameters,
        )
    }

    /// Constructs a layer with observables retained in output-column order.
    ///
    /// The observable list must be non-empty. The two parameter mappings must
    /// together cover every circuit symbol exactly once without duplicates.
    pub fn new_multi(
        circuit: Circuit,
        observables: Vec<Observable>,
        input_parameters: Vec<ParameterId>,
        trainable_parameters: Vec<ParameterId>,
    ) -> QuantumResult<Self> {
        if observables.is_empty()
        {
            return Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            });
        }
        let circuit_ids: BTreeSet<_> = circuit.parameter_ids().into_iter().collect();
        let mut mapped = BTreeSet::new();
        for &id in input_parameters.iter().chain(&trainable_parameters)
        {
            if !mapped.insert(id)
            {
                return Err(QuantumError::InvalidParameterMapping {
                    reason: "a parameter ID appears more than once",
                });
            }
            circuit.parameter_occurrences(id)?;
        }
        if mapped != circuit_ids
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "mappings must cover every circuit symbol exactly once",
            });
        }
        Ok(Self {
            circuit,
            observables,
            input_parameters,
            trainable_parameters,
        })
    }

    /// Adds one single-sample, single-observable expectation node.
    ///
    /// This preserves the original tensor contract: features are
    /// `1 x input_parameter_count`, parameters are
    /// `1 x trainable_parameter_count`, and the output is `1 x 1`. A layer
    /// constructed with multiple observables is rejected; use
    /// [`Self::forward_batch`] for that case.
    pub fn forward<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
    ) -> QuantumResult<Var<'t>> {
        if self.observables.len() != 1
        {
            return Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: Some(1),
                actual: self.observables.len(),
            });
        }
        if classical_features.shape().0 != 1
        {
            let (actual_rows, actual_cols) = classical_features.shape();
            return Err(QuantumError::InvalidTensorShape {
                tensor: "classical_features",
                expected_rows: Some(1),
                expected_cols: Some(self.input_parameters.len()),
                actual_rows,
                actual_cols,
            });
        }
        self.forward_batch(classical_features, quantum_parameters)
    }

    /// Adds one deterministic batched, multi-observable autograd node.
    ///
    /// `classical_features` has shape `batch x input_parameter_count` with
    /// `batch >= 1`. `quantum_parameters` has exactly one row and is shared by
    /// every sample. The row-major output has shape `batch x observable_count`,
    /// where `output[sample, observable]` is evaluated after executing that
    /// sample's bound circuit once.
    ///
    /// Reverse mode accumulates in fixed sample, mapped-parameter, then
    /// observable order. Feature gradients keep the feature tensor shape;
    /// shared-parameter gradients sum across samples into one row. No implicit
    /// batch averaging is performed.
    pub fn forward_batch<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
    ) -> QuantumResult<Var<'t>> {
        classical_features.try_quantum_expectations(quantum_parameters, self)
    }

    /// Circuit template.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// The sole measured Pauli product for a single-observable layer.
    ///
    /// For a multi-observable layer use [`Self::observables`] or
    /// [`Self::try_observable`].
    ///
    /// # Panics
    ///
    /// Panics if this layer contains more than one observable.
    pub fn observable(&self) -> &Observable {
        self.try_observable()
            .expect("QuantumLayer::observable requires exactly one observable")
    }

    /// Returns the sole observable, rejecting a multi-observable layer.
    pub fn try_observable(&self) -> QuantumResult<&Observable> {
        if self.observables.len() == 1
        {
            Ok(&self.observables[0])
        }
        else
        {
            Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: Some(1),
                actual: self.observables.len(),
            })
        }
    }

    /// Measured Pauli products in deterministic output-column order.
    pub fn observables(&self) -> &[Observable] {
        &self.observables
    }

    /// Feature-column parameter IDs.
    pub fn input_parameters(&self) -> &[ParameterId] {
        &self.input_parameters
    }

    /// Trainable-tensor parameter IDs.
    pub fn trainable_parameters(&self) -> &[ParameterId] {
        &self.trainable_parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::quantum::gradient::expectation_value;
    use crate::quantum::ir::{Operation, Parameter, ParameterValues};

    const FEATURES: [f32; 3] = [-0.71, 0.13, 1.04];
    const THETA: f32 = -0.27;
    const UPSTREAM: [f32; 6] = [0.7, -1.1, -0.4, 0.3, 1.2, 0.5];
    const ANALYTIC_TOLERANCE: f32 = 7.0e-5;
    const FINITE_DIFFERENCE_TOLERANCE: f32 = 1.5e-3;
    const FINITE_DIFFERENCE_EPSILON: f32 = 1.0e-3;

    fn layer() -> QuantumLayer {
        let input = ParameterId(0);
        let trainable = ParameterId(1);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(input),
            })
            .unwrap()
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(trainable),
            })
            .unwrap();
        QuantumLayer::new(circuit, Observable::z(0), vec![input], vec![trainable]).unwrap()
    }

    fn multi_layer() -> QuantumLayer {
        let input = ParameterId(0);
        let trainable = ParameterId(1);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(input),
            })
            .unwrap()
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(trainable),
            })
            .unwrap();
        QuantumLayer::new_multi(
            circuit,
            vec![Observable::z(0), Observable::x(0)],
            vec![input],
            vec![trainable],
        )
        .unwrap()
    }

    fn parameter_values(feature: f32, theta: f32) -> ParameterValues {
        ParameterValues::new()
            .with(ParameterId(0), feature)
            .unwrap()
            .with(ParameterId(1), theta)
            .unwrap()
    }

    fn weighted_autograd(layer: &QuantumLayer) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(FEATURES.to_vec(), 3, 1));
        let parameters = tape.input(Tensor::from_vec(vec![THETA], 1, 1));
        let output = layer.forward_batch(features, parameters).unwrap();
        let output_data = tape.value(output.idx()).data;
        let upstream = tape.input(Tensor::from_vec(UPSTREAM.to_vec(), 3, 2));
        output.hadamard(upstream).sum().backward();
        (
            output_data,
            tape.grad(features.idx()).data,
            tape.grad(parameters.idx()).data,
        )
    }

    fn direct_weighted_objective(layer: &QuantumLayer, theta: f32) -> f32 {
        let mut objective = 0.0f32;
        for (sample, &feature) in FEATURES.iter().enumerate()
        {
            let values = parameter_values(feature, theta);
            let state = layer
                .circuit()
                .bind(&values)
                .unwrap()
                .execute_dense()
                .unwrap();
            for (observable, measured) in layer.observables().iter().enumerate()
            {
                objective +=
                    UPSTREAM[sample * 2 + observable] * state.expectation(measured).unwrap();
            }
        }
        objective
    }

    #[test]
    fn multi_observable_forward_matches_closed_form_for_three_rows() {
        let layer = multi_layer();
        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(FEATURES.to_vec(), 3, 1));
        let parameters = tape.input(Tensor::from_vec(vec![THETA], 1, 1));
        let output = layer.forward_batch(features, parameters).unwrap();
        let actual = tape.value(output.idx());
        assert_eq!(actual.shape(), (3, 2));

        let mut max_error = 0.0f32;
        for (sample, &feature) in FEATURES.iter().enumerate()
        {
            let phi = feature + THETA;
            let expected = [phi.cos(), phi.sin()];
            for (observable, &expected) in expected.iter().enumerate()
            {
                let error = (actual.data[sample * 2 + observable] - expected).abs();
                max_error = max_error.max(error);
                assert!(error <= ANALYTIC_TOLERANCE, "forward error {error}");
            }
        }
        eprintln!("quantum batched max analytic forward error: {max_error:.9e}");
    }

    #[test]
    fn batched_forward_matches_independently_executed_scalar_circuits() {
        let layer = multi_layer();
        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(FEATURES.to_vec(), 3, 1));
        let parameters = tape.input(Tensor::from_vec(vec![THETA], 1, 1));
        let output = layer.forward_batch(features, parameters).unwrap();
        let batched = tape.value(output.idx());

        for (sample, &feature) in FEATURES.iter().enumerate()
        {
            let values = parameter_values(feature, THETA);
            for (observable, measured) in layer.observables().iter().enumerate()
            {
                let scalar = expectation_value(layer.circuit(), &values, measured).unwrap();
                assert_eq!(batched.data[sample * 2 + observable], scalar);
            }
        }
    }

    #[test]
    fn feature_gradients_contract_nonuniform_upstream_weights() {
        let (_, actual, _) = weighted_autograd(&multi_layer());
        let mut max_error = 0.0f32;
        for (sample, &feature) in FEATURES.iter().enumerate()
        {
            let phi = feature + THETA;
            let expected = UPSTREAM[sample * 2] * -phi.sin() + UPSTREAM[sample * 2 + 1] * phi.cos();
            let error = (actual[sample] - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= ANALYTIC_TOLERANCE,
                "feature gradient error {error}"
            );
        }
        eprintln!("quantum batched max analytic feature-gradient error: {max_error:.9e}");
    }

    #[test]
    fn shared_parameter_gradient_sums_every_sample_and_observable() {
        let (_, _, actual) = weighted_autograd(&multi_layer());
        let mut expected = 0.0f32;
        for (sample, &feature) in FEATURES.iter().enumerate()
        {
            let phi = feature + THETA;
            expected += UPSTREAM[sample * 2] * -phi.sin() + UPSTREAM[sample * 2 + 1] * phi.cos();
        }
        let error = (actual[0] - expected).abs();
        eprintln!("quantum batched analytic shared-parameter error: {error:.9e}");
        assert!(
            error <= ANALYTIC_TOLERANCE,
            "parameter gradient error {error}"
        );
    }

    #[test]
    fn weighted_batched_objective_matches_central_finite_difference() {
        let layer = multi_layer();
        let (_, _, analytic) = weighted_autograd(&layer);
        let upper = direct_weighted_objective(&layer, THETA + FINITE_DIFFERENCE_EPSILON);
        let lower = direct_weighted_objective(&layer, THETA - FINITE_DIFFERENCE_EPSILON);
        let finite = (upper - lower) / (2.0 * FINITE_DIFFERENCE_EPSILON);
        let error = (analytic[0] - finite).abs();
        eprintln!("quantum batched finite-difference comparison error: {error:.9e}");
        assert!(
            error <= FINITE_DIFFERENCE_TOLERANCE,
            "analytic {}, finite difference {finite}, error {error}",
            analytic[0]
        );
    }

    #[test]
    fn batched_forward_and_backward_are_exactly_repeatable_on_fresh_tapes() {
        let layer = multi_layer();
        assert_eq!(weighted_autograd(&layer), weighted_autograd(&layer));
    }

    #[test]
    fn batched_tensor_contract_rejections_are_structured() {
        let layer = multi_layer();
        let tape = Tape::new();
        let parameters = tape.input(Tensor::from_vec(vec![THETA], 1, 1));

        let zero_batch = tape.input(Tensor::from_vec(Vec::new(), 0, 1));
        assert_eq!(
            layer.forward_batch(zero_batch, parameters).unwrap_err(),
            QuantumError::InvalidBatchSize {
                minimum: 1,
                actual: 0,
            }
        );

        let wrong_feature_columns = tape.input(Tensor::from_vec(vec![0.0; 6], 3, 2));
        assert_eq!(
            layer
                .forward_batch(wrong_feature_columns, parameters)
                .unwrap_err(),
            QuantumError::InvalidTensorShape {
                tensor: "classical_features",
                expected_rows: None,
                expected_cols: Some(1),
                actual_rows: 3,
                actual_cols: 2,
            }
        );

        let features = tape.input(Tensor::from_vec(FEATURES.to_vec(), 3, 1));
        let per_sample_parameters = tape.input(Tensor::from_vec(vec![THETA; 2], 2, 1));
        assert_eq!(
            layer
                .forward_batch(features, per_sample_parameters)
                .unwrap_err(),
            QuantumError::InvalidTensorShape {
                tensor: "quantum_parameters",
                expected_rows: Some(1),
                expected_cols: Some(1),
                actual_rows: 2,
                actual_cols: 1,
            }
        );

        let wrong_parameter_columns = tape.input(Tensor::from_vec(vec![THETA; 2], 1, 2));
        assert_eq!(
            layer
                .forward_batch(features, wrong_parameter_columns)
                .unwrap_err(),
            QuantumError::InvalidTensorShape {
                tensor: "quantum_parameters",
                expected_rows: Some(1),
                expected_cols: Some(1),
                actual_rows: 1,
                actual_cols: 2,
            }
        );

        assert_eq!(
            layer.forward(features, parameters).unwrap_err(),
            QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: Some(1),
                actual: 2,
            }
        );

        let other_tape = Tape::new();
        let other_parameters = other_tape.input(Tensor::from_vec(vec![THETA], 1, 1));
        assert_eq!(
            layer.forward_batch(features, other_parameters).unwrap_err(),
            QuantumError::MismatchedAutodiffTapes
        );

        assert_eq!(
            QuantumLayer::new_multi(
                layer.circuit().clone(),
                Vec::new(),
                vec![ParameterId(0)],
                vec![ParameterId(1)],
            ),
            Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            })
        );
        assert_eq!(
            layer.try_observable().unwrap_err(),
            QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: Some(1),
                actual: 2,
            }
        );
    }

    #[test]
    fn single_output_constructor_and_forward_keep_the_original_contract() {
        let layer = layer();
        let tape = Tape::new();
        let feature = tape.input(Tensor::from_vec(vec![0.31], 1, 1));
        let parameter = tape.input(Tensor::from_vec(vec![-0.17], 1, 1));
        let output = layer.forward(feature, parameter).unwrap();
        let actual = tape.value(output.idx());
        assert_eq!(actual.shape(), (1, 1));
        assert!((actual.data[0] - (0.31f32 - 0.17).cos()).abs() < ANALYTIC_TOLERANCE);
        assert_eq!(layer.observable(), &Observable::z(0));
        assert_eq!(layer.observables(), &[Observable::z(0)]);
    }

    #[test]
    fn autograd_matches_closed_form_for_inputs_and_parameters() {
        // output = cos(input + theta)
        let tape = Tape::new();
        let input = tape.input(Tensor::from_vec(vec![0.31], 1, 1));
        let theta = tape.input(Tensor::from_vec(vec![-0.17], 1, 1));
        let output = layer().forward(input, theta).unwrap();
        output.backward();
        let expected = -(0.31f32 - 0.17).sin();
        assert!((tape.value(output.idx()).data[0] - (0.31f32 - 0.17).cos()).abs() < 5.0e-5);
        assert!((tape.grad(input.idx()).data[0] - expected).abs() < 5.0e-5);
        assert!((tape.grad(theta.idx()).data[0] - expected).abs() < 5.0e-5);
    }

    #[test]
    fn gradient_reaches_a_preceding_classical_weight() {
        // feature = x*w; output = cos(feature + theta), so doutput/dw = -sin(...)*x.
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.6], 1, 1));
        let weight = tape.input(Tensor::from_vec(vec![0.4], 1, 1));
        let theta = tape.input(Tensor::from_vec(vec![0.2], 1, 1));
        let feature = x.matmul(weight);
        let output = layer().forward(feature, theta).unwrap();
        output.backward();
        let expected = -(0.6f32 * 0.4 + 0.2).sin() * 0.6;
        assert!((tape.grad(weight.idx()).data[0] - expected).abs() < 7.0e-5);
        assert!(tape.grad(weight.idx()).data[0].abs() > 0.1);
    }

    #[test]
    fn incomplete_or_duplicate_mapping_is_rejected() {
        let parameter = ParameterId(5);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        assert!(
            QuantumLayer::new(circuit.clone(), Observable::z(0), Vec::new(), Vec::new(),).is_err()
        );
        assert!(
            QuantumLayer::new(circuit, Observable::z(0), vec![parameter], vec![parameter],)
                .is_err()
        );
    }
}
