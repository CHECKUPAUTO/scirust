//! SciRust reverse-mode integration for one exact expectation output.

use super::error::{QuantumError, QuantumResult};
use super::ir::{Circuit, ParameterId};
use super::observable::Observable;
use crate::autodiff::reverse::Var;
use std::collections::BTreeSet;

/// A single-sample quantum layer backed by exact dense CPU execution.
///
/// `input_parameters[i]` binds feature tensor column `i`; likewise,
/// `trainable_parameters[i]` binds quantum-parameter tensor column `i`.
/// Every circuit symbol must occur exactly once across those two mappings.
/// Backward uses parameter shift for `Rx`, `Ry`, and `Rz`, including encoded
/// inputs, so gradients can reach classical layers before this operation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumLayer {
    circuit: Circuit,
    observable: Observable,
    input_parameters: Vec<ParameterId>,
    trainable_parameters: Vec<ParameterId>,
}

impl QuantumLayer {
    /// Validates a complete, duplicate-free mapping onto supported rotation parameters.
    pub fn new(
        circuit: Circuit,
        observable: Observable,
        input_parameters: Vec<ParameterId>,
        trainable_parameters: Vec<ParameterId>,
    ) -> QuantumResult<Self> {
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
            observable,
            input_parameters,
            trainable_parameters,
        })
    }

    /// Adds one quantum expectation node to the same tape as both inputs.
    /// Inputs must each have shape `1 x number_of_mapped_parameters`.
    pub fn forward<'t>(
        &self,
        classical_features: Var<'t>,
        quantum_parameters: Var<'t>,
    ) -> QuantumResult<Var<'t>> {
        classical_features.try_quantum_expectation(quantum_parameters, self)
    }

    /// Circuit template.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Measured Pauli product.
    pub fn observable(&self) -> &Observable {
        &self.observable
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
    use crate::quantum::ir::{Operation, Parameter};

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
