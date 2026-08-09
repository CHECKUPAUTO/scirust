//! Stable optimizer identity across fresh reverse-mode tapes.
//!
//! Tape-node indices are intentionally local to one
//! [`Tape`](crate::autodiff::reverse::Tape). Stateful optimizers such as Adam
//! therefore need an identity that does not depend on a node index when a
//! VQNet-like model rebuilds its tape every training step. This module adapts
//! SciRust's existing raw-slice optimizers to that contract; it does not
//! reimplement their update rules.

use super::QuantumForward;
use crate::autodiff::reverse::Tensor;
use crate::quantum::{QuantumError, QuantumResult};

/// Adapter contract for stateful optimizers whose state is keyed independently
/// from one reverse-mode tape.
///
/// Implementations must update `parameters` in place using `gradient` and keep
/// any optimizer state associated with `parameter_key`. The trait requires
/// [`Clone`] so [`OptimizerSlot::step`] can make the optimizer update
/// transactional with respect to non-finite output validation.
pub trait PersistentParameterOptimizer: Clone {
    /// Applies one optimizer step to one persistent parameter vector.
    fn step(&mut self, parameter_key: &str, gradient: &[f32], parameters: &mut [f32]);
}

impl PersistentParameterOptimizer for crate::optim::AdamW {
    fn step(&mut self, parameter_key: &str, gradient: &[f32], parameters: &mut [f32]) {
        crate::optim::AdamW::step(self, parameter_key, gradient, parameters);
    }
}

impl PersistentParameterOptimizer for crate::optim::LAMB {
    fn step(&mut self, parameter_key: &str, gradient: &[f32], parameters: &mut [f32]) {
        crate::optim::LAMB::step(self, parameter_key, gradient, parameters);
    }
}

/// Stable string identity for one trainable parameter slot.
///
/// The key is independent of tape construction order and can therefore be
/// reused across fresh tapes without aliasing optimizer moments to a temporary
/// node index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OptimizerSlot {
    key: String,
}

impl OptimizerSlot {
    /// Creates a stable optimizer slot.
    ///
    /// Empty or surrounding-whitespace-only names are rejected so logs,
    /// checkpoints, and composed models can use the key without ambiguity.
    pub fn new(key: impl Into<String>) -> QuantumResult<Self> {
        let key = key.into();
        if key.is_empty() || key.trim() != key
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "VQNet optimizer slot key must be non-empty and trimmed",
            });
        }
        Ok(Self { key })
    }

    /// Stable optimizer key.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Applies one persistent optimizer step to the parameter variable from
    /// `forward`.
    ///
    /// The gradient and current parameter value are read from the tape. The
    /// optimizer and parameter vector are first updated on clones. Both are
    /// committed only if the resulting parameter vector is finite and retains
    /// the exact shape, so a failing custom optimizer cannot partially corrupt
    /// either the tape value or its optimizer state.
    pub fn step<O: PersistentParameterOptimizer>(
        &self,
        optimizer: &mut O,
        forward: QuantumForward<'_>,
    ) -> QuantumResult<()> {
        let parameter = forward.parameters();
        let tape = parameter.tape;
        let value = tape.value(parameter.idx());
        let gradient = tape.grad(parameter.idx());

        if value.shape() != gradient.shape()
        {
            return Err(QuantumError::InvalidTensorShape {
                tensor: "vqnet_parameter_gradient",
                expected_rows: Some(value.rows),
                expected_cols: Some(value.cols),
                actual_rows: gradient.rows,
                actual_cols: gradient.cols,
            });
        }
        if !gradient.data.iter().all(|value| value.is_finite())
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "VQNet parameter gradient",
            });
        }

        let mut candidate_values = value.data.clone();
        let mut candidate_optimizer = optimizer.clone();
        candidate_optimizer.step(self.key(), &gradient.data, &mut candidate_values);

        if !candidate_values.iter().all(|value| value.is_finite())
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "VQNet optimizer output",
            });
        }

        tape.set_value(
            parameter.idx(),
            Tensor::from_vec(candidate_values, value.rows, value.cols),
        );
        *optimizer = candidate_optimizer;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::optim::AdamW;
    use crate::vqnet::{
        ParameterInitializer, QuantumModule, RotationAxis, VariationalCircuit,
        VariationalCircuitBuilder,
    };

    fn one_qubit_circuit() -> VariationalCircuit {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder.hardware_efficient_ansatz(1).unwrap();
        builder.measure_all_z().unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn slot_requires_a_stable_trimmed_name() {
        for invalid in ["", " quantum", "quantum ", " quantum "]
        {
            assert_eq!(
                OptimizerSlot::new(invalid).unwrap_err(),
                QuantumError::InvalidParameterMapping {
                    reason: "VQNet optimizer slot key must be non-empty and trimmed",
                }
            );
        }
        assert_eq!(
            OptimizerSlot::new("classifier.quantum").unwrap().key(),
            "classifier.quantum"
        );
    }

    #[test]
    fn raw_adamw_state_matches_reference_across_fresh_tapes() {
        let mut module =
            QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Constant(0.3)).unwrap();
        let slot = OptimizerSlot::new("classifier.quantum").unwrap();
        let mut optimizer = AdamW::new(0.025).with_weight_decay(0.0);
        let mut reference_optimizer = AdamW::new(0.025).with_weight_decay(0.0);
        let mut reference_values = module.parameters().values().to_vec();

        for features_data in [[0.2f32, -0.4], [0.7, 0.1], [-0.3, 0.9]]
        {
            let tape = Tape::new();
            let features = tape.input(Tensor::from_vec(features_data.to_vec(), 2, 1));
            let forward = module.forward_batch(&tape, features).unwrap();
            forward.output().sum().backward();

            let gradient = tape.grad(forward.parameter_index()).data;
            reference_optimizer.step(slot.key(), &gradient, &mut reference_values);

            slot.step(&mut optimizer, forward).unwrap();
            module.sync_parameters(forward.parameters()).unwrap();

            assert_eq!(module.parameters().values(), reference_values.as_slice());
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct InvalidOptimizer {
        calls: usize,
    }

    impl PersistentParameterOptimizer for InvalidOptimizer {
        fn step(&mut self, _parameter_key: &str, _gradient: &[f32], parameters: &mut [f32]) {
            self.calls += 1;
            parameters[0] = f32::NAN;
        }
    }

    #[test]
    fn invalid_optimizer_output_is_transactional() {
        let module =
            QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Constant(0.3)).unwrap();
        let slot = OptimizerSlot::new("classifier.quantum").unwrap();
        let mut optimizer = InvalidOptimizer { calls: 0 };

        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(vec![0.2, -0.4], 2, 1));
        let forward = module.forward_batch(&tape, features).unwrap();
        forward.output().sum().backward();
        let before = tape.value(forward.parameter_index());

        assert_eq!(
            slot.step(&mut optimizer, forward).unwrap_err(),
            QuantumError::NonFiniteParameter {
                what: "VQNet optimizer output",
            }
        );
        assert_eq!(optimizer.calls, 0);
        assert_eq!(tape.value(forward.parameter_index()).data, before.data);
    }
}
