//! Module-owned trainable state for the high-level VQNet-like facade.
//!
//! SciRust's reverse-mode tape is intentionally rebuilt for iterative training.
//! This module keeps quantum parameter values outside any one tape, injects one
//! shared `1 × parameter_count` row for a forward pass, and can synchronize an
//! optimizer-updated tape value back into persistent module state.

use super::VariationalCircuit;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::quantum::{QuantumError, QuantumResult};

/// Deterministic initialization policy for trainable quantum angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterInitializer {
    /// Initialize every trainable parameter to zero.
    Zeros,
    /// Initialize every trainable parameter to the same finite value.
    Constant(f32),
    /// Draw deterministic values from `[low, high)` using a local SplitMix64
    /// stream seeded only by `seed`.
    Uniform { seed: u64, low: f32, high: f32 },
}

impl ParameterInitializer {
    fn initialize(self, count: usize) -> QuantumResult<Vec<f32>> {
        match self
        {
            Self::Zeros => Ok(vec![0.0; count]),
            Self::Constant(value) =>
            {
                ensure_finite(value, "VQNet constant initializer")?;
                Ok(vec![value; count])
            },
            Self::Uniform { seed, low, high } =>
            {
                ensure_finite(low, "VQNet uniform initializer lower bound")?;
                ensure_finite(high, "VQNet uniform initializer upper bound")?;
                if low >= high
                {
                    return Err(QuantumError::InvalidParameterMapping {
                        reason: "VQNet uniform initializer requires low < high",
                    });
                }

                let width = high - low;
                if !width.is_finite()
                {
                    return Err(QuantumError::NonFiniteParameter {
                        what: "VQNet uniform initializer width",
                    });
                }

                let mut state = seed;
                let values = (0..count)
                    .map(|_| {
                        let unit = splitmix_unit_f32(&mut state);
                        low + width * unit
                    })
                    .collect();
                Ok(values)
            },
        }
    }
}

/// Persistent trainable values for one [`VariationalCircuit`].
#[derive(Debug, Clone, PartialEq)]
pub struct VariationalParameters {
    values: Vec<f32>,
}

impl VariationalParameters {
    /// Creates persistent values after checking the circuit's exact trainable
    /// parameter count and finite-value invariant.
    pub fn from_values(expected_count: usize, values: Vec<f32>) -> QuantumResult<Self> {
        if values.len() != expected_count
        {
            return Err(QuantumError::InvalidTensorShape {
                tensor: "vqnet_parameters",
                expected_rows: Some(1),
                expected_cols: Some(expected_count),
                actual_rows: 1,
                actual_cols: values.len(),
            });
        }
        validate_values(&values)?;
        Ok(Self { values })
    }

    /// Creates persistent values from a deterministic initializer.
    pub fn initialized(count: usize, initializer: ParameterInitializer) -> QuantumResult<Self> {
        Self::from_values(count, initializer.initialize(count)?)
    }

    /// Trainable parameter values in circuit allocation order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of trainable quantum parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this parameter set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Creates the shared `1 × parameter_count` tape input expected by
    /// [`VariationalCircuit::forward_batch`].
    pub fn attach<'t>(&self, tape: &'t Tape) -> Var<'t> {
        tape.input(self.as_tensor())
    }

    /// Copies an optimizer-updated tape variable back into persistent state.
    ///
    /// The variable must retain the exact `1 × parameter_count` shape and all
    /// values must remain finite. The owned state is unchanged on validation
    /// failure.
    pub fn sync_from(&mut self, parameters: Var<'_>) -> QuantumResult<()> {
        self.sync_tensor(parameters.tape.value(parameters.idx()))
    }

    pub(super) fn as_tensor(&self) -> Tensor {
        Tensor::from_vec(self.values.clone(), 1, self.values.len())
    }

    pub(super) fn sync_tensor(&mut self, tensor: Tensor) -> QuantumResult<()> {
        let expected_count = self.values.len();
        if tensor.shape() != (1, expected_count)
        {
            return Err(QuantumError::InvalidTensorShape {
                tensor: "vqnet_parameters",
                expected_rows: Some(1),
                expected_cols: Some(expected_count),
                actual_rows: tensor.rows,
                actual_cols: tensor.cols,
            });
        }
        validate_values(&tensor.data)?;
        self.values = tensor.data;
        Ok(())
    }
}

/// One forward-pass handle exposing both model output and the tape variable that
/// an existing SciRust optimizer should update.
#[derive(Debug, Clone, Copy)]
pub struct QuantumForward<'t> {
    output: Var<'t>,
    parameters: Var<'t>,
}

impl<'t> QuantumForward<'t> {
    /// Model output variable.
    #[must_use]
    pub const fn output(&self) -> Var<'t> {
        self.output
    }

    /// Shared trainable quantum-parameter variable for this tape.
    #[must_use]
    pub const fn parameters(&self) -> Var<'t> {
        self.parameters
    }

    /// Tape node index to pass to a 2-D tape optimizer.
    #[must_use]
    pub fn parameter_index(&self) -> usize {
        self.parameters.idx()
    }
}

/// High-level variational circuit together with persistent trainable values.
///
/// The module does not duplicate optimizer state. Existing SciRust tape
/// optimizers can update [`QuantumForward::parameters`], while the
/// `PersistentParameterOptimizer` path can use a stable optimizer key across
/// fresh tapes. Both paths synchronize validated values back into this module.
#[derive(Debug)]
pub struct QuantumModule {
    circuit: VariationalCircuit,
    parameters: VariationalParameters,
    last_parameter_index: Option<usize>,
}

impl Clone for QuantumModule {
    fn clone(&self) -> Self {
        Self {
            circuit: self.circuit.clone(),
            parameters: self.parameters.clone(),
            last_parameter_index: None,
        }
    }
}

impl PartialEq for QuantumModule {
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit && self.parameters == other.parameters
    }
}

impl QuantumModule {
    /// Constructs a module and deterministically initializes all trainable
    /// quantum parameters.
    pub fn new(
        circuit: VariationalCircuit,
        initializer: ParameterInitializer,
    ) -> QuantumResult<Self> {
        let parameters =
            VariationalParameters::initialized(circuit.trainable_parameter_count(), initializer)?;
        Ok(Self {
            circuit,
            parameters,
            last_parameter_index: None,
        })
    }

    /// Constructs a module from explicit trainable values in the circuit's
    /// deterministic trainable-parameter order.
    pub fn from_values(circuit: VariationalCircuit, values: Vec<f32>) -> QuantumResult<Self> {
        let parameters =
            VariationalParameters::from_values(circuit.trainable_parameter_count(), values)?;
        Ok(Self {
            circuit,
            parameters,
            last_parameter_index: None,
        })
    }

    /// Underlying differentiable variational circuit.
    #[must_use]
    pub const fn circuit(&self) -> &VariationalCircuit {
        &self.circuit
    }

    /// Persistent trainable state.
    #[must_use]
    pub const fn parameters(&self) -> &VariationalParameters {
        &self.parameters
    }

    /// Runs one single-sample, single-observable forward pass using persistent
    /// module parameters attached to `tape`.
    pub fn forward<'t>(
        &self,
        tape: &'t Tape,
        classical_features: Var<'t>,
    ) -> QuantumResult<QuantumForward<'t>> {
        let parameters = self.parameters.attach(tape);
        let output = self.circuit.forward(classical_features, parameters)?;
        Ok(QuantumForward { output, parameters })
    }

    /// Runs one batched, ordered multi-observable forward pass using persistent
    /// module parameters attached to `tape`.
    pub fn forward_batch<'t>(
        &self,
        tape: &'t Tape,
        classical_features: Var<'t>,
    ) -> QuantumResult<QuantumForward<'t>> {
        let parameters = self.parameters.attach(tape);
        let output = self.circuit.forward_batch(classical_features, parameters)?;
        Ok(QuantumForward { output, parameters })
    }

    /// Persists an optimizer-updated parameter variable for the next fresh tape.
    pub fn sync_parameters(&mut self, parameters: Var<'_>) -> QuantumResult<()> {
        self.parameters.sync_from(parameters)
    }

    pub(super) fn record_parameter_index(&mut self, index: usize) {
        self.last_parameter_index = Some(index);
    }

    pub(super) const fn last_parameter_index(&self) -> Option<usize> {
        self.last_parameter_index
    }

    pub(super) fn sync_last_from_tape(&mut self, tape: &Tape) -> QuantumResult<()> {
        if let Some(index) = self.last_parameter_index
        {
            self.parameters.sync_tensor(tape.value(index))?;
        }
        Ok(())
    }

    pub(super) fn replace_parameter_tensor(&mut self, tensor: Tensor) -> QuantumResult<()> {
        self.parameters.sync_tensor(tensor)?;
        self.last_parameter_index = None;
        Ok(())
    }

    pub(super) fn parameter_tensor(&self) -> Tensor {
        self.parameters.as_tensor()
    }
}

fn validate_values(values: &[f32]) -> QuantumResult<()> {
    if values.iter().all(|value| value.is_finite())
    {
        Ok(())
    }
    else
    {
        Err(QuantumError::NonFiniteParameter {
            what: "VQNet trainable parameter",
        })
    }
}

fn ensure_finite(value: f32, what: &'static str) -> QuantumResult<()> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(QuantumError::NonFiniteParameter { what })
    }
}

fn splitmix_unit_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;

    // Populate the 23 mantissa bits of a number in [1, 2), then subtract one.
    // This avoids integer-to-float conversion dependence in the random source.
    let mantissa = (value >> 41) as u32;
    f32::from_bits(0x3f80_0000 | mantissa) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::optim::{Optimizer, Sgd};
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::vqnet::{RotationAxis, VariationalCircuitBuilder};

    fn one_qubit_circuit() -> VariationalCircuit {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        builder.angle_encoding(RotationAxis::Y, &[0]).unwrap();
        builder.hardware_efficient_ansatz(1).unwrap();
        builder.measure_all_z().unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn seeded_uniform_initialization_is_exactly_repeatable() {
        let initializer = ParameterInitializer::Uniform {
            seed: 0x51a7_cafe_1234_5678,
            low: -0.25,
            high: 0.25,
        };
        let first = VariationalParameters::initialized(16, initializer).unwrap();
        let second = VariationalParameters::initialized(16, initializer).unwrap();
        let different = VariationalParameters::initialized(
            16,
            ParameterInitializer::Uniform {
                seed: 0x51a7_cafe_1234_5679,
                low: -0.25,
                high: 0.25,
            },
        )
        .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, different);
        assert!(
            first
                .values()
                .iter()
                .all(|value| (-0.25..0.25).contains(value))
        );
    }

    #[test]
    fn explicit_values_must_match_exact_trainable_count() {
        let circuit = one_qubit_circuit();
        assert_eq!(
            QuantumModule::from_values(circuit, vec![0.1]).unwrap_err(),
            QuantumError::InvalidTensorShape {
                tensor: "vqnet_parameters",
                expected_rows: Some(1),
                expected_cols: Some(2),
                actual_rows: 1,
                actual_cols: 1,
            }
        );
    }

    #[test]
    fn invalid_initializers_are_rejected() {
        let circuit = one_qubit_circuit();
        assert_eq!(
            QuantumModule::new(circuit.clone(), ParameterInitializer::Constant(f32::NAN))
                .unwrap_err(),
            QuantumError::NonFiniteParameter {
                what: "VQNet constant initializer",
            }
        );
        assert_eq!(
            QuantumModule::new(
                circuit,
                ParameterInitializer::Uniform {
                    seed: 7,
                    low: 1.0,
                    high: 1.0,
                },
            )
            .unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "VQNet uniform initializer requires low < high",
            }
        );
    }

    #[test]
    fn module_parameters_survive_a_fresh_tape_optimizer_step() {
        let circuit = one_qubit_circuit();
        let mut module = QuantumModule::new(circuit, ParameterInitializer::Constant(0.3)).unwrap();
        let initial = module.parameters().values().to_vec();

        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(vec![0.2, -0.4], 2, 1));
        let pass = module.forward_batch(&tape, features).unwrap();
        pass.output().sum().backward();

        let mut optimizer = Sgd::new(0.05);
        optimizer.step(&[pass.parameter_index()], &tape);
        let updated_on_tape = tape.value(pass.parameter_index()).data;
        module.sync_parameters(pass.parameters()).unwrap();

        assert_eq!(module.parameters().values(), updated_on_tape.as_slice());
        assert_ne!(module.parameters().values()[0], initial[0]);
        assert!((module.parameters().values()[1] - initial[1]).abs() <= 2.0e-6);
    }

    #[test]
    fn forward_rejects_features_from_a_different_tape() {
        let module = QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Zeros).unwrap();
        let feature_tape = Tape::new();
        let module_tape = Tape::new();
        let features = feature_tape.input(Tensor::from_vec(vec![0.2], 1, 1));

        assert_eq!(
            module.forward_batch(&module_tape, features).unwrap_err(),
            QuantumError::MismatchedAutodiffTapes
        );
    }

    #[test]
    fn sync_rejects_non_finite_optimizer_output_without_mutation() {
        let mut parameters = VariationalParameters::from_values(2, vec![0.1, 0.2]).unwrap();
        let tape = Tape::new();
        let invalid = tape.input(Tensor::from_vec(vec![0.3, f32::INFINITY], 1, 2));

        assert_eq!(
            parameters.sync_from(invalid).unwrap_err(),
            QuantumError::NonFiniteParameter {
                what: "VQNet trainable parameter",
            }
        );
        assert_eq!(parameters.values(), &[0.1, 0.2]);
    }

    #[test]
    fn clone_resets_ephemeral_tape_identity() {
        let mut module =
            QuantumModule::new(one_qubit_circuit(), ParameterInitializer::Zeros).unwrap();
        module.record_parameter_index(17);
        let cloned = module.clone();

        assert_eq!(module, cloned);
        assert_eq!(cloned.last_parameter_index(), None);
    }
}
