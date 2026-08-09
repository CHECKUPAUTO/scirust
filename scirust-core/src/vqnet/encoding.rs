//! Reusable classical-feature encodings for variational circuits.

use super::{RotationAxis, VariationalCircuitBuilder, rotation_operation, validate_qubit};
use crate::quantum::{Parameter, ParameterId, QuantumError, QuantumResult};

/// Stable mapping from classical feature columns to encoded qubits.
///
/// A handle is created by [`VariationalCircuitBuilder::angle_encoding_with_handle`]
/// and can be re-applied later with [`VariationalCircuitBuilder::reupload_angle_encoding`].
/// Re-uploading reuses the exact same symbolic [`ParameterId`] values, so one
/// classical input feature can occur in multiple rotation gates while remaining
/// one input column in the `QuantumLayer` tensor contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AngleEncodingHandle {
    qubits: Vec<usize>,
    parameters: Vec<ParameterId>,
}

impl AngleEncodingHandle {
    /// Encoded qubits in classical feature-column order.
    #[must_use]
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// Reused symbolic input parameter IDs in feature-column order.
    #[must_use]
    pub fn parameters(&self) -> &[ParameterId] {
        &self.parameters
    }

    /// Number of encoded classical feature columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Whether the handle contains no feature mapping.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }
}

impl VariationalCircuitBuilder {
    /// Appends angle encoding and returns a stable handle for later data
    /// re-uploading.
    ///
    /// The first encoding has exactly the same gate and parameter allocation
    /// semantics as [`Self::angle_encoding`]. An empty qubit list is rejected
    /// because an empty handle cannot represent a reusable feature map.
    pub fn angle_encoding_with_handle(
        &mut self,
        axis: RotationAxis,
        qubits: &[usize],
    ) -> QuantumResult<AngleEncodingHandle> {
        if qubits.is_empty()
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "angle encoding handle needs at least one qubit",
            });
        }

        let first_parameter = self.input_parameters.len();
        self.angle_encoding(axis, qubits)?;
        let parameters = self.input_parameters[first_parameter..].to_vec();

        Ok(AngleEncodingHandle {
            qubits: qubits.to_vec(),
            parameters,
        })
    }

    /// Re-applies one prior angle encoding without allocating new input columns.
    ///
    /// Every handle parameter must already belong to this builder's input
    /// mapping. Validation is completed before the circuit is mutated. The
    /// reused symbols can occur arbitrarily many times in `Rx`, `Ry`, or `Rz`;
    /// SciRust's existing adjoint differentiation accumulates their derivatives
    /// into the original feature column.
    pub fn reupload_angle_encoding(
        &mut self,
        handle: &AngleEncodingHandle,
        axis: RotationAxis,
    ) -> QuantumResult<&mut Self> {
        if handle.qubits.len() != handle.parameters.len() || handle.is_empty()
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "invalid VQNet angle encoding handle",
            });
        }

        for (&qubit, &parameter) in handle.qubits.iter().zip(&handle.parameters)
        {
            validate_qubit(qubit, self.num_qubits())?;
            if !self.input_parameters.contains(&parameter)
            {
                return Err(QuantumError::InvalidParameterMapping {
                    reason: "angle encoding handle is not mapped by this builder",
                });
            }
        }

        for (&qubit, &parameter) in handle.qubits.iter().zip(&handle.parameters)
        {
            self.circuit.push(rotation_operation(
                axis,
                qubit,
                Parameter::Symbol(parameter),
            ))?;
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};
    use crate::quantum::Operation;
    use crate::vqnet::{EntanglementTopology, EntanglingGate};

    const TOLERANCE: f32 = 3.0e-5;

    #[test]
    fn reupload_reuses_exact_input_parameter_ids() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        let handle = builder
            .angle_encoding_with_handle(RotationAxis::Y, &[0, 1])
            .unwrap();
        builder
            .reupload_angle_encoding(&handle, RotationAxis::X)
            .unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        assert_eq!(handle.parameters(), &[ParameterId(0), ParameterId(1)]);
        assert_eq!(model.input_parameter_count(), 2);
        assert_eq!(model.input_parameters(), handle.parameters());
        assert!(matches!(
            model.circuit().operations()[2],
            Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(ParameterId(0)),
            }
        ));
        assert!(matches!(
            model.circuit().operations()[3],
            Operation::Rx {
                target: 1,
                parameter: Parameter::Symbol(ParameterId(1)),
            }
        ));
    }

    #[test]
    fn repeated_feature_occurrences_accumulate_exact_adjoint_gradient() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        let handle = builder
            .angle_encoding_with_handle(RotationAxis::Y, &[0])
            .unwrap();
        builder
            .reupload_angle_encoding(&handle, RotationAxis::Y)
            .unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        assert_eq!(model.input_parameter_count(), 1);
        assert_eq!(model.trainable_parameter_count(), 1);

        let features_data = [0.2f32, -0.4];
        let theta = 0.3f32;
        let tape = Tape::new();
        let features = tape.input(Tensor::from_vec(features_data.to_vec(), 2, 1));
        let parameters = tape.input(Tensor::from_vec(vec![theta], 1, 1));
        let output = model.forward_batch(features, parameters).unwrap();
        let output_data = tape.value(output.idx()).data;

        for (sample, feature) in features_data.into_iter().enumerate()
        {
            let phase = 2.0 * feature + theta;
            assert!((output_data[sample] - phase.cos()).abs() <= TOLERANCE);
        }

        output.sum().backward();
        let feature_gradient = tape.grad(features.idx()).data;
        let parameter_gradient = tape.grad(parameters.idx()).data;
        let mut expected_parameter_gradient = 0.0f32;

        for (sample, feature) in features_data.into_iter().enumerate()
        {
            let phase = 2.0 * feature + theta;
            let expected_feature_gradient = -2.0 * phase.sin();
            expected_parameter_gradient -= phase.sin();
            assert!((feature_gradient[sample] - expected_feature_gradient).abs() <= TOLERANCE);
        }
        assert!((parameter_gradient[0] - expected_parameter_gradient).abs() <= TOLERANCE);
    }

    #[test]
    fn incompatible_handle_is_rejected_before_mutation() {
        let mut source = VariationalCircuitBuilder::new(1).unwrap();
        let handle = source
            .angle_encoding_with_handle(RotationAxis::Y, &[0])
            .unwrap();

        let mut target = VariationalCircuitBuilder::new(1).unwrap();
        assert_eq!(
            target
                .reupload_angle_encoding(&handle, RotationAxis::Z)
                .unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "angle encoding handle is not mapped by this builder",
            }
        );
        assert!(target.circuit.operations().is_empty());
        assert!(target.input_parameters.is_empty());
    }

    #[test]
    fn empty_handle_creation_is_rejected_before_mutation() {
        let mut builder = VariationalCircuitBuilder::new(1).unwrap();
        assert_eq!(
            builder
                .angle_encoding_with_handle(RotationAxis::Y, &[])
                .unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "angle encoding handle needs at least one qubit",
            }
        );
        assert!(builder.circuit.operations().is_empty());
        assert!(builder.input_parameters.is_empty());
    }
}
