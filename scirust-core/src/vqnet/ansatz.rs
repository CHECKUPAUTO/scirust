//! Reusable variational ansatz templates for the VQNet-like facade.

use super::{RotationAxis, VariationalCircuitBuilder, rotation_operation};
use crate::quantum::{Operation, Parameter, QuantumError, QuantumResult};

/// Deterministic two-qubit gate used by a variational entanglement layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglingGate {
    /// Controlled-X / CNOT.
    Cnot,
    /// Controlled-Z.
    Cz,
}

/// Deterministic connectivity pattern for one variational entanglement layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglementTopology {
    /// No two-qubit entangling gates.
    None,
    /// Nearest-neighbour chain `0→1→...→n-1`.
    Linear,
    /// Linear chain plus `n-1→0` when at least three qubits are present.
    ///
    /// For two qubits this intentionally equals [`Self::Linear`] so a symmetric
    /// CZ layer is not applied twice and CNOT does not gain a surprising reverse
    /// edge solely because the circuit has size two.
    Ring,
    /// Every unordered qubit pair exactly once, ordered lexicographically as
    /// `(0,1), (0,2), ..., (n-2,n-1)`. The lower qubit is the control for CNOT.
    Full,
}

impl VariationalCircuitBuilder {
    /// Appends a configurable deterministic variational ansatz.
    ///
    /// For each layer, symbolic rotations are allocated in exact
    /// `rotations`-slice order and then ascending qubit order. Entanglers are
    /// appended afterward in the deterministic order documented by
    /// [`EntanglementTopology`]. At least one layer and one rotation axis are
    /// required.
    pub fn variational_ansatz(
        &mut self,
        layers: usize,
        rotations: &[RotationAxis],
        topology: EntanglementTopology,
        entangler: EntanglingGate,
    ) -> QuantumResult<&mut Self> {
        if layers == 0
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "variational ansatz needs at least one layer",
            });
        }
        if rotations.is_empty()
        {
            return Err(QuantumError::InvalidParameterMapping {
                reason: "variational ansatz needs at least one rotation axis",
            });
        }

        for _ in 0..layers
        {
            for &axis in rotations
            {
                for qubit in 0..self.num_qubits()
                {
                    let parameter = self.allocate_trainable_parameter()?;
                    self.circuit.push(rotation_operation(
                        axis,
                        qubit,
                        Parameter::Symbol(parameter),
                    ))?;
                }
            }
            self.append_entanglement(topology, entangler)?;
        }

        Ok(self)
    }

    fn append_entanglement(
        &mut self,
        topology: EntanglementTopology,
        entangler: EntanglingGate,
    ) -> QuantumResult<()> {
        match topology
        {
            EntanglementTopology::None =>
            {},
            EntanglementTopology::Linear | EntanglementTopology::Ring =>
            {
                for control in 0..self.num_qubits().saturating_sub(1)
                {
                    self.push_entangler(entangler, control, control + 1)?;
                }
                if topology == EntanglementTopology::Ring && self.num_qubits() > 2
                {
                    self.push_entangler(entangler, self.num_qubits() - 1, 0)?;
                }
            },
            EntanglementTopology::Full =>
            {
                for control in 0..self.num_qubits()
                {
                    for target in (control + 1)..self.num_qubits()
                    {
                        self.push_entangler(entangler, control, target)?;
                    }
                }
            },
        }
        Ok(())
    }

    fn push_entangler(
        &mut self,
        entangler: EntanglingGate,
        control: usize,
        target: usize,
    ) -> QuantumResult<()> {
        let operation = match entangler
        {
            EntanglingGate::Cnot => Operation::Cnot { control, target },
            EntanglingGate::Cz => Operation::Cz { control, target },
        };
        self.circuit.push(operation)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::{Observable, ParameterId};

    #[test]
    fn configurable_linear_ansatz_matches_legacy_hardware_efficient_layout() {
        let mut legacy = VariationalCircuitBuilder::new(3).unwrap();
        legacy.hardware_efficient_ansatz(2).unwrap();
        legacy.measure(Observable::z(0)).unwrap();
        let legacy = legacy.build().unwrap();

        let mut configurable = VariationalCircuitBuilder::new(3).unwrap();
        configurable
            .variational_ansatz(
                2,
                &[RotationAxis::Y, RotationAxis::Z],
                EntanglementTopology::Linear,
                EntanglingGate::Cnot,
            )
            .unwrap();
        configurable.measure(Observable::z(0)).unwrap();
        let configurable = configurable.build().unwrap();

        assert_eq!(legacy.circuit(), configurable.circuit());
        assert_eq!(
            configurable.trainable_parameters(),
            &[
                ParameterId(0),
                ParameterId(1),
                ParameterId(2),
                ParameterId(3),
                ParameterId(4),
                ParameterId(5),
                ParameterId(6),
                ParameterId(7),
                ParameterId(8),
                ParameterId(9),
                ParameterId(10),
                ParameterId(11),
            ]
        );
    }

    #[test]
    fn ring_has_one_closing_edge_only_when_it_is_distinct() {
        let mut two = VariationalCircuitBuilder::new(2).unwrap();
        two.variational_ansatz(
            1,
            &[RotationAxis::X],
            EntanglementTopology::Ring,
            EntanglingGate::Cz,
        )
        .unwrap();
        two.measure_all_z().unwrap();
        let two = two.build().unwrap();
        assert_eq!(two.circuit().operations().len(), 3);
        assert!(matches!(
            two.circuit().operations()[2],
            Operation::Cz {
                control: 0,
                target: 1
            }
        ));

        let mut three = VariationalCircuitBuilder::new(3).unwrap();
        three
            .variational_ansatz(
                1,
                &[RotationAxis::X],
                EntanglementTopology::Ring,
                EntanglingGate::Cz,
            )
            .unwrap();
        three.measure_all_z().unwrap();
        let three = three.build().unwrap();
        assert_eq!(three.circuit().operations().len(), 6);
        assert!(matches!(
            three.circuit().operations()[5],
            Operation::Cz {
                control: 2,
                target: 0
            }
        ));
    }

    #[test]
    fn full_topology_emits_each_pair_once_in_lexicographic_order() {
        let mut builder = VariationalCircuitBuilder::new(4).unwrap();
        builder
            .variational_ansatz(
                1,
                &[RotationAxis::Y],
                EntanglementTopology::Full,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        let expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (operation, &(control, target)) in
            model.circuit().operations()[4..].iter().zip(&expected)
        {
            assert!(matches!(
                operation,
                Operation::Cnot {
                    control: actual_control,
                    target: actual_target,
                } if (*actual_control, *actual_target) == (control, target)
            ));
        }
    }

    #[test]
    fn no_entanglement_keeps_only_requested_rotations() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        builder
            .variational_ansatz(
                2,
                &[RotationAxis::X, RotationAxis::Z],
                EntanglementTopology::None,
                EntanglingGate::Cnot,
            )
            .unwrap();
        builder.measure_all_z().unwrap();
        let model = builder.build().unwrap();

        assert_eq!(model.circuit().operations().len(), 8);
        assert_eq!(model.trainable_parameter_count(), 8);
    }

    #[test]
    fn configurable_ansatz_rejects_empty_structure_before_mutation() {
        let mut builder = VariationalCircuitBuilder::new(2).unwrap();
        assert_eq!(
            builder
                .variational_ansatz(
                    0,
                    &[RotationAxis::Y],
                    EntanglementTopology::Linear,
                    EntanglingGate::Cnot,
                )
                .unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "variational ansatz needs at least one layer",
            }
        );
        assert!(builder.circuit.operations().is_empty());

        assert_eq!(
            builder
                .variational_ansatz(1, &[], EntanglementTopology::Linear, EntanglingGate::Cnot,)
                .unwrap_err(),
            QuantumError::InvalidParameterMapping {
                reason: "variational ansatz needs at least one rotation axis",
            }
        );
        assert!(builder.circuit.operations().is_empty());
    }
}
