//! Typed circuit structure, symbolic parameters, and deterministic binding.

use super::dense::DenseStateVector;
use super::error::{QuantumError, QuantumResult};
use std::collections::{BTreeMap, BTreeSet};

/// Stable identifier for a symbolic circuit parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(pub u32);

/// A fixed angle or a symbolic angle supplied at execution time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Parameter {
    /// Immutable angle in radians.
    Fixed(f32),
    /// Symbolic angle in radians.
    Symbol(ParameterId),
}

/// Deterministically ordered symbolic parameter bindings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParameterValues {
    values: BTreeMap<ParameterId, f32>,
}

impl ParameterValues {
    /// Creates an empty binding set.
    pub const fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Inserts a finite angle in radians, returning the previous value if any.
    pub fn insert(&mut self, id: ParameterId, value: f32) -> QuantumResult<Option<f32>> {
        if !value.is_finite()
        {
            return Err(QuantumError::NonFiniteParameter {
                what: "circuit parameter",
            });
        }
        Ok(self.values.insert(id, value))
    }

    /// Builder-style finite binding insertion.
    pub fn with(mut self, id: ParameterId, value: f32) -> QuantumResult<Self> {
        self.insert(id, value)?;
        Ok(self)
    }

    /// Returns a bound value.
    pub fn get(&self, id: ParameterId) -> Option<f32> {
        self.values.get(&id).copied()
    }

    /// Iterates in ascending parameter-ID order.
    pub fn iter(&self) -> impl Iterator<Item = (ParameterId, f32)> + '_ {
        self.values.iter().map(|(&id, &value)| (id, value))
    }
}

/// A typed quantum gate. Operation order is vector order in [`Circuit`].
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    I { target: usize },
    H { target: usize },
    X { target: usize },
    Y { target: usize },
    Z { target: usize },
    S { target: usize },
    Sdg { target: usize },
    T { target: usize },
    Tdg { target: usize },
    Rx { target: usize, parameter: Parameter },
    Ry { target: usize, parameter: Parameter },
    Rz { target: usize, parameter: Parameter },
    PhaseShift { target: usize, parameter: Parameter },
    Cnot { control: usize, target: usize },
    Cz { control: usize, target: usize },
    Swap { first: usize, second: usize },
}

/// Circuit structure independent of a simulator and parameter values.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    num_qubits: usize,
    operations: Vec<Operation>,
}

impl Circuit {
    /// Creates an empty circuit for at least one qubit.
    pub fn new(num_qubits: usize) -> QuantumResult<Self> {
        if num_qubits == 0
        {
            return Err(QuantumError::StateDimensionOverflow { num_qubits });
        }
        Ok(Self {
            num_qubits,
            operations: Vec::new(),
        })
    }

    /// Number of qubits.
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Gates in execution order.
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Validates and appends one operation.
    pub fn push(&mut self, operation: Operation) -> QuantumResult<&mut Self> {
        validate_operation(self.num_qubits, &operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// IDs used by symbolic operations in ascending order.
    pub fn parameter_ids(&self) -> Vec<ParameterId> {
        let mut ids = BTreeSet::new();
        for operation in &self.operations
        {
            if let Some(Parameter::Symbol(id)) = operation_parameter(operation)
            {
                ids.insert(id);
            }
        }
        ids.into_iter().collect()
    }

    /// Resolves every symbolic parameter and rejects missing or extraneous IDs.
    pub fn bind(&self, values: &ParameterValues) -> QuantumResult<BoundCircuit> {
        let used: BTreeSet<_> = self.parameter_ids().into_iter().collect();
        for (id, _) in values.iter()
        {
            if !used.contains(&id)
            {
                return Err(QuantumError::UnknownParameter { parameter: id.0 });
            }
        }
        let operations = self
            .operations
            .iter()
            .map(|operation| bind_operation(operation, values))
            .collect::<QuantumResult<Vec<_>>>()?;
        Ok(BoundCircuit {
            num_qubits: self.num_qubits,
            operations,
        })
    }

    pub(crate) fn parameter_occurrences(
        &self,
        parameter: ParameterId,
    ) -> QuantumResult<Vec<usize>> {
        let mut occurrences = Vec::new();
        for (index, operation) in self.operations.iter().enumerate()
        {
            if operation_parameter(operation) == Some(Parameter::Symbol(parameter))
            {
                match operation
                {
                    Operation::Rx { .. } | Operation::Ry { .. } | Operation::Rz { .. } =>
                    {
                        occurrences.push(index);
                    },
                    _ =>
                    {
                        return Err(QuantumError::UnsupportedGradientRule {
                            parameter: parameter.0,
                        });
                    },
                }
            }
        }
        if occurrences.is_empty()
        {
            return Err(QuantumError::UnknownParameter {
                parameter: parameter.0,
            });
        }
        Ok(occurrences)
    }

    pub(crate) fn bind_with_occurrence_shift(
        &self,
        values: &ParameterValues,
        operation_index: usize,
        shift: f32,
    ) -> QuantumResult<BoundCircuit> {
        let mut bound = self.bind(values)?;
        let operation =
            bound
                .operations
                .get_mut(operation_index)
                .ok_or(QuantumError::NumericalFailure {
                    operation: "parameter occurrence lookup",
                })?;
        match operation
        {
            BoundOperation::Rx { theta, .. }
            | BoundOperation::Ry { theta, .. }
            | BoundOperation::Rz { theta, .. } =>
            {
                *theta += shift;
                if !theta.is_finite()
                {
                    return Err(QuantumError::NonFiniteParameter {
                        what: "shifted circuit parameter",
                    });
                }
            },
            _ =>
            {
                return Err(QuantumError::NumericalFailure {
                    operation: "shifted operation is not a rotation",
                });
            },
        }
        Ok(bound)
    }
}

/// An operation after every angle has been resolved to a finite `f32` value.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundOperation {
    I { target: usize },
    H { target: usize },
    X { target: usize },
    Y { target: usize },
    Z { target: usize },
    S { target: usize },
    Sdg { target: usize },
    T { target: usize },
    Tdg { target: usize },
    Rx { target: usize, theta: f32 },
    Ry { target: usize, theta: f32 },
    Rz { target: usize, theta: f32 },
    PhaseShift { target: usize, theta: f32 },
    Cnot { control: usize, target: usize },
    Cz { control: usize, target: usize },
    Swap { first: usize, second: usize },
}

/// A fully bound executable circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCircuit {
    num_qubits: usize,
    operations: Vec<BoundOperation>,
}

impl BoundCircuit {
    /// Number of qubits.
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Bound gates in execution order.
    pub fn operations(&self) -> &[BoundOperation] {
        &self.operations
    }

    /// Executes from `|00...0>` on the dense CPU oracle.
    pub fn execute_dense(&self) -> QuantumResult<DenseStateVector> {
        let mut state = DenseStateVector::zero(self.num_qubits)?;
        for operation in &self.operations
        {
            match *operation
            {
                BoundOperation::I { target } => state.i(target)?,
                BoundOperation::H { target } => state.h(target)?,
                BoundOperation::X { target } => state.x(target)?,
                BoundOperation::Y { target } => state.y(target)?,
                BoundOperation::Z { target } => state.z(target)?,
                BoundOperation::S { target } => state.s(target)?,
                BoundOperation::Sdg { target } => state.sdg(target)?,
                BoundOperation::T { target } => state.t(target)?,
                BoundOperation::Tdg { target } => state.tdg(target)?,
                BoundOperation::Rx { target, theta } => state.rx(target, theta)?,
                BoundOperation::Ry { target, theta } => state.ry(target, theta)?,
                BoundOperation::Rz { target, theta } => state.rz(target, theta)?,
                BoundOperation::PhaseShift { target, theta } => state.phase_shift(target, theta)?,
                BoundOperation::Cnot { control, target } => state.cnot(control, target)?,
                BoundOperation::Cz { control, target } => state.cz(control, target)?,
                BoundOperation::Swap { first, second } => state.swap(first, second)?,
            }
        }
        Ok(state)
    }
}

fn validate_operation(num_qubits: usize, operation: &Operation) -> QuantumResult<()> {
    let validate_qubit = |qubit| {
        if qubit < num_qubits
        {
            Ok(())
        }
        else
        {
            Err(QuantumError::InvalidQubitIndex { qubit, num_qubits })
        }
    };
    match operation
    {
        Operation::I { target }
        | Operation::H { target }
        | Operation::X { target }
        | Operation::Y { target }
        | Operation::Z { target }
        | Operation::S { target }
        | Operation::Sdg { target }
        | Operation::T { target }
        | Operation::Tdg { target } => validate_qubit(*target),
        Operation::Rx { target, parameter }
        | Operation::Ry { target, parameter }
        | Operation::Rz { target, parameter }
        | Operation::PhaseShift { target, parameter } =>
        {
            validate_qubit(*target)?;
            if let Parameter::Fixed(value) = parameter
                && !value.is_finite()
            {
                return Err(QuantumError::NonFiniteParameter {
                    what: "fixed circuit parameter",
                });
            }
            Ok(())
        },
        Operation::Cnot { control, target } | Operation::Cz { control, target } =>
        {
            validate_qubit(*control)?;
            validate_qubit(*target)?;
            if control == target
            {
                return Err(QuantumError::InvalidControlTarget {
                    control: *control,
                    target: *target,
                });
            }
            Ok(())
        },
        Operation::Swap { first, second } =>
        {
            validate_qubit(*first)?;
            validate_qubit(*second)?;
            if first == second
            {
                return Err(QuantumError::DuplicateQubit { qubit: *first });
            }
            Ok(())
        },
    }
}

fn operation_parameter(operation: &Operation) -> Option<Parameter> {
    match *operation
    {
        Operation::Rx { parameter, .. }
        | Operation::Ry { parameter, .. }
        | Operation::Rz { parameter, .. }
        | Operation::PhaseShift { parameter, .. } => Some(parameter),
        _ => None,
    }
}

fn resolve(parameter: Parameter, values: &ParameterValues) -> QuantumResult<f32> {
    let value = match parameter
    {
        Parameter::Fixed(value) => value,
        Parameter::Symbol(id) => values
            .get(id)
            .ok_or(QuantumError::UnboundParameter { parameter: id.0 })?,
    };
    if value.is_finite()
    {
        Ok(value)
    }
    else
    {
        Err(QuantumError::NonFiniteParameter {
            what: "bound circuit parameter",
        })
    }
}

fn bind_operation(
    operation: &Operation,
    values: &ParameterValues,
) -> QuantumResult<BoundOperation> {
    Ok(match *operation
    {
        Operation::I { target } => BoundOperation::I { target },
        Operation::H { target } => BoundOperation::H { target },
        Operation::X { target } => BoundOperation::X { target },
        Operation::Y { target } => BoundOperation::Y { target },
        Operation::Z { target } => BoundOperation::Z { target },
        Operation::S { target } => BoundOperation::S { target },
        Operation::Sdg { target } => BoundOperation::Sdg { target },
        Operation::T { target } => BoundOperation::T { target },
        Operation::Tdg { target } => BoundOperation::Tdg { target },
        Operation::Rx { target, parameter } => BoundOperation::Rx {
            target,
            theta: resolve(parameter, values)?,
        },
        Operation::Ry { target, parameter } => BoundOperation::Ry {
            target,
            theta: resolve(parameter, values)?,
        },
        Operation::Rz { target, parameter } => BoundOperation::Rz {
            target,
            theta: resolve(parameter, values)?,
        },
        Operation::PhaseShift { target, parameter } => BoundOperation::PhaseShift {
            target,
            theta: resolve(parameter, values)?,
        },
        Operation::Cnot { control, target } => BoundOperation::Cnot { control, target },
        Operation::Cz { control, target } => BoundOperation::Cz { control, target },
        Operation::Swap { first, second } => BoundOperation::Swap { first, second },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_order_and_binding_are_stable() {
        let theta = ParameterId(7);
        let mut circuit = Circuit::new(2).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::Ry {
                target: 1,
                parameter: Parameter::Symbol(theta),
            })
            .unwrap()
            .push(Operation::Cnot {
                control: 0,
                target: 1,
            })
            .unwrap();
        assert_eq!(circuit.parameter_ids(), vec![theta]);
        let values = ParameterValues::new().with(theta, 0.25).unwrap();
        let bound = circuit.bind(&values).unwrap();
        assert!(matches!(bound.operations()[0], BoundOperation::H { .. }));
        assert!(matches!(
            bound.operations()[1],
            BoundOperation::Ry { theta: 0.25, .. }
        ));
        assert!(matches!(bound.operations()[2], BoundOperation::Cnot { .. }));
    }

    #[test]
    fn invalid_and_unbound_parameters_are_structured_errors() {
        let theta = ParameterId(2);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(theta),
            })
            .unwrap();
        assert_eq!(
            circuit.bind(&ParameterValues::new()),
            Err(QuantumError::UnboundParameter { parameter: 2 })
        );
        assert!(ParameterValues::new().with(theta, f32::NAN).is_err());
    }
}
