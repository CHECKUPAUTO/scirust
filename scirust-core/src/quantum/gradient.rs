//! Verification-first gradients for exact expectation values.

use super::complex::Complex32;
use super::dense::DenseStateVector;
use super::error::{QuantumError, QuantumResult};
use super::ir::{BoundOperation, Circuit, Operation, Parameter, ParameterId, ParameterValues};
use super::observable::{Observable, Pauli};
use std::collections::BTreeMap;

/// Executes a bound circuit and evaluates one exact Pauli expectation.
pub fn expectation_value(
    circuit: &Circuit,
    values: &ParameterValues,
    observable: &Observable,
) -> QuantumResult<f32> {
    circuit
        .bind(values)?
        .execute_dense()?
        .expectation(observable)
}

/// Central finite-difference reference derivative.
///
/// This is a numerical oracle only. Cancellation dominates when `epsilon` is
/// too small for `f32`, while truncation dominates when it is too large.
pub fn finite_difference_gradient(
    circuit: &Circuit,
    values: &ParameterValues,
    observable: &Observable,
    parameter: ParameterId,
    epsilon: f32,
) -> QuantumResult<f32> {
    if !epsilon.is_finite() || epsilon <= 0.0
    {
        return Err(QuantumError::NonFiniteParameter {
            what: "finite-difference epsilon",
        });
    }
    let center = values
        .get(parameter)
        .ok_or(QuantumError::UnboundParameter {
            parameter: parameter.0,
        })?;
    let mut plus = values.clone();
    let mut minus = values.clone();
    plus.insert(parameter, center + epsilon)?;
    minus.insert(parameter, center - epsilon)?;
    let upper = expectation_value(circuit, &plus, observable)?;
    let lower = expectation_value(circuit, &minus, observable)?;
    Ok((upper - lower) / (2.0 * epsilon))
}

/// Exact parameter-shift derivative for `Rx`, `Ry`, and `Rz` occurrences with
/// generator `P/2`. If one symbolic parameter occurs in multiple gates, each
/// occurrence is shifted separately and the contributions are summed.
pub fn parameter_shift_gradient(
    circuit: &Circuit,
    values: &ParameterValues,
    observable: &Observable,
    parameter: ParameterId,
) -> QuantumResult<f32> {
    Ok(parameter_shift_gradients(
        circuit,
        values,
        core::slice::from_ref(observable),
        parameter,
    )?[0])
}

/// Exact parameter-shift derivatives for multiple observables.
///
/// For each occurrence of `parameter`, the positively and negatively shifted
/// circuits are each executed once. Every observable is then evaluated on
/// those two states in slice order, so adding observables does not multiply the
/// number of shifted circuit executions. Contributions from reused symbolic
/// parameters are accumulated in deterministic occurrence and observable
/// order.
pub fn parameter_shift_gradients(
    circuit: &Circuit,
    values: &ParameterValues,
    observables: &[Observable],
    parameter: ParameterId,
) -> QuantumResult<Vec<f32>> {
    if observables.is_empty()
    {
        return Err(QuantumError::InvalidObservableCount {
            minimum: 1,
            maximum: None,
            actual: 0,
        });
    }
    if values.get(parameter).is_none()
    {
        return Err(QuantumError::UnboundParameter {
            parameter: parameter.0,
        });
    }
    let occurrences = circuit.parameter_occurrences(parameter)?;
    let shift = core::f32::consts::FRAC_PI_2;
    let mut gradients = vec![0.0f32; observables.len()];
    for operation_index in occurrences
    {
        let upper_state = circuit
            .bind_with_occurrence_shift(values, operation_index, shift)?
            .execute_dense()?;
        let lower_state = circuit
            .bind_with_occurrence_shift(values, operation_index, -shift)?
            .execute_dense()?;
        for (observable_index, observable) in observables.iter().enumerate()
        {
            let upper = upper_state.expectation(observable)?;
            let lower = lower_state.expectation(observable)?;
            gradients[observable_index] += 0.5 * (upper - lower);
        }
    }
    if gradients.iter().all(|gradient| gradient.is_finite())
    {
        Ok(gradients)
    }
    else
    {
        Err(QuantumError::NumericalFailure {
            operation: "multi-observable parameter-shift gradient",
        })
    }
}

/// Exact dense adjoint Jacobian for all symbolic circuit parameters.
///
/// The returned [`BTreeMap`] is ordered by ascending [`ParameterId`]. Each
/// vector preserves `observables` order. Reused symbolic parameters accumulate
/// their gate-occurrence contributions in ascending circuit-operation order.
///
/// One dense forward execution constructs the final ket. The reverse sweep
/// propagates that ket and one adjoint state per observable through every
/// inverse gate. No shifted circuit is executed.
pub fn adjoint_jacobian(
    circuit: &Circuit,
    values: &ParameterValues,
    observables: &[Observable],
) -> QuantumResult<BTreeMap<ParameterId, Vec<f32>>> {
    if observables.is_empty()
    {
        return Err(QuantumError::InvalidObservableCount {
            minimum: 1,
            maximum: None,
            actual: 0,
        });
    }

    let parameter_ids = circuit.parameter_ids();
    for &parameter in &parameter_ids
    {
        // This validates every symbolic occurrence and rejects symbolic gates
        // without a supported exact adjoint rule, notably PhaseShift.
        circuit.parameter_occurrences(parameter)?;
    }

    let bound = circuit.bind(values)?;
    let final_state = bound.execute_dense()?;
    let mut ket = final_state.clone();

    let mut adjoints = Vec::with_capacity(observables.len());
    for observable in observables
    {
        let mut adjoint = final_state.clone();
        apply_observable(&mut adjoint, observable)?;
        adjoints.push(adjoint);
    }

    // The reverse sweep discovers occurrences in descending operation order.
    // Store contributions at their original indices, then sum them forward so
    // reused parameters retain the same deterministic occurrence ordering as
    // parameter shift.
    let mut occurrence_contributions: Vec<Option<(ParameterId, Vec<f32>)>> =
        vec![None; bound.operations().len()];

    for operation_index in (0..bound.operations().len()).rev()
    {
        let bound_operation = &bound.operations()[operation_index];
        let symbolic_generator = match &circuit.operations()[operation_index]
        {
            Operation::Rx {
                target,
                parameter: Parameter::Symbol(parameter),
            } => Some((*parameter, *target, Pauli::X)),
            Operation::Ry {
                target,
                parameter: Parameter::Symbol(parameter),
            } => Some((*parameter, *target, Pauli::Y)),
            Operation::Rz {
                target,
                parameter: Parameter::Symbol(parameter),
            } => Some((*parameter, *target, Pauli::Z)),
            Operation::PhaseShift {
                parameter: Parameter::Symbol(parameter),
                ..
            } =>
            {
                return Err(QuantumError::UnsupportedGradientRule {
                    parameter: parameter.0,
                });
            },
            _ => None,
        };

        if let Some((parameter, target, generator)) = symbolic_generator
        {
            let mut derivatives = Vec::with_capacity(observables.len());
            for adjoint in &adjoints
            {
                // For U(theta) = exp(-i theta P / 2):
                //
                //   d< O >/dtheta = Im <lambda | P | ket>,
                //
                // where ket is the state immediately after U and lambda is the
                // corresponding observable adjoint at the same circuit depth.
                derivatives.push(pauli_cross_inner_product(adjoint, &ket, target, generator)?.im);
            }
            occurrence_contributions[operation_index] = Some((parameter, derivatives));
        }

        for adjoint in &mut adjoints
        {
            apply_adjoint_operation(adjoint, bound_operation)?;
        }
        apply_adjoint_operation(&mut ket, bound_operation)?;
    }

    let mut jacobian = BTreeMap::new();
    for parameter in parameter_ids
    {
        jacobian.insert(parameter, vec![0.0f32; observables.len()]);
    }

    for (parameter, derivatives) in occurrence_contributions.into_iter().flatten()
    {
        let accumulated = jacobian
            .get_mut(&parameter)
            .ok_or(QuantumError::NumericalFailure {
                operation: "adjoint parameter occurrence accumulation",
            })?;
        for (total, derivative) in accumulated.iter_mut().zip(derivatives)
        {
            *total += derivative;
        }
    }

    if jacobian
        .values()
        .flat_map(|derivatives| derivatives.iter())
        .all(|derivative| derivative.is_finite())
    {
        Ok(jacobian)
    }
    else
    {
        Err(QuantumError::NumericalFailure {
            operation: "dense adjoint Jacobian",
        })
    }
}

/// Exact dense adjoint derivatives for one parameter and multiple observables.
pub fn adjoint_gradients(
    circuit: &Circuit,
    values: &ParameterValues,
    observables: &[Observable],
    parameter: ParameterId,
) -> QuantumResult<Vec<f32>> {
    adjoint_jacobian(circuit, values, observables)?
        .remove(&parameter)
        .ok_or(QuantumError::UnknownParameter {
            parameter: parameter.0,
        })
}

/// Exact dense adjoint derivative for one parameter and one observable.
pub fn adjoint_gradient(
    circuit: &Circuit,
    values: &ParameterValues,
    observable: &Observable,
    parameter: ParameterId,
) -> QuantumResult<f32> {
    Ok(adjoint_gradients(
        circuit,
        values,
        core::slice::from_ref(observable),
        parameter,
    )?[0])
}

fn apply_observable(state: &mut DenseStateVector, observable: &Observable) -> QuantumResult<()> {
    for term in observable.terms()
    {
        match term.pauli
        {
            Pauli::X => state.x(term.qubit)?,
            Pauli::Y => state.y(term.qubit)?,
            Pauli::Z => state.z(term.qubit)?,
        }
    }
    Ok(())
}

fn pauli_cross_inner_product(
    left: &DenseStateVector,
    right: &DenseStateVector,
    target: usize,
    pauli: Pauli,
) -> QuantumResult<Complex32> {
    if left.amplitudes().len() != right.amplitudes().len()
    {
        return Err(QuantumError::InvalidStateDimension {
            expected: left.amplitudes().len(),
            actual: right.amplitudes().len(),
        });
    }
    if target >= right.num_qubits()
    {
        return Err(QuantumError::InvalidQubitIndex {
            qubit: target,
            num_qubits: right.num_qubits(),
        });
    }

    let mask = 1usize << target;
    let mut value = Complex32::zero();

    for (input_index, &right_amplitude) in right.amplitudes().iter().enumerate()
    {
        let bit_is_one = input_index & mask != 0;
        let (output_index, coefficient) = match pauli
        {
            Pauli::X => (input_index ^ mask, Complex32::one()),
            Pauli::Y => (
                input_index ^ mask,
                if bit_is_one
                {
                    Complex32::new(0.0, -1.0)
                }
                else
                {
                    Complex32::new(0.0, 1.0)
                },
            ),
            Pauli::Z => (
                input_index,
                if bit_is_one
                {
                    Complex32::new(-1.0, 0.0)
                }
                else
                {
                    Complex32::one()
                },
            ),
        };

        value += left.amplitudes()[output_index].conj() * coefficient * right_amplitude;
    }

    if value.is_finite()
    {
        Ok(value)
    }
    else
    {
        Err(QuantumError::NumericalFailure {
            operation: "Pauli cross-state inner product",
        })
    }
}

#[cfg(test)]
fn apply_bound_operation(
    state: &mut DenseStateVector,
    operation: &BoundOperation,
) -> QuantumResult<()> {
    match *operation
    {
        BoundOperation::I { target } => state.i(target),
        BoundOperation::H { target } => state.h(target),
        BoundOperation::X { target } => state.x(target),
        BoundOperation::Y { target } => state.y(target),
        BoundOperation::Z { target } => state.z(target),
        BoundOperation::S { target } => state.s(target),
        BoundOperation::Sdg { target } => state.sdg(target),
        BoundOperation::T { target } => state.t(target),
        BoundOperation::Tdg { target } => state.tdg(target),
        BoundOperation::Rx { target, theta } => state.rx(target, theta),
        BoundOperation::Ry { target, theta } => state.ry(target, theta),
        BoundOperation::Rz { target, theta } => state.rz(target, theta),
        BoundOperation::PhaseShift { target, theta } => state.phase_shift(target, theta),
        BoundOperation::Cnot { control, target } => state.cnot(control, target),
        BoundOperation::Cz { control, target } => state.cz(control, target),
        BoundOperation::Swap { first, second } => state.swap(first, second),
    }
}

fn apply_adjoint_operation(
    state: &mut DenseStateVector,
    operation: &BoundOperation,
) -> QuantumResult<()> {
    match *operation
    {
        BoundOperation::I { target } => state.i(target),
        BoundOperation::H { target } => state.h(target),
        BoundOperation::X { target } => state.x(target),
        BoundOperation::Y { target } => state.y(target),
        BoundOperation::Z { target } => state.z(target),
        BoundOperation::S { target } => state.sdg(target),
        BoundOperation::Sdg { target } => state.s(target),
        BoundOperation::T { target } => state.tdg(target),
        BoundOperation::Tdg { target } => state.t(target),
        BoundOperation::Rx { target, theta } => state.rx(target, -theta),
        BoundOperation::Ry { target, theta } => state.ry(target, -theta),
        BoundOperation::Rz { target, theta } => state.rz(target, -theta),
        BoundOperation::PhaseShift { target, theta } => state.phase_shift(target, -theta),
        BoundOperation::Cnot { control, target } => state.cnot(control, target),
        BoundOperation::Cz { control, target } => state.cz(control, target),
        BoundOperation::Swap { first, second } => state.swap(first, second),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::ir::{Operation, Parameter};
    use crate::quantum::observable::{Pauli, PauliTerm};

    const ANALYTIC_TOLERANCE: f32 = 5.0e-5;
    const FINITE_DIFFERENCE_TOLERANCE: f32 = 8.0e-4;
    const EPSILON: f32 = 1.0e-3;

    fn assert_gradients(
        circuit: &Circuit,
        parameter: ParameterId,
        observable: &Observable,
        theta: f32,
        expected_value: f32,
        expected_gradient: f32,
    ) {
        let values = ParameterValues::new().with(parameter, theta).unwrap();
        let value = expectation_value(circuit, &values, observable).unwrap();
        let shifted = parameter_shift_gradient(circuit, &values, observable, parameter).unwrap();
        let finite =
            finite_difference_gradient(circuit, &values, observable, parameter, EPSILON).unwrap();
        assert!((value - expected_value).abs() <= ANALYTIC_TOLERANCE);
        assert!((shifted - expected_gradient).abs() <= ANALYTIC_TOLERANCE);
        assert!((finite - expected_gradient).abs() <= FINITE_DIFFERENCE_TOLERANCE);
        assert!((shifted - finite).abs() <= FINITE_DIFFERENCE_TOLERANCE);
    }

    #[test]
    fn multi_observable_shift_preserves_order_and_rejects_empty_input() {
        let parameter = ParameterId(12);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        let theta = 0.37f32;
        let values = ParameterValues::new().with(parameter, theta).unwrap();
        let observables = [Observable::z(0), Observable::x(0)];
        let gradients =
            parameter_shift_gradients(&circuit, &values, &observables, parameter).unwrap();
        assert_eq!(gradients.len(), 2);
        assert!((gradients[0] + theta.sin()).abs() <= ANALYTIC_TOLERANCE);
        assert!((gradients[1] - theta.cos()).abs() <= ANALYTIC_TOLERANCE);
        assert_eq!(
            parameter_shift_gradient(&circuit, &values, &observables[0], parameter).unwrap(),
            gradients[0]
        );
        assert_eq!(
            parameter_shift_gradients(&circuit, &values, &[], parameter),
            Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            })
        );
    }

    #[test]
    fn ry_z_matches_cosine_and_negative_sine_across_fixed_angles() {
        let parameter = ParameterId(0);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        let angles = [
            0.0,
            1.0e-3,
            core::f32::consts::FRAC_PI_4,
            core::f32::consts::FRAC_PI_2,
            -0.73,
            core::f32::consts::PI - 1.0e-3,
        ];
        for theta in angles
        {
            assert_gradients(
                &circuit,
                parameter,
                &Observable::z(0),
                theta,
                theta.cos(),
                -theta.sin(),
            );
        }
    }

    #[test]
    fn rx_y_after_h_has_nonzero_closed_form_gradient() {
        // H|0> = |+>; Rx only adds global phase, so use S first:
        // H,S prepares (|0>+i|1>)/sqrt(2), with <Z> after Rx(theta) = sin(theta).
        let parameter = ParameterId(1);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::S { target: 0 })
            .unwrap()
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        for theta in [0.0, 0.2, -0.8, core::f32::consts::FRAC_PI_2]
        {
            assert_gradients(
                &circuit,
                parameter,
                &Observable::z(0),
                theta,
                theta.sin(),
                theta.cos(),
            );
        }
    }

    #[test]
    fn entangled_circuit_gradients_match_reference_oracles() {
        // Bell state followed by Ry(theta) on q1 has <Z0 Z1> = cos(theta).
        let parameter = ParameterId(3);
        let mut circuit = Circuit::new(2).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::Cnot {
                control: 0,
                target: 1,
            })
            .unwrap()
            .push(Operation::Ry {
                target: 1,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        let observable = Observable::new(vec![
            PauliTerm::new(0, Pauli::Z),
            PauliTerm::new(1, Pauli::Z),
        ])
        .unwrap();
        for theta in [0.0, 0.11, core::f32::consts::FRAC_PI_4, -1.2]
        {
            assert_gradients(
                &circuit,
                parameter,
                &observable,
                theta,
                theta.cos(),
                -theta.sin(),
            );
        }
    }

    #[test]
    fn reused_parameter_sums_per_occurrence_shifts() {
        // Ry(theta)Ry(theta)|0> = Ry(2theta)|0>, so <Z>=cos(2theta).
        let parameter = ParameterId(4);
        let mut circuit = Circuit::new(1).unwrap();
        for _ in 0..2
        {
            circuit
                .push(Operation::Ry {
                    target: 0,
                    parameter: Parameter::Symbol(parameter),
                })
                .unwrap();
        }
        let theta = 0.31;
        assert_gradients(
            &circuit,
            parameter,
            &Observable::z(0),
            theta,
            (2.0 * theta).cos(),
            -2.0 * (2.0 * theta).sin(),
        );
    }

    #[test]
    fn phase_shift_does_not_claim_the_rotation_shift_rule() {
        let parameter = ParameterId(9);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::PhaseShift {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        let values = ParameterValues::new().with(parameter, 0.2).unwrap();
        assert_eq!(
            parameter_shift_gradient(&circuit, &values, &Observable::z(0), parameter),
            Err(QuantumError::UnsupportedGradientRule { parameter: 9 })
        );
    }

    #[test]
    fn adjoint_matches_closed_form_and_parameter_shift_for_ry() {
        let parameter = ParameterId(21);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();

        for theta in [
            0.0,
            1.0e-3,
            core::f32::consts::FRAC_PI_4,
            core::f32::consts::FRAC_PI_2,
            -0.73,
            core::f32::consts::PI - 1.0e-3,
        ]
        {
            let values = ParameterValues::new().with(parameter, theta).unwrap();
            let adjoint =
                adjoint_gradient(&circuit, &values, &Observable::z(0), parameter).unwrap();
            let shifted =
                parameter_shift_gradient(&circuit, &values, &Observable::z(0), parameter).unwrap();
            let expected = -theta.sin();

            assert!(
                (adjoint - expected).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, expected={expected}"
            );
            assert!(
                (adjoint - shifted).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, shifted={shifted}"
            );
        }
    }

    #[test]
    fn adjoint_matches_closed_form_and_parameter_shift_for_rx() {
        // H then S prepares (|0> + i|1>) / sqrt(2).
        // After Rx(theta), <Z> = sin(theta), hence d<Z>/dtheta = cos(theta).
        let parameter = ParameterId(22);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::S { target: 0 })
            .unwrap()
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();

        for theta in [
            0.0,
            1.0e-3,
            core::f32::consts::FRAC_PI_4,
            core::f32::consts::FRAC_PI_2,
            -0.83,
            core::f32::consts::PI - 1.0e-3,
        ]
        {
            let values = ParameterValues::new().with(parameter, theta).unwrap();
            let adjoint =
                adjoint_gradient(&circuit, &values, &Observable::z(0), parameter).unwrap();
            let shifted =
                parameter_shift_gradient(&circuit, &values, &Observable::z(0), parameter).unwrap();
            let expected = theta.cos();

            assert!(
                (adjoint - expected).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, expected={expected}"
            );
            assert!(
                (adjoint - shifted).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, shifted={shifted}"
            );
        }
    }

    #[test]
    fn adjoint_matches_closed_form_and_parameter_shift_for_rz() {
        // H prepares |+>. After Rz(theta), <X> = cos(theta), hence
        // d<X>/dtheta = -sin(theta).
        let parameter = ParameterId(23);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::Rz {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();

        for theta in [
            0.0,
            1.0e-3,
            core::f32::consts::FRAC_PI_4,
            core::f32::consts::FRAC_PI_2,
            -0.57,
            core::f32::consts::PI - 1.0e-3,
        ]
        {
            let values = ParameterValues::new().with(parameter, theta).unwrap();
            let adjoint =
                adjoint_gradient(&circuit, &values, &Observable::x(0), parameter).unwrap();
            let shifted =
                parameter_shift_gradient(&circuit, &values, &Observable::x(0), parameter).unwrap();
            let expected = -theta.sin();

            assert!(
                (adjoint - expected).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, expected={expected}"
            );
            assert!(
                (adjoint - shifted).abs() <= ANALYTIC_TOLERANCE,
                "theta={theta}, adjoint={adjoint}, shifted={shifted}"
            );
        }
    }

    #[test]
    fn adjoint_matches_parameter_shift_for_complex_entangled_circuit() {
        let alpha = ParameterId(9);
        let beta = ParameterId(2);
        let gamma = ParameterId(17);

        let mut circuit = Circuit::new(3).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::S { target: 0 })
            .unwrap()
            .push(Operation::Rx {
                target: 1,
                parameter: Parameter::Symbol(alpha),
            })
            .unwrap()
            .push(Operation::Cnot {
                control: 0,
                target: 1,
            })
            .unwrap()
            .push(Operation::Ry {
                target: 2,
                parameter: Parameter::Symbol(beta),
            })
            .unwrap()
            .push(Operation::Cz {
                control: 1,
                target: 2,
            })
            .unwrap()
            .push(Operation::Rz {
                target: 0,
                parameter: Parameter::Symbol(gamma),
            })
            .unwrap()
            .push(Operation::PhaseShift {
                target: 2,
                parameter: Parameter::Fixed(0.41),
            })
            .unwrap()
            .push(Operation::Swap {
                first: 0,
                second: 2,
            })
            .unwrap();

        let observables = [
            Observable::y(2),
            Observable::new(vec![
                PauliTerm::new(0, Pauli::X),
                PauliTerm::new(1, Pauli::Z),
            ])
            .unwrap(),
            Observable::new(vec![
                PauliTerm::new(0, Pauli::Y),
                PauliTerm::new(1, Pauli::X),
                PauliTerm::new(2, Pauli::Z),
            ])
            .unwrap(),
            Observable::z(1),
        ];

        let values = ParameterValues::new()
            .with(alpha, 0.37)
            .unwrap()
            .with(beta, -0.61)
            .unwrap()
            .with(gamma, 0.29)
            .unwrap();

        let jacobian = adjoint_jacobian(&circuit, &values, &observables).unwrap();
        assert_eq!(
            jacobian.keys().copied().collect::<Vec<_>>(),
            vec![beta, alpha, gamma]
        );

        let mut max_error = 0.0f32;
        for parameter in [alpha, beta, gamma]
        {
            let adjoint = jacobian.get(&parameter).unwrap();
            let shifted =
                parameter_shift_gradients(&circuit, &values, &observables, parameter).unwrap();
            assert_eq!(adjoint.len(), observables.len());
            for (&actual, &expected) in adjoint.iter().zip(&shifted)
            {
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 2.0e-4,
                    "parameter={parameter:?}, adjoint={actual}, shifted={expected}, error={error}"
                );
            }
        }
        eprintln!("adjoint complex-entangled max parameter-shift error: {max_error:.9e}");
    }

    #[test]
    fn adjoint_reused_parameter_sums_occurrences_in_forward_order() {
        let parameter = ParameterId(31);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::Ry {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap()
            .push(Operation::Rz {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap()
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();

        let values = ParameterValues::new().with(parameter, 0.43).unwrap();
        let observables = [Observable::y(0), Observable::z(0), Observable::x(0)];

        let adjoint = adjoint_gradients(&circuit, &values, &observables, parameter).unwrap();
        let shifted =
            parameter_shift_gradients(&circuit, &values, &observables, parameter).unwrap();

        assert_eq!(adjoint.len(), 3);
        for (observable_index, (&actual, &expected)) in adjoint.iter().zip(&shifted).enumerate()
        {
            let error = (actual - expected).abs();
            assert!(
                error <= 2.0e-4,
                "observable={observable_index}, adjoint={actual}, shifted={expected}, error={error}"
            );
        }
    }

    #[test]
    fn adjoint_matches_central_finite_difference() {
        let parameter = ParameterId(44);
        let mut circuit = Circuit::new(2).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::Cnot {
                control: 0,
                target: 1,
            })
            .unwrap()
            .push(Operation::Ry {
                target: 1,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap()
            .push(Operation::S { target: 0 })
            .unwrap();

        let observable = Observable::new(vec![
            PauliTerm::new(0, Pauli::Y),
            PauliTerm::new(1, Pauli::X),
        ])
        .unwrap();
        let values = ParameterValues::new().with(parameter, -0.38).unwrap();

        let adjoint = adjoint_gradient(&circuit, &values, &observable, parameter).unwrap();
        let finite =
            finite_difference_gradient(&circuit, &values, &observable, parameter, EPSILON).unwrap();
        let error = (adjoint - finite).abs();

        eprintln!("adjoint finite-difference error: {error:.9e}");
        assert!(
            error <= FINITE_DIFFERENCE_TOLERANCE,
            "adjoint={adjoint}, finite={finite}, error={error}"
        );
    }

    #[test]
    fn adjoint_is_exactly_repeatable_and_preserves_observable_order() {
        let parameter = ParameterId(55);
        let mut circuit = Circuit::new(1).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::Rz {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();

        let values = ParameterValues::new().with(parameter, 0.27).unwrap();
        let observables = [Observable::y(0), Observable::z(0), Observable::x(0)];

        let first = adjoint_jacobian(&circuit, &values, &observables).unwrap();
        let second = adjoint_jacobian(&circuit, &values, &observables).unwrap();

        assert_eq!(first, second);

        let derivatives = first.get(&parameter).unwrap();
        let shifted =
            parameter_shift_gradients(&circuit, &values, &observables, parameter).unwrap();

        let mut max_error = 0.0f32;
        for (observable, (&actual, &expected)) in derivatives.iter().zip(&shifted).enumerate()
        {
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= ANALYTIC_TOLERANCE,
                "observable={observable}, adjoint={actual}, shifted={expected}, error={error}"
            );
        }
        eprintln!("adjoint observable-order max parameter-shift error: {max_error:.9e}");
    }

    #[test]
    fn adjoint_rejects_empty_observables_unknown_parameter_and_symbolic_phase() {
        let parameter = ParameterId(61);
        let unknown = ParameterId(62);

        let mut rotation = Circuit::new(1).unwrap();
        rotation
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        let values = ParameterValues::new().with(parameter, 0.2).unwrap();

        assert_eq!(
            adjoint_jacobian(&rotation, &values, &[]),
            Err(QuantumError::InvalidObservableCount {
                minimum: 1,
                maximum: None,
                actual: 0,
            })
        );
        assert_eq!(
            adjoint_gradient(&rotation, &values, &Observable::z(0), unknown),
            Err(QuantumError::UnknownParameter {
                parameter: unknown.0,
            })
        );

        let mut phase = Circuit::new(1).unwrap();
        phase
            .push(Operation::PhaseShift {
                target: 0,
                parameter: Parameter::Symbol(parameter),
            })
            .unwrap();
        assert_eq!(
            adjoint_jacobian(&phase, &values, &[Observable::z(0)]),
            Err(QuantumError::UnsupportedGradientRule {
                parameter: parameter.0,
            })
        );
    }

    #[test]
    fn every_supported_bound_gate_round_trips_through_its_adjoint() {
        let rx = ParameterId(70);
        let ry = ParameterId(71);
        let rz = ParameterId(72);

        let mut circuit = Circuit::new(3).unwrap();
        circuit
            .push(Operation::H { target: 0 })
            .unwrap()
            .push(Operation::X { target: 1 })
            .unwrap()
            .push(Operation::Y { target: 2 })
            .unwrap()
            .push(Operation::Z { target: 0 })
            .unwrap()
            .push(Operation::S { target: 1 })
            .unwrap()
            .push(Operation::Sdg { target: 2 })
            .unwrap()
            .push(Operation::T { target: 0 })
            .unwrap()
            .push(Operation::Tdg { target: 1 })
            .unwrap()
            .push(Operation::Rx {
                target: 0,
                parameter: Parameter::Symbol(rx),
            })
            .unwrap()
            .push(Operation::Ry {
                target: 1,
                parameter: Parameter::Symbol(ry),
            })
            .unwrap()
            .push(Operation::Rz {
                target: 2,
                parameter: Parameter::Symbol(rz),
            })
            .unwrap()
            .push(Operation::PhaseShift {
                target: 0,
                parameter: Parameter::Fixed(0.23),
            })
            .unwrap()
            .push(Operation::Cnot {
                control: 0,
                target: 1,
            })
            .unwrap()
            .push(Operation::Cz {
                control: 1,
                target: 2,
            })
            .unwrap()
            .push(Operation::Swap {
                first: 0,
                second: 2,
            })
            .unwrap();

        let values = ParameterValues::new()
            .with(rx, 0.31)
            .unwrap()
            .with(ry, -0.47)
            .unwrap()
            .with(rz, 0.68)
            .unwrap();
        let bound = circuit.bind(&values).unwrap();

        let mut state = DenseStateVector::zero(3).unwrap();
        for operation in bound.operations()
        {
            apply_bound_operation(&mut state, operation).unwrap();
        }
        for operation in bound.operations().iter().rev()
        {
            apply_adjoint_operation(&mut state, operation).unwrap();
        }

        let mut max_error = 0.0f32;
        for (index, &amplitude) in state.amplitudes().iter().enumerate()
        {
            let expected = if index == 0
            {
                Complex32::one()
            }
            else
            {
                Complex32::zero()
            };
            let error = (amplitude - expected).norm();
            max_error = max_error.max(error);
            assert!(
                error <= 5.0e-5,
                "index={index}, amplitude={amplitude:?}, error={error}"
            );
        }
        eprintln!("adjoint inverse round-trip max amplitude error: {max_error:.9e}");
    }
}
