//! Verification-first gradients for exact expectation values.

use super::error::{QuantumError, QuantumResult};
use super::ir::{Circuit, ParameterId, ParameterValues};
use super::observable::Observable;

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
    if values.get(parameter).is_none()
    {
        return Err(QuantumError::UnboundParameter {
            parameter: parameter.0,
        });
    }
    let occurrences = circuit.parameter_occurrences(parameter)?;
    let shift = core::f32::consts::FRAC_PI_2;
    let mut gradient = 0.0f32;
    for operation_index in occurrences
    {
        let upper = circuit
            .bind_with_occurrence_shift(values, operation_index, shift)?
            .execute_dense()?
            .expectation(observable)?;
        let lower = circuit
            .bind_with_occurrence_shift(values, operation_index, -shift)?
            .execute_dense()?
            .expectation(observable)?;
        gradient += 0.5 * (upper - lower);
    }
    if gradient.is_finite()
    {
        Ok(gradient)
    }
    else
    {
        Err(QuantumError::NumericalFailure {
            operation: "parameter-shift gradient",
        })
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
}
