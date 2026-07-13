use scirust_fractional::{
    FractionalError, FractionalOrder, caputo_l1_uniform, grunwald_letnikov_weights,
    riemann_liouville_gl_uniform,
};
use scirust_special::gamma;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;

    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

#[test]
fn order_rejects_invalid_values() {
    assert_eq!(
        FractionalOrder::new(0.0),
        Err(FractionalError::InvalidOrder(0.0))
    );
    assert_eq!(
        FractionalOrder::new(1.0),
        Err(FractionalError::InvalidOrder(1.0))
    );
    assert!(FractionalOrder::new(f64::NAN).is_err());
    assert!(FractionalOrder::new(f64::INFINITY).is_err());
}

#[test]
fn half_order_weights_match_hand_derived_values() {
    let order = FractionalOrder::new(0.5).unwrap();
    let weights = grunwald_letnikov_weights(order, 5);
    let expected = [1.0, -0.5, -0.125, -0.0625, -0.039_062_5];

    for (actual, expected) in weights.iter().zip(expected)
    {
        assert_close(*actual, expected, 1.0e-15);
    }
}

#[test]
fn caputo_derivative_of_a_constant_is_zero() {
    let order = FractionalOrder::new(0.37).unwrap();
    let samples = vec![4.25; 128];

    let derivative = caputo_l1_uniform(&samples, 0.05, order).unwrap();

    assert_eq!(derivative.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn caputo_l1_is_exact_for_a_linear_function_on_uniform_grid() {
    let alpha = 0.5;
    let order = FractionalOrder::new(alpha).unwrap();
    let step = 0.125_f64;
    let intervals = 64_usize;

    let samples: Vec<f64> = (0..=intervals).map(|i| i as f64 * step).collect();

    let t_end = intervals as f64 * step;
    let expected = t_end.powf(1.0 - alpha) / gamma(2.0 - alpha);
    let actual = caputo_l1_uniform(&samples, step, order).unwrap();

    assert_close(actual, expected, 2.0e-14);
}

#[test]
fn grunwald_letnikov_converges_for_a_quadratic_power() {
    let alpha = 0.5;
    let order = FractionalOrder::new(alpha).unwrap();
    let intervals = 4096_usize;
    let step = 1.0 / intervals as f64;

    let samples: Vec<f64> = (0..=intervals)
        .map(|i| {
            let t = i as f64 * step;
            t * t
        })
        .collect();

    let expected = gamma(3.0) / gamma(3.0 - alpha);
    let actual = riemann_liouville_gl_uniform(&samples, step, order).unwrap();

    assert_close(actual, expected, 5.0e-4);
}

#[test]
fn repeated_evaluation_is_bit_identical() {
    let order = FractionalOrder::new(0.63).unwrap();
    let samples: Vec<f64> = (0..=256)
        .map(|i| {
            let t = i as f64 * 0.01;
            t.sin() + 0.25 * t
        })
        .collect();

    let first = caputo_l1_uniform(&samples, 0.01, order).unwrap();
    let second = caputo_l1_uniform(&samples, 0.01, order).unwrap();

    assert_eq!(first.to_bits(), second.to_bits());
}

#[test]
fn invalid_inputs_return_explicit_errors() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_uniform(&[], 0.1, order),
        Err(FractionalError::EmptySamples)
    );
    assert_eq!(
        caputo_l1_uniform(&[1.0], 0.1, order),
        Err(FractionalError::TooFewSamples)
    );
    assert_eq!(
        caputo_l1_uniform(&[0.0, 1.0], 0.0, order),
        Err(FractionalError::InvalidStep(0.0))
    );
    assert_eq!(
        caputo_l1_uniform(&[0.0, f64::NAN], 0.1, order),
        Err(FractionalError::NonFiniteSample(1))
    );
}
