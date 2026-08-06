use scirust_fractional::{
    FractionalError, FractionalOrder, caputo_l1_nonuniform, caputo_l1_uniform,
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
fn caputo_l1_nonuniform_is_exact_for_a_linear_function() {
    let alpha = 0.5;
    let order = FractionalOrder::new(alpha).unwrap();

    // Deliberately irregular spacing: no common step.
    let sample_times = [0.0, 0.05, 0.19, 0.28, 0.5, 0.61, 0.9, 1.0, 1.37, 1.5];
    let slope = 2.3;
    let samples: Vec<f64> = sample_times.iter().map(|t| slope * t).collect();

    let t_end = *sample_times.last().unwrap();
    let expected = slope * t_end.powf(1.0 - alpha) / gamma(2.0 - alpha);
    let actual = caputo_l1_nonuniform(&samples, &sample_times, order).unwrap();

    assert_close(actual, expected, 1.0e-13);
}

#[test]
fn caputo_l1_nonuniform_matches_uniform_on_a_uniform_grid() {
    let order = FractionalOrder::new(0.63).unwrap();
    let step = 0.01;
    let samples: Vec<f64> = (0..=256)
        .map(|i| {
            let t = i as f64 * step;
            t.sin() + 0.25 * t
        })
        .collect();
    let sample_times: Vec<f64> = (0..=256).map(|i| i as f64 * step).collect();

    let uniform = caputo_l1_uniform(&samples, step, order).unwrap();
    let nonuniform = caputo_l1_nonuniform(&samples, &sample_times, order).unwrap();

    assert_close(nonuniform, uniform, 1.0e-11);
}

#[test]
fn caputo_l1_nonuniform_derivative_of_a_constant_is_zero() {
    let order = FractionalOrder::new(0.37).unwrap();
    let samples = vec![4.25; 32];
    let sample_times: Vec<f64> = (0..32)
        .map(|i| (i as f64) * 0.05 + 0.001 * (i as f64).sin())
        .collect();

    let derivative = caputo_l1_nonuniform(&samples, &sample_times, order).unwrap();

    assert_eq!(derivative.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn caputo_l1_nonuniform_repeated_evaluation_is_bit_identical() {
    let order = FractionalOrder::new(0.5).unwrap();
    let sample_times: [f64; 8] = [0.0, 0.05, 0.19, 0.28, 0.5, 0.61, 0.9, 1.0];
    let samples: Vec<f64> = sample_times.iter().map(|t| t.cos()).collect();

    let first = caputo_l1_nonuniform(&samples, &sample_times, order).unwrap();
    let second = caputo_l1_nonuniform(&samples, &sample_times, order).unwrap();

    assert_eq!(first.to_bits(), second.to_bits());
}

#[test]
fn caputo_l1_nonuniform_rejects_mismatched_lengths() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[0.0, 1.0, 2.0], &[0.0, 1.0], order),
        Err(FractionalError::MismatchedLengths {
            samples: 3,
            sample_times: 2,
        })
    );
}

#[test]
fn caputo_l1_nonuniform_rejects_too_few_samples() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[1.0], &[0.0], order),
        Err(FractionalError::TooFewSamples)
    );
}

#[test]
fn caputo_l1_nonuniform_rejects_non_finite_sample_time() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[0.0, 1.0, 2.0], &[0.0, f64::NAN, 1.0], order),
        Err(FractionalError::NonFiniteSampleTime(1))
    );
}

#[test]
fn caputo_l1_nonuniform_rejects_non_finite_sample() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[0.0, f64::NAN, 2.0], &[0.0, 0.5, 1.0], order),
        Err(FractionalError::NonFiniteSample(1))
    );
}

#[test]
fn caputo_l1_nonuniform_rejects_non_monotonic_sample_times() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[0.0, 1.0, 2.0], &[0.0, 0.5, 0.5], order),
        Err(FractionalError::NonMonotonicSampleTimes { index: 2 })
    );
    assert_eq!(
        caputo_l1_nonuniform(&[0.0, 1.0, 2.0], &[0.0, 0.5, 0.3], order),
        Err(FractionalError::NonMonotonicSampleTimes { index: 2 })
    );
}

#[test]
fn caputo_l1_nonuniform_rejects_empty_inputs() {
    let order = FractionalOrder::new(0.5).unwrap();

    assert_eq!(
        caputo_l1_nonuniform(&[], &[], order),
        Err(FractionalError::EmptySamples)
    );
}
