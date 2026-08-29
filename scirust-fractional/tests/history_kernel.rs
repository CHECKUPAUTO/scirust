use scirust_fractional::{
    CaputoL1HistoryKernel, FractionalOrder, caputo_l1_nonuniform, caputo_l1_uniform,
};
use scirust_history::{
    BoundedHistory, CompleteHistory, HistoryBackend, HistoryEntry, HistoryKernel,
};
use scirust_special::gamma;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;
    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn complete_history(samples: &[f64], positions: &[f64]) -> CompleteHistory<f64, f64> {
    let mut history = CompleteHistory::with_capacity(samples.len());
    for (&sample, &position) in samples.iter().zip(positions)
    {
        history.push(HistoryEntry::new(sample, position)).unwrap();
    }
    history
}

#[test]
fn history_kernel_matches_direct_nonuniform_operator() {
    let order = FractionalOrder::new(0.5).unwrap();
    let positions: [f64; 8] = [0.0, 0.05, 0.19, 0.28, 0.5, 0.61, 0.9, 1.0];
    let samples: Vec<f64> = positions.iter().map(|t| t.cos()).collect();
    let history = complete_history(&samples, &positions);
    let kernel = CaputoL1HistoryKernel::new(order);

    let direct = caputo_l1_nonuniform(&samples, &positions, order).unwrap();
    let adapted = kernel.evaluate(&history.view()).unwrap();

    assert_eq!(adapted.to_bits(), direct.to_bits());
}

#[test]
fn history_kernel_preserves_linear_nonuniform_oracle() {
    let alpha = 0.5;
    let order = FractionalOrder::new(alpha).unwrap();
    let positions: [f64; 10] = [0.0, 0.05, 0.19, 0.28, 0.5, 0.61, 0.9, 1.0, 1.37, 1.5];
    let slope = 2.3;
    let samples: Vec<f64> = positions.iter().map(|t| slope * t).collect();
    let history = complete_history(&samples, &positions);

    let expected = slope * positions.last().unwrap().powf(1.0 - alpha) / gamma(2.0 - alpha);
    let actual = CaputoL1HistoryKernel::new(order)
        .evaluate(&history.view())
        .unwrap();

    assert_close(actual, expected, 1.0e-13);
}

#[test]
fn history_kernel_matches_uniform_operator_on_uniform_grid() {
    let order = FractionalOrder::new(0.63).unwrap();
    let step = 0.01;
    let samples: Vec<f64> = (0..=256)
        .map(|i| {
            let t = i as f64 * step;
            t.sin() + 0.25 * t
        })
        .collect();
    let positions: Vec<f64> = (0..=256).map(|i| i as f64 * step).collect();
    let history = complete_history(&samples, &positions);

    let uniform = caputo_l1_uniform(&samples, step, order).unwrap();
    let adapted = CaputoL1HistoryKernel::new(order)
        .evaluate(&history.view())
        .unwrap();

    assert_close(adapted, uniform, 1.0e-11);
}

#[test]
fn bounded_and_complete_agree_when_window_covers_all_samples() {
    let order = FractionalOrder::new(0.41).unwrap();
    let positions: [f64; 5] = [0.0, 0.1, 0.31, 0.7, 1.2];
    let samples: Vec<f64> = positions.iter().map(|t| t * t + 0.5 * t).collect();
    let complete = complete_history(&samples, &positions);
    let mut bounded = BoundedHistory::new(samples.len()).unwrap();
    for (&sample, &position) in samples.iter().zip(&positions)
    {
        bounded.push(HistoryEntry::new(sample, position)).unwrap();
    }

    let kernel = CaputoL1HistoryKernel::new(order);
    let reference = kernel.evaluate(&complete.view()).unwrap();
    let approximation = kernel.evaluate(&bounded.view()).unwrap();

    assert_eq!(reference.to_bits(), approximation.to_bits());
}

#[test]
fn repeated_history_kernel_evaluation_is_bit_identical() {
    let order = FractionalOrder::new(0.5).unwrap();
    let positions: [f64; 8] = [0.0, 0.05, 0.19, 0.28, 0.5, 0.61, 0.9, 1.0];
    let samples: Vec<f64> = positions.iter().map(|t| t.cos()).collect();
    let history = complete_history(&samples, &positions);
    let kernel = CaputoL1HistoryKernel::new(order);

    let first = kernel.evaluate(&history.view()).unwrap();
    let second = kernel.evaluate(&history.view()).unwrap();

    assert_eq!(first.to_bits(), second.to_bits());
}
