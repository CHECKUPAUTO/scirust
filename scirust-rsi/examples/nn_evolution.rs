//! Train a **real neural network** with the gradient-free `(1+λ)`-ES.
//!
//! The network is a scirust-core ReLU MLP (`IbpMlp`) — concrete `f32` weights,
//! no autograd. `OnePlusLambda` searches the flat parameter vector to minimise
//! mean-squared error on a regression target, and we print a scirust-learning
//! linear-regression baseline for comparison (a line can't fit a curve, so the
//! evolved MLP should win).
//!
//! Run with: `cargo run -p scirust-rsi --example nn_evolution`

use scirust_core::nn::ibp::{IbpLinear, IbpMlp};
use scirust_rsi::Guard;
use scirust_rsi::evo::OnePlusLambda;

// MLP shape: 1 -> 8 (ReLU) -> 1.
const IN: usize = 1;
const HID: usize = 8;
const OUT: usize = 1;
// Parameter layout: W1(IN*HID), b1(HID), W2(HID*OUT), b2(OUT).
const N_PARAMS: usize = IN * HID + HID + HID * OUT + OUT;

/// Slice a flat `f64` parameter vector into a concrete ReLU MLP.
fn build_mlp(p: &[f64]) -> IbpMlp {
    let f = |x: f64| x as f32;
    let (w1, rest) = p.split_at(IN * HID);
    let (b1, rest) = rest.split_at(HID);
    let (w2, b2) = rest.split_at(HID * OUT);
    IbpMlp::new(vec![
        IbpLinear::new(
            w1.iter().map(|&v| f(v)).collect(),
            b1.iter().map(|&v| f(v)).collect(),
            IN,
            HID,
        ),
        IbpLinear::new(
            w2.iter().map(|&v| f(v)).collect(),
            b2.iter().map(|&v| f(v)).collect(),
            HID,
            OUT,
        ),
    ])
}

fn main() {
    // Target: a smooth nonlinear curve a ReLU MLP can approximate piecewise.
    let xs: Vec<f64> = (0..41).map(|i| -2.0 + i as f64 * 0.1).collect();
    let target = |x: f64| (1.5 * x).sin();
    let ys: Vec<f64> = xs.iter().map(|&x| target(x)).collect();

    let mse = |p: &[f64]| {
        let net = build_mlp(p);
        let s: f64 = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| {
                let pred = net.forward(&[x as f32])[0] as f64;
                (pred - y).powi(2)
            })
            .sum();
        s / xs.len() as f64
    };

    // Fitness = -MSE (the ES maximises).
    let fitness = |p: &[f64]| -mse(p);

    println!("=== (1+λ)-ES training a real scirust-core ReLU MLP ({N_PARAMS} params) ===");
    let x0 = vec![0.0; N_PARAMS];
    println!("  initial MSE         : {:.5}", mse(&x0));

    let (best, fit, report) = OnePlusLambda::new(0x5C1).lambda(24).sigma0(0.5).optimize(
        x0,
        fitness,
        &Guard::new().max_iters(20_000).target(-1e-3),
    );

    println!(
        "  evolved MLP MSE     : {:.5}  ({} gens, {:?}, monotone = {})",
        -fit,
        report.iterations,
        report.stop_reason,
        report.is_monotone()
    );

    // scirust-learning baseline: best straight line through the data.
    let (slope, intercept) = scirust_learning::linear_regression(&xs, &ys);
    let lin_mse: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| (slope * x + intercept - y).powi(2))
        .sum::<f64>()
        / xs.len() as f64;
    println!("  linear baseline MSE : {lin_mse:.5}  (scirust_learning::linear_regression)");

    let net = build_mlp(&best);
    println!("\n  x      target    MLP");
    for &x in &[-1.5, -0.5, 0.5, 1.5]
    {
        println!(
            "  {:+.1}   {:+.3}   {:+.3}",
            x,
            target(x),
            net.forward(&[x as f32])[0]
        );
    }
    println!(
        "\nThe gradient-free ES trained a real network end-to-end — non-regressing by construction."
    );
}
