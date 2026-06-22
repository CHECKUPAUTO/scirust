//! Reproducible head-to-head: two gradient-free RSI loops train the **same**
//! real scirust-core ReLU MLP on the same regression task.
//!
//! - `(1+λ)-ES` (`OnePlusLambda`): one self-adapting parent.
//! - **PBT neuro-evolution** (`Pbt`): a population where each member runs a
//!   `(1+1)` evolution step and PBT self-tunes each member's mutation scale `σ`.
//!
//! Everything is seeded, so the printed numbers reproduce run to run.
//!
//! Run with: `cargo run -p scirust-rsi --example optimizer_bench`

use rand::Rng;
use rand::rngs::StdRng;
use scirust_core::nn::ibp::{IbpLinear, IbpMlp};
use scirust_rsi::Guard;
use scirust_rsi::evo::OnePlusLambda;
use scirust_rsi::pbt::{Pbt, PbtTask};

// MLP shape: 1 -> 8 (ReLU) -> 1.
const IN: usize = 1;
const HID: usize = 8;
const OUT: usize = 1;
const N_PARAMS: usize = IN * HID + HID + HID * OUT + OUT;

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

/// Shared regression task: fit `sin(1.5 x)` on [-2, 2].
struct Task {
    xs: Vec<f64>,
    ys: Vec<f64>,
}
impl Task {
    fn new() -> Self {
        let xs: Vec<f64> = (0..41).map(|i| -2.0 + i as f64 * 0.1).collect();
        let ys = xs.iter().map(|&x| (1.5 * x).sin()).collect();
        Self { xs, ys }
    }
    /// Fitness = -MSE (higher is better).
    fn fitness(&self, p: &[f64]) -> f64 {
        let net = build_mlp(p);
        let s: f64 = self
            .xs
            .iter()
            .zip(&self.ys)
            .map(|(&x, &y)| (net.forward(&[x as f32])[0] as f64 - y).powi(2))
            .sum();
        -s / self.xs.len() as f64
    }
}

/// PBT member = NN weights; the self-tuned hyper-parameter is the mutation σ.
/// Each `step` is one `(1+1)`-ES trial at the member's current σ.
struct NeuroPbt {
    task: Task,
}
impl PbtTask for NeuroPbt {
    type Hyper = f64; // mutation scale σ

    fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, f64) {
        let params = (0..N_PARAMS).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let sigma = rng.gen_range(0.05..0.6);
        (params, sigma)
    }

    fn step(&self, params: &mut Vec<f64>, &sigma: &f64, rng: &mut StdRng) -> f64 {
        let cur = self.task.fitness(params);
        let cand: Vec<f64> = params
            .iter()
            .map(|p| p + sigma * rng.gen_range(-1.0..1.0))
            .collect();
        let cf = self.task.fitness(&cand);
        if cf > cur
        {
            *params = cand;
            cf
        }
        else
        {
            cur
        }
    }

    fn perturb(&self, &sigma: &f64, rng: &mut StdRng) -> f64 {
        let factor = if rng.gen_bool(0.5) { 0.7 } else { 1.4 };
        (sigma * factor).clamp(1e-3, 1.0)
    }
}

fn main() {
    let guard = Guard::new().max_iters(4_000).target(-1e-3);

    println!("=== Training the same scirust-core MLP ({N_PARAMS} params) two ways ===\n");
    println!(
        "  {:<24} {:>10} {:>8}  {:>8}",
        "method", "MSE", "iters", "monotone"
    );

    // --- (1+λ)-ES -------------------------------------------------------
    let task = Task::new();
    let (_x, fit_es, rep_es) = OnePlusLambda::new(0xE5).lambda(24).sigma0(0.5).optimize(
        vec![0.0; N_PARAMS],
        |p| task.fitness(p),
        &guard,
    );
    println!(
        "  {:<24} {:>10.5} {:>8} {:>9}",
        "(1+λ)-ES",
        -fit_es,
        rep_es.iterations,
        rep_es.is_monotone()
    );

    // --- PBT neuro-evolution -------------------------------------------
    let (_params, best_sigma, rep_pbt) = Pbt::new(0xB7)
        .pop_size(24)
        .steps_per_gen(8)
        .run(&NeuroPbt { task: Task::new() }, &guard);
    println!(
        "  {:<24} {:>10.5} {:>8} {:>9}",
        "PBT neuro-evolution",
        -rep_pbt.best_fitness,
        rep_pbt.iterations,
        rep_pbt.is_monotone()
    );
    println!("\n  PBT self-tuned the winning member's σ to {best_sigma:.4}.");
    println!("  Both loops are seeded → these numbers reproduce exactly.");
}
