//! End-to-end demo of every recursive-self-improvement loop in `scirust-rsi`.
//!
//! Run with: `cargo run -p scirust-rsi --example rsi_demo`

use rand::Rng;
use rand::rngs::StdRng;
use scirust_rsi::evo::OnePlusLambda;
use scirust_rsi::pbt::{Pbt, PbtTask};
use scirust_rsi::refine::{RefineTask, SelfRefiner};
use scirust_rsi::{Fitness, Guard, bench};

fn banner(title: &str) {
    println!("\n=== {title} ===");
}

// --- Self-Refine: shrink a vector toward the origin -------------------------
struct Shrink;
impl RefineTask for Shrink {
    type Solution = Vec<f64>;
    fn initial(&self, _rng: &mut StdRng) -> Vec<f64> {
        vec![5.0, -4.0, 3.0]
    }
    fn score(&self, s: &Vec<f64>) -> Fitness {
        -s.iter().map(|v| v * v).sum::<f64>()
    }
    fn refine(&self, s: &Vec<f64>, rng: &mut StdRng) -> Vec<f64> {
        let mut out = s.clone();
        let i = (0..out.len())
            .max_by(|&a, &b| out[a].abs().partial_cmp(&out[b].abs()).unwrap())
            .unwrap();
        out[i] *= rng.gen_range(0.3..0.9);
        out
    }
}

// --- PBT: tune the learning rate of 1-D gradient descent --------------------
struct LrSearch;
impl PbtTask for LrSearch {
    type Hyper = f64;
    fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, f64) {
        (vec![10.0], rng.gen_range(0.001..1.5))
    }
    fn step(&self, params: &mut Vec<f64>, hyper: &f64, _rng: &mut StdRng) -> Fitness {
        let x = params[0];
        params[0] = x - hyper * 2.0 * x;
        -params[0] * params[0]
    }
    fn perturb(&self, hyper: &f64, rng: &mut StdRng) -> f64 {
        let f = if rng.gen_bool(0.5) { 0.8 } else { 1.25 };
        (hyper * f).clamp(1e-4, 1.5)
    }
}

fn main() {
    banner("Self-Refine (critique-and-revise)");
    let (sol, rep) = SelfRefiner::new(1).run(&Shrink, &Guard::new().max_iters(200).patience(20));
    println!("  best solution : {sol:?}");
    println!(
        "  fitness {:.6} in {} iters ({:?}), monotone = {}",
        rep.best_fitness,
        rep.iterations,
        rep.stop_reason,
        rep.is_monotone()
    );

    banner("(1+λ)-ES with the 1/5 success rule");
    for (name, f) in [
        ("sphere", bench::sphere as fn(&[f64]) -> f64),
        ("rosenbrock", bench::rosenbrock),
        ("rastrigin", bench::rastrigin),
    ]
    {
        let (_x, fit, rep) = OnePlusLambda::new(7).lambda(20).sigma0(1.0).optimize(
            vec![3.0; 4],
            f,
            &Guard::new().max_iters(5_000).target(-1e-8),
        );
        println!(
            "  {name:<11}: fitness {:.6} in {} gens ({:?})",
            fit, rep.iterations, rep.stop_reason
        );
    }

    banner("Population-Based Training (self-tuning the learning rate)");
    let (params, lr, rep) = Pbt::new(2024)
        .pop_size(24)
        .run(&LrSearch, &Guard::new().max_iters(100).target(-1e-9));
    println!(
        "  converged x = {:.2e}, discovered lr = {:.4}, {} gens, monotone = {}",
        params[0],
        lr,
        rep.iterations,
        rep.is_monotone()
    );

    println!("\nAll loops are bounded, elitist (non-regressing) and seeded. See the");
    println!("module docs for STaR and Expert Iteration, which follow the same shape.");
}
