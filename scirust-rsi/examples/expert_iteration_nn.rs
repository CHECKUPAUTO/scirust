//! Expert Iteration training a **real** scirust-core ReLU MLP.
//!
//! The *policy* is the network's flat weight vector. The *expert* is that policy
//! augmented with search — a short `(1+1)`-ES look-ahead that finds better
//! weights than the bare policy. Each round, several experts improve the current
//! policy, the best improvement is **distilled** back in (adopted only if it
//! beats the incumbent), and the now-stronger policy makes the next round's
//! experts start from a better place. That feedback is the recursion.
//!
//! Run with: `cargo run -p scirust-rsi --example expert_iteration_nn`

use rand::Rng;
use rand::rngs::StdRng;
use scirust_core::nn::ibp::{IbpLinear, IbpMlp};
use scirust_rsi::expert_iteration::{ExpertIteration, ExpertIterationTask};
use scirust_rsi::{Fitness, Guard};

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

/// Regression task: fit `sin(1.5 x)` on [-2, 2]; fitness = -MSE.
struct NnExit {
    xs: Vec<f64>,
    ys: Vec<f64>,
    expert_trials: usize,
}
impl NnExit {
    fn fitness(&self, p: &[f64]) -> f64 {
        let net = build_mlp(p);
        -self
            .xs
            .iter()
            .zip(&self.ys)
            .map(|(&x, &y)| (net.forward(&[x as f32])[0] as f64 - y).powi(2))
            .sum::<f64>()
            / self.xs.len() as f64
    }
}

impl ExpertIterationTask for NnExit {
    type Sample = (); // single-task; each "sample" is one expert restart
    type Policy = Vec<f64>; // network weights
    type Target = Vec<f64>; // improved weights from the expert search

    fn samples(&self, _rng: &mut StdRng) -> Vec<()> {
        vec![(); 4] // 4 expert restarts per round
    }

    fn base_policy(&self) -> Vec<f64> {
        vec![0.0; N_PARAMS] // start from scratch
    }

    fn expert(&self, policy: &Vec<f64>, _s: &(), rng: &mut StdRng) -> Vec<f64> {
        // Search-augmented "expert": a short (1+1)-ES around the policy.
        let mut cur = policy.clone();
        let mut cf = self.fitness(&cur);
        let sigma = 0.3;
        for _ in 0..self.expert_trials
        {
            let cand: Vec<f64> = cur
                .iter()
                .map(|p| p + sigma * rng.gen_range(-1.0..1.0))
                .collect();
            let f = self.fitness(&cand);
            if f > cf
            {
                cf = f;
                cur = cand;
            }
        }
        cur
    }

    fn distil(&self, base: &Vec<f64>, data: &[((), Vec<f64>)]) -> Vec<f64> {
        // Imitate the strongest expert (so the distilled policy never regresses).
        data.iter()
            .map(|(_, t)| t.clone())
            .max_by(|a, b| self.fitness(a).partial_cmp(&self.fitness(b)).unwrap())
            .unwrap_or_else(|| base.clone())
    }

    fn evaluate(&self, policy: &Vec<f64>) -> Fitness {
        self.fitness(policy)
    }
}

fn main() {
    let xs: Vec<f64> = (0..41).map(|i| -2.0 + i as f64 * 0.1).collect();
    let ys = xs.iter().map(|&x| (1.5 * x).sin()).collect();
    let task = NnExit {
        xs,
        ys,
        expert_trials: 300,
    };

    println!("=== Expert Iteration training a real scirust-core MLP ({N_PARAMS} params) ===");
    println!("  initial MSE : {:.5}", -task.evaluate(&task.base_policy()));

    let (policy, report) =
        ExpertIteration::new(0xE1).run(&task, &Guard::new().max_iters(150).target(-1e-2));

    println!(
        "  final MSE   : {:.5}  ({} rounds, {:?}, accepted {} distillations, monotone = {})",
        -report.best_fitness,
        report.iterations,
        report.stop_reason,
        report.accepted,
        report.is_monotone()
    );

    let net = build_mlp(&policy);
    println!("\n  x      target    policy");
    for &x in &[-1.5_f64, -0.5, 0.5, 1.5]
    {
        println!(
            "  {:+.1}   {:+.3}   {:+.3}",
            x,
            (1.5 * x).sin(),
            net.forward(&[x as f32])[0]
        );
    }
    println!("\nApprentice ← distil(expert search) — every round, non-regressing by construction.");
}
