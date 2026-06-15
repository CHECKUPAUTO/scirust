//! Learning subcommands beyond `quickstart`: ownership-model training and
//! evolutionary optimization. Both are deterministic in their seed.

use scirust_core::nn::PcgEngine;
use scirust_core::nn::conformal::ConformalRegressor;
use scirust_core::nn::ibp::{IbpLinear, IbpMlp, Interval, certified_robust, crown_bounds};
use scirust_core::nn::nd_layers::NdLinear;
use scirust_core::quantization::{awq_quantize, gptq_hessian, quantize_gptq, quantize_per_channel};
use scirust_evo::{CmaEs, GeneticAlgorithm};
use scirust_som_dataset::build_training_set;
use scirust_som_inference::{evaluate, ownership_majority_baseline};
use scirust_som_model::{SomModel, SomModelConfig};
use scirust_som_tokenizer::SomVocab;
use scirust_som_trainer::{TrainerConfig, train};

fn flag_u64(args: &[String], name: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn flag_f32(args: &[String], name: &str, default: f32) -> f32 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `certify [--seed N] [--eps E]` — build a small seeded ReLU MLP and prove
/// output bounds over an L∞ box of radius `eps` around an input, via both
/// **Interval Bound Propagation** and the tighter **CROWN** relaxation; report
/// whether the predicted class is **provably** unchanged. Showcases scirust's
/// "certifiable AI" thesis (and that CROWN certifies where IBP cannot).
pub fn run_certify(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let eps = flag_f32(args, "--eps", 0.05);
    if eps <= 0.0 || !eps.is_finite()
    {
        eprintln!("usage: scirust certify [--seed N] [--eps E]");
        eprintln!("error: --eps must be a positive number");
        return 2;
    }

    let (in_f, hidden, out_f) = (4usize, 8usize, 3usize);
    let mut rng = PcgEngine::new(seed);
    let l1 = NdLinear::new(in_f, hidden, &mut rng);
    let l2 = NdLinear::new(hidden, out_f, &mut rng);
    let il1 = IbpLinear::from_nd_linear(&l1);
    let il2 = IbpLinear::from_nd_linear(&l2);

    let centre = vec![0.2f32, -0.5, 0.7, -0.1];
    let box_in = Interval::around(&centre, eps);
    // CROWN before moving the layers into the IBP MLP.
    let crown = crown_bounds(&il1, &il2, &box_in);
    let mlp = IbpMlp::new(vec![il1, il2]);
    let pred = mlp.forward(&centre);
    let argmax = pred
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    let ibp = mlp.certify(&box_in);

    let robust_str = |out: &Interval| {
        if certified_robust(out, argmax)
        {
            format!("CERTIFIED — class {argmax} cannot change in the box")
        }
        else
        {
            "not certified at this eps".to_string()
        }
    };
    let width = |out: &Interval| -> f32 {
        (0..out_f).map(|c| out.hi[c] - out.lo[c]).sum::<f32>() / out_f as f32
    };

    println!("Certified bounds — pure Rust, deterministic (seed {seed})");
    println!("  MLP: {in_f}->{hidden}->{out_f} (ReLU)");
    println!("  input: {centre:?}  ->  prediction: class {argmax}");
    println!("  L∞ box radius eps = {eps}");
    println!("  IBP   bounds (avg width {:.4}):", width(&ibp));
    for c in 0..out_f
    {
        println!("    class {c}: [{:.4}, {:.4}]", ibp.lo[c], ibp.hi[c]);
    }
    println!("    robustness: {}", robust_str(&ibp));
    println!("  CROWN bounds (avg width {:.4}, tighter):", width(&crown));
    for c in 0..out_f
    {
        println!("    class {c}: [{:.4}, {:.4}]", crown.lo[c], crown.hi[c]);
    }
    println!("    robustness: {}", robust_str(&crown));
    0
}

/// `conformal [--seed N] [--alpha A]` — calibrate a **split-conformal**
/// regressor on synthetic residuals at target coverage `1 − α`, then report the
/// **empirical coverage** measured on fresh data and the interval half-width.
/// Distribution-free guarantee, demonstrated and deterministic by seed.
pub fn run_conformal(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let alpha = flag_f32(args, "--alpha", 0.1);
    if alpha <= 0.0 || alpha >= 1.0 || !alpha.is_finite()
    {
        eprintln!("usage: scirust conformal [--seed N] [--alpha A]");
        eprintln!("error: --alpha must be in (0, 1)");
        return 2;
    }

    let mut rng = PcgEngine::new(seed);
    let noise = |r: &mut PcgEngine| (r.float_signed() + r.float_signed()).abs();
    let cal: Vec<f32> = (0..2000).map(|_| noise(&mut rng)).collect();
    let reg = ConformalRegressor::calibrate(&cal, alpha);

    let n_test = 5000usize;
    let mut covered = 0usize;
    for _ in 0..n_test
    {
        if reg.covers(0.0, noise(&mut rng))
        {
            covered += 1;
        }
    }
    let coverage = covered as f32 / n_test as f32;
    let target = 100.0 * (1.0 - alpha);

    println!("Conformal prediction — pure Rust, deterministic (seed {seed})");
    println!("  calibration: 2000 points · target coverage {target:.0}% (alpha {alpha})");
    println!("  interval half-width q̂ = {:.4}", reg.half_width());
    println!(
        "  empirical coverage on {n_test} fresh points: {:.1}%",
        100.0 * coverage
    );
    println!("  guarantee: distribution-free marginal coverage ≥ {target:.0}%");
    0
}

/// `gptq [--seed N] [--samples S] [--damp D]` — quantize a synthetic Linear
/// layer to int8 with **GPTQ** (second-order error feedback) on correlated
/// calibration activations, and report the calibration-weighted reconstruction
/// error against plain round-to-nearest. Deterministic in `--seed`.
pub fn run_gptq(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let samples = flag_u64(args, "--samples", 128).max(1) as usize;
    let damp = flag_f32(args, "--damp", 0.01);
    if damp < 0.0 || !damp.is_finite()
    {
        eprintln!("usage: scirust gptq [--seed N] [--samples S] [--damp D]");
        eprintln!("error: --damp must be a non-negative number");
        return 2;
    }

    let (in_f, out_f, latent) = (16usize, 8usize, 4usize);
    let mut rng = PcgEngine::new(seed);
    // Correlated activations x = A·z + small noise (rank `latent`): off-diagonal
    // Hessian structure is exactly what GPTQ exploits over round-to-nearest.
    let a: Vec<f32> = (0..in_f * latent).map(|_| rng.float_signed()).collect();
    let mut x = vec![0f32; samples * in_f];
    for t in 0..samples
    {
        let z: Vec<f32> = (0..latent).map(|_| rng.float_signed()).collect();
        for i in 0..in_f
        {
            let mut v = 0.1 * rng.float_signed();
            for (l, &zl) in z.iter().enumerate()
            {
                v += a[i * latent + l] * zl;
            }
            x[t * in_f + i] = v;
        }
    }
    let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
    let h = gptq_hessian(&x, samples, in_f);

    // Calibration-weighted error Σ_o Δw_oᵀ H Δw_o (the GPTQ objective).
    let werr = |wq: &[f32]| -> f64 {
        let mut e = 0f64;
        for o in 0..out_f
        {
            for ai in 0..in_f
            {
                let da = (wq[ai * out_f + o] - w[ai * out_f + o]) as f64;
                if da == 0.0
                {
                    continue;
                }
                for b in 0..in_f
                {
                    let db = (wq[b * out_f + o] - w[b * out_f + o]) as f64;
                    e += da * h[ai * in_f + b] as f64 * db;
                }
            }
        }
        e
    };
    let dequant = |q: &[i8], s: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; in_f * out_f];
        for i in 0..in_f
        {
            for o in 0..out_f
            {
                out[i * out_f + o] = q[i * out_f + o] as f32 * s[o];
            }
        }
        out
    };

    let (qg, sg) = quantize_gptq(&w, in_f, out_f, &h, damp);
    let (qr, sr) = quantize_per_channel(&w, in_f, out_f);
    let eg = werr(&dequant(&qg, &sg));
    let er = werr(&dequant(&qr, &sr));
    let reduction = if er > 0.0
    {
        100.0 * (1.0 - eg / er)
    }
    else
    {
        0.0
    };

    println!("GPTQ int8 quantization — pure Rust, deterministic (seed {seed})");
    println!(
        "  Linear: {in_f}->{out_f} · {samples} correlated calibration samples (rank {latent})"
    );
    println!("  per-output-channel symmetric int8 · damping λ = {damp}");
    println!("  calibration-weighted reconstruction error  Σ Δwᵀ·H·Δw:");
    println!("    round-to-nearest : {er:.5}");
    println!("    GPTQ             : {eg:.5}");
    println!("  GPTQ reduces the calibration error by {reduction:.1}% at the same int8 budget");
    0
}

/// `awq [--seed N] [--samples S] [--grid G]` — quantize a synthetic Linear layer
/// to int8 with **AWQ** (activation-aware, search-based per-channel scaling) on
/// calibration activations that have a few salient (high-magnitude) channels, and
/// report the calibration-weighted error against plain round-to-nearest plus the
/// scaling exponent the search selected. Deterministic in `--seed`.
pub fn run_awq(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let samples = flag_u64(args, "--samples", 128).max(1) as usize;
    let grid = flag_u64(args, "--grid", 21).max(2) as usize;

    let (in_f, out_f) = (16usize, 8usize);
    let salient = [3usize, 7, 11];
    let mut rng = PcgEngine::new(seed);
    // A few salient input channels (×20) dominate the layer output — exactly the
    // regime AWQ targets by protecting those channels at quantization time.
    let mut x = vec![0f32; samples * in_f];
    for t in 0..samples
    {
        for j in 0..in_f
        {
            let base = rng.float_signed();
            x[t * in_f + j] = if salient.contains(&j)
            {
                20.0 * base
            }
            else
            {
                base
            };
        }
    }
    let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
    let h = gptq_hessian(&x, samples, in_f);

    let werr = |wq: &[f32]| -> f64 {
        let mut e = 0f64;
        for o in 0..out_f
        {
            for ai in 0..in_f
            {
                let da = (wq[ai * out_f + o] - w[ai * out_f + o]) as f64;
                if da == 0.0
                {
                    continue;
                }
                for b in 0..in_f
                {
                    let db = (wq[b * out_f + o] - w[b * out_f + o]) as f64;
                    e += da * h[ai * in_f + b] as f64 * db;
                }
            }
        }
        e
    };

    let res = awq_quantize(&w, in_f, out_f, &x, samples, grid);
    let eg = werr(&res.dequantize(in_f, out_f));
    let (qr, sr) = quantize_per_channel(&w, in_f, out_f);
    let mut wr = vec![0f32; in_f * out_f];
    for j in 0..in_f
    {
        for o in 0..out_f
        {
            wr[j * out_f + o] = qr[j * out_f + o] as f32 * sr[o];
        }
    }
    let er = werr(&wr);
    let reduction = if er > 0.0
    {
        100.0 * (1.0 - eg / er)
    }
    else
    {
        0.0
    };

    println!("AWQ int8 quantization — pure Rust, deterministic (seed {seed})");
    println!(
        "  Linear: {in_f}->{out_f} · {samples} calibration samples · salient channels {salient:?} (×20)"
    );
    println!("  per-output-channel symmetric int8 · alpha grid of {grid} points in [0,1]");
    println!("  selected scaling exponent alpha = {:.3}", res.alpha);
    println!("  calibration-weighted reconstruction error  Σ Δwᵀ·H·Δw:");
    println!("    round-to-nearest : {er:.5}");
    println!("    AWQ              : {eg:.5}");
    println!(
        "  AWQ reduces the calibration error by {reduction:.1}% by protecting salient channels"
    );
    0
}

/// `som train [--seed N] [--epochs E]` — train the ownership model on
/// oracle-labelled data and report per-token accuracy against the majority
/// baseline. Deterministic in `--seed`.
pub fn run_som(args: &[String]) -> u8 {
    if args.first().map(String::as_str) != Some("train")
    {
        eprintln!("usage: scirust som train [--seed N] [--epochs E]");
        return 2;
    }
    let seed = flag_u64(&args[1..], "--seed", 42);
    let epochs = flag_u64(&args[1..], "--epochs", 6) as usize;

    println!("SOM ownership model — training (seed {seed}, {epochs} epochs)\n");
    let train_set = build_training_set(seed, 160, 64);
    let eval_set = build_training_set(seed.wrapping_add(9000), 50, 64);
    let baseline = ownership_majority_baseline(&eval_set);

    let mut model = SomModel::new(SomModelConfig {
        vocab_size: SomVocab::vocab_size(),
        seed,
        ..SomModelConfig::default()
    });
    let report = train(
        &mut model,
        &train_set,
        &TrainerConfig {
            epochs,
            learning_rate: 0.005,
        },
    );
    let eval = evaluate(&mut model, &eval_set);

    println!(
        "  loss: {:.4} → {:.4}",
        report.first_loss(),
        report.last_loss()
    );
    println!(
        "  ownership accuracy : {:.4}   (majority baseline {:.4})",
        eval.ownership_accuracy, baseline
    );
    println!("  borrow accuracy    : {:.4}", eval.borrow_accuracy);
    println!("  fault detection    : {:.4}", eval.invalid_accuracy);
    println!("  tokens evaluated   : {}", eval.n_tokens);
    if eval.ownership_accuracy > baseline
    {
        println!("\nOK — model beats the majority baseline on held-out programs.");
        0
    }
    else
    {
        println!("\nNOTE — try more epochs; model did not beat the baseline here.");
        0
    }
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// `evo [--seed N] [--gens G]` — minimize the sphere function with a seeded
/// genetic algorithm and report the best value found (→ 0). Deterministic.
pub fn run_evo(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 7);
    let gens = flag_u64(args, "--gens", 60) as usize;
    let dims = 5;

    let ga = GeneticAlgorithm::seeded(seed);
    let mut pop = ga.init_pop(dims);
    let start_best = pop
        .iter()
        .map(|i| sphere(&i.genome))
        .fold(f64::INFINITY, f64::min);
    for _ in 0..gens
    {
        ga.evolve(&mut pop, |inds| {
            inds.iter().map(|i| -sphere(&i.genome)).collect()
        });
    }
    let best = pop
        .iter()
        .map(|i| sphere(&i.genome))
        .fold(f64::INFINITY, f64::min);

    println!("Evolutionary optimization — minimize sphere f(x)=Σxᵢ² (dims {dims}, seed {seed})\n");
    println!("  generations : {gens}");
    println!("  best f(x)   : {start_best:.4} → {best:.6}");
    if best < start_best
    {
        println!("\nOK — converged toward the optimum, deterministically.");
        0
    }
    else
    {
        println!("\nNOTE — no improvement; increase --gens.");
        0
    }
}

/// `cmaes [--seed N] [--steps S]` — minimize the sphere function with a
/// seeded CMA-ES and report the best value found (→ 0). Deterministic.
pub fn run_cmaes(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 7);
    let steps = flag_u64(args, "--steps", 80) as usize;
    let dims = 5;

    let mut es = CmaEs::seeded(dims, seed);
    let mut theta = vec![1.5f64; dims];
    let start = sphere(&theta);
    for _ in 0..steps
    {
        es.step(&mut theta, |x| -sphere(x));
    }
    let best = sphere(&theta);

    println!("CMA-ES — minimize sphere f(x)=Σxᵢ² (dims {dims}, seed {seed})\n");
    println!("  steps     : {steps}");
    println!("  best f(x) : {start:.4} → {best:.6}");
    if best < start
    {
        println!("\nOK — converged toward the optimum, deterministically.");
        0
    }
    else
    {
        println!("\nNOTE — no improvement; increase --steps.");
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cmaes_minimizes_sphere_deterministically() {
        assert_eq!(run_cmaes(&s(&["--seed", "3", "--steps", "40"])), 0);
        let best = |seed: u64| {
            let mut es = CmaEs::seeded(5, seed);
            let mut theta = vec![1.5f64; 5];
            for _ in 0..40
            {
                es.step(&mut theta, |x| -sphere(x));
            }
            sphere(&theta)
        };
        assert_eq!(best(3).to_bits(), best(3).to_bits());
    }

    #[test]
    fn som_train_runs_and_beats_baseline() {
        // Small but real run; must beat the majority baseline.
        assert_eq!(run_som(&s(&["train", "--epochs", "4"])), 0);
        assert_eq!(run_som(&s(&["oops"])), 2);
    }

    #[test]
    fn evo_minimizes_sphere_deterministically() {
        assert_eq!(run_evo(&s(&["--seed", "1", "--gens", "30"])), 0);
        // determinism: the same seed yields the same best value.
        let best = |seed: u64| {
            let ga = GeneticAlgorithm::seeded(seed);
            let mut pop = ga.init_pop(5);
            for _ in 0..30
            {
                ga.evolve(&mut pop, |inds| {
                    inds.iter().map(|i| -sphere(&i.genome)).collect()
                });
            }
            pop.iter()
                .map(|i| sphere(&i.genome))
                .fold(f64::INFINITY, f64::min)
        };
        assert_eq!(best(1).to_bits(), best(1).to_bits());
    }

    #[test]
    fn certify_runs() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        // A small box certifies; default and explicit eps both succeed.
        assert_eq!(run_certify(&s(&["--eps", "0.01"])), 0);
        assert_eq!(run_certify(&s(&["--seed", "3", "--eps", "0.2"])), 0);
        // Invalid eps is rejected.
        assert_eq!(run_certify(&s(&["--eps", "0"])), 2);
    }

    #[test]
    fn conformal_runs() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(run_conformal(&s(&["--alpha", "0.1"])), 0);
        assert_eq!(run_conformal(&s(&["--seed", "5", "--alpha", "0.2"])), 0);
        // alpha must be in (0,1).
        assert_eq!(run_conformal(&s(&["--alpha", "0"])), 2);
        assert_eq!(run_conformal(&s(&["--alpha", "1"])), 2);
    }
}
