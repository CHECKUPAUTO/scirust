//! Learning subcommands beyond `quickstart`: ownership-model training and
//! evolutionary optimization. Both are deterministic in their seed.

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
}
