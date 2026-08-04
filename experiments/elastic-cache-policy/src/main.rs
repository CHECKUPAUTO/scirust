use scirust_cache_policy::{
    DiscoveryConfig, compare_on_holdout, discover_linear_policy, discover_symbolic_surrogate,
    read_trace_csv, split_by_trajectory, synthetic_trace, write_trace_csv,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    trace: Option<PathBuf>,
    write_synthetic: Option<PathBuf>,
    seed: u64,
    steps: usize,
    max_quality_loss: f64,
    trajectories: usize,
    rows_per_trajectory: usize,
    symbolic: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            trace: None,
            write_synthetic: None,
            seed: 20_260_804,
            steps: 600,
            max_quality_loss: 0.05,
            trajectories: 400,
            rows_per_trajectory: 64,
            symbolic: false,
        }
    }
}

fn usage() {
    eprintln!(
        "usage: scirust-cache-policy [--trace FILE] [--write-synthetic FILE] \\
         [--seed N] [--steps N] [--max-quality-loss X] \\
         [--trajectories N] [--rows-per-trajectory N] [--symbolic]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next()
    {
        match arg.as_str()
        {
            "--trace" =>
            {
                parsed.trace = Some(PathBuf::from(args.next().ok_or("--trace requires a path")?));
            },
            "--write-synthetic" =>
            {
                parsed.write_synthetic = Some(PathBuf::from(
                    args.next().ok_or("--write-synthetic requires a path")?,
                ));
            },
            "--seed" =>
            {
                parsed.seed = args
                    .next()
                    .ok_or("--seed requires an integer")?
                    .parse()
                    .map_err(|error| format!("invalid --seed: {error}"))?;
            },
            "--steps" =>
            {
                parsed.steps = args
                    .next()
                    .ok_or("--steps requires an integer")?
                    .parse()
                    .map_err(|error| format!("invalid --steps: {error}"))?;
            },
            "--max-quality-loss" =>
            {
                parsed.max_quality_loss = args
                    .next()
                    .ok_or("--max-quality-loss requires a number")?
                    .parse()
                    .map_err(|error| format!("invalid --max-quality-loss: {error}"))?;
            },
            "--trajectories" =>
            {
                parsed.trajectories = args
                    .next()
                    .ok_or("--trajectories requires an integer")?
                    .parse()
                    .map_err(|error| format!("invalid --trajectories: {error}"))?;
            },
            "--rows-per-trajectory" =>
            {
                parsed.rows_per_trajectory = args
                    .next()
                    .ok_or("--rows-per-trajectory requires an integer")?
                    .parse()
                    .map_err(|error| format!("invalid --rows-per-trajectory: {error}"))?;
            },
            "--symbolic" => parsed.symbolic = true,
            "-h" | "--help" =>
            {
                usage();
                std::process::exit(0);
            },
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(parsed)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (rows, source) = if let Some(path) = &args.trace
    {
        (read_trace_csv(path)?, format!("trace:{}", path.display()))
    }
    else
    {
        (
            synthetic_trace(args.trajectories, args.rows_per_trajectory, args.seed),
            "synthetic-oracle".to_string(),
        )
    };

    if let Some(path) = &args.write_synthetic
    {
        write_trace_csv(path, &rows)?;
    }

    let (training, validation, test) = split_by_trajectory(&rows);
    if training.is_empty() || validation.is_empty() || test.is_empty()
    {
        return Err(
            "trajectory split produced an empty partition; provide at least five trajectory IDs"
                .into(),
        );
    }

    let config = DiscoveryConfig {
        seed: args.seed,
        steps: args.steps,
        max_quality_loss: args.max_quality_loss,
        ..DiscoveryConfig::default()
    };
    let calibration_budget = config.max_quality_loss * config.calibration_budget_fraction;
    let result = discover_linear_policy(&training, &validation, config)?;
    let comparison = compare_on_holdout(&result.policy, &test, args.max_quality_loss);

    println!("source={source}");
    println!(
        "rows total={} train={} validation={} test={}",
        rows.len(),
        training.len(),
        validation.len(),
        test.len()
    );
    println!("seed={} steps={}", args.seed, args.steps);
    println!(
        "quality_budget={:.8} calibration_budget={:.8}",
        args.max_quality_loss, calibration_budget
    );
    println!("weights={:?}", result.policy.weights);
    println!("threshold={:.17}", result.policy.threshold);
    println!(
        "validation quality_loss={:.8} compute={:.8} refresh_rate={:.8}",
        result.validation.quality_loss_fraction,
        result.validation.compute_fraction,
        result.validation.refresh_rate
    );
    println!(
        "test learned quality_loss={:.8} compute={:.8} refresh_rate={:.8}",
        comparison.learned.quality_loss_fraction,
        comparison.learned.compute_fraction,
        comparison.learned.refresh_rate
    );
    println!(
        "test best_gamma gamma={:.8} quality_loss={:.8} compute={:.8} refresh_rate={:.8}",
        comparison.fixed_gamma.gamma,
        comparison.fixed_gamma.metrics.quality_loss_fraction,
        comparison.fixed_gamma.metrics.compute_fraction,
        comparison.fixed_gamma.metrics.refresh_rate
    );
    println!(
        "learned_meets_budget={} fixed_gamma_meets_budget={} constrained_better={}",
        comparison.learned_meets_budget,
        comparison.fixed_gamma_meets_budget,
        comparison.constrained_better
    );
    println!(
        "relative_compute_improvement={:.8} pareto_dominates={}",
        comparison.relative_compute_improvement, comparison.pareto_dominates
    );

    if args.trace.is_none()
    {
        println!("evidence_scope=synthetic-only; no claim about LLaDA or Dream");
    }

    if args.symbolic
    {
        let capped = &training[..training.len().min(4_096)];
        let seeds = [
            args.seed,
            args.seed.wrapping_add(1),
            args.seed.wrapping_add(2),
        ];
        let front = discover_symbolic_surrogate(capped, &seeds, 120, 14, 25, 24);
        println!("symbolic_front_size={}", front.len());
        for candidate in front.iter().take(12)
        {
            println!(
                "symbolic size={} mse={:.10} expr={}",
                candidate.size, candidate.mse, candidate.expression
            );
        }
    }

    Ok(())
}

fn main() {
    if let Err(error) = run()
    {
        usage();
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
