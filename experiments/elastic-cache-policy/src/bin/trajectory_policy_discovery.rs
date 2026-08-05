use scirust_cache_policy::trajectory::{
    TrajectoryDiscoveryConfig, discover_trajectory_policy,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    dataset: PathBuf,
    output: PathBuf,
    config: TrajectoryDiscoveryConfig,
}

fn usage() {
    eprintln!(
        "usage: trajectory_policy_discovery --dataset FILE --output FILE \\
         [--seed N] [--crf-epochs N] [--crf-learning-rate X] \\
         [--crf-l2 X] [--nsga-population N] [--nsga-generations N] \\
         [--minimum-holdout-coverage X] [--no-symbolic]"
    );
}

fn parse_value<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))?
        .parse::<T>()
        .map_err(|error| format!("invalid {option}: {error}"))
}

fn parse_args() -> Result<Args, String> {
    let mut dataset = None;
    let mut output = None;
    let mut config = TrajectoryDiscoveryConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--dataset" => {
                dataset = Some(PathBuf::from(
                    args.next().ok_or("--dataset requires a path")?,
                ));
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            "--seed" => config.seed = parse_value(&mut args, "--seed")?,
            "--crf-epochs" => {
                config.crf_epochs = parse_value(&mut args, "--crf-epochs")?;
            }
            "--crf-learning-rate" => {
                config.crf_learning_rate =
                    parse_value(&mut args, "--crf-learning-rate")?;
            }
            "--crf-l2" => {
                config.crf_l2_penalty = parse_value(&mut args, "--crf-l2")?;
            }
            "--nsga-population" => {
                config.nsga_population = parse_value(&mut args, "--nsga-population")?;
            }
            "--nsga-generations" => {
                config.nsga_generations = parse_value(&mut args, "--nsga-generations")?;
            }
            "--minimum-holdout-coverage" => {
                config.minimum_holdout_coverage =
                    parse_value(&mut args, "--minimum-holdout-coverage")?;
            }
            "--no-symbolic" => config.symbolic = false,
            "--symbolic" => config.symbolic = true,
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Args {
        dataset: dataset.ok_or("--dataset is required")?,
        output: output.ok_or("--output is required")?,
        config,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let report = discover_trajectory_policy(&args.dataset, &args.config)?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!("cannot create output directory {}: {error}", parent.display())
        })?;
    }
    let encoded = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot serialize trajectory report: {error}"))?;
    std::fs::write(&args.output, format!("{encoded}\n")).map_err(|error| {
        format!("cannot write trajectory report {}: {error}", args.output.display())
    })?;
    println!("{encoded}");
    println!("\nRapport: {}", args.output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        usage();
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
