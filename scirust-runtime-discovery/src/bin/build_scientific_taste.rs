use scirust_runtime_discovery::{
    DatasetEvaluationReport, ProposalReview, build_sciagent_research_brief,
    derive_scientific_taste,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    review: PathBuf,
    evaluation: PathBuf,
    taste_output: PathBuf,
    brief_output: PathBuf,
    maximum_targets: usize,
}

fn usage() {
    eprintln!(
        "usage: build-scientific-taste --review FILE --evaluation FILE \\\n         --taste-output FILE --brief-output FILE [--maximum-targets N]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut review = None;
    let mut evaluation = None;
    let mut taste_output = None;
    let mut brief_output = None;
    let mut maximum_targets = 16usize;
    let mut args = std::env::args().skip(1);

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--review" => review = Some(PathBuf::from(next_value(&mut args, "--review")?)),
            "--evaluation" => {
                evaluation = Some(PathBuf::from(next_value(&mut args, "--evaluation")?));
            }
            "--taste-output" => {
                taste_output = Some(PathBuf::from(next_value(&mut args, "--taste-output")?));
            }
            "--brief-output" => {
                brief_output = Some(PathBuf::from(next_value(&mut args, "--brief-output")?));
            }
            "--maximum-targets" => {
                maximum_targets = next_value(&mut args, "--maximum-targets")?
                    .parse()
                    .map_err(|error| format!("invalid --maximum-targets: {error}"))?;
            }
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    if maximum_targets == 0 {
        return Err("--maximum-targets must be positive".to_string());
    }

    Ok(Args {
        review: review.ok_or("--review is required")?,
        evaluation: evaluation.ok_or("--evaluation is required")?,
        taste_output: taste_output.ok_or("--taste-output is required")?,
        brief_output: brief_output.ok_or("--brief-output is required")?,
        maximum_targets,
    })
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let review: ProposalReview = read_json(&args.review)?;
    let evaluation: DatasetEvaluationReport = read_json(&args.evaluation)?;

    if review.experiment_id != evaluation.experiment_id {
        return Err(format!(
            "experiment mismatch: review={} evaluation={}",
            review.experiment_id, evaluation.experiment_id
        ));
    }

    let taste = derive_scientific_taste(&evaluation);
    let brief = build_sciagent_research_brief(&review, &evaluation, args.maximum_targets);
    write_json(&args.taste_output, &taste)?;
    write_json(&args.brief_output, &brief)?;

    println!(
        "scientific taste: {} events -> {}",
        taste.events.len(),
        args.taste_output.display()
    );
    println!(
        "SciAgent brief: {} targets -> {}",
        brief.targets.len(),
        args.brief_output.display()
    );
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(value)
        .map_err(|error| format!("cannot serialize output: {error}"))?;
    std::fs::write(path, format!("{encoded}\n"))
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn main() {
    if let Err(error) = run() {
        usage();
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
