use scirust_runtime_discovery::{ProposalReview, evaluate_review_on_jsonl};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    review: PathBuf,
    dataset: PathBuf,
    output: PathBuf,
}

fn usage() {
    eprintln!(
        "usage: evaluate-feature-proposals --review FILE --dataset FILE --output FILE"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut review = None;
    let mut dataset = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--review" => review = Some(PathBuf::from(args.next().ok_or("--review requires a path")?)),
            "--dataset" => dataset = Some(PathBuf::from(args.next().ok_or("--dataset requires a path")?)),
            "--output" => output = Some(PathBuf::from(args.next().ok_or("--output requires a path")?)),
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            },
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Args {
        review: review.ok_or("--review is required")?,
        dataset: dataset.ok_or("--dataset is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let encoded = std::fs::read_to_string(&args.review)
        .map_err(|error| format!("cannot read review {}: {error}", args.review.display()))?;
    let review: ProposalReview = serde_json::from_str(&encoded)
        .map_err(|error| format!("invalid review JSON: {error}"))?;
    let report = evaluate_review_on_jsonl(&review, &args.dataset)?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("cannot serialize evaluation report: {error}"))?;
    std::fs::write(&args.output, format!("{encoded}\n"))
        .map_err(|error| format!("cannot write {}: {error}", args.output.display()))?;
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
