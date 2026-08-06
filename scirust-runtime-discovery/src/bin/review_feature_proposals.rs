use scirust_runtime_discovery::{
    DiscoveryRequest, FeatureCatalog, ProposalBatch, review_proposals, summarize_rejections,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    request: PathBuf,
    catalog: PathBuf,
    proposals: PathBuf,
    output: PathBuf,
}

fn usage() {
    eprintln!(
        "usage: review-feature-proposals --request FILE --catalog FILE \\\n         --proposals FILE --output FILE"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut request = None;
    let mut catalog = None;
    let mut proposals = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next()
    {
        match argument.as_str()
        {
            "--request" =>
            {
                request = Some(PathBuf::from(
                    args.next().ok_or("--request requires a path")?,
                ));
            },
            "--catalog" =>
            {
                catalog = Some(PathBuf::from(
                    args.next().ok_or("--catalog requires a path")?,
                ));
            },
            "--proposals" =>
            {
                proposals = Some(PathBuf::from(
                    args.next().ok_or("--proposals requires a path")?,
                ));
            },
            "--output" =>
            {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            },
            "-h" | "--help" =>
            {
                usage();
                std::process::exit(0);
            },
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    Ok(Args {
        request: request.ok_or("--request is required")?,
        catalog: catalog.ok_or("--catalog is required")?,
        proposals: proposals.ok_or("--proposals is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot decode {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let request: DiscoveryRequest = read_json(&args.request)?;
    request.validate()?;
    let catalog: FeatureCatalog = read_json(&args.catalog)?;
    catalog.validate()?;
    let proposals: ProposalBatch = read_json(&args.proposals)?;

    if request.experiment_id != catalog.experiment_id
        || request.experiment_id != proposals.experiment_id
    {
        return Err("request, catalog, and proposals must share experiment_id".to_string());
    }

    let existing_ids: Vec<String> = catalog
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.id.clone())
        .collect();
    let mut available_signals = request.available_signals.clone();
    available_signals.extend(request.base_features.iter().cloned());
    available_signals.sort();
    available_signals.dedup();

    let review = review_proposals(&proposals, &available_signals, &existing_ids)?;
    let encoded = serde_json::to_string_pretty(&review)
        .map_err(|error| format!("cannot serialize proposal review: {error}"))?;
    if let Some(parent) = args.output.parent()
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create output directory {}: {error}",
                parent.display()
            )
        })?;
    }
    std::fs::write(&args.output, format!("{encoded}\n")).map_err(|error| {
        format!(
            "cannot write proposal review {}: {error}",
            args.output.display()
        )
    })?;

    println!("{encoded}");
    println!("\nAccepted: {}", review.accepted.len());
    println!("Rejected: {}", review.rejected.len());
    for (reason, count) in summarize_rejections(&review)
    {
        println!("Rejected {count}: {reason}");
    }
    println!("Review: {}", args.output.display());
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
