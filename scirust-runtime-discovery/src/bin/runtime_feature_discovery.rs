use scirust_runtime_discovery::{DiscoveryRequest, generate_catalog};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    request: PathBuf,
    output: PathBuf,
}

fn usage() {
    eprintln!("usage: runtime-feature-discovery --request REQUEST.json --output CATALOG.json");
}

fn parse_args() -> Result<Args, String> {
    let mut request = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--request" => {
                request = Some(PathBuf::from(
                    args.next().ok_or("--request requires a path")?,
                ));
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Args {
        request: request.ok_or("--request is required")?,
        output: output.ok_or("--output is required")?,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let encoded = std::fs::read_to_string(&args.request).map_err(|error| {
        format!(
            "cannot read discovery request {}: {error}",
            args.request.display()
        )
    })?;
    let request: DiscoveryRequest = serde_json::from_str(&encoded).map_err(|error| {
        format!(
            "invalid discovery request {}: {error}",
            args.request.display()
        )
    })?;
    let catalog = generate_catalog(&request)?;
    let output = serde_json::to_string_pretty(&catalog)
        .map_err(|error| format!("cannot serialize feature catalog: {error}"))?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!("cannot create output directory {}: {error}", parent.display())
        })?;
    }
    std::fs::write(&args.output, format!("{output}\n")).map_err(|error| {
        format!("cannot write feature catalog {}: {error}", args.output.display())
    })?;
    println!("{output}");
    println!("\nCatalog: {}", args.output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        usage();
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
