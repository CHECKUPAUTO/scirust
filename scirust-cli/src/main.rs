//! `scirust` — unified command-line entry point. See `scirust help`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(scirust_cli::run(&args))
}
