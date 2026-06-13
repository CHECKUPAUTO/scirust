//! `scirust-verify` — emit/verify inference proof certificates.
//!
//! Thin wrapper over `scirust_runtime::proofcli` (the same logic backs the
//! unified `scirust verify` subcommand). See that module for the format.
//!
//!   scirust-verify emit   <model.qsr1> <out.proof> `[batch]` `[seeds...]`
//!   scirust-verify verify <bundle.proof> <model.qsr1>

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(scirust_runtime::proofcli::run(&args))
}
