//! `som-analyze` — ownership analysis of a real Rust file.
//!
//! Usage:
//!   som-analyze [--sarif] <file.rs>
//!
//! Prints a per-token ownership/borrow/fault report (markdown) or a SARIF
//! 2.1.0 document with `--sarif` (for CI code-scanning upload). Exit codes:
//! 0 = no fault, 1 = at least one ownership fault, 2 = usage/IO/syntax.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(scirust_som_cli::run(&args, "som-analyze"))
}
