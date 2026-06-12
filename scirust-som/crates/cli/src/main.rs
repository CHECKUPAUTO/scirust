//! `som-analyze` — run the SOM ownership oracle on a real Rust file.
//!
//! Usage:
//!   som-analyze <file.rs>
//!
//! Parses real Rust with the `syn`-based frontend, lowers the supported
//! subset to the ownership IR, runs the deterministic oracle, and prints a
//! per-token ownership/borrow/fault table plus diagnostics. Exits with
//! status 1 when the oracle reports at least one ownership fault, so it can
//! be used as a check in scripts.

use std::process::ExitCode;

use scirust_som_frontend::lower_str;
use scirust_som_symbolic::OwnershipOracle;
use scirust_som_visualizer::render_markdown;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let path = match args.next()
    {
        Some(p) => p,
        None =>
        {
            eprintln!("usage: som-analyze <file.rs>");
            return ExitCode::from(2);
        },
    };

    let src = match std::fs::read_to_string(&path)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::from(2);
        },
    };

    let lowered = match lower_str(&src)
    {
        Ok(l) => l,
        Err(e) =>
        {
            eprintln!("error: {path} is not valid Rust: {e}");
            return ExitCode::from(2);
        },
    };

    let analysis = OwnershipOracle::new().analyze(&lowered.ast);

    println!("# SOM ownership analysis — {path}\n");
    println!("{}", render_markdown(&analysis));

    if !lowered.approximations.is_empty()
    {
        println!("\nApproximations applied:");
        for a in &lowered.approximations
        {
            println!("- {a}");
        }
    }
    if !lowered.unsupported.is_empty()
    {
        println!("\nConstructs skipped (not modelled):");
        for u in &lowered.unsupported
        {
            println!("- {u}");
        }
    }

    let faults = analysis.diagnostics.len();
    println!(
        "\nSummary: {} token(s), {} ownership fault(s).",
        analysis.tokens.len(),
        faults
    );

    if faults > 0
    {
        ExitCode::FAILURE
    }
    else
    {
        ExitCode::SUCCESS
    }
}
