//! Deterministic CSV exporter for the Phase 1 baseline suite.

use scirust_elastic_latent_kv::phase1::{run_standard_suite, suite_to_csv};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run_standard_suite()
    {
        Ok(reports) =>
        {
            print!("{}", suite_to_csv(&reports));
            ExitCode::SUCCESS
        },
        Err(error) =>
        {
            eprintln!("phase1 harness failed: {error}");
            ExitCode::FAILURE
        },
    }
}
