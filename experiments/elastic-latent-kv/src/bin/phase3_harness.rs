//! Deterministic CSV exporter for the Elastic Latent KV Phase 3 suite.

use scirust_elastic_latent_kv::phase3::{run_standard_suite, suite_to_csv};

fn main() {
    match run_standard_suite()
    {
        Ok(reports) => print!("{}", suite_to_csv(&reports)),
        Err(error) =>
        {
            eprintln!("phase 3 harness failed: {error}");
            std::process::exit(1);
        },
    }
}
