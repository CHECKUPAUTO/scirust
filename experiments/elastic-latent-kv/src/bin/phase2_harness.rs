//! Emits the deterministic Phase 2 projection suite as CSV.

use scirust_elastic_latent_kv::phase2::{run_standard_suite, suite_to_csv};

fn main() {
    match run_standard_suite()
    {
        Ok(reports) => print!("{}", suite_to_csv(&reports)),
        Err(error) =>
        {
            eprintln!("phase2 harness failed: {error}");
            std::process::exit(1);
        },
    }
}
