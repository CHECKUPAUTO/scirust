//! Deterministic Phase 5 sparse residual-channel harness.

use scirust_elastic_latent_kv::phase5::{run_standard_suite, suite_to_csv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reports = run_standard_suite()?;
    print!("{}", suite_to_csv(&reports));
    Ok(())
}
