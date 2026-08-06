//! Deterministic Phase 6 quantization harness.

#[path = "../phase6.rs"]
mod phase6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reports = phase6::run_standard_suite()?;
    print!("{}", phase6::suite_to_csv(&reports));
    Ok(())
}
