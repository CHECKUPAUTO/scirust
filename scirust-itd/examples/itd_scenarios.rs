//! Reproduces the reference ITD entrypoint: runs the three canonical scenarios
//! on the published 161×161 / 401-step configuration and prints the two-axis
//! (intensity, structure) diagnostic, asserting the same invariants the
//! reference does.
//!
//! Run with: `cargo run -p scirust-itd --example itd_scenarios`

use scirust_itd::{Config, Scenario, SimConfig, simulate_canonical};

fn main() {
    let config = Config::default();
    let sim = SimConfig::default();

    let results: Vec<_> = [Scenario::Calm, Scenario::Coherent, Scenario::Multi]
        .into_iter()
        .map(|s| simulate_canonical(s, &config, &sim).expect("simulation failed"))
        .collect();

    println!("=== ITD SIMULATOR (SciRust port) ===");
    println!("structural length : {:.6}", sim.structural_length);
    for r in &results
    {
        println!();
        println!("scenario          : {}", r.name);
        println!("intensity index   : {:.12}", r.intensity_index);
        println!("structure index   : {:.12}", r.structure_index);
        println!("coupled diagnostic: {:.12}", r.coupled_index);
    }

    let calm = &results[0];
    let coherent = &results[1];
    let multi = &results[2];

    assert!(
        calm.intensity_index < 1.0e-20,
        "calm field must be quasi-irrotational"
    );
    assert!(
        calm.structure_index < 1.0e-20,
        "calm field must be structureless"
    );
    assert!(
        coherent.intensity_index > multi.intensity_index,
        "the coherent vortex must be the most intense"
    );
    assert!(
        multi.structure_index > coherent.structure_index,
        "the multi-vortex field must be the most structurally complex"
    );

    println!();
    println!("two-axis validation : PASSED");
}
