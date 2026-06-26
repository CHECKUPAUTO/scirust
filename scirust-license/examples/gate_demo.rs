//! End-to-end demo: a vendor issues a license unlocking a chosen set of
//! modules, and a runtime gates feature access on the verified entitlements.
//!
//! Run with: `cargo run -p scirust-license --example gate_demo`

use scirust_license::{Entitlements, License, LicenseError, Module, Vendor, verify_license};

/// A stand-in for a real licensed feature: navigation fusion. In a production
/// crate this guard would sit at the entry point of `scirust-nav`.
fn run_navigation(ent: &Entitlements) -> Result<String, LicenseError> {
    ent.require(Module::Navigation)?;
    Ok("navigation fix computed".to_string())
}

/// Another feature, deliberately *not* in our license, to show the gate biting.
fn run_water_analytics(ent: &Entitlements) -> Result<String, LicenseError> {
    ent.require(Module::Water)?;
    Ok("water-network analysis computed".to_string())
}

fn main() {
    // --- Vendor side (offline; holds the secret seed) ----------------------
    let vendor = Vendor::from_seed(&[42u8; 32], 8);
    let root = vendor.root(); // the only thing the runtime embeds
    println!(
        "vendor public root: {}",
        scirust_license::hashsig::hex_encode(&root)
    );

    // Issue a license for "Acme Robotics" unlocking Navigation + Control,
    // valid in the window [1000, 2000].
    let license = License::new(
        "Acme Robotics",
        "L-2026-001",
        [Module::Navigation, Module::Control],
        1_000,
        Some(2_000),
    );
    let signed = vendor.issue_with_leaf(license, 0);
    println!("\nissued license:\n{}", signed.to_json());

    // --- Runtime side (ships to the customer; holds only `root`) -----------
    let now = 1_500;
    let ent = verify_license(&signed, &root, now).expect("license should verify");
    println!("\nverified entitlements for: {}", ent.licensee());

    match run_navigation(&ent)
    {
        Ok(msg) => println!("  navigation -> OK: {msg}"),
        Err(e) => println!("  navigation -> blocked: {e}"),
    }
    match run_water_analytics(&ent)
    {
        Ok(msg) => println!("  water      -> OK: {msg}"),
        Err(e) => println!("  water      -> blocked: {e}"),
    }

    // --- Tamper attempt: customer edits the module list --------------------
    let mut forged = signed.clone();
    forged.license.modules.push(Module::Water);
    match verify_license(&forged, &root, now)
    {
        Ok(_) => println!("\nFORGERY ACCEPTED — this must never print"),
        Err(e) => println!("\ntamper attempt correctly rejected: {e}"),
    }
}
