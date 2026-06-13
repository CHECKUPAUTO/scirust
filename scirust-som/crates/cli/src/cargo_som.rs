//! `cargo som` — the SOM ownership analyzer as a cargo subcommand.
//!
//! Install with `cargo install --path scirust-som/crates/cli`, then:
//!   cargo som [--sarif] <file.rs>
//!
//! Cargo invokes this binary as `cargo-som som <args…>`; the injected
//! subcommand word is skipped before delegating to the shared driver.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("som")
    {
        args.remove(0);
    }
    ExitCode::from(scirust_som_cli::run(&args, "cargo som"))
}
