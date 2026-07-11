//! `finaudit` — consolidated audit / baseline tamper-check for scirust-finmigrate.
//!
//! Recomputes every committed Golden Baseline's SHA-256, verifies it against the
//! recorded digest, prints a consolidated audit report, and exits non-zero if any
//! baseline fails to verify. Intended for CI and for an auditor's one-shot check.
//!
//! Run: `cargo run -p scirust-finmigrate --bin finaudit`

use scirust_finmigrate::audit::{all_ok, audit_units, render_report};
use std::process::ExitCode;

fn main() -> ExitCode {
    let units = audit_units();
    print!("{}", render_report(&units));
    if all_ok(&units)
    {
        ExitCode::SUCCESS
    }
    else
    {
        ExitCode::FAILURE
    }
}
