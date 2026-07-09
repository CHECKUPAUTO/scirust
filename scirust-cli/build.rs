//! Build script: expose the *workspace* version and the git commit to the CLI
//! so `scirust --version` reports the real project version (0.14.x) instead of
//! this helper crate's own `0.1.0`, plus a short SHA for traceable bug reports.
//!
//! Deliberately no wall-clock build date: this project's headline property is
//! bit-exact reproducibility, and an embedded timestamp would make the binary
//! non-reproducible. Version + commit SHA are both reproducible.

use std::path::Path;
use std::process::Command;

fn main() {
    // --- Workspace version, parsed from the root Cargo.toml's [package] ---
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let root_manifest = Path::new(&manifest_dir).join("..").join("Cargo.toml");
    let version = std::fs::read_to_string(&root_manifest)
        .ok()
        .and_then(|s| parse_package_version(&s))
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    println!("cargo:rustc-env=SCIRUST_VERSION={version}");

    // --- Short git SHA (best-effort; "unknown" outside a checkout) ---
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=SCIRUST_GIT_SHA={sha}");

    // Rebuild when the version or the checked-out commit changes.
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../.git/HEAD");
}

/// Extract the first `version = "..."` line that appears inside the `[package]`
/// section of a Cargo manifest. Stops at the next `[section]` header so a
/// dependency's version cannot be mistaken for the package version.
fn parse_package_version(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines()
    {
        let trimmed = line.trim();
        if trimmed.starts_with('[')
        {
            in_package = trimmed == "[package]";
            continue;
        }
        if in_package
        {
            if let Some(rest) = trimmed.strip_prefix("version")
            {
                // rest looks like `= "0.14.0"` (allow spaces around `=`).
                if let Some(eq) = rest.find('=')
                {
                    let val = rest[eq + 1..].trim().trim_matches('"');
                    if !val.is_empty()
                    {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}
