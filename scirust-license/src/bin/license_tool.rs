//! `license-tool` — the SciRust licensing command-line tool.
//!
//! A thin wrapper over [`scirust_license::cli::run`]: it supplies the process
//! arguments and the real wall-clock time, prints the captured output and
//! propagates the exit code. All command logic (and its tests) lives in the
//! library's `cli` module so it stays deterministic.

use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let result = scirust_license::cli::run(&args, now);
    print!("{}", result.stdout);
    std::process::exit(result.exit);
}
