mod commands;
mod workspace;

use std::fmt;
use std::io;
use std::process::ExitCode;

use workspace::Workspace;

type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
struct AppError {
    message: String,
}

impl AppError {
    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn io(error: io::Error) -> Self {
        Self::message(error.to_string())
    }

    fn command(name: &str, code: Option<i32>, stderr: &[u8]) -> Self {
        let stderr = String::from_utf8_lossy(stderr);
        Self::message(format!(
            "{name} failed with status {:?}: {}",
            code,
            stderr.trim()
        ))
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cargo-scirust: {error}");
            ExitCode::from(1)
        }
    }
}

fn real_main() -> AppResult<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    // Cargo plugins are invoked as `cargo-scirust scirust ...`; direct binary
    // invocation is `cargo-scirust ...`. Supporting both makes development and
    // installation equally predictable.
    if args.first().map(String::as_str) == Some("scirust") {
        args.remove(0);
    }

    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        "version" | "-V" | "--version" => {
            println!("cargo-scirust {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "affected" => commands::affected(&Workspace::load()?, &args[1..]),
        "check" => commands::check(&Workspace::load()?, &args[1..]),
        "parity" => commands::parity(&Workspace::load()?, &args[1..]),
        "determinism" => commands::determinism(&Workspace::load()?, &args[1..]),
        "cost" => commands::cost(&Workspace::load()?, &args[1..]),
        "features" => commands::features(&Workspace::load()?, &args[1..]),
        "bench" => commands::bench(&Workspace::load()?, &args[1..]),
        "calibrate" => commands::calibrate(&Workspace::load()?, &args[1..]),
        other => Err(AppError::message(format!(
            "unknown command `{other}`; run `cargo scirust help`"
        ))),
    }
}

fn print_help() {
    println!("cargo-scirust — fast repository-aware commands for SciRust");
    println!();
    println!("USAGE:");
    println!("  cargo scirust <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("  affected      Map changed files to direct + transitive workspace crates");
    println!("  check         Run fmt/clippy/tests only where the dependency graph requires");
    println!("  parity        Compare two commands byte-for-byte (exit/stdout/stderr)");
    println!("  determinism   Repeat a command and require exact reproducible output");
    println!("  cost          Scan Rust source for copy/allocation/GPU-sync cost indicators");
    println!("  features      Inspect features or execute a bounded pairwise feature matrix");
    println!("  bench         Run cargo bench only for affected or selected crates");
    println!("  calibrate     Learn S/M/L/XL/XXL/XXXL tokenizer piece-size boundaries");
    println!("  version       Print tool version");
    println!();
    println!("Run `cargo scirust <COMMAND> --help` for command-specific usage.");
}
