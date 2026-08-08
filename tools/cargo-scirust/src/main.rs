mod complete;
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
        },
    }
}

fn real_main() -> AppResult<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    // Cargo plugins are invoked as `cargo-scirust scirust ...`; direct binary
    // invocation is `cargo-scirust ...`.
    if args.first().map(String::as_str) == Some("scirust") {
        args.remove(0);
    }

    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    if args[1..]
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        if print_command_help(command) {
            return Ok(());
        }
    }

    match command {
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        },
        "version" | "-V" | "--version" => {
            println!("cargo-scirust {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        },
        "affected" => complete::affected(&Workspace::load()?, &args[1..]),
        "check" => complete::check(&Workspace::load()?, &args[1..]),
        "parity" => complete::parity(&Workspace::load()?, &args[1..]),
        "determinism" => complete::determinism(&Workspace::load()?, &args[1..]),
        "cost" => complete::cost(&Workspace::load()?, &args[1..]),
        "features" => complete::features(&Workspace::load()?, &args[1..]),
        "bench" => complete::bench(&Workspace::load()?, &args[1..]),
        "calibrate" => complete::calibrate(&Workspace::load()?, &args[1..]),
        other => Err(AppError::message(format!(
            "unknown command `{other}`; run `cargo scirust help`"
        ))),
    }
}

fn print_command_help(command: &str) -> bool {
    match command {
        "affected" => println!(
            "cargo scirust affected [--base REF] [--head REF] [--json] [--names-only] [--direct-only] [--fail-if-empty]"
        ),
        "check" => println!(
            "cargo scirust check [--base REF] [--head REF] [--all] [--full] [--dry-run] [--all-features] [--unlocked] [--no-fmt] [--no-clippy] [--no-test]"
        ),
        "parity" => println!(
            "cargo scirust parity --left \"COMMAND\" --right \"COMMAND\" [--repeat N] [--ignore-stderr] [--allow-failure] [--json]"
        ),
        "determinism" => println!(
            "cargo scirust determinism [--repeat N] [--ignore-stderr] [--allow-failure] [--json] -- PROGRAM [ARGS...]"
        ),
        "cost" => println!(
            "cargo scirust cost [--path PATH | -p PACKAGE] [--limit N] [--json] [--no-static] [--measure N] [--warmup N] [--inherit-io] [-- PROGRAM ARGS...]"
        ),
        "features" => println!(
            "cargo scirust features <package> [--cover pairwise] [--execute] [--max N] [--json] [--allow-incompatible] [--include-default]"
        ),
        "bench" => println!(
            "cargo scirust bench [--base REF] [--head REF] [--all] [-p PACKAGE] [--repeat N] [--dry-run] [-- <cargo bench args>]"
        ),
        "calibrate" => {
            println!(
                "cargo scirust calibrate --tokenizer FILE --input PATH [--input PATH...] --output PROFILE.json [--recursive] [--probe-lengths CSV] [--cases-per-length N] [--warmup-runs N] [--measured-runs N] [--extension CSV] [--device NAME] [--debug] [--dry-run]"
            );
            println!("cargo scirust calibrate (--pieces FILE | --lengths FILE) [--json]");
        },
        _ => return false,
    }
    true
}

fn print_help() {
    println!(
        "cargo-scirust — repository-aware validation, parity, cost and ElasticTokenizer tooling"
    );
    println!();
    println!("USAGE:");
    println!("  cargo scirust <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("  affected      Map changes to direct/transitive workspace crates");
    println!("  check         Run locked fmt/clippy/tests only where required");
    println!(
        "  parity        Compare two successful commands exactly, with repeat/JSON diagnostics"
    );
    println!("  determinism   Repeat a successful command and require reproducible output");
    println!("  cost          Static cost audit plus optional measured wall-clock command cost");
    println!("  features      Inspect or execute a full pairwise compatibility matrix");
    println!("  bench         Benchmark affected/selected crates, optionally repeatedly");
    println!(
        "  calibrate     Run full semantics-gated ElasticTokenizer autotune or size-only analysis"
    );
    println!("  version       Print tool version");
    println!();
    println!("Run `cargo scirust <COMMAND> --help` for command-specific usage.");
}
