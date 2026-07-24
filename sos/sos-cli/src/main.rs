//! `sos` — entry point. See `sos help`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(sos_cli::run(&args))
}
