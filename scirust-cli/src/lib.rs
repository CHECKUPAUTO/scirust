//! `scirust` — one entry point for the whole toolkit.
//!
//! A thin, discoverable dispatcher over capabilities that already exist and
//! are tested elsewhere in the workspace: it adds no new compute, only a
//! simple command surface so users don't have to hand-write the library
//! API for common tasks. `scirust help` lists everything.

pub mod quickstart;

/// One registered command: name, one-line help, and what it does.
struct Command {
    name: &'static str,
    args: &'static str,
    about: &'static str,
}

const COMMANDS: &[Command] = &[
    Command {
        name: "quickstart",
        args: "",
        about: "Train the XOR demo classifier end to end (deterministic) and report.",
    },
    Command {
        name: "analyze",
        args: "<file.rs> [--sarif]",
        about: "Ownership analysis of a real Rust file (use-after-move, borrows). SARIF for CI.",
    },
    Command {
        name: "verify",
        args: "emit|verify <args...>",
        about: "Emit or check a deterministic inference proof certificate.",
    },
    Command {
        name: "help",
        args: "",
        about: "Show this list of commands.",
    },
    Command {
        name: "version",
        args: "",
        about: "Print the scirust CLI version.",
    },
];

fn print_help() {
    println!("scirust — pure-Rust deterministic ML toolkit\n");
    println!("usage: scirust <command> [args]\n");
    println!("commands:");
    let width = COMMANDS
        .iter()
        .map(|c| c.name.len() + c.args.len())
        .max()
        .unwrap_or(0);
    for c in COMMANDS
    {
        let sig = if c.args.is_empty()
        {
            c.name.to_string()
        }
        else
        {
            format!("{} {}", c.name, c.args)
        };
        println!("  {sig:<width$}  {}", c.about, width = width + 1);
    }
    println!("\nrun `scirust <command>` with no further args for per-command usage.");
}

/// Dispatch `args` (excluding the program name). Returns the exit code.
pub fn run(args: &[String]) -> u8 {
    match args.first().map(String::as_str)
    {
        None | Some("help") | Some("-h") | Some("--help") =>
        {
            print_help();
            0
        },
        Some("version") | Some("--version") | Some("-V") =>
        {
            println!("scirust {}", env!("CARGO_PKG_VERSION"));
            0
        },
        Some("quickstart") => quickstart::run(),
        Some("analyze") => scirust_som_cli::run(&args[1..], "scirust analyze"),
        Some("verify") => scirust_runtime::proofcli::run(&args[1..]),
        Some(other) =>
        {
            eprintln!("unknown command: `{other}`\n");
            print_help();
            2
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn help_and_version_succeed() {
        assert_eq!(run(&s(&["help"])), 0);
        assert_eq!(run(&[]), 0);
        assert_eq!(run(&s(&["version"])), 0);
    }

    #[test]
    fn unknown_command_is_rejected() {
        assert_eq!(run(&s(&["frobnicate"])), 2);
    }

    #[test]
    fn quickstart_via_dispatch_succeeds() {
        assert_eq!(run(&s(&["quickstart"])), 0);
    }

    #[test]
    fn analyze_missing_arg_is_usage_error() {
        // No file → the som driver returns its usage code (2).
        assert_eq!(run(&s(&["analyze"])), 2);
    }

    #[test]
    fn verify_missing_arg_is_usage_error() {
        assert_eq!(run(&s(&["verify"])), 2);
    }
}
