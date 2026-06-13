//! `scirust` — one entry point for the whole toolkit.
//!
//! A thin, discoverable dispatcher over capabilities that already exist and
//! are tested elsewhere in the workspace: it adds no new compute, only a
//! command surface so users don't have to hand-write the library API for
//! common tasks. `scirust help` lists everything; `scirust info` describes
//! the guarantees.

pub mod learning;
pub mod quickstart;
pub mod symbolic;

/// One registered command for the help listing.
struct Command {
    name: &'static str,
    args: &'static str,
    about: &'static str,
}

/// Commands grouped by theme, in display order.
const GROUPS: &[(&str, &[Command])] = &[
    (
        "LEARNING",
        &[
            Command {
                name: "quickstart",
                args: "",
                about: "Train the XOR demo MLP (deterministic) end to end → 4/4.",
            },
            Command {
                name: "som train",
                args: "[--seed N] [--epochs E]",
                about: "Train the ownership model; report accuracy vs baseline.",
            },
            Command {
                name: "evo",
                args: "[--seed N] [--gens G]",
                about: "Minimize the sphere function with a seeded genetic algorithm.",
            },
        ],
    ),
    (
        "SYMBOLIC MATH",
        &[
            Command {
                name: "diff",
                args: "<expr> [var]",
                about: "Symbolic derivative, e.g. `diff \"x^2 + 3*x\"`.",
            },
            Command {
                name: "simplify",
                args: "<expr>",
                about: "Algebraic simplification of an expression.",
            },
            Command {
                name: "eval",
                args: "<expr> [x=.. ..]",
                about: "Evaluate an expression at given variable values.",
            },
            Command {
                name: "solve",
                args: "<expr> [var]",
                about: "Real roots of `expr = 0` (linear / quadratic).",
            },
        ],
    ),
    (
        "CODE ANALYSIS",
        &[Command {
            name: "analyze",
            args: "<file.rs> [--sarif]",
            about: "Ownership analysis of real Rust (use-after-move, borrows). SARIF for CI.",
        }],
    ),
    (
        "INFERENCE INTEGRITY",
        &[Command {
            name: "verify",
            args: "emit|verify <args..>",
            about: "Emit or check a deterministic inference proof certificate.",
        }],
    ),
    (
        "META",
        &[
            Command {
                name: "info",
                args: "",
                about: "Capabilities, guarantees, determinism.",
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
        ],
    ),
];

fn print_help() {
    println!("scirust — pure-Rust deterministic ML & scientific-computing toolkit\n");
    println!("usage: scirust <command> [args]\n");
    let width = GROUPS
        .iter()
        .flat_map(|(_, cs)| cs.iter())
        .map(|c| c.name.len() + c.args.len() + 1)
        .max()
        .unwrap_or(0);
    for (group, cmds) in GROUPS
    {
        println!("{group}");
        for c in *cmds
        {
            let sig = if c.args.is_empty()
            {
                c.name.to_string()
            }
            else
            {
                format!("{} {}", c.name, c.args)
            };
            println!("  {sig:<width$}  {}", c.about);
        }
        println!();
    }
    println!("run a command with no further args for its specific usage.");
}

fn print_info() {
    println!(
        "scirust {} — pure Rust, zero FFI\n",
        env!("CARGO_PKG_VERSION")
    );
    println!("Guarantees:");
    println!("  • Deterministic: seeded PCG RNG everywhere; same seed ⇒ bit-identical output.");
    println!("  • Oracle-validated: every numeric primitive is tested against a reference.");
    println!("  • Stable Rust: the whole workspace builds and tests on stable (nightly only");
    println!("    for the optional `portable-simd` feature).");
    println!(
        "  • Auditable: pure Rust, no C/C++/Python, Cargo.lock committed, cargo-deny in CI.\n"
    );
    println!("Highlights reachable from this CLI:");
    println!("  • Deep-learning core + reverse-mode autodiff (`quickstart`).");
    println!("  • Ownership analysis of real Rust source (`analyze`, `som train`).");
    println!("  • Symbolic math: differentiation, simplification, solving (`diff`/`solve`/…).");
    println!("  • Evolutionary optimization (`evo`).");
    println!("  • Verifiable, reproducible inference certificates (`verify`).\n");
    println!("Docs: README.md · docs/REFERENCE.md · `cargo doc --workspace --no-deps --open`");
}

/// Dispatch `args` (excluding the program name). Returns the exit code.
pub fn run(args: &[String]) -> u8 {
    let rest = if args.len() > 1 { &args[1..] } else { &[] };
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
        Some("info") =>
        {
            print_info();
            0
        },
        Some("quickstart") => quickstart::run(),
        Some("som") => learning::run_som(rest),
        Some("evo") => learning::run_evo(rest),
        Some("diff") => symbolic::run_diff(rest),
        Some("simplify") => symbolic::run_simplify(rest),
        Some("eval") => symbolic::run_eval(rest),
        Some("solve") => symbolic::run_solve(rest),
        Some("analyze") => scirust_som_cli::run(rest, "scirust analyze"),
        Some("verify") => scirust_runtime::proofcli::run(rest),
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
    fn meta_commands_succeed() {
        assert_eq!(run(&[]), 0);
        assert_eq!(run(&s(&["help"])), 0);
        assert_eq!(run(&s(&["version"])), 0);
        assert_eq!(run(&s(&["info"])), 0);
    }

    #[test]
    fn unknown_command_is_rejected() {
        assert_eq!(run(&s(&["frobnicate"])), 2);
    }

    #[test]
    fn dispatch_reaches_each_group() {
        assert_eq!(run(&s(&["quickstart"])), 0);
        assert_eq!(run(&s(&["diff", "x*x"])), 0);
        assert_eq!(run(&s(&["solve", "x^2 - 4"])), 0);
        assert_eq!(run(&s(&["evo", "--gens", "20"])), 0);
        assert_eq!(run(&s(&["som", "train", "--epochs", "3"])), 0);
    }

    #[test]
    fn usage_errors_return_two() {
        assert_eq!(run(&s(&["analyze"])), 2);
        assert_eq!(run(&s(&["verify"])), 2);
        assert_eq!(run(&s(&["diff"])), 2);
        assert_eq!(run(&s(&["eval"])), 2);
    }
}
