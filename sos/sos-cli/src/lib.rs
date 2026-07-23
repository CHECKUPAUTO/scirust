//! # `sos` — the Scientific Operating System command-line porcelain
//!
//! A thin, git-shaped shell over capabilities that already exist and are
//! tested in their own crates (RFC-0002 §10.4): it adds no new compute of its
//! own, only a discoverable command surface so a user does not have to
//! hand-write the library API for common tasks.
//!
//! ```text
//! sos init [path]                          create a reasoning repository
//! sos clone <src> <dest>                   copy a repository (local paths)
//! sos push <src> <dest>                    same operation, opposite direction
//! sos log [--store <path>]                 list every object, oldest first
//! sos know [--store <path>] <sub> ...       query the knowledge graph
//! sos ask [--store <path>] [--limit N]      run a curiosity sweep
//! sos why [--store <path>] <object>        print the provenance behind an object
//! sos verify [--store <path>] <object>     check an object's structural + content identity
//! sos diff [--store <path>] <a> <b>        compare two studies' ancestor sets
//! sos plan <candidates.json> [--floor N]   recommend the next experiment
//!          [--budget N]
//! sos publish <publication.json>           seal (and render) a publication
//!          [--author <n>] [--format <f>]
//!          [--store <path>]
//! sos plugins <descriptors.json>           list/find plugins
//!          [--role <r>] [--domain <d>]
//! sos help | --help | -h                   this text
//! sos version | --version | -V             version
//! ```
//!
//! ## What is deliberately not here
//!
//! `sos run <manifest>` (executing a discovery workflow) and a true `sos
//! merge` (reconciling two labs' divergent graphs) are not implemented: the
//! former needs a real [`sos_workflow::StageExecutor`] backend — which is
//! `sos-scirust`'s job and does not exist yet — and inventing one here would
//! either be fake execution or a stub, both forbidden. The latter needs
//! conflict-resolution semantics no crate in this workspace has designed yet.
//! `sos clone`/`sos push` cover the local, no-network sharing case (mirroring
//! how `git clone`/`git push` already work against a local path); a real
//! network remote is `sos-mcp`'s domain, not this one.
//!
//! ## Example
//!
//! Each subcommand is a thin function over the same store any other `sos-*`
//! engine already reads and writes — `sos_cli::run` is what the `sos` binary
//! calls, but the individual command modules are directly usable as a library
//! too (this is how the test suite drives them without spawning a process):
//!
//! ```
//! use sos_core::{Author, Object};
//! use sos_reasoning::Derivation;
//! use sos_store::{FileStore, TypedStore};
//!
//! # let root = std::env::temp_dir().join(format!("sos-cli-doctest-{}", std::process::id()));
//! # let _ = std::fs::remove_dir_all(&root);
//! let path = root.to_str().unwrap();
//!
//! // `sos init` — creates the on-disk store.
//! let msg = sos_cli::init::run(Some(path)).unwrap();
//! assert!(msg.contains("Initialized"));
//!
//! // Some other tool (or a human, via a library call) writes a real object.
//! let mut store = FileStore::open(path).unwrap();
//! let axiom = Object::builder(Derivation::undetermined("an axiom")).author(Author::human("ada")).seal();
//! store.put_object(&axiom).unwrap();
//!
//! // `sos log` — the porcelain sees it immediately (same on-disk store).
//! let log = sos_cli::log::run(Some(path)).unwrap();
//! assert!(log.contains("Derivation"));
//! # std::fs::remove_dir_all(&root).ok();
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod args;
pub mod ask;
pub mod clone;
pub mod diff;
pub mod error;
pub mod header;
pub mod init;
pub mod know;
pub mod log;
pub mod plan;
pub mod plugins;
pub mod publish;
pub mod store;
pub mod verify;
pub mod why;

use args::{Args, parse_object_id};
use error::CliError;

/// Every dispatchable command name, for the "unknown command" message.
const ALL_COMMANDS: &[&str] = &[
    "init", "clone", "push", "log", "know", "ask", "why", "verify", "diff", "plan", "publish",
    "plugins", "help", "version",
];

/// Run the `sos` command line (`args` excludes the program name) and return
/// the process exit code: `0` on success, `1` on any error.
#[must_use]
pub fn run(args: &[String]) -> u8 {
    match dispatch(args)
    {
        Ok(text) =>
        {
            if !text.is_empty()
            {
                println!("{text}");
            }
            0
        },
        Err(e) =>
        {
            eprintln!("sos: {e}");
            1
        },
    }
}

/// Dispatch on the first argument (the subcommand name).
fn dispatch(args: &[String]) -> error::Result<String> {
    let rest: Vec<String> = if args.len() > 1
    {
        args[1..].to_vec()
    }
    else
    {
        Vec::new()
    };

    match args.first().map(String::as_str)
    {
        None | Some("help") | Some("-h") | Some("--help") => Ok(help_text()),
        Some("version") | Some("--version") | Some("-V") =>
        {
            Ok(format!("sos {}", env!("CARGO_PKG_VERSION")))
        },
        Some("init") => init::run(rest.first().map(String::as_str)),
        Some("clone") =>
        {
            let a = Args::parse(&rest)?;
            let src = a.positional(0, "src")?;
            let dest = a.positional(1, "dest")?;
            clone::run(src, dest)
        },
        Some("push") =>
        {
            let a = Args::parse(&rest)?;
            let src = a.positional(0, "src")?;
            let dest = a.positional(1, "dest")?;
            clone::run(src, dest)
        },
        Some("log") =>
        {
            let a = Args::parse(&rest)?;
            log::run(a.flag("store"))
        },
        Some("know") =>
        {
            let a = Args::parse(&rest)?;
            let store_flag = a.flag("store").map(str::to_owned);
            know::run(store_flag.as_deref(), &a)
        },
        Some("ask") =>
        {
            let a = Args::parse(&rest)?;
            let store_flag = a.flag("store").map(str::to_owned);
            ask::run(store_flag.as_deref(), &a)
        },
        Some("why") =>
        {
            let a = Args::parse(&rest)?;
            let id = parse_object_id(a.positional(0, "object")?)?;
            why::run(a.flag("store"), id)
        },
        Some("verify") =>
        {
            let a = Args::parse(&rest)?;
            let id = parse_object_id(a.positional(0, "object")?)?;
            verify::run(a.flag("store"), id)
        },
        Some("diff") =>
        {
            let a = Args::parse(&rest)?;
            let root_a = parse_object_id(a.positional(0, "root-a")?)?;
            let root_b = parse_object_id(a.positional(1, "root-b")?)?;
            diff::run(a.flag("store"), root_a, root_b)
        },
        Some("plan") =>
        {
            let a = Args::parse(&rest)?;
            plan::run(&a)
        },
        Some("publish") =>
        {
            let a = Args::parse(&rest)?;
            publish::run(&a)
        },
        Some("plugins") =>
        {
            let a = Args::parse(&rest)?;
            plugins::run(&a)
        },
        Some(other) => Err(CliError::Usage(format!(
            "unknown command `{other}` (expected one of: {})",
            ALL_COMMANDS.join(", ")
        ))),
    }
}

/// The `sos help` text.
fn help_text() -> String {
    concat!(
        "sos — the Scientific Operating System command-line porcelain\n\n",
        "USAGE:\n",
        "  sos init [path]                        create a reasoning repository\n",
        "  sos clone <src> <dest>                  copy a repository (local paths)\n",
        "  sos push <src> <dest>                   same operation, opposite direction\n",
        "  sos log [--store <path>]                list every object\n",
        "  sos know [--store <path>] <sub> ...      query the knowledge graph\n",
        "           (neighbors|in-neighbors <id> <relation>, related <a> <b>,\n",
        "            path <a> <b> [relation])\n",
        "  sos ask [--store <path>] [--limit N]     run a curiosity sweep\n",
        "  sos why [--store <path>] <object>        print the provenance behind an object\n",
        "  sos verify [--store <path>] <object>     check structural + content identity\n",
        "  sos diff [--store <path>] <a> <b>        compare two studies\n",
        "  sos plan <candidates.json> [--floor N]   recommend the next experiment\n",
        "           [--budget N]\n",
        "  sos publish <publication.json>           seal (and render) a publication\n",
        "           [--author <name>] [--format md|html|json] [--store <path>]\n",
        "  sos plugins <descriptors.json>           list/find plugins\n",
        "           [--role <role>] [--domain <tag>]\n",
        "  sos help | version"
    )
    .to_owned()
}
