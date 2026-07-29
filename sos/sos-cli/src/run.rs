//! `sos run` — execute a study manifest to completion (RFC-0002 §10.4).
//!
//! This is the command the CLI has been unable to offer. Its two blockers are
//! both gone: `sos-workflow` resolves a TOML study to a validated
//! [`Plan`](sos_workflow::Plan), and `sos-scirust` supplies real
//! [`StageExecutor`] backends. What was left was
//! the binding — nothing in this binary could map a study's plugin names to
//! code. That is what this module is.
//!
//! ## Why only the model catalogue is bound
//!
//! `sos-scirust` ships five `Simulate` backends and two stage handlers, and
//! exactly one of them can be driven from a command line. That is not an
//! arbitrary limit of this command; it follows from what a *file* can say.
//!
//! [`OdeStageHandler`](sos_scirust::stage::OdeStageHandler) integrates
//! `dy/dt = f(t, y)` where `f` is a Rust closure. No JSON file can name a
//! closure, so no amount of CLI plumbing could let a user supply one — a
//! study naming that plugin would have to be run from Rust that already has
//! the right-hand side compiled in.
//! [`CatalogStageHandler`] runs
//! [`ModelRun`]s, which are *data*: a model name plus its parameters plus an
//! initial state. Those can be written down, so they can be handed to a
//! binary.
//!
//! That is the practical payoff of a model having an identity, and it is why
//! `sos run` binds the catalogue plugin and nothing else. When another backend
//! gets a data-only config and a handler, it joins the table
//! ([`HANDLERS`]).
//!
//! ## What the command needs, and why each piece is asked for rather than
//! guessed
//!
//! * **The study** (`<manifest.toml>`) — stages, their plugin pins, and the
//!   content address of each stage's configuration.
//! * **The runs** (`<runs.json>`) — a JSON array of [`ModelRun`]s, the same
//!   "consume already-known data from a file" convention `sos plan` and
//!   `sos plugins` already use, since there is no other persistence format to
//!   point at. The manifest pins each stage's config *by address*, so
//!   supplying the wrong file cannot silently substitute a different
//!   experiment: the address will not be on offer and the stage fails.
//! * **The plugins** (`--plugins <descriptors.json>`) — the registry the
//!   manifest's `pin` is checked against. Same file format `sos plugins`
//!   reads.
//! * **The capabilities** (`--allow <cap>[,<cap>]`) — **empty by default**.
//!   A study that needs `effectful` must say so on the command line; granting
//!   everything by default would make the capability gate decorative.
//! * **The environment** (`--env <label>`) — folded into every cache key.
//!   Defaults to a fixed label, so the same study memoizes identically on
//!   every machine, matching the host-independence the stage handlers already
//!   record. Pass `--env` when results should be bound to a specific host.
//!
//! ## What this does not do
//!
//! Memoization is per-invocation: a second `sos run` recomputes rather than
//! reading a persisted memo table. It is not *wrong* — the backends are `L3`,
//! so a re-run produces byte-identical objects at the same addresses and the
//! store does not grow — but the CPU is spent again. A persistent memo table
//! is separate work, not stubbed here.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{Digest, HashAlgo, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry};
use sos_scirust::model::ModelRun;
use sos_scirust::stage::{CatalogStageHandler, model_config_address, sim_backend};
use sos_workflow::{Dispatch, MemoTable, StageExecutor, resolve_manifest, run_plan};

use crate::args::Args;
use crate::error::{CliError, Result};
use crate::store;

/// The plugin name `sos run` binds to the model catalogue.
///
/// A study wanting catalogued models names this plugin. The name is part of
/// the study text, so it is fixed rather than configurable — a study that ran
/// yesterday must name the same plugin today.
pub const CATALOG_PLUGIN: &str = "sim-catalog";

/// Every plugin name this binary can bind, with a one-line description — the
/// CLI's handler table, and the thing whose absence kept `sos run` from
/// existing.
///
/// One entry today. See the module docs on why it is one and not five.
pub const HANDLERS: &[(&str, &str)] = &[(
    CATALOG_PLUGIN,
    "scirust-sim's catalogued models (L3, fixed-step RK4)",
)];

/// Run `sos run <manifest.toml> <runs.json> [--plugins <f>] [--allow <caps>]
/// [--env <label>] [--store <path>]`.
///
/// # Errors
/// [`CliError::Usage`] for a missing argument, an unreadable file, a manifest
/// that does not resolve, or an unknown capability name;
/// [`CliError::Workflow`] if the study fails to execute — including the case
/// where a stage pins a configuration the runs file does not contain, which is
/// a refusal rather than a substitution.
pub fn run(args: &Args) -> Result<String> {
    let manifest_path = args.positional(0, "manifest.toml")?;
    let runs_path = args.positional(1, "runs.json")?;

    let text = std::fs::read_to_string(manifest_path)?;
    let plan =
        resolve_manifest(&text).map_err(|e| CliError::Usage(format!("{manifest_path}: {e}")))?;

    let runs: Vec<ModelRun> = serde_json::from_slice(&std::fs::read(runs_path)?)?;
    let grant = parse_grant(args.flag("allow"))?;
    let registry = load_registry(args.flag("plugins"))?;

    let root = store::resolve_root(args.flag("store"))?;
    let store = Rc::new(RefCell::new(store::open(&root)?));

    let mut handler = CatalogStageHandler::new(
        Rc::clone(&store),
        sim_backend(SemVer::new(0, 1, 0), backend_hash()),
    );
    let offered: Vec<Digest> = runs.into_iter().map(|r| handler.offer(r)).collect();

    let mut dispatch = Dispatch::new(&registry, grant);
    dispatch.register(CATALOG_PLUGIN, Box::new(handler) as Box<dyn StageExecutor>);

    let env = HashAlgo::Sha256.hash(
        b"sos-cli:env",
        args.flag("env").unwrap_or("unspecified").as_bytes(),
    );
    let ledger = run_plan(&plan, &env, &mut MemoTable::new(), &mut dispatch)?;

    let mut out = format!(
        "ran {} of {} stage(s) from {manifest_path} ({} run(s) on offer)\n",
        ledger.ran_count(),
        ledger.steps.len(),
        offered.len()
    );
    for step in &ledger.steps
    {
        out.push_str(&format!(
            "  {:<20} {}\n",
            step.stage.0,
            step.outputs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    out.push_str(&format!("stored under {}", root.display()));
    Ok(out)
}

/// The backend build stamped into every result's [`ReproMeta`].
///
/// A content hash of this binary's own version, which is the most specific
/// honest thing available here: the CLI does not know `scirust-sim`'s build
/// hash, and inventing a plausible-looking digest would be worse than a
/// derived one that is at least stable and traceable to a release.
fn backend_hash() -> Digest {
    HashAlgo::Sha256.hash(b"sos-cli:backend", env!("CARGO_PKG_VERSION").as_bytes())
}

/// Parse `--allow effectful,network` into a [`Grant`]. Absent means **no**
/// capabilities, not all of them.
fn parse_grant(flag: Option<&str>) -> Result<Grant> {
    let mut grant = Grant::new();
    let Some(list) = flag
    else
    {
        return Ok(grant);
    };
    for name in list.split(',').map(str::trim).filter(|s| !s.is_empty())
    {
        grant = grant.allow(parse_capability(name)?);
    }
    Ok(grant)
}

/// The capability named by `s`.
///
/// Unknown names are an error rather than a [`Capability::Custom`]: a typo in
/// `--allow effectfull` must not silently grant nothing and leave the user
/// puzzling over a refused stage.
fn parse_capability(s: &str) -> Result<Capability> {
    match s
    {
        "effectful" => Ok(Capability::Effectful),
        "network" => Ok(Capability::Network),
        "gpu" => Ok(Capability::Gpu),
        "filesystem" => Ok(Capability::Filesystem),
        other => Err(CliError::Usage(format!(
            "unknown capability `{other}` (expected one of: effectful, network, gpu, filesystem)"
        ))),
    }
}

/// Load the plugin registry from a descriptors file, or build the default one
/// naming just the catalogue plugin.
///
/// The default exists so a simple study does not need a descriptors file at
/// all, but it is not a shortcut around pinning: the manifest's `pin` must
/// still match, and [`default_registry`] uses the same well-known hash a study
/// author can compute.
fn load_registry(path: Option<&str>) -> Result<Registry> {
    let Some(path) = path
    else
    {
        return Ok(default_registry());
    };
    let descriptors: Vec<PluginDescriptor> = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut registry = Registry::new();
    for d in descriptors
    {
        registry.register(d);
    }
    Ok(registry)
}

/// The well-known content hash of the built-in catalogue plugin — what a study
/// must write in its stage `pin` to run without a descriptors file.
#[must_use]
pub fn catalog_plugin_hash() -> Digest {
    HashAlgo::Sha256.hash(b"sos-cli:plugin", b"sim-catalog@1.0.0")
}

/// The registry `sos run` uses when no descriptors file is given: the
/// catalogue plugin, at the hash [`catalog_plugin_hash`] publishes.
#[must_use]
pub fn default_registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(
        PluginDescriptor::new(
            CATALOG_PLUGIN,
            SemVer::new(1, 0, 0),
            catalog_plugin_hash(),
            sos_registry::Role::Simulation,
        )
        .needs(Capability::Effectful),
    );
    registry
}

/// The content address a study must write in a stage's `config` field to
/// select `run` — re-exported here so a study author can compute it from the
/// same crate the CLI resolves it with.
#[must_use]
pub fn run_address(run: &ModelRun) -> Digest {
    model_config_address(run)
}

/// Run `sos address <runs.json>` — print the manifest fields a study author
/// has to write, for every run in the file.
///
/// Without this, `sos run` is unusable from a shell: a manifest pins its
/// stages by content address, and there was no way to *obtain* those hex
/// strings short of writing Rust. Content addresses are the point, so making
/// them unobtainable would have made the whole command a library API wearing a
/// CLI costume.
///
/// The output is deliberately paste-ready TOML rather than a bare list of
/// digests.
///
/// # Errors
/// [`CliError::Usage`] if the path is missing; [`CliError::Io`]/
/// [`CliError::Serde`] if the file cannot be read or parsed — including a run
/// naming a model this build does not have, which fails rather than loading as
/// a different one.
pub fn address(args: &Args) -> Result<String> {
    let path = args.positional(0, "runs.json")?;
    let runs: Vec<ModelRun> = serde_json::from_slice(&std::fs::read(path)?)?;
    if runs.is_empty()
    {
        return Ok(format!("{path} contains no runs"));
    }

    let mut out = format!(
        "# stage fields for {path} — plugin `{CATALOG_PLUGIN}` v1.0.0\n\
         # pin = \"{pin}\"\n",
        pin = catalog_plugin_hash().to_hex()
    );
    for (i, run) in runs.iter().enumerate()
    {
        out.push_str(&format!(
            "\n[[stage]]\nid      = \"stage-{i}\"  # {model}\n\
             plugin  = \"{CATALOG_PLUGIN}\"\nversion = \"1.0.0\"\n\
             pin     = \"{pin}\"\nconfig  = \"{cfg}\"\n",
            model = run.model.kind,
            pin = catalog_plugin_hash().to_hex(),
            cfg = run_address(run).to_hex(),
        ));
    }
    out.pop();
    Ok(out)
}
