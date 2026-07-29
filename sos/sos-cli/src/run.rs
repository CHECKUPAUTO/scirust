//! `sos run` — execute a study manifest to completion (RFC-0002 §10.4).
//!
//! This is the command the CLI has been unable to offer. Its two blockers are
//! both gone: `sos-workflow` resolves a TOML study to a validated
//! [`Plan`](sos_workflow::Plan), and `sos-scirust` supplies real
//! [`StageExecutor`] backends. What was left was
//! the binding — nothing in this binary could map a study's plugin names to
//! code. That is what this module is.
//!
//! ## Which backends can be bound, and why it is not a matter of effort
//!
//! `sos-scirust` ships six `Simulate` backends. Exactly **three** can be
//! driven from a command line, and that follows from what a *file* can say
//! rather than from how much plumbing is here.
//!
//! [`OdeStageHandler`](sos_scirust::stage::OdeStageHandler) integrates
//! `dy/dt = f(t, y)` where `f` is a Rust closure; `Dopri5OdeSimulator`,
//! `QuadratureSimulator` and `BroydenRootSimulator` likewise take a function —
//! a right-hand side, an integrand, a residual. **No JSON file can name a
//! function**, so no amount of CLI plumbing could let a user supply one; those
//! backends need a transport that ships *code* (RFC-0002 §10's WASM component
//! or MCP transports), which is a different mechanism, not a bigger version of
//! this one.
//!
//! The three that can be bound are the three whose configuration is data:
//!
//! * [`CatalogStageHandler`] runs [`ModelRun`]s — a model name, its
//!   parameters, an initial state.
//! * [`SpectrumStageHandler`] runs [`SpectrumConfig`]s — samples, a sample
//!   rate, a named window.
//! * [`WelchStageHandler`] runs [`WelchConfig`]s — the same, plus a segment
//!   length and an overlap.
//!
//! Both are written down in the same file, tagged by kind (see
//! [`StageConfig`]), and both appear in [`HANDLERS`].
//!
//! ## What the command needs, and why each piece is asked for rather than
//! guessed
//!
//! * **The study** (`<manifest.toml>`) — stages, their plugin pins, and the
//!   content address of each stage's configuration.
//! * **The runs** (`<runs.json>`) — a JSON array of tagged [`StageConfig`]s,
//!   the same "consume already-known data from a file" convention `sos plan`
//!   and `sos plugins` already use, since there is no other persistence format
//!   to point at. The manifest pins each stage's config *by address*, so
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
//!   This binary's version is folded in either way ([`env_digest`]).
//!
//! ## Memoization survives the process
//!
//! Results are remembered in a [`FileMemo`] beside the object store, so
//! re-running an unchanged study is nearly free rather than a full recompute
//! onto the same object ids. Persisting a cache introduces a staleness hazard
//! an in-memory one cannot have — see [`crate::memo`]'s docs — which is why
//! the environment digest binds this binary's own version.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{Digest, HashAlgo, SemVer};
use sos_registry::{Capability, Grant, PluginDescriptor, Registry};
use sos_scirust::model::ModelRun;
use sos_scirust::spectrum::{SpectrumConfig, WelchConfig};
use sos_scirust::stage::{
    CatalogStageHandler, SpectrumStageHandler, WelchStageHandler, model_config_address,
    signal_backend, sim_backend, spectrum_config_address, welch_config_address,
};
use sos_workflow::{Dispatch, StageExecutor, resolve_manifest, run_plan};

use crate::args::Args;
use crate::error::{CliError, Result};
use crate::memo::{DEFAULT_ENV_LABEL, FileMemo, env_digest};
use crate::store;

/// The plugin name `sos run` binds to the model catalogue.
///
/// A study wanting catalogued models names this plugin. The name is part of
/// the study text, so it is fixed rather than configurable — a study that ran
/// yesterday must name the same plugin today.
pub const CATALOG_PLUGIN: &str = "sim-catalog";

/// The plugin name `sos run` binds to the spectral backend.
pub const SPECTRUM_PLUGIN: &str = "signal-periodogram";

/// The plugin name `sos run` binds to the Welch averaged-periodogram backend.
pub const WELCH_PLUGIN: &str = "signal-welch";

/// Every plugin name this binary can bind, with a one-line description — the
/// CLI's handler table, and the thing whose absence kept `sos run` from
/// existing.
///
/// Three entries. See the module docs on why it is three and not six.
pub const HANDLERS: &[(&str, &str)] = &[
    (
        CATALOG_PLUGIN,
        "scirust-sim's catalogued models (L3, fixed-step RK4)",
    ),
    (
        SPECTRUM_PLUGIN,
        "scirust-signal's windowed periodogram (L3, reproducible not accurate)",
    ),
    (
        WELCH_PLUGIN,
        "scirust-signal's Welch averaged periodogram (L3, variance falls as 1/segments)",
    ),
];

/// One entry in a runs file: a stage configuration, tagged by which backend
/// consumes it.
///
/// Tagged rather than inferred from shape, because guessing which backend a
/// JSON object was meant for is exactly the kind of silent substitution the
/// rest of this system refuses. An unknown tag is an error.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum StageConfig {
    /// A catalogued model run, for [`CATALOG_PLUGIN`].
    Model(ModelRun),
    /// A spectral measurement, for [`SPECTRUM_PLUGIN`].
    Spectrum(SpectrumConfig),
    /// A Welch averaged-periodogram measurement, for [`WELCH_PLUGIN`].
    Welch(WelchConfig),
}

impl StageConfig {
    /// The content address a study must write in a stage's `config` field to
    /// select this configuration.
    #[must_use]
    pub fn address(&self) -> Digest {
        match self
        {
            Self::Model(run) => model_config_address(run),
            Self::Spectrum(config) => spectrum_config_address(config),
            Self::Welch(config) => welch_config_address(config),
        }
    }

    /// The plugin that consumes this configuration.
    #[must_use]
    pub const fn plugin(&self) -> &'static str {
        match self
        {
            Self::Model(_) => CATALOG_PLUGIN,
            Self::Spectrum(_) => SPECTRUM_PLUGIN,
            Self::Welch(_) => WELCH_PLUGIN,
        }
    }

    /// A short description of what this configuration will do, for
    /// [`address`]'s comments.
    #[must_use]
    pub fn summary(&self) -> String {
        match self
        {
            Self::Model(run) => run.model.kind.to_string(),
            Self::Spectrum(c) => format!("{} window over {} samples", c.window, c.signal.len()),
            Self::Welch(c) => format!(
                "{} window, {}-sample segments, {} overlap",
                c.window, c.segment_len, c.overlap
            ),
        }
    }
}

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

    let configs: Vec<StageConfig> = serde_json::from_slice(&std::fs::read(runs_path)?)?;
    let grant = parse_grant(args.flag("allow"))?;
    let registry = load_registry(args.flag("plugins"))?;

    let root = store::resolve_root(args.flag("store"))?;
    let store = Rc::new(RefCell::new(store::open(&root)?));

    let mut models = CatalogStageHandler::new(
        Rc::clone(&store),
        sim_backend(SemVer::new(0, 1, 0), backend_hash()),
    );
    let mut spectra = SpectrumStageHandler::new(
        Rc::clone(&store),
        signal_backend(SemVer::new(0, 1, 0), backend_hash()),
    );
    let mut welch = WelchStageHandler::new(
        Rc::clone(&store),
        signal_backend(SemVer::new(0, 1, 0), backend_hash()),
    );
    let offered: Vec<Digest> = configs
        .into_iter()
        .map(|c| match c
        {
            StageConfig::Model(run) => models.offer(run),
            StageConfig::Spectrum(config) => spectra.offer(config),
            StageConfig::Welch(config) => welch.offer(config),
        })
        .collect();

    let mut dispatch = Dispatch::new(&registry, grant);
    dispatch.register(CATALOG_PLUGIN, Box::new(models) as Box<dyn StageExecutor>);
    dispatch.register(SPECTRUM_PLUGIN, Box::new(spectra) as Box<dyn StageExecutor>);
    dispatch.register(WELCH_PLUGIN, Box::new(welch) as Box<dyn StageExecutor>);

    let env = env_digest(args.flag("env").unwrap_or(DEFAULT_ENV_LABEL));
    let mut memo = FileMemo::open(&root)?;
    let ledger = run_plan(&plan, &env, &mut memo, &mut dispatch)?;
    memo.flush()?;

    let mut out = format!(
        "ran {} of {} stage(s) from {manifest_path} ({} config(s) on offer)\n",
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

/// The well-known content hash of a built-in plugin — what a study must write
/// in its stage `pin` to run without a descriptors file.
#[must_use]
pub fn plugin_hash(plugin: &str) -> Digest {
    HashAlgo::Sha256.hash(b"sos-cli:plugin", format!("{plugin}@1.0.0").as_bytes())
}

/// The catalogue plugin's well-known hash. Kept as a named function because
/// study authors and tests reference it directly.
#[must_use]
pub fn catalog_plugin_hash() -> Digest {
    plugin_hash(CATALOG_PLUGIN)
}

/// The spectral plugin's well-known hash.
#[must_use]
pub fn spectrum_plugin_hash() -> Digest {
    plugin_hash(SPECTRUM_PLUGIN)
}

/// The registry `sos run` uses when no descriptors file is given: every
/// built-in plugin, at the hash [`plugin_hash`] publishes.
#[must_use]
pub fn default_registry() -> Registry {
    let mut registry = Registry::new();
    for (name, _) in HANDLERS
    {
        registry.register(
            PluginDescriptor::new(
                *name,
                SemVer::new(1, 0, 0),
                plugin_hash(name),
                sos_registry::Role::Simulation,
            )
            .needs(Capability::Effectful),
        );
    }
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
    let configs: Vec<StageConfig> = serde_json::from_slice(&std::fs::read(path)?)?;
    if configs.is_empty()
    {
        return Ok(format!("{path} contains no configurations"));
    }

    let mut out = format!("# stage fields for {path}\n");
    for (i, config) in configs.iter().enumerate()
    {
        out.push_str(&format!(
            "\n[[stage]]\nid      = \"stage-{i}\"  # {what}\n\
             plugin  = \"{plugin}\"\nversion = \"1.0.0\"\n\
             pin     = \"{pin}\"\nconfig  = \"{cfg}\"\n",
            what = config.summary(),
            plugin = config.plugin(),
            pin = plugin_hash(config.plugin()).to_hex(),
            cfg = config.address().to_hex(),
        ));
    }
    out.pop();
    Ok(out)
}
