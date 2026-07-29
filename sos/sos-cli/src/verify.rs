//! `sos verify` — check an object's structural identity, and its content hash
//! where the object's kind is one this CLI recognizes.
//!
//! [`sos_store::TypedStore::get_object`] is generic over a compile-time body
//! type, so *recomputing* a content hash needs to know the concrete Rust type
//! — there is no body-type-erased hash check anywhere in the kernel (a body's
//! canonical encoding is inherently type-specific, by design). This command
//! therefore always reports the structural header (kind, determinism level,
//! parent count — read generically, the same way `sos log` does), and
//! *additionally* recomputes and checks the content address for every kind
//! this CLI links against — every [`sos_core::Body`] type across the engines
//! landed so far, **including the four kinds `sos run` produces**. An
//! unrecognized kind still gets the structural report, honestly labelled as
//! such rather than silently skipped.
//!
//! ## `--rerun`: identity is not reproduction
//!
//! Without a flag this checks **identity** — is this object what it says it
//! is — which is a real check but not the one RFC-0002 §10.4 describes.
//! `sos verify --rerun <object> <runs.json>` does the other one: it finds the
//! run that produced the object, re-executes its plan, and checks the
//! reproduction contract node by node via `sos-repro`'s `verify_object`.
//!
//! Two things about that path are load-bearing.
//!
//! **The re-run must not be served from the cache.** `sos run` persists a
//! memo table, and reusing it here would replay the recorded outputs and
//! "verify" nothing at all — the strongest possible false pass. `--rerun`
//! therefore always starts from an empty
//! [`MemoTable`]. Re-executing is the point; a cache
//! hit is the failure mode.
//!
//! **Two store handles, deliberately.** The re-execution's stage handlers
//! hold the store through `Rc<RefCell<_>>` to write results, while
//! `verify_object` wants `&Store` to read ledgers and levels — borrowing the
//! same `RefCell` for both at once would panic. A second
//! [`FileStore`](sos_store::FileStore) over the same path is safe here for
//! reasons specific to that type: it caches no objects (every `get_raw` and
//! `object_ids` reads the directory), and writes are content-addressed and
//! first-wins, so the reading handle sees the re-run's output the moment it
//! lands and no write can conflict with another.
//!
//! Certification uses [`NoCertifier`], which refuses
//! every `L2`/`L1` node rather than certifying one no backend examined. All
//! three backends `sos run` can bind are `L3`, so this passes for anything it
//! produced — and a study that somehow contained an `L2` node is *told* so
//! instead of quietly passing.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{EnvRecord, ObjectId, SemVer};
use sos_repro::{EnvLock, NoCertifier, verify_object};
use sos_scirust::stage::{CatalogStageHandler, SpectrumStageHandler, WelchStageHandler};
use sos_store::TypedStore;
use sos_workflow::{Dispatch, MemoTable, StageExecutor};

use crate::args::Args;
use crate::error::Result;
use crate::header::GenericHeader;
use crate::run::{
    CATALOG_PLUGIN, SPECTRUM_PLUGIN, StageConfig, WELCH_PLUGIN, load_registry, parse_grant,
    signal_backend_version, sim_backend_version,
};
use crate::store;

/// Run `sos verify --rerun <object> <runs.json>` — re-execute the run that
/// produced `id` and check the reproduction contract.
///
/// # Errors
/// [`crate::error::CliError::Usage`] for a missing argument or an unreadable
/// file; [`crate::error::CliError::Repro`] if no stored run produced the
/// object, if more than one did, or if the re-run diverges.
pub fn rerun(path: Option<&str>, id: ObjectId, args: &Args) -> Result<String> {
    let runs_path = args
        .flag("rerun")
        .ok_or_else(|| crate::error::CliError::Usage("--rerun needs a runs.json".to_owned()))?;
    let configs: Vec<StageConfig> = serde_json::from_slice(&std::fs::read(runs_path)?)?;
    let registry = load_registry(args.flag("plugins"))?;
    let grant = parse_grant(args.flag("allow"))?;

    let root = store::resolve_root(path)?;
    // See the module docs: a *second* handle, because the handlers below hold
    // the first one mutably for the whole re-execution.
    let reader = store::open(&root)?;
    let writer = Rc::new(RefCell::new(store::open(&root)?));

    let mut models = CatalogStageHandler::new(Rc::clone(&writer), sim_backend_version());
    let mut spectra = SpectrumStageHandler::new(Rc::clone(&writer), signal_backend_version());
    let mut welch = WelchStageHandler::new(Rc::clone(&writer), signal_backend_version());
    for config in configs
    {
        match config
        {
            StageConfig::Model(run) => drop(models.offer(run)),
            StageConfig::Spectrum(c) => drop(spectra.offer(c)),
            StageConfig::Welch(c) => drop(welch.offer(c)),
        }
    }
    let mut dispatch = Dispatch::new(&registry, grant);
    dispatch.register(CATALOG_PLUGIN, Box::new(models) as Box<dyn StageExecutor>);
    dispatch.register(SPECTRUM_PLUGIN, Box::new(spectra) as Box<dyn StageExecutor>);
    dispatch.register(WELCH_PLUGIN, Box::new(welch) as Box<dyn StageExecutor>);

    // An empty memo, always. Reusing `sos run`'s persisted one would replay
    // the recorded outputs and verify nothing.
    let mut memo = MemoTable::new();
    let lock = EnvLock::pin(EnvRecord::new(
        "unspecified",
        vec![sos_core::BackendVersion::new(
            "sos-cli",
            SemVer::new(0, 1, 0),
            crate::run::plugin_hash("verify"),
        )],
        "unspecified",
        "unspecified",
    ));

    let report = verify_object(
        id,
        &reader,
        &lock,
        &mut memo,
        &mut dispatch,
        &mut NoCertifier,
    )?;

    let mut out = format!(
        "{id}\n  re-executed: {} node(s)\n  contract level: {}\n",
        report.nodes.len(),
        report.level.code()
    );
    for node in &report.nodes
    {
        out.push_str(&format!(
            "  {} {} {:?}\n",
            node.node,
            node.level.code(),
            node.verdict
        ));
    }
    out.push_str(
        if report.reproduced()
        {
            "  REPRODUCED (every node matched at its declared level)"
        }
        else
        {
            "  DID NOT REPRODUCE — see the node verdicts above"
        },
    );
    Ok(out)
}

/// Run `sos verify [path] <object>`.
///
/// # Errors
/// [`crate::error::CliError::NotFound`] if no object is stored at the given id.
pub fn run(path: Option<&str>, id: ObjectId) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let header = store::header_of(&s, id)?;

    let mut out = format!(
        "{id}\n  kind: {}\n  level: {}\n  parents: {}\n  author: {:?}\n",
        header.kind,
        header.level.code(),
        header.parents.len(),
        header.author
    );
    out.push_str(&typed_check(&s, &header));
    Ok(out)
}

/// Attempt a typed content-hash verification for every recognized kind.
fn typed_check<S: sos_store::ObjectStore>(s: &S, header: &GenericHeader) -> String {
    macro_rules! check {
        ($ty:ty) => {
            match s.get_object::<$ty>(header.id)
            {
                Ok(Some(obj)) => format!("  content hash: {}", verdict(obj.verify_id())),
                Ok(None) => "  content hash: object vanished mid-check".to_owned(),
                Err(e) => format!("  content hash: could not verify — {e}"),
            }
        };
    }

    match header.kind.name.as_str()
    {
        "Derivation" => check!(sos_reasoning::Derivation),
        "Contradiction" => check!(sos_reasoning::Contradiction),
        "Theory" => check!(sos_theory::Theory),
        "RunLedger" => check!(sos_workflow::RunLedger),
        "EnvLock" => check!(sos_repro::EnvLock),
        // Two crates ship a `Plan`; they declare distinct kinds so a record
        // can only ever be read as the type that wrote it.
        "ExperimentPlan" => check!(sos_planner::Plan),
        "Plan" => check!(sos_workflow::Plan),
        "Publication" => check!(sos_publication::Publication),
        "ReleaseManifest" => check!(sos_publication::ReleaseManifest),
        "Proposal" => check!(sos_ccos::Proposal),
        "Admission" => check!(sos_ccos::Admission),
        "Edge" => check!(sos_knowledge::Edge),
        "ScientificQuestion" => check!(sos_curiosity::ScientificQuestion),
        "CuriosityPolicy" => check!(sos_curiosity::CuriosityPolicy),
        // The kinds `sos run` itself produces. Their absence meant this
        // command could not verify the objects the same binary had just
        // written — reporting "unrecognized kind" for its own output.
        "OdeTrajectory" => check!(sos_scirust::stage::TrajectoryBody),
        "ModeledTrajectory" => check!(sos_scirust::model::ModeledTrajectoryBody),
        "Spectrum" => check!(sos_scirust::spectrum::SpectrumBody),
        "AveragedSpectrum" => check!(sos_scirust::spectrum::AveragedSpectrumBody),
        other => format!("  content hash: not checked (unrecognized kind `{other}`)"),
    }
}

/// A short verdict label.
fn verdict(ok: bool) -> &'static str {
    if ok
    {
        "OK (recomputed address matches)"
    }
    else
    {
        "MISMATCH — tampered or corrupted"
    }
}
