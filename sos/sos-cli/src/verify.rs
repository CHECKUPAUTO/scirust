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
//! ## Certifying an `L2` node
//!
//! An `L2` node cannot be verified the way an `L3` one is. `L3` means
//! bit-identical, so comparing object ids settles it. `L2` means *within a
//! declared tolerance*, and two adaptive runs that both meet `rtol` need not
//! agree bit for bit — demanding they do would fail correct reproductions.
//!
//! So this command carries a [`TrajectoryCertifier`]: it loads both certified
//! trajectories and asks `sos-scirust` whether they agree within the tolerance
//! the original certified itself to. The backend that made the claim is the
//! one that judges it. Anything it cannot certify is refused rather than waved
//! through — [`NoCertifier`](sos_repro::NoCertifier)'s behaviour, kept for
//! every kind this CLI does not understand.
//!
//! That branch was unreachable until the adaptive catalogue backend existed:
//! every CLI-bindable backend was `L3`, so `NoCertifier` was the honest
//! choice. It is no longer.

use std::cell::RefCell;
use std::rc::Rc;

use sos_core::{DeterminismLevel, EnvRecord, ObjectId, SemVer};
use sos_repro::{Certifier, EnvLock, Reproduced, verify_object};
use sos_scirust::model::CertifiedModeledTrajectoryBody;
use sos_scirust::ode::CertifiedTrajectory;
use sos_scirust::stage::{
    AdaptiveCatalogStageHandler, CatalogStageHandler, SpectrumStageHandler,
    TrajectorySpectrogramStageHandler, TrajectorySpectrumStageHandler, WelchStageHandler,
};
use sos_scirust::verdict::{certify_trajectory_endpoint, certify_trajectory_pointwise};
use sos_store::TypedStore;
use sos_workflow::{Dispatch, MemoTable, StageExecutor};

use crate::args::Args;
use crate::error::Result;
use crate::header::GenericHeader;
use crate::run::{
    ADAPTIVE_CATALOG_PLUGIN, CATALOG_PLUGIN, SPECTROGRAM_PLUGIN, SPECTRUM_PLUGIN, StageConfig,
    TRAJECTORY_PLUGIN, WELCH_PLUGIN, load_registry, parse_grant, signal_backend_version,
    sim_backend_version,
};
use crate::store;

/// A [`Certifier`] that judges an `L2` node by the tolerance it declared.
///
/// [`NoCertifier`] refuses every `L2`/`L1` node, which was the honest answer
/// while no `L2` result was reachable from `sos run`: certifying a node no
/// backend had examined would be a lie. Now that the adaptive catalogue
/// backend exists, that branch of the reproduction contract is real, and
/// refusing it would make `--rerun` unable to verify the one kind of result
/// that most needs it.
///
/// An `L2` node cannot be verified the way an `L3` one is. Comparing object
/// ids is exactly wrong: two adaptive runs that both meet `rtol` need not be
/// bit-identical, and demanding they be would fail correct reproductions.
/// This loads both trajectories and asks `sos-scirust` whether they agree
/// **within the tolerance the original certified itself to** — the backend
/// that made the claim is the one that gets to judge it.
///
/// Anything that is not a kind this CLI can certify is still refused rather
/// than waved through.
struct TrajectoryCertifier<'s, S> {
    store: &'s S,
}

impl<S: sos_store::ObjectStore> Certifier for TrajectoryCertifier<'_, S> {
    fn certify(
        &mut self,
        stage: &sos_workflow::StageId,
        level: DeterminismLevel,
        original: ObjectId,
        rerun: ObjectId,
    ) -> core::result::Result<bool, String> {
        if level != DeterminismLevel::L2
        {
            return Err(format!(
                "stage {stage} produced a {level:?} node; this CLI can certify L2                  trajectories only"
            ));
        }
        let load = |id: ObjectId| -> core::result::Result<CertifiedTrajectory, String> {
            self.store
                .get_object::<CertifiedModeledTrajectoryBody>(id)
                .map_err(|e| format!("{id}: {e}"))?
                .map(|o| o.body.certified)
                .ok_or_else(|| format!("{id} is not in the store"))
        };
        let (a, b) = (load(original)?, load(rerun)?);
        // Pointwise where the grids line up; the endpoint otherwise, since two
        // adaptive runs are entitled to choose different grids.
        let agreement =
            match certify_trajectory_pointwise(&a, &b)
            {
                Ok(agreement) => agreement,
                Err(_) => certify_trajectory_endpoint(&a, &b)
                    .map_err(|e| format!("stage {stage}: {e}"))?,
            };
        Ok(matches!(agreement.verdict(), Reproduced::Certified(true)))
    }
}

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
    let mut trajectories =
        TrajectorySpectrumStageHandler::new(Rc::clone(&writer), signal_backend_version());
    let mut adaptive = AdaptiveCatalogStageHandler::new(Rc::clone(&writer), sim_backend_version());
    let mut spectrograms =
        TrajectorySpectrogramStageHandler::new(Rc::clone(&writer), signal_backend_version());
    for config in configs
    {
        match config
        {
            StageConfig::Model(run) => drop(models.offer(run)),
            StageConfig::Spectrum(c) => drop(spectra.offer(c)),
            StageConfig::Welch(c) => drop(welch.offer(c)),
            StageConfig::Trajectory(c) => drop(trajectories.offer(c)),
            StageConfig::AdaptiveModel(r) => drop(adaptive.offer(r)),
            StageConfig::Spectrogram(c) => drop(spectrograms.offer(c)),
        }
    }
    let mut dispatch = Dispatch::new(&registry, grant);
    dispatch.register(CATALOG_PLUGIN, Box::new(models) as Box<dyn StageExecutor>);
    dispatch.register(SPECTRUM_PLUGIN, Box::new(spectra) as Box<dyn StageExecutor>);
    dispatch.register(WELCH_PLUGIN, Box::new(welch) as Box<dyn StageExecutor>);
    dispatch.register(
        TRAJECTORY_PLUGIN,
        Box::new(trajectories) as Box<dyn StageExecutor>,
    );
    dispatch.register(
        ADAPTIVE_CATALOG_PLUGIN,
        Box::new(adaptive) as Box<dyn StageExecutor>,
    );
    dispatch.register(
        SPECTROGRAM_PLUGIN,
        Box::new(spectrograms) as Box<dyn StageExecutor>,
    );

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

    let mut certifier = TrajectoryCertifier { store: &reader };
    let report = verify_object(id, &reader, &lock, &mut memo, &mut dispatch, &mut certifier)?;

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
        "TrajectorySpectrum" => check!(sos_scirust::pipeline::TrajectorySpectrumBody),
        "TrajectorySpectrogram" => check!(sos_scirust::pipeline::TrajectorySpectrogramBody),
        "CertifiedModeledTrajectory" =>
        {
            check!(sos_scirust::model::CertifiedModeledTrajectoryBody)
        },
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
