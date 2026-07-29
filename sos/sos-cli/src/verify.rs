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
//! ## What this is not, yet
//!
//! RFC-0002 §10.4 describes `sos verify <object>` as "re-execute and check the
//! reproducibility contract", and this checks *identity*, not reproduction: it
//! answers "is this object what it says it is", not "does running the
//! experiment again produce it". `sos-repro`'s `verify_object` does the
//! latter, and now has what it needs — `sos run` seals its `Plan` and
//! `RunLedger`, which is how that function finds the run that produced an
//! object. Wiring it here additionally needs a second store handle: the
//! re-execution's stage handlers hold the store through `Rc<RefCell<_>>`
//! while `verify_object` wants `&Store` for the same instant, which would
//! panic on the borrow. That is a real API question, not plumbing, so it is
//! left rather than forced.

use sos_core::ObjectId;
use sos_store::TypedStore;

use crate::error::Result;
use crate::header::GenericHeader;
use crate::store;

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
