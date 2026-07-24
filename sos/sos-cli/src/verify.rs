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
//! landed so far. An unrecognized kind still gets the structural report,
//! honestly labelled as such rather than silently skipped.

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
        "Plan" => check!(sos_planner::Plan),
        "Publication" => check!(sos_publication::Publication),
        "ReleaseManifest" => check!(sos_publication::ReleaseManifest),
        "Proposal" => check!(sos_ccos::Proposal),
        "Admission" => check!(sos_ccos::Admission),
        "Edge" => check!(sos_knowledge::Edge),
        "ScientificQuestion" => check!(sos_curiosity::ScientificQuestion),
        "CuriosityPolicy" => check!(sos_curiosity::CuriosityPolicy),
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
