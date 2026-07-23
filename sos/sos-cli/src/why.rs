//! `sos why` — print the provenance that justifies an object: "why do we
//! believe this?"

use sos_core::ObjectId;
use sos_provenance::ProvenanceGraph;

use crate::error::Result;
use crate::store;

/// Run `sos why [path] <object>`: print every ancestor of `id` (the transitive
/// closure over provenance parents) — the full chain of derivation and
/// evidence behind it.
///
/// # Errors
/// [`crate::error::CliError::Provenance`] if the store's headers cannot be
/// walked.
pub fn run(path: Option<&str>, id: ObjectId) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let graph = ProvenanceGraph::build(&s)?;

    let ancestors = graph.ancestors(id);
    if ancestors.is_empty()
    {
        return Ok(format!("{id} has no recorded ancestors (it is a root)"));
    }

    let mut out = format!("{id} is justified by {} ancestor(s):\n", ancestors.len());
    for a in &ancestors
    {
        let header = store::header_of(&s, *a)?;
        out.push_str(&format!("  {a}  [{}]\n", header.kind));
    }
    out.pop();
    Ok(out)
}
