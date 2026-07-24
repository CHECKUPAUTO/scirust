//! `sos diff` — compare two studies by their reachable ancestor sets.
//!
//! Reuses [`sos_publication`]'s dependency-closure walker rather than
//! reimplementing a graph traversal: the same "everything this object was
//! derived from" primitive the Publication Engine uses to fix a document's
//! declared scope is exactly what a study-vs-study diff needs.

use std::collections::BTreeSet;

use sos_core::ObjectId;
use sos_publication::{StoreSource, dependency_closure};

use crate::error::Result;
use crate::store;

/// Run `sos diff <path> <root-a> <root-b>`: report the objects reachable
/// (by provenance ancestry) from `root_a` but not `root_b`, and vice versa.
///
/// # Errors
/// [`crate::error::CliError::Store`] if the store cannot be opened;
/// [`crate::error::CliError::Source`] if the graph cannot be walked.
pub fn run(path: Option<&str>, root_a: ObjectId, root_b: ObjectId) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let source = StoreSource::new(&s);

    let a = dependency_closure(&source, &[root_a])?;
    let b = dependency_closure(&source, &[root_b])?;

    let only_a: BTreeSet<_> = a.difference(&b).collect();
    let only_b: BTreeSet<_> = b.difference(&a).collect();

    let mut out = String::new();
    out.push_str(&format!("Only in {root_a} ({} object(s)):\n", only_a.len()));
    for id in &only_a
    {
        out.push_str(&format!("  - {id}\n"));
    }
    out.push_str(&format!("Only in {root_b} ({} object(s)):\n", only_b.len()));
    for id in &only_b
    {
        out.push_str(&format!("  + {id}\n"));
    }
    out.push_str(&format!("Shared: {} object(s)", a.intersection(&b).count()));
    Ok(out)
}
