//! [`diff`] — a semantic difference between two publications.
//!
//! A byte diff of two sealed objects tells you only *that* they differ. Review
//! needs to know *what* changed: which claims were added, dropped, or reworded;
//! which exhibits changed recipe; whether the declared scope or the governing
//! policy moved. [`PublicationDiff`] answers that at the level of the registries,
//! comparing claims by their content address (so a reordering is not a change)
//! and exhibits by their full spec. It is the reviewer's companion to
//! [`check_release`](crate::verify::check_release)'s yes/no.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::key::{ClaimKey, FigureKey, TableKey};
use crate::publication::Publication;

/// The semantic difference between two publications (`old` → `new`).
///
/// Membership lists (`*_added`, `*_removed`) are keyed; `*_changed` lists entries
/// present under the same key in both but with different content. All lists are
/// sorted by key for a deterministic, reviewable result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationDiff {
    /// Whether the front-matter (title, subtitle, authors, abstract) changed.
    pub meta_changed: bool,
    /// Whether the ordered section body changed.
    pub sections_changed: bool,
    /// Claims present in `new` but not `old`.
    pub claims_added: Vec<ClaimKey>,
    /// Claims present in `old` but not `new`.
    pub claims_removed: Vec<ClaimKey>,
    /// Claims under the same key whose content address changed (restated or
    /// rebound).
    pub claims_changed: Vec<ClaimKey>,
    /// Figures added in `new`.
    pub figures_added: Vec<FigureKey>,
    /// Figures removed from `old`.
    pub figures_removed: Vec<FigureKey>,
    /// Figures under the same key whose spec changed.
    pub figures_changed: Vec<FigureKey>,
    /// Tables added in `new`.
    pub tables_added: Vec<TableKey>,
    /// Tables removed from `old`.
    pub tables_removed: Vec<TableKey>,
    /// Tables under the same key whose spec changed.
    pub tables_changed: Vec<TableKey>,
    /// Whether the declared root set changed.
    pub declared_roots_changed: bool,
    /// Whether the governing support policy changed.
    pub policy_changed: bool,
    /// Whether the reproducibility bar changed.
    pub reproducibility_changed: bool,
}

impl PublicationDiff {
    /// Whether the two publications are semantically identical.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.meta_changed
            && !self.sections_changed
            && self.claims_added.is_empty()
            && self.claims_removed.is_empty()
            && self.claims_changed.is_empty()
            && self.figures_added.is_empty()
            && self.figures_removed.is_empty()
            && self.figures_changed.is_empty()
            && self.tables_added.is_empty()
            && self.tables_removed.is_empty()
            && self.tables_changed.is_empty()
            && !self.declared_roots_changed
            && !self.policy_changed
            && !self.reproducibility_changed
    }
}

/// Compute the semantic difference from `old` to `new`.
#[must_use]
pub fn diff(old: &Publication, new: &Publication) -> PublicationDiff {
    // Claims are compared by content address, so reordering is not a change.
    let old_claims: BTreeMap<ClaimKey, sos_core::ObjectId> = old
        .claims
        .iter()
        .map(|c| (c.key.clone(), c.content_id()))
        .collect();
    let new_claims: BTreeMap<ClaimKey, sos_core::ObjectId> = new
        .claims
        .iter()
        .map(|c| (c.key.clone(), c.content_id()))
        .collect();
    let (claims_added, claims_removed, claims_changed) =
        keyed_diff(&old_claims, &new_claims, |a, b| a == b);

    let old_figures: BTreeMap<FigureKey, &_> =
        old.figures.iter().map(|f| (f.key.clone(), f)).collect();
    let new_figures: BTreeMap<FigureKey, &_> =
        new.figures.iter().map(|f| (f.key.clone(), f)).collect();
    let (figures_added, figures_removed, figures_changed) =
        keyed_diff(&old_figures, &new_figures, |a, b| a == b);

    let old_tables: BTreeMap<TableKey, &_> =
        old.tables.iter().map(|t| (t.key.clone(), t)).collect();
    let new_tables: BTreeMap<TableKey, &_> =
        new.tables.iter().map(|t| (t.key.clone(), t)).collect();
    let (tables_added, tables_removed, tables_changed) =
        keyed_diff(&old_tables, &new_tables, |a, b| a == b);

    PublicationDiff {
        meta_changed: old.meta != new.meta,
        sections_changed: old.sections != new.sections,
        claims_added,
        claims_removed,
        claims_changed,
        figures_added,
        figures_removed,
        figures_changed,
        tables_added,
        tables_removed,
        tables_changed,
        declared_roots_changed: old.declared_roots != new.declared_roots,
        policy_changed: old.verification_policy != new.verification_policy,
        reproducibility_changed: old.reproducibility != new.reproducibility,
    }
}

/// Split two keyed maps into (added, removed, changed) key lists. `same` decides
/// whether two same-keyed values are unchanged. Keys are `Ord`, so a `BTreeMap`
/// makes every output list sorted and deterministic.
fn keyed_diff<K, V>(
    old: &BTreeMap<K, V>,
    new: &BTreeMap<K, V>,
    same: impl Fn(&V, &V) -> bool,
) -> (Vec<K>, Vec<K>, Vec<K>)
where
    K: Ord + Clone,
{
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (key, new_value) in new
    {
        match old.get(key)
        {
            None => added.push(key.clone()),
            Some(old_value) if !same(old_value, new_value) => changed.push(key.clone()),
            Some(_) =>
            {},
        }
    }
    for key in old.keys()
    {
        if !new.contains_key(key)
        {
            removed.push(key.clone());
        }
    }
    (added, removed, changed)
}
