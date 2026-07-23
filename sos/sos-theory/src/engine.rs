//! The [`TheoryEngine`] trait and the store-backed [`Theories`] engine:
//! revision, revision lineage, and rival comparison over a shared scope.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_store::{ObjectStore, TypedStore};

use crate::error::{Result, TheoryError};
use crate::scope::Scope;
use crate::theory::Theory;

/// The basis a [`Ranking`] was computed on, so a ranking explains itself
/// (Invariant VI — no opaque scoring).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RankBasis {
    /// Retained **evidential balance**: `|supporting| − |contradicting|`. This is
    /// the deterministic ranking available today; Bayes-factor `Confidence`
    /// ranking (posterior odds restricted to the shared domain) awaits the
    /// statistics backend and is deferred per Invariant VIII (RFC-0002 §07.3).
    EvidentialBalance,
}

/// One rival's standing in a [`Ranking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedTheory {
    /// The theory ranked.
    pub theory: ObjectId,
    /// Count of supporting evidence.
    pub supporting: usize,
    /// Count of contradicting evidence.
    pub contradicting: usize,
    /// `supporting − contradicting` (the value ranked by), saturating.
    pub net: i64,
}

/// A comparison of rival theories over a shared [`Scope`], best first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ranking {
    /// What the ranking was computed from.
    pub basis: RankBasis,
    /// The scope the rivals were compared over.
    pub scope: Scope,
    /// The in-scope rivals, ordered best-first (ties broken by id).
    pub ranked: Vec<RankedTheory>,
}

/// The Theory Engine syscall surface (RFC-0002 §07.3).
///
/// The `discriminating_experiment` operation — "which experiment best separates
/// two rivals" — is *not* here: it is an expected-information-gain query to the
/// Planning Engine (`sos-planner`), deferred per Invariant VIII. This trait ships
/// the operations that are fully deterministic over the store.
pub trait TheoryEngine {
    /// Build a successor revising `parent_id`, forced by `forced_by`. Loads the
    /// parent and delegates to [`Theory::revise`].
    ///
    /// # Errors
    /// [`TheoryError::UnknownTheory`] if `parent_id` is absent, or
    /// [`TheoryError::Store`] on a backend failure.
    fn revise(&self, parent_id: ObjectId, forced_by: &[ObjectId]) -> Result<Theory>;

    /// The revision lineage from `id` back to its root ancestor, newest first,
    /// following [`Theory::revises`]. The whole chain stays queryable — old
    /// theories are never deleted.
    ///
    /// # Errors
    /// [`TheoryError::UnknownTheory`] / [`TheoryError::Store`] if any theory in
    /// the chain cannot be loaded.
    fn revision_chain(&self, id: ObjectId) -> Result<Vec<ObjectId>>;

    /// Rank `rivals` over `scope`, keeping only those that **claim validity
    /// there** (their domain contains the scope), ordered by
    /// [`RankBasis::EvidentialBalance`] (ties by id ascending). Rivals whose
    /// domain does not cover `scope` are excluded, not penalized.
    ///
    /// # Errors
    /// [`TheoryError::UnknownTheory`] / [`TheoryError::Store`] if any rival cannot
    /// be loaded.
    fn compare(&self, rivals: &[ObjectId], scope: &Scope) -> Result<Ranking>;
}

/// A deterministic Theory Engine backed by an [`ObjectStore`].
#[derive(Debug, Clone, Copy)]
pub struct Theories<'s, S: ObjectStore + ?Sized> {
    store: &'s S,
}

impl<'s, S: ObjectStore + ?Sized> Theories<'s, S> {
    /// Create a theory engine over `store`.
    #[must_use]
    pub fn new(store: &'s S) -> Self {
        Self { store }
    }

    /// Load a [`Theory`] body by id.
    ///
    /// # Errors
    /// [`TheoryError::UnknownTheory`] if absent, [`TheoryError::Store`] on a
    /// backend or integrity failure.
    pub fn get(&self, id: ObjectId) -> Result<Theory> {
        match self.store.get_object::<Theory>(id)?
        {
            Some(obj) => Ok(obj.body),
            None => Err(TheoryError::UnknownTheory(id)),
        }
    }
}

impl<S: ObjectStore + ?Sized> TheoryEngine for Theories<'_, S> {
    fn revise(&self, parent_id: ObjectId, forced_by: &[ObjectId]) -> Result<Theory> {
        let parent = self.get(parent_id)?;
        Ok(parent.revise(parent_id, forced_by))
    }

    fn revision_chain(&self, id: ObjectId) -> Result<Vec<ObjectId>> {
        let mut chain = vec![id];
        let mut seen = BTreeSet::from([id]);
        let mut current = self.get(id)?;
        while let Some(parent) = current.revises
        {
            // Guard against a cycle (a well-formed lineage is a DAG; this keeps a
            // corrupted store from looping forever).
            if !seen.insert(parent)
            {
                break;
            }
            chain.push(parent);
            current = self.get(parent)?;
        }
        Ok(chain)
    }

    fn compare(&self, rivals: &[ObjectId], scope: &Scope) -> Result<Ranking> {
        let mut ranked = Vec::new();
        for &id in rivals
        {
            let theory = self.get(id)?;
            // Only rank rivals that claim validity across the queried scope.
            if !theory.domain_of_validity.contains(scope)
            {
                continue;
            }
            let supporting = theory.supporting.len();
            let contradicting = theory.contradicting.len();
            let net = i64::try_from(supporting)
                .unwrap_or(i64::MAX)
                .saturating_sub(i64::try_from(contradicting).unwrap_or(i64::MAX));
            ranked.push(RankedTheory {
                theory: id,
                supporting,
                contradicting,
                net,
            });
        }
        // Deterministic: best (highest net) first, ties broken by id ascending.
        ranked.sort_by(|a, b| b.net.cmp(&a.net).then(a.theory.cmp(&b.theory)));
        Ok(Ranking {
            basis: RankBasis::EvidentialBalance,
            scope: scope.clone(),
            ranked,
        })
    }
}
