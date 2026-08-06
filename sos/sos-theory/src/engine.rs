//! The [`TheoryEngine`] trait and the store-backed [`Theories`] engine:
//! revision, revision lineage, and rival comparison over a shared scope.

use std::collections::BTreeSet;

use std::collections::BTreeMap;

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
    /// Retained **evidential balance**: `|supporting| − |contradicting|` —
    /// the ranking this crate can compute alone, from the graph.
    EvidentialBalance,
    /// **Weight of evidence**, in millibans, as supplied by the statistics
    /// backend (`sos-scirust`). This crate ranks by it; it never computes it
    /// (Invariant VIII). See [`WeightOfEvidence`] and
    /// [`TheoryEngine::compare_by_evidence`].
    PosteriorOdds,
}

/// One rival's weight of evidence, in **millibans**, computed elsewhere.
///
/// A *ban* is `log₁₀` of a Bayes factor — Good and Turing's unit for weight
/// of evidence — so a milliban is a thousandth of that, and the integer
/// keeps the value float-free for canonical hashing exactly as
/// `sos-planner` keeps expected information gain in millibits. Positive
/// favours the theory; `0` is genuinely neutral evidence, which is why a
/// *missing* weight is an error rather than a zero (see
/// [`TheoryEngine::compare_by_evidence`]).
///
/// This crate never constructs one from data. `sos-scirust`'s `bayes` module
/// does, from real likelihoods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightOfEvidence {
    /// The theory this weight is for.
    pub theory: ObjectId,
    /// `1000 · log₁₀(Bayes factor)`. Positive favours `theory`.
    pub milli_bans: i64,
}

impl WeightOfEvidence {
    /// Record `milli_bans` of evidence for `theory`.
    #[must_use]
    pub fn new(theory: ObjectId, milli_bans: i64) -> Self {
        Self { theory, milli_bans }
    }
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
    /// `supporting − contradicting` (the value ranked by under
    /// [`RankBasis::EvidentialBalance`]), saturating.
    pub net: i64,
    /// The supplied weight of evidence in millibans, when ranked under
    /// [`RankBasis::PosteriorOdds`]. `None` under any other basis — absent
    /// rather than `0`, since neutral evidence and no evidence are different
    /// claims.
    #[serde(default)]
    pub weight_milli_bans: Option<i64>,
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

    /// Rank `rivals` over `scope` by **supplied** weight of evidence rather
    /// than by retained counts — the Bayes-factor ranking RFC-0002 §07.3
    /// calls for, with the numerics where Invariant VIII puts them.
    ///
    /// The scope restriction is identical to [`compare`](Self::compare):
    /// rivals whose domain does not cover `scope` are **excluded, not
    /// penalized**, which is what "posterior odds restricted to the shared
    /// domain" means — a theory that never claimed to apply here is not
    /// losing an argument it did not enter.
    ///
    /// Every in-scope rival must have a weight. A missing one is an error,
    /// not a zero: `0` millibans asserts the evidence is *neutral*, and
    /// silently promoting "nobody computed this" into that claim is exactly
    /// the kind of fabricated verdict this crate refuses elsewhere.
    ///
    /// # Errors
    /// [`TheoryError::UnknownTheory`] / [`TheoryError::Store`] if a rival
    /// cannot be loaded; [`TheoryError::MissingWeight`] if an in-scope rival
    /// has no supplied weight; [`TheoryError::WeightForNonRival`] if a weight
    /// names a theory that is not among `rivals`;
    /// [`TheoryError::DuplicateWeight`] if one is supplied twice.
    fn compare_by_evidence(
        &self,
        rivals: &[ObjectId],
        scope: &Scope,
        weights: &[WeightOfEvidence],
    ) -> Result<Ranking>;
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
                weight_milli_bans: None,
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

    fn compare_by_evidence(
        &self,
        rivals: &[ObjectId],
        scope: &Scope,
        weights: &[WeightOfEvidence],
    ) -> Result<Ranking> {
        let mut supplied: BTreeMap<ObjectId, i64> = BTreeMap::new();
        for w in weights
        {
            if !rivals.contains(&w.theory)
            {
                return Err(TheoryError::WeightForNonRival(w.theory));
            }
            if supplied.insert(w.theory, w.milli_bans).is_some()
            {
                return Err(TheoryError::DuplicateWeight(w.theory));
            }
        }

        let mut ranked = Vec::new();
        for &id in rivals
        {
            let theory = self.get(id)?;
            // Identical scope restriction to `compare`: excluded, not penalized.
            if !theory.domain_of_validity.contains(scope)
            {
                continue;
            }
            let milli_bans = *supplied.get(&id).ok_or(TheoryError::MissingWeight(id))?;
            let supporting = theory.supporting.len();
            let contradicting = theory.contradicting.len();
            ranked.push(RankedTheory {
                theory: id,
                supporting,
                contradicting,
                net: i64::try_from(supporting)
                    .unwrap_or(i64::MAX)
                    .saturating_sub(i64::try_from(contradicting).unwrap_or(i64::MAX)),
                weight_milli_bans: Some(milli_bans),
            });
        }
        // Deterministic: strongest evidence first, ties broken by id ascending.
        ranked.sort_by(|a, b| {
            b.weight_milli_bans
                .cmp(&a.weight_milli_bans)
                .then(a.theory.cmp(&b.theory))
        });
        Ok(Ranking {
            basis: RankBasis::PosteriorOdds,
            scope: scope.clone(),
            ranked,
        })
    }
}
