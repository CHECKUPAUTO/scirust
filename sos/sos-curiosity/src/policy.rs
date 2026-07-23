//! Scoring: [`CuriosityPolicy`], the raw [`Features`], and the auditable
//! [`Priority`] it produces.
//!
//! Scoring is **integer, fixed-point, saturating** end-to-end, so a sweep's
//! ranking is bit-exact (`L3`) and portable — the kernel's canonical encoder is
//! deliberately float-free (floats are not portable to hash), and saturating
//! arithmetic means no weight, however large, can overflow-panic. There is no
//! opaque scoring: a [`Priority`] exposes every weighted term (Invariant VI).

use serde::{Deserialize, Serialize};
use sos_core::Body;
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// Fixed-point scale for fractional signals, in parts per million (`1.0` maps to
/// `SCALE`). Declared precision keeps novelty and inverse-cost integer-exact.
pub const SCALE: i64 = 1_000_000;

/// The raw, un-weighted signals a scanner extracts about a candidate question,
/// before a [`CuriosityPolicy`] turns them into a [`Priority`]. Kept separate so
/// scoring is a pure, inspectable function of `(features, policy)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    /// Structural novelty — how weakly-connected the subject is: `SCALE / (1 + degree)`.
    pub novelty: i64,
    /// Contradiction severity — `SCALE` for a contradiction question, else `0`.
    pub contradiction: i64,
    /// Inverse investigation cost — `SCALE / subject_size` (fewer subjects, cheaper).
    pub inv_cost: i64,
    /// Expected information gain — **always `0` here**. EIG/BOED is the Planning
    /// Engine's job (`sos-planner`), deferred per Invariant VIII; the field
    /// exists so a planner-supplied candidate can be scored by the same policy.
    pub info_gain: i64,
}

impl Features {
    /// Derive the fixed-point features from raw structural counts.
    ///
    /// `degree` is the subject's connectivity, `subject_size` the number of nodes
    /// the question concerns (clamped to at least 1), and `is_contradiction`
    /// whether it came from the contradiction lens.
    #[must_use]
    pub fn from_structure(degree: usize, subject_size: usize, is_contradiction: bool) -> Self {
        let degree = i64::try_from(degree).unwrap_or(i64::MAX);
        let subject_size = i64::try_from(subject_size).unwrap_or(i64::MAX).max(1);
        Self {
            novelty: SCALE / degree.saturating_add(1),
            contradiction: if is_contradiction { SCALE } else { 0 },
            inv_cost: SCALE / subject_size,
            info_gain: 0,
        }
    }
}

/// The explicit, versioned scoring policy (RFC-0002 §06.4): how a sweep ranks
/// candidate questions.
///
/// Integer weights keep ranking deterministic and the policy is a
/// content-addressed [`Body`], so *why* SOS prioritizes what it does is itself
/// auditable and citable — not an opaque hunch (Invariant VI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CuriosityPolicy {
    /// Weight on expected information gain (planner-supplied; see [`Features::info_gain`]).
    pub w_info_gain: i64,
    /// Weight on structural novelty.
    pub w_novelty: i64,
    /// Weight on contradiction severity.
    pub w_contradiction: i64,
    /// Weight on inverse cost.
    pub w_inv_cost: i64,
    /// Whether cognitive-backend proposals are admitted (they are still scored by
    /// this same policy). `sos-ccos` supplies such proposals; absent that backend
    /// the flag has no effect on this crate's purely graph-driven sweep.
    pub allow_cognitive_proposals: bool,
}

impl Default for CuriosityPolicy {
    /// A principled priority ordering: **severity > information-gain > novelty >
    /// inverse-cost**. The weights are deliberately spaced so a contradiction
    /// (severity `SCALE`) outranks *any* purely structural signal — the default
    /// agenda resolves known inconsistencies before chasing merely isolated or
    /// cheap questions. Every fractional feature is bounded by `SCALE`, so with
    /// `w_contradiction = 4` and the structural weights summing to `3`, a
    /// contradiction's term (`4·SCALE`) exceeds the maximum structural total
    /// (`3·SCALE`). Cognitive proposals are admitted (and still scored here).
    fn default() -> Self {
        Self {
            w_info_gain: 3,
            w_novelty: 2,
            w_contradiction: 4,
            w_inv_cost: 1,
            allow_cognitive_proposals: true,
        }
    }
}

impl CuriosityPolicy {
    /// Score `features` into an auditable [`Priority`] with saturating integer
    /// arithmetic — deterministic and overflow-proof for any weights.
    #[must_use]
    pub fn score(&self, features: Features) -> Priority {
        let info_gain = self.w_info_gain.saturating_mul(features.info_gain);
        let novelty = self.w_novelty.saturating_mul(features.novelty);
        let contradiction = self.w_contradiction.saturating_mul(features.contradiction);
        let inv_cost = self.w_inv_cost.saturating_mul(features.inv_cost);
        let total = info_gain
            .saturating_add(novelty)
            .saturating_add(contradiction)
            .saturating_add(inv_cost);
        Priority {
            info_gain,
            novelty,
            contradiction,
            inv_cost,
            total,
        }
    }
}

impl Canonical for CuriosityPolicy {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.i64(self.w_info_gain);
        enc.i64(self.w_novelty);
        enc.i64(self.w_contradiction);
        enc.i64(self.w_inv_cost);
        enc.bool(self.allow_cognitive_proposals);
    }
}

impl Body for CuriosityPolicy {
    const KIND: &'static str = "CuriosityPolicy";
    const SCHEMA_VERSION: u32 = 1;
}

/// An auditable score: the weighted contribution of each term plus the `total`.
///
/// Ranking is by `total` (ties broken by the sweep on the question's canonical
/// bytes), and every term is exposed so a score can be explained and challenged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority {
    /// Weighted information-gain term.
    pub info_gain: i64,
    /// Weighted novelty term.
    pub novelty: i64,
    /// Weighted contradiction term.
    pub contradiction: i64,
    /// Weighted inverse-cost term.
    pub inv_cost: i64,
    /// The sum of the four weighted terms — the value questions are ranked by.
    pub total: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contradiction_outranks_a_bare_leaf_under_default_policy() {
        let p = CuriosityPolicy::default();
        // A contradiction question about two well-connected nodes...
        let contra = p.score(Features::from_structure(5, 2, true));
        // ...versus an under-connected question about one isolated node.
        let leaf = p.score(Features::from_structure(0, 1, false));
        // The default policy spaces the weights so a contradiction outranks even
        // a maximally-isolated single node.
        assert!(contra.total > leaf.total);
        assert!(contra.contradiction > 0);
        assert_eq!(leaf.contradiction, 0);
    }

    #[test]
    fn weights_are_honoured_and_saturating() {
        let p = CuriosityPolicy {
            w_contradiction: 10,
            ..CuriosityPolicy::default()
        };
        let s = p.score(Features::from_structure(1, 2, true));
        assert_eq!(s.contradiction, 10 * SCALE);
        // Adversarial weight cannot overflow-panic.
        let sat_policy = CuriosityPolicy {
            w_contradiction: i64::MAX,
            ..CuriosityPolicy::default()
        };
        let sat = sat_policy.score(Features::from_structure(1, 2, true));
        assert_eq!(sat.contradiction, i64::MAX);
        assert_eq!(sat.total, i64::MAX);
    }

    #[test]
    fn info_gain_is_zero_from_structural_features() {
        let f = Features::from_structure(3, 1, true);
        assert_eq!(f.info_gain, 0);
    }

    #[test]
    fn policy_is_canonical_and_content_addressed() {
        let a = CuriosityPolicy::default();
        let b = CuriosityPolicy {
            w_contradiction: 2,
            ..CuriosityPolicy::default()
        };
        assert_eq!(
            a.canonical_bytes(),
            CuriosityPolicy::default().canonical_bytes()
        );
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }
}
