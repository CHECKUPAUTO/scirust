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
use sos_planner::{Cost, Estimate};

/// Fixed-point scale for fractional signals, in parts per million (`1.0` maps to
/// `SCALE`). Declared precision keeps novelty and inverse-cost integer-exact.
pub const SCALE: i64 = 1_000_000;

/// The raw, un-weighted signals a scanner extracts about a candidate question,
/// before a [`CuriosityPolicy`] turns them into a [`Priority`]. Kept separate so
/// scoring is a pure, inspectable function of `(features, policy)`.
///
/// **Every fractional feature is bounded by [`SCALE`]** — the default policy's
/// priority ordering depends on it ([`CuriosityPolicy::default`]), so any new
/// feature derivation must preserve that bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    /// Structural novelty — how weakly-connected the subject is: `SCALE / (1 + degree)`.
    pub novelty: i64,
    /// Contradiction severity — `SCALE` for a contradiction question, else `0`.
    pub contradiction: i64,
    /// Inverse investigation cost — `SCALE / cost`, where cost is the subject
    /// size for a structural question ([`Features::from_structure`]) and the
    /// planner's real [`Cost::total`] for a design ([`Features::from_design`]).
    pub inv_cost: i64,
    /// Expected information gain, `0` for the purely structural lenses (graph
    /// shape says nothing about how much an experiment would teach) and derived
    /// from a planner-supplied [`Estimate`] by [`Features::from_design`].
    pub info_gain: i64,
}

impl Features {
    /// Structural novelty from a node's connectivity: `SCALE / (1 + degree)`,
    /// so an isolated node reads `SCALE` and a well-connected one tends to `0`.
    fn novelty_from_degree(degree: usize) -> i64 {
        let degree = i64::try_from(degree).unwrap_or(i64::MAX);
        SCALE / degree.saturating_add(1)
    }

    /// Inverse cost: `SCALE / cost`, clamped so a zero/negative cost is treated
    /// as one unit and the result stays bounded by [`SCALE`].
    fn inv_cost_from(cost: i64) -> i64 {
        SCALE / cost.max(1)
    }

    /// Derive the fixed-point features from raw structural counts.
    ///
    /// `degree` is the subject's connectivity, `subject_size` the number of nodes
    /// the question concerns (clamped to at least 1), and `is_contradiction`
    /// whether it came from the contradiction lens.
    #[must_use]
    pub fn from_structure(degree: usize, subject_size: usize, is_contradiction: bool) -> Self {
        Self {
            novelty: Self::novelty_from_degree(degree),
            contradiction: if is_contradiction { SCALE } else { 0 },
            inv_cost: Self::inv_cost_from(i64::try_from(subject_size).unwrap_or(i64::MAX)),
            info_gain: 0,
        }
    }

    /// Derive the features of a **planner-supplied design**: an experiment whose
    /// expected information gain the Planning Engine has actually estimated.
    ///
    /// `degree` is the design node's connectivity in the graph (novelty, exactly
    /// as for a structural question); `eig` and `cost` come from the planner's
    /// [`sos_planner::Candidate`].
    ///
    /// Two honesty properties are deliberate:
    ///
    /// * **The conservative bound, not the point estimate.** EIG is scored from
    ///   [`Estimate::lower_bound`] (point − standard error, floored at `0`) —
    ///   the value `sos-planner` documents for "a planner that wants to avoid
    ///   over-claiming". A noisy `0.9 ± 0.8` bit estimate therefore contributes
    ///   what it can defend, not what it hopes.
    /// * **The `SCALE` bound is preserved** via `eig_saturation_milli`: the EIG,
    ///   in millibits, at or above which this feature reads a full [`SCALE`].
    ///   EIG is unbounded but features must not be, so the saturation point is
    ///   an *explicit, caller-declared* scale rather than a magic constant —
    ///   "2 bits is as interesting as it gets here" is a scientific judgement
    ///   the caller should own and record. Clamped to at least `1` so it can
    ///   never divide by zero.
    #[must_use]
    pub fn from_design(
        degree: usize,
        eig: &Estimate,
        cost: &Cost,
        eig_saturation_milli: i64,
    ) -> Self {
        let saturation = eig_saturation_milli.max(1);
        let defensible = eig.lower_bound().max(0).min(saturation);
        Self {
            novelty: Self::novelty_from_degree(degree),
            contradiction: 0,
            inv_cost: Self::inv_cost_from(cost.total()),
            info_gain: defensible.saturating_mul(SCALE) / saturation,
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
    /// Weight on expected information gain (planner-supplied via
    /// [`Features::from_design`]; see [`Features::info_gain`]).
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
    /// inverse-cost**, with severity spaced so that a contradiction outranks
    /// *any* non-contradiction question — the default agenda resolves known
    /// inconsistencies before chasing merely informative, isolated, or cheap
    /// ones. Cognitive proposals are admitted (and still scored here).
    ///
    /// The spacing is load-bearing, so it is stated as arithmetic. Every
    /// fractional feature is bounded by [`SCALE`], so a non-contradiction
    /// question totals at most `(w_info_gain + w_novelty + w_inv_cost)·SCALE =
    /// 6·SCALE`, while a contradiction scores at least `w_contradiction·SCALE =
    /// 7·SCALE` from its severity term alone. `7 > 6`, so the dominance holds
    /// for every possible feature vector, not merely typical ones (the
    /// `contradiction_dominates_any_non_contradiction` test pins it).
    ///
    /// `w_contradiction` is `7` rather than the `4` this policy shipped with
    /// before the information lens existed. That original spacing compared
    /// severity against `w_novelty + w_inv_cost = 3` and **excluded**
    /// `w_info_gain`, which was sound only while [`Features::info_gain`] was
    /// hardcoded to `0`; once [`Features::from_design`] made it live, `4·SCALE`
    /// no longer dominated the `6·SCALE` ceiling and a sufficiently attractive
    /// design could displace an unresolved contradiction from the top of the
    /// agenda. Raising the weight restores the documented intent rather than
    /// silently changing what the default agenda does.
    fn default() -> Self {
        Self {
            w_info_gain: 3,
            w_novelty: 2,
            w_contradiction: 7,
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
    use sos_core::DeterminismLevel;

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

    /// The default policy's load-bearing spacing, checked as arithmetic rather
    /// than on a handful of sampled feature vectors: because every fractional
    /// feature is bounded by `SCALE`, the worst case for the guarantee is a
    /// non-contradiction question maxing out *every* other feature against a
    /// contradiction with the barest possible structure.
    ///
    /// This is the invariant that quietly broke when `Features::info_gain`
    /// stopped being hardcoded to `0`: the original `w_contradiction = 4` was
    /// spaced only against `w_novelty + w_inv_cost`.
    #[test]
    fn contradiction_dominates_any_non_contradiction() {
        let p = CuriosityPolicy::default();

        // The ceiling for a question that is not a contradiction: every
        // fractional feature saturated at SCALE.
        let best_non_contradiction = p.score(Features {
            novelty: SCALE,
            contradiction: 0,
            inv_cost: SCALE,
            info_gain: SCALE,
        });
        // The floor for a contradiction: severity only, nothing else going for it.
        let worst_contradiction = p.score(Features {
            novelty: 0,
            contradiction: SCALE,
            inv_cost: 0,
            info_gain: 0,
        });

        assert!(
            worst_contradiction.total > best_non_contradiction.total,
            "severity must dominate: worst contradiction {} vs best other {}",
            worst_contradiction.total,
            best_non_contradiction.total
        );

        // And the stated ordering of the weights themselves holds.
        assert!(p.w_contradiction > p.w_info_gain);
        assert!(p.w_info_gain > p.w_novelty);
        assert!(p.w_novelty > p.w_inv_cost);
        // The dominance is exactly "severity exceeds everything else combined".
        assert!(p.w_contradiction > p.w_info_gain + p.w_novelty + p.w_inv_cost);
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
    fn design_info_gain_scales_against_the_declared_saturation() {
        // 1 bit of EIG against a 2-bit saturation ⇒ exactly half of SCALE.
        let half = Features::from_design(
            0,
            &Estimate::exact(1_000),
            &Cost::new(1, 0, 0, 0),
            2_000, // 2 bits
        );
        assert_eq!(half.info_gain, SCALE / 2);

        // At (and beyond) saturation the feature reads a full SCALE, never more —
        // the bound the default policy's ordering depends on.
        let at = Features::from_design(0, &Estimate::exact(2_000), &Cost::new(1, 0, 0, 0), 2_000);
        assert_eq!(at.info_gain, SCALE);
        let beyond =
            Features::from_design(0, &Estimate::exact(500_000), &Cost::new(1, 0, 0, 0), 2_000);
        assert_eq!(beyond.info_gain, SCALE);
    }

    #[test]
    fn design_info_gain_uses_the_conservative_lower_bound() {
        // 0.9 ± 0.8 bits: the defensible claim is 0.1 bits, not 0.9.
        let noisy = Features::from_design(
            0,
            &Estimate::new(900, 800, DeterminismLevel::L1),
            &Cost::new(1, 0, 0, 0),
            1_000,
        );
        assert_eq!(noisy.info_gain, SCALE / 10);

        // The same point estimate, measured exactly, defends the whole claim.
        let exact = Features::from_design(0, &Estimate::exact(900), &Cost::new(1, 0, 0, 0), 1_000);
        assert!(exact.info_gain > noisy.info_gain);
        assert_eq!(exact.info_gain, 900 * SCALE / 1_000);
    }

    #[test]
    fn design_info_gain_never_goes_negative() {
        // 0.02 ± 0.03 bits — the lower bound is below zero, but a feature that
        // went negative would silently *subtract* priority.
        let f = Features::from_design(
            0,
            &Estimate::new(20, 30, DeterminismLevel::L1),
            &Cost::new(1, 0, 0, 0),
            1_000,
        );
        assert_eq!(f.info_gain, 0);
    }

    #[test]
    fn design_inv_cost_uses_the_planners_real_cost() {
        let cheap = Features::from_design(0, &Estimate::exact(500), &Cost::new(1, 0, 0, 0), 1_000);
        let dear = Features::from_design(0, &Estimate::exact(500), &Cost::new(5, 3, 2, 0), 1_000);
        assert_eq!(cheap.inv_cost, SCALE);
        assert_eq!(dear.inv_cost, SCALE / 10); // total() == 10
        assert!(cheap.inv_cost > dear.inv_cost);

        // A free experiment is costed as one unit, never a division by zero.
        let free = Features::from_design(0, &Estimate::exact(500), &Cost::default(), 1_000);
        assert_eq!(free.inv_cost, SCALE);
    }

    #[test]
    fn design_saturation_is_clamped_away_from_zero() {
        // A zero (or negative) saturation must not divide by zero.
        for saturation in [0, -1, i64::MIN]
        {
            let f =
                Features::from_design(0, &Estimate::exact(500), &Cost::new(1, 0, 0, 0), saturation);
            assert_eq!(f.info_gain, SCALE); // everything saturates at scale 1
        }
    }

    #[test]
    fn design_novelty_matches_the_structural_formula() {
        // Novelty is one formula, shared: a design node and a structural node of
        // the same degree are equally novel.
        for degree in [0, 1, 5, 100]
        {
            let d =
                Features::from_design(degree, &Estimate::exact(0), &Cost::new(1, 0, 0, 0), 1_000);
            let s = Features::from_structure(degree, 1, false);
            assert_eq!(d.novelty, s.novelty);
        }
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
