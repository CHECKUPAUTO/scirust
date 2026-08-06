//! Discriminating-experiment planning — *"which experiment best separates two
//! rivals?"*
//!
//! This crate's docs listed the question as deferred because it *"is an
//! expected-information-gain query to `sos-planner`"*. That sentence is also
//! the design: the question splits cleanly in two, and this module answers
//! only its half.
//!
//! | Question | Whose | Answered by |
//! |---|---|---|
//! | Does this design *bear on* the rivals at all? | theory's | here |
//! | How much would it teach, and is that worth its cost? | planner's | [`sos_planner`] |
//!
//! Neither half can answer the other's. A design that would be enormously
//! informative about something irrelevant does not discriminate these rivals;
//! a design that bears on them but teaches nothing is not worth running. So
//! this module states the *bearing*, hands the designs to
//! [`GreedyPlanner`], and returns the planner's ranking **unmodified**.
//!
//! ## What this module does not do, deliberately
//!
//! * **It never estimates expected information gain.** Every
//!   [`Candidate`]'s [`eig`](Candidate::eig) arrives already computed — by
//!   `sos-scirust`, per Invariant VIII. This module has no path that
//!   constructs an estimate, and `sos-theory` does not re-export the type,
//!   so a caller cannot be led into building one here.
//! * **It never re-ranks.** The returned [`sos_planner::Plan`] is exactly
//!   what the planner produced: same order, same utilities, same verdict.
//!   Re-sorting it "by how well it separates the rivals" would be this crate
//!   quietly running its own utility policy.
//! * **It never softens
//!   [`InformationExhausted`](sos_planner::StopVerdict::InformationExhausted).**
//!   If no design clears the planner's floor, that verdict passes through. A
//!   theory engine that answered "well, run this one anyway" would be
//!   overriding the one signal that says the rivals cannot currently be
//!   separated — which is a real scientific finding, not a failure to
//!   produce a recommendation.
//!
//! The dependency lint enforces this boundary mechanically rather than
//! trusting the prose: `sos-theory -> sos-planner` is sanctioned for this
//! purpose alone, and `sos/scripts/lint-deps.py` fails the build if this
//! crate names the planner's stopping rules or its fixed-point utility
//! internals — the items a reimplementation would need.
//!
//! ## Relevance is checked, and it is this crate's job
//!
//! The planner admits designs by *informativeness* (its EIG floor). This
//! module admits them by *relevance*: a candidate must name two distinct
//! rivals from the problem it is submitted against. That is not a second
//! opinion on the planner's rule — it is the question the planner cannot see,
//! because a [`Candidate`] carries an experiment id and an estimate, and
//! nothing about which theories it would tell apart.
//!
//! ## Example
//!
//! ```
//! use sos_core::{HashAlgo, ObjectId};
//! use sos_planner::{Candidate, Cost, Estimate, StopVerdict, UtilityPolicy};
//! use sos_core::DeterminismLevel;
//! use sos_theory::discriminate::{Discrimination, DiscriminatingCandidate, discriminate};
//!
//! let oid = |t: &[u8]| ObjectId::compute(HashAlgo::Sha256, b"doc", t);
//! let (newton, einstein) = (oid(b"newton"), oid(b"einstein"));
//!
//! // Two designs. Their EIG was estimated elsewhere — this crate never does it.
//! let cheap = Candidate::new(
//!     oid(b"pendulum"),
//!     Estimate::new(400, 0, DeterminismLevel::L3),
//!     Cost::new(1, 0, 0, 0),
//! );
//! let decisive = Candidate::new(
//!     oid(b"perihelion"),
//!     Estimate::new(3_000, 0, DeterminismLevel::L3),
//!     Cost::new(2, 0, 0, 0),
//! );
//!
//! let problem = Discrimination::between(vec![newton, einstein]).unwrap();
//! let plan = discriminate(
//!     &problem,
//!     &[
//!         DiscriminatingCandidate::new(cheap, newton, einstein),
//!         DiscriminatingCandidate::new(decisive, newton, einstein),
//!     ],
//!     UtilityPolicy::EigPerCost,
//!     0,
//! )
//! .unwrap();
//!
//! // The planner ranked them; this crate reports which rivals the winner separates.
//! assert_eq!(plan.plan.verdict, StopVerdict::Recommend(oid(b"perihelion")));
//! assert_eq!(plan.separated_by(oid(b"perihelion")), Some((newton, einstein)));
//! ```

use std::collections::BTreeMap;

use sos_core::ObjectId;
use sos_planner::{Candidate, GreedyPlanner, Planner, UtilityPolicy};

use crate::error::{Result, TheoryError};

/// A discrimination problem: the rival theories to be told apart.
///
/// Rivals are recorded as [`ObjectId`]s into the graph, like every other
/// theory reference in this crate — a discrimination problem is a view over
/// provenance, not a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discrimination {
    rivals: Vec<ObjectId>,
}

impl Discrimination {
    /// A problem over the given rivals.
    ///
    /// # Errors
    /// [`TheoryError::TooFewRivals`] with fewer than two — there is nothing to
    /// discriminate between — or [`TheoryError::DuplicateRival`] if a theory
    /// is listed twice, since no experiment separates a theory from itself.
    pub fn between(rivals: Vec<ObjectId>) -> Result<Self> {
        if rivals.len() < 2
        {
            return Err(TheoryError::TooFewRivals(rivals.len()));
        }
        let mut seen = rivals.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != rivals.len()
        {
            return Err(TheoryError::DuplicateRival);
        }
        Ok(Self { rivals })
    }

    /// The rivals, in the order supplied.
    #[must_use]
    pub fn rivals(&self) -> &[ObjectId] {
        &self.rivals
    }

    /// Whether `id` is one of the rivals.
    #[must_use]
    pub fn is_rival(&self, id: ObjectId) -> bool {
        self.rivals.contains(&id)
    }
}

/// A candidate design together with the claim that it separates two specific
/// rivals.
///
/// The [`Candidate`] is the planner's own type, carrying an EIG estimate
/// produced by the numerics backend. The pair is this crate's contribution:
/// the planner cannot see which theories a design would tell apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscriminatingCandidate {
    /// The design, with its already-estimated EIG and cost.
    pub candidate: Candidate,
    /// One rival it would separate.
    pub left: ObjectId,
    /// The other.
    pub right: ObjectId,
}

impl DiscriminatingCandidate {
    /// Claim that `candidate` separates `left` from `right`.
    #[must_use]
    pub fn new(candidate: Candidate, left: ObjectId, right: ObjectId) -> Self {
        Self {
            candidate,
            left,
            right,
        }
    }
}

/// The planner's ranking, plus the bearing this crate attached to each design.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscriminationPlan {
    /// The planner's plan, exactly as it produced it — ranking, utilities and
    /// verdict unmodified.
    pub plan: sos_planner::Plan,
    bearing: BTreeMap<ObjectId, (ObjectId, ObjectId)>,
}

impl DiscriminationPlan {
    /// Which rivals the design `experiment` was submitted as separating.
    #[must_use]
    pub fn separated_by(&self, experiment: ObjectId) -> Option<(ObjectId, ObjectId)> {
        self.bearing.get(&experiment).copied()
    }

    /// The recommended experiment and the rivals it separates, when the
    /// planner recommended one.
    ///
    /// `None` when the verdict is
    /// [`InformationExhausted`](sos_planner::StopVerdict::InformationExhausted)
    /// — the rivals cannot currently be separated by any submitted design,
    /// which this crate reports rather than works around.
    #[must_use]
    pub fn recommendation(&self) -> Option<(ObjectId, (ObjectId, ObjectId))> {
        match self.plan.verdict
        {
            sos_planner::StopVerdict::Recommend(experiment) =>
            {
                self.separated_by(experiment).map(|pair| (experiment, pair))
            },
            sos_planner::StopVerdict::InformationExhausted => None,
        }
    }
}

/// Rank the designs that could separate `problem`'s rivals.
///
/// Checks that each candidate bears on two distinct rivals of this problem,
/// then delegates ranking wholesale to [`GreedyPlanner`] under `policy` and
/// `eig_floor_milli`. The returned plan is the planner's.
///
/// # Errors
/// [`TheoryError::NoDiscriminatingCandidates`] if none were supplied;
/// [`TheoryError::NotARival`] if a candidate names a theory outside the
/// problem; [`TheoryError::SelfSeparation`] if it names the same theory
/// twice; [`TheoryError::Planner`] if the planner rejects the candidates.
pub fn discriminate(
    problem: &Discrimination,
    candidates: &[DiscriminatingCandidate],
    policy: UtilityPolicy,
    eig_floor_milli: i64,
) -> Result<DiscriminationPlan> {
    if candidates.is_empty()
    {
        return Err(TheoryError::NoDiscriminatingCandidates);
    }

    let mut bearing = BTreeMap::new();
    let mut designs = Vec::with_capacity(candidates.len());
    for c in candidates
    {
        // Relevance — this crate's admission rule, and the question the
        // planner structurally cannot ask.
        if c.left == c.right
        {
            return Err(TheoryError::SelfSeparation(c.left));
        }
        for side in [c.left, c.right]
        {
            if !problem.is_rival(side)
            {
                return Err(TheoryError::NotARival(side));
            }
        }
        bearing.insert(c.candidate.experiment, (c.left, c.right));
        designs.push(c.candidate);
    }

    // Informativeness, cost and ranking — entirely the planner's, including
    // the information-exhaustion verdict.
    let plan = GreedyPlanner::new().recommend(&designs, policy, eig_floor_milli)?;
    Ok(DiscriminationPlan { plan, bearing })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::{DeterminismLevel, HashAlgo};
    use sos_planner::{Cost, Estimate, StopVerdict};

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::Sha256, b"test", tag)
    }

    fn candidate(tag: &[u8], bits: i64, cost: i64) -> Candidate {
        Candidate::new(
            oid(tag),
            Estimate::new(bits, 0, DeterminismLevel::L3),
            Cost::new(cost, 0, 0, 0),
        )
    }

    fn rivals() -> (ObjectId, ObjectId) {
        (oid(b"newton"), oid(b"einstein"))
    }

    #[test]
    fn a_problem_needs_at_least_two_distinct_rivals() {
        let (a, b) = rivals();
        assert!(Discrimination::between(vec![a, b]).is_ok());
        assert!(matches!(
            Discrimination::between(vec![a]),
            Err(TheoryError::TooFewRivals(1))
        ));
        assert!(matches!(
            Discrimination::between(vec![]),
            Err(TheoryError::TooFewRivals(0))
        ));
        // No experiment separates a theory from itself.
        assert!(matches!(
            Discrimination::between(vec![a, a]),
            Err(TheoryError::DuplicateRival)
        ));
    }

    #[test]
    fn the_planners_ranking_is_returned_unmodified() {
        // The higher-EIG design wins on EigPerCost here, and this crate does
        // not reorder — re-sorting "by how well it separates" would be a
        // second utility policy.
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        let weak = candidate(b"weak", 400, 1);
        let strong = candidate(b"strong", 3_000, 1);

        let plan = discriminate(
            &problem,
            &[
                DiscriminatingCandidate::new(weak, a, b),
                DiscriminatingCandidate::new(strong, a, b),
            ],
            UtilityPolicy::EigPerCost,
            0,
        )
        .unwrap();

        let direct = GreedyPlanner::new()
            .recommend(&[weak, strong], UtilityPolicy::EigPerCost, 0)
            .unwrap();
        assert_eq!(plan.plan, direct, "byte-for-byte the planner's own plan");
        assert_eq!(plan.plan.verdict, StopVerdict::Recommend(oid(b"strong")));
    }

    #[test]
    fn the_bearing_survives_alongside_the_ranking() {
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        let design = candidate(b"perihelion", 3_000, 2);
        let plan = discriminate(
            &problem,
            &[DiscriminatingCandidate::new(design, a, b)],
            UtilityPolicy::EigPerCost,
            0,
        )
        .unwrap();

        assert_eq!(plan.separated_by(oid(b"perihelion")), Some((a, b)));
        assert_eq!(plan.separated_by(oid(b"never-submitted")), None);
        assert_eq!(plan.recommendation(), Some((oid(b"perihelion"), (a, b))));
    }

    #[test]
    fn information_exhaustion_passes_through_rather_than_being_softened() {
        // The rivals cannot currently be separated. That is a finding, and
        // this crate reports it instead of recommending something anyway.
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        let negligible = candidate(b"negligible", 5, 1);

        let plan = discriminate(
            &problem,
            &[DiscriminatingCandidate::new(negligible, a, b)],
            UtilityPolicy::EigPerCost,
            1_000, // a floor the design cannot clear
        )
        .unwrap();

        assert_eq!(plan.plan.verdict, StopVerdict::InformationExhausted);
        assert_eq!(plan.recommendation(), None);
        // The design is still *reported*, with its bearing — exhaustion is
        // not amnesia about what was considered.
        assert_eq!(plan.separated_by(oid(b"negligible")), Some((a, b)));
    }

    #[test]
    fn a_design_bearing_on_a_non_rival_is_refused() {
        // Relevance is this crate's admission rule. A design about some other
        // theory is not a discriminating experiment for *these* rivals,
        // however informative it is.
        let (a, b) = rivals();
        let stranger = oid(b"unrelated-theory");
        let problem = Discrimination::between(vec![a, b]).unwrap();

        assert!(matches!(
            discriminate(
                &problem,
                &[DiscriminatingCandidate::new(
                    candidate(b"d", 3_000, 1),
                    a,
                    stranger
                )],
                UtilityPolicy::EigPerCost,
                0,
            ),
            Err(TheoryError::NotARival(id)) if id == stranger
        ));
    }

    #[test]
    fn a_design_claiming_to_separate_a_theory_from_itself_is_refused() {
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        assert!(matches!(
            discriminate(
                &problem,
                &[DiscriminatingCandidate::new(candidate(b"d", 3_000, 1), a, a)],
                UtilityPolicy::EigPerCost,
                0,
            ),
            Err(TheoryError::SelfSeparation(id)) if id == a
        ));
    }

    #[test]
    fn no_candidates_is_an_error_not_an_empty_recommendation() {
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        assert!(matches!(
            discriminate(&problem, &[], UtilityPolicy::EigPerCost, 0),
            Err(TheoryError::NoDiscriminatingCandidates)
        ));
    }

    #[test]
    fn more_than_two_rivals_are_supported_pairwise() {
        // Three-way disagreements are ordinary; each design still names the
        // specific pair it would settle.
        let (a, b) = rivals();
        let c = oid(b"brans-dicke");
        let problem = Discrimination::between(vec![a, b, c]).unwrap();

        let plan = discriminate(
            &problem,
            &[
                DiscriminatingCandidate::new(candidate(b"ab", 1_000, 1), a, b),
                DiscriminatingCandidate::new(candidate(b"bc", 3_000, 1), b, c),
            ],
            UtilityPolicy::EigPerCost,
            0,
        )
        .unwrap();

        let (winner, pair) = plan.recommendation().unwrap();
        assert_eq!(winner, oid(b"bc"));
        assert_eq!(pair, (b, c), "the report names which rivals it settles");
    }

    #[test]
    fn a_cost_budget_is_the_planners_call_and_is_honoured() {
        // Cost-aware utility belongs to the planner; this crate only passes
        // the policy through, and must not second-guess an exclusion.
        let (a, b) = rivals();
        let problem = Discrimination::between(vec![a, b]).unwrap();
        let cheap = candidate(b"cheap", 1_000, 1);
        let dear = candidate(b"dear", 9_000, 100);

        let plan = discriminate(
            &problem,
            &[
                DiscriminatingCandidate::new(cheap, a, b),
                DiscriminatingCandidate::new(dear, a, b),
            ],
            UtilityPolicy::EigBudgeted { budget: 10 },
            0,
        )
        .unwrap();

        assert_eq!(
            plan.plan.verdict,
            StopVerdict::Recommend(oid(b"cheap")),
            "the over-budget design is excluded by the planner's policy"
        );
    }
}
