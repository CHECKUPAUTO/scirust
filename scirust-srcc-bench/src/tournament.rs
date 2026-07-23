//! Adaptive estimator tournament with abstention (phase 4E.5).
//!
//! Program 3 closed on a specific conclusion: robust benefit **cannot** be read
//! off a single marginal moment; the honest policy is *measure competing methods
//! under a leakage-free protocol, quantify uncertainty, and make an explicit
//! promote / hold / reject / inconclusive decision — never a one-number
//! shortcut*. [`PromotionGate`](crate::PromotionGate) does this for one incumbent
//! versus one challenger. This module generalizes it to a **field** of candidates
//! and, crucially, lets the tournament **abstain**.
//!
//! Like the promotion gate, the tournament is **metric-level**: it consumes
//! per-unit validation scores that the caller produced under a leakage-free
//! protocol (fit on training rows, score on held-out rows), and it decides. It
//! never fits an estimator itself — keeping the "protocol, not verdict" contract
//! and making the decision logic exhaustively testable on constructed scores. The
//! accompanying `industrial-estimator-tournament` binary shows the end-to-end
//! leakage-free wiring.
//!
//! # The five outcomes
//!
//! Every comparison is a seeded paired bootstrap (reusing
//! [`paired_bootstrap`](crate::paired::paired_bootstrap)) over the per-unit score
//! differences, so each verdict rests on an interval, not a point estimate.
//!
//! - [`TournamentDecision::Select`] — exactly one candidate defensibly beats the
//!   incumbent (improvement interval lower bound above `min_improvement`) and
//!   defensibly beats every other candidate that also beat the incumbent.
//! - [`TournamentDecision::HoldIncumbent`] — we can **rule out** that any
//!   candidate beats the incumbent by the margin (every improvement interval sits
//!   at or below `min_improvement`). Keep what you have.
//! - [`TournamentDecision::Tie`] — two or more candidates beat the incumbent but
//!   are statistically **indistinguishable from each other**; the tournament
//!   refuses to pick arbitrarily and names the contenders for a human or a
//!   secondary criterion to break.
//! - [`TournamentDecision::Inconclusive`] — at least one candidate *might* beat
//!   the incumbent (interval reaches above the margin) but none defensibly does.
//!   The evidence is too weak; collect more units.
//! - [`TournamentDecision::RejectAll`] — an absolute `quality_floor` is set and
//!   even the best-scoring candidate fails it. Nothing here is deployable,
//!   regardless of the relative ranking. This verdict takes precedence over the
//!   four relative ones (safety before ranking).
//!
//! Determinism: candidates are processed in a canonical name order and every
//! bootstrap seed is derived from the configured seed and canonical indices, so
//! the verdict is invariant to the order candidates are supplied and identical
//! across runs and platforms.

use core::fmt;

use scirust_bench_schema::ConfidenceInterval;

use crate::paired::{PairedComparisonError, paired_bootstrap};
use crate::promotion::Orientation;

/// An odd multiplier (golden-ratio scramble) for deriving per-comparison seeds.
const SEED_SCRAMBLE: u64 = 0x9E37_79B9_7F4A_7C15;

/// One competitor's per-unit validation scores, evaluated leakage-free on the
/// same held-out units, in the same order, as every other entry.
#[derive(Clone, Debug, PartialEq)]
pub struct TournamentEntry {
    /// Unique competitor name (the incumbent's included).
    pub name: String,
    /// Per-unit validation scores (e.g. per-sample absolute error).
    pub scores: Vec<f64>,
}

impl TournamentEntry {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, scores: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            scores,
        }
    }
}

/// Configuration of an adaptive estimator tournament.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatorTournament {
    /// Whether a lower or a higher score is better.
    pub orientation: Orientation,
    /// Practical-significance margin: a candidate beats the incumbent only if the
    /// bootstrap lower bound of its (oriented) improvement exceeds this, in metric
    /// units. `0.0` means "any statistically defensible improvement".
    pub min_improvement: f64,
    /// Two candidates that both beat the incumbent are declared tied unless the
    /// better one's head-to-head improvement lower bound exceeds this margin.
    pub tie_margin: f64,
    /// Optional absolute quality floor. If set, the best-scoring candidate's mean
    /// score must be at least as good (in the natural orientation) as this value;
    /// otherwise the verdict is [`TournamentDecision::RejectAll`].
    pub quality_floor: Option<f64>,
    /// Bootstrap resample count.
    pub resamples: usize,
    /// Confidence level in `(0, 1)`.
    pub level: f64,
    /// Base bootstrap seed (recorded; per-comparison seeds derive from it).
    pub seed: u64,
}

impl Default for EstimatorTournament {
    fn default() -> Self {
        Self {
            orientation: Orientation::LowerIsBetter,
            min_improvement: 0.0,
            tie_margin: 0.0,
            quality_floor: None,
            resamples: 2000,
            level: 0.95,
            seed: 0x5352_4343_5445,
        }
    }
}

/// One candidate's evidence versus the incumbent.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateFinding {
    /// Candidate name.
    pub name: String,
    /// Mean validation score in natural units (lower or higher better per the
    /// tournament orientation).
    pub mean_score: f64,
    /// Mean improvement over the incumbent, oriented so positive means the
    /// candidate is better.
    pub mean_improvement: f64,
    /// Bootstrap interval for `mean_improvement`.
    pub improvement_interval: ConfidenceInterval,
    /// Paired effect size (`None` when the per-unit improvements are constant).
    pub effect_size: Option<f64>,
    /// Whether the candidate defensibly beat the incumbent (interval lower bound
    /// strictly above `min_improvement`).
    pub beats_incumbent: bool,
}

/// The tournament's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TournamentDecision {
    /// Promote this single candidate.
    Select {
        /// The winning candidate's name.
        winner: String,
    },
    /// Keep the incumbent — no candidate is a defensible improvement.
    HoldIncumbent,
    /// Multiple candidates beat the incumbent but are indistinguishable from each
    /// other; a human or secondary criterion must break the tie.
    Tie {
        /// The tied contenders' names, ascending.
        contenders: Vec<String>,
    },
    /// The evidence is too weak to decide; collect more units.
    Inconclusive,
    /// No candidate meets the absolute quality floor.
    RejectAll,
}

/// The full, reproducible tournament report.
#[derive(Clone, Debug, PartialEq)]
pub struct TournamentReport {
    /// The verdict.
    pub decision: TournamentDecision,
    /// The incumbent's name.
    pub incumbent: String,
    /// Each candidate's evidence, best-first by mean improvement (ties by name).
    pub findings: Vec<CandidateFinding>,
    /// Human-readable justification of the verdict.
    pub reasons: Vec<String>,
}

/// Typed tournament errors.
#[derive(Clone, Debug, PartialEq)]
pub enum TournamentError {
    /// No candidates were supplied.
    NoCandidates,
    /// An entry's score vector length differs from the incumbent's.
    LengthMismatch {
        /// The offending entry.
        name: String,
        /// Incumbent unit count.
        expected: usize,
        /// The entry's unit count.
        found: usize,
    },
    /// Fewer than two units (a paired bootstrap needs at least two).
    TooFewUnits {
        /// The unit count supplied.
        found: usize,
    },
    /// A score is `NaN` or `±∞`.
    NonFiniteScore {
        /// The offending entry.
        name: String,
        /// The offending unit index.
        index: usize,
    },
    /// Two entries (incumbent or candidates) share a name.
    DuplicateName {
        /// The duplicated name.
        name: String,
    },
    /// A configuration value was out of range.
    InvalidConfig {
        /// What was wrong.
        detail: String,
    },
    /// An underlying paired comparison failed.
    Comparison(PairedComparisonError),
}

impl fmt::Display for TournamentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoCandidates => formatter.write_str("the tournament has no candidates"),
            Self::LengthMismatch {
                name,
                expected,
                found,
            } => write!(
                formatter,
                "entry '{name}' has {found} scores but the incumbent has {expected}"
            ),
            Self::TooFewUnits { found } => write!(
                formatter,
                "the tournament needs at least 2 units, found {found}"
            ),
            Self::NonFiniteScore { name, index } =>
            {
                write!(
                    formatter,
                    "entry '{name}' has a non-finite score at unit {index}"
                )
            },
            Self::DuplicateName { name } =>
            {
                write!(formatter, "two entries share the name '{name}'")
            },
            Self::InvalidConfig { detail } => write!(formatter, "invalid configuration: {detail}"),
            Self::Comparison(error) => write!(formatter, "paired comparison failed: {error}"),
        }
    }
}

impl std::error::Error for TournamentError {}

impl From<PairedComparisonError> for TournamentError {
    fn from(error: PairedComparisonError) -> Self {
        Self::Comparison(error)
    }
}

impl EstimatorTournament {
    /// Runs the tournament: score the incumbent against every candidate under a
    /// paired bootstrap, then apply the five-outcome decision rule.
    ///
    /// # Errors
    ///
    /// [`TournamentError`] on no candidates, mismatched score lengths, fewer than
    /// two units, a non-finite score, duplicate names, an out-of-range
    /// configuration, or an underlying bootstrap failure.
    pub fn evaluate(
        &self,
        incumbent: &TournamentEntry,
        candidates: &[TournamentEntry],
    ) -> Result<TournamentReport, TournamentError> {
        self.validate(incumbent, candidates)?;

        // Canonical processing order (by name) so the verdict is order-independent.
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| candidates[a].name.cmp(&candidates[b].name));

        let mut findings = Vec::with_capacity(candidates.len());
        for (canonical_index, &original) in order.iter().enumerate()
        {
            let candidate = &candidates[original];
            let differences =
                oriented_improvement(&candidate.scores, &incumbent.scores, self.orientation);
            let seed = self.seed ^ (canonical_index as u64).wrapping_mul(SEED_SCRAMBLE);
            let report = paired_bootstrap(&differences, self.resamples, self.level, seed)?;
            let mean_score = mean(&candidate.scores);
            findings.push((
                canonical_index,
                CandidateFinding {
                    name: candidate.name.clone(),
                    mean_score,
                    mean_improvement: report.mean_difference,
                    improvement_interval: report.confidence_interval,
                    effect_size: report.effect_size,
                    beats_incumbent: report.confidence_interval.lo > self.min_improvement,
                },
            ));
        }

        let (decision, reasons) = self.decide(incumbent, candidates, &order, &findings)?;

        // Report findings best-first by mean improvement (ties by name).
        let mut reported: Vec<CandidateFinding> =
            findings.into_iter().map(|(_, finding)| finding).collect();
        reported.sort_by(|a, b| {
            b.mean_improvement
                .total_cmp(&a.mean_improvement)
                .then_with(|| a.name.cmp(&b.name))
        });

        Ok(TournamentReport {
            decision,
            incumbent: incumbent.name.clone(),
            findings: reported,
            reasons,
        })
    }

    fn validate(
        &self,
        incumbent: &TournamentEntry,
        candidates: &[TournamentEntry],
    ) -> Result<(), TournamentError> {
        if !(self.min_improvement.is_finite() && self.min_improvement >= 0.0)
        {
            return Err(TournamentError::InvalidConfig {
                detail: "min_improvement must be finite and non-negative".to_string(),
            });
        }
        if !(self.tie_margin.is_finite() && self.tie_margin >= 0.0)
        {
            return Err(TournamentError::InvalidConfig {
                detail: "tie_margin must be finite and non-negative".to_string(),
            });
        }
        if let Some(floor) = self.quality_floor
            && !floor.is_finite()
        {
            return Err(TournamentError::InvalidConfig {
                detail: "quality_floor must be finite".to_string(),
            });
        }
        if !self.level.is_finite() || self.level <= 0.0 || self.level >= 1.0
        {
            return Err(TournamentError::InvalidConfig {
                detail: "level must be in (0, 1)".to_string(),
            });
        }
        if self.resamples == 0
        {
            return Err(TournamentError::InvalidConfig {
                detail: "resamples must be positive".to_string(),
            });
        }
        if candidates.is_empty()
        {
            return Err(TournamentError::NoCandidates);
        }

        let units = incumbent.scores.len();
        if units < 2
        {
            return Err(TournamentError::TooFewUnits { found: units });
        }

        check_finite(incumbent)?;
        for candidate in candidates
        {
            if candidate.scores.len() != units
            {
                return Err(TournamentError::LengthMismatch {
                    name: candidate.name.clone(),
                    expected: units,
                    found: candidate.scores.len(),
                });
            }
            check_finite(candidate)?;
        }

        // Names (incumbent + candidates) must be distinct.
        let mut names: Vec<&str> = Vec::with_capacity(candidates.len() + 1);
        names.push(incumbent.name.as_str());
        for candidate in candidates
        {
            names.push(candidate.name.as_str());
        }
        names.sort_unstable();
        for pair in names.windows(2)
        {
            if pair[0] == pair[1]
            {
                return Err(TournamentError::DuplicateName {
                    name: pair[0].to_string(),
                });
            }
        }

        Ok(())
    }

    /// The five-outcome decision rule. `findings` carries each candidate's
    /// canonical index (for seed derivation in head-to-head comparisons).
    fn decide(
        &self,
        _incumbent: &TournamentEntry,
        candidates: &[TournamentEntry],
        order: &[usize],
        findings: &[(usize, CandidateFinding)],
    ) -> Result<(TournamentDecision, Vec<String>), TournamentError> {
        // Safety before ranking: the absolute quality floor.
        if let Some(floor) = self.quality_floor
        {
            let best_score = findings
                .iter()
                .map(|(_, finding)| finding.mean_score)
                .reduce(|a, b| self.better_score(a, b))
                .expect("candidates are non-empty");
            if !self.score_is_acceptable(best_score, floor)
            {
                return Ok((
                    TournamentDecision::RejectAll,
                    vec![format!(
                        "best candidate mean score {best_score:.6} fails the quality floor {floor:.6}"
                    )],
                ));
            }
        }

        let beaters: Vec<&(usize, CandidateFinding)> = findings
            .iter()
            .filter(|(_, finding)| finding.beats_incumbent)
            .collect();

        if beaters.is_empty()
        {
            // Could any candidate still plausibly beat the incumbent?
            let plausible = findings
                .iter()
                .any(|(_, finding)| finding.improvement_interval.hi > self.min_improvement);
            if plausible
            {
                return Ok((
                    TournamentDecision::Inconclusive,
                    vec![
                        "at least one candidate's improvement interval reaches above the margin, \
                         but none clears it — insufficient evidence"
                            .to_string(),
                    ],
                ));
            }
            return Ok((
                TournamentDecision::HoldIncumbent,
                vec![
                    "every candidate's improvement interval sits at or below the margin — no \
                     defensible improvement"
                        .to_string(),
                ],
            ));
        }

        // Pick the best beater by mean improvement (ties by name), then test
        // whether it defensibly beats every other beater head to head.
        let best = beaters
            .iter()
            .copied()
            .reduce(
                |a, b| match a.1.mean_improvement.total_cmp(&b.1.mean_improvement)
                {
                    core::cmp::Ordering::Greater => a,
                    core::cmp::Ordering::Less => b,
                    core::cmp::Ordering::Equal =>
                    {
                        if a.1.name <= b.1.name
                        {
                            a
                        }
                        else
                        {
                            b
                        }
                    },
                },
            )
            .expect("beaters is non-empty");

        let best_original = order[best.0];
        let mut tied: Vec<String> = Vec::new();
        for entry in &beaters
        {
            let canonical_index = entry.0;
            let finding = &entry.1;
            if canonical_index == best.0
            {
                continue;
            }
            let other_original = order[canonical_index];
            let differences = oriented_improvement(
                &candidates[best_original].scores,
                &candidates[other_original].scores,
                self.orientation,
            );
            // Symmetric, deterministic head-to-head seed from both canonical indices.
            let seed = self.seed
                ^ (best.0 as u64).wrapping_mul(SEED_SCRAMBLE)
                ^ (canonical_index as u64)
                    .wrapping_mul(SEED_SCRAMBLE)
                    .rotate_left(32);
            let head_to_head = paired_bootstrap(&differences, self.resamples, self.level, seed)?;
            if head_to_head.confidence_interval.lo <= self.tie_margin
            {
                tied.push(finding.name.clone());
            }
        }

        if tied.is_empty()
        {
            Ok((
                TournamentDecision::Select {
                    winner: best.1.name.clone(),
                },
                vec![format!(
                    "'{}' defensibly beats the incumbent and every other contender",
                    best.1.name
                )],
            ))
        }
        else
        {
            tied.push(best.1.name.clone());
            tied.sort();
            Ok((
                TournamentDecision::Tie {
                    contenders: tied.clone(),
                },
                vec![format!(
                    "candidates {tied:?} beat the incumbent but are statistically indistinguishable"
                )],
            ))
        }
    }

    /// Returns the better of two mean scores under the orientation.
    fn better_score(&self, a: f64, b: f64) -> f64 {
        match self.orientation
        {
            Orientation::LowerIsBetter => a.min(b),
            Orientation::HigherIsBetter => a.max(b),
        }
    }

    /// Whether a mean score is at least as good as the floor, per orientation.
    fn score_is_acceptable(&self, score: f64, floor: f64) -> bool {
        match self.orientation
        {
            Orientation::LowerIsBetter => score <= floor,
            Orientation::HigherIsBetter => score >= floor,
        }
    }
}

fn check_finite(entry: &TournamentEntry) -> Result<(), TournamentError> {
    for (index, score) in entry.scores.iter().enumerate()
    {
        if !score.is_finite()
        {
            return Err(TournamentError::NonFiniteScore {
                name: entry.name.clone(),
                index,
            });
        }
    }
    Ok(())
}

/// Per-unit improvement of `a` over `b`, oriented so positive means `a` is
/// better.
fn oriented_improvement(a: &[f64], b: &[f64], orientation: Orientation) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(&ai, &bi)| match orientation
        {
            Orientation::LowerIsBetter => bi - ai,
            Orientation::HigherIsBetter => ai - bi,
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_is_better() -> EstimatorTournament {
        EstimatorTournament {
            orientation: Orientation::LowerIsBetter,
            min_improvement: 0.0,
            tie_margin: 0.0,
            quality_floor: None,
            resamples: 2000,
            level: 0.95,
            seed: 0x0000_0BEE,
        }
    }

    fn constant(value: f64, units: usize) -> Vec<f64> {
        vec![value; units]
    }

    /// Alternates `hi`/`lo` so the per-unit pattern has spread but a known mean.
    fn alternating(hi: f64, lo: f64, units: usize) -> Vec<f64> {
        (0..units)
            .map(|i| if i % 2 == 0 { hi } else { lo })
            .collect()
    }

    #[test]
    fn selects_a_single_clear_winner() {
        // Incumbent errors 1.0; A errors 0.5 (clearly best); B errors 0.9.
        // Every per-unit improvement is constant, so each interval is exact.
        let incumbent = TournamentEntry::new("ols", constant(1.0, 12));
        let candidates = vec![
            TournamentEntry::new("a", constant(0.5, 12)),
            TournamentEntry::new("b", constant(0.9, 12)),
        ];
        let report = lower_is_better().evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(
            report.decision,
            TournamentDecision::Select {
                winner: "a".to_string()
            }
        );
        // Best-first ordering: a (improvement 0.5) before b (0.1).
        assert_eq!(report.findings[0].name, "a");
        assert_eq!(report.findings[0].mean_improvement, 0.5);
        assert!(report.findings[0].beats_incumbent);
        assert!(report.findings[1].beats_incumbent);
    }

    #[test]
    fn holds_the_incumbent_when_nothing_improves() {
        // Both candidates are confidently worse (constant negative improvement).
        let incumbent = TournamentEntry::new("ols", constant(1.0, 12));
        let candidates = vec![
            TournamentEntry::new("a", constant(1.5, 12)),
            TournamentEntry::new("b", constant(1.2, 12)),
        ];
        let report = lower_is_better().evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(report.decision, TournamentDecision::HoldIncumbent);
        assert!(report.findings.iter().all(|f| !f.beats_incumbent));
    }

    #[test]
    fn abstains_as_inconclusive_when_the_signal_is_too_weak() {
        // Candidate improvement alternates +1 / -1: mean 0, interval straddles 0
        // — it *might* beat the incumbent but does not defensibly do so.
        let units = 24;
        let incumbent = TournamentEntry::new("ols", constant(1.0, units));
        let candidates = vec![TournamentEntry::new("a", alternating(0.0, 2.0, units))];
        let report = lower_is_better().evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(report.decision, TournamentDecision::Inconclusive);
        let finding = &report.findings[0];
        assert!(!finding.beats_incumbent);
        assert!(finding.improvement_interval.lo <= 0.0);
        assert!(finding.improvement_interval.hi > 0.0);
    }

    #[test]
    fn declares_a_tie_between_indistinguishable_winners() {
        // A and B both beat the incumbent (errors ~0.95 vs 2.0) but their
        // head-to-head difference alternates ±0.1 (mean 0) — indistinguishable.
        let units = 24;
        let incumbent = TournamentEntry::new("ols", constant(2.0, units));
        let candidates = vec![
            TournamentEntry::new("a", alternating(1.0, 0.9, units)),
            TournamentEntry::new("b", alternating(0.9, 1.0, units)),
        ];
        let report = lower_is_better().evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(
            report.decision,
            TournamentDecision::Tie {
                contenders: vec!["a".to_string(), "b".to_string()]
            }
        );
        assert!(report.findings.iter().all(|f| f.beats_incumbent));
    }

    #[test]
    fn rejects_all_when_the_quality_floor_is_unmet() {
        // A beats the incumbent relatively, but the absolute floor (MAE <= 0.5)
        // is unmet by every candidate — deploy nothing.
        let incumbent = TournamentEntry::new("ols", constant(2.0, 12));
        let candidates = vec![
            TournamentEntry::new("a", constant(1.0, 12)),
            TournamentEntry::new("b", constant(1.2, 12)),
        ];
        let mut tournament = lower_is_better();
        tournament.quality_floor = Some(0.5);
        let report = tournament.evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(report.decision, TournamentDecision::RejectAll);
        // The relative evidence is still reported (A does beat the incumbent).
        assert!(report.findings[0].beats_incumbent);
        assert!(report.reasons[0].contains("quality floor"));
    }

    #[test]
    fn a_met_floor_still_holds_when_nothing_beats() {
        // Floor is met (scores <= 2.5) but no candidate beats the incumbent.
        let incumbent = TournamentEntry::new("ols", constant(2.0, 12));
        let candidates = vec![
            TournamentEntry::new("a", constant(2.0, 12)), // equal
            TournamentEntry::new("b", constant(2.1, 12)), // worse
        ];
        let mut tournament = lower_is_better();
        tournament.quality_floor = Some(2.5);
        let report = tournament.evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(report.decision, TournamentDecision::HoldIncumbent);
    }

    #[test]
    fn higher_is_better_orientation_selects_the_top_scorer() {
        // AUROC-style: higher is better. A (0.90) beats incumbent (0.70) and B (0.75).
        let incumbent = TournamentEntry::new("ols", constant(0.70, 12));
        let candidates = vec![
            TournamentEntry::new("a", constant(0.90, 12)),
            TournamentEntry::new("b", constant(0.75, 12)),
        ];
        let tournament = EstimatorTournament {
            orientation: Orientation::HigherIsBetter,
            ..lower_is_better()
        };
        let report = tournament.evaluate(&incumbent, &candidates).unwrap();
        assert_eq!(
            report.decision,
            TournamentDecision::Select {
                winner: "a".to_string()
            }
        );
        assert_eq!(report.findings[0].name, "a");
        // Oriented improvement is positive for a better (higher) score.
        assert!((report.findings[0].mean_improvement - 0.20).abs() < 1e-12);
    }

    #[test]
    fn is_deterministic_and_order_independent() {
        let incumbent = TournamentEntry::new("ols", alternating(1.0, 1.2, 20));
        let a = TournamentEntry::new("a", alternating(0.4, 0.6, 20));
        let b = TournamentEntry::new("b", alternating(0.8, 0.7, 20));
        let c = TournamentEntry::new("c", alternating(0.9, 1.1, 20));

        let forward = lower_is_better()
            .evaluate(&incumbent, &[a.clone(), b.clone(), c.clone()])
            .unwrap();
        let again = lower_is_better()
            .evaluate(&incumbent, &[a.clone(), b.clone(), c.clone()])
            .unwrap();
        assert_eq!(forward, again);

        // Reversed candidate order must not change the verdict or the (sorted)
        // findings.
        let reversed = lower_is_better().evaluate(&incumbent, &[c, b, a]).unwrap();
        assert_eq!(forward.decision, reversed.decision);
        assert_eq!(forward.findings, reversed.findings);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let incumbent = TournamentEntry::new("ols", constant(1.0, 6));
        let ok = TournamentEntry::new("a", constant(0.5, 6));

        assert_eq!(
            lower_is_better().evaluate(&incumbent, &[]),
            Err(TournamentError::NoCandidates)
        );
        assert_eq!(
            lower_is_better().evaluate(&incumbent, &[TournamentEntry::new("a", constant(0.5, 5))]),
            Err(TournamentError::LengthMismatch {
                name: "a".to_string(),
                expected: 6,
                found: 5
            })
        );
        assert_eq!(
            lower_is_better().evaluate(
                &TournamentEntry::new("ols", constant(1.0, 1)),
                &[TournamentEntry::new("a", constant(0.5, 1))]
            ),
            Err(TournamentError::TooFewUnits { found: 1 })
        );
        assert_eq!(
            lower_is_better().evaluate(
                &incumbent,
                &[TournamentEntry::new(
                    "a",
                    vec![0.5, f64::NAN, 0.5, 0.5, 0.5, 0.5]
                )]
            ),
            Err(TournamentError::NonFiniteScore {
                name: "a".to_string(),
                index: 1
            })
        );
        assert_eq!(
            lower_is_better()
                .evaluate(&incumbent, &[TournamentEntry::new("ols", constant(0.5, 6))]),
            Err(TournamentError::DuplicateName {
                name: "ols".to_string()
            })
        );

        let candidate = std::slice::from_ref(&ok);

        let mut bad_level = lower_is_better();
        bad_level.level = 1.0;
        assert!(matches!(
            bad_level.evaluate(&incumbent, candidate),
            Err(TournamentError::InvalidConfig { .. })
        ));

        let mut bad_resamples = lower_is_better();
        bad_resamples.resamples = 0;
        assert!(matches!(
            bad_resamples.evaluate(&incumbent, candidate),
            Err(TournamentError::InvalidConfig { .. })
        ));

        let mut bad_margin = lower_is_better();
        bad_margin.min_improvement = -1.0;
        assert!(matches!(
            bad_margin.evaluate(&incumbent, candidate),
            Err(TournamentError::InvalidConfig { .. })
        ));
    }
}
