//! Unified certified decision pipeline (phase 4E.7).
//!
//! The program built the measure-and-decide pieces separately: an uncertainty-
//! aware [estimator tournament that may abstain](crate::tournament) and
//! [conformal coverage that survives sub-populations and drift](crate::conditional_conformal).
//! This chains them into one flow that turns leakage-free evidence into a single
//! **certified decision** — and, crucially, an explicit statement of *what the
//! certificate does and does not promise*.
//!
//! Like every decision layer in this crate it is **metric-level**: the caller
//! fits each candidate under a leakage-free three-way split (train / calibrate /
//! test) and hands the pipeline each candidate's **calibration** and **test**
//! signed residuals `yᵢ − ŷᵢ`. The pipeline never fits a model, so its logic is
//! exhaustively testable on constructed residuals; the
//! `industrial-certified-pipeline` binary shows the end-to-end wiring.
//!
//! Steps:
//! 1. **Select** — run the tournament on the candidates' calibration absolute
//!    errors. A [`Select`](crate::TournamentDecision::Select) deploys the winner;
//!    a [`HoldIncumbent`](crate::TournamentDecision::HoldIncumbent) retains the
//!    incumbent; a [`Tie`](crate::TournamentDecision::Tie) or
//!    [`Inconclusive`](crate::TournamentDecision::Inconclusive) **abstains** (no
//!    unique deployment) but still reports a *provisional* incumbent certificate;
//!    a [`RejectAll`](crate::TournamentDecision::RejectAll) deploys nothing and
//!    issues no coverage certificate.
//! 2. **Calibrate** — fit a conformal band on the selected estimator's
//!    calibration residuals: marginal [`SplitConformal`](crate::SplitConformal),
//!    or per-group [`MondrianConformal`](crate::MondrianConformal) when group
//!    labels are supplied.
//! 3. **Certify** — measure the band's *empirical coverage on the held-out test
//!    residuals* and assemble the certificate with an explicit list of guarantees
//!    and caveats.
//!
//! Honesty is the whole point: selection is heuristic (best-observed, never
//! proven-optimal), coverage is marginal-or-conditional-on-group (never
//! conditional-on-`x`) and assumes calibration/test exchangeability (no drift
//! guarantee), and an abstention is reported as an abstention, not silently
//! resolved. All deterministic.

use core::fmt;

use crate::conditional_conformal::{ConditionalConformalError, GroupBand, MondrianConformal};
use crate::conformal::{ConformalError, SplitConformal};
use crate::tournament::{
    EstimatorTournament, TournamentDecision, TournamentEntry, TournamentError, TournamentReport,
};

/// One candidate's leakage-free residual evidence, `yᵢ − ŷᵢ` on each fold.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatorEvidence {
    /// Unique estimator name.
    pub name: String,
    /// Signed residuals on the calibration fold.
    pub calibration_residuals: Vec<f64>,
    /// Signed residuals on the held-out test fold.
    pub test_residuals: Vec<f64>,
}

impl EstimatorEvidence {
    /// Convenience constructor.
    pub fn new(
        name: impl Into<String>,
        calibration_residuals: Vec<f64>,
        test_residuals: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            calibration_residuals,
            test_residuals,
        }
    }
}

/// How coverage is certified.
#[derive(Clone, Debug, PartialEq)]
pub enum CoverageMode {
    /// A single marginal band.
    Marginal,
    /// Per-group (Mondrian) bands. `calibration_groups` aligns with the selected
    /// estimator's calibration residuals, `test_groups` with its test residuals.
    GroupConditional {
        /// Group key per calibration residual.
        calibration_groups: Vec<u64>,
        /// Group key per test residual.
        test_groups: Vec<u64>,
    },
}

/// Configuration for [`run_certified_pipeline`].
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedPipelineConfig {
    /// The tournament that selects (or abstains) from calibration errors.
    pub tournament: EstimatorTournament,
    /// Target coverage level for the conformal certificate, in `(0, 1)`.
    pub coverage_level: f64,
    /// Marginal or group-conditional coverage.
    pub coverage_mode: CoverageMode,
}

/// One group's certified coverage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupCoverage {
    /// The group key.
    pub key: u64,
    /// The group's half-width.
    pub half_width: f64,
    /// Empirical coverage on this group's test residuals (`NaN` if the group has
    /// no test residuals).
    pub empirical_test_coverage: f64,
    /// Whether the group had its own finite-sample band (else it borrowed the
    /// pooled marginal band — only a marginal guarantee for that group).
    pub conditionally_valid: bool,
}

/// The kind of coverage certificate produced.
#[derive(Clone, Debug, PartialEq)]
pub enum CoverageKind {
    /// A single marginal band with its half-width.
    Marginal {
        /// The symmetric half-width.
        half_width: f64,
    },
    /// Per-group bands.
    GroupConditional {
        /// Per-group coverage, ascending by key.
        per_group: Vec<GroupCoverage>,
    },
}

/// A coverage certificate for the deployed (or provisionally-retained) estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageCertificate {
    /// The estimator this certifies.
    pub estimator: String,
    /// The nominal coverage level.
    pub level: f64,
    /// Overall empirical coverage on the test residuals.
    pub empirical_test_coverage: f64,
    /// Calibration residuals used.
    pub calibration_count: usize,
    /// Test residuals evaluated.
    pub test_count: usize,
    /// Marginal or per-group detail.
    pub kind: CoverageKind,
    /// `true` when selection abstained and this certifies the incumbent only as a
    /// provisional fallback (not a deployment decision).
    pub provisional: bool,
}

/// The certified pipeline's full report.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedPipelineReport {
    /// The tournament's selection evidence and verdict.
    pub selection: TournamentReport,
    /// The estimator to deploy (`Some` for a firm Select/Hold; `None` when the
    /// tournament abstained or rejected all).
    pub selected_estimator: Option<String>,
    /// The coverage certificate (`None` only on RejectAll).
    pub coverage: Option<CoverageCertificate>,
    /// What the certificate guarantees.
    pub guarantees: Vec<String>,
    /// What it explicitly does not guarantee.
    pub caveats: Vec<String>,
}

/// Typed certified-pipeline errors.
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineError {
    /// The coverage level was not in `(0, 1)`.
    InvalidLevel {
        /// The rejected level.
        level: f64,
    },
    /// A candidate's calibration/test residual count did not match the group
    /// labels supplied for it.
    GroupLengthMismatch {
        /// Which fold ("calibration" or "test").
        fold: &'static str,
        /// Residual count.
        residuals: usize,
        /// Group-label count.
        groups: usize,
    },
    /// The tournament stage failed.
    Tournament(TournamentError),
    /// The marginal conformal stage failed.
    Conformal(ConformalError),
    /// The group-conditional conformal stage failed.
    Conditional(ConditionalConformalError),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidLevel { level } =>
            {
                write!(formatter, "coverage level {level} must lie in (0, 1)")
            },
            Self::GroupLengthMismatch {
                fold,
                residuals,
                groups,
            } => write!(
                formatter,
                "{fold} residual count {residuals} does not match its group-label count {groups}"
            ),
            Self::Tournament(error) => write!(formatter, "selection stage failed: {error}"),
            Self::Conformal(error) => write!(formatter, "coverage stage failed: {error}"),
            Self::Conditional(error) =>
            {
                write!(
                    formatter,
                    "group-conditional coverage stage failed: {error}"
                )
            },
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<TournamentError> for PipelineError {
    fn from(error: TournamentError) -> Self {
        Self::Tournament(error)
    }
}
impl From<ConformalError> for PipelineError {
    fn from(error: ConformalError) -> Self {
        Self::Conformal(error)
    }
}
impl From<ConditionalConformalError> for PipelineError {
    fn from(error: ConditionalConformalError) -> Self {
        Self::Conditional(error)
    }
}

/// Runs the unified certified pipeline: select an estimator (or abstain), then
/// certify its coverage.
///
/// # Errors
///
/// [`PipelineError`] on an out-of-range level, group labels misaligned with the
/// selected estimator's residuals, or a failure in the tournament or conformal
/// stage.
pub fn run_certified_pipeline(
    incumbent: &EstimatorEvidence,
    challengers: &[EstimatorEvidence],
    config: &CertifiedPipelineConfig,
) -> Result<CertifiedPipelineReport, PipelineError> {
    if !(config.coverage_level > 0.0 && config.coverage_level < 1.0)
    {
        return Err(PipelineError::InvalidLevel {
            level: config.coverage_level,
        });
    }

    // Stage 1 — selection from calibration absolute errors.
    let incumbent_entry = TournamentEntry::new(
        incumbent.name.clone(),
        absolute(&incumbent.calibration_residuals),
    );
    let candidate_entries: Vec<TournamentEntry> = challengers
        .iter()
        .map(|c| TournamentEntry::new(c.name.clone(), absolute(&c.calibration_residuals)))
        .collect();
    let selection = config
        .tournament
        .evaluate(&incumbent_entry, &candidate_entries)?;

    // Map the verdict to a deployment (or an abstention) plus which estimator, if
    // any, the coverage certificate should describe.
    let (selected_estimator, certify_name, provisional) = match &selection.decision
    {
        TournamentDecision::Select { winner } =>
        {
            (Some(winner.clone()), Some(winner.clone()), false)
        },
        TournamentDecision::HoldIncumbent => (
            Some(incumbent.name.clone()),
            Some(incumbent.name.clone()),
            false,
        ),
        TournamentDecision::Tie { contenders } =>
        {
            // Abstain on the choice, but describe a representative tied contender
            // (they are statistically equivalent) — not the beaten incumbent.
            let representative = contenders
                .first()
                .cloned()
                .unwrap_or_else(|| incumbent.name.clone());
            (None, Some(representative), true)
        },
        TournamentDecision::Inconclusive =>
        {
            // No challenger was established as better; the incumbent is the safe
            // fallback, certified provisionally.
            (None, Some(incumbent.name.clone()), true)
        },
        TournamentDecision::RejectAll => (None, None, false),
    };

    let coverage = match &certify_name
    {
        None => None,
        Some(name) =>
        {
            let evidence = find_evidence(name, incumbent, challengers)
                .expect("selected name is always the incumbent or a challenger");
            Some(certify_coverage(evidence, config, provisional)?)
        },
    };

    let (guarantees, caveats) =
        describe_certificate(&selection.decision, config, coverage.as_ref());

    Ok(CertifiedPipelineReport {
        selection,
        selected_estimator,
        coverage,
        guarantees,
        caveats,
    })
}

fn certify_coverage(
    evidence: &EstimatorEvidence,
    config: &CertifiedPipelineConfig,
    provisional: bool,
) -> Result<CoverageCertificate, PipelineError> {
    let level = config.coverage_level;
    match &config.coverage_mode
    {
        CoverageMode::Marginal =>
        {
            let band = SplitConformal::fit(&evidence.calibration_residuals, level)?;
            let covered = evidence
                .test_residuals
                .iter()
                .filter(|&&r| band.covers(0.0, r))
                .count();
            let test_count = evidence.test_residuals.len();
            Ok(CoverageCertificate {
                estimator: evidence.name.clone(),
                level,
                empirical_test_coverage: empirical(covered, test_count),
                calibration_count: band.calibration_count(),
                test_count,
                kind: CoverageKind::Marginal {
                    half_width: band.half_width(),
                },
                provisional,
            })
        },
        CoverageMode::GroupConditional {
            calibration_groups,
            test_groups,
        } =>
        {
            if calibration_groups.len() != evidence.calibration_residuals.len()
            {
                return Err(PipelineError::GroupLengthMismatch {
                    fold: "calibration",
                    residuals: evidence.calibration_residuals.len(),
                    groups: calibration_groups.len(),
                });
            }
            if test_groups.len() != evidence.test_residuals.len()
            {
                return Err(PipelineError::GroupLengthMismatch {
                    fold: "test",
                    residuals: evidence.test_residuals.len(),
                    groups: test_groups.len(),
                });
            }
            let mondrian =
                MondrianConformal::fit(calibration_groups, &evidence.calibration_residuals, level)?;

            let per_group: Vec<GroupCoverage> = mondrian
                .bands()
                .iter()
                .map(|band| group_coverage(band, &mondrian, test_groups, &evidence.test_residuals))
                .collect();

            let covered = test_groups
                .iter()
                .zip(&evidence.test_residuals)
                .filter(|(g, r)| mondrian.covers(**g, 0.0, **r))
                .count();
            let test_count = evidence.test_residuals.len();

            Ok(CoverageCertificate {
                estimator: evidence.name.clone(),
                level,
                empirical_test_coverage: empirical(covered, test_count),
                calibration_count: evidence.calibration_residuals.len(),
                test_count,
                kind: CoverageKind::GroupConditional { per_group },
                provisional,
            })
        },
    }
}

fn group_coverage(
    band: &GroupBand,
    mondrian: &MondrianConformal,
    test_groups: &[u64],
    test_residuals: &[f64],
) -> GroupCoverage {
    let mut covered = 0usize;
    let mut total = 0usize;
    for (&g, &r) in test_groups.iter().zip(test_residuals)
    {
        if g == band.key
        {
            total += 1;
            if mondrian.covers(g, 0.0, r)
            {
                covered += 1;
            }
        }
    }
    GroupCoverage {
        key: band.key,
        half_width: band.half_width,
        empirical_test_coverage: empirical(covered, total),
        conditionally_valid: band.conditionally_valid,
    }
}

fn describe_certificate(
    decision: &TournamentDecision,
    config: &CertifiedPipelineConfig,
    coverage: Option<&CoverageCertificate>,
) -> (Vec<String>, Vec<String>) {
    let mut guarantees = Vec::new();
    let mut caveats = Vec::new();

    match decision
    {
        TournamentDecision::Select { winner } =>
        {
            guarantees.push(format!(
                "'{winner}' defensibly beat the incumbent and every co-contender under a paired bootstrap"
            ));
        },
        TournamentDecision::HoldIncumbent =>
        {
            guarantees.push(
                "no challenger was a defensible improvement; the incumbent is retained".to_string(),
            );
        },
        TournamentDecision::Tie { contenders } =>
        {
            caveats.push(format!(
                "selection ABSTAINED: contenders {contenders:?} are statistically indistinguishable — a secondary criterion must decide"
            ));
        },
        TournamentDecision::Inconclusive =>
        {
            caveats.push(
                "selection ABSTAINED: evidence too weak to name a winner — gather more calibration units"
                    .to_string(),
            );
        },
        TournamentDecision::RejectAll =>
        {
            caveats.push(
                "no candidate met the quality floor: nothing is deployable and no coverage is certified"
                    .to_string(),
            );
        },
    }

    if let Some(cert) = coverage
    {
        match &cert.kind
        {
            CoverageKind::Marginal { .. } =>
            {
                guarantees.push(format!(
                    "finite-sample MARGINAL coverage >= {:.3} for '{}' under calibration/test exchangeability",
                    cert.level, cert.estimator
                ));
                caveats.push("coverage is marginal, NOT conditional on the features x".to_string());
            },
            CoverageKind::GroupConditional { .. } =>
            {
                guarantees.push(format!(
                    "finite-sample coverage >= {:.3} for '{}' CONDITIONAL ON GROUP (Mondrian)",
                    cert.level, cert.estimator
                ));
                caveats.push(
                    "coverage is conditional on the discrete group, NOT on the continuous features x; groups flagged conditionally_valid=false hold only marginally"
                        .to_string(),
                );
            },
        }
        if cert.provisional
        {
            caveats.push(format!(
                "this coverage certificate is PROVISIONAL (selection abstained); it describes '{}' only, not a committed deployment",
                cert.estimator
            ));
        }
        caveats.push(
            "exchangeability is assumed: no coverage guarantee under distribution drift (use adaptive conformal for streams)"
                .to_string(),
        );
    }

    // Always-true honesty caveat about the selection procedure.
    if !matches!(decision, TournamentDecision::RejectAll)
    {
        caveats.push(
            "estimator selection is heuristic and best-observed under the given panel, not proven globally optimal"
                .to_string(),
        );
    }
    let _ = config;
    (guarantees, caveats)
}

fn find_evidence<'a>(
    name: &str,
    incumbent: &'a EstimatorEvidence,
    challengers: &'a [EstimatorEvidence],
) -> Option<&'a EstimatorEvidence> {
    if incumbent.name == name
    {
        return Some(incumbent);
    }
    challengers.iter().find(|c| c.name == name)
}

fn absolute(residuals: &[f64]) -> Vec<f64> {
    residuals.iter().map(|r| r.abs()).collect()
}

fn empirical(covered: usize, total: usize) -> f64 {
    if total == 0
    {
        f64::NAN
    }
    else
    {
        covered as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::promotion::Orientation;

    fn tournament() -> EstimatorTournament {
        EstimatorTournament {
            orientation: Orientation::LowerIsBetter,
            min_improvement: 0.0,
            tie_margin: 0.0,
            quality_floor: None,
            resamples: 2000,
            level: 0.95,
            seed: 0x0000_C0DE,
        }
    }

    fn marginal_config() -> CertifiedPipelineConfig {
        CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: 0.9,
            coverage_mode: CoverageMode::Marginal,
        }
    }

    /// A deterministic spread of residuals of a given magnitude.
    fn residuals(magnitude: f64, n: usize, offset: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = (((i + offset) * 7) % 11) as f64 / 11.0 - 0.5; // [-0.5, 0.5)
                u * magnitude
            })
            .collect()
    }

    #[test]
    fn selects_the_winner_and_certifies_its_coverage() {
        // Challenger 'robust' has much smaller residuals than the incumbent.
        let incumbent = EstimatorEvidence::new("ols", residuals(8.0, 60, 0), residuals(8.0, 40, 1));
        let robust = EstimatorEvidence::new("robust", residuals(1.0, 60, 2), residuals(1.0, 40, 3));
        let report = run_certified_pipeline(&incumbent, &[robust], &marginal_config()).unwrap();

        assert_eq!(
            report.selection.decision,
            TournamentDecision::Select {
                winner: "robust".to_string()
            }
        );
        assert_eq!(report.selected_estimator.as_deref(), Some("robust"));
        let cert = report.coverage.unwrap();
        assert_eq!(cert.estimator, "robust");
        assert!(!cert.provisional);
        assert!(cert.empirical_test_coverage >= 0.9);
        assert!(!report.guarantees.is_empty());
    }

    #[test]
    fn holds_the_incumbent_when_nothing_beats() {
        let incumbent = EstimatorEvidence::new("ols", residuals(2.0, 60, 0), residuals(2.0, 40, 1));
        let similar = EstimatorEvidence::new("other", residuals(2.0, 60, 0), residuals(2.0, 40, 1));
        let report = run_certified_pipeline(&incumbent, &[similar], &marginal_config()).unwrap();
        assert_eq!(report.selection.decision, TournamentDecision::HoldIncumbent);
        assert_eq!(report.selected_estimator.as_deref(), Some("ols"));
        assert!(!report.coverage.unwrap().provisional);
    }

    #[test]
    fn abstains_on_a_tie_with_a_provisional_certificate() {
        // Two challengers both clearly beat the incumbent but tie each other.
        let incumbent = EstimatorEvidence::new("ols", residuals(9.0, 60, 0), residuals(9.0, 40, 1));
        let a = EstimatorEvidence::new(
            "a",
            (0..60)
                .map(|i| if i % 2 == 0 { 1.0 } else { 0.9 })
                .collect(),
            residuals(1.0, 40, 5),
        );
        let b = EstimatorEvidence::new(
            "b",
            (0..60)
                .map(|i| if i % 2 == 0 { 0.9 } else { 1.0 })
                .collect(),
            residuals(1.0, 40, 6),
        );
        let report = run_certified_pipeline(&incumbent, &[a, b], &marginal_config()).unwrap();
        assert!(matches!(
            report.selection.decision,
            TournamentDecision::Tie { .. }
        ));
        assert_eq!(report.selected_estimator, None);
        // A provisional certificate for a representative tied contender (not the
        // beaten incumbent) is still produced.
        let cert = report.coverage.unwrap();
        assert_eq!(cert.estimator, "a");
        assert!(cert.provisional);
        assert!(report.caveats.iter().any(|c| c.contains("ABSTAINED")));
    }

    #[test]
    fn rejects_all_below_the_quality_floor() {
        let incumbent = EstimatorEvidence::new("ols", residuals(8.0, 60, 0), residuals(8.0, 40, 1));
        let robust = EstimatorEvidence::new("robust", residuals(6.0, 60, 2), residuals(6.0, 40, 3));
        let mut config = marginal_config();
        // Require mean abs error <= 1.0; nobody meets it.
        config.tournament.quality_floor = Some(1.0);
        let report = run_certified_pipeline(&incumbent, &[robust], &config).unwrap();
        assert_eq!(report.selection.decision, TournamentDecision::RejectAll);
        assert_eq!(report.selected_estimator, None);
        assert!(report.coverage.is_none());
        assert!(
            report
                .caveats
                .iter()
                .any(|c| c.contains("nothing is deployable"))
        );
    }

    #[test]
    fn group_conditional_certificate_reports_per_group_coverage() {
        // One challenger wins; certify per-group coverage over two groups.
        let n_cal = 80;
        let n_test = 80;
        let incumbent =
            EstimatorEvidence::new("ols", residuals(9.0, n_cal, 0), residuals(9.0, n_test, 1));
        // Robust residuals: group 0 tight, group 1 wide, alternating by index.
        let robust_cal: Vec<f64> = (0..n_cal)
            .map(|i| {
                if i % 2 == 0
                {
                    residuals(1.0, 1, i)[0]
                }
                else
                {
                    residuals(5.0, 1, i)[0]
                }
            })
            .collect();
        let robust_test: Vec<f64> = (0..n_test)
            .map(|i| {
                if i % 2 == 0
                {
                    residuals(1.0, 1, i + 3)[0]
                }
                else
                {
                    residuals(5.0, 1, i + 3)[0]
                }
            })
            .collect();
        let robust = EstimatorEvidence::new("robust", robust_cal, robust_test);

        let groups: Vec<u64> = (0..n_cal).map(|i| (i % 2) as u64).collect();
        let test_groups: Vec<u64> = (0..n_test).map(|i| (i % 2) as u64).collect();

        let config = CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: 0.9,
            coverage_mode: CoverageMode::GroupConditional {
                calibration_groups: groups,
                test_groups,
            },
        };
        let report = run_certified_pipeline(&incumbent, &[robust], &config).unwrap();
        let cert = report.coverage.unwrap();
        match cert.kind
        {
            CoverageKind::GroupConditional { per_group } =>
            {
                assert_eq!(per_group.len(), 2);
                // Group 0 (tight) should have a smaller band than group 1 (wide).
                assert!(per_group[0].half_width < per_group[1].half_width);
                assert!(per_group.iter().all(|g| g.conditionally_valid));
            },
            other => panic!("expected group-conditional, got {other:?}"),
        }
    }

    #[test]
    fn invalid_level_and_group_mismatch_are_typed_errors() {
        let incumbent = EstimatorEvidence::new("ols", residuals(2.0, 20, 0), residuals(2.0, 10, 1));
        let mut bad = marginal_config();
        bad.coverage_level = 1.5;
        assert_eq!(
            run_certified_pipeline(&incumbent, &[], &bad),
            Err(PipelineError::InvalidLevel { level: 1.5 })
        );

        // Empty challenger set -> tournament NoCandidates surfaces as a typed error.
        assert!(matches!(
            run_certified_pipeline(&incumbent, &[], &marginal_config()),
            Err(PipelineError::Tournament(_))
        ));

        // Group labels misaligned with the winner's residuals.
        let robust = EstimatorEvidence::new("robust", residuals(0.5, 20, 2), residuals(0.5, 10, 3));
        let config = CertifiedPipelineConfig {
            tournament: tournament(),
            coverage_level: 0.9,
            coverage_mode: CoverageMode::GroupConditional {
                calibration_groups: vec![0; 19], // wrong length (20 residuals)
                test_groups: vec![0; 10],
            },
        };
        assert_eq!(
            run_certified_pipeline(&incumbent, &[robust], &config),
            Err(PipelineError::GroupLengthMismatch {
                fold: "calibration",
                residuals: 20,
                groups: 19
            })
        );
    }

    #[test]
    fn is_deterministic() {
        let incumbent = EstimatorEvidence::new("ols", residuals(7.0, 50, 0), residuals(7.0, 30, 1));
        let robust = EstimatorEvidence::new("robust", residuals(1.5, 50, 2), residuals(1.5, 30, 3));
        let challengers = [robust];
        let first = run_certified_pipeline(&incumbent, &challengers, &marginal_config()).unwrap();
        let second = run_certified_pipeline(&incumbent, &challengers, &marginal_config()).unwrap();
        assert_eq!(first, second);
    }
}
