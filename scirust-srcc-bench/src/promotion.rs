//! Deterministic shadow-deployment promotion gates (phase 729).
//!
//! The benchmark harness produces *evidence*; a deployment needs a *decision*.
//! A [`PromotionGate`] turns a preregistered rule into a reproducible
//! promote/hold verdict on a **shadow comparison**: a candidate model is scored
//! alongside the incumbent on the same units, and the candidate is promoted only
//! when the evidence clears the bar the operator committed to *in advance*.
//!
//! The rule has two parts, both decided on the seeded paired bootstrap in
//! [`crate::paired`] (never on a point estimate):
//!
//! - a **primary criterion** — the candidate must improve the primary metric by
//!   at least `min_improvement`, and the *lower* bound of the improvement's
//!   confidence interval must exceed it (a statistically defensible gain, not a
//!   lucky mean);
//! - zero or more **guardrails** — metrics the candidate must not regress on:
//!   the *upper* bound of each regression's confidence interval must stay below
//!   `max_regression` (a defensible non-regression, symmetric to the primary
//!   test).
//!
//! Promotion requires the primary to pass **and** every guardrail to hold; any
//! failure holds the candidate and records the reason. Each metric carries its
//! own [`Orientation`], so the gate needs no per-criterion sign bookkeeping.
//!
//! Determinism: the whole decision is the paired bootstrap's seeded percentile
//! interval — identical inputs and seed give an identical verdict on every
//! platform. No RNG beyond that seed.

use core::fmt;

use scirust_bench_schema::ConfidenceInterval;

use crate::paired::{PairedComparisonError, paired_bootstrap, paired_differences};

/// Whether a lower or a higher value of a metric is better.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Smaller is better (errors, latencies, false-alarm rates).
    LowerIsBetter,
    /// Larger is better (AUROC, recall, throughput).
    HigherIsBetter,
}

/// Per-unit paired metric values: the incumbent and the candidate scored on the
/// same units (samples, machines, splits), in matching order.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedMetric {
    /// Metric name (matched against the gate's criteria).
    pub metric: String,
    /// Which direction counts as an improvement.
    pub orientation: Orientation,
    /// Incumbent per-unit values.
    pub incumbent: Vec<f64>,
    /// Candidate per-unit values, aligned with `incumbent`.
    pub candidate: Vec<f64>,
}

/// The preregistered primary promotion criterion.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimaryCriterion {
    /// The primary metric name.
    pub metric: String,
    /// Minimum candidate-over-incumbent improvement (in metric units) whose
    /// bootstrap lower bound must be exceeded to promote. Use `0.0` for "any
    /// statistically defensible improvement".
    pub min_improvement: f64,
}

/// A guardrail: a metric the candidate must not regress on beyond tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct Guardrail {
    /// The guardrail metric name.
    pub metric: String,
    /// Maximum tolerated regression (candidate worse than incumbent, in metric
    /// units); the regression's bootstrap upper bound must stay strictly below
    /// this. Use `0.0` for "no defensible regression at all".
    pub max_regression: f64,
}

/// A preregistered promotion gate.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionGate {
    /// The primary improvement criterion.
    pub primary: PrimaryCriterion,
    /// Guardrail non-regression criteria (may be empty).
    pub guardrails: Vec<Guardrail>,
    /// Bootstrap resample count.
    pub resamples: usize,
    /// Confidence level in `(0, 1)`.
    pub level: f64,
    /// Bootstrap seed (recorded in the report).
    pub seed: u64,
}

/// One criterion's evaluated evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CriterionFinding {
    /// The metric this finding is about.
    pub metric: String,
    /// Mean improvement (primary) or mean regression (guardrail), candidate vs
    /// incumbent, oriented so positive is "candidate better" for the primary and
    /// "candidate worse" for a guardrail.
    pub mean: f64,
    /// The bootstrap interval for `mean`.
    pub confidence_interval: ConfidenceInterval,
    /// Whether this criterion passed.
    pub passed: bool,
}

/// The gate's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The candidate cleared the primary criterion and every guardrail.
    Promote,
    /// The candidate was held back; see [`PromotionReport::reasons`].
    Hold,
}

/// The full, reproducible promotion report.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionReport {
    /// Promote or hold.
    pub decision: Decision,
    /// The primary criterion's evidence.
    pub primary: CriterionFinding,
    /// Each guardrail's evidence, in gate order.
    pub guardrails: Vec<CriterionFinding>,
    /// Human-readable reasons a hold was issued (empty when promoted).
    pub reasons: Vec<String>,
}

/// Typed promotion errors.
#[derive(Clone, Debug, PartialEq)]
pub enum PromotionError {
    /// A criterion referenced a metric absent from the shadow comparison.
    MetricMissing {
        /// The missing metric name.
        metric: String,
    },
    /// The underlying paired comparison failed (shape, finiteness, level, …).
    Comparison(PairedComparisonError),
}

impl fmt::Display for PromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::MetricMissing { metric } =>
            {
                write!(
                    formatter,
                    "no paired metric named '{metric}' in the comparison"
                )
            },
            Self::Comparison(error) => write!(formatter, "paired comparison failed: {error}"),
        }
    }
}

impl std::error::Error for PromotionError {}

impl From<PairedComparisonError> for PromotionError {
    fn from(error: PairedComparisonError) -> Self {
        Self::Comparison(error)
    }
}

impl PromotionGate {
    /// Evaluates the shadow comparison and returns a promote/hold report.
    ///
    /// # Errors
    ///
    /// [`PromotionError::MetricMissing`] when a criterion names an absent metric;
    /// [`PromotionError::Comparison`] when the paired bootstrap rejects the
    /// inputs (misaligned lengths, non-finite values, too few units, …).
    pub fn decide(&self, metrics: &[PairedMetric]) -> Result<PromotionReport, PromotionError> {
        let primary_metric = find_metric(metrics, &self.primary.metric)?;
        // Per-unit improvement, oriented so positive means "candidate better".
        let improvement = improvement_vector(primary_metric)?;
        let primary_report = paired_bootstrap(&improvement, self.resamples, self.level, self.seed)?;
        let primary_passed = primary_report.confidence_interval.lo > self.primary.min_improvement;

        let primary = CriterionFinding {
            metric: self.primary.metric.clone(),
            mean: primary_report.mean_difference,
            confidence_interval: primary_report.confidence_interval,
            passed: primary_passed,
        };

        let mut reasons: Vec<String> = Vec::new();

        if !primary_passed
        {
            reasons.push(format!(
                "primary '{}' improvement CI lower bound {:.4} did not exceed required {:.4}",
                self.primary.metric, primary.confidence_interval.lo, self.primary.min_improvement
            ));
        }

        let mut guardrails: Vec<CriterionFinding> = Vec::with_capacity(self.guardrails.len());

        for guardrail in &self.guardrails
        {
            let metric = find_metric(metrics, &guardrail.metric)?;
            // Per-unit regression = negated improvement (positive means worse).
            let regression: Vec<f64> = improvement_vector(metric)?
                .iter()
                .map(|value| -value)
                .collect();
            let report = paired_bootstrap(&regression, self.resamples, self.level, self.seed)?;
            let passed = report.confidence_interval.hi < guardrail.max_regression;

            if !passed
            {
                reasons.push(format!(
                    "guardrail '{}' regression CI upper bound {:.4} reached tolerance {:.4}",
                    guardrail.metric, report.confidence_interval.hi, guardrail.max_regression
                ));
            }

            guardrails.push(CriterionFinding {
                metric: guardrail.metric.clone(),
                mean: report.mean_difference,
                confidence_interval: report.confidence_interval,
                passed,
            });
        }

        let decision = if primary_passed && guardrails.iter().all(|finding| finding.passed)
        {
            Decision::Promote
        }
        else
        {
            Decision::Hold
        };

        Ok(PromotionReport {
            decision,
            primary,
            guardrails,
            reasons,
        })
    }
}

fn find_metric<'a>(
    metrics: &'a [PairedMetric],
    name: &str,
) -> Result<&'a PairedMetric, PromotionError> {
    metrics
        .iter()
        .find(|metric| metric.metric == name)
        .ok_or_else(|| PromotionError::MetricMissing {
            metric: name.to_string(),
        })
}

/// Per-unit candidate-over-incumbent improvement, oriented so positive is always
/// "candidate better" regardless of the metric's [`Orientation`].
fn improvement_vector(metric: &PairedMetric) -> Result<Vec<f64>, PairedComparisonError> {
    match metric.orientation
    {
        // Lower is better: improvement = incumbent − candidate.
        Orientation::LowerIsBetter => paired_differences(&metric.incumbent, &metric.candidate),
        // Higher is better: improvement = candidate − incumbent.
        Orientation::HigherIsBetter => paired_differences(&metric.candidate, &metric.incumbent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> PromotionGate {
        PromotionGate {
            primary: PrimaryCriterion {
                metric: "signal_rmse".to_string(),
                min_improvement: 0.0,
            },
            guardrails: vec![Guardrail {
                metric: "median_abs_error".to_string(),
                max_regression: 0.5,
            }],
            resamples: 999,
            level: 0.95,
            seed: 0x0729_0001,
        }
    }

    fn metric(
        name: &str,
        orientation: Orientation,
        incumbent: Vec<f64>,
        candidate: Vec<f64>,
    ) -> PairedMetric {
        PairedMetric {
            metric: name.to_string(),
            orientation,
            incumbent,
            candidate,
        }
    }

    #[test]
    fn promotes_a_clear_win_with_guardrail_held() {
        // Candidate lower (better) on the primary error everywhere, and no worse
        // on the guardrail.
        let incumbent: Vec<f64> = (0..30).map(|i| 10.0 + (i % 3) as f64).collect();
        let candidate: Vec<f64> = incumbent.iter().map(|v| v - 2.0).collect();
        let guard_inc: Vec<f64> = (0..30).map(|i| 1.0 + 0.01 * (i % 4) as f64).collect();
        let guard_can: Vec<f64> = guard_inc.iter().map(|v| v - 0.1).collect();

        let report = gate()
            .decide(&[
                metric(
                    "signal_rmse",
                    Orientation::LowerIsBetter,
                    incumbent,
                    candidate,
                ),
                metric(
                    "median_abs_error",
                    Orientation::LowerIsBetter,
                    guard_inc,
                    guard_can,
                ),
            ])
            .unwrap();

        assert_eq!(report.decision, Decision::Promote);
        assert!(report.primary.passed);
        assert!(report.reasons.is_empty());
    }

    #[test]
    fn holds_when_the_primary_improvement_straddles_zero() {
        // Candidate alternates better/worse → mean near zero, CI straddles.
        let incumbent: Vec<f64> = (0..30).map(|i| 5.0 + (i % 5) as f64).collect();
        let candidate: Vec<f64> = incumbent
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 2 == 0 { v - 0.05 } else { v + 0.05 })
            .collect();
        let guard: Vec<f64> = vec![1.0; 30];

        let report = gate()
            .decide(&[
                metric(
                    "signal_rmse",
                    Orientation::LowerIsBetter,
                    incumbent,
                    candidate,
                ),
                metric(
                    "median_abs_error",
                    Orientation::LowerIsBetter,
                    guard.clone(),
                    guard,
                ),
            ])
            .unwrap();

        assert_eq!(report.decision, Decision::Hold);
        assert!(!report.primary.passed);
        assert!(!report.reasons.is_empty());
    }

    #[test]
    fn holds_when_a_guardrail_regresses_despite_a_primary_win() {
        // Primary clearly improves, but the guardrail regresses beyond 0.5.
        let incumbent: Vec<f64> = (0..30).map(|i| 10.0 + (i % 3) as f64).collect();
        let candidate: Vec<f64> = incumbent.iter().map(|v| v - 3.0).collect();
        let guard_inc: Vec<f64> = vec![1.0; 30];
        let guard_can: Vec<f64> = vec![2.0; 30]; // +1.0 regression, over the 0.5 tolerance

        let report = gate()
            .decide(&[
                metric(
                    "signal_rmse",
                    Orientation::LowerIsBetter,
                    incumbent,
                    candidate,
                ),
                metric(
                    "median_abs_error",
                    Orientation::LowerIsBetter,
                    guard_inc,
                    guard_can,
                ),
            ])
            .unwrap();

        assert_eq!(report.decision, Decision::Hold);
        assert!(report.primary.passed);
        assert_eq!(report.guardrails[0].passed, false);
        assert!(report.reasons.iter().any(|r| r.contains("guardrail")));
    }

    #[test]
    fn higher_is_better_orientation_is_respected() {
        // AUROC-like metric: candidate higher is the improvement.
        let incumbent: Vec<f64> = (0..30).map(|i| 0.5 + 0.001 * (i % 3) as f64).collect();
        let candidate: Vec<f64> = incumbent.iter().map(|v| v + 0.1).collect();

        let simple = PromotionGate {
            primary: PrimaryCriterion {
                metric: "auroc".to_string(),
                min_improvement: 0.0,
            },
            guardrails: Vec::new(),
            resamples: 500,
            level: 0.95,
            seed: 1,
        };

        let report = simple
            .decide(&[metric(
                "auroc",
                Orientation::HigherIsBetter,
                incumbent,
                candidate,
            )])
            .unwrap();

        assert_eq!(report.decision, Decision::Promote);
        assert!(report.primary.mean > 0.0);
    }

    #[test]
    fn missing_metric_is_a_typed_error() {
        let err = gate()
            .decide(&[metric(
                "something_else",
                Orientation::LowerIsBetter,
                vec![1.0, 2.0],
                vec![1.0, 2.0],
            )])
            .unwrap_err();
        assert_eq!(
            err,
            PromotionError::MetricMissing {
                metric: "signal_rmse".to_string()
            }
        );
    }

    #[test]
    fn decision_is_deterministic() {
        let incumbent: Vec<f64> = (0..25).map(|i| 3.0 + (i % 4) as f64).collect();
        let candidate: Vec<f64> = incumbent.iter().map(|v| v - 1.0).collect();
        let guard: Vec<f64> = vec![1.0; 25];
        let comparison = [
            metric(
                "signal_rmse",
                Orientation::LowerIsBetter,
                incumbent,
                candidate,
            ),
            metric(
                "median_abs_error",
                Orientation::LowerIsBetter,
                guard.clone(),
                guard,
            ),
        ];

        assert_eq!(gate().decide(&comparison), gate().decide(&comparison));
    }
}
