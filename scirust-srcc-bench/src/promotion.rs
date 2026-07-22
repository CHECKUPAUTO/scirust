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
    /// An [`ExtendedPromotionGate`] defined no weighted composite metric.
    EmptyComposite,
    /// A composite weight was negative or non-finite.
    InvalidWeight {
        /// The offending metric name.
        metric: String,
        /// The rejected weight.
        weight: f64,
    },
    /// A composite scale was non-positive or non-finite.
    InvalidScale {
        /// The offending metric name.
        metric: String,
        /// The rejected scale.
        scale: f64,
    },
    /// The composite weights summed to zero (no metric carries any weight).
    ZeroTotalWeight,
    /// An [`ExtendedPromotionGate`] was handed no shadow windows.
    NoShadowWindows,
    /// A shadow window lacked a metric the composite requires.
    WindowMetricMissing {
        /// The window whose values were incomplete.
        window: String,
        /// The composite metric missing from that window.
        metric: String,
    },
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
            Self::EmptyComposite =>
            {
                write!(formatter, "the extended gate defines no composite metric")
            },
            Self::InvalidWeight { metric, weight } =>
            {
                write!(
                    formatter,
                    "composite weight {weight} for metric '{metric}' must be finite and non-negative"
                )
            },
            Self::InvalidScale { metric, scale } =>
            {
                write!(
                    formatter,
                    "composite scale {scale} for metric '{metric}' must be finite and positive"
                )
            },
            Self::ZeroTotalWeight =>
            {
                write!(formatter, "the composite weights sum to zero")
            },
            Self::NoShadowWindows =>
            {
                write!(formatter, "the extended gate received no shadow windows")
            },
            Self::WindowMetricMissing { window, metric } =>
            {
                write!(
                    formatter,
                    "shadow window '{window}' has no paired metric named '{metric}'"
                )
            },
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

// ---------------------------------------------------------------------------
// Extended promotion gate (axis 4): weighted composite, switching cost, and
// temporal shadow windows. Additive — the flat `PromotionGate` above is
// unchanged; this layers three operator-facing controls on the same seeded
// paired-bootstrap machinery.
// ---------------------------------------------------------------------------

/// A metric contributing to an [`ExtendedPromotionGate`]'s weighted composite.
///
/// Each metric's per-unit improvement is oriented (so positive is "candidate
/// better"), divided by `scale` to make heterogeneous metrics comparable, then
/// multiplied by `weight`. The composite per-unit improvement is the sum of
/// these contributions — a single dimensionless quantity the bootstrap can act
/// on directly.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMetric {
    /// Metric name (matched against each shadow window's values).
    pub metric: String,
    /// Which direction counts as an improvement.
    pub orientation: Orientation,
    /// Non-negative weight; the composite normalizes by the total weight so only
    /// the *relative* weights matter.
    pub weight: f64,
    /// Strictly-positive per-metric scale (e.g. the metric's typical magnitude)
    /// applied as `improvement / scale` before weighting.
    pub scale: f64,
}

/// One metric's paired per-unit values within a single shadow window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowMetricValues {
    /// Metric name (matched against the composite).
    pub metric: String,
    /// Incumbent per-unit values.
    pub incumbent: Vec<f64>,
    /// Candidate per-unit values, aligned with `incumbent`.
    pub candidate: Vec<f64>,
}

/// One temporal shadow window: every composite metric measured on that window's
/// units (a time slice, a deployment cohort, a machine group).
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowWindow {
    /// Human-readable window label (recorded in the report).
    pub label: String,
    /// The window's paired metric values; must cover every composite metric.
    pub values: Vec<WindowMetricValues>,
}

/// A preregistered **extended** promotion gate.
///
/// It composes three operator controls on top of the flat gate:
///
/// - **weighted composite** — the decision runs on a single per-unit score that
///   linearly combines several oriented, scaled metrics (`composite`), so a
///   trade-off (e.g. bulk error down, tail error up) is priced in advance rather
///   than argued after the fact;
/// - **switching cost** — the pooled composite improvement's bootstrap *lower
///   bound* must exceed `switching_cost`, a hysteresis deadband: promote only
///   when the candidate is better by more than the cost of switching models,
///   never on a marginal or lucky gain;
/// - **temporal shadow windows** — every window's *mean* composite improvement
///   must reach `min_window_improvement`, so a candidate that wins on pooled
///   data only by excelling in one period (and regressing in another) is held.
///
/// Promotion requires the pooled switching-cost test **and** every window's
/// consistency floor. Determinism: pooled and per-window verdicts are seeded
/// paired-bootstrap percentile intervals; identical inputs and seed give an
/// identical decision on every platform.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedPromotionGate {
    /// The weighted composite definition (at least one metric).
    pub composite: Vec<WeightedMetric>,
    /// Hysteresis deadband: the pooled composite improvement CI lower bound must
    /// exceed this. Use `0.0` for "any defensible improvement, cost-free".
    pub switching_cost: f64,
    /// Per-window floor: each window's mean composite improvement must be at
    /// least this. Use `0.0` for "no window may regress on the composite".
    pub min_window_improvement: f64,
    /// Bootstrap resample count.
    pub resamples: usize,
    /// Confidence level in `(0, 1)`.
    pub level: f64,
    /// Bootstrap seed (recorded in the report).
    pub seed: u64,
}

/// One shadow window's evaluated composite evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowFinding {
    /// The window this finding is about.
    pub label: String,
    /// Mean composite improvement over the window's units (positive = candidate
    /// better).
    pub mean_improvement: f64,
    /// The bootstrap interval for `mean_improvement`.
    pub confidence_interval: ConfidenceInterval,
    /// Whether the window cleared `min_window_improvement`.
    pub passed: bool,
}

/// The full, reproducible extended-promotion report.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedPromotionReport {
    /// Promote or hold.
    pub decision: Decision,
    /// Mean composite improvement pooled across all windows.
    pub pooled_mean_improvement: f64,
    /// The bootstrap interval for the pooled composite improvement.
    pub pooled_confidence_interval: ConfidenceInterval,
    /// Whether the pooled improvement cleared the switching cost.
    pub pooled_passed: bool,
    /// Per-window evidence, in gate-input order.
    pub windows: Vec<WindowFinding>,
    /// Human-readable reasons a hold was issued (empty when promoted).
    pub reasons: Vec<String>,
}

impl ExtendedPromotionGate {
    /// Evaluates the weighted composite over every shadow window and returns a
    /// promote/hold report.
    ///
    /// # Errors
    ///
    /// Returns [`PromotionError::EmptyComposite`] / [`PromotionError::NoShadowWindows`]
    /// for an empty gate or input; [`PromotionError::InvalidWeight`] /
    /// [`PromotionError::InvalidScale`] / [`PromotionError::ZeroTotalWeight`] for a
    /// malformed composite; [`PromotionError::WindowMetricMissing`] when a window
    /// omits a composite metric; and [`PromotionError::Comparison`] when the paired
    /// bootstrap rejects the values (misaligned lengths, non-finite, too few units).
    pub fn decide(
        &self,
        windows: &[ShadowWindow],
    ) -> Result<ExtendedPromotionReport, PromotionError> {
        if self.composite.is_empty()
        {
            return Err(PromotionError::EmptyComposite);
        }

        if windows.is_empty()
        {
            return Err(PromotionError::NoShadowWindows);
        }

        let mut total_weight = 0.0;

        for weighted in &self.composite
        {
            if !weighted.weight.is_finite() || weighted.weight < 0.0
            {
                return Err(PromotionError::InvalidWeight {
                    metric: weighted.metric.clone(),
                    weight: weighted.weight,
                });
            }

            if !weighted.scale.is_finite() || weighted.scale <= 0.0
            {
                return Err(PromotionError::InvalidScale {
                    metric: weighted.metric.clone(),
                    scale: weighted.scale,
                });
            }

            total_weight += weighted.weight;
        }

        if total_weight <= 0.0
        {
            return Err(PromotionError::ZeroTotalWeight);
        }

        let mut window_findings: Vec<WindowFinding> = Vec::with_capacity(windows.len());
        let mut pooled_composite: Vec<f64> = Vec::new();
        let mut reasons: Vec<String> = Vec::new();

        for window in windows
        {
            let composite = self.window_composite(window, total_weight)?;
            let report = paired_bootstrap(&composite, self.resamples, self.level, self.seed)?;
            let passed = report.mean_difference >= self.min_window_improvement;

            if !passed
            {
                reasons.push(format!(
                    "shadow window '{}' mean composite improvement {:.4} below floor {:.4}",
                    window.label, report.mean_difference, self.min_window_improvement
                ));
            }

            window_findings.push(WindowFinding {
                label: window.label.clone(),
                mean_improvement: report.mean_difference,
                confidence_interval: report.confidence_interval,
                passed,
            });

            pooled_composite.extend_from_slice(&composite);
        }

        let pooled_report =
            paired_bootstrap(&pooled_composite, self.resamples, self.level, self.seed)?;
        let pooled_passed = pooled_report.confidence_interval.lo > self.switching_cost;

        if !pooled_passed
        {
            reasons.push(format!(
                "pooled composite improvement CI lower bound {:.4} did not exceed switching cost {:.4}",
                pooled_report.confidence_interval.lo, self.switching_cost
            ));
        }

        let decision = if pooled_passed && window_findings.iter().all(|finding| finding.passed)
        {
            Decision::Promote
        }
        else
        {
            Decision::Hold
        };

        Ok(ExtendedPromotionReport {
            decision,
            pooled_mean_improvement: pooled_report.mean_difference,
            pooled_confidence_interval: pooled_report.confidence_interval,
            pooled_passed,
            windows: window_findings,
            reasons,
        })
    }

    /// Per-unit composite improvement for a single window: the weight-normalized
    /// sum of each metric's oriented, scaled per-unit improvement.
    fn window_composite(
        &self,
        window: &ShadowWindow,
        total_weight: f64,
    ) -> Result<Vec<f64>, PromotionError> {
        let mut composite: Option<Vec<f64>> = None;

        for weighted in &self.composite
        {
            let values = window
                .values
                .iter()
                .find(|candidate| candidate.metric == weighted.metric)
                .ok_or_else(|| PromotionError::WindowMetricMissing {
                    window: window.label.clone(),
                    metric: weighted.metric.clone(),
                })?;

            let paired = PairedMetric {
                metric: weighted.metric.clone(),
                orientation: weighted.orientation,
                incumbent: values.incumbent.clone(),
                candidate: values.candidate.clone(),
            };
            let improvement = improvement_vector(&paired)?;

            let contribution = weighted.weight / (total_weight * weighted.scale);

            match &mut composite
            {
                None =>
                {
                    composite = Some(
                        improvement
                            .iter()
                            .map(|value| value * contribution)
                            .collect(),
                    );
                },
                Some(accumulator) =>
                {
                    if accumulator.len() != improvement.len()
                    {
                        return Err(PromotionError::Comparison(
                            PairedComparisonError::LengthMismatch {
                                left: accumulator.len(),
                                right: improvement.len(),
                            },
                        ));
                    }

                    for (slot, value) in accumulator.iter_mut().zip(&improvement)
                    {
                        *slot += value * contribution;
                    }
                },
            }
        }

        // `composite` is `Some` because the composite is non-empty (checked in `decide`).
        Ok(composite.unwrap_or_default())
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
        assert!(!report.guardrails[0].passed);
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

    // --- extended gate (axis 4) ---

    fn weighted(name: &str, orientation: Orientation, weight: f64, scale: f64) -> WeightedMetric {
        WeightedMetric {
            metric: name.to_string(),
            orientation,
            weight,
            scale,
        }
    }

    fn window_values(name: &str, incumbent: Vec<f64>, candidate: Vec<f64>) -> WindowMetricValues {
        WindowMetricValues {
            metric: name.to_string(),
            incumbent,
            candidate,
        }
    }

    fn extended_gate(switching_cost: f64, min_window_improvement: f64) -> ExtendedPromotionGate {
        ExtendedPromotionGate {
            composite: vec![
                weighted("mae", Orientation::LowerIsBetter, 0.75, 1.0),
                weighted("rmse", Orientation::LowerIsBetter, 0.25, 1.0),
            ],
            switching_cost,
            min_window_improvement,
            resamples: 999,
            level: 0.95,
            seed: 0x0729_0042,
        }
    }

    /// Two windows; the candidate is uniformly better on both composite metrics.
    fn consistent_windows(delta: f64) -> Vec<ShadowWindow> {
        (0..2)
            .map(|w| {
                let base: Vec<f64> = (0..20).map(|i| 5.0 + (i % 4) as f64 + w as f64).collect();
                let better: Vec<f64> = base.iter().map(|v| v - delta).collect();
                ShadowWindow {
                    label: format!("window_{w}"),
                    values: vec![
                        window_values("mae", base.clone(), better.clone()),
                        window_values("rmse", base, better),
                    ],
                }
            })
            .collect()
    }

    #[test]
    fn extended_promotes_a_consistent_win_above_switching_cost() {
        // A clear, consistent 1.0-unit improvement clears a 0.2 switching cost.
        let report = extended_gate(0.2, 0.0)
            .decide(&consistent_windows(1.0))
            .unwrap();
        assert_eq!(report.decision, Decision::Promote);
        assert!(report.pooled_passed);
        assert!(report.windows.iter().all(|w| w.passed));
        assert!(report.reasons.is_empty());
    }

    #[test]
    fn extended_holds_when_improvement_is_below_the_switching_cost() {
        // A real but tiny 0.05 improvement does not clear a 0.5 switching cost.
        let report = extended_gate(0.5, 0.0)
            .decide(&consistent_windows(0.05))
            .unwrap();
        assert_eq!(report.decision, Decision::Hold);
        assert!(!report.pooled_passed);
        assert!(report.reasons.iter().any(|r| r.contains("switching cost")));
    }

    #[test]
    fn extended_holds_when_one_window_regresses_despite_a_good_pool() {
        // Window 0 improves strongly; window 1 regresses. Pooled may look fine,
        // but the per-window consistency floor must catch the regression.
        let strong_base: Vec<f64> = (0..20).map(|i| 10.0 + (i % 3) as f64).collect();
        let strong_better: Vec<f64> = strong_base.iter().map(|v| v - 2.0).collect();
        let weak_base: Vec<f64> = (0..20).map(|i| 4.0 + (i % 3) as f64).collect();
        let weak_worse: Vec<f64> = weak_base.iter().map(|v| v + 1.0).collect();

        let windows = vec![
            ShadowWindow {
                label: "recent_good".to_string(),
                values: vec![
                    window_values("mae", strong_base.clone(), strong_better.clone()),
                    window_values("rmse", strong_base, strong_better),
                ],
            },
            ShadowWindow {
                label: "recent_bad".to_string(),
                values: vec![
                    window_values("mae", weak_base.clone(), weak_worse.clone()),
                    window_values("rmse", weak_base, weak_worse),
                ],
            },
        ];

        let report = extended_gate(0.0, 0.0).decide(&windows).unwrap();
        assert_eq!(report.decision, Decision::Hold);
        assert!(!report.windows[1].passed);
        assert!(report.reasons.iter().any(|r| r.contains("recent_bad")));
    }

    #[test]
    fn extended_weights_and_scales_shape_the_composite() {
        // mae improves by +1, rmse regresses by −1 (in raw units). With mae
        // weighted 0.75 and rmse 0.25 (equal scale), the composite is positive.
        let base: Vec<f64> = (0..24).map(|i| 6.0 + (i % 3) as f64).collect();
        let mae_better: Vec<f64> = base.iter().map(|v| v - 1.0).collect();
        let rmse_worse: Vec<f64> = base.iter().map(|v| v + 1.0).collect();

        let windows = vec![ShadowWindow {
            label: "w".to_string(),
            values: vec![
                window_values("mae", base.clone(), mae_better),
                window_values("rmse", base, rmse_worse),
            ],
        }];

        let report = extended_gate(0.0, 0.0).decide(&windows).unwrap();
        // Weighted composite mean = 0.75*(+1) + 0.25*(−1) = +0.5.
        assert!((report.pooled_mean_improvement - 0.5).abs() < 1e-9);
        assert_eq!(report.decision, Decision::Promote);
    }

    #[test]
    fn extended_rejects_a_malformed_composite_or_input() {
        let empty = ExtendedPromotionGate {
            composite: Vec::new(),
            ..extended_gate(0.0, 0.0)
        };
        assert_eq!(
            empty.decide(&consistent_windows(1.0)).unwrap_err(),
            PromotionError::EmptyComposite
        );

        assert_eq!(
            extended_gate(0.0, 0.0).decide(&[]).unwrap_err(),
            PromotionError::NoShadowWindows
        );

        let bad_weight = ExtendedPromotionGate {
            composite: vec![weighted("mae", Orientation::LowerIsBetter, -1.0, 1.0)],
            ..extended_gate(0.0, 0.0)
        };
        assert_eq!(
            bad_weight.decide(&consistent_windows(1.0)).unwrap_err(),
            PromotionError::InvalidWeight {
                metric: "mae".to_string(),
                weight: -1.0
            }
        );

        let bad_scale = ExtendedPromotionGate {
            composite: vec![weighted("mae", Orientation::LowerIsBetter, 1.0, 0.0)],
            ..extended_gate(0.0, 0.0)
        };
        assert_eq!(
            bad_scale.decide(&consistent_windows(1.0)).unwrap_err(),
            PromotionError::InvalidScale {
                metric: "mae".to_string(),
                scale: 0.0
            }
        );

        let zero_total = ExtendedPromotionGate {
            composite: vec![weighted("mae", Orientation::LowerIsBetter, 0.0, 1.0)],
            ..extended_gate(0.0, 0.0)
        };
        assert_eq!(
            zero_total.decide(&consistent_windows(1.0)).unwrap_err(),
            PromotionError::ZeroTotalWeight
        );
    }

    #[test]
    fn extended_reports_a_missing_window_metric() {
        let windows = vec![ShadowWindow {
            label: "w".to_string(),
            values: vec![window_values("mae", vec![1.0; 20], vec![0.5; 20])],
        }];
        // The gate also needs "rmse", absent here.
        assert_eq!(
            extended_gate(0.0, 0.0).decide(&windows).unwrap_err(),
            PromotionError::WindowMetricMissing {
                window: "w".to_string(),
                metric: "rmse".to_string()
            }
        );
    }

    #[test]
    fn extended_decision_is_deterministic() {
        let windows = consistent_windows(0.8);
        assert_eq!(
            extended_gate(0.1, 0.0).decide(&windows),
            extended_gate(0.1, 0.0).decide(&windows)
        );
    }
}
