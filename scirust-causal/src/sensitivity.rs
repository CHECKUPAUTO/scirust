//! Quantitative **sensitivity analysis for unmeasured confounding**, via the
//! omitted-variable-bias framework of Cinelli & Hazlett (2020), *Making Sense
//! of Sensitivity: Extending Omitted Variable Bias* (JRSS-B 82(1)).
//!
//! # Why this exists
//!
//! `crate::effect_estimation` identifies an effect **relative to a graph the
//! caller supplies**, and therefore assumes causal sufficiency rather than
//! establishing it. The crate's own adversarial test demonstrates what that
//! costs: a latent confounder omitted from both the data and the graph
//! produced an estimate of `1.497` against a truth of `0.7` — certified
//! `Identifiable`, with a standard error so small the bias was 121 standard
//! errors wide. Up to this point the only signal available was a *qualitative*
//! sensitivity note in prose.
//!
//! This module replaces that prose with arithmetic. It answers: **how strong
//! would an unmeasured confounder have to be to overturn this conclusion?**
//!
//! # What is computed, and what it assumes
//!
//! Everything here is a closed-form function of the fitted model's
//! `(β̂, se(β̂), df)` — no simulation, no RNG, no data re-fit (except
//! [`benchmark_covariate`], which residualizes). It assumes the *observed*
//! model is a correctly specified linear regression; it is a statement about
//! what an omitted linear term would do to that model, **not** a test for
//! whether a confounder exists. No data can supply that.
//!
//! - [`SensitivityAnalysis::partial_r2_treatment_outcome`] — `R²_{Y~D|X}`, the
//!   *extreme-scenario* benchmark: even a confounder explaining 100% of the
//!   treatment's residual variance must explain at least this share of the
//!   outcome's to drive the estimate to zero.
//! - [`SensitivityAnalysis::robustness_value`] — `RV_q`, the minimum share of
//!   residual variance a confounder must explain in **both** treatment and
//!   outcome to move the estimate by `100·q` percent. Reported on `[0, 1)`.
//! - [`bound_effect_under_confounder`] — for a *hypothesized* confounder
//!   strength, the exact worst-case bias, the adjusted estimate interval, and
//!   the adjusted standard error.
//! - [`benchmark_covariate`] — expresses a hypothetical confounder in units of
//!   an **observed** covariate ("as strong as `age`"), which is what makes a
//!   robustness value interpretable rather than an abstract number.
//!
//! # The lesson this module's own tests record
//!
//! A high robustness value is **not** by itself reassurance. On the crate's
//! sensitivity benchmark, a latent confounder produced an estimate of `1.5163`
//! against a truth of `0.7`, and `RV₁ = 0.9335` — "a confounder would need to
//! explain 93% of both residual variances to explain this away", which reads
//! as very robust.
//!
//! But the confounder that was actually there explained **90.1%** of the
//! treatment's residual variance and **43.7%** of the outcome's. Feeding
//! *those* measured strengths to [`bound_effect_under_confounder`] predicts a
//! bias of `0.8333` against an actual bias of `0.8163` — agreement to **2.1%**.
//!
//! So the machinery works; what fails is reading `RV` without asking whether a
//! confounder that strong is plausible in the domain. Here one nearly that
//! strong was not merely plausible, it was present. `RV` is a threshold to
//! compare against domain knowledge — which is what [`benchmark_covariate`]
//! exists to supply — not a verdict on its own. The crate's tests pin both
//! halves of this down.

use crate::certificate::IdentifiabilityStatus;
use crate::conditional_independence::{RegimeSelection, extract_column, select_rows};
use crate::dataset::CausalDataset;
use crate::effect_estimation::EffectEstimate;
use crate::error::CausalError;
use crate::partial_correlation::{pearson_correlation, residualize};

/// A hypothesized unmeasured confounder, described by how much residual
/// variance it explains in the treatment and in the outcome.
///
/// Both are *partial* R² values on `[0, 1)`: `treatment_share` is
/// `R²_{D~U|X}` (share of the treatment's variance left unexplained by the
/// adjustment set that `U` accounts for), and `outcome_share` is
/// `R²_{Y~U|D,X}` (likewise for the outcome, additionally controlling for the
/// treatment).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConfounderScenario {
    pub treatment_share: f64,
    pub outcome_share: f64,
}

impl ConfounderScenario {
    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] unless both shares are finite,
    /// non-negative, and strictly below 1. `treatment_share == 1` would mean
    /// the confounder explains the treatment entirely, leaving no variation
    /// to identify anything and making the bias formula divide by zero.
    pub fn new(treatment_share: f64, outcome_share: f64) -> Result<Self, CausalError> {
        for (name, value) in [
            ("treatment_share", treatment_share),
            ("outcome_share", outcome_share),
        ]
        {
            if !value.is_finite() || !(0.0..1.0).contains(&value)
            {
                return Err(CausalError::InvalidConfiguration { name, value });
            }
        }
        Ok(Self {
            treatment_share,
            outcome_share,
        })
    }
}

/// The closed-form sensitivity summary of one identified effect.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SensitivityAnalysis {
    pub treatment: usize,
    pub outcome: usize,
    pub estimate: f64,
    pub standard_error: f64,
    pub degrees_of_freedom: usize,
    pub t_statistic: f64,
    /// `R²_{Y~D|X}` — the extreme-scenario benchmark (see module docs).
    pub partial_r2_treatment_outcome: f64,
    /// The `q` the robustness value was computed for: the fraction of the
    /// estimate a confounder would have to explain away. `1.0` means "reduce
    /// it to zero".
    pub q: f64,
    /// `RV_q` on `[0, 1)`.
    pub robustness_value: f64,
}

/// Computes the sensitivity summary of an identified effect.
///
/// `q` is the fraction of the estimate a hypothetical confounder would have to
/// explain away — `1.0` for "reduce it to exactly zero", `0.5` for "halve it".
///
/// # Errors
///
/// [`CausalError::InvalidContract`] if `estimate`'s certificate is not
/// [`IdentifiabilityStatus::Identifiable`] — there is deliberately no way to
/// run a sensitivity analysis on a result that was never identified, since
/// "how robust is this estimate?" presupposes an estimate;
/// [`CausalError::InvalidConfiguration`] if `q` is not finite and in `(0, 1]`;
/// [`CausalError::InsufficientSamples`] if the fit left fewer than two
/// residual degrees of freedom (the adjusted-standard-error formula divides
/// by `df − 1`).
pub fn analyze_sensitivity(
    estimate: &EffectEstimate,
    q: f64,
) -> Result<SensitivityAnalysis, CausalError> {
    if estimate.certificate.status() != IdentifiabilityStatus::Identifiable
    {
        return Err(CausalError::InvalidContract {
            detail: "sensitivity analysis requires an Identifiable effect estimate",
        });
    }
    if !q.is_finite() || q <= 0.0 || q > 1.0
    {
        return Err(CausalError::InvalidConfiguration {
            name: "q",
            value: q,
        });
    }

    // `Identifiable` guarantees all three are present; the certificate layer
    // enforces it, and `effect_estimation` populates them together.
    let (Some(point), Some(standard_error), Some(adjustment)) = (
        estimate.estimate,
        estimate.standard_error,
        estimate.adjustment_set.as_ref(),
    )
    else
    {
        return Err(CausalError::InvalidContract {
            detail: "an Identifiable estimate must carry a point estimate, standard error, \
                     and adjustment set",
        });
    };

    let parameters = adjustment.len() + 2;
    if estimate.sample_count <= parameters + 1
    {
        return Err(CausalError::InsufficientSamples {
            required: parameters + 2,
            actual: estimate.sample_count,
        });
    }
    let degrees_of_freedom = estimate.sample_count - parameters;

    if standard_error <= 0.0
    {
        return Err(CausalError::InvalidConfiguration {
            name: "standard_error",
            value: standard_error,
        });
    }
    let t_statistic = point / standard_error;
    let df = degrees_of_freedom as f64;

    // Partial R² of the treatment with the outcome: t² / (t² + df).
    let t_squared = t_statistic * t_statistic;
    let partial_r2_treatment_outcome = t_squared / (t_squared + df);

    // Partial Cohen's f, scaled by the fraction to be explained away.
    let f_q = q * t_statistic.abs() / df.sqrt();
    let robustness_value = robustness_value_from_f(f_q);

    Ok(SensitivityAnalysis {
        treatment: estimate.treatment,
        outcome: estimate.outcome,
        estimate: point,
        standard_error,
        degrees_of_freedom,
        t_statistic,
        partial_r2_treatment_outcome,
        q,
        robustness_value,
    })
}

/// `RV = ½·(√(f⁴ + 4f²) − f²)`, the Cinelli–Hazlett robustness value.
///
/// Bounded in `[0, 1)` by construction: as `f → 0` it behaves like `f`, and as
/// `f → ∞` it approaches (but never reaches) 1 — a confounder can never need
/// to explain *more* than all of the residual variance.
fn robustness_value_from_f(f_q: f64) -> f64 {
    let f2 = f_q * f_q;
    let value = 0.5 * ((f2 * f2 + 4.0 * f2).sqrt() - f2);
    value.clamp(0.0, 1.0)
}

/// What a hypothesized confounder would do to an identified effect.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdjustedEffect {
    pub scenario: ConfounderScenario,
    /// The worst-case absolute bias such a confounder could induce.
    pub bias_bound: f64,
    /// `estimate − bias_bound` and `estimate + bias_bound`: the range the true
    /// effect could occupy if a confounder this strong were present.
    pub adjusted_estimate_lower: f64,
    pub adjusted_estimate_upper: f64,
    /// The standard error the estimate would have had, had the confounder been
    /// measured and adjusted for.
    pub adjusted_standard_error: f64,
    /// `true` iff the adjusted range excludes zero — i.e. the *sign* of the
    /// effect survives a confounder this strong. This is a statement about the
    /// point range only; it is not a hypothesis test.
    pub sign_survives: bool,
}

/// The exact worst-case bias a confounder of the given strength would induce,
/// plus the resulting adjusted estimate range and standard error.
///
/// `bias = se(β̂)·√df·√( R²_{Y~U|D,X}·R²_{D~U|X} / (1 − R²_{D~U|X}) )`
///
/// This is exact for a linear confounder of exactly that strength (the "bound"
/// is over the *direction*, which the data cannot pin down — hence a symmetric
/// range rather than a single adjusted point).
///
/// # Errors
///
/// [`CausalError::NonFiniteComputation`] if the arithmetic overflows to a
/// non-finite value.
pub fn bound_effect_under_confounder(
    analysis: &SensitivityAnalysis,
    scenario: &ConfounderScenario,
) -> Result<AdjustedEffect, CausalError> {
    let df = analysis.degrees_of_freedom as f64;
    let r2_d = scenario.treatment_share;
    let r2_y = scenario.outcome_share;

    let bias_factor = (r2_y * r2_d / (1.0 - r2_d)).sqrt();
    let bias_bound = analysis.standard_error * df.sqrt() * bias_factor;

    let adjusted_standard_error =
        analysis.standard_error * ((1.0 - r2_y) / (1.0 - r2_d)).sqrt() * (df / (df - 1.0)).sqrt();

    for (index, value) in [bias_bound, adjusted_standard_error]
        .into_iter()
        .enumerate()
    {
        if !value.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "bound_effect_under_confounder",
                index,
                value,
            });
        }
    }

    let lower = analysis.estimate - bias_bound;
    let upper = analysis.estimate + bias_bound;
    Ok(AdjustedEffect {
        scenario: *scenario,
        bias_bound,
        adjusted_estimate_lower: lower,
        adjusted_estimate_upper: upper,
        adjusted_standard_error,
        sign_survives: lower > 0.0 || upper < 0.0,
    })
}

/// Expresses an *observed* covariate as a [`ConfounderScenario`]: how much
/// residual variance it explains in the treatment and in the outcome, holding
/// the rest of the adjustment set fixed.
///
/// This is what makes a robustness value actionable. `RV = 0.93` means little
/// on its own; "`RV = 0.93`, and the strongest covariate we actually measured
/// reaches only 0.04" means a great deal — as does the opposite.
///
/// `covariate` must be a member of `adjustment`; the returned shares are
/// computed with the *other* adjustment variables partialled out, matching how
/// a confounder's strength is defined in [`ConfounderScenario`].
///
/// # Errors
///
/// [`CausalError::UnknownVariableIndex`] for an out-of-range index;
/// [`CausalError::InvalidContract`] if `covariate` is not in `adjustment`;
/// [`CausalError::ZeroVariance`] if the covariate has no residual variation
/// left once the others are partialled out (it is collinear with them, so
/// "as strong as this covariate" is not a well-defined scenario);
/// [`CausalError::RankDeficientConditioningSet`] / [`CausalError::SolverFailure`]
/// from the underlying residualization.
pub fn benchmark_covariate(
    dataset: &CausalDataset,
    treatment: usize,
    outcome: usize,
    adjustment: &[usize],
    covariate: usize,
    regime: &RegimeSelection,
    rank_tolerance: f64,
) -> Result<ConfounderScenario, CausalError> {
    let n_vars = dataset.variables.len();
    for &v in std::iter::once(&treatment)
        .chain(std::iter::once(&outcome))
        .chain(adjustment.iter())
        .chain(std::iter::once(&covariate))
    {
        if v >= n_vars
        {
            return Err(CausalError::UnknownVariableIndex { index: v });
        }
    }
    if !adjustment.contains(&covariate)
    {
        return Err(CausalError::InvalidContract {
            detail: "the benchmarked covariate must be a member of the adjustment set",
        });
    }

    let rows = select_rows(dataset, regime)?;
    let others: Vec<usize> = adjustment
        .iter()
        .copied()
        .filter(|&v| v != covariate)
        .collect();

    let treatment_column = extract_column(dataset, treatment, &rows);
    let outcome_column = extract_column(dataset, outcome, &rows);
    let covariate_column = extract_column(dataset, covariate, &rows);
    let other_columns: Vec<Vec<f64>> = others
        .iter()
        .map(|&z| extract_column(dataset, z, &rows))
        .collect();
    let other_refs: Vec<&[f64]> = other_columns.iter().map(Vec::as_slice).collect();

    // R²_{D~C | X_-c}: partial correlation of covariate with treatment.
    let treatment_residual = residualize(&treatment_column, &other_refs, rank_tolerance)?;
    let covariate_on_others = residualize(&covariate_column, &other_refs, rank_tolerance)?;
    let treatment_share = pearson_correlation(&treatment_residual, &covariate_on_others)
        .ok_or(CausalError::ZeroVariance {
            variable: covariate,
        })?
        .powi(2);

    // R²_{Y~C | D, X_-c}: same, additionally controlling for the treatment.
    let mut with_treatment: Vec<&[f64]> = vec![treatment_column.as_slice()];
    with_treatment.extend_from_slice(&other_refs);
    let outcome_residual = residualize(&outcome_column, &with_treatment, rank_tolerance)?;
    let covariate_on_treatment_and_others =
        residualize(&covariate_column, &with_treatment, rank_tolerance)?;
    let outcome_share = pearson_correlation(&outcome_residual, &covariate_on_treatment_and_others)
        .ok_or(CausalError::ZeroVariance {
            variable: covariate,
        })?
        .powi(2);

    ConfounderScenario::new(treatment_share, outcome_share)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RV` behaves as the algebra requires at both ends of its range.
    #[test]
    fn robustness_value_is_bounded_and_monotone() {
        assert!((robustness_value_from_f(0.0) - 0.0).abs() < 1e-12);
        // Small f: RV ≈ f.
        let small = robustness_value_from_f(0.01);
        assert!((small - 0.01).abs() < 1e-3, "got {small}");
        // Large f: RV → 1 from below, never reaching it.
        let large = robustness_value_from_f(1000.0);
        assert!(large < 1.0 && large > 0.999, "got {large}");
        // Monotone increasing in f.
        let mut previous = 0.0;
        for step in 1..200
        {
            let value = robustness_value_from_f(step as f64 * 0.05);
            assert!(value > previous, "RV must increase with f");
            assert!((0.0..1.0).contains(&value));
            previous = value;
        }
    }

    /// Hand-computed check of the closed form at f = 1: RV = ½(√5 − 1).
    #[test]
    fn robustness_value_matches_a_hand_computed_case() {
        // f = 1 ⇒ RV = ½(√(1 + 4) − 1) = ½(√5 − 1) ≈ 0.6180339887 (1/φ).
        let expected = 0.5 * (5.0_f64.sqrt() - 1.0);
        assert!((robustness_value_from_f(1.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn confounder_scenario_rejects_out_of_range_shares() {
        assert!(ConfounderScenario::new(0.5, 0.5).is_ok());
        assert!(ConfounderScenario::new(0.0, 0.0).is_ok());
        // A treatment share of exactly 1 would divide by zero in the bias.
        assert!(matches!(
            ConfounderScenario::new(1.0, 0.5),
            Err(CausalError::InvalidConfiguration {
                name: "treatment_share",
                ..
            })
        ));
        assert!(matches!(
            ConfounderScenario::new(0.5, 1.0),
            Err(CausalError::InvalidConfiguration {
                name: "outcome_share",
                ..
            })
        ));
        assert!(ConfounderScenario::new(-0.1, 0.5).is_err());
        assert!(ConfounderScenario::new(f64::NAN, 0.5).is_err());
    }

    /// A zero-strength confounder induces exactly zero bias.
    #[test]
    fn a_null_confounder_induces_no_bias() {
        let analysis = SensitivityAnalysis {
            treatment: 0,
            outcome: 1,
            estimate: 2.0,
            standard_error: 0.1,
            degrees_of_freedom: 100,
            t_statistic: 20.0,
            partial_r2_treatment_outcome: 0.8,
            q: 1.0,
            robustness_value: 0.5,
        };
        let scenario = ConfounderScenario::new(0.0, 0.0).unwrap();
        let adjusted = bound_effect_under_confounder(&analysis, &scenario).unwrap();
        assert!((adjusted.bias_bound - 0.0).abs() < 1e-12);
        assert!((adjusted.adjusted_estimate_lower - 2.0).abs() < 1e-12);
        assert!((adjusted.adjusted_estimate_upper - 2.0).abs() < 1e-12);
        assert!(adjusted.sign_survives);
    }

    /// The bias bound grows with confounder strength on both axes.
    #[test]
    fn bias_bound_increases_with_confounder_strength() {
        let analysis = SensitivityAnalysis {
            treatment: 0,
            outcome: 1,
            estimate: 1.0,
            standard_error: 0.05,
            degrees_of_freedom: 500,
            t_statistic: 20.0,
            partial_r2_treatment_outcome: 0.44,
            q: 1.0,
            robustness_value: 0.5,
        };
        let weak =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.1, 0.1).unwrap())
                .unwrap();
        let strong =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.5, 0.5).unwrap())
                .unwrap();
        assert!(strong.bias_bound > weak.bias_bound);
        // And a strong enough confounder eventually flips the sign verdict.
        let overwhelming =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.95, 0.95).unwrap())
                .unwrap();
        assert!(weak.sign_survives);
        assert!(!overwhelming.sign_survives);
    }

    /// A confounder at exactly the robustness value drives the estimate to
    /// (approximately) zero — this is the defining property of `RV₁`, checked
    /// against the independent bias formula rather than assumed.
    #[test]
    fn a_confounder_at_the_robustness_value_explains_the_effect_away() {
        // Construct a concrete analysis, then feed RV back through the bias
        // formula: the induced bias must equal the estimate.
        let standard_error = 0.05_f64;
        let degrees_of_freedom = 500usize;
        let estimate = 1.0_f64;
        let t_statistic = estimate / standard_error;
        let f = t_statistic.abs() / (degrees_of_freedom as f64).sqrt();
        let rv = robustness_value_from_f(f);

        let analysis = SensitivityAnalysis {
            treatment: 0,
            outcome: 1,
            estimate,
            standard_error,
            degrees_of_freedom,
            t_statistic,
            partial_r2_treatment_outcome: t_statistic * t_statistic
                / (t_statistic * t_statistic + degrees_of_freedom as f64),
            q: 1.0,
            robustness_value: rv,
        };
        let at_rv =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(rv, rv).unwrap())
                .unwrap();

        assert!(
            (at_rv.bias_bound - estimate).abs() < 1e-9,
            "a confounder at RV must induce a bias equal to the estimate: bias={} estimate={estimate}",
            at_rv.bias_bound
        );
        assert!(
            !at_rv.sign_survives,
            "at exactly RV the adjusted range must touch zero"
        );
    }

    /// The adjusted standard error responds in the documented directions.
    #[test]
    fn adjusted_standard_error_moves_in_the_expected_directions() {
        let analysis = SensitivityAnalysis {
            treatment: 0,
            outcome: 1,
            estimate: 1.0,
            standard_error: 0.05,
            degrees_of_freedom: 500,
            t_statistic: 20.0,
            partial_r2_treatment_outcome: 0.44,
            q: 1.0,
            robustness_value: 0.5,
        };
        // Explaining outcome variance shrinks the residual variance ⇒ smaller se.
        let outcome_only =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.0, 0.5).unwrap())
                .unwrap();
        assert!(outcome_only.adjusted_standard_error < analysis.standard_error);
        // Explaining treatment variance removes identifying variation ⇒ larger se.
        let treatment_only =
            bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.5, 0.0).unwrap())
                .unwrap();
        assert!(treatment_only.adjusted_standard_error > analysis.standard_error);
    }
}
