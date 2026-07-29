//! **Invariant Causal Prediction** (ICP) — Peters, Bühlmann & Meinshausen
//! (2016), *Causal inference using invariant prediction: identification and
//! confidence intervals* (JRSS-B 78(5)).
//!
//! # The idea
//!
//! If `S` really is the set of direct causes of `Y`, then the mechanism
//! `Y ← f(X_S) + ε` is a property of nature, not of the regime the data was
//! collected under. Intervening elsewhere in the system changes the
//! distribution of the `X`s, but **not** the conditional distribution of `Y`
//! given its own causes. So: regress `Y` on each candidate subset `S`, pooled
//! across environments, and ask whether the residuals look the same in every
//! environment. A subset whose residuals *shift* between environments cannot
//! be the causal set.
//!
//! Every subset that survives is "plausible". The variables in the
//! **intersection** of all plausible subsets are the ones no surviving
//! explanation can do without — and under the stated assumptions that
//! intersection is, with probability at least `1 − α`, a **subset of the true
//! direct causes**.
//!
//! # Why this complements — and does not replace — backdoor adjustment
//!
//! `crate::effect_estimation` needs a graph supplied by the caller and
//! *assumes* causal sufficiency; the crate's own adversarial test shows it
//! confidently returning a badly wrong number when that assumption fails, with
//! no way to notice. ICP needs **no graph at all**, and has a property that
//! backdoor adjustment structurally cannot have: when its own core assumption
//! fails, it can **say so**. If no subset survives the invariance test, that
//! is not a weak result — it is positive evidence that the model is
//! misspecified ([`InvariantPredictionOutcome::AssumptionsViolated`]).
//!
//! The trade is directness. ICP answers *which variables are direct causes*,
//! not *how large is the effect*, and it is conservative by construction: it
//! returns a subset, never a complete parent set, and frequently returns the
//! empty set when the environments are too few or too similar to discriminate.
//!
//! # What must hold, and what happens when it does not
//!
//! - **Invariance** ([`CausalAssumption::InvarianceAcrossEnvironments`]): the
//!   mechanism generating `Y` from its causes is identical in every
//!   environment.
//! - **No intervention acts directly on the target.** If an environment
//!   intervenes on `Y` itself, invariance fails *for the true causal set too*,
//!   and ICP will report `AssumptionsViolated` or an empty set — never the
//!   right answer. This module records that as a named assumption and cannot
//!   check it: the [`crate::Environment`] labels say which variables were
//!   intervened on, but nothing in the data says whether the labels are true.
//! - **Linearity**, since the mechanism is fitted by linear regression.
//!
//! # Cost
//!
//! The search is over subsets, so it is exponential in the candidate count.
//! [`InvarianceConfig::max_predictor_set_size`] bounds it; a bounded search
//! that finds an intersection reports one that is valid *among the subsets it
//! examined*, which is recorded as a warning rather than quietly presented as
//! a complete search.

use crate::assumptions::CausalAssumption;
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::conditional_independence::extract_column;
use crate::dataset::CausalDataset;
use crate::error::CausalError;
use crate::partial_correlation::residualize;
use crate::skeleton_discovery::combinations;
use crate::variable::VariableKind;
use scirust_stats::describe::variance;
use scirust_stats::dist::{Distribution, FisherF};
use scirust_stats::htest::{Tail, t_test_two_sample};
use std::collections::BTreeMap;

/// Configuration for [`invariant_causal_prediction`].
#[derive(Debug, Clone, PartialEq)]
pub struct InvarianceConfig {
    /// Level at which a candidate subset is *rejected*. The resulting causal
    /// set holds with confidence at least `1 − significance_level`.
    pub significance_level: f64,
    /// Largest candidate subset to examine. `None` searches all subsets —
    /// exponential in the candidate count. A bounded search that still finds
    /// an intersection records a warning saying so.
    pub max_predictor_set_size: Option<usize>,
    /// Relative singular-value tolerance for rank-deficiency detection in the
    /// pooled regression.
    pub rank_tolerance: f64,
}

impl Default for InvarianceConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            max_predictor_set_size: None,
            rank_tolerance: 1e-9,
        }
    }
}

impl InvarianceConfig {
    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] if `significance_level` is not
    /// finite and in `(0, 1)`.
    pub fn new(significance_level: f64) -> Result<Self, CausalError> {
        if !significance_level.is_finite() || significance_level <= 0.0 || significance_level >= 1.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "significance_level",
                value: significance_level,
            });
        }
        Ok(Self {
            significance_level,
            ..Self::default()
        })
    }

    #[must_use]
    pub fn with_max_predictor_set_size(mut self, size: usize) -> Self {
        self.max_predictor_set_size = Some(size);
        self
    }
}

/// The three-way outcome of an ICP run. None of these is a failure mode; the
/// last is a *positive finding* about the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InvariantPredictionOutcome {
    /// At least one subset survived, and their intersection is non-empty.
    /// Those variables are direct causes of the target with confidence at
    /// least `1 − significance_level`.
    CausalPredictorsIdentified,
    /// Subsets survived, but their intersection is empty: no individual
    /// variable is required by every surviving explanation. Usually too few
    /// or too similar environments to discriminate — an honest "cannot tell",
    /// not evidence that nothing is causal.
    NoPredictorConfirmed,
    /// **No** subset survived. Under the stated assumptions at least one
    /// subset — the true causal set — should have. That none did is evidence
    /// the assumptions themselves fail: a mechanism that is not invariant
    /// across these environments, an intervention acting directly on the
    /// target, or a non-linear mechanism.
    AssumptionsViolated,
}

/// One candidate subset that survived the invariance test.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AcceptedSet {
    pub predictors: Vec<usize>,
    /// The subset's Bonferroni-combined p-value; larger means more plausible.
    pub p_value: f64,
}

/// One ICP run's full, reproducible outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InvariantPredictionResult {
    pub target: usize,
    pub outcome: InvariantPredictionOutcome,
    /// The intersection of all accepted subsets — a **subset** of the true
    /// direct causes with confidence `1 − significance_level`, never a
    /// complete parent set. Empty unless the outcome is
    /// [`InvariantPredictionOutcome::CausalPredictorsIdentified`].
    pub causal_predictors: Vec<usize>,
    pub accepted_sets: Vec<AcceptedSet>,
    pub sets_tested: usize,
    pub environments: Vec<String>,
    pub significance_level: f64,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

/// The assumptions every ICP conclusion rests on.
fn invariance_assumptions() -> Vec<CausalAssumption> {
    vec![
        CausalAssumption::InvarianceAcrossEnvironments,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::AdequateSampleSize,
        CausalAssumption::Other(
            "no environment intervenes directly on the target variable".to_string(),
        ),
    ]
}

/// Two-sided F-test of `H₀: var(a) = var(b)`. Returns `None` when either side
/// is too small or degenerate for the ratio to be defined.
fn variance_ratio_p_value(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 2 || b.len() < 2
    {
        return None;
    }
    let (va, vb) = (variance(a), variance(b));
    // Finiteness first, so the subsequent comparisons cannot silently pass a
    // NaN through (`NaN <= 0.0` is false).
    if !va.is_finite() || !vb.is_finite() || va <= 0.0 || vb <= 0.0
    {
        return None;
    }
    let f = va / vb;
    let distribution = FisherF::new((a.len() - 1) as f64, (b.len() - 1) as f64);
    let p = 2.0 * distribution.cdf(f).min(distribution.sf(f));
    if p.is_finite()
    {
        Some(p.clamp(0.0, 1.0))
    }
    else
    {
        None
    }
}

/// Groups every row of `dataset` by its environment id, in sorted id order.
fn group_rows_by_environment(
    dataset: &CausalDataset,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<String>) {
    let mut by_environment: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for (block_index, block) in dataset.blocks.iter().enumerate()
    {
        let entry = by_environment
            .entry(block.environment.id.as_str())
            .or_default();
        for row in 0..block.n_samples()
        {
            entry.push((block_index, row));
        }
    }

    let mut rows = Vec::new();
    let mut environment_of_row = Vec::new();
    let mut environment_ids = Vec::new();
    for (index, (id, block_rows)) in by_environment.into_iter().enumerate()
    {
        environment_ids.push(id.to_string());
        for row in block_rows
        {
            rows.push(row);
            environment_of_row.push(index);
        }
    }
    (rows, environment_of_row, environment_ids)
}

/// The Bonferroni-combined p-value for one candidate subset: for each
/// environment, the residuals inside it are compared against those outside it
/// in both mean and variance; the two are Bonferroni-combined, then the
/// per-environment values are Bonferroni-combined again.
///
/// Returns `None` when no environment yielded a usable comparison at all — the
/// subset is then untestable rather than rejected.
fn subset_p_value(
    residuals: &[f64],
    environment_of_row: &[usize],
    n_environments: usize,
) -> Option<f64> {
    let mut smallest: Option<f64> = None;
    for environment in 0..n_environments
    {
        let inside: Vec<f64> = residuals
            .iter()
            .zip(environment_of_row)
            .filter(|(_, &e)| e == environment)
            .map(|(&r, _)| r)
            .collect();
        let outside: Vec<f64> = residuals
            .iter()
            .zip(environment_of_row)
            .filter(|(_, &e)| e != environment)
            .map(|(&r, _)| r)
            .collect();

        let mean_p = t_test_two_sample(&inside, &outside, false, Tail::TwoSided).map(|r| r.p_value);
        let variance_p = variance_ratio_p_value(&inside, &outside);

        let combined = match (mean_p, variance_p)
        {
            (Some(m), Some(v)) => Some(2.0 * m.min(v)),
            (Some(m), None) => Some(m),
            (None, Some(v)) => Some(v),
            (None, None) => None,
        };
        if let Some(value) = combined
        {
            if value.is_finite()
            {
                let value = value.clamp(0.0, 1.0);
                smallest = Some(smallest.map_or(value, |s: f64| s.min(value)));
            }
        }
    }
    smallest.map(|s| (s * n_environments as f64).clamp(0.0, 1.0))
}

/// Runs Invariant Causal Prediction for `target` over `candidate_predictors`.
///
/// See the module docs for the assumptions, the meaning of each outcome, and
/// the exponential cost of the subset search.
///
/// # Errors
///
/// [`CausalError::UnknownVariableIndex`] for an out-of-range index;
/// [`CausalError::SameVariable`] if the target is among its own candidate
/// predictors; [`CausalError::DuplicateConditioningVariable`] for a repeated
/// candidate; [`CausalError::UnsupportedVariableKind`] for a non-continuous
/// variable; [`CausalError::InvalidContract`] if the dataset carries fewer
/// than two distinct environments, which is the one thing ICP cannot work
/// without.
pub fn invariant_causal_prediction(
    dataset: &CausalDataset,
    target: usize,
    candidate_predictors: &[usize],
    config: &InvarianceConfig,
) -> Result<InvariantPredictionResult, CausalError> {
    let n_vars = dataset.variables.len();
    if target >= n_vars
    {
        return Err(CausalError::UnknownVariableIndex { index: target });
    }
    let mut candidates: Vec<usize> = candidate_predictors.to_vec();
    candidates.sort_unstable();
    for pair in candidates.windows(2)
    {
        if pair[0] == pair[1]
        {
            return Err(CausalError::DuplicateConditioningVariable { variable: pair[0] });
        }
    }
    for &v in &candidates
    {
        if v >= n_vars
        {
            return Err(CausalError::UnknownVariableIndex { index: v });
        }
        if v == target
        {
            return Err(CausalError::SameVariable { variable: v });
        }
    }
    for &v in std::iter::once(&target).chain(candidates.iter())
    {
        if dataset.variables[v].kind != VariableKind::Continuous
        {
            return Err(CausalError::UnsupportedVariableKind { variable: v });
        }
    }

    let (rows, environment_of_row, environment_ids) = group_rows_by_environment(dataset);
    if environment_ids.len() < 2
    {
        return Err(CausalError::InvalidContract {
            detail: "invariant causal prediction requires at least two distinct environments",
        });
    }

    let target_column = extract_column(dataset, target, &rows);
    let candidate_columns: BTreeMap<usize, Vec<f64>> = candidates
        .iter()
        .map(|&v| (v, extract_column(dataset, v, &rows)))
        .collect();

    let max_size = config
        .max_predictor_set_size
        .unwrap_or(candidates.len())
        .min(candidates.len());

    let mut accepted: Vec<AcceptedSet> = Vec::new();
    let mut sets_tested = 0usize;
    let mut warnings: Vec<String> = Vec::new();

    for size in 0..=max_size
    {
        for subset in combinations(&candidates, size)
        {
            sets_tested += 1;
            let columns: Vec<&[f64]> = subset
                .iter()
                .map(|v| candidate_columns[v].as_slice())
                .collect();

            let residuals = match residualize(&target_column, &columns, config.rank_tolerance)
            {
                Ok(residuals) => residuals,
                Err(
                    error @ (CausalError::RankDeficientConditioningSet { .. }
                    | CausalError::SolverFailure { .. }),
                ) =>
                {
                    warnings.push(format!(
                        "subset {subset:?} could not be fitted ({error}); skipped"
                    ));
                    continue;
                },
                Err(error) => return Err(error),
            };

            match subset_p_value(&residuals, &environment_of_row, environment_ids.len())
            {
                Some(p) if p > config.significance_level =>
                {
                    accepted.push(AcceptedSet {
                        predictors: subset,
                        p_value: p,
                    });
                },
                Some(_) =>
                {},
                None =>
                {
                    warnings.push(format!(
                        "subset {subset:?} yielded no usable invariance comparison; skipped"
                    ));
                },
            }
        }
    }

    // The intersection of every accepted subset.
    let causal_predictors: Vec<usize> = if accepted.is_empty()
    {
        Vec::new()
    }
    else
    {
        candidates
            .iter()
            .copied()
            .filter(|v| accepted.iter().all(|set| set.predictors.contains(v)))
            .collect()
    };

    let outcome = if accepted.is_empty()
    {
        InvariantPredictionOutcome::AssumptionsViolated
    }
    else if causal_predictors.is_empty()
    {
        InvariantPredictionOutcome::NoPredictorConfirmed
    }
    else
    {
        InvariantPredictionOutcome::CausalPredictorsIdentified
    };

    if config.max_predictor_set_size.is_some()
        && max_size < candidates.len()
        && outcome == InvariantPredictionOutcome::CausalPredictorsIdentified
    {
        warnings.push(format!(
            "the subset search was bounded at size {max_size} of {} candidates, so this \
             intersection is valid among the subsets examined and is not a complete search",
            candidates.len()
        ));
    }

    let confidence = 1.0 - config.significance_level;
    let (status, evidence, sensitivity) = match outcome
    {
        InvariantPredictionOutcome::CausalPredictorsIdentified => (
            IdentifiabilityStatus::Identifiable,
            format!(
                "{} of {sets_tested} candidate subsets were invariant across {} environments \
                 {environment_ids:?}; their intersection is {causal_predictors:?}",
                accepted.len(),
                environment_ids.len()
            ),
            "these variables are a SUBSET of the direct causes, not the complete parent set; \
             the claim holds only if the mechanism is invariant across these environments, no \
             environment intervened directly on the target, and the mechanism is linear"
                .to_string(),
        ),
        InvariantPredictionOutcome::NoPredictorConfirmed => (
            IdentifiabilityStatus::Inconclusive,
            format!(
                "{} of {sets_tested} candidate subsets were invariant across {} environments, \
                 but their intersection is empty, so no individual variable is required by \
                 every surviving explanation",
                accepted.len(),
                environment_ids.len()
            ),
            "an empty intersection is a statement about this evidence, not about the world: \
             usually too few or too mutually similar environments to discriminate"
                .to_string(),
        ),
        InvariantPredictionOutcome::AssumptionsViolated => (
            IdentifiabilityStatus::Inconclusive,
            format!(
                "none of the {sets_tested} candidate subsets was invariant across the {} \
                 environments {environment_ids:?}",
                environment_ids.len()
            ),
            "under the stated assumptions the true causal set should have survived; that none \
             did is evidence the assumptions themselves fail — a non-invariant mechanism, an \
             intervention acting directly on the target, or a non-linear mechanism"
                .to_string(),
        ),
    };

    if outcome == InvariantPredictionOutcome::AssumptionsViolated
    {
        warnings.push(
            "no subset survived the invariance test: this is positive evidence of model \
             misspecification, not merely an absence of findings"
                .to_string(),
        );
    }

    let certificate = CausalCertificate::builder(
        format!(
            "direct causes of {} among {candidates:?} (invariant causal prediction at \
             confidence {confidence:.2})",
            dataset.variables[target].name
        ),
        status,
        invariance_assumptions(),
        evidence,
    )
    .with_sensitivity(sensitivity)
    .finalize()?;

    Ok(InvariantPredictionResult {
        target,
        outcome,
        causal_predictors,
        accepted_sets: accepted,
        sets_tested,
        environments: environment_ids,
        significance_level: config.significance_level,
        certificate,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_ratio_detects_a_clear_difference_and_ignores_a_match() {
        let tight: Vec<f64> = (0..200).map(|i| (i % 7) as f64 * 0.01).collect();
        let wide: Vec<f64> = (0..200).map(|i| (i % 7) as f64 * 10.0).collect();
        let p_different = variance_ratio_p_value(&tight, &wide).unwrap();
        assert!(
            p_different < 1e-6,
            "wildly different variances must be flagged, got {p_different}"
        );

        let same: Vec<f64> = (0..200).map(|i| (i % 7) as f64 * 0.01).collect();
        let p_same = variance_ratio_p_value(&tight, &same).unwrap();
        assert!(
            p_same > 0.5,
            "identical variances must not be flagged, got {p_same}"
        );
    }

    #[test]
    fn variance_ratio_declines_degenerate_input() {
        assert!(variance_ratio_p_value(&[1.0], &[1.0, 2.0]).is_none());
        assert!(variance_ratio_p_value(&[1.0, 1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn the_subset_p_value_is_bonferroni_scaled_by_environment_count() {
        // Two environments whose residuals are drawn identically: the per-
        // environment comparison should be unremarkable, and the reported
        // value should be that minimum scaled by the environment count.
        let residuals: Vec<f64> = (0..400).map(|i| ((i * 37) % 101) as f64 - 50.0).collect();
        let environment_of_row: Vec<usize> = (0..400).map(|i| i % 2).collect();
        let p = subset_p_value(&residuals, &environment_of_row, 2).unwrap();
        assert!(
            (0.0..=1.0).contains(&p),
            "a p-value must stay in [0, 1], got {p}"
        );
        assert!(
            p > 0.05,
            "identically-distributed residuals must not be rejected, got {p}"
        );
    }

    #[test]
    fn the_subset_p_value_flags_a_mean_shift_between_environments() {
        // Environment 1's residuals are shifted well away from environment 0's.
        let residuals: Vec<f64> = (0..400)
            .map(|i| {
                let base = ((i * 37) % 101) as f64 - 50.0;
                if i % 2 == 0 { base } else { base + 500.0 }
            })
            .collect();
        let environment_of_row: Vec<usize> = (0..400).map(|i| i % 2).collect();
        let p = subset_p_value(&residuals, &environment_of_row, 2).unwrap();
        assert!(
            p < 0.01,
            "a large mean shift between environments must be rejected, got {p}"
        );
    }
}
