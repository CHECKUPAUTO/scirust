//! Deterministic, robust **conditional-independence (CI) testing**: the
//! statistical oracle later causal-discovery algorithms (e.g. PC-Stable) need
//! for statements of the form `X ⟂ Y | Z`.
//!
//! # What this tests, and what it does not
//!
//! A test here evaluates a *statistical* hypothesis: is the (linear) partial
//! association between `X` and `Y`, controlling for `Z`, distinguishable from
//! noise under a stated model, calibration, and significance level? This is
//! **not** causal discovery. In particular, a
//! [`IndependenceDecision::IndependentWithinThreshold`] result does **not**
//! establish:
//!
//! - that a causal edge is absent;
//! - that the (eventual) discovered graph is acyclic;
//! - causal sufficiency (no latent confounding);
//! - faithfulness;
//! - the absence of selection bias;
//! - correct temporal ordering.
//!
//! `X ⟂ Y | Z` (or its rejection) is *evidence* a future discovery procedure
//! consumes, not a causal conclusion in itself — see the crate root's "Causal
//! interpretation" section, which this module's results are subject to
//! exactly the same way.
//!
//! # The three methods, in one sentence each
//!
//! - [`ConditionalIndependenceMethod::GaussianPartialCorrelation`] —
//!   `crate::partial_correlation`: QR-residualized Pearson correlation,
//!   optionally Fisher-z calibrated; assumes an (approximately) linear,
//!   (approximately) Gaussian-residual relationship.
//! - [`ConditionalIndependenceMethod::RobustPartialCorrelation`] —
//!   `crate::robust_partial_correlation`: the same *shape* of statistic,
//!   computed via the existing OGK robust scatter estimator instead of OLS;
//!   never fabricates a classical p-value unless explicitly asked to.
//! - [`ConditionalIndependenceMethod::PermutationPartialCorrelation`] —
//!   the classical statistic, calibrated by deterministic residual
//!   permutation (`crate::permutation_calibration`) instead of the Fisher-z
//!   asymptotic approximation.
//!
//! All three are **linear** association measures. A linear partial
//! correlation can be exactly zero while `X` and `Y` remain conditionally
//! *dependent* through a nonlinear relationship — see the crate's nonlinear
//! adversarial test for an explicit, undisguised demonstration of this
//! failure mode.
//!
//! # Decision semantics
//!
//! [`IndependenceDecision`] has three outcomes, never collapsed to a boolean:
//! [`IndependenceDecision::Dependent`] (the null is rejected at
//! `significance_level`), [`IndependenceDecision::IndependentWithinThreshold`]
//! (the null was *not* rejected — not "proven true"), and
//! [`IndependenceDecision::Inconclusive`] (no calibrated p-value exists at
//! all: insufficient degrees of freedom, `NoPValue` calibration chosen, or a
//! similar honest non-answer). Malformed *inputs* (unknown variable, `x==y`,
//! …) are typed [`CausalError`]s, not `Inconclusive` — the distinction in
//! the crate's honesty rules is between a request that cannot be answered
//! (error) and one that was answered with "no evidence either way" (a valid,
//! reported [`IndependenceDecision::Inconclusive`]).
//!
//! # Determinism contract
//!
//! Given the same dataset, `x`/`y`/`conditioned_on` indices, config, and
//! (where applicable) seed: the conditioning set is canonicalized by sorting
//! before any computation, so callers passing the same set in a different
//! order get identical results (tested); row selection and column extraction
//! use a fixed block-then-row order; QR/SVD (`scirust-solvers`) and OGK
//! (`scirust-multivariate`) are both deterministic by construction (no
//! internal RNG, fixed accumulation order); the one seeded procedure
//! (permutation calibration) is a single continuing
//! [`scirust_stats::SplitMix64`] stream, entirely determined by its `seed`
//! and the sample count (see `crate::permutation_calibration`). No
//! floating-point sort occurs anywhere in this module: `scirust-solvers`'s
//! SVD already returns singular values pre-sorted descending, and permutation
//! exceedance counting is a direct `>=` comparison on values already
//! validated finite — so `f64::total_cmp` is not needed here.

use crate::assumptions::CausalAssumption;
use crate::dataset::CausalDataset;
use crate::error::CausalError;
use crate::partial_correlation::{classical_partial_correlation, fisher_z_p_value};
use crate::permutation_calibration::{apply_permutation, calibrate_by_permutation};
use crate::robust_partial_correlation::{
    RobustCalibration, apply_robust_calibration, robust_partial_correlation,
};
use crate::variable::VariableKind;
use scirust_multivariate::RobustScatterConfig;
use scirust_stats::describe::variance;

const ZERO_VARIANCE_TOLERANCE: f64 = 1e-14;

/// Which regime's rows a test draws its (observational) sample from.
///
/// A conditional-independence test assumes one *homogeneous* observational
/// distribution; mixing rows from different interventional regimes (or
/// observational with interventional) without saying so would silently
/// answer a different question than the one asked. This forces the choice
/// to be explicit.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RegimeSelection {
    /// Every row whose block's [`crate::Environment`] carries no
    /// interventions.
    ObservationalOnly,
    /// Every row whose block's [`crate::Environment::id`] equals this string,
    /// regardless of whether that environment carries interventions.
    Environment(String),
    /// Exactly these global row indices (block-then-row order, matching
    /// [`crate::CausalDataset::total_samples`]'s own summation order).
    ExplicitRows(Vec<usize>),
}

/// How to handle a value that cannot be used.
///
/// Under [`crate::CausalDataset`]'s current invariants every stored value is
/// already guaranteed finite ([`crate::SampleBlock::new`] rejects non-finite
/// entries at construction), so **neither variant can currently do anything
/// non-trivial** — `CompleteCases` observably removes zero rows, and `Error`
/// can never fire. This policy exists for the dataset type this crate may
/// have in the future, not because today's data can be incomplete; it is
/// still implemented and tested precisely so that claim is checked, not
/// assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MissingValuePolicy {
    /// A non-finite selected value is a typed [`CausalError::NonFiniteSample`].
    Error,
    /// Drop any row with a non-finite selected value; report the effective
    /// sample count actually used.
    CompleteCases,
}

/// Which residualization the permutation-calibrated method uses to compute
/// its underlying statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResidualizationMethod {
    /// QR-based ordinary least squares (the only implemented method).
    OrdinaryLeastSquares,
}

/// Which conditional-independence method (and, where relevant, calibration)
/// to run. See the module docs for a one-sentence summary of each.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConditionalIndependenceMethod {
    /// QR-residualized Pearson correlation. `fisher_z = true` calibrates via
    /// `crate::partial_correlation::fisher_z_p_value`; `false` reports the
    /// statistic with `p_value = None`.
    GaussianPartialCorrelation { fisher_z: bool },
    /// OGK-residualized Pearson correlation (see
    /// `crate::robust_partial_correlation`), calibrated per `calibration`.
    RobustPartialCorrelation {
        scatter: RobustScatterConfig,
        calibration: RobustCalibration,
    },
    /// The classical statistic, calibrated by deterministic residual
    /// permutation instead of Fisher-z.
    PermutationPartialCorrelation {
        permutations: usize,
        seed: u64,
        residualization: ResidualizationMethod,
    },
}

/// What calibration actually produced a result's `p_value` (or its
/// deliberate absence) — the *reported* counterpart to the *requested*
/// choices inside [`ConditionalIndependenceMethod`] / [`RobustCalibration`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CalibrationMethod {
    FisherZ,
    NoPValue,
    Permutation { permutations: usize, seed: u64 },
}

/// The three-way outcome a conditional-independence test reports. Never
/// collapsed into a boolean — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndependenceDecision {
    /// The null (`X ⟂ Y | Z`) was rejected at `significance_level`.
    Dependent,
    /// The null was **not** rejected under the declared model, calibration,
    /// `significance_level`, and sample — not a proof of independence.
    IndependentWithinThreshold,
    /// No calibrated p-value exists (insufficient degrees of freedom,
    /// `NoPValue` calibration, or another honest non-answer).
    Inconclusive,
}

/// One conditional-independence test's full, reproducible outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConditionalIndependenceResult {
    pub x: usize,
    pub y: usize,
    /// Canonicalized (sorted, deduplicated-by-construction) conditioning set.
    pub conditioned_on: Vec<usize>,
    /// The signed statistic (a correlation coefficient in `[-1, 1]`).
    pub statistic: f64,
    /// `|statistic|` — an unsigned effect-size summary.
    pub effect_size: f64,
    pub p_value: Option<f64>,
    pub significance_level: f64,
    pub decision: IndependenceDecision,
    /// Rows actually used (after regime selection and missing-value policy).
    pub sample_count: usize,
    /// The conditioning design's numerical rank (`0` for an empty conditioning
    /// set — there is no design to have a rank).
    pub effective_rank: usize,
    pub method: ConditionalIndependenceMethod,
    pub calibration: CalibrationMethod,
    pub assumptions: Vec<CausalAssumption>,
    pub warnings: Vec<String>,
}

/// A test of `X ⟂ Y | Z`. See the module docs for the exact scientific scope.
pub trait ConditionalIndependenceTest {
    /// # Errors
    ///
    /// A typed [`CausalError`] for a malformed request (unknown/duplicate/
    /// endpoint-overlapping variable, insufficient samples, rank-deficient
    /// conditioning set, …) — see [`CausalError`]'s variants. A scientifically
    /// unresolved but *well-formed* request reports
    /// [`IndependenceDecision::Inconclusive`] in `Ok(..)`, never an `Err`.
    fn test(
        &self,
        dataset: &CausalDataset,
        x: usize,
        y: usize,
        conditioned_on: &[usize],
    ) -> Result<ConditionalIndependenceResult, CausalError>;
}

/// Configuration for [`PartialCorrelationTest`]. Constructed via
/// [`ConditionalIndependenceConfig::new`] (validated), refined with the
/// chainable `with_*` methods.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalIndependenceConfig {
    pub significance_level: f64,
    pub method: ConditionalIndependenceMethod,
    /// Relative singular-value tolerance for rank-deficiency detection in
    /// [`ConditionalIndependenceMethod::GaussianPartialCorrelation`] /
    /// [`ConditionalIndependenceMethod::PermutationPartialCorrelation`]'s QR
    /// residualization (unused by the robust method, whose own OGK fit
    /// rejects a singular scatter directly).
    pub rank_tolerance: f64,
    pub regime: RegimeSelection,
    pub missing_value_policy: MissingValuePolicy,
}

impl ConditionalIndependenceConfig {
    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] if `significance_level` is not
    /// finite and in `(0, 1)`, or if `method` carries a permutation count of
    /// `0`.
    pub fn new(
        significance_level: f64,
        method: ConditionalIndependenceMethod,
    ) -> Result<Self, CausalError> {
        if !significance_level.is_finite() || significance_level <= 0.0 || significance_level >= 1.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "significance_level",
                value: significance_level,
            });
        }
        Self::validate_method(&method)?;
        Ok(Self {
            significance_level,
            method,
            rank_tolerance: 1e-9,
            regime: RegimeSelection::ObservationalOnly,
            missing_value_policy: MissingValuePolicy::Error,
        })
    }

    fn validate_method(method: &ConditionalIndependenceMethod) -> Result<(), CausalError> {
        let permutations = match method
        {
            ConditionalIndependenceMethod::PermutationPartialCorrelation {
                permutations, ..
            } => Some(*permutations),
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                calibration: RobustCalibration::Permutation { permutations, .. },
                ..
            } => Some(*permutations),
            _ => None,
        };
        if permutations == Some(0)
        {
            return Err(CausalError::InvalidConfiguration {
                name: "permutations",
                value: 0.0,
            });
        }
        Ok(())
    }

    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] if `rank_tolerance` is not
    /// finite and positive.
    pub fn with_rank_tolerance(mut self, rank_tolerance: f64) -> Result<Self, CausalError> {
        if !rank_tolerance.is_finite() || rank_tolerance <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "rank_tolerance",
                value: rank_tolerance,
            });
        }
        self.rank_tolerance = rank_tolerance;
        Ok(self)
    }

    #[must_use]
    pub fn with_regime(mut self, regime: RegimeSelection) -> Self {
        self.regime = regime;
        self
    }

    #[must_use]
    pub fn with_missing_value_policy(mut self, policy: MissingValuePolicy) -> Self {
        self.missing_value_policy = policy;
        self
    }
}

/// The one [`ConditionalIndependenceTest`] implementor this phase ships:
/// partial correlation, dispatched over classical / robust / permutation
/// calibration by [`ConditionalIndependenceConfig::method`].
pub struct PartialCorrelationTest {
    config: ConditionalIndependenceConfig,
}

impl PartialCorrelationTest {
    #[must_use]
    pub fn new(config: ConditionalIndependenceConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn config(&self) -> &ConditionalIndependenceConfig {
        &self.config
    }

    fn zero_variance_error(
        x_residual: &[f64],
        y_residual: &[f64],
        x: usize,
        y: usize,
    ) -> CausalError {
        let variable = if variance(x_residual) <= ZERO_VARIANCE_TOLERANCE
        {
            x
        }
        else
        {
            y
        };
        let _ = y_residual; // documents symmetry of the check above
        CausalError::ZeroVariance { variable }
    }
}

/// Global row index (block-then-row order) → `(block_index, row_within_block)`.
fn locate_global_row(dataset: &CausalDataset, global_row: usize) -> Option<(usize, usize)> {
    let mut remaining = global_row;
    for (block_index, block) in dataset.blocks.iter().enumerate()
    {
        if remaining < block.n_samples()
        {
            return Some((block_index, remaining));
        }
        remaining -= block.n_samples();
    }
    None
}

fn select_rows(
    dataset: &CausalDataset,
    regime: &RegimeSelection,
) -> Result<Vec<(usize, usize)>, CausalError> {
    match regime
    {
        RegimeSelection::ObservationalOnly => Ok(dataset
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.environment.is_observational())
            .flat_map(|(block_index, block)| {
                (0..block.n_samples()).map(move |row| (block_index, row))
            })
            .collect()),
        RegimeSelection::Environment(id) => Ok(dataset
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| &block.environment.id == id)
            .flat_map(|(block_index, block)| {
                (0..block.n_samples()).map(move |row| (block_index, row))
            })
            .collect()),
        RegimeSelection::ExplicitRows(rows) => rows
            .iter()
            .map(|&global_row| {
                locate_global_row(dataset, global_row)
                    .ok_or(CausalError::UnknownVariableIndex { index: global_row })
            })
            .collect(),
    }
}

fn extract_column(dataset: &CausalDataset, variable: usize, rows: &[(usize, usize)]) -> Vec<f64> {
    rows.iter()
        .map(|&(block_index, row)| {
            let block = &dataset.blocks[block_index];
            block.data()[row * block.n_variables() + variable]
        })
        .collect()
}

/// Applies [`MissingValuePolicy`] over the selected rows for the specific
/// columns this test will use. See [`MissingValuePolicy`]'s docs: under the
/// current [`CausalDataset`] invariants this can never actually remove or
/// reject anything, since every stored value is already finite.
fn apply_missing_value_policy(
    dataset: &CausalDataset,
    rows: &[(usize, usize)],
    x: usize,
    y: usize,
    conditioned_on: &[usize],
    policy: MissingValuePolicy,
) -> Result<Vec<(usize, usize)>, CausalError> {
    let mut columns = vec![x, y];
    columns.extend_from_slice(conditioned_on);

    let is_row_complete = |&(block_index, row): &(usize, usize)| -> Result<bool, CausalError> {
        let block = &dataset.blocks[block_index];
        for &variable in &columns
        {
            let value = block.data()[row * block.n_variables() + variable];
            if !value.is_finite()
            {
                match policy
                {
                    MissingValuePolicy::Error =>
                    {
                        return Err(CausalError::NonFiniteSample {
                            row,
                            variable,
                            value,
                        });
                    },
                    MissingValuePolicy::CompleteCases => return Ok(false),
                }
            }
        }
        Ok(true)
    };

    let mut kept = Vec::with_capacity(rows.len());
    for &row in rows
    {
        if is_row_complete(&row)?
        {
            kept.push(row);
        }
    }
    Ok(kept)
}

impl ConditionalIndependenceTest for PartialCorrelationTest {
    fn test(
        &self,
        dataset: &CausalDataset,
        x: usize,
        y: usize,
        conditioned_on: &[usize],
    ) -> Result<ConditionalIndependenceResult, CausalError> {
        let n_vars = dataset.variables.len();
        if x >= n_vars
        {
            return Err(CausalError::UnknownVariableIndex { index: x });
        }
        if y >= n_vars
        {
            return Err(CausalError::UnknownVariableIndex { index: y });
        }
        for &z in conditioned_on
        {
            if z >= n_vars
            {
                return Err(CausalError::UnknownVariableIndex { index: z });
            }
        }
        if x == y
        {
            return Err(CausalError::SameVariable { variable: x });
        }

        let mut conditioned_on: Vec<usize> = conditioned_on.to_vec();
        conditioned_on.sort_unstable();
        for pair in conditioned_on.windows(2)
        {
            if pair[0] == pair[1]
            {
                return Err(CausalError::DuplicateConditioningVariable { variable: pair[0] });
            }
        }
        if conditioned_on.contains(&x)
        {
            return Err(CausalError::ConditioningContainsEndpoint { variable: x });
        }
        if conditioned_on.contains(&y)
        {
            return Err(CausalError::ConditioningContainsEndpoint { variable: y });
        }

        for &variable in std::iter::once(&x)
            .chain(std::iter::once(&y))
            .chain(conditioned_on.iter())
        {
            if dataset.variables[variable].kind != VariableKind::Continuous
            {
                return Err(CausalError::UnsupportedVariableKind { variable });
            }
        }

        let selected_rows = select_rows(dataset, &self.config.regime)?;
        let selected_rows = apply_missing_value_policy(
            dataset,
            &selected_rows,
            x,
            y,
            &conditioned_on,
            self.config.missing_value_policy,
        )?;

        let min_required = if conditioned_on.is_empty()
        {
            2
        }
        else
        {
            conditioned_on.len() + 2
        };
        if selected_rows.len() < min_required
        {
            return Err(CausalError::InsufficientSamples {
                required: min_required,
                actual: selected_rows.len(),
            });
        }

        let x_col = extract_column(dataset, x, &selected_rows);
        let y_col = extract_column(dataset, y, &selected_rows);
        let z_cols: Vec<Vec<f64>> = conditioned_on
            .iter()
            .map(|&z| extract_column(dataset, z, &selected_rows))
            .collect();
        let z_refs: Vec<&[f64]> = z_cols.iter().map(Vec::as_slice).collect();

        let sample_count = selected_rows.len();
        let conditioning_size = conditioned_on.len();

        match self.config.method.clone()
        {
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z } =>
            {
                let outcome = classical_partial_correlation(
                    &x_col,
                    &y_col,
                    &z_refs,
                    self.config.rank_tolerance,
                )?;
                let r = outcome.r.ok_or_else(|| {
                    Self::zero_variance_error(&outcome.x_residual, &outcome.y_residual, x, y)
                })?;

                let (p_value, mut warnings) = if fisher_z
                {
                    match fisher_z_p_value(r, sample_count, conditioning_size)
                    {
                        Some(p) => (Some(p), Vec::new()),
                        None => (
                            None,
                            vec![
                                "insufficient residual degrees of freedom for the Fisher-z \
                                 approximation"
                                    .to_string(),
                            ],
                        ),
                    }
                }
                else
                {
                    (None, Vec::new())
                };

                let mut assumptions = vec![CausalAssumption::CorrectFunctionalForm];
                if fisher_z
                {
                    assumptions.push(CausalAssumption::AdequateSampleSize);
                }
                warnings.sort();

                Ok(build_result(
                    x,
                    y,
                    conditioned_on,
                    r,
                    p_value,
                    self.config.significance_level,
                    sample_count,
                    outcome.rank,
                    ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z },
                    if fisher_z
                    {
                        CalibrationMethod::FisherZ
                    }
                    else
                    {
                        CalibrationMethod::NoPValue
                    },
                    assumptions,
                    warnings,
                ))
            },
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                scatter,
                calibration,
            } =>
            {
                let outcome = robust_partial_correlation(&x_col, &y_col, &z_refs, &scatter)?;
                let r = outcome.r.ok_or_else(|| {
                    Self::zero_variance_error(&outcome.x_residual, &outcome.y_residual, x, y)
                })?;

                let calibration_outcome = apply_robust_calibration(
                    r,
                    &outcome.x_residual,
                    &outcome.y_residual,
                    sample_count,
                    conditioning_size,
                    calibration,
                    &scatter,
                )?;

                let calibration_method = match calibration
                {
                    RobustCalibration::NoPValue => CalibrationMethod::NoPValue,
                    RobustCalibration::GaussianApproximation => CalibrationMethod::FisherZ,
                    RobustCalibration::Permutation { permutations, seed } =>
                    {
                        CalibrationMethod::Permutation { permutations, seed }
                    },
                };
                let mut assumptions = vec![CausalAssumption::CorrectFunctionalForm];
                match calibration
                {
                    RobustCalibration::GaussianApproximation =>
                    {
                        assumptions.push(CausalAssumption::AdequateSampleSize);
                    },
                    RobustCalibration::Permutation { .. } =>
                    {
                        assumptions.push(CausalAssumption::ResidualExchangeability);
                    },
                    RobustCalibration::NoPValue =>
                    {},
                }

                let effective_rank = if conditioning_size == 0
                {
                    0
                }
                else
                {
                    1 + conditioning_size
                };

                Ok(build_result(
                    x,
                    y,
                    conditioned_on,
                    r,
                    calibration_outcome.p_value,
                    self.config.significance_level,
                    sample_count,
                    effective_rank,
                    ConditionalIndependenceMethod::RobustPartialCorrelation {
                        scatter,
                        calibration,
                    },
                    calibration_method,
                    assumptions,
                    calibration_outcome.warnings,
                ))
            },
            ConditionalIndependenceMethod::PermutationPartialCorrelation {
                permutations,
                seed,
                residualization,
            } =>
            {
                let ResidualizationMethod::OrdinaryLeastSquares = residualization;
                let outcome = classical_partial_correlation(
                    &x_col,
                    &y_col,
                    &z_refs,
                    self.config.rank_tolerance,
                )?;
                let r = outcome.r.ok_or_else(|| {
                    Self::zero_variance_error(&outcome.x_residual, &outcome.y_residual, x, y)
                })?;

                let calibration_outcome =
                    calibrate_by_permutation(r, sample_count, permutations, seed, |order| {
                        let permuted_y = apply_permutation(&outcome.y_residual, order);
                        crate::partial_correlation::pearson_correlation(
                            &outcome.x_residual,
                            &permuted_y,
                        )
                    })?;

                let mut warnings = Vec::new();
                if calibration_outcome.completed_permutations
                    < calibration_outcome.requested_permutations
                {
                    warnings.push(format!(
                        "{} of {} requested permutations were skipped (degenerate resample)",
                        calibration_outcome.requested_permutations
                            - calibration_outcome.completed_permutations,
                        calibration_outcome.requested_permutations
                    ));
                }

                Ok(build_result(
                    x,
                    y,
                    conditioned_on,
                    r,
                    Some(calibration_outcome.p_value),
                    self.config.significance_level,
                    sample_count,
                    outcome.rank,
                    ConditionalIndependenceMethod::PermutationPartialCorrelation {
                        permutations,
                        seed,
                        residualization,
                    },
                    CalibrationMethod::Permutation { permutations, seed },
                    vec![
                        CausalAssumption::CorrectFunctionalForm,
                        CausalAssumption::ResidualExchangeability,
                    ],
                    warnings,
                ))
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    x: usize,
    y: usize,
    conditioned_on: Vec<usize>,
    statistic: f64,
    p_value: Option<f64>,
    significance_level: f64,
    sample_count: usize,
    effective_rank: usize,
    method: ConditionalIndependenceMethod,
    calibration: CalibrationMethod,
    assumptions: Vec<CausalAssumption>,
    warnings: Vec<String>,
) -> ConditionalIndependenceResult {
    let decision = match p_value
    {
        Some(p) if p <= significance_level => IndependenceDecision::Dependent,
        Some(_) => IndependenceDecision::IndependentWithinThreshold,
        None => IndependenceDecision::Inconclusive,
    };
    ConditionalIndependenceResult {
        x,
        y,
        conditioned_on,
        statistic,
        effect_size: statistic.abs(),
        p_value,
        significance_level,
        decision,
        sample_count,
        effective_rank,
        method,
        calibration,
        assumptions,
        warnings,
    }
}
