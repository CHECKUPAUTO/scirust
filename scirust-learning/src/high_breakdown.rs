//! Deterministic high-breakdown regression (phase 4E.3).
//!
//! Phase 4E.2's benchmark showed Huber IRLS collapses under **bad leverage**: a
//! minority of high-leverage points off the line drags the fit, and Huber's
//! vertical-residual reweighting cannot resist it (slope error ≈ the OLS error).
//! High-breakdown regression separates three concerns the existing single-start
//! trimmed least squares blurs:
//!
//! 1. **Initialization** — a *many-start* FAST-LTS search (Rousseeuw & Van
//!    Driessen, 2006): deterministic seeded elemental subsets, each concentrated
//!    by C-steps, keeping the subset with the smallest trimmed sum of squares.
//!    Many starts are what make it resist bad leverage that a single OLS start
//!    walks straight into.
//! 2. **Scale** — a proper **S-scale** (Tukey `ρ`, 50 % breakdown consistency)
//!    from the LTS fit's residuals, solved by monotone bisection.
//! 3. **Efficiency** — an optional **MM** step: a redescending Tukey
//!    M-estimation refinement at 95 % Gaussian efficiency, with the S-scale held
//!    fixed, started from the LTS fit.
//!
//! # Honesty
//!
//! The subset search is a **heuristic** over finitely many deterministic starts;
//! the result is the **best observed**, never claimed to be the global optimum
//! (`proven_optimal` is always `false`). Determinism: a fixed `SplitMix64` seed,
//! canonical (sorted) subsets, `f64::total_cmp` ranking with an index tiebreak,
//! and bounded C-step / M-iteration budgets — identical inputs and seed give a
//! bit-identical fit on every platform. Singular elemental subsets and C-step
//! cycles are counted and reported, never hidden.

use core::fmt;

use scirust_solvers::linalg::{Matrix, qr_decompose, solve_qr_least_squares};
use scirust_stats::{SplitMix64, describe};

/// Tukey S-tuning constant (≈50 % breakdown) and its normal-consistency target.
const S_TUNING: f64 = 1.547_6;
const S_TARGET: f64 = 0.5;
/// Tukey M-tuning constant (≈95 % Gaussian efficiency).
const M_TUNING: f64 = 4.685;

/// Which high-breakdown estimator to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighBreakdownMethod {
    /// Least Trimmed Squares only (high breakdown, lower Gaussian efficiency).
    LeastTrimmedSquares,
    /// MM: LTS initialization → S-scale → efficient redescending M refinement.
    MmEstimator,
}

/// Configuration for [`fit_high_breakdown`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighBreakdownConfig {
    /// The estimator.
    pub method: HighBreakdownMethod,
    /// Fraction of observations retained by LTS (`0.5..=1.0`). Lower = higher
    /// breakdown, lower efficiency. `0.5` is the maximum-breakdown choice.
    pub coverage: f64,
    /// Number of deterministic elemental subset starts.
    pub subset_starts: usize,
    /// Fit a per-model intercept.
    pub fit_intercept: bool,
    /// Maximum C-steps per start.
    pub maximum_concentration_steps: usize,
    /// Maximum MM M-refinement iterations.
    pub maximum_m_iterations: usize,
    /// `SplitMix64` seed for the elemental subset draws (recorded in the report).
    pub seed: u64,
    /// Relative coefficient-change tolerance for the MM refinement.
    pub tolerance: f64,
}

impl Default for HighBreakdownConfig {
    fn default() -> Self {
        Self {
            method: HighBreakdownMethod::MmEstimator,
            coverage: 0.75,
            subset_starts: 200,
            fit_intercept: true,
            maximum_concentration_steps: 50,
            maximum_m_iterations: 50,
            seed: 0x5152_4353,
            tolerance: 1.0e-9,
        }
    }
}

/// A reproducible high-breakdown regression report.
#[derive(Debug, Clone, PartialEq)]
pub struct HighBreakdownRegressionReport {
    /// Fitted feature coefficients (length `p`).
    pub coefficients: Vec<f64>,
    /// Fitted intercept (`0.0` when `fit_intercept` was false).
    pub intercept: f64,
    /// The initialization method (`"fast_lts"`).
    pub initial_method: String,
    /// The final estimator (`"least_trimmed_squares"` or `"mm"`).
    pub final_method: String,
    /// Indices retained by the winning LTS support (ascending).
    pub retained_indices: Vec<usize>,
    /// Indices trimmed by LTS (ascending).
    pub rejected_indices: Vec<usize>,
    /// Robust residual scale (the S-scale).
    pub residual_scale: f64,
    /// Final-stage iterations (C-steps for LTS, M-iterations for MM).
    pub iterations: usize,
    /// Whether the final stage reached its stability/tolerance criterion.
    pub converged: bool,
    /// How many starts entered a C-step cycle (kept at their fixed point).
    pub detected_cycles: usize,
    /// Elemental subsets skipped because they were rank deficient.
    pub singular_starts: usize,
    /// Whether the trimmed optimum is *proven* global (always `false`: the search
    /// is a heuristic over finitely many starts — the result is best-observed).
    pub proven_optimal: bool,
    /// Non-fatal notes.
    pub warnings: Vec<String>,
}

impl HighBreakdownRegressionReport {
    /// Predict the response for one feature row.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut value = self.intercept;
        for (coefficient, feature) in self.coefficients.iter().zip(features)
        {
            value += coefficient * feature;
        }
        value
    }
}

/// Typed high-breakdown regression errors.
#[derive(Debug, Clone, PartialEq)]
pub enum HighBreakdownError {
    /// The design has zero rows or zero feature columns.
    EmptyDesign,
    /// The target length does not match the feature row count.
    TargetLengthMismatch {
        /// Feature row count.
        rows: usize,
        /// Target length.
        targets: usize,
    },
    /// A feature or target entry is non-finite.
    NonFiniteValue {
        /// Row of the offending entry.
        row: usize,
        /// Column (`usize::MAX` marks the target).
        col: usize,
        /// The value.
        value: f64,
    },
    /// A configuration value was out of range.
    InvalidConfig {
        /// What was wrong.
        detail: String,
    },
    /// Fewer rows than the fitted column count plus one.
    TooFewObservations {
        /// Minimum rows required.
        required: usize,
        /// Rows supplied.
        found: usize,
    },
    /// Every elemental subset start was rank deficient — no fit could be formed.
    AllStartsSingular,
}

impl fmt::Display for HighBreakdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDesign => formatter.write_str("the design has zero rows or columns"),
            Self::TargetLengthMismatch { rows, targets } => write!(
                formatter,
                "target length {targets} does not match the {rows} feature rows"
            ),
            Self::NonFiniteValue { row, col, value } =>
            {
                write!(
                    formatter,
                    "non-finite value {value} at row {row}, column {col}"
                )
            },
            Self::InvalidConfig { detail } => write!(formatter, "invalid configuration: {detail}"),
            Self::TooFewObservations { required, found } => write!(
                formatter,
                "high-breakdown regression needs at least {required} rows, found {found}"
            ),
            Self::AllStartsSingular =>
            {
                formatter.write_str("every elemental subset start was rank deficient")
            },
        }
    }
}

impl std::error::Error for HighBreakdownError {}

/// Fit a deterministic high-breakdown regression.
///
/// # Errors
///
/// [`HighBreakdownError`] on empty/mismatched/non-finite input, an invalid
/// configuration, too few rows, or an all-singular set of starts.
pub fn fit_high_breakdown(
    features: &Matrix,
    targets: &[f64],
    config: HighBreakdownConfig,
) -> Result<HighBreakdownRegressionReport, HighBreakdownError> {
    let n = features.rows();
    let p = features.cols();
    if n == 0 || p == 0
    {
        return Err(HighBreakdownError::EmptyDesign);
    }
    if targets.len() != n
    {
        return Err(HighBreakdownError::TargetLengthMismatch {
            rows: n,
            targets: targets.len(),
        });
    }
    validate_config(&config)?;
    for i in 0..n
    {
        for j in 0..p
        {
            let value = features[(i, j)];
            if !value.is_finite()
            {
                return Err(HighBreakdownError::NonFiniteValue {
                    row: i,
                    col: j,
                    value,
                });
            }
        }
        if !targets[i].is_finite()
        {
            return Err(HighBreakdownError::NonFiniteValue {
                row: i,
                col: usize::MAX,
                value: targets[i],
            });
        }
    }

    let fitted_columns = p + usize::from(config.fit_intercept);
    if n < fitted_columns + 1
    {
        return Err(HighBreakdownError::TooFewObservations {
            required: fitted_columns + 1,
            found: n,
        });
    }

    let coverage_count =
        ((config.coverage * n as f64).ceil() as usize).clamp(fitted_columns + 1, n);

    // ── FAST-LTS: many deterministic elemental starts, each concentrated. ──
    let mut rng = SplitMix64::new(config.seed);
    let mut best: Option<LtsCandidate> = None;
    let mut detected_cycles = 0;
    let mut singular_starts = 0;

    for _ in 0..config.subset_starts
    {
        let subset = elemental_subset(&mut rng, n, fitted_columns);
        let Some(start) = solve_ols(features, targets, &subset, config.fit_intercept)
        else
        {
            singular_starts += 1;
            continue;
        };
        let outcome = concentrate(
            features,
            targets,
            start,
            coverage_count,
            config.fit_intercept,
            config.maximum_concentration_steps,
        );
        let Some(candidate) = outcome
        else
        {
            singular_starts += 1;
            continue;
        };
        if candidate.cycled
        {
            detected_cycles += 1;
        }
        if best
            .as_ref()
            .is_none_or(|current| candidate.trimmed_sse < current.trimmed_sse)
        {
            best = Some(candidate);
        }
    }

    let Some(lts) = best
    else
    {
        return Err(HighBreakdownError::AllStartsSingular);
    };

    let mut warnings = Vec::new();
    let all_residuals = residuals(features, targets, &lts.coefficients, config.fit_intercept);
    let residual_scale = m_scale(&all_residuals, S_TUNING, S_TARGET);
    if residual_scale <= 0.0
    {
        warnings.push(
            "S-scale is zero: at least half the observations fit the LTS line exactly".to_string(),
        );
    }

    let mut retained_indices = lts.retained.clone();
    retained_indices.sort_unstable();
    let retained_set: std::collections::BTreeSet<usize> =
        retained_indices.iter().copied().collect();
    let rejected_indices: Vec<usize> = (0..n).filter(|i| !retained_set.contains(i)).collect();

    let (coefficients_full, iterations, converged, final_method) = match config.method
    {
        HighBreakdownMethod::LeastTrimmedSquares => (
            lts.coefficients.clone(),
            lts.steps,
            true,
            "least_trimmed_squares",
        ),
        HighBreakdownMethod::MmEstimator =>
        {
            if residual_scale > 0.0
            {
                let refinement = mm_refine(
                    features,
                    targets,
                    &lts.coefficients,
                    residual_scale,
                    config.fit_intercept,
                    config.maximum_m_iterations,
                    config.tolerance,
                );
                (
                    refinement.coefficients,
                    refinement.iterations,
                    refinement.converged,
                    "mm",
                )
            }
            else
            {
                warnings.push(
                    "MM refinement skipped: the S-scale is zero, so the LTS fit is returned"
                        .to_string(),
                );
                (
                    lts.coefficients.clone(),
                    lts.steps,
                    true,
                    "least_trimmed_squares",
                )
            }
        },
    };

    let (coefficients, intercept) = split_intercept(&coefficients_full, p, config.fit_intercept);

    Ok(HighBreakdownRegressionReport {
        coefficients,
        intercept,
        initial_method: "fast_lts".to_string(),
        final_method: final_method.to_string(),
        retained_indices,
        rejected_indices,
        residual_scale,
        iterations,
        converged,
        detected_cycles,
        singular_starts,
        proven_optimal: false,
        warnings,
    })
}

fn validate_config(config: &HighBreakdownConfig) -> Result<(), HighBreakdownError> {
    if !(config.coverage.is_finite() && config.coverage >= 0.5 && config.coverage <= 1.0)
    {
        return Err(HighBreakdownError::InvalidConfig {
            detail: "coverage must be in [0.5, 1.0]".to_string(),
        });
    }
    if config.subset_starts == 0
    {
        return Err(HighBreakdownError::InvalidConfig {
            detail: "subset_starts must be positive".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0)
    {
        return Err(HighBreakdownError::InvalidConfig {
            detail: "tolerance must be finite and positive".to_string(),
        });
    }
    Ok(())
}

/// One concentrated LTS start.
struct LtsCandidate {
    coefficients: Vec<f64>,
    retained: Vec<usize>,
    trimmed_sse: f64,
    steps: usize,
    cycled: bool,
}

struct MmOutcome {
    coefficients: Vec<f64>,
    iterations: usize,
    converged: bool,
}

/// A deterministic elemental subset of `size` distinct indices from `0..n`, drawn
/// by a partial Fisher–Yates shuffle so the draw order is canonical for the seed.
fn elemental_subset(rng: &mut SplitMix64, n: usize, size: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for position in 0..size
    {
        let span = (n - position) as u64;
        let pick = position + (rng.next_u64() % span) as usize;
        pool.swap(position, pick);
    }
    let mut subset: Vec<usize> = pool[..size].to_vec();
    subset.sort_unstable();
    subset
}

/// Solve OLS on the given rows through the QR of `[features | intercept]`.
/// Returns `None` if the subset design is rank deficient.
fn solve_ols(
    features: &Matrix,
    targets: &[f64],
    rows: &[usize],
    fit_intercept: bool,
) -> Option<Vec<f64>> {
    let p = features.cols();
    let fitted_columns = p + usize::from(fit_intercept);
    let mut design = Matrix::zeros(rows.len(), fitted_columns);
    let mut response = vec![0.0_f64; rows.len()];
    for (local, &row) in rows.iter().enumerate()
    {
        for j in 0..p
        {
            design[(local, j)] = features[(row, j)];
        }
        if fit_intercept
        {
            design[(local, p)] = 1.0;
        }
        response[local] = targets[row];
    }
    let factorization = qr_decompose(design).ok()?;
    let reciprocal_condition = factorization.rcond().unwrap_or(0.0);
    if reciprocal_condition < 1.0e-12
    {
        return None;
    }
    solve_qr_least_squares(&factorization, &response).ok()
}

/// Weighted OLS through the QR of the `√w`-scaled design.
fn solve_weighted(
    features: &Matrix,
    targets: &[f64],
    weights: &[f64],
    fit_intercept: bool,
) -> Option<Vec<f64>> {
    let n = features.rows();
    let p = features.cols();
    let fitted_columns = p + usize::from(fit_intercept);
    let mut design = Matrix::zeros(n, fitted_columns);
    let mut response = vec![0.0_f64; n];
    for i in 0..n
    {
        let root = weights[i].max(0.0).sqrt();
        for j in 0..p
        {
            design[(i, j)] = root * features[(i, j)];
        }
        if fit_intercept
        {
            design[(i, p)] = root;
        }
        response[i] = root * targets[i];
    }
    let factorization = qr_decompose(design).ok()?;
    if factorization.rcond().unwrap_or(0.0) < 1.0e-12
    {
        return None;
    }
    solve_qr_least_squares(&factorization, &response).ok()
}

fn predict(features: &Matrix, row: usize, beta: &[f64], fit_intercept: bool) -> f64 {
    let p = features.cols();
    let mut value = 0.0;
    for j in 0..p
    {
        value += features[(row, j)] * beta[j];
    }
    if fit_intercept
    {
        value += beta[p];
    }
    value
}

fn residuals(features: &Matrix, targets: &[f64], beta: &[f64], fit_intercept: bool) -> Vec<f64> {
    (0..features.rows())
        .map(|i| targets[i] - predict(features, i, beta, fit_intercept))
        .collect()
}

/// Concentration (C-steps): repeatedly refit on the `coverage_count` smallest
/// absolute residuals until the support is stable, a cycle repeats, or the step
/// budget is spent. Returns `None` if a refit is rank deficient.
fn concentrate(
    features: &Matrix,
    targets: &[f64],
    mut beta: Vec<f64>,
    coverage_count: usize,
    fit_intercept: bool,
    maximum_steps: usize,
) -> Option<LtsCandidate> {
    let n = features.rows();
    let mut seen: Vec<Vec<usize>> = Vec::new();
    let mut retained: Vec<usize> = Vec::new();
    let mut trimmed_sse = f64::INFINITY;
    let mut steps = 0;
    let mut cycled = false;

    for _ in 0..maximum_steps
    {
        steps += 1;
        let residual = residuals(features, targets, &beta, fit_intercept);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            residual[a]
                .abs()
                .total_cmp(&residual[b].abs())
                .then(a.cmp(&b))
        });
        let mut next: Vec<usize> = order[..coverage_count].to_vec();
        next.sort_unstable();
        let objective: f64 = next.iter().map(|&i| residual[i] * residual[i]).sum();

        if next == retained
        {
            trimmed_sse = objective;
            break;
        }
        if seen.contains(&next)
        {
            cycled = true;
            trimmed_sse = objective;
            retained = next;
            break;
        }
        seen.push(retained.clone());
        retained = next;
        trimmed_sse = objective;
        beta = solve_ols(features, targets, &retained, fit_intercept)?;
    }

    Some(LtsCandidate {
        coefficients: beta,
        retained,
        trimmed_sse,
        steps,
        cycled,
    })
}

/// Normalized Tukey biweight `ρ`, scaled to `[0, 1]` (`ρ(∞) = 1`).
fn tukey_rho(u: f64, c: f64) -> f64 {
    let t = u / c;
    if t.abs() <= 1.0
    {
        let quad = 1.0 - t * t;
        1.0 - quad * quad * quad
    }
    else
    {
        1.0
    }
}

/// Tukey biweight IRLS weight `ψ(u)/u = (1 − (u/c)²)²` for `|u| ≤ c`, else `0`.
fn tukey_weight(u: f64, c: f64) -> f64 {
    let t = u / c;
    if t.abs() <= 1.0
    {
        let quad = 1.0 - t * t;
        quad * quad
    }
    else
    {
        0.0
    }
}

/// The S-scale: the `s > 0` solving `mean(ρ_c(rᵢ / s)) = target`, by bisection on
/// the monotone-decreasing objective. Returns `0.0` when at least half the
/// residuals are exactly zero.
fn m_scale(residual: &[f64], c: f64, target: f64) -> f64 {
    let n = residual.len() as f64;
    let absolute: Vec<f64> = residual.iter().map(|r| r.abs()).collect();
    let median_abs = describe::median(&absolute);
    if median_abs <= 0.0
    {
        return 0.0;
    }
    let objective = |scale: f64| {
        absolute
            .iter()
            .map(|&r| tukey_rho(r / scale, c))
            .sum::<f64>()
            / n
    };

    let mut low = median_abs * 1.0e-3;
    let mut high = median_abs * 1.0e3 + 1.0;
    for _ in 0..200
    {
        let mid = 0.5 * (low + high);
        if objective(mid) > target
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
        if (high - low) <= 1.0e-12 * high
        {
            break;
        }
    }
    0.5 * (low + high)
}

/// MM refinement: redescending Tukey M-estimation (efficiency tuning) with the
/// S-scale fixed, started from the LTS coefficients.
fn mm_refine(
    features: &Matrix,
    targets: &[f64],
    initial: &[f64],
    scale: f64,
    fit_intercept: bool,
    maximum_iterations: usize,
    tolerance: f64,
) -> MmOutcome {
    let n = features.rows();
    let mut beta = initial.to_vec();
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..maximum_iterations
    {
        iterations += 1;
        let residual = residuals(features, targets, &beta, fit_intercept);
        let weights: Vec<f64> = (0..n)
            .map(|i| tukey_weight(residual[i] / scale, M_TUNING))
            .collect();
        let Some(next) = solve_weighted(features, targets, &weights, fit_intercept)
        else
        {
            break;
        };
        let change: f64 = beta
            .iter()
            .zip(&next)
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);
        let magnitude = beta.iter().map(|b| b.abs()).fold(1.0_f64, f64::max);
        beta = next;
        if change <= tolerance * magnitude
        {
            converged = true;
            break;
        }
    }

    MmOutcome {
        coefficients: beta,
        iterations,
        converged,
    }
}

fn split_intercept(full: &[f64], p: usize, fit_intercept: bool) -> (Vec<f64>, f64) {
    let coefficients = full[..p].to_vec();
    let intercept = if fit_intercept { full[p] } else { 0.0 };
    (coefficients, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(xs: &[f64]) -> Matrix {
        Matrix::from_row_major(xs.len(), 1, xs.to_vec())
    }

    fn clean(n: usize) -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n)
            .map(|i| 2.0 * i as f64 + (((i * 7) % 11) as f64 - 5.0) * 0.2)
            .collect();
        (xs, ys)
    }

    fn with_bad_leverage() -> (Vec<f64>, Vec<f64>) {
        let (mut xs, mut ys) = clean(50);
        for k in 0..10
        {
            let x = 120.0 + k as f64;
            xs.push(x);
            ys.push(2.0 * x - 160.0);
        }
        (xs, ys)
    }

    fn with_vertical() -> (Vec<f64>, Vec<f64>) {
        let (mut xs, mut ys) = clean(50);
        for k in 0..10
        {
            let x = (k * 4) as f64;
            xs.push(x);
            ys.push(2.0 * x + 70.0);
        }
        (xs, ys)
    }

    fn fit(xs: &[f64], ys: &[f64], method: HighBreakdownMethod) -> HighBreakdownRegressionReport {
        fit_high_breakdown(
            &design(xs),
            ys,
            HighBreakdownConfig {
                method,
                ..HighBreakdownConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn recovers_the_true_slope_on_clean_data() {
        let (xs, ys) = clean(60);
        let mm = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert!(
            (mm.coefficients[0] - 2.0).abs() < 0.05,
            "slope {}",
            mm.coefficients[0]
        );
        assert_eq!(mm.final_method, "mm");
        assert_eq!(mm.initial_method, "fast_lts");
        assert!(!mm.proven_optimal);
    }

    #[test]
    fn recovers_the_true_slope_under_bad_leverage() {
        // The headline: this is exactly where phase-4E.2 showed Huber fails.
        let (xs, ys) = with_bad_leverage();
        let mm = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert!(
            (mm.coefficients[0] - 2.0).abs() < 0.1,
            "MM slope {} should be ~2 under bad leverage",
            mm.coefficients[0]
        );
        let lts = fit(&xs, &ys, HighBreakdownMethod::LeastTrimmedSquares);
        assert!(
            (lts.coefficients[0] - 2.0).abs() < 0.1,
            "LTS slope {}",
            lts.coefficients[0]
        );
        // The ten off-line points should be among the rejected set.
        let rejected: std::collections::BTreeSet<usize> =
            lts.rejected_indices.iter().copied().collect();
        let caught = (50..60).filter(|i| rejected.contains(i)).count();
        assert!(
            caught >= 9,
            "expected the leverage cluster rejected, caught {caught}/10"
        );
    }

    #[test]
    fn recovers_the_true_slope_under_vertical_contamination() {
        let (xs, ys) = with_vertical();
        let mm = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert!(
            (mm.coefficients[0] - 2.0).abs() < 0.1,
            "slope {}",
            mm.coefficients[0]
        );
    }

    #[test]
    fn is_deterministic_and_seed_reproducible() {
        let (xs, ys) = with_bad_leverage();
        let a = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        let b = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert_eq!(a, b, "same seed → bit-identical report");
        // A different seed still recovers the slope (best-observed over starts).
        let c = fit_high_breakdown(
            &design(&xs),
            &ys,
            HighBreakdownConfig {
                seed: 0xABCD_1234,
                ..HighBreakdownConfig::default()
            },
        )
        .unwrap();
        assert!(
            (c.coefficients[0] - 2.0).abs() < 0.1,
            "seed-varied slope {}",
            c.coefficients[0]
        );
    }

    #[test]
    fn mm_is_more_efficient_than_lts_on_clean_data() {
        // On clean data both are near the truth, but MM (95% efficiency) should be
        // at least as close to the OLS/true slope as raw LTS.
        let (xs, ys) = clean(60);
        let lts = fit(&xs, &ys, HighBreakdownMethod::LeastTrimmedSquares);
        let mm = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        let lts_error = (lts.coefficients[0] - 2.0).abs();
        let mm_error = (mm.coefficients[0] - 2.0).abs();
        assert!(
            mm_error <= lts_error + 1e-9,
            "MM {mm_error} vs LTS {lts_error}"
        );
    }

    #[test]
    fn reports_residual_scale_and_retained_partition() {
        let (xs, ys) = with_bad_leverage();
        let report = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert!(report.residual_scale > 0.0);
        assert_eq!(
            report.retained_indices.len() + report.rejected_indices.len(),
            xs.len()
        );
        // coverage 0.75 of 60 = 45 retained.
        assert_eq!(report.retained_indices.len(), 45);
    }

    #[test]
    fn above_breakdown_contamination_is_not_silently_trusted() {
        // 55% contamination exceeds any high-breakdown guarantee; we do NOT assert
        // recovery — only that the fit is produced deterministically without panic
        // and reports a scale. (An honest non-recovery is a valid outcome.)
        let mut xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        for k in 0..50
        {
            let x = 100.0 + k as f64;
            xs.push(x);
            ys.push(-3.0 * x + 500.0);
        }
        let report = fit(&xs, &ys, HighBreakdownMethod::MmEstimator);
        assert!(report.residual_scale >= 0.0);
        assert!(report.coefficients[0].is_finite());
    }

    #[test]
    fn invalid_inputs_and_config_are_typed_errors() {
        let (xs, ys) = clean(40);
        assert!(matches!(
            fit_high_breakdown(&design(&xs), &ys[..39], HighBreakdownConfig::default())
                .unwrap_err(),
            HighBreakdownError::TargetLengthMismatch { .. }
        ));
        assert!(matches!(
            fit_high_breakdown(
                &design(&xs),
                &ys,
                HighBreakdownConfig {
                    coverage: 0.2,
                    ..HighBreakdownConfig::default()
                }
            )
            .unwrap_err(),
            HighBreakdownError::InvalidConfig { .. }
        ));
        let tiny = design(&[1.0, 2.0]);
        assert!(matches!(
            fit_high_breakdown(&tiny, &[1.0, 2.0], HighBreakdownConfig::default()).unwrap_err(),
            HighBreakdownError::TooFewObservations { .. }
        ));
    }

    #[test]
    fn m_scale_solves_the_estimating_equation() {
        // For residuals with a clear scale, the S-scale is positive and finite.
        let residual: Vec<f64> = (0..40).map(|i| ((i * 3) % 7) as f64 - 3.0).collect();
        let scale = m_scale(&residual, S_TUNING, S_TARGET);
        assert!(scale > 0.0 && scale.is_finite());
        // Mean rho at the solved scale is close to the 0.5 target.
        let mean_rho: f64 = residual
            .iter()
            .map(|&r| tukey_rho(r / scale, S_TUNING))
            .sum::<f64>()
            / 40.0;
        assert!((mean_rho - 0.5).abs() < 0.05, "mean rho {mean_rho}");
    }
}
