//! Robust partial correlation: reuses the existing OGK robust scatter
//! estimator (`scirust-multivariate`, Program 4 phase 4E.1) — this module
//! does **not** reimplement OGK.
//!
//! # What is computed, precisely — two robust stages, not one
//!
//! **Stage 1 — residualization.** For a target variable `V` and conditioning
//! columns `Z`, [`robust_residualize_one`] fits OGK jointly on `{V} ∪ Z` and
//! extracts `V`'s **OGK-implied linear residual** against `Z`: writing `P`
//! for the fitted precision matrix
//! ([`scirust_multivariate::RobustScatterModel::inverse_scatter`]) with index
//! `0` reserved for `V`, the standard precision-matrix conditional-mean
//! identity gives the coefficient of `Z_j` in the (robust) linear projection
//! of `V` onto `Z` as `-P[0, 1+j] / P[0, 0]`; the residual is `V`'s
//! robust-centered value minus that projection. For an empty conditioning set
//! this stage is a no-op (there is nothing to project out): the "residual" is
//! just `V` itself.
//!
//! **Stage 2 — correlating the residuals.** This is the part an earlier
//! version of this module got wrong, and is worth being explicit about:
//! ordinary Pearson correlation of two *already-centered* vectors is **not**
//! robust, because [`pearson_correlation`] recomputes its own mean
//! internally — any prior (even robust) centering is mathematically
//! inconsequential to its output (correlation is translation-invariant by
//! construction). Robustly *centering* the residuals is therefore not enough
//! to make the final number robust. [`correlation_from_robust_fit`] instead
//! fits OGK a **second** time, jointly on the two (Stage-1) residual vectors
//! (a 2-dimensional fit), and reads the correlation directly off that fit's
//! precision matrix via the standard identity `r = -P[0,1] / sqrt(P[0,0] *
//! P[1,1])` — for a 2x2 matrix this is exactly equivalent to reading the
//! correlation off the forward scatter matrix, so this is a real,
//! bounded-influence robust correlation coefficient, not a relabeled
//! classical one. `pearson_correlation` is still used, but only as a
//! **zero-variance detector** (its actual returned coefficient is discarded)
//! so the crate's existing, tested zero-variance-attribution logic keeps
//! working unchanged.
//!
//! Doing Stage 1 once for `X` and once for `Y`, then Stage 2 on the two
//! residuals, is the robust analogue of `crate::partial_correlation`'s
//! "QR-residualize, then correlate" — same overall shape, with *both* steps
//! (not just residualization) replaced by a bounded-influence procedure.
//!
//! # Invariances and honesty caveats
//!
//! - OGK's own equivariance is **not** full affine equivariance (see
//!   [`scirust_multivariate::AchievedEquivariance`]); it is translation- and
//!   scaling-equivariant per coordinate. Both fitting stages above inherit
//!   whatever equivariance the underlying fit achieves — no stronger claim is
//!   made here.
//! - A dimension OGK marks inactive (near-degenerate robust scale) is still
//!   included in the projection sum with whatever coefficient the fit
//!   produced; this module does not special-case inactive dimensions beyond
//!   what OGK itself already guards (see
//!   [`scirust_multivariate::RobustScatterModel::active_dimensions`]).
//! - This remains fundamentally an **association measure**: a small robust
//!   partial correlation is not proof of conditional independence, robust or
//!   otherwise.
//! - Robust does not mean breakdown-proof at any contamination level: a
//!   large enough or sufficiently *structured* (internally correlated)
//!   contaminating block can still move this statistic — see the crate's
//!   adversarial tests for an explicit, undisguised demonstration.
//! - **On clean, outlier-free data this statistic is routinely bit-identical
//!   to the classical one** (see the deterministic benchmark), not merely
//!   close — this is an expected, correct property of `RobustScatterConfig`'s
//!   default hard-reweighting OGK
//!   ([`scirust_multivariate::RobustScatterModel`]'s χ²-cutoff C-step), not a
//!   coincidence or an unused robust path: when every row passes the cutoff,
//!   the reweighted scatter *is* the ordinary covariance of all rows,
//!   formula-for-formula. The two methods only visibly diverge once some
//!   rows are actually down-weighted or rejected.
//! - **The classical Fisher-z null distribution is not proven exact for an
//!   OGK-derived statistic.** [`RobustCalibration::GaussianApproximation`]
//!   applies the same asymptotic formula as the classical method purely as an
//!   approximation, and every result computed under it is documented as such
//!   — never as a calibrated exact p-value. The honest default,
//!   [`RobustCalibration::NoPValue`], reports the statistic and effect size
//!   with `p_value = None`.
//! - Permutation calibration ([`RobustCalibration::Permutation`]) recomputes
//!   *the same* Stage-2 fit-and-read-off-precision-matrix statistic on every
//!   permuted arrangement (not ordinary Pearson) — this keeps the observed
//!   and permuted values methodologically consistent draws of the identical
//!   statistic, as a permutation test requires. A permutation whose 2-D OGK
//!   refit itself fails (e.g. a degenerate resample) is excluded from that
//!   permutation's completed/exceedance counts exactly as a `None` from the
//!   recompute closure already is (see `crate::permutation_calibration`),
//!   not propagated as a hard error.

use crate::error::CausalError;
use crate::partial_correlation::{fisher_z_p_value, pearson_correlation};
use crate::permutation_calibration::{apply_permutation, calibrate_by_permutation};
use scirust_multivariate::{Matrix as MultivariateMatrix, RobustScatterConfig, RobustScatterModel};

/// Row-major columns → the nested-`Vec` `Matrix` layout
/// [`scirust_multivariate::RobustScatterModel::fit`] expects.
fn columns_to_multivariate_matrix(columns: &[&[f64]]) -> MultivariateMatrix {
    let cols = columns.len();
    let rows = columns[0].len();
    let data = (0..rows)
        .map(|row| (0..cols).map(|col| columns[col][row]).collect())
        .collect();
    MultivariateMatrix { rows, cols, data }
}

fn fit_ogk(
    columns: &[&[f64]],
    scatter_config: &RobustScatterConfig,
) -> Result<RobustScatterModel, CausalError> {
    let matrix = columns_to_multivariate_matrix(columns);
    RobustScatterModel::fit(&matrix, *scatter_config).map_err(CausalError::ScatterFailure)
}

/// Fits OGK once on `{focal} ∪ z` and returns `focal`'s robust residual
/// against `z` (row-major, one entry per sample).
fn robust_residualize_one(
    focal: &[f64],
    z_columns: &[&[f64]],
    scatter_config: &RobustScatterConfig,
) -> Result<(Vec<f64>, RobustScatterModel), CausalError> {
    let mut joint_columns = Vec::with_capacity(1 + z_columns.len());
    joint_columns.push(focal);
    joint_columns.extend_from_slice(z_columns);

    let model = fit_ogk(&joint_columns, scatter_config)?;
    let p = &model.inverse_scatter;
    let p00 = p.data[0][0];
    let location0 = model.location[0];

    let residual = if z_columns.is_empty() || p00.abs() <= 0.0
    {
        focal.iter().map(|&v| v - location0).collect()
    }
    else
    {
        let coefficients: Vec<f64> = (1..p.cols).map(|j| -p.data[0][j] / p00).collect();
        (0..focal.len())
            .map(|row| {
                let projection: f64 = coefficients
                    .iter()
                    .zip(z_columns)
                    .zip(1..p.cols)
                    .map(|((&coef, z_col), j)| coef * (z_col[row] - model.location[j]))
                    .sum();
                focal[row] - location0 - projection
            })
            .collect()
    };
    Ok((residual, model))
}

/// The observed robust statistic plus the residual vectors it was computed
/// from (the latter reused directly for permutation calibration).
///
/// `r` is `None` exactly when [`pearson_correlation`] finds zero variance on
/// either residual — `x_residual`/`y_residual` are still populated so the
/// caller can attribute *which* side was degenerate.
pub(crate) struct RobustPartialCorrelationOutcome {
    pub r: Option<f64>,
    pub x_residual: Vec<f64>,
    pub y_residual: Vec<f64>,
}

/// Computes the robust partial correlation of `x` and `y` given `z_columns`
/// (see the module docs for exactly what is computed).
///
/// # Errors
///
/// [`CausalError::ScatterFailure`] if the underlying OGK fit fails (see
/// [`scirust_multivariate::RobustGeometryError`]).
pub(crate) fn robust_partial_correlation(
    x: &[f64],
    y: &[f64],
    z_columns: &[&[f64]],
    scatter_config: &RobustScatterConfig,
) -> Result<RobustPartialCorrelationOutcome, CausalError> {
    let (x_residual, y_residual) = if z_columns.is_empty()
    {
        // Nothing to project out; Stage 2 (correlation_from_robust_fit) does
        // its own robust centering internally, so no pre-centering is needed
        // here (see the module docs on why pre-centering would be
        // inconsequential to an ordinary-Pearson read-off anyway).
        (x.to_vec(), y.to_vec())
    }
    else
    {
        let (x_residual, _) = robust_residualize_one(x, z_columns, scatter_config)?;
        let (y_residual, _) = robust_residualize_one(y, z_columns, scatter_config)?;
        (x_residual, y_residual)
    };

    // Zero-variance detection reuses the classical method's own
    // tolerance-checked formula; the coefficient it returns is otherwise
    // discarded in favor of `correlation_from_robust_fit` below (see the
    // module docs for why that second fit, not this one, is what actually
    // makes the reported statistic robust).
    if pearson_correlation(&x_residual, &y_residual).is_none()
    {
        return Ok(RobustPartialCorrelationOutcome {
            r: None,
            x_residual,
            y_residual,
        });
    }

    let r = correlation_from_robust_fit(&x_residual, &y_residual, scatter_config)?;
    Ok(RobustPartialCorrelationOutcome {
        r,
        x_residual,
        y_residual,
    })
}

/// Stage 2 of this module's robust statistic: a **second**, two-dimensional
/// OGK fit on `a` and `b` (typically the Stage-1 residuals), reading the
/// correlation directly off that fit's precision matrix via
/// `r = -P[0,1] / sqrt(P[0,0] * P[1,1])`. See the module docs for why this —
/// rather than ordinary Pearson correlation of two robust-centered vectors —
/// is the step that actually makes this method's output robust.
///
/// Returns `Ok(None)` (not an error) if the fitted precision matrix's own
/// diagonal is degenerate; the caller is expected to have already ruled out
/// an ordinary zero-variance input via [`pearson_correlation`], so this is
/// expected to be rare in practice.
///
/// # Errors
///
/// [`CausalError::ScatterFailure`] if the underlying OGK fit fails.
pub(crate) fn correlation_from_robust_fit(
    a: &[f64],
    b: &[f64],
    scatter_config: &RobustScatterConfig,
) -> Result<Option<f64>, CausalError> {
    let joint = fit_ogk(&[a, b], scatter_config)?;
    let p = &joint.inverse_scatter;
    if p.data[0][0] <= 0.0 || p.data[1][1] <= 0.0
    {
        return Ok(None);
    }
    let raw = -p.data[0][1] / (p.data[0][0] * p.data[1][1]).sqrt();
    const OVERSHOOT_TOLERANCE: f64 = 1e-9;
    let r = if raw > 1.0 && raw <= 1.0 + OVERSHOOT_TOLERANCE
    {
        1.0
    }
    else if (-1.0 - OVERSHOOT_TOLERANCE..-1.0).contains(&raw)
    {
        -1.0
    }
    else
    {
        raw
    };
    Ok(Some(r))
}

/// The calibration strategy for a [`crate::ConditionalIndependenceMethod::RobustPartialCorrelation`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RobustCalibration {
    /// Report the statistic and effect size only; `p_value = None`. The
    /// honest default — no classical null is claimed to apply.
    NoPValue,
    /// Apply the same Fisher-z asymptotic formula the classical method uses,
    /// to this robust statistic. **Not proven exact for an OGK-derived
    /// estimate** — always reported with that caveat in `warnings`.
    GaussianApproximation,
    /// Deterministic residual-permutation calibration (see
    /// `crate::permutation_calibration`), applied to this robust
    /// statistic's own residuals.
    Permutation { permutations: usize, seed: u64 },
}

/// The outcome of applying a [`RobustCalibration`] to a computed robust
/// statistic: a p-value (or its deliberate absence) plus any warnings the
/// calibration choice itself warrants.
pub(crate) struct RobustCalibrationOutcome {
    pub p_value: Option<f64>,
    pub warnings: Vec<String>,
}

/// Calibrates an already-computed, confirmed-non-degenerate robust statistic
/// (`observed_r`, from the residuals `x_residual`/`y_residual`). The caller is
/// responsible for having already ruled out `RobustPartialCorrelationOutcome::r
/// == None` (a zero-variance residual) before calling this. `scatter_config`
/// must be the same configuration `observed_r` was computed with — permutation
/// calibration re-fits [`correlation_from_robust_fit`] under it for every
/// resample, and a mismatched config would silently compare two different
/// statistics.
pub(crate) fn apply_robust_calibration(
    observed_r: f64,
    x_residual: &[f64],
    y_residual: &[f64],
    sample_count: usize,
    conditioning_size: usize,
    calibration: RobustCalibration,
    scatter_config: &RobustScatterConfig,
) -> Result<RobustCalibrationOutcome, CausalError> {
    match calibration
    {
        RobustCalibration::NoPValue => Ok(RobustCalibrationOutcome {
            p_value: None,
            warnings: Vec::new(),
        }),
        RobustCalibration::GaussianApproximation =>
        {
            let p_value = fisher_z_p_value(observed_r, sample_count, conditioning_size);
            let mut warnings = vec![
                "Gaussian-approximation calibration applied to a robust (OGK-derived) \
                 statistic: the classical Fisher-z null distribution is not proven exact here, \
                 only used as an approximation."
                    .to_string(),
            ];
            if p_value.is_none()
            {
                warnings.push(
                    "insufficient residual degrees of freedom for the Fisher-z approximation"
                        .to_string(),
                );
            }
            Ok(RobustCalibrationOutcome { p_value, warnings })
        },
        RobustCalibration::Permutation { permutations, seed } =>
        {
            let calibration_outcome =
                calibrate_by_permutation(observed_r, sample_count, permutations, seed, |order| {
                    let permuted_y = apply_permutation(y_residual, order);
                    correlation_from_robust_fit(x_residual, &permuted_y, scatter_config)
                        .ok()
                        .flatten()
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
            Ok(RobustCalibrationOutcome {
                p_value: Some(calibration_outcome.p_value),
                warnings,
            })
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_multivariate::RobustScatterMethod;

    fn default_scatter_config() -> RobustScatterConfig {
        RobustScatterConfig::default()
    }

    #[test]
    fn columns_to_matrix_is_row_major() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let m = columns_to_multivariate_matrix(&[&a, &b]);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data[0], vec![1.0, 10.0]);
        assert_eq!(m.data[2], vec![3.0, 30.0]);
    }

    #[test]
    fn empty_z_reduces_to_a_robust_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 14.2, 15.8];
        let outcome = robust_partial_correlation(&x, &y, &[], &default_scatter_config()).unwrap();
        let r = outcome.r.unwrap();
        assert!(
            r > 0.9,
            "clean perfect linear relation, expected r near 1, got {r}"
        );
    }

    /// Regression guard for a real bug this module once had: because
    /// [`pearson_correlation`] recenters internally, robustly *centering* `x`
    /// and `y` and then Pearson-correlating them is mathematically identical
    /// to Pearson-correlating the raw values — providing **zero** actual
    /// robustness. Under contamination, a correct implementation must differ
    /// from plain Pearson; this pins that down so the bug cannot silently
    /// return.
    #[test]
    fn contaminated_empty_z_result_differs_from_plain_pearson_correlation() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut y: Vec<f64> = vec![1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9];
        x.extend_from_slice(&[11.0, -11.0]);
        y.extend_from_slice(&[-40.0, 40.0]);

        let robust = robust_partial_correlation(&x, &y, &[], &default_scatter_config())
            .unwrap()
            .r
            .unwrap();
        let classical = pearson_correlation(&x, &y).unwrap();
        assert!(
            (robust - classical).abs() > 0.05,
            "robust ({robust}) must differ meaningfully from plain Pearson ({classical}) under \
             this contamination, or the fit is providing no actual robustness"
        );
    }

    #[test]
    fn no_p_value_calibration_reports_none() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.3, 11.7, 14.2, 15.8];
        let outcome = robust_partial_correlation(&x, &y, &[], &default_scatter_config()).unwrap();
        let calibration = apply_robust_calibration(
            outcome.r.unwrap(),
            &outcome.x_residual,
            &outcome.y_residual,
            x.len(),
            0,
            RobustCalibration::NoPValue,
            &default_scatter_config(),
        )
        .unwrap();
        assert_eq!(calibration.p_value, None);
        assert!(calibration.warnings.is_empty());
    }

    #[test]
    fn gaussian_approximation_always_warns_about_its_own_inexactness() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [2.0, 3.0, 5.0, 4.0, 8.0, 7.0, 9.0, 10.0, 12.0, 15.0];
        let outcome = robust_partial_correlation(&x, &y, &[], &default_scatter_config()).unwrap();
        let calibration = apply_robust_calibration(
            outcome.r.unwrap(),
            &outcome.x_residual,
            &outcome.y_residual,
            x.len(),
            0,
            RobustCalibration::GaussianApproximation,
            &default_scatter_config(),
        )
        .unwrap();
        assert!(calibration.p_value.is_some());
        assert!(
            calibration
                .warnings
                .iter()
                .any(|w| w.contains("not proven exact")),
            "expected an explicit inexactness caveat, got {:?}",
            calibration.warnings
        );
    }

    #[test]
    fn permutation_calibration_is_deterministic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0];
        let outcome = robust_partial_correlation(&x, &y, &[], &default_scatter_config()).unwrap();
        let run = || {
            apply_robust_calibration(
                outcome.r.unwrap(),
                &outcome.x_residual,
                &outcome.y_residual,
                x.len(),
                0,
                RobustCalibration::Permutation {
                    permutations: 200,
                    seed: 7,
                },
                &default_scatter_config(),
            )
            .unwrap()
            .p_value
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn residualize_one_reduces_to_centering_with_empty_z() {
        let x = [3.0, 5.0, 7.0, 9.0];
        let (residual, model) = robust_residualize_one(&x, &[], &default_scatter_config()).unwrap();
        for (r, &v) in residual.iter().zip(&x)
        {
            assert!((r - (v - model.location[0])).abs() < 1e-12);
        }
    }

    #[test]
    fn ogk_config_default_is_ogk_not_classical() {
        assert!(matches!(
            RobustScatterConfig::default().method,
            RobustScatterMethod::Ogk { .. }
        ));
    }
}
