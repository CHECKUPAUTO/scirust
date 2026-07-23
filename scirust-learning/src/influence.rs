//! Deterministic leverage and influence diagnostics (phase 4E.2).
//!
//! Direction 3D showed a single residual moment (excess kurtosis) does not
//! predict when robustness pays. A central missing variable is **leverage**:
//! observations unusual in *feature* space move an OLS fit even when their
//! post-fit residual looks unremarkable. This module reports, per observation,
//! the classical influence diagnostics *and* a robust feature-space distance
//! (via the phase-4E.1 OGK scatter), then classifies each point.
//!
//! # What is and is not claimed
//!
//! These are **evidence, not truth**. The hat leverage and studentized residuals
//! are exact OLS quantities; the classification thresholds are conventional
//! (documented, configurable) rule-of-thumb cutoffs, not decisions. A point
//! flagged `BadLeverage` is *worth inspecting*, not *proven* corrupt.
//!
//! # Numerics
//!
//! Leverage is the hat-matrix diagonal `hᵢ = ‖Qᵢ‖²` from the thin QR of the
//! (optionally intercept-augmented) design — never an explicit `XᵀX` inverse.
//! Because the hat matrix is invariant to any non-singular reparameterization of
//! the columns, raw and standardized features give identical leverage when an
//! intercept is present. Rank deficiency (a tiny reciprocal condition number) is
//! a typed error, never a silent pseudo-inverse. Determinism: pure fixed-order
//! loops; the only randomness-free "search" is the OGK fit, itself deterministic.

use core::fmt;

use scirust_multivariate::{
    Matrix as GeometryMatrix, RobustGeometryError, RobustScatterConfig, RobustScatterMethod,
    RobustScatterModel, RobustUnivariateScale,
};
use scirust_solvers::linalg::{Matrix, qr_decompose, solve_qr_least_squares};
use scirust_stats::{ChiSquared, Distribution};

/// Typed leverage/influence errors.
#[derive(Debug, Clone, PartialEq)]
pub enum InfluenceError {
    /// The design has zero rows or zero feature columns.
    EmptyDesign,
    /// The target vector length does not match the feature row count.
    TargetLengthMismatch {
        /// Feature row count.
        rows: usize,
        /// Target length.
        targets: usize,
    },
    /// A feature or target entry is `NaN` or `±∞`.
    NonFiniteValue {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry (`usize::MAX` marks the target).
        col: usize,
        /// The non-finite value.
        value: f64,
    },
    /// Fewer observations than the residual degrees of freedom need
    /// (`rows >= fitted_columns + 2`).
    TooFewObservations {
        /// Minimum rows required.
        required: usize,
        /// Rows supplied.
        found: usize,
    },
    /// The design is (numerically) rank deficient: its reciprocal condition
    /// number fell below `minimum_reciprocal_condition`. Leverage from a
    /// rank-deficient design is not trustworthy, so it is refused rather than
    /// computed through a pseudo-inverse.
    RankDeficient {
        /// The observed reciprocal condition number.
        reciprocal_condition: f64,
        /// The configured floor.
        floor: f64,
    },
    /// A configuration value was out of range.
    InvalidConfig {
        /// What was wrong.
        detail: String,
    },
    /// The QR factorization or least-squares solve failed.
    Factorization {
        /// Underlying detail.
        detail: String,
    },
}

impl fmt::Display for InfluenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDesign => formatter.write_str("the design has zero rows or columns"),
            Self::TargetLengthMismatch { rows, targets } => write!(
                formatter,
                "target length {targets} does not match the {rows} feature rows"
            ),
            Self::NonFiniteValue { row, col, value } => write!(
                formatter,
                "non-finite value {value} at row {row}, column {col}"
            ),
            Self::TooFewObservations { required, found } => write!(
                formatter,
                "leverage diagnostics need at least {required} rows, found {found}"
            ),
            Self::RankDeficient {
                reciprocal_condition,
                floor,
            } => write!(
                formatter,
                "design is rank deficient (reciprocal condition {reciprocal_condition:.3e} < floor {floor:.3e})"
            ),
            Self::InvalidConfig { detail } => write!(formatter, "invalid configuration: {detail}"),
            Self::Factorization { detail } => write!(formatter, "factorization failed: {detail}"),
        }
    }
}

impl std::error::Error for InfluenceError {}

/// A per-observation influence classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationInfluenceClass {
    /// Unremarkable in both feature space and residual.
    Regular,
    /// A large residual with ordinary leverage (a "vertical" outlier).
    VerticalOutlier,
    /// High leverage but a small residual — unusual in feature space yet
    /// consistent with the fit (helps, does not distort).
    GoodLeverage,
    /// High leverage *and* a large residual — the dangerous case that can drag
    /// an OLS fit toward itself.
    BadLeverage,
    /// The classical hat leverage and the robust feature-space distance disagree
    /// on whether the point is high leverage; treat with care.
    Ambiguous,
}

/// The diagnostics for one observation.
#[derive(Debug, Clone, PartialEq)]
pub struct InfluenceRecord {
    /// The observation's row index.
    pub row_index: usize,
    /// Hat-matrix diagonal `hᵢ ∈ [0, 1]`.
    pub leverage: f64,
    /// Robust OGK Mahalanobis distance in feature space (`None` if the robust
    /// scatter could not be fitted — for example rank-deficient features).
    pub robust_distance: Option<f64>,
    /// OLS residual `yᵢ − ŷᵢ`.
    pub residual: f64,
    /// Internally studentized residual `rᵢ / (σ̂ √(1 − hᵢ))` (`None` when `1 − hᵢ`
    /// or the residual scale is degenerate).
    pub studentized_residual: Option<f64>,
    /// Cook's distance (`None` under the same degeneracies).
    pub cook_distance: Option<f64>,
    /// `‖DFBETAᵢ‖₂` — the Euclidean size of the coefficient shift caused by
    /// deleting this observation (`None` under degeneracy).
    pub coefficient_displacement: Option<f64>,
    /// The influence class.
    pub class: ObservationInfluenceClass,
}

/// Configuration for [`InfluenceReport::fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfluenceConfig {
    /// Include an intercept column in the design.
    pub fit_intercept: bool,
    /// High-leverage cutoff as a multiple of the mean leverage `p/n`
    /// (conventionally `2` or `3`).
    pub leverage_multiple: f64,
    /// `|studentized residual|` above which a residual counts as large
    /// (conventionally `2` to `3`).
    pub residual_threshold: f64,
    /// Chi-squared quantile (on the feature dimension) for the robust-distance
    /// cutoff (conventionally `0.975`).
    pub robust_distance_quantile: f64,
    /// Apply OGK reweighting when fitting the robust feature-space scatter.
    pub ogk_reweight: bool,
    /// Reject the design as rank deficient below this reciprocal condition
    /// number.
    pub minimum_reciprocal_condition: f64,
}

impl Default for InfluenceConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            leverage_multiple: 2.0,
            residual_threshold: 2.5,
            robust_distance_quantile: 0.975,
            ogk_reweight: true,
            minimum_reciprocal_condition: 1.0e-10,
        }
    }
}

/// A reproducible leverage/influence report over a dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct InfluenceReport {
    /// One record per observation, in input order.
    pub records: Vec<InfluenceRecord>,
    /// The residual scale `σ̂ = √(RSS / (n − p))`.
    pub residual_scale: f64,
    /// Fitted column count `p` (features plus the intercept when present).
    pub fitted_columns: usize,
    /// Observation count `n`.
    pub observation_count: usize,
    /// The high-leverage threshold used (`leverage_multiple · p / n`).
    pub leverage_threshold: f64,
    /// The large-residual threshold used.
    pub residual_threshold: f64,
    /// The robust-distance threshold used (`None` when robust distances were
    /// unavailable).
    pub robust_distance_threshold: Option<f64>,
    /// Non-fatal notes (for example a robust scatter that could not be fitted).
    pub warnings: Vec<String>,
}

impl InfluenceReport {
    /// Compute leverage and influence diagnostics for the OLS fit of `targets`
    /// on `features`.
    ///
    /// # Errors
    ///
    /// [`InfluenceError`] on empty/mismatched/non-finite input, too few rows, an
    /// invalid configuration, a rank-deficient design, or a factorization
    /// failure.
    pub fn fit(
        features: &Matrix,
        targets: &[f64],
        config: InfluenceConfig,
    ) -> Result<Self, InfluenceError> {
        let n = features.rows();
        let p = features.cols();
        if n == 0 || p == 0
        {
            return Err(InfluenceError::EmptyDesign);
        }
        if targets.len() != n
        {
            return Err(InfluenceError::TargetLengthMismatch {
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
                    return Err(InfluenceError::NonFiniteValue {
                        row: i,
                        col: j,
                        value,
                    });
                }
            }
            if !targets[i].is_finite()
            {
                return Err(InfluenceError::NonFiniteValue {
                    row: i,
                    col: usize::MAX,
                    value: targets[i],
                });
            }
        }

        let fitted_columns = p + usize::from(config.fit_intercept);
        let required = fitted_columns + 2;
        if n < required
        {
            return Err(InfluenceError::TooFewObservations { required, found: n });
        }

        // Design [features | intercept-last], matching the crate's fitting order.
        let mut design = Matrix::zeros(n, fitted_columns);
        for i in 0..n
        {
            for j in 0..p
            {
                design[(i, j)] = features[(i, j)];
            }
            if config.fit_intercept
            {
                design[(i, p)] = 1.0;
            }
        }

        let factorization =
            qr_decompose(design).map_err(|error| InfluenceError::Factorization {
                detail: error.to_string(),
            })?;
        // A singular design can make `rcond` fail to compute; treat that as the
        // most ill-conditioned case (reciprocal condition 0), i.e. rank deficient.
        let reciprocal_condition = factorization.rcond().unwrap_or(0.0);
        if reciprocal_condition < config.minimum_reciprocal_condition
        {
            return Err(InfluenceError::RankDeficient {
                reciprocal_condition,
                floor: config.minimum_reciprocal_condition,
            });
        }

        let coefficients = solve_qr_least_squares(&factorization, targets).map_err(|error| {
            InfluenceError::Factorization {
                detail: error.to_string(),
            }
        })?;
        let q = factorization.q();
        let r = factorization.r();

        // Residuals and the hat diagonal.
        let mut residuals = vec![0.0_f64; n];
        let mut leverage = vec![0.0_f64; n];
        let mut residual_sum_of_squares = 0.0;
        for i in 0..n
        {
            let mut prediction = 0.0;
            for j in 0..p
            {
                prediction += features[(i, j)] * coefficients[j];
            }
            if config.fit_intercept
            {
                prediction += coefficients[p];
            }
            residuals[i] = targets[i] - prediction;
            residual_sum_of_squares += residuals[i] * residuals[i];

            let mut hat = 0.0;
            for c in 0..fitted_columns
            {
                let entry = q[(i, c)];
                hat += entry * entry;
            }
            leverage[i] = hat.clamp(0.0, 1.0);
        }

        let degrees_of_freedom = n - fitted_columns;
        let variance = residual_sum_of_squares / degrees_of_freedom as f64;
        let residual_scale = variance.sqrt();

        // Robust feature-space distances (best effort — never fatal).
        let (robust_distances, robust_warning) = robust_distances(features, config.ogk_reweight);
        let mut warnings = Vec::new();
        if let Some(message) = robust_warning
        {
            warnings.push(message);
        }

        let leverage_threshold = config.leverage_multiple * fitted_columns as f64 / n as f64;
        let robust_distance_threshold = if robust_distances.is_some()
        {
            Some(
                ChiSquared::new(p as f64)
                    .quantile(config.robust_distance_quantile)
                    .sqrt(),
            )
        }
        else
        {
            None
        };

        let mut records = Vec::with_capacity(n);
        for i in 0..n
        {
            let hat = leverage[i];
            let one_minus_hat = 1.0 - hat;
            let residual = residuals[i];

            let (studentized_residual, cook_distance) = if one_minus_hat > 1.0e-12 && variance > 0.0
            {
                // Internally studentized residual rᵢ / (σ̂ √(1 − hᵢ)). More stable
                // than the external (leave-one-out) form, whose scale can collapse
                // for a single very influential point; documented as internal.
                let studentized = residual / (residual_scale * one_minus_hat.sqrt());
                let cook = (residual * residual / (fitted_columns as f64 * variance))
                    * (hat / (one_minus_hat * one_minus_hat));
                (Some(studentized), Some(cook))
            }
            else
            {
                (None, None)
            };

            let coefficient_displacement = coefficient_displacement(
                &r,
                features,
                i,
                config.fit_intercept,
                residual,
                one_minus_hat,
            );

            let robust_distance = robust_distances.as_ref().map(|distances| distances[i]);
            let class = classify(
                hat,
                leverage_threshold,
                robust_distance,
                robust_distance_threshold,
                studentized_residual,
                config.residual_threshold,
            );

            records.push(InfluenceRecord {
                row_index: i,
                leverage: hat,
                robust_distance,
                residual,
                studentized_residual,
                cook_distance,
                coefficient_displacement,
                class,
            });
        }

        Ok(Self {
            records,
            residual_scale,
            fitted_columns,
            observation_count: n,
            leverage_threshold,
            residual_threshold: config.residual_threshold,
            robust_distance_threshold,
            warnings,
        })
    }

    /// The number of records with a given influence class.
    pub fn count(&self, class: ObservationInfluenceClass) -> usize {
        self.records
            .iter()
            .filter(|record| record.class == class)
            .count()
    }
}

fn validate_config(config: &InfluenceConfig) -> Result<(), InfluenceError> {
    if !(config.leverage_multiple.is_finite() && config.leverage_multiple > 0.0)
    {
        return Err(InfluenceError::InvalidConfig {
            detail: "leverage_multiple must be finite and positive".to_string(),
        });
    }
    if !(config.residual_threshold.is_finite() && config.residual_threshold > 0.0)
    {
        return Err(InfluenceError::InvalidConfig {
            detail: "residual_threshold must be finite and positive".to_string(),
        });
    }
    if !(config.robust_distance_quantile.is_finite()
        && config.robust_distance_quantile > 0.0
        && config.robust_distance_quantile < 1.0)
    {
        return Err(InfluenceError::InvalidConfig {
            detail: "robust_distance_quantile must be in (0, 1)".to_string(),
        });
    }
    if !(config.minimum_reciprocal_condition.is_finite()
        && config.minimum_reciprocal_condition >= 0.0)
    {
        return Err(InfluenceError::InvalidConfig {
            detail: "minimum_reciprocal_condition must be finite and non-negative".to_string(),
        });
    }
    Ok(())
}

/// `‖(XᵀX)⁻¹ xᵢ · rᵢ / (1 − hᵢ)‖₂`, computed through the QR's `R` factor with two
/// triangular solves — never an explicit inverse. Returns `None` under degeneracy.
fn coefficient_displacement(
    r: &Matrix,
    features: &Matrix,
    row: usize,
    fit_intercept: bool,
    residual: f64,
    one_minus_hat: f64,
) -> Option<f64> {
    if one_minus_hat <= 1.0e-12
    {
        return None;
    }
    let p = features.cols();
    let fitted_columns = p + usize::from(fit_intercept);

    // The design row xᵢ.
    let mut x = vec![0.0_f64; fitted_columns];
    for j in 0..p
    {
        x[j] = features[(row, j)];
    }
    if fit_intercept
    {
        x[p] = 1.0;
    }

    // Solve Rᵀ a = x (forward, Rᵀ lower triangular).
    let mut a = vec![0.0_f64; fitted_columns];
    for i in 0..fitted_columns
    {
        let mut sum = x[i];
        for j in 0..i
        {
            sum -= r[(j, i)] * a[j];
        }
        let diagonal = r[(i, i)];
        if diagonal.abs() <= 1.0e-300
        {
            return None;
        }
        a[i] = sum / diagonal;
    }
    // Solve R b = a (back, R upper triangular).
    let mut b = vec![0.0_f64; fitted_columns];
    for i in (0..fitted_columns).rev()
    {
        let mut sum = a[i];
        for j in (i + 1)..fitted_columns
        {
            sum -= r[(i, j)] * b[j];
        }
        let diagonal = r[(i, i)];
        if diagonal.abs() <= 1.0e-300
        {
            return None;
        }
        b[i] = sum / diagonal;
    }

    let factor = residual / one_minus_hat;
    let norm_squared: f64 = b.iter().map(|value| (value * factor).powi(2)).sum();
    Some(norm_squared.sqrt())
}

/// Robust OGK Mahalanobis distances in feature space (best effort). Returns
/// `(None, Some(message))` when the robust scatter could not be fitted.
fn robust_distances(features: &Matrix, reweight: bool) -> (Option<Vec<f64>>, Option<String>) {
    let n = features.rows();
    let p = features.cols();
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..p).map(|j| features[(i, j)]).collect())
        .collect();
    let geometry = GeometryMatrix {
        rows: n,
        cols: p,
        data,
    };

    let config = RobustScatterConfig {
        method: RobustScatterMethod::Ogk {
            scale: RobustUnivariateScale::MedianAbsoluteDeviation,
            reweight,
        },
        ridge: 0.0,
        ..RobustScatterConfig::default()
    };

    match RobustScatterModel::fit(&geometry, config)
    {
        Ok(model) =>
        {
            let mut distances = vec![0.0_f64; n];
            for (i, row) in geometry.data.iter().enumerate()
            {
                match model.mahalanobis(row)
                {
                    Ok(distance) => distances[i] = distance,
                    Err(error) =>
                    {
                        return (None, Some(format!("robust distance unavailable: {error}")));
                    },
                }
            }
            (Some(distances), None)
        },
        Err(RobustGeometryError::SingularScatter { .. }) => (
            None,
            Some(
                "robust feature-space distance unavailable: OGK scatter is singular (features not \
full rank); classification falls back to hat leverage"
                    .to_string(),
            ),
        ),
        Err(error) => (
            None,
            Some(format!(
                "robust feature-space distance unavailable: {error}"
            )),
        ),
    }
}

fn classify(
    leverage: f64,
    leverage_threshold: f64,
    robust_distance: Option<f64>,
    robust_distance_threshold: Option<f64>,
    studentized_residual: Option<f64>,
    residual_threshold: f64,
) -> ObservationInfluenceClass {
    let hat_high = leverage > leverage_threshold;
    let robust_high = match (robust_distance, robust_distance_threshold)
    {
        (Some(distance), Some(threshold)) => distance > threshold,
        // No robust distance: fall back to the hat verdict (no disagreement).
        _ => hat_high,
    };
    let residual_high = studentized_residual.is_some_and(|value| value.abs() > residual_threshold);

    // Disagreement between the two leverage notions is itself informative.
    if robust_distance.is_some() && robust_distance_threshold.is_some() && hat_high != robust_high
    {
        return ObservationInfluenceClass::Ambiguous;
    }

    let leverage_high = hat_high || robust_high;
    match (leverage_high, residual_high)
    {
        (true, true) => ObservationInfluenceClass::BadLeverage,
        (true, false) => ObservationInfluenceClass::GoodLeverage,
        (false, true) => ObservationInfluenceClass::VerticalOutlier,
        (false, false) => ObservationInfluenceClass::Regular,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(columns: &[Vec<f64>]) -> Matrix {
        let n = columns[0].len();
        let p = columns.len();
        let mut m = Matrix::zeros(n, p);
        for (j, col) in columns.iter().enumerate()
        {
            for (i, &v) in col.iter().enumerate()
            {
                m[(i, j)] = v;
            }
        }
        m
    }

    /// Clean 1-feature line y = 2x + bounded wiggle over x = 0..n.
    fn clean_line(n: usize) -> (Matrix, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..n)
            .map(|i| 2.0 * i as f64 + (((i * 7) % 11) as f64 - 5.0) * 0.2)
            .collect();
        (design(&[xs]), ys)
    }

    #[test]
    fn hat_leverage_trace_equals_fitted_columns() {
        let (features, targets) = clean_line(40);
        let report = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let trace: f64 = report.records.iter().map(|r| r.leverage).sum();
        // trace(H) = p (features + intercept) exactly for a full-rank design.
        assert!(
            (trace - report.fitted_columns as f64).abs() < 1e-9,
            "trace {trace}"
        );
        assert_eq!(report.fitted_columns, 2);
        for record in &report.records
        {
            assert!(record.leverage >= 0.0 && record.leverage <= 1.0);
        }
    }

    #[test]
    fn clean_data_is_mostly_regular() {
        let (features, targets) = clean_line(50);
        let report = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let regular = report.count(ObservationInfluenceClass::Regular);
        assert!(regular >= 45, "expected mostly regular, got {regular}/50");
    }

    #[test]
    fn vertical_outlier_is_detected() {
        let (features, mut targets) = clean_line(40);
        // A big vertical shift at a mid-range x (ordinary leverage, huge residual).
        targets[20] += 60.0;
        let report = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let record = &report.records[20];
        assert_eq!(record.class, ObservationInfluenceClass::VerticalOutlier);
        assert!(record.studentized_residual.unwrap().abs() > 2.5);
        assert!(record.leverage < report.leverage_threshold);
    }

    #[test]
    fn good_leverage_is_detected() {
        // A clean line plus a far cluster ON the line (high leverage, tiny residual).
        let mut xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        for k in 0..5
        {
            let x = 150.0 + k as f64;
            xs.push(x);
            ys.push(2.0 * x); // exactly on the line
        }
        let report = InfluenceReport::fit(&design(&[xs]), &ys, InfluenceConfig::default()).unwrap();
        for i in 40..45
        {
            let record = &report.records[i];
            assert!(
                record.leverage > report.leverage_threshold,
                "row {i} leverage"
            );
            assert_eq!(
                record.class,
                ObservationInfluenceClass::GoodLeverage,
                "row {i}"
            );
        }
    }

    #[test]
    fn bad_leverage_is_detected() {
        // A clean line plus ONE moderate-leverage point well off the line. A single
        // outlier can't drag the fit to itself (no self-masking), so it keeps a
        // large residual AND elevated leverage → BadLeverage.
        let mut xs: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let mut ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        xs.push(60.0);
        ys.push(2.0 * 60.0 - 100.0); // off the line by −100
        let report = InfluenceReport::fit(&design(&[xs]), &ys, InfluenceConfig::default()).unwrap();
        let record = &report.records[40];
        assert!(
            record.leverage > report.leverage_threshold,
            "leverage {}",
            record.leverage
        );
        assert!(
            record.studentized_residual.unwrap().abs() > 2.5,
            "studentized"
        );
        assert!(record.robust_distance.unwrap() > 2.0, "robust distance");
        assert_eq!(record.class, ObservationInfluenceClass::BadLeverage);
    }

    #[test]
    fn rank_deficient_design_is_a_typed_error() {
        // Two identical feature columns → rank deficient.
        let xs: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let features = design(&[xs.clone(), xs]);
        let targets: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let error =
            InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap_err();
        assert!(matches!(error, InfluenceError::RankDeficient { .. }));
    }

    #[test]
    fn cook_distance_flags_the_influential_point() {
        let (features, mut targets) = clean_line(40);
        targets[39] += 40.0; // last point, higher leverage + residual
        let report = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let influential = report.records[39].cook_distance.unwrap();
        let typical: f64 = (0..39)
            .map(|i| report.records[i].cook_distance.unwrap())
            .sum::<f64>()
            / 39.0;
        assert!(
            influential > 10.0 * typical,
            "cook {influential} vs typical {typical}"
        );
    }

    #[test]
    fn diagnostics_are_deterministic() {
        let (features, targets) = clean_line(45);
        let a = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let b = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let (features, targets) = clean_line(40);
        assert!(matches!(
            InfluenceReport::fit(&features, &targets[..39], InfluenceConfig::default())
                .unwrap_err(),
            InfluenceError::TargetLengthMismatch { .. }
        ));
        let tiny = design(&[vec![1.0, 2.0, 3.0]]);
        assert!(matches!(
            InfluenceReport::fit(&tiny, &[1.0, 2.0, 3.0], InfluenceConfig::default()).unwrap_err(),
            InfluenceError::TooFewObservations { .. }
        ));
        assert!(matches!(
            InfluenceReport::fit(
                &features,
                &targets,
                InfluenceConfig {
                    residual_threshold: -1.0,
                    ..InfluenceConfig::default()
                }
            )
            .unwrap_err(),
            InfluenceError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn intercept_toggle_changes_fitted_columns() {
        let (features, targets) = clean_line(40);
        let with = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let without = InfluenceReport::fit(
            &features,
            &targets,
            InfluenceConfig {
                fit_intercept: false,
                ..InfluenceConfig::default()
            },
        )
        .unwrap();
        assert_eq!(with.fitted_columns, 2);
        assert_eq!(without.fitted_columns, 1);
    }
}
