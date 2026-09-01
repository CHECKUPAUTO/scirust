//! Cox proportional-hazards regression for right-censored observations.
//!
//! The implementation maximizes Cox's partial log-likelihood with a
//! deterministic Newton-Raphson iteration. Event-time ties can be handled with
//! either the Breslow or Efron approximation. Covariates are internally centered
//! before likelihood evaluation so the fit is insensitive to arbitrary additive
//! origins and the observed information avoids catastrophic cancellation.

use crate::survival::RightCensoredObservation;
use core::fmt;

/// Treatment used for tied event times in the Cox partial likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoxTieMethod {
    /// Breslow's approximation: every tied event uses the full risk-set denominator.
    Breslow,
    /// Efron's approximation: progressively removes fractions of tied-event risk mass.
    Efron,
}

/// One right-censored observation together with its covariate vector.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxObservation {
    survival: RightCensoredObservation,
    covariates: Vec<f64>,
}

impl CoxObservation {
    /// Construct a Cox regression observation.
    ///
    /// At least one covariate is required, and every covariate must be finite.
    pub fn new(survival: RightCensoredObservation, covariates: Vec<f64>) -> Result<Self, CoxError> {
        if covariates.is_empty()
        {
            return Err(CoxError::EmptyCovariates);
        }
        for (index, &value) in covariates.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(CoxError::NonFiniteCovariate { index, value });
            }
        }
        Ok(Self {
            survival,
            covariates,
        })
    }

    /// Right-censored time/event observation.
    #[must_use]
    pub fn survival(&self) -> RightCensoredObservation {
        self.survival
    }

    /// Covariates in model-column order.
    #[must_use]
    pub fn covariates(&self) -> &[f64] {
        &self.covariates
    }
}

/// Numerical options for Cox proportional-hazards fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoxFitOptions {
    /// Tie treatment used in the partial likelihood.
    pub tie_method: CoxTieMethod,
    /// Maximum number of Newton updates.
    pub max_iterations: usize,
    /// Convergence threshold applied to both the Newton step and score.
    pub tolerance: f64,
    /// Relative pivot threshold used when solving/inverting observed information.
    pub singularity_tolerance: f64,
    /// Maximum deterministic step halvings used to obtain non-decreasing likelihood.
    pub max_step_halvings: usize,
}

impl Default for CoxFitOptions {
    fn default() -> Self {
        Self {
            tie_method: CoxTieMethod::Efron,
            max_iterations: 64,
            tolerance: 1.0e-10,
            singularity_tolerance: 1.0e-12,
            max_step_halvings: 20,
        }
    }
}

/// Result of a Cox proportional-hazards fit.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxFitResult {
    /// Estimated log-hazard coefficients in covariate order.
    pub coefficients: Vec<f64>,
    /// Standard errors from inverse observed information.
    pub standard_errors: Vec<f64>,
    /// Row-major variance-covariance matrix.
    pub variance_covariance: Vec<f64>,
    /// Maximized partial log-likelihood at returned coefficients.
    pub log_partial_likelihood: f64,
    /// Number of Newton updates accepted.
    pub iterations: usize,
    /// Whether the configured convergence tolerance was reached.
    pub converged: bool,
    /// Tie treatment used for this fit.
    pub tie_method: CoxTieMethod,
}

/// Failure returned by Cox proportional-hazards fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoxError {
    /// At least one regression observation is required.
    EmptySample,
    /// A Cox model requires at least one covariate.
    EmptyCovariates,
    /// All observations were censored.
    NoEvents,
    /// A row has a different covariate dimension from the first row.
    CovariateDimensionMismatch {
        /// Zero-based row index.
        row: usize,
        /// Expected covariate count.
        expected: usize,
        /// Actual covariate count.
        actual: usize,
    },
    /// A covariate is NaN or infinite.
    NonFiniteCovariate {
        /// Zero-based covariate index.
        index: usize,
        /// Invalid value.
        value: f64,
    },
    /// Numerical options are invalid.
    InvalidOptions,
    /// The observed information matrix is singular or rank deficient.
    SingularInformation,
    /// Newton line search could not find a finite non-decreasing step.
    LineSearchFailed,
    /// A diagonal variance is negative or non-finite.
    InvalidVariance {
        /// Covariate index.
        index: usize,
        /// Invalid variance estimate.
        value: f64,
    },
}

impl fmt::Display for CoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptySample => write!(f, "Cox regression sample must not be empty"),
            Self::EmptyCovariates => write!(f, "Cox regression requires at least one covariate"),
            Self::NoEvents => write!(f, "Cox regression requires at least one observed event"),
            Self::CovariateDimensionMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Cox covariate dimension mismatch at row {row}: expected {expected}, got {actual}"
            ),
            Self::NonFiniteCovariate { index, value } =>
            {
                write!(f, "Cox covariate {index} must be finite, got {value}")
            },
            Self::InvalidOptions => write!(f, "invalid Cox fitting options"),
            Self::SingularInformation => write!(f, "Cox observed information matrix is singular"),
            Self::LineSearchFailed => write!(f, "Cox Newton line search failed"),
            Self::InvalidVariance { index, value } => write!(
                f,
                "Cox variance estimate for covariate {index} is invalid: {value}"
            ),
        }
    }
}

impl std::error::Error for CoxError {}

#[derive(Debug)]
struct PartialLikelihood {
    log_likelihood: f64,
    score: Vec<f64>,
    information: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}

fn validate_data(data: &[CoxObservation], options: CoxFitOptions) -> Result<usize, CoxError> {
    if data.is_empty()
    {
        return Err(CoxError::EmptySample);
    }
    if options.max_iterations == 0
        || options.max_step_halvings == 0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || !options.singularity_tolerance.is_finite()
        || options.singularity_tolerance <= 0.0
    {
        return Err(CoxError::InvalidOptions);
    }

    let dimension = data[0].covariates.len();
    if dimension == 0
    {
        return Err(CoxError::EmptyCovariates);
    }
    let mut any_event = false;
    for (row, observation) in data.iter().enumerate()
    {
        if observation.covariates.len() != dimension
        {
            return Err(CoxError::CovariateDimensionMismatch {
                row,
                expected: dimension,
                actual: observation.covariates.len(),
            });
        }
        any_event |= observation.survival.event_observed();
        for (index, &value) in observation.covariates.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(CoxError::NonFiniteCovariate { index, value });
            }
        }
    }
    if !any_event
    {
        return Err(CoxError::NoEvents);
    }
    Ok(dimension)
}

fn scaled_copy(data: &[CoxObservation], dimension: usize) -> (Vec<CoxObservation>, Vec<f64>) {
    let origin = &data[0].covariates;
    let centered: Vec<CoxObservation> = data
        .iter()
        .map(|row| CoxObservation {
            survival: row.survival,
            covariates: (0..dimension)
                .map(|j| row.covariates[j] - origin[j])
                .collect(),
        })
        .collect();

    let mut scales = vec![0.0; dimension];
    for row in &centered
    {
        for (scale, &value) in scales.iter_mut().zip(&row.covariates)
        {
            *scale = (*scale).max(value.abs());
        }
    }
    for scale in &mut scales
    {
        if *scale == 0.0
        {
            *scale = 1.0;
        }
    }

    let scaled = centered
        .into_iter()
        .map(|mut row| {
            for (value, &scale) in row.covariates.iter_mut().zip(&scales)
            {
                *value /= scale;
            }
            row
        })
        .collect();
    (scaled, scales)
}

fn partial_likelihood(
    data: &[CoxObservation],
    beta: &[f64],
    tie_method: CoxTieMethod,
) -> PartialLikelihood {
    let p = beta.len();
    let mut event_times: Vec<f64> = data
        .iter()
        .filter(|row| row.survival.event_observed())
        .map(|row| row.survival.time())
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();

    let mut log_likelihood = 0.0;
    let mut score = vec![0.0; p];
    let mut information = vec![0.0; p * p];

    for time in event_times
    {
        let mut max_eta = f64::NEG_INFINITY;
        for row in data.iter().filter(|row| row.survival.time() >= time)
        {
            max_eta = max_eta.max(dot(beta, &row.covariates));
        }

        let mut risk0 = 0.0;
        let mut risk1 = vec![0.0; p];
        let mut risk2 = vec![0.0; p * p];
        let mut tied0 = 0.0;
        let mut tied1 = vec![0.0; p];
        let mut tied2 = vec![0.0; p * p];
        let mut event_x = vec![0.0; p];
        let mut event_eta = 0.0;
        let mut event_count = 0usize;

        for row in data
        {
            if row.survival.time() < time
            {
                continue;
            }
            let is_event = row.survival.time() == time && row.survival.event_observed();
            let eta = dot(beta, &row.covariates);
            let weight = (eta - max_eta).exp();
            risk0 += weight;
            for j in 0..p
            {
                risk1[j] += weight * row.covariates[j];
                for k in 0..p
                {
                    risk2[j * p + k] += weight * row.covariates[j] * row.covariates[k];
                }
            }
            if is_event
            {
                event_count += 1;
                event_eta += eta;
                tied0 += weight;
                for j in 0..p
                {
                    event_x[j] += row.covariates[j];
                    tied1[j] += weight * row.covariates[j];
                    for k in 0..p
                    {
                        tied2[j * p + k] += weight * row.covariates[j] * row.covariates[k];
                    }
                }
            }
        }

        log_likelihood += event_eta;
        for j in 0..p
        {
            score[j] += event_x[j];
        }

        for tied_index in 0..event_count
        {
            let fraction = match tie_method
            {
                CoxTieMethod::Breslow => 0.0,
                CoxTieMethod::Efron => tied_index as f64 / event_count as f64,
            };
            let denominator = risk0 - fraction * tied0;
            log_likelihood -= max_eta + denominator.ln();

            let mut mean = vec![0.0; p];
            for j in 0..p
            {
                let first = risk1[j] - fraction * tied1[j];
                mean[j] = first / denominator;
                score[j] -= mean[j];
            }
            for j in 0..p
            {
                for k in 0..p
                {
                    let second = risk2[j * p + k] - fraction * tied2[j * p + k];
                    information[j * p + k] += second / denominator - mean[j] * mean[k];
                }
            }
        }
    }

    PartialLikelihood {
        log_likelihood,
        score,
        information,
    }
}

fn solve_linear_system(
    matrix: &[f64],
    rhs: &[f64],
    dimension: usize,
    singularity_tolerance: f64,
) -> Result<Vec<f64>, CoxError> {
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    let scale = a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if !scale.is_finite() || scale == 0.0
    {
        return Err(CoxError::SingularInformation);
    }
    let threshold = singularity_tolerance * scale;

    for column in 0..dimension
    {
        let mut pivot = column;
        let mut pivot_abs = a[column * dimension + column].abs();
        for row in (column + 1)..dimension
        {
            let candidate = a[row * dimension + column].abs();
            if candidate > pivot_abs
            {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= threshold
        {
            return Err(CoxError::SingularInformation);
        }
        if pivot != column
        {
            for k in 0..dimension
            {
                a.swap(column * dimension + k, pivot * dimension + k);
            }
            b.swap(column, pivot);
        }

        let diag = a[column * dimension + column];
        for row in (column + 1)..dimension
        {
            let factor = a[row * dimension + column] / diag;
            if factor == 0.0
            {
                continue;
            }
            a[row * dimension + column] = 0.0;
            for k in (column + 1)..dimension
            {
                a[row * dimension + k] -= factor * a[column * dimension + k];
            }
            b[row] -= factor * b[column];
        }
    }

    let mut x = vec![0.0; dimension];
    for row in (0..dimension).rev()
    {
        let mut value = b[row];
        for k in (row + 1)..dimension
        {
            value -= a[row * dimension + k] * x[k];
        }
        let diag = a[row * dimension + row];
        if !diag.is_finite() || diag.abs() <= threshold
        {
            return Err(CoxError::SingularInformation);
        }
        x[row] = value / diag;
    }
    Ok(x)
}

fn invert_matrix(
    matrix: &[f64],
    dimension: usize,
    singularity_tolerance: f64,
) -> Result<Vec<f64>, CoxError> {
    let mut inverse = vec![0.0; dimension * dimension];
    for column in 0..dimension
    {
        let mut unit = vec![0.0; dimension];
        unit[column] = 1.0;
        let solution = solve_linear_system(matrix, &unit, dimension, singularity_tolerance)?;
        for row in 0..dimension
        {
            inverse[row * dimension + column] = solution[row];
        }
    }
    Ok(inverse)
}

/// Fit a Cox proportional-hazards regression model.
///
/// Covariates are internally translated by a fixed column origin and scaled
/// by their largest centered magnitude before evaluating risk-set moments. The
/// Cox partial likelihood is invariant to translations and the coefficients,
/// standard errors, and covariance matrix are transformed back to the caller's
/// original covariate units. Centered and scaled moments avoid both large
/// common-offset cancellation and tiny-scale information matrices.
pub fn cox_proportional_hazards(
    data: &[CoxObservation],
    options: CoxFitOptions,
) -> Result<CoxFitResult, CoxError> {
    let dimension = validate_data(data, options)?;
    let (scaled, covariate_scales) = scaled_copy(data, dimension);
    let data = scaled.as_slice();
    let mut beta = vec![0.0; dimension];
    let mut current = partial_likelihood(data, &beta, options.tie_method);
    let mut iterations = 0usize;
    let mut converged = max_abs(&current.score) <= options.tolerance;

    while !converged && iterations < options.max_iterations
    {
        let direction = solve_linear_system(
            &current.information,
            &current.score,
            dimension,
            options.singularity_tolerance,
        )?;
        let mut accepted = None;
        let mut step_scale = 1.0;
        for _ in 0..options.max_step_halvings
        {
            let candidate_beta: Vec<f64> = beta
                .iter()
                .zip(&direction)
                .map(|(&b, &d)| b + step_scale * d)
                .collect();
            let candidate = partial_likelihood(data, &candidate_beta, options.tie_method);
            if candidate.log_likelihood.is_finite()
                && candidate.log_likelihood >= current.log_likelihood
            {
                accepted = Some((candidate_beta, candidate, step_scale));
                break;
            }
            step_scale *= 0.5;
        }
        let Some((candidate_beta, candidate, accepted_scale)) = accepted
        else
        {
            return Err(CoxError::LineSearchFailed);
        };

        iterations += 1;
        let step_size = accepted_scale * max_abs(&direction);
        beta = candidate_beta;
        current = candidate;
        converged = step_size <= options.tolerance || max_abs(&current.score) <= options.tolerance;
    }

    let scaled_covariance = invert_matrix(
        &current.information,
        dimension,
        options.singularity_tolerance,
    )?;
    let mut scaled_standard_errors = Vec::with_capacity(dimension);
    for index in 0..dimension
    {
        let variance = scaled_covariance[index * dimension + index];
        if !variance.is_finite() || variance < 0.0
        {
            return Err(CoxError::InvalidVariance {
                index,
                value: variance,
            });
        }
        scaled_standard_errors.push(variance.sqrt());
    }

    let inverse_scales: Vec<f64> = covariate_scales.iter().map(|scale| 1.0 / scale).collect();
    let coefficients = beta
        .iter()
        .zip(&inverse_scales)
        .map(|(&coefficient, &scale)| coefficient * scale)
        .collect();
    let standard_errors = scaled_standard_errors
        .iter()
        .zip(&inverse_scales)
        .map(|(&standard_error, &scale)| standard_error * scale)
        .collect();
    let variance_covariance = scaled_covariance
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let row = index / dimension;
            let column = index % dimension;
            value * inverse_scales[row] * inverse_scales[column]
        })
        .collect();

    Ok(CoxFitResult {
        coefficients,
        standard_errors,
        variance_covariance,
        log_partial_likelihood: current.log_likelihood,
        iterations,
        converged,
        tie_method: options.tie_method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(time: f64, is_event: bool, x: f64) -> CoxObservation {
        CoxObservation::new(
            RightCensoredObservation::new(time, is_event).unwrap(),
            vec![x],
        )
        .unwrap()
    }

    fn base_data(scale: f64, shift: f64) -> Vec<CoxObservation> {
        [
            (1.0, true, 1.0),
            (2.0, false, 0.0),
            (3.0, true, 0.0),
            (4.0, false, 1.0),
            (5.0, true, 1.0),
            (6.0, false, 0.0),
        ]
        .into_iter()
        .map(|(time, event, x)| row(time, event, shift + scale * x))
        .collect()
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }

    #[test]
    fn hand_solved_log_two_coefficient() {
        let fit = cox_proportional_hazards(&base_data(1.0, 0.0), CoxFitOptions::default()).unwrap();
        assert!(fit.converged);
        close(fit.coefficients[0], 2.0_f64.ln(), 1.0e-10);
        close(fit.standard_errors[0], 1.5_f64.sqrt(), 1.0e-10);
    }

    #[test]
    fn large_common_offset_preserves_fit() {
        let a = cox_proportional_hazards(&base_data(1.0, 0.0), CoxFitOptions::default()).unwrap();
        let b = cox_proportional_hazards(&base_data(1.0, 1.0e8), CoxFitOptions::default()).unwrap();
        close(a.coefficients[0], b.coefficients[0], 1.0e-10);
        close(a.standard_errors[0], b.standard_errors[0], 1.0e-10);
    }

    #[test]
    fn small_covariate_scale_remains_identifiable() {
        let base =
            cox_proportional_hazards(&base_data(1.0, 0.0), CoxFitOptions::default()).unwrap();
        let scaled =
            cox_proportional_hazards(&base_data(1.0e-7, 0.0), CoxFitOptions::default()).unwrap();
        assert!(scaled.converged);
        close(
            scaled.coefficients[0] * 1.0e-7,
            base.coefficients[0],
            1.0e-9,
        );
        close(
            scaled.standard_errors[0] * 1.0e-7,
            base.standard_errors[0],
            1.0e-9,
        );
    }

    #[test]
    fn tied_breslow_and_efron_match_closed_forms() {
        let data = [
            row(1.0, true, -1.0),
            row(1.0, true, 1.0),
            row(2.0, false, -1.0),
            row(2.0, false, 1.0),
        ];
        let breslow = cox_proportional_hazards(
            &data,
            CoxFitOptions {
                tie_method: CoxTieMethod::Breslow,
                ..CoxFitOptions::default()
            },
        )
        .unwrap();
        let efron = cox_proportional_hazards(&data, CoxFitOptions::default()).unwrap();
        close(breslow.coefficients[0], 0.0, 1.0e-15);
        close(efron.coefficients[0], 0.0, 1.0e-15);
        close(breslow.log_partial_likelihood, -2.0 * 4.0_f64.ln(), 2.0e-15);
        close(efron.log_partial_likelihood, -12.0_f64.ln(), 2.0e-15);
    }

    #[test]
    fn rejects_collinearity_and_bad_input() {
        assert_eq!(
            cox_proportional_hazards(&[], CoxFitOptions::default()),
            Err(CoxError::EmptySample)
        );
        let all_censored = [row(1.0, false, 0.0), row(2.0, false, 1.0)];
        assert_eq!(
            cox_proportional_hazards(&all_censored, CoxFitOptions::default()),
            Err(CoxError::NoEvents)
        );
        let data = [
            CoxObservation::new(
                RightCensoredObservation::new(1.0, true).unwrap(),
                vec![1.0, 2.0],
            )
            .unwrap(),
            CoxObservation::new(
                RightCensoredObservation::new(2.0, true).unwrap(),
                vec![0.0, 0.0],
            )
            .unwrap(),
            CoxObservation::new(
                RightCensoredObservation::new(3.0, false).unwrap(),
                vec![1.0, 2.0],
            )
            .unwrap(),
            CoxObservation::new(
                RightCensoredObservation::new(4.0, false).unwrap(),
                vec![0.0, 0.0],
            )
            .unwrap(),
        ];
        assert_eq!(
            cox_proportional_hazards(&data, CoxFitOptions::default()),
            Err(CoxError::SingularInformation)
        );
    }

    #[test]
    fn repeated_fit_is_numerically_reproducible() {
        let data = base_data(1.0, 0.0);
        let a = cox_proportional_hazards(&data, CoxFitOptions::default()).unwrap();
        let b = cox_proportional_hazards(&data, CoxFitOptions::default()).unwrap();
        close(a.coefficients[0], b.coefficients[0], 2.0e-9);
        close(a.standard_errors[0], b.standard_errors[0], 1.0e-9);
        close(a.log_partial_likelihood, b.log_partial_likelihood, 4.0e-15);
    }
}
