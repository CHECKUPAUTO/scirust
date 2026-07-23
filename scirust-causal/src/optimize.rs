//! Deterministic augmented-Lagrangian optimizer for the cubic causal score.
//!
//! # Honesty of the termination contract
//!
//! This optimizer minimizes a smooth score plus an acyclicity penalty. It is a
//! first-order (BFGS) method, so it can only certify *stationarity*, never global
//! optimality, and the returned [`CausalOptimizationResult::termination`] must be
//! consulted before the result is trusted:
//!
//! - [`TerminationReason::Converged`] — the acyclicity penalty is feasible and the
//!   gradient (or objective change) is below tolerance **after at least one
//!   descent step**.
//! - [`TerminationReason::StationaryAtInitialPoint`] — the optimizer performed
//!   **zero** descent steps because the initial point was already first-order
//!   stationary. The result equals the initial guess and is **not** a certified
//!   minimum. In particular the all-zeros interaction matrix is a degenerate
//!   *saddle* of the cubic score (its gradient vanishes there for any data), so
//!   initializing at zero returns the empty graph unchanged. Initialize away from
//!   exact zero to actually optimize.
//! - [`TerminationReason::MaxOuterIterations`] / [`TerminationReason::MaxInnerIterations`]
//!   / [`TerminationReason::LineSearchFailure`] / [`TerminationReason::PenaltyLimitReached`]
//!   — the run did **not** converge; the interaction matrix has no optimality
//!   guarantee and should not be thresholded into a graph without review.
//!
//! Non-convergence is a first-class *result*, not an error: the numerics
//! completed, but the scientific conclusion is "no minimizer certified".

use crate::error::CausalError;
use scirust_solvers::Matrix;

/// Why the optimizer stopped. Every variant is reachable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// Feasible acyclicity and a small gradient/objective change after ≥1 step.
    Converged,
    /// Zero descent steps were taken: the initial point was already stationary.
    /// The result equals the initial guess and is not a certified minimum.
    StationaryAtInitialPoint,
    /// The outer augmented-Lagrangian loop exhausted without reaching feasibility.
    MaxOuterIterations,
    /// The inner BFGS loop hit its iteration cap on the final outer step.
    MaxInnerIterations,
    /// The inner line search could not find a descent step on the final outer step.
    LineSearchFailure,
    /// The acyclicity penalty reached its cap without achieving feasibility.
    PenaltyLimitReached,
}

/// The reproducible outcome of an optimization run.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalOptimizationResult {
    /// The (row-major) interaction matrix at termination.
    pub interactions: Matrix,
    /// Final objective value.
    pub objective: f64,
    /// Final acyclicity residual `h(A)` (0 ⇔ acyclic).
    pub acyclicity: f64,
    /// Final gradient norm.
    pub gradient_norm: f64,
    /// Outer (augmented-Lagrangian) iterations performed.
    pub outer_iterations: usize,
    /// Total inner (BFGS) iterations across all outer steps.
    pub inner_iterations: usize,
    /// Why the run stopped — **consult this before trusting the result**.
    pub termination: TerminationReason,
    /// The final Lagrange multiplier `α` reached (penalty-state provenance).
    pub final_alpha: f64,
    /// The final penalty coefficient `ρ` reached (penalty-state provenance).
    pub final_rho: f64,
    /// Non-fatal notes. The first entry echoes the run configuration; later
    /// entries record stationarity / non-convergence findings.
    pub warnings: Vec<String>,
}

/// Optimizer configuration. All fields are validated at construction.
pub struct OptimizerConfig {
    /// Maximum inner (BFGS) iterations per outer step.
    pub inner_max_iter: usize,
    /// Maximum outer (augmented-Lagrangian) iterations.
    pub outer_max_iter: usize,
    /// Gradient-norm convergence tolerance (also the acyclicity feasibility bar).
    pub gradient_tol: f64,
    /// Relative objective-change convergence tolerance for the inner loop.
    pub objective_tol: f64,
}

impl OptimizerConfig {
    /// Validates and constructs a configuration.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidConfiguration`] when an iteration cap is zero or a
    /// tolerance is non-finite or non-positive.
    pub fn new(
        inner_max_iter: usize,
        outer_max_iter: usize,
        gradient_tol: f64,
        objective_tol: f64,
    ) -> Result<Self, CausalError> {
        if inner_max_iter == 0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "inner_max_iter",
                value: inner_max_iter as f64,
            });
        }
        if outer_max_iter == 0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "outer_max_iter",
                value: outer_max_iter as f64,
            });
        }
        if !gradient_tol.is_finite() || gradient_tol <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "gradient_tol",
                value: gradient_tol,
            });
        }
        if !objective_tol.is_finite() || objective_tol <= 0.0
        {
            return Err(CausalError::InvalidConfiguration {
                name: "objective_tol",
                value: objective_tol,
            });
        }
        Ok(Self {
            inner_max_iter,
            outer_max_iter,
            gradient_tol,
            objective_tol,
        })
    }
}

/// The largest penalty coefficient the outer loop will grow `ρ` to.
const PENALTY_CAP: f64 = 1e8;

/// Why the inner BFGS loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InnerStop {
    /// Gradient norm or relative objective change fell below tolerance.
    Converged,
    /// The inner iteration cap was reached.
    MaxIterations,
    /// The line search could not find a descent step.
    LineSearchFailed,
}

fn bfgs_direction(h: &Matrix, g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut p = vec![0.0; n];
    for i in 0..n
    {
        for j in 0..n
        {
            p[i] -= h[(i, j)] * g[j];
        }
    }
    p
}

fn line_search_armijo<F>(
    fg: &F,
    x: &[f64],
    fx: f64,
    grad: &[f64],
    direction: &[f64],
) -> Option<(Vec<f64>, f64, Vec<f64>)>
where
    F: Fn(&[f64]) -> Result<(f64, Vec<f64>), CausalError>,
{
    let armijo = 1e-4;
    let max_iters = 40;
    let mut alpha = 1.0;
    let dir_grad: f64 = grad.iter().zip(direction).map(|(g, d)| g * d).sum();

    if dir_grad >= 0.0
    {
        return None;
    }

    for _ in 0..max_iters
    {
        let trial: Vec<f64> = x
            .iter()
            .zip(direction)
            .map(|(x_i, d_i)| x_i + alpha * d_i)
            .collect();

        if let Ok((ftrial, gtrial)) = fg(&trial)
        {
            if ftrial <= fx + armijo * alpha * dir_grad
            {
                return Some((trial, ftrial, gtrial));
            }
        }
        alpha *= 0.5;
    }
    None
}

fn bfgs_update(h: &mut Matrix, s: &[f64], y: &[f64]) {
    let n = s.len();
    let sy: f64 = s.iter().zip(y).map(|(s_i, y_i)| s_i * y_i).sum();

    if sy <= 0.0
    {
        return;
    }

    let rho = 1.0 / sy;
    let mut hy = vec![0.0; n];
    for i in 0..n
    {
        for j in 0..n
        {
            hy[i] += h[(i, j)] * y[j];
        }
    }

    let yhy: f64 = y.iter().zip(&hy).map(|(y_i, hy_i)| y_i * hy_i).sum();

    for i in 0..n
    {
        for j in 0..n
        {
            let update = rho * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j])
                + yhy * rho * rho * s[i] * s[j];
            h[(i, j)] += update;
        }
    }
}

/// Runs inner BFGS. Returns `(x, fx, gradient_norm, iterations, why_stopped)`.
///
/// `iterations == 0` with [`InnerStop::Converged`] means the initial point was
/// already stationary — the caller distinguishes this degenerate case.
fn bfgs_optimize<F>(
    fg: F,
    x0: Vec<f64>,
    max_iter: usize,
    grad_tol: f64,
    objective_tol: f64,
) -> Result<(Vec<f64>, f64, f64, usize, InnerStop), CausalError>
where
    F: Fn(&[f64]) -> Result<(f64, Vec<f64>), CausalError>,
{
    let n = x0.len();
    let mut x = x0;
    let (mut fx, mut grad) = fg(&x)?;
    let mut hessian_inv = Matrix::identity(n);

    for iter in 0..max_iter
    {
        let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        if gn < grad_tol
        {
            return Ok((x, fx, gn, iter, InnerStop::Converged));
        }

        let dir = bfgs_direction(&hessian_inv, &grad);

        match line_search_armijo(&fg, &x, fx, &grad, &dir)
        {
            Some((x_new, fx_new, grad_new)) =>
            {
                let objective_change = (fx_new - fx).abs();
                let s: Vec<f64> = x_new.iter().zip(&x).map(|(xn, xo)| xn - xo).collect();
                let y: Vec<f64> = grad_new.iter().zip(&grad).map(|(gn, go)| gn - go).collect();
                bfgs_update(&mut hessian_inv, &s, &y);

                x = x_new;
                fx = fx_new;
                grad = grad_new;

                // Objective-change convergence: a genuine step whose relative
                // objective improvement is below tolerance.
                if objective_change <= objective_tol * (1.0 + fx.abs())
                {
                    let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                    return Ok((x, fx, gn, iter + 1, InnerStop::Converged));
                }
            },
            None =>
            {
                let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                return Ok((x, fx, gn, iter, InnerStop::LineSearchFailed));
            },
        }
    }

    let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    Ok((x, fx, gn, max_iter, InnerStop::MaxIterations))
}

/// Minimizes the cubic causal score with an augmented-Lagrangian acyclicity
/// penalty.
///
/// Non-convergence and stationarity are reported through
/// [`CausalOptimizationResult::termination`], never as an `Err` — the numerics
/// completing without certifying a minimizer is a scientific result, not a
/// failure. See the module docs for the meaning of each termination reason; in
/// particular an all-zeros `initial_interactions` returns
/// [`TerminationReason::StationaryAtInitialPoint`] (the empty graph unchanged).
///
/// # Errors
///
/// [`CausalError`] only for genuinely invalid numerics (non-finite objective
/// during evaluation, invalid sub-configuration, dimension issues).
#[allow(clippy::too_many_arguments)]
pub fn optimize_causal(
    samples: &Matrix,
    initial_interactions: &Matrix,
    lambda_l1: f64,
    smooth_l1_epsilon: f64,
    alpha_init: f64,
    rho_init: f64,
    config: &OptimizerConfig,
) -> Result<CausalOptimizationResult, CausalError> {
    let dim = initial_interactions.rows();
    let mut a = initial_interactions.clone();
    let mut alpha = alpha_init;
    let mut rho = rho_init;
    let mut total_inner = 0;
    let mut final_acyclicity;
    let mut final_grad_norm = 0.0;
    let mut final_objective = 0.0;
    let mut last_inner_stop = InnerStop::Converged;
    let mut penalty_capped = false;

    let mut warnings = vec![format!(
        "config: lambda_l1={lambda_l1:.6e} smooth_l1_epsilon={smooth_l1_epsilon:.6e} \
         alpha_init={alpha_init:.6e} rho_init={rho_init:.6e} gradient_tol={:.6e} \
         objective_tol={:.6e} inner_max_iter={} outer_max_iter={}",
        config.gradient_tol, config.objective_tol, config.inner_max_iter, config.outer_max_iter
    )];

    for outer_iter in 0..config.outer_max_iter
    {
        let al_config = crate::objective::AugmentedLagrangianConfig::new(
            lambda_l1,
            alpha,
            rho,
            smooth_l1_epsilon,
        )
        .map_err(|_| CausalError::InvalidConfiguration {
            name: "augmented_lagrangian",
            value: rho,
        })?;

        let a_flat = a.data().to_vec();

        let fg = |x: &[f64]| -> Result<(f64, Vec<f64>), CausalError> {
            let mat = Matrix::from_row_major(dim, dim, x.to_vec());
            let eval = crate::objective::CausalObjective::evaluate(samples, &mat, &al_config)?;
            let g_flat = eval.gradient.data().to_vec();
            Ok((eval.total, g_flat))
        };

        let (x_opt, obj_val, grad_norm, inner_iters, inner_stop) = bfgs_optimize(
            fg,
            a_flat,
            config.inner_max_iter,
            config.gradient_tol,
            config.objective_tol,
        )?;
        total_inner += inner_iters;
        final_objective = obj_val;
        final_grad_norm = grad_norm;
        last_inner_stop = inner_stop;

        a = Matrix::from_row_major(dim, dim, x_opt);
        final_acyclicity = crate::acyclicity::PolynomialAcyclicity::value(&a)?;

        // Feasible + first-order stationary ⇒ converged (unless nothing moved).
        if final_acyclicity < config.gradient_tol && grad_norm < config.gradient_tol
        {
            let termination = if total_inner == 0
            {
                warnings.push(
                    "optimizer took zero descent steps: the initial point is stationary \
                     (for an all-zeros start this is the empty-graph saddle) — result is the \
                     initial guess, not a certified minimum"
                        .to_string(),
                );
                TerminationReason::StationaryAtInitialPoint
            }
            else
            {
                TerminationReason::Converged
            };

            return Ok(CausalOptimizationResult {
                interactions: a,
                objective: final_objective,
                acyclicity: final_acyclicity,
                gradient_norm: final_grad_norm,
                outer_iterations: outer_iter + 1,
                inner_iterations: total_inner,
                termination,
                final_alpha: alpha,
                final_rho: rho,
                warnings,
            });
        }

        alpha += rho * final_acyclicity;
        let next_rho = (rho * 2.0).min(PENALTY_CAP);
        if next_rho >= PENALTY_CAP
        {
            penalty_capped = true;
        }
        rho = next_rho;
    }

    final_acyclicity = crate::acyclicity::PolynomialAcyclicity::value(&a)?;

    // The outer loop finished without certifying convergence: report the most
    // specific non-convergence reason available.
    let termination = if penalty_capped
    {
        TerminationReason::PenaltyLimitReached
    }
    else
    {
        match last_inner_stop
        {
            InnerStop::LineSearchFailed => TerminationReason::LineSearchFailure,
            InnerStop::MaxIterations => TerminationReason::MaxInnerIterations,
            InnerStop::Converged => TerminationReason::MaxOuterIterations,
        }
    };
    warnings.push(format!(
        "did not certify convergence: {termination:?} (final acyclicity {final_acyclicity:.6e}, \
         gradient norm {final_grad_norm:.6e}) — do not threshold this matrix into a graph without \
         review"
    ));

    Ok(CausalOptimizationResult {
        interactions: a,
        objective: final_objective,
        acyclicity: final_acyclicity,
        gradient_norm: final_grad_norm,
        outer_iterations: config.outer_max_iter,
        inner_iterations: total_inner,
        termination,
        final_alpha: alpha,
        final_rho: rho,
        warnings,
    })
}
