use crate::error::CausalError;
use scirust_solvers::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    Converged,
    MaxOuterIterations,
    MaxInnerIterations,
    NonFiniteObjective,
    NonFiniteGradient,
    LineSearchFailure,
    PenaltyLimitReached,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalOptimizationResult {
    pub interactions: Matrix,
    pub objective: f64,
    pub acyclicity: f64,
    pub gradient_norm: f64,
    pub outer_iterations: usize,
    pub inner_iterations: usize,
    pub termination: TerminationReason,
}

pub struct OptimizerConfig {
    pub inner_max_iter: usize,
    pub outer_max_iter: usize,
    pub gradient_tol: f64,
    pub objective_tol: f64,
}

impl OptimizerConfig {
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

fn bfgs_optimize<F>(
    fg: F,
    x0: Vec<f64>,
    max_iter: usize,
    grad_tol: f64,
) -> Result<(Vec<f64>, f64, f64, usize), CausalError>
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
            return Ok((x, fx, gn, iter));
        }

        let dir = bfgs_direction(&hessian_inv, &grad);

        match line_search_armijo(&fg, &x, fx, &grad, &dir)
        {
            Some((x_new, fx_new, grad_new)) =>
            {
                let s: Vec<f64> = x_new.iter().zip(&x).map(|(xn, xo)| xn - xo).collect();
                let y: Vec<f64> = grad_new.iter().zip(&grad).map(|(gn, go)| gn - go).collect();
                bfgs_update(&mut hessian_inv, &s, &y);

                x = x_new;
                fx = fx_new;
                grad = grad_new;
            },
            None =>
            {
                let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                return Ok((x, fx, gn, iter));
            },
        }

        if !fx.is_finite()
        {
            return Err(CausalError::NonFiniteComputation {
                operation: "bfgs_objective",
                index: 0,
                value: fx,
            });
        }
    }

    let gn: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    Ok((x, fx, gn, max_iter))
}

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

        let result = bfgs_optimize(fg, a_flat, config.inner_max_iter, config.gradient_tol)?;
        let (x_opt, obj_val, grad_norm, inner_iters) = result;
        total_inner += inner_iters;
        final_objective = obj_val;
        final_grad_norm = grad_norm;

        a = Matrix::from_row_major(dim, dim, x_opt);

        if !obj_val.is_finite()
        {
            return Ok(CausalOptimizationResult {
                interactions: a,
                objective: final_objective,
                acyclicity: 0.0,
                gradient_norm: final_grad_norm,
                outer_iterations: outer_iter + 1,
                inner_iterations: total_inner,
                termination: TerminationReason::NonFiniteObjective,
            });
        }

        final_acyclicity = crate::acyclicity::PolynomialAcyclicity::value(&a)?;

        if final_acyclicity < config.gradient_tol && grad_norm < config.gradient_tol
        {
            return Ok(CausalOptimizationResult {
                interactions: a,
                objective: final_objective,
                acyclicity: final_acyclicity,
                gradient_norm: final_grad_norm,
                outer_iterations: outer_iter + 1,
                inner_iterations: total_inner,
                termination: TerminationReason::Converged,
            });
        }

        alpha += rho * final_acyclicity;
        rho = (rho * 2.0).min(1e8);
    }

    final_acyclicity = crate::acyclicity::PolynomialAcyclicity::value(&a)?;

    Ok(CausalOptimizationResult {
        interactions: a,
        objective: final_objective,
        acyclicity: final_acyclicity,
        gradient_norm: final_grad_norm,
        outer_iterations: config.outer_max_iter,
        inner_iterations: total_inner,
        termination: TerminationReason::MaxOuterIterations,
    })
}
