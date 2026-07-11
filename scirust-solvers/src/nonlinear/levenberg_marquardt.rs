//! Levenberg-Marquardt for nonlinear least squares: minimize `½‖r(x)‖²` for a
//! residual `r: R^n → R^m` (`m ≥ 1`, not necessarily `= n` — unlike
//! [`super::newton::newton_system`], which solves a square system exactly,
//! this handles the overdetermined curve-fitting case).
//!
//! Uses Marquardt's diagonally-scaled damping `(JᵀJ + λ·diag(JᵀJ))δ = -Jᵀr`
//! together with Nielsen's adaptive update of `λ` from the actual/predicted
//! reduction ratio — the standard, well-documented strategy (Levenberg 1944;
//! Marquardt 1963; Nielsen, *Damping Parameter in Marquardt's Method*, IMM
//! Tech. Report 1999; Madsen, Nielsen & Tingleff, *Methods for Non-Linear
//! Least Squares Problems*, 2004, §3.2 — the same reference algorithm behind
//! most production LM implementations, e.g. Ceres Solver's Levenberg-Marquardt
//! strategy).
//!
//! ## Numerical safety
//! - `check_finite` on the residual and Jacobian at every evaluation
//! - No artificial floor is added to `diag(JᵀJ)`: a parameter with a
//!   genuinely zero column in `J` is truly undetermined by the data, and the
//!   resulting singular normal-equations system surfaces as
//!   `SolverError::Singular` rather than being silently patched
//! - A step is only ever accepted when it actually reduces the cost
//!   (`ρ > 0`); rejected steps just grow `λ` and retry from the same point

use crate::linalg::{self, Matrix};
use crate::{Solution, SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use tracing::warn;

fn check_finite(value: f64, location: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        warn!(target: "solver", "non-finite value at {location}: {value}");
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Evaluate `r(x)` and its Jacobian `J` (m×n) together, one autodiff pass per
/// column — the same pattern `newton_system` uses, generalized to `m ≠ n`.
fn eval_residual_and_jacobian<F: Fn(&[Dual], &mut [Dual])>(
    f: &F,
    x: &[f64],
    buf_in: &mut [Dual],
    buf_out: &mut [Dual],
    r: &mut [f64],
    jac: &mut Matrix,
) -> SolverResult<()> {
    let n = x.len();
    let m = r.len();
    for j in 0..n
    {
        for i in 0..n
        {
            buf_in[i] = Dual::new(x[i], if i == j { 1.0 } else { 0.0 });
        }
        f(buf_in, buf_out);
        for i in 0..m
        {
            let d = buf_out[i].deriv;
            check_finite(d, &format!("J[{i},{j}]"))?;
            jac[(i, j)] = d;
            if j == 0
            {
                let v = buf_out[i].value;
                check_finite(v, &format!("r[{i}]"))?;
                r[i] = v;
            }
        }
    }
    Ok(())
}

/// Evaluate `r(x)` alone (primal only), for trial points during damping.
fn eval_residual<F: Fn(&[Dual], &mut [Dual])>(
    f: &F,
    x: &[f64],
    buf_in: &mut [Dual],
    buf_out: &mut [Dual],
    r: &mut [f64],
) -> SolverResult<()> {
    for (i, &xi) in x.iter().enumerate()
    {
        buf_in[i] = Dual::primal(xi);
    }
    f(buf_in, buf_out);
    for (i, ri) in r.iter_mut().enumerate()
    {
        *ri = buf_out[i].value;
        check_finite(*ri, &format!("r[{i}]"))?;
    }
    Ok(())
}

/// Levenberg-Marquardt nonlinear least squares. `residual` computes `r(x)`
/// (length `m`) with `x` and outputs passed as [`Dual`] numbers so the
/// Jacobian falls out via forward-mode autodiff, exactly as
/// [`super::newton::newton_system`] does for square systems.
///
/// `m` is the residual dimension (`m ≥ n` for a well-posed overdetermined
/// fit; `m = n` reduces to a Gauss-Newton root-find, and LM still applies —
/// see `rosenbrock_root_via_least_squares` in the tests below).
pub fn levenberg_marquardt<F>(
    residual: F,
    x0: Vec<f64>,
    m: usize,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual], &mut [Dual]),
{
    let n = x0.len();
    assert!(n > 0, "x0 must be non-empty");
    assert!(m > 0, "m must be > 0");

    let mut x = x0;
    let mut buf_in = vec![Dual::primal(0.0); n];
    let mut buf_out = vec![Dual::primal(0.0); m];
    let mut r = vec![0.0; m];
    let mut jac = Matrix::zeros(m, n);

    eval_residual_and_jacobian(&residual, &x, &mut buf_in, &mut buf_out, &mut r, &mut jac)?;
    let mut cost = 0.5 * linalg::dot(&r, &r);

    // g = Jᵀr (gradient of the cost), H = JᵀJ (Gauss-Newton Hessian approx).
    let mut g = vec![0.0; n];
    let mut h = Matrix::zeros(n, n);
    compute_gradient_and_hessian(&jac, &r, &mut g, &mut h);

    let mut d_diag: Vec<f64> = (0..n).map(|i| h[(i, i)]).collect();
    let g_inf = linalg::norm_inf(&g);
    if g_inf < tol.abs
    {
        return Ok(Solution::new(x, 0, cost));
    }

    const TAU: f64 = 1e-3;
    let mut lambda = TAU * d_diag.iter().cloned().fold(0.0f64, f64::max);
    if lambda <= 0.0
    {
        lambda = TAU;
    }
    let mut nu = 2.0;

    for k in 0..tol.max_iter
    {
        // Solve (H + lambda*D) delta = -g.
        let mut damped = h.clone();
        for i in 0..n
        {
            damped[(i, i)] += lambda * d_diag[i];
        }
        let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
        let delta = linalg::solve(damped, &rhs).map_err(|e| {
            warn!(target: "solver", "LM: damped normal equations singular at iteration {k}: {e:?}");
            e
        })?;
        for (i, &di) in delta.iter().enumerate()
        {
            check_finite(di, &format!("delta[{i}] LM k={k}"))?;
        }

        let step_norm = linalg::norm2(&delta);
        let x_norm = linalg::norm2(&x);
        if step_norm <= tol.rel * (x_norm + tol.rel)
        {
            return Ok(Solution::new(x, k, cost));
        }

        let mut x_new = x.clone();
        for i in 0..n
        {
            x_new[i] += delta[i];
        }
        let mut r_new = vec![0.0; m];
        eval_residual(&residual, &x_new, &mut buf_in, &mut buf_out, &mut r_new)?;
        let cost_new = 0.5 * linalg::dot(&r_new, &r_new);

        // Nielsen's predicted reduction: L(0) - L(delta) = 0.5*delta.(lambda*D.*delta - g).
        let predicted: f64 = (0..n)
            .map(|i| delta[i] * (lambda * d_diag[i] * delta[i] - g[i]))
            .sum::<f64>()
            * 0.5;
        let actual = cost - cost_new;

        if predicted > 0.0 && actual > 0.0
        {
            let rho = actual / predicted;
            x = x_new;
            r = r_new;
            cost = cost_new;
            eval_residual_and_jacobian(&residual, &x, &mut buf_in, &mut buf_out, &mut r, &mut jac)?;
            compute_gradient_and_hessian(&jac, &r, &mut g, &mut h);
            d_diag = (0..n).map(|i| h[(i, i)]).collect();

            if linalg::norm_inf(&g) < tol.abs
            {
                return Ok(Solution::new(x, k + 1, cost));
            }

            let shrink = 1.0 - (2.0 * rho - 1.0).powi(3);
            lambda *= shrink.max(1.0 / 3.0);
            nu = 2.0;
        }
        else
        {
            lambda *= nu;
            nu *= 2.0;
            if !lambda.is_finite() || lambda > 1e300
            {
                warn!(target: "solver", "LM: damping diverged at iteration {k}");
                return Err(SolverError::NoConvergence {
                    iterations: k,
                    residual: cost.sqrt(),
                });
            }
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: cost.sqrt(),
    })
}

fn compute_gradient_and_hessian(jac: &Matrix, r: &[f64], g: &mut [f64], h: &mut Matrix) {
    let m = jac.rows();
    let n = jac.cols();
    for j in 0..n
    {
        let mut gj = 0.0;
        for i in 0..m
        {
            gj += jac[(i, j)] * r[i];
        }
        g[j] = gj;
    }
    for j1 in 0..n
    {
        for j2 in j1..n
        {
            let mut hij = 0.0;
            for i in 0..m
            {
                hij += jac[(i, j1)] * jac[(i, j2)];
            }
            h[(j1, j2)] = hij;
            h[(j2, j1)] = hij;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rosenbrock_root_via_least_squares() {
        // Same residual as newton_system's rosenbrock_root test (m = n = 2,
        // exact zero at (1,1)) — LM should reduce to a Gauss-Newton root
        // find here.
        let s = levenberg_marquardt(
            |x, out| {
                out[0] = (x[1] - x[0] * x[0]) * 10.0;
                out[1] = -x[0] + 1.0;
            },
            vec![-1.2, 1.0],
            2,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(s.value[1], 1.0, epsilon = 1e-8);
        assert!(s.info.residual < 1e-12, "residual {}", s.info.residual);
    }

    #[test]
    fn overdetermined_exponential_fit_recovers_true_parameters() {
        // y = a * exp(b * t), noise-free synthetic data at 6 points (m=6,
        // n=2) — a genuine overdetermined fit, not a square root-find.
        let a_true = 2.5_f64;
        let b_true = -0.7_f64;
        let ts: [f64; 6] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let ys: Vec<f64> = ts.iter().map(|&t| a_true * (b_true * t).exp()).collect();

        let s = levenberg_marquardt(
            |params, out| {
                let a = params[0];
                let b = params[1];
                for (i, &t) in ts.iter().enumerate()
                {
                    out[i] = a * (b * t).exp() - ys[i];
                }
            },
            vec![1.0, 0.0],
            ts.len(),
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], a_true, epsilon = 1e-6);
        assert_relative_eq!(s.value[1], b_true, epsilon = 1e-6);
    }

    #[test]
    fn linear_residual_converges_to_the_closed_form_least_squares_solution() {
        // For r(x) = A*x - b (linear in x), the Gauss-Newton model is exact
        // (no higher-order terms dropped), so every accepted step lands
        // exactly on the quadratic cost's minimizer along that step's
        // direction and the damping ratio rho is exactly 1 — LM still takes
        // a handful of iterations to shrink lambda toward 0 (each accepted
        // step only shrinks it by Nielsen's factor, not to zero outright),
        // but must land on the closed-form solution x = (AᵀA)⁻¹Aᵀb to high
        // precision, regardless of the starting point.
        let a = [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]; // 3x2, overdetermined
        let b = [4.0, 9.0, 5.0];
        let s = levenberg_marquardt(
            |x, out| {
                for i in 0..3
                {
                    out[i] = a[i][0] * x[0] + a[i][1] * x[1] - b[i];
                }
            },
            vec![0.0, 0.0],
            3,
            Tolerance::default(),
        )
        .unwrap();
        // Closed-form normal equations solution for this specific A, b.
        // AᵀA = [[5,1],[1,10]], Aᵀb = [13, 32] -> x = [2, 3].
        assert_relative_eq!(s.value[0], 2.0, epsilon = 1e-9);
        assert_relative_eq!(s.value[1], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn converges_immediately_when_x0_is_already_the_minimizer() {
        let s = levenberg_marquardt(
            |x, out| {
                out[0] = x[0] - 3.0;
                out[1] = x[1] - 4.0;
            },
            vec![3.0, 4.0],
            2,
            Tolerance::default(),
        )
        .unwrap();
        assert_eq!(s.info.iterations, 0);
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(s.value[1], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn nonzero_residual_fit_still_reaches_a_stationary_point() {
        // Fit a single constant `c` to 3 data points that can't be matched
        // exactly (residual can't reach zero) — the real defining property
        // of a least-squares fit isn't "residual reaches zero" (the other
        // tests above all happen to have an exact fit available), it's
        // "gradient Jᵀr reaches zero". The closed-form optimum for fitting a
        // constant under sum-of-squares is the mean of the data.
        let ys = [1.0, 2.0, 6.0];
        let s = levenberg_marquardt(
            |x, out| {
                for i in 0..3
                {
                    out[i] = x[0] - ys[i];
                }
            },
            vec![0.0],
            3,
            Tolerance::default(),
        )
        .unwrap();
        let mean = ys.iter().sum::<f64>() / 3.0;
        assert_relative_eq!(s.value[0], mean, epsilon = 1e-8);
        assert!(
            s.info.residual > 1.0,
            "expected a genuinely nonzero residual, got {}",
            s.info.residual
        );
    }
}

/// Property-based tests: for any random overdetermined linear system
/// `r(x) = A*x - b`, the Gauss-Newton model is exact, so LM must converge to
/// the same closed-form least-squares solution `x = (AᵀA)⁻¹Aᵀb` regardless
/// of `A`, `b`, or the (fixed) starting point — generalizing the
/// hand-picked `linear_residual_converges_to_the_closed_form_least_squares_
/// solution` test above to many random systems.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::linalg::Matrix;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn linear_least_squares_matches_normal_equations(
            raw_a in prop::collection::vec(-10.0f64..10.0, 12), // 4x3
            raw_b in prop::collection::vec(-10.0f64..10.0, 4),
        ) {
            let m = 4;
            let n = 3;
            // Force AᵀA to be well-conditioned (add n*I via A -> [A; sqrt(n)*I]
            // is more invasive; instead just require a minimum diagonal
            // dominance on AᵀA post hoc by regenerating on failure is not
            // available in proptest, so nudge A's diagonal-ish entries up
            // front to keep the system well-posed).
            let mut a_data = raw_a.clone();
            for i in 0..n {
                a_data[i * n + i] += 15.0;
            }
            let a = Matrix::from_row_major(m, n, a_data);
            let b = raw_b;

            let s = levenberg_marquardt(
                |x, out| {
                    for i in 0..m {
                        let mut acc = Dual::primal(-b[i]);
                        for j in 0..n {
                            acc = acc + a[(i, j)] * x[j];
                        }
                        out[i] = acc;
                    }
                },
                vec![0.0; n],
                m,
                Tolerance::default(),
            );
            let s = s.expect("well-conditioned overdetermined linear LM fit must converge");

            // Independent closed-form oracle via the crate's own LU solve on
            // the normal equations AᵀA x = Aᵀb.
            let at = a.transpose();
            let ata = at.matmul(&a).unwrap();
            let atb = at.matvec(&b).unwrap();
            let x_star = linalg::solve(ata, &atb).unwrap();

            for i in 0..n {
                prop_assert!(
                    (s.value[i] - x_star[i]).abs() < 1e-6 * (1.0 + x_star[i].abs()),
                    "component {i}: LM={} closed-form={}", s.value[i], x_star[i]
                );
            }
        }
    }
}
