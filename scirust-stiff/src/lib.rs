//! # `scirust-stiff` — integrators for stiff ODE systems and index-1 DAEs
//!
//! Explicit integrators (forward Euler, classical Runge–Kutta) must take steps
//! smaller than the fastest time-scale of a system or they blow up. For *stiff*
//! problems — where that fastest scale is orders of magnitude below the scale of
//! interest — that constraint makes explicit methods unusable. This crate
//! provides **implicit** and **linearly-implicit** integrators whose stability
//! is decoupled from the fast transients, so the step size can track accuracy
//! rather than stability.
//!
//! ## What is provided
//!
//! - [`backward_euler`] — fixed-step implicit (backward) Euler. Each step solves
//!   the nonlinear stage equation with a modified-Newton iteration driven by an
//!   internal finite-difference Jacobian and a dense LU factorization. A-stable
//!   (in fact L-stable): it stays bounded on stiff decay with arbitrarily large
//!   steps.
//! - [`rosenbrock23`] — an adaptive, linearly-implicit Rosenbrock–Wanner method
//!   of order 2 with an embedded order-3 error estimate (an `ode23s`-type
//!   scheme). One Jacobian and three linear solves per step; no Newton loop.
//!   Because the order-2 result is insensitive to the accuracy of the Jacobian
//!   (only a finite-difference approximation is used), it is a *Rosenbrock-W*
//!   method. Step size is chosen by elementary error control on the embedded
//!   estimate.
//! - [`backward_euler_dae`] — the mass-matrix form `M·y' = f(t, y)`. When `M`
//!   is singular (some rows all zero), those rows are algebraic constraints, so
//!   this integrates semi-explicit index-1 DAEs.
//!
//! Everything is self-contained: the dense LU with partial pivoting, the
//! finite-difference Jacobian and the vector-norm helpers are implemented here.
//! There are no dependencies, no `unsafe`, no randomness and no global state —
//! identical inputs yield identical outputs on every platform.
//!
//! ## Error handling
//!
//! Fallible operations return [`StiffError`]. Newton non-convergence, a singular
//! linear system, step-size underflow and malformed inputs are reported rather
//! than panicking.
//!
//! ## Example
//!
//! The linear scalar test problem `y' = -50 y`, `y(0) = 1`, whose exact solution
//! is `e^{-50 t}`, is stiff: forward Euler needs `h < 0.04` or it diverges.
//! Backward Euler stays bounded and decays for *any* step.
//!
//! ```
//! use scirust_stiff::backward_euler;
//!
//! // A step 2.5x larger than the explicit stability limit.
//! let sol = backward_euler(|_t: f64, y: &[f64]| vec![-50.0 * y[0]], 0.0, &[1.0], 1.0, 0.1)
//!     .expect("integration succeeds");
//!
//! // The solution never grew and has decayed toward zero.
//! let last = sol.y.last().unwrap();
//! assert!(last[0] >= 0.0 && last[0] < 1e-3);
//! assert!(sol.y.iter().all(|row| row[0] <= 1.0));
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::f64::consts::SQRT_2;
use std::fmt;

// Square root of `f64::EPSILON`; the standard perturbation scale for
// forward-difference derivative approximations.
const SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

// Modified-Newton iteration budget for the implicit steppers.
const NEWTON_MAX_ITER: usize = 50;
// Absolute / relative scales for the Newton convergence test. Convergence is
// declared when the scaled max-norm of the update drops to 1.
const NEWTON_ATOL: f64 = 1e-10;
const NEWTON_RTOL: f64 = 1e-8;

// Hard cap on the number of Rosenbrock step attempts, so a pathological
// right-hand side cannot spin forever.
const MAX_STEPS: usize = 5_000_000;

/// The result of an integration: a list of times and the state row at each time.
///
/// `t[i]` is the `i`-th output time and `y[i]` is the full state vector there,
/// so `y[i][k]` is component `k` at time `t[i]`. The first row is always the
/// initial condition.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Output times, strictly increasing, starting at `t0`.
    pub t: Vec<f64>,
    /// State rows; `y[i]` is the state vector at `t[i]`.
    pub y: Vec<Vec<f64>>,
}

/// Errors returned by the integrators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum StiffError {
    /// The right-hand side returned a vector whose length differs from the
    /// state dimension.
    DimMismatch {
        /// The state dimension that was expected.
        expected: usize,
        /// The length that was actually returned.
        got: usize,
    },
    /// A dense linear solve hit a zero (or non-finite) pivot: the iteration
    /// matrix is singular. For a DAE this usually means the constraints do not
    /// determine the algebraic variables.
    SingularMatrix {
        /// The pivot column at which the factorization failed.
        column: usize,
    },
    /// The modified-Newton iteration failed to converge within the budget.
    NewtonDivergence {
        /// The step index (0-based) on which convergence failed.
        step: usize,
        /// The number of iterations performed.
        iterations: usize,
        /// The final scaled residual norm (may be non-finite on blow-up).
        residual: f64,
    },
    /// The adaptive step size fell below the smallest permissible value while
    /// trying to meet the requested tolerance (e.g. approaching a singularity).
    StepUnderflow {
        /// The time at which progress stalled.
        t: f64,
        /// The rejected (too-small) step size.
        h: f64,
    },
    /// The adaptive integrator exceeded its total step-attempt budget.
    MaxStepsExceeded {
        /// The number of attempts made before giving up.
        steps: usize,
    },
    /// An input argument was invalid; the message explains why.
    BadInput(String),
}

impl fmt::Display for StiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            StiffError::DimMismatch { expected, got } => write!(
                f,
                "right-hand side returned length {got}, but the state dimension is {expected}"
            ),
            StiffError::SingularMatrix { column } =>
            {
                write!(
                    f,
                    "singular iteration matrix (zero pivot at column {column})"
                )
            },
            StiffError::NewtonDivergence {
                step,
                iterations,
                residual,
            } => write!(
                f,
                "Newton iteration did not converge on step {step} after {iterations} iterations (residual {residual:e})"
            ),
            StiffError::StepUnderflow { t, h } =>
            {
                write!(f, "step size underflow at t = {t:e} (h = {h:e})")
            },
            StiffError::MaxStepsExceeded { steps } =>
            {
                write!(f, "exceeded the maximum of {steps} step attempts")
            },
            StiffError::BadInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for StiffError {}

// ============================================================ //
//  Internal linear-algebra helpers                             //
// ============================================================ //

// In-place LU factorization with partial (row) pivoting. On return `a` holds
// the combined unit-lower / upper factors and the returned vector `piv` records
// the row permutation (`piv[i]` is the original row now sitting at row `i`).
// Returns `SingularMatrix` on a zero or non-finite pivot.
fn lu_factor(a: &mut [Vec<f64>]) -> Result<Vec<usize>, StiffError> {
    let n = a.len();
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n
    {
        // Locate the pivot: largest magnitude in column k, on or below the
        // diagonal.
        let mut p = k;
        let mut best = a[k][k].abs();
        for (i, row) in a.iter().enumerate().skip(k + 1)
        {
            let v = row[k].abs();
            if v > best
            {
                best = v;
                p = i;
            }
        }
        // `best` is a magnitude, so `<= 0.0` means an exact zero pivot; the NaN
        // guard rejects a non-finite matrix.
        if best <= 0.0 || best.is_nan()
        {
            return Err(StiffError::SingularMatrix { column: k });
        }
        if p != k
        {
            a.swap(p, k);
            piv.swap(p, k);
        }
        let akk = a[k][k];
        // Split so the pivot row can be read while the rows below it are
        // updated in place.
        let (upper, lower) = a.split_at_mut(k + 1);
        let pivot_row = &upper[k];
        for row in lower.iter_mut()
        {
            let factor = row[k] / akk;
            row[k] = factor;
            for (elem, &p_elem) in row.iter_mut().zip(pivot_row.iter()).skip(k + 1)
            {
                *elem -= factor * p_elem;
            }
        }
    }
    Ok(piv)
}

// Solve `A x = b` given the LU factors and permutation from `lu_factor`.
fn lu_solve(lu: &[Vec<f64>], piv: &[usize], b: &[f64]) -> Vec<f64> {
    let n = lu.len();
    // Apply the row permutation to the right-hand side.
    let mut x: Vec<f64> = (0..n).map(|i| b[piv[i]]).collect();
    // Forward substitution (unit lower triangle).
    for i in 1..n
    {
        let mut s = x[i];
        for j in 0..i
        {
            s -= lu[i][j] * x[j];
        }
        x[i] = s;
    }
    // Back substitution (upper triangle).
    for i in (0..n).rev()
    {
        let mut s = x[i];
        for j in (i + 1)..n
        {
            s -= lu[i][j] * x[j];
        }
        x[i] = s / lu[i][i];
    }
    x
}

// Evaluate the right-hand side, checking that it returns the right length.
fn eval_f<F>(f: &F, t: f64, y: &[f64]) -> Result<Vec<f64>, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let v = f(t, y);
    if v.len() != y.len()
    {
        return Err(StiffError::DimMismatch {
            expected: y.len(),
            got: v.len(),
        });
    }
    Ok(v)
}

// Forward-difference Jacobian `J[i][j] = d f_i / d y_j` at `(t, y)`. `f0` must be
// `f(t, y)` already evaluated.
fn fd_jacobian<F>(f: &F, t: f64, y: &[f64], f0: &[f64]) -> Result<Vec<Vec<f64>>, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y.len();
    let mut jac = vec![vec![0.0; n]; n];
    let mut yp = y.to_vec();
    for col in 0..n
    {
        let saved = yp[col];
        let step = SQRT_EPS * saved.abs().max(1.0);
        yp[col] = saved + step;
        // The realized step after rounding; never zero because `step` is well
        // above the ULP of `saved`.
        let dstep = yp[col] - saved;
        let fp = eval_f(f, t, &yp)?;
        yp[col] = saved;
        for row in 0..n
        {
            jac[row][col] = (fp[row] - f0[row]) / dstep;
        }
    }
    Ok(jac)
}

// Weighted root-mean-square norm used for adaptive error control. `ya`, `yb`
// bracket the step so the scale tracks the larger endpoint magnitude.
fn wrms_norm(v: &[f64], ya: &[f64], yb: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = v.len();
    if n == 0
    {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n
    {
        let scale = atol + rtol * ya[i].abs().max(yb[i].abs());
        let r = v[i] / scale;
        s += r * r;
    }
    (s / n as f64).sqrt()
}

// Scaled max-norm of a Newton update relative to the current iterate.
fn newton_norm(dv: &[f64], y: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..dv.len()
    {
        let scale = NEWTON_ATOL + NEWTON_RTOL * y[i].abs();
        let r = (dv[i] / scale).abs();
        // A NaN component means the iterate has blown up; report it as NaN so
        // the caller treats the step as non-converged rather than silently
        // dropping it (`r > m` is false for NaN).
        if r.is_nan()
        {
            return f64::NAN;
        }
        if r > m
        {
            m = r;
        }
    }
    m
}

// ============================================================ //
//  Input validation                                            //
// ============================================================ //

fn validate_common(y0: &[f64], t0: f64, t_end: f64) -> Result<(), StiffError> {
    if y0.is_empty()
    {
        return Err(StiffError::BadInput(
            "state vector y0 must be non-empty".to_string(),
        ));
    }
    if t_end < t0 || t0.is_nan() || t_end.is_nan()
    {
        return Err(StiffError::BadInput(
            "t_end must be greater than or equal to t0".to_string(),
        ));
    }
    Ok(())
}

fn require_positive(value: f64, name: &str) -> Result<(), StiffError> {
    if value <= 0.0 || value.is_nan()
    {
        return Err(StiffError::BadInput(format!("{name} must be positive")));
    }
    Ok(())
}

// ============================================================ //
//  Implicit (backward-Euler) machinery                         //
// ============================================================ //

// One backward-Euler / DAE step by modified Newton.
//
// Solves `G(y) = M·(y - yn) - h·f(t1, y) = 0` for `y = y_{n+1}`, where `M` is the
// mass matrix (identity when `mass` is `None`). The Jacobian of `G`,
// `N = M - h·(df/dy)`, is formed once from a finite-difference Jacobian at
// `(t1, yn)` and its LU factorization is reused across the iteration (hence
// *modified* Newton).
fn newton_step<F>(
    f: &F,
    mass: Option<&[Vec<f64>]>,
    t1: f64,
    yn: &[f64],
    h: f64,
    step_idx: usize,
) -> Result<Vec<f64>, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = yn.len();
    let f_yn = eval_f(f, t1, yn)?;
    let jac = fd_jacobian(f, t1, yn, &f_yn)?;

    // Iteration matrix N = M - h J.
    let mut nm = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        for k in 0..n
        {
            let m_ik = match mass
            {
                Some(m) => m[i][k],
                None =>
                {
                    if i == k
                    {
                        1.0
                    }
                    else
                    {
                        0.0
                    }
                },
            };
            nm[i][k] = m_ik - h * jac[i][k];
        }
    }
    let piv = lu_factor(&mut nm)?;

    let mut y = yn.to_vec();
    let mut residual = f64::INFINITY;
    let mut iters = 0;
    while iters < NEWTON_MAX_ITER
    {
        iters += 1;
        let fy = eval_f(f, t1, &y)?;
        // g = M (y - yn) - h f(t1, y)
        let mut g = vec![0.0; n];
        for i in 0..n
        {
            let mv = match mass
            {
                Some(m) =>
                {
                    let mut acc = 0.0;
                    for k in 0..n
                    {
                        acc += m[i][k] * (y[k] - yn[k]);
                    }
                    acc
                },
                None => y[i] - yn[i],
            };
            g[i] = mv - h * fy[i];
        }
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let dy = lu_solve(&nm, &piv, &neg_g);
        for i in 0..n
        {
            y[i] += dy[i];
        }
        residual = newton_norm(&dy, &y);
        if residual <= 1.0
        {
            return Ok(y);
        }
        // Detect blow-up early so we do not burn the whole budget on infinities.
        if !residual.is_finite() && iters >= 3
        {
            break;
        }
    }
    Err(StiffError::NewtonDivergence {
        step: step_idx,
        iterations: iters,
        residual,
    })
}

/// Fixed-step **backward (implicit) Euler** for `y' = f(t, y)`.
///
/// Integrates from `t0` to `t_end` in steps of `h` (the final step is shortened
/// so the last output time is exactly `t_end`). Each step solves the implicit
/// stage equation `y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1})` with a modified-Newton
/// iteration using an internal finite-difference Jacobian and dense LU solve.
///
/// The method is L-stable, so on stiff decay it stays bounded for *any* step
/// size — unlike an explicit method, which diverges once `h` exceeds the fast
/// time-scale. Accuracy is first order: the global error is `O(h)`.
///
/// # Errors
///
/// Returns [`StiffError::BadInput`] for an empty `y0`, non-positive `h`, or
/// `t_end < t0`; [`StiffError::DimMismatch`] if `f` returns the wrong length;
/// [`StiffError::SingularMatrix`] if the Newton iteration matrix is singular;
/// and [`StiffError::NewtonDivergence`] if the iteration fails to converge.
pub fn backward_euler<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> Result<Solution, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_common(y0, t0, t_end)?;
    require_positive(h, "step size h")?;
    // Validate the right-hand side dimension up front.
    eval_f(&f, t0, y0)?;

    integrate_backward_euler(&f, None, t0, y0, t_end, h)
}

/// Fixed-step **backward Euler for the mass-matrix form** `M·y' = f(t, y)`.
///
/// `mass` is a dense `n×n` matrix (`n = y0.len()`). Rows of `M` that are
/// entirely zero turn the corresponding equations into algebraic constraints,
/// so this integrates semi-explicit **index-1 DAEs**: differential rows advance
/// the state while algebraic rows enforce `f_i(t, y) = 0` at every step (to
/// Newton tolerance). Each step applies modified Newton to the residual
/// `M·(y_{n+1} - y_n)/h - f(t_{n+1}, y_{n+1}) = 0`.
///
/// The initial condition `y0` should be consistent (it should already satisfy
/// the algebraic constraints).
///
/// # Errors
///
/// As [`backward_euler`], plus [`StiffError::BadInput`] if `mass` is not square
/// or its size does not match `y0`. A structurally singular DAE (constraints
/// that do not pin down the algebraic variables) surfaces as
/// [`StiffError::SingularMatrix`].
pub fn backward_euler_dae<F>(
    mass: &[Vec<f64>],
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> Result<Solution, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_common(y0, t0, t_end)?;
    require_positive(h, "step size h")?;
    let n = y0.len();
    if mass.len() != n || mass.iter().any(|row| row.len() != n)
    {
        return Err(StiffError::BadInput(format!(
            "mass matrix must be {n}x{n} to match y0"
        )));
    }
    eval_f(&f, t0, y0)?;

    integrate_backward_euler(&f, Some(mass), t0, y0, t_end, h)
}

// Shared fixed-step driver for both the plain and mass-matrix backward Euler.
fn integrate_backward_euler<F>(
    f: &F,
    mass: Option<&[Vec<f64>]>,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    h: f64,
) -> Result<Solution, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let span = t_end - t0;
    let tiny = span.abs() * 1e-12 + f64::MIN_POSITIVE;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut ts = vec![t0];
    let mut ys = vec![y0.to_vec()];
    let mut step = 0;

    while t < t_end - tiny
    {
        // Shorten the last step so it lands exactly on t_end.
        let hh = if t + h > t_end { t_end - t } else { h };
        let t1 = t + hh;
        y = newton_step(f, mass, t1, &y, hh, step)?;
        t = t1;
        ts.push(t);
        ys.push(y.clone());
        step += 1;
    }

    Ok(Solution { t: ts, y: ys })
}

/// Adaptive **Rosenbrock-W (order 2, embedded order 3)** for `y' = f(t, y)`.
///
/// A linearly-implicit `ode23s`-type scheme: per step it forms one
/// finite-difference Jacobian `J` and one factorization of `W = I - h·d·J`
/// (`d = 1/(2 + √2)`), then performs three back-solves for the stage vectors —
/// no nonlinear iteration. The difference between the embedded order-3 and the
/// order-2 result gives a local error estimate that drives elementary
/// (`I`-controller) step-size selection with the mixed tolerance
/// `atol + rtol·|y|`. The order-2 update is L-stable and its accuracy does not
/// depend on the Jacobian being exact, which is what makes the finite-difference
/// Jacobian admissible (a *W*-method).
///
/// `h0` is the initial step; it is grown or shrunk automatically. Steps whose
/// estimated error exceeds tolerance are rejected and retried with a smaller
/// step (reusing the same Jacobian).
///
/// # Errors
///
/// Returns [`StiffError::BadInput`] for an empty `y0`, non-positive `rtol`,
/// `atol` or `h0`, or `t_end < t0`; [`StiffError::DimMismatch`] on a wrong-length
/// right-hand side; [`StiffError::SingularMatrix`] if `W` is singular;
/// [`StiffError::StepUnderflow`] if the step must shrink below the floor to meet
/// tolerance (e.g. near a finite-time singularity); and
/// [`StiffError::MaxStepsExceeded`] if the attempt budget is exhausted.
pub fn rosenbrock23<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
    h0: f64,
) -> Result<Solution, StiffError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_common(y0, t0, t_end)?;
    require_positive(rtol, "rtol")?;
    require_positive(atol, "atol")?;
    require_positive(h0, "initial step h0")?;
    let n = y0.len();
    eval_f(&f, t0, y0)?;

    let span = t_end - t0;
    let mut ts = vec![t0];
    let mut ys = vec![y0.to_vec()];
    if span <= 0.0
    {
        return Ok(Solution { t: ts, y: ys });
    }

    // Rosenbrock-W(2,3) coefficients (ode23s).
    let d = 1.0 / (2.0 + SQRT_2);
    let e32 = 6.0 + SQRT_2;
    // Step-controller constants.
    let safety = 0.9;
    let facmin = 0.2;
    let facmax = 5.0;

    let tiny = span * 1e-8;
    let hmin = span * 1e-10;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h0.min(span).max(hmin);
    let mut attempts = 0usize;

    while t < t_end - tiny
    {
        if h < hmin
        {
            return Err(StiffError::StepUnderflow { t, h });
        }
        if t + h > t_end
        {
            h = t_end - t;
        }

        // One Jacobian, one directional t-derivative, per step position.
        let f0 = eval_f(&f, t, &y)?;
        let jac = fd_jacobian(&f, t, &y, &f0)?;
        let dt = SQRT_EPS * t.abs().max(1.0);
        let f_dt = eval_f(&f, t + dt, &y)?;
        let ft: Vec<f64> = (0..n).map(|i| (f_dt[i] - f0[i]) / dt).collect();

        // Attempt the step, shrinking h on rejection (Jacobian frozen).
        let (y_new, err_norm) = loop
        {
            attempts += 1;
            if attempts > MAX_STEPS
            {
                return Err(StiffError::MaxStepsExceeded { steps: attempts });
            }

            // W = I - h d J, factorized once and reused for the three solves.
            let mut w = vec![vec![0.0; n]; n];
            for i in 0..n
            {
                for k in 0..n
                {
                    w[i][k] = -h * d * jac[i][k];
                }
                w[i][i] += 1.0;
            }
            let piv = lu_factor(&mut w)?;

            let rhs1: Vec<f64> = (0..n).map(|i| f0[i] + h * d * ft[i]).collect();
            let k1 = lu_solve(&w, &piv, &rhs1);

            let y_mid: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
            let f1 = eval_f(&f, t + 0.5 * h, &y_mid)?;
            let rhs2: Vec<f64> = (0..n).map(|i| f1[i] - k1[i]).collect();
            let mut k2 = lu_solve(&w, &piv, &rhs2);
            for i in 0..n
            {
                k2[i] += k1[i];
            }

            let y_next: Vec<f64> = (0..n).map(|i| y[i] + h * k2[i]).collect();
            let f2 = eval_f(&f, t + h, &y_next)?;
            let rhs3: Vec<f64> = (0..n)
                .map(|i| f2[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + h * d * ft[i])
                .collect();
            let k3 = lu_solve(&w, &piv, &rhs3);

            let err: Vec<f64> = (0..n)
                .map(|i| (h / 6.0) * (k1[i] - 2.0 * k2[i] + k3[i]))
                .collect();
            let en = wrms_norm(&err, &y, &y_next, rtol, atol);

            if en <= 1.0
            {
                break (y_next, en);
            }

            // Reject: shrink and retry.
            let fac = if en.is_finite() && en > 0.0
            {
                (safety * (1.0 / en).cbrt()).clamp(facmin, 1.0)
            }
            else
            {
                facmin
            };
            h *= fac;
            if h < hmin || h.is_nan()
            {
                return Err(StiffError::StepUnderflow { t, h });
            }
        };

        // Accept.
        t += h;
        y = y_new;
        ts.push(t);
        ys.push(y.clone());

        // Propose the next step.
        let fac = if err_norm > 0.0
        {
            (safety * (1.0 / err_norm).cbrt()).clamp(facmin, facmax)
        }
        else
        {
            facmax
        };
        h *= fac;
    }

    Ok(Solution { t: ts, y: ys })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Max absolute difference between two equal-length vectors.
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // ---- backward Euler: stiff stability and accuracy ------------------- //

    #[test]
    fn backward_euler_is_stable_on_stiff_decay_with_large_step() {
        // y' = -50 y. Explicit Euler needs h < 2/50 = 0.04; we use h = 0.5.
        let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let sol = backward_euler(f, 0.0, &[1.0], 5.0, 0.5).unwrap();

        // Bounded, non-negative, monotonically decreasing toward zero.
        for row in &sol.y
        {
            assert!(row[0] >= 0.0 && row[0] <= 1.0);
        }
        for w in sol.y.windows(2)
        {
            assert!(w[1][0] <= w[0][0] + 1e-15);
        }
        assert!(sol.y.last().unwrap()[0] < 1e-3);

        // Contrast: explicit Euler with the same step blows up.
        let mut ye = 1.0_f64;
        for _ in 0..10
        {
            ye += 0.5 * (-50.0 * ye);
        }
        assert!(ye.abs() > 1e6, "explicit Euler must diverge, got {ye}");
    }

    #[test]
    fn backward_euler_matches_exponential_with_small_step() {
        // With a small step, match e^{-50 t} to a documented tolerance.
        // Backward Euler for y' = -50 y has relative error ~ 1250·t·h, so at
        // t = 0.2 with h = 1e-5 the error is ~2.5e-3; we assert 1e-2.
        let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let h = 1e-5;
        let sol = backward_euler(f, 0.0, &[1.0], 0.2, h).unwrap();
        for (t, row) in sol.t.iter().zip(&sol.y)
        {
            let exact = (-50.0 * t).exp();
            let rel = (row[0] - exact).abs() / exact;
            assert!(rel < 1e-2, "t={t}, rel err {rel}");
        }
    }

    #[test]
    fn backward_euler_two_d_linear_matches_matrix_exponential() {
        // A = [[-100, 1], [0, -1]], stiffness ratio 100.
        let f = |_t: f64, y: &[f64]| vec![-100.0 * y[0] + y[1], -y[1]];
        let h = 1e-4;
        let sol = backward_euler(f, 0.0, &[1.0, 1.0], 1.0, h).unwrap();
        // y2(t) = e^{-t}; first-order BE error ~ h at t=1.
        let last = sol.y.last().unwrap();
        assert!((last[1] - (-1.0_f64).exp()).abs() < 5e-3);
    }

    // ---- rosenbrock23: adaptive stiff integration ----------------------- //

    // Exact solution of y' = A y, A = [[-100,1],[0,-1]], y0 = [1,1].
    fn exact_2d(t: f64) -> [f64; 2] {
        let e100 = (-100.0 * t).exp();
        let e1 = (-t).exp();
        let y1 = e100 - (e100 - e1) / 99.0;
        [y1, e1]
    }

    #[test]
    fn rosenbrock_matches_exponential_and_takes_sane_step_count() {
        let f = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let sol = rosenbrock23(f, 0.0, &[1.0], 1.0, 1e-6, 1e-9, 1e-2).unwrap();

        // Accuracy vs e^{-50 t}: ~1e-4 target (documented).
        let mut worst = 0.0_f64;
        for (t, row) in sol.t.iter().zip(&sol.y)
        {
            let exact = (-50.0 * t).exp();
            worst = worst.max((row[0] - exact).abs());
        }
        assert!(worst < 1e-4, "worst abs error {worst}");

        // Monotone-decreasing state (up to the atol noise floor, below which the
        // decayed solution is numerically zero), and a sane step count.
        for w in sol.y.windows(2)
        {
            assert!(w[1][0] <= w[0][0] + 1e-8);
        }
        assert!(
            sol.t.len() >= 3 && sol.t.len() < 500,
            "steps {}",
            sol.t.len()
        );
    }

    #[test]
    fn rosenbrock_two_d_linear_matches_matrix_exponential() {
        let f = |_t: f64, y: &[f64]| vec![-100.0 * y[0] + y[1], -y[1]];
        let sol = rosenbrock23(f, 0.0, &[1.0, 1.0], 1.0, 1e-7, 1e-10, 1e-3).unwrap();
        for (t, row) in sol.t.iter().zip(&sol.y)
        {
            let e = exact_2d(*t);
            assert!(max_abs_diff(row, &e) < 1e-3, "t={t}");
        }
    }

    #[test]
    fn rosenbrock_two_d_matches_tiny_step_backward_euler() {
        // Reference from a tiny-step backward Euler on the same system.
        let f = |_t: f64, y: &[f64]| vec![-100.0 * y[0] + y[1], -y[1]];
        let reference = backward_euler(f, 0.0, &[1.0, 1.0], 1.0, 1e-5).unwrap();
        let ros = rosenbrock23(f, 0.0, &[1.0, 1.0], 1.0, 1e-7, 1e-10, 1e-3).unwrap();
        let r_last = reference.y.last().unwrap();
        let s_last = ros.y.last().unwrap();
        assert!(max_abs_diff(r_last, s_last) < 5e-3);
    }

    #[test]
    fn rosenbrock_nonlinear_van_der_pol_matches_reference() {
        // Van der Pol at moderate stiffness (mu = 5).
        let vdp = |_t: f64, y: &[f64]| {
            let mu = 5.0;
            vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
        };
        let reference = backward_euler(vdp, 0.0, &[2.0, 0.0], 1.0, 1e-4).unwrap();
        let ros = rosenbrock23(vdp, 0.0, &[2.0, 0.0], 1.0, 1e-6, 1e-9, 1e-2).unwrap();
        let r_last = reference.y.last().unwrap();
        let s_last = ros.y.last().unwrap();
        assert!(
            max_abs_diff(r_last, s_last) < 5e-2,
            "ref {r_last:?} vs ros {s_last:?}"
        );
    }

    #[test]
    fn rosenbrock_tracks_nonautonomous_stiff_forcing() {
        // y' = -50(y - cos t) - sin t has exact solution y = cos t and is stiff,
        // exercising the d f/d t term of the Rosenbrock scheme.
        let f = |t: f64, y: &[f64]| vec![-50.0 * (y[0] - t.cos()) - t.sin()];
        let sol = rosenbrock23(f, 0.0, &[1.0], 3.0, 1e-6, 1e-9, 1e-2).unwrap();
        for (t, row) in sol.t.iter().zip(&sol.y)
        {
            assert!((row[0] - t.cos()).abs() < 5e-3, "t={t}");
        }
    }

    // ---- DAE: semi-explicit index-1 ------------------------------------- //

    #[test]
    fn dae_index1_constraint_holds_every_step() {
        // y1' = y2, 0 = y1 + y2 - t. Consistent init y1(0)=0.5, y2(0)=-0.5.
        // Reduces to y1' = t - y1, exact y1 = t - 1 + 1.5 e^{-t}.
        let mass = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let f = |t: f64, y: &[f64]| vec![y[1], y[0] + y[1] - t];
        let sol = backward_euler_dae(&mass, f, 0.0, &[0.5, -0.5], 1.0, 1e-3).unwrap();

        // Algebraic constraint y1 + y2 = t at every stored step.
        for (t, row) in sol.t.iter().zip(&sol.y)
        {
            let c = row[0] + row[1] - t;
            assert!(c.abs() < 1e-8, "constraint residual {c} at t={t}");
        }

        // Differential variable matches the analytic solution (BE first order).
        let tf = *sol.t.last().unwrap();
        let exact_y1 = tf - 1.0 + 1.5 * (-tf).exp();
        assert!((sol.y.last().unwrap()[0] - exact_y1).abs() < 5e-3);
    }

    // ---- error paths ---------------------------------------------------- //

    #[test]
    fn dimension_mismatch_is_reported() {
        // f returns length 1 for a length-2 state.
        let f = |_t: f64, _y: &[f64]| vec![0.0];
        let r = backward_euler(f, 0.0, &[1.0, 2.0], 1.0, 0.1);
        assert!(matches!(
            r,
            Err(StiffError::DimMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn structurally_singular_dae_is_reported() {
        // f2 is constant, so both the mass row and the Jacobian row 2 vanish:
        // the iteration matrix is singular.
        let mass = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let f = |_t: f64, y: &[f64]| vec![y[1], 0.0];
        let r = backward_euler_dae(&mass, f, 0.0, &[1.0, 1.0], 1.0, 0.1);
        assert!(matches!(r, Err(StiffError::SingularMatrix { .. })));
    }

    #[test]
    fn newton_divergence_is_reported() {
        // y = y_n + h(y^2 + 1) has no real root for h=1, y_n=1 (discriminant<0),
        // so the implicit stage equation is unsolvable and Newton diverges.
        let f = |_t: f64, y: &[f64]| vec![y[0] * y[0] + 1.0];
        let r = backward_euler(f, 0.0, &[1.0], 1.0, 1.0);
        assert!(matches!(r, Err(StiffError::NewtonDivergence { .. })));
    }

    #[test]
    fn finite_time_blowup_triggers_step_failure() {
        // y' = y^2 blows up at t = 1; integrating to t = 2 forces the adaptive
        // step below the floor.
        let f = |_t: f64, y: &[f64]| vec![y[0] * y[0]];
        let r = rosenbrock23(f, 0.0, &[1.0], 2.0, 1e-6, 1e-9, 1e-2);
        assert!(
            matches!(
                r,
                Err(StiffError::StepUnderflow { .. })
                    | Err(StiffError::SingularMatrix { .. })
                    | Err(StiffError::MaxStepsExceeded { .. })
            ),
            "expected a step failure, got {r:?}"
        );
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        assert!(matches!(
            backward_euler(f, 0.0, &[], 1.0, 0.1),
            Err(StiffError::BadInput(_))
        ));
        assert!(matches!(
            backward_euler(f, 0.0, &[1.0], 1.0, -0.1),
            Err(StiffError::BadInput(_))
        ));
        assert!(matches!(
            backward_euler(f, 0.0, &[1.0], -1.0, 0.1),
            Err(StiffError::BadInput(_))
        ));
        assert!(matches!(
            rosenbrock23(f, 0.0, &[1.0], 1.0, 0.0, 1e-9, 1e-2),
            Err(StiffError::BadInput(_))
        ));
        assert!(matches!(
            backward_euler_dae(&[vec![1.0, 0.0]], f, 0.0, &[1.0], 1.0, 0.1),
            Err(StiffError::BadInput(_))
        ));
    }

    // ---- internal helpers ---------------------------------------------- //

    #[test]
    fn lu_solves_a_permuted_system() {
        // A = [[0,1],[1,0]], b = [1,2] -> x = [2,1].
        let mut a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let piv = lu_factor(&mut a).unwrap();
        let x = lu_solve(&a, &piv, &[1.0, 2.0]);
        assert!((x[0] - 2.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lu_reports_singular_matrix() {
        let mut a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(matches!(
            lu_factor(&mut a),
            Err(StiffError::SingularMatrix { .. })
        ));
    }

    #[test]
    fn finite_difference_jacobian_is_accurate() {
        // f(y) = [y0^2, y0*y1] -> J = [[2 y0, 0], [y1, y0]] at (3, 4).
        let f = |_t: f64, y: &[f64]| vec![y[0] * y[0], y[0] * y[1]];
        let y = [3.0, 4.0];
        let f0 = f(0.0, &y);
        let j = fd_jacobian(&f, 0.0, &y, &f0).unwrap();
        assert!((j[0][0] - 6.0).abs() < 1e-6);
        assert!(j[0][1].abs() < 1e-6);
        assert!((j[1][0] - 4.0).abs() < 1e-6);
        assert!((j[1][1] - 3.0).abs() < 1e-6);
    }
}
