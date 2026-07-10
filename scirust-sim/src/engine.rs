//! Deterministic time-stepping engine: the [`System`] and
//! [`SecondOrderSystem`] traits, the fixed-step integrators and the
//! [`Trajectory`] they produce.

use std::error::Error;
use std::fmt;

/// Hard cap on the number of integration steps a single call may take, so a
/// pathological `(t_end - t0) / h` cannot spin forever.
const MAX_STEPS: usize = 10_000_000;

/// A continuous-time dynamical system `y' = f(t, y)`.
///
/// The derivative is written in place into `dydt` (whose length equals
/// [`dim`](System::dim)), the same shape used by the closures accepted by
/// `scirust_solvers::ode::dopri5`, so implementors can be handed to that
/// adaptive integrator with a one-line closure adapter.
pub trait System {
    /// Dimension of the state vector.
    fn dim(&self) -> usize;
    /// Write `f(t, y)` into `dydt`. Both slices have length [`dim`](System::dim).
    fn derivatives(&self, t: f64, y: &[f64], dydt: &mut [f64]);
}

/// A mechanical system `q'' = a(t, q, v)` with `v = q'`.
///
/// Used by [`simulate_second_order`], which integrates with the symplectic
/// (semi-implicit) Euler method. When the acceleration depends only on `q`
/// (a separable Hamiltonian: gravity, springs, Kepler attraction), that
/// method conserves a perturbed energy, so the energy error stays *bounded*
/// over arbitrarily long horizons instead of drifting.
pub trait SecondOrderSystem {
    /// Number of degrees of freedom (length of `q` and `v`).
    fn dof(&self) -> usize;
    /// Write `a(t, q, v)` into `acc`. All slices have length [`dof`](SecondOrderSystem::dof).
    fn acceleration(&self, t: f64, q: &[f64], v: &[f64], acc: &mut [f64]);
}

/// View of a [`SecondOrderSystem`] as a first-order [`System`] with state
/// `y = [q, v]`, so mechanical systems can also be integrated by [`simulate`]
/// (classical RK4) when short-horizon accuracy matters more than long-horizon
/// energy behaviour.
pub struct FirstOrderForm<'a, S: SecondOrderSystem>(pub &'a S);

impl<S: SecondOrderSystem> System for FirstOrderForm<'_, S> {
    fn dim(&self) -> usize {
        2 * self.0.dof()
    }

    fn derivatives(&self, t: f64, y: &[f64], dydt: &mut [f64]) {
        let n = self.0.dof();
        let (q, v) = y.split_at(n);
        let (dq, dv) = dydt.split_at_mut(n);
        dq.copy_from_slice(v);
        self.0.acceleration(t, q, v, dv);
    }
}

/// The result of a simulation: a list of times and the state row at each time.
///
/// `t[i]` is the `i`-th output time and `y[i]` is the full state vector
/// there, so `y[i][k]` is component `k` at time `t[i]`. The first row is
/// always the initial condition and the last time is exactly `t_end`.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Output times, strictly increasing, starting at `t0`.
    pub t: Vec<f64>,
    /// State rows; `y[i]` is the state vector at `t[i]`.
    pub y: Vec<Vec<f64>>,
}

impl Trajectory {
    /// Number of stored samples (including the initial condition).
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// `true` when no samples are stored.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    /// The final time, if any samples are stored.
    pub fn last_time(&self) -> Option<f64> {
        self.t.last().copied()
    }

    /// The final state row, if any samples are stored.
    pub fn last_state(&self) -> Option<&[f64]> {
        self.y.last().map(Vec::as_slice)
    }

    /// The time series of state component `k`, or `None` when `k` is out of
    /// range or the trajectory is empty.
    pub fn column(&self, k: usize) -> Option<Vec<f64>> {
        if self.y.first().is_none_or(|row| k >= row.len())
        {
            return None;
        }
        Some(self.y.iter().map(|row| row[k]).collect())
    }
}

/// Errors returned by the simulation engine and the domain models.
#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    /// An input argument was invalid; the message explains why.
    BadInput(String),
    /// A state or action vector had the wrong length.
    DimMismatch {
        /// The length that was expected.
        expected: usize,
        /// The length that was actually supplied.
        got: usize,
    },
    /// The state stopped being finite (overflow or NaN) at time `t`; the step
    /// size is too large for the system's fastest time-scale, or the model is
    /// genuinely divergent.
    NonFinite {
        /// The time at which a non-finite component first appeared.
        t: f64,
    },
    /// The adaptive step size collapsed below the smallest permissible value
    /// while trying to meet the requested tolerance (e.g. approaching a
    /// singularity, or tolerances asked tighter than `f64` can deliver).
    StepUnderflow {
        /// The time at which progress stalled.
        t: f64,
    },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            SimError::BadInput(msg) => write!(f, "invalid input: {msg}"),
            SimError::DimMismatch { expected, got } =>
            {
                write!(
                    f,
                    "dimension mismatch: expected length {expected}, got {got}"
                )
            },
            SimError::NonFinite { t } =>
            {
                write!(
                    f,
                    "state became non-finite at t = {t}; reduce the step size"
                )
            },
            SimError::StepUnderflow { t } =>
            {
                write!(
                    f,
                    "adaptive step size underflowed at t = {t}; the tolerance is \
                     unreachable (approaching a singularity, or too tight for f64)"
                )
            },
        }
    }
}

impl Error for SimError {}

fn validate_run(dim: usize, y0: &[f64], t0: f64, t_end: f64, h: f64) -> Result<usize, SimError> {
    if dim == 0
    {
        return Err(SimError::BadInput("system dimension is zero".to_string()));
    }
    if y0.len() != dim
    {
        return Err(SimError::DimMismatch {
            expected: dim,
            got: y0.len(),
        });
    }
    if y0.iter().any(|c| !c.is_finite())
    {
        return Err(SimError::BadInput(
            "initial state has a non-finite component".to_string(),
        ));
    }
    if !t0.is_finite() || !t_end.is_finite() || t_end <= t0
    {
        return Err(SimError::BadInput(format!(
            "time span [{t0}, {t_end}] must be finite with t_end > t0"
        )));
    }
    if !h.is_finite() || h <= 0.0
    {
        return Err(SimError::BadInput(format!(
            "step size {h} must be finite and positive"
        )));
    }
    let steps = ((t_end - t0) / h).ceil() as usize;
    if steps > MAX_STEPS
    {
        return Err(SimError::BadInput(format!(
            "time span requires {steps} steps, above the {MAX_STEPS} budget"
        )));
    }
    Ok(steps.max(1))
}

/// Integrate `system` from `y0` over `[t0, t_end]` with the classical
/// fixed-step fourth-order Runge–Kutta method.
///
/// Every step of size `h` is recorded; the final step is shortened so the
/// trajectory lands exactly on `t_end`. Returns [`SimError::BadInput`] on a
/// malformed request and [`SimError::NonFinite`] if the state blows up.
///
/// RK4's error is `O(h^4)` per unit time, and like every Runge–Kutta method
/// it preserves *linear* invariants (total population, total mass) to
/// round-off exactly.
pub fn simulate<S: System>(
    system: &S,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> Result<Trajectory, SimError> {
    let dim = system.dim();
    let steps = validate_run(dim, y0, t0, t_end, h)?;

    let mut traj = Trajectory {
        t: Vec::with_capacity(steps + 1),
        y: Vec::with_capacity(steps + 1),
    };
    traj.t.push(t0);
    traj.y.push(y0.to_vec());

    let mut y = y0.to_vec();
    let mut k1 = vec![0.0; dim];
    let mut k2 = vec![0.0; dim];
    let mut k3 = vec![0.0; dim];
    let mut k4 = vec![0.0; dim];
    let mut stage = vec![0.0; dim];

    let mut t = t0;
    while t < t_end
    {
        // Land exactly on t_end; the comparison above guarantees dt > 0.
        let dt = h.min(t_end - t);

        system.derivatives(t, &y, &mut k1);
        for i in 0..dim
        {
            stage[i] = y[i] + 0.5 * dt * k1[i];
        }
        system.derivatives(t + 0.5 * dt, &stage, &mut k2);
        for i in 0..dim
        {
            stage[i] = y[i] + 0.5 * dt * k2[i];
        }
        system.derivatives(t + 0.5 * dt, &stage, &mut k3);
        for i in 0..dim
        {
            stage[i] = y[i] + dt * k3[i];
        }
        system.derivatives(t + dt, &stage, &mut k4);
        for i in 0..dim
        {
            y[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        t = if dt < h { t_end } else { t + h };
        if y.iter().any(|c| !c.is_finite())
        {
            return Err(SimError::NonFinite { t });
        }
        traj.t.push(t);
        traj.y.push(y.clone());
    }
    Ok(traj)
}

// The Dormand–Prince 5(4) Butcher tableau (the method behind MATLAB's
// `ode45`): a seven-stage explicit Runge–Kutta pair whose fifth-order solution
// advances the state while the embedded fourth-order solution supplies the
// local-error estimate that drives step-size control.
//
// Nodes c_i for stages 2..=6 (c1 = 0 and c7 = 1 are handled inline).
const DP_C: [f64; 5] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0];
// Strictly-lower-triangular coupling coefficients a_ij, one row per stage
// 2..=6 (row for stage i has i-1 entries).
const DP_A2: [f64; 1] = [1.0 / 5.0];
const DP_A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];
const DP_A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
const DP_A5: [f64; 4] = [
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
const DP_A6: [f64; 5] = [
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];
// Fifth-order weights b_i for k1..k6 (b7 = 0); these advance the solution.
const DP_B5: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];
// Fourth-order weights b*_i for k1..k7, used only for the error estimate.
const DP_B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

// Write `stage[i] = y[i] + h·Σ_j a[j]·k[j][i]` for one Runge–Kutta stage.
fn stage_input(y: &[f64], h: f64, k: &[Vec<f64>], a: &[f64], stage: &mut [f64]) {
    for i in 0..y.len()
    {
        let mut acc = 0.0;
        for j in 0..a.len()
        {
            acc += a[j] * k[j][i];
        }
        stage[i] = y[i] + h * acc;
    }
}

// Automatic initial step size (Hairer, Nørsett & Wanner, *Solving Ordinary
// Differential Equations I*, §II.4): balance the scaled sizes of `y`, `f` and
// the finite-difference second derivative so the very first step is neither
// wildly too large nor needlessly tiny.
fn initial_step<S: System>(
    system: &S,
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    span: f64,
    rtol: f64,
    atol: f64,
) -> f64 {
    let dim = y0.len();
    let sc = |i: usize| atol + rtol * y0[i].abs();
    let rms = |v: &[f64]| -> f64 {
        let mut s = 0.0;
        for (i, &vi) in v.iter().enumerate()
        {
            let r = vi / sc(i);
            s += r * r;
        }
        (s / dim as f64).sqrt()
    };
    let d0 = rms(y0);
    let d1 = rms(f0);
    let h0 = if d0 < 1e-5 || d1 < 1e-5
    {
        1e-6
    }
    else
    {
        0.01 * d0 / d1
    };

    // One explicit-Euler probe to estimate the second derivative's scale.
    let mut y1 = vec![0.0; dim];
    for (y1i, (&y0i, &f0i)) in y1.iter_mut().zip(y0.iter().zip(f0.iter()))
    {
        *y1i = y0i + h0 * f0i;
    }
    let mut f1 = vec![0.0; dim];
    system.derivatives(t0 + h0, &y1, &mut f1);
    let mut d2 = 0.0;
    for (i, (&f1i, &f0i)) in f1.iter().zip(f0.iter()).enumerate()
    {
        let r = (f1i - f0i) / sc(i);
        d2 += r * r;
    }
    d2 = (d2 / dim as f64).sqrt() / h0;

    // Require the estimated local error ~ 0.01 for a method of order p = 5.
    let h1 = if d1.max(d2) <= 1e-15
    {
        (h0 * 1e-3).max(1e-6)
    }
    else
    {
        (0.01 / d1.max(d2)).powf(1.0 / 6.0)
    };
    (100.0 * h0).min(h1).min(span)
}

/// Integrate `system` from `y0` over `[t0, t_end]` with the adaptive,
/// error-controlled **Dormand–Prince 5(4)** method — the scheme behind
/// MATLAB's `ode45`.
///
/// Unlike [`simulate`], the step size is chosen automatically to keep the
/// estimated local error of each component below `atol + rtol·|y|`: the
/// integrator takes small steps through fast transients and long steps
/// through smooth stretches, so a sharp initial layer followed by a slow tail
/// is handled efficiently and accurately in a single call. The returned
/// [`Trajectory`] samples the *accepted* steps (their number and spacing are
/// solution-dependent), always starting at `t0` and ending exactly at
/// `t_end`.
///
/// `rtol` and `atol` must be finite and positive. Returns
/// [`SimError::StepUnderflow`] when the tolerance cannot be met (approaching a
/// singularity, or asked tighter than `f64` can resolve),
/// [`SimError::NonFinite`] on blow-up, and [`SimError::BadInput`] on a
/// malformed request.
///
/// # Example
///
/// ```
/// use scirust_sim::{simulate_adaptive, System};
///
/// struct Decay;
/// impl System for Decay {
///     fn dim(&self) -> usize {
///         1
///     }
///     fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
///         dydt[0] = -y[0];
///     }
/// }
///
/// let traj = simulate_adaptive(&Decay, &[1.0], 0.0, 10.0, 1e-9, 1e-12).unwrap();
/// // Ends exactly on t_end and matches e^{-10} to the requested accuracy.
/// assert_eq!(traj.last_time(), Some(10.0));
/// assert!((traj.last_state().unwrap()[0] - (-10.0f64).exp()).abs() < 1e-8);
/// ```
pub fn simulate_adaptive<S: System>(
    system: &S,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
) -> Result<Trajectory, SimError> {
    let dim = system.dim();
    if dim == 0
    {
        return Err(SimError::BadInput("system dimension is zero".to_string()));
    }
    if y0.len() != dim
    {
        return Err(SimError::DimMismatch {
            expected: dim,
            got: y0.len(),
        });
    }
    if y0.iter().any(|c| !c.is_finite())
    {
        return Err(SimError::BadInput(
            "initial state has a non-finite component".to_string(),
        ));
    }
    if !t0.is_finite() || !t_end.is_finite() || t_end <= t0
    {
        return Err(SimError::BadInput(format!(
            "time span [{t0}, {t_end}] must be finite with t_end > t0"
        )));
    }
    if !rtol.is_finite() || rtol <= 0.0 || !atol.is_finite() || atol <= 0.0
    {
        return Err(SimError::BadInput(format!(
            "tolerances rtol = {rtol}, atol = {atol} must be finite and positive"
        )));
    }

    let span = t_end - t0;
    let h_min = 16.0 * f64::EPSILON * span.max(1.0);

    let mut traj = Trajectory {
        t: vec![t0],
        y: vec![y0.to_vec()],
    };

    let mut y = y0.to_vec();
    let mut t = t0;

    // Seven stage-derivative buffers, a stage-input buffer and the candidate.
    let mut k: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; dim]);
    let mut stage = vec![0.0; dim];
    let mut y_next = vec![0.0; dim];

    // k[0] = f(t0, y0); reused across the run by the FSAL property below.
    system.derivatives(t, &y, &mut k[0]);
    let mut h = initial_step(system, t, &y, &k[0], span, rtol, atol);

    let mut rejected_last = false;
    let mut steps = 0usize;
    while t < t_end
    {
        if steps >= MAX_STEPS
        {
            return Err(SimError::BadInput(format!(
                "adaptive integration exceeded the {MAX_STEPS}-step budget"
            )));
        }
        steps += 1;

        // Never step past t_end; the last accepted step lands exactly on it.
        if t + h > t_end
        {
            h = t_end - t;
        }
        if h < h_min
        {
            return Err(SimError::StepUnderflow { t });
        }

        // k[0] already holds f(t, y). Stages 2..=6:
        stage_input(&y, h, &k[..1], &DP_A2, &mut stage);
        system.derivatives(t + DP_C[0] * h, &stage, &mut k[1]);
        stage_input(&y, h, &k[..2], &DP_A3, &mut stage);
        system.derivatives(t + DP_C[1] * h, &stage, &mut k[2]);
        stage_input(&y, h, &k[..3], &DP_A4, &mut stage);
        system.derivatives(t + DP_C[2] * h, &stage, &mut k[3]);
        stage_input(&y, h, &k[..4], &DP_A5, &mut stage);
        system.derivatives(t + DP_C[3] * h, &stage, &mut k[4]);
        stage_input(&y, h, &k[..5], &DP_A6, &mut stage);
        system.derivatives(t + DP_C[4] * h, &stage, &mut k[5]);

        // Fifth-order solution advances the state (b7 = 0).
        for i in 0..dim
        {
            let mut acc = 0.0;
            for s in 0..6
            {
                acc += DP_B5[s] * k[s][i];
            }
            y_next[i] = y[i] + h * acc;
        }
        // Seventh stage at (t + h, y_next): feeds the error estimate and, on
        // acceptance, becomes the next step's k[0] (FSAL).
        system.derivatives(t + h, &y_next, &mut k[6]);

        // Scaled error norm: RMS over components of err_i / (atol + rtol·|y|).
        let mut err_sq = 0.0;
        for i in 0..dim
        {
            let mut e = 0.0;
            for s in 0..7
            {
                let b5 = if s < 6 { DP_B5[s] } else { 0.0 };
                e += (b5 - DP_B4[s]) * k[s][i];
            }
            let sc = atol + rtol * y[i].abs().max(y_next[i].abs());
            let ratio = h * e / sc;
            err_sq += ratio * ratio;
        }
        let err_norm = (err_sq / dim as f64).sqrt();
        if !err_norm.is_finite()
        {
            return Err(SimError::NonFinite { t: t + h });
        }

        // Elementary I-controller; the estimator has order 4, hence the 1/5
        // exponent. Safety and clamp factors are the textbook defaults.
        const SAFETY: f64 = 0.9;
        const MIN_SCALE: f64 = 0.2;
        const MAX_SCALE: f64 = 5.0;
        let scale = if err_norm == 0.0
        {
            MAX_SCALE
        }
        else
        {
            (SAFETY * err_norm.powf(-0.2)).clamp(MIN_SCALE, MAX_SCALE)
        };

        if err_norm <= 1.0
        {
            t += h;
            // Erase float rounding on the final shortened step.
            if t_end - t < h_min
            {
                t = t_end;
            }
            y.copy_from_slice(&y_next);
            if y.iter().any(|c| !c.is_finite())
            {
                return Err(SimError::NonFinite { t });
            }
            traj.t.push(t);
            traj.y.push(y.clone());
            // FSAL: the seventh stage becomes the next first stage.
            k.swap(0, 6);
            // Forbid step growth immediately after a rejection (Hairer's rule)
            // to avoid a reject/accept limit cycle.
            let grow = if rejected_last { scale.min(1.0) } else { scale };
            h *= grow;
            rejected_last = false;
        }
        else
        {
            // Reject: shrink and retry; nothing is recorded and (t, y, k[0])
            // are unchanged, so the retry is exact.
            h *= scale;
            rejected_last = true;
        }
    }
    Ok(traj)
}

/// Integrate a mechanical system with the symplectic (semi-implicit) Euler
/// method: `v += h·a(t, q, v)` then `q += h·v`.
///
/// The state rows of the returned trajectory are `[q, v]` concatenated
/// (length `2·dof`). First-order accurate, but for accelerations that depend
/// only on position the method is symplectic: orbits stay closed and the
/// energy error stays bounded over arbitrarily many periods, where explicit
/// Euler spirals outward (demonstrated in the [`orbital`](crate::orbital)
/// tests).
pub fn simulate_second_order<S: SecondOrderSystem>(
    system: &S,
    q0: &[f64],
    v0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
) -> Result<Trajectory, SimError> {
    let n = system.dof();
    if v0.len() != n
    {
        return Err(SimError::DimMismatch {
            expected: n,
            got: v0.len(),
        });
    }
    let steps = validate_run(n, q0, t0, t_end, h)?;
    if v0.iter().any(|c| !c.is_finite())
    {
        return Err(SimError::BadInput(
            "initial velocity has a non-finite component".to_string(),
        ));
    }

    let mut traj = Trajectory {
        t: Vec::with_capacity(steps + 1),
        y: Vec::with_capacity(steps + 1),
    };
    let row = |q: &[f64], v: &[f64]| {
        let mut r = Vec::with_capacity(2 * n);
        r.extend_from_slice(q);
        r.extend_from_slice(v);
        r
    };
    traj.t.push(t0);
    traj.y.push(row(q0, v0));

    let mut q = q0.to_vec();
    let mut v = v0.to_vec();
    let mut acc = vec![0.0; n];
    let mut t = t0;
    while t < t_end
    {
        let dt = h.min(t_end - t);
        system.acceleration(t, &q, &v, &mut acc);
        for i in 0..n
        {
            v[i] += dt * acc[i];
        }
        for i in 0..n
        {
            q[i] += dt * v[i];
        }
        t = if dt < h { t_end } else { t + h };
        if q.iter().chain(v.iter()).any(|c| !c.is_finite())
        {
            return Err(SimError::NonFinite { t });
        }
        traj.t.push(t);
        traj.y.push(row(&q, &v));
    }
    Ok(traj)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `y' = -y`, exact solution `e^{-t}`.
    struct Decay;

    impl System for Decay {
        fn dim(&self) -> usize {
            1
        }

        fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -y[0];
        }
    }

    /// Harmonic oscillator `q'' = -q`.
    struct Harmonic;

    impl SecondOrderSystem for Harmonic {
        fn dof(&self) -> usize {
            1
        }

        fn acceleration(&self, _t: f64, q: &[f64], _v: &[f64], acc: &mut [f64]) {
            acc[0] = -q[0];
        }
    }

    #[test]
    fn rk4_matches_exponential_decay() {
        let traj = simulate(&Decay, &[1.0], 0.0, 5.0, 0.01).unwrap();
        for (t, y) in traj.t.iter().zip(traj.y.iter())
        {
            assert!((y[0] - (-t).exp()).abs() < 1e-9, "t = {t}");
        }
    }

    #[test]
    fn rk4_order_four_convergence() {
        // Halving h must shrink the endpoint error by ~2^4.
        let err = |h: f64| {
            let traj = simulate(&Decay, &[1.0], 0.0, 1.0, h).unwrap();
            (traj.last_state().unwrap()[0] - (-1.0f64).exp()).abs()
        };
        let ratio = err(0.1) / err(0.05);
        assert!(ratio > 12.0 && ratio < 20.0, "observed ratio {ratio}");
    }

    #[test]
    fn trajectory_lands_exactly_on_t_end() {
        // 0.35 is not a multiple of 0.1: the last step must be shortened.
        let traj = simulate(&Decay, &[1.0], 0.0, 0.35, 0.1).unwrap();
        assert_eq!(traj.last_time(), Some(0.35));
        assert_eq!(traj.len(), 5); // t = 0, 0.1, 0.2, 0.3, 0.35
        // RK4 with h = 0.1: local error ~ h^5/5! per step, ~1e-7 in total.
        assert!((traj.last_state().unwrap()[0] - (-0.35f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn column_extracts_a_component_and_rejects_bad_index() {
        let traj = simulate(&Decay, &[1.0], 0.0, 1.0, 0.5).unwrap();
        assert_eq!(traj.column(0).unwrap().len(), traj.len());
        assert!(traj.column(1).is_none());
    }

    #[test]
    fn bad_inputs_are_rejected_not_panicked() {
        assert!(matches!(
            simulate(&Decay, &[1.0, 2.0], 0.0, 1.0, 0.1),
            Err(SimError::DimMismatch {
                expected: 1,
                got: 2
            })
        ));
        assert!(matches!(
            simulate(&Decay, &[1.0], 0.0, 1.0, 0.0),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate(&Decay, &[1.0], 0.0, 1.0, -0.1),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate(&Decay, &[1.0], 1.0, 1.0, 0.1),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate(&Decay, &[f64::NAN], 0.0, 1.0, 0.1),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate(&Decay, &[1.0], 0.0, f64::INFINITY, 0.1),
            Err(SimError::BadInput(_))
        ));
    }

    #[test]
    fn blow_up_is_reported_as_non_finite() {
        /// `y' = y^2` from y(0) = 1 blows up at t = 1.
        struct BlowUp;
        impl System for BlowUp {
            fn dim(&self) -> usize {
                1
            }

            fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
                dydt[0] = y[0] * y[0];
            }
        }
        assert!(matches!(
            simulate(&BlowUp, &[1.0], 0.0, 2.0, 0.01),
            Err(SimError::NonFinite { .. })
        ));
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn symplectic_euler_keeps_oscillator_energy_bounded() {
        // 100 periods of the harmonic oscillator with a coarse step: the
        // energy H = (q^2 + v^2)/2 must stay within a few percent of its
        // initial value at *every* recorded step (no secular drift).
        let t_end = 100.0 * 2.0 * std::f64::consts::PI;
        let traj = simulate_second_order(&Harmonic, &[1.0], &[0.0], 0.0, t_end, 0.05).unwrap();
        for row in &traj.y
        {
            let energy = 0.5 * (row[0] * row[0] + row[1] * row[1]);
            assert!(
                (energy - 0.5).abs() < 0.05 * 0.5,
                "energy drifted to {energy}"
            );
        }
    }

    #[test]
    fn first_order_form_matches_analytic_oscillator() {
        // RK4 on the wrapped second-order system: q(t) = cos t, v(t) = -sin t.
        let sys = FirstOrderForm(&Harmonic);
        let traj = simulate(&sys, &[1.0, 0.0], 0.0, 6.0, 0.01).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - 6.0f64.cos()).abs() < 1e-8);
        assert!((last[1] + 6.0f64.sin()).abs() < 1e-8);
    }

    #[test]
    fn second_order_rejects_mismatched_velocity() {
        assert!(matches!(
            simulate_second_order(&Harmonic, &[1.0], &[0.0, 0.0], 0.0, 1.0, 0.1),
            Err(SimError::DimMismatch {
                expected: 1,
                got: 2
            })
        ));
    }

    #[test]
    fn errors_display_as_sentences() {
        let text = SimError::NonFinite { t: 2.5 }.to_string();
        assert!(text.contains("2.5"));
        let text = SimError::DimMismatch {
            expected: 3,
            got: 1,
        }
        .to_string();
        assert!(text.contains('3') && text.contains('1'));
        let text = SimError::StepUnderflow { t: 0.9 }.to_string();
        assert!(text.contains("0.9") && text.contains("underflow"));
    }

    /// `y' = -50 y`: a fast transient that decays to a flat tail — the case
    /// adaptive stepping is meant to handle in one pass.
    struct StiffDecay;

    impl System for StiffDecay {
        fn dim(&self) -> usize {
            1
        }

        fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -50.0 * y[0];
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn adaptive_matches_exponential_decay_to_tolerance() {
        let traj = simulate_adaptive(&Decay, &[1.0], 0.0, 10.0, 1e-9, 1e-12).unwrap();
        // Ends exactly on t_end and every accepted sample matches e^{-t}.
        assert_eq!(traj.last_time(), Some(10.0));
        for (t, y) in traj.t.iter().zip(traj.y.iter())
        {
            assert!((y[0] - (-t).exp()).abs() < 1e-8, "t = {t}: {}", y[0]);
        }
        // A smooth problem needs far fewer steps than the fixed-step method
        // would at comparable accuracy: fixed RK4 at 1e-9 would need h ~ 6e-3
        // (~1700 steps over this span); the adaptive solver uses a small
        // fraction of that.
        assert!(traj.len() < 300, "took {} steps", traj.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn adaptive_endpoint_error_tightens_with_tolerance() {
        let endpoint_err = |tol: f64| {
            let traj = simulate_adaptive(&Decay, &[1.0], 0.0, 5.0, tol, tol * 1e-3).unwrap();
            (traj.last_state().unwrap()[0] - (-5.0f64).exp()).abs()
        };
        let loose = endpoint_err(1e-4);
        let tight = endpoint_err(1e-9);
        assert!(tight < loose, "tight {tight} not below loose {loose}");
        assert!(tight < 1e-7, "tight endpoint error {tight}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn adaptive_grows_the_step_once_the_transient_settles() {
        // On y' = -50y the solution changes fast near t = 0 and is flat by
        // t = 5; the average accepted step in the tail must exceed that near
        // the start.
        let traj = simulate_adaptive(&StiffDecay, &[1.0], 0.0, 5.0, 1e-6, 1e-9).unwrap();
        let dt: Vec<f64> = traj.t.windows(2).map(|w| w[1] - w[0]).collect();
        let quarter = dt.len() / 4;
        let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let early = mean(&dt[..quarter]);
        let late = mean(&dt[3 * quarter..]);
        assert!(
            late > 5.0 * early,
            "no step growth: early {early}, late {late}"
        );
        // And accuracy held throughout.
        for (t, y) in traj.t.iter().zip(traj.y.iter())
        {
            assert!((y[0] - (-50.0 * t).exp()).abs() < 1e-5, "t = {t}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn adaptive_handles_the_vector_valued_oscillator() {
        // Two-component state through the second-order-to-first-order wrapper:
        // q(t) = cos t, v(t) = -sin t.
        let sys = FirstOrderForm(&Harmonic);
        let traj = simulate_adaptive(&sys, &[1.0, 0.0], 0.0, 6.0, 1e-10, 1e-12).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - 6.0f64.cos()).abs() < 1e-8, "q = {}", last[0]);
        assert!((last[1] + 6.0f64.sin()).abs() < 1e-8, "v = {}", last[1]);
    }

    #[test]
    fn adaptive_lands_exactly_on_t_end() {
        let traj = simulate_adaptive(&Decay, &[1.0], 0.0, 3.7, 1e-6, 1e-9).unwrap();
        assert_eq!(traj.last_time(), Some(3.7));
        assert!(
            traj.t.windows(2).all(|w| w[1] > w[0]),
            "times must strictly increase"
        );
    }

    #[test]
    fn adaptive_rejects_bad_requests() {
        assert!(matches!(
            simulate_adaptive(&Decay, &[1.0, 2.0], 0.0, 1.0, 1e-6, 1e-9),
            Err(SimError::DimMismatch { .. })
        ));
        assert!(matches!(
            simulate_adaptive(&Decay, &[1.0], 0.0, 1.0, 0.0, 1e-9),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate_adaptive(&Decay, &[1.0], 0.0, 1.0, 1e-6, -1e-9),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate_adaptive(&Decay, &[1.0], 1.0, 1.0, 1e-6, 1e-9),
            Err(SimError::BadInput(_))
        ));
        assert!(matches!(
            simulate_adaptive(&Decay, &[f64::NAN], 0.0, 1.0, 1e-6, 1e-9),
            Err(SimError::BadInput(_))
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn adaptive_reports_a_finite_time_singularity() {
        // y' = y^2 from y(0) = 1 blows up at t = 1; integrating to t = 2 must
        // fail (step underflow as the step collapses, or non-finite state) —
        // never a silent bogus success.
        struct BlowUp;
        impl System for BlowUp {
            fn dim(&self) -> usize {
                1
            }

            fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
                dydt[0] = y[0] * y[0];
            }
        }
        let result = simulate_adaptive(&BlowUp, &[1.0], 0.0, 2.0, 1e-8, 1e-10);
        assert!(matches!(
            result,
            Err(SimError::StepUnderflow { .. }) | Err(SimError::NonFinite { .. })
        ));
    }
}
