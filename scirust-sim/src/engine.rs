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
    }
}
