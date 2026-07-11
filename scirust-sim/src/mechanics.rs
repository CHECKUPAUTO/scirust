//! Classical mechanics models: spring–mass–damper, pendulum, projectile with
//! linear drag, and the chaotic double pendulum. Each is validated against an
//! analytic solution or an energy argument (and, for the double pendulum,
//! sensitive dependence on initial conditions) in the tests.

use crate::engine::{SimError, System};

fn check_finite(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite"
        )))
    }
}

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
    check_finite(name, value)?;
    if value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be positive"
        )))
    }
}

fn check_non_negative(name: &str, value: f64) -> Result<(), SimError> {
    check_finite(name, value)?;
    if value >= 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be non-negative"
        )))
    }
}

/// A mass on a linear spring with viscous damping:
/// `m·x'' + c·x' + k·x = 0`, state `y = [x, v]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpringMassDamper {
    mass: f64,
    damping: f64,
    stiffness: f64,
}

impl SpringMassDamper {
    /// Create the model. `mass` must be positive; `damping` and `stiffness`
    /// must be non-negative; all must be finite.
    pub fn new(mass: f64, damping: f64, stiffness: f64) -> Result<Self, SimError> {
        check_positive("mass", mass)?;
        check_non_negative("damping", damping)?;
        check_non_negative("stiffness", stiffness)?;
        Ok(SpringMassDamper {
            mass,
            damping,
            stiffness,
        })
    }

    /// Total mechanical energy `½·m·v² + ½·k·x²` of a state `[x, v]`, or
    /// `None` when the state does not have length 2. Conserved when the
    /// damping is zero.
    pub fn energy(&self, state: &[f64]) -> Option<f64> {
        let [x, v] = *state
        else
        {
            return None;
        };
        Some(0.5 * self.mass * v * v + 0.5 * self.stiffness * x * x)
    }
}

impl System for SpringMassDamper {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] = -(self.damping * y[1] + self.stiffness * y[0]) / self.mass;
    }
}

/// A rigid pendulum of length `L` under gravity `g` with viscous damping:
/// `θ'' = -(g/L)·sin θ - c·θ'`, state `y = [θ, ω]`.
///
/// The full nonlinear equation — no small-angle linearization — so it also
/// covers large-amplitude swings.
#[derive(Debug, Clone, PartialEq)]
pub struct Pendulum {
    length: f64,
    gravity: f64,
    damping: f64,
}

impl Pendulum {
    /// Create the model. `length` and `gravity` must be positive, `damping`
    /// non-negative, all finite.
    pub fn new(length: f64, gravity: f64, damping: f64) -> Result<Self, SimError> {
        check_positive("length", length)?;
        check_positive("gravity", gravity)?;
        check_non_negative("damping", damping)?;
        Ok(Pendulum {
            length,
            gravity,
            damping,
        })
    }

    /// Period of small oscillations, `2π·√(L/g)`.
    pub fn small_angle_period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.length / self.gravity).sqrt()
    }

    /// Mechanical energy per unit mass, `½·L²·ω² + g·L·(1 - cos θ)`, of a
    /// state `[θ, ω]`, or `None` when the state does not have length 2.
    /// Conserved when the damping is zero.
    pub fn energy(&self, state: &[f64]) -> Option<f64> {
        let [theta, omega] = *state
        else
        {
            return None;
        };
        Some(
            0.5 * self.length * self.length * omega * omega
                + self.gravity * self.length * (1.0 - theta.cos()),
        )
    }
}

impl System for Pendulum {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] = -(self.gravity / self.length) * y[0].sin() - self.damping * y[1];
    }
}

/// A point projectile under gravity with drag proportional to velocity
/// (Stokes regime): `v' = -g·ĵ - k·v`, state `y = [x, y, vx, vy]`.
///
/// Linear drag keeps the model analytically solvable, which is what the
/// oracle test exercises; `drag = 0` recovers the textbook parabola.
#[derive(Debug, Clone, PartialEq)]
pub struct Projectile {
    gravity: f64,
    drag: f64,
}

impl Projectile {
    /// Create the model. `gravity` must be positive and `drag` (per unit
    /// mass, unit 1/s) non-negative, both finite.
    pub fn new(gravity: f64, drag: f64) -> Result<Self, SimError> {
        check_positive("gravity", gravity)?;
        check_non_negative("drag", drag)?;
        Ok(Projectile { gravity, drag })
    }
}

impl System for Projectile {
    fn dim(&self) -> usize {
        4
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -self.drag * y[2];
        dydt[3] = -self.gravity - self.drag * y[3];
    }
}

/// A planar double pendulum: a bob (`m1`) on a rigid rod (`l1`) hangs from a
/// fixed pivot, and a second bob (`m2`) hangs from the first on its own rod
/// (`l2`). Angles `θ1`, `θ2` are measured from the downward vertical. State
/// `y = [θ1, ω1, θ2, ω2]`.
///
/// This is the textbook example of *deterministic chaos*: the motion is fully
/// determined by the initial conditions, yet two nearby starts diverge
/// exponentially (a positive Lyapunov exponent). Undamped, the total
/// mechanical energy is exactly conserved by the true dynamics — the oracle
/// the tests check. The accelerations are the standard Lagrangian form.
#[derive(Debug, Clone, PartialEq)]
pub struct DoublePendulum {
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
    gravity: f64,
}

impl DoublePendulum {
    /// Create the model. Both masses, both lengths and `gravity` must be
    /// finite and positive.
    pub fn new(m1: f64, m2: f64, l1: f64, l2: f64, gravity: f64) -> Result<Self, SimError> {
        check_positive("m1", m1)?;
        check_positive("m2", m2)?;
        check_positive("l1", l1)?;
        check_positive("l2", l2)?;
        check_positive("gravity", gravity)?;
        Ok(DoublePendulum {
            m1,
            m2,
            l1,
            l2,
            gravity,
        })
    }

    /// Total mechanical energy (kinetic + gravitational potential, heights
    /// measured downward from the pivot) of a state `[θ1, ω1, θ2, ω2]`, or
    /// `None` when the state does not have length 4. Conserved by the exact
    /// undamped dynamics.
    pub fn energy(&self, state: &[f64]) -> Option<f64> {
        let [t1, w1, t2, w2] = *state
        else
        {
            return None;
        };
        let (m1, m2, l1, l2, g) = (self.m1, self.m2, self.l1, self.l2, self.gravity);
        // Bob-2's velocity is the vector sum of both rods' contributions, hence
        // the cross term in cos(θ1 − θ2).
        let ke = 0.5 * m1 * l1 * l1 * w1 * w1
            + 0.5
                * m2
                * (l1 * l1 * w1 * w1
                    + l2 * l2 * w2 * w2
                    + 2.0 * l1 * l2 * w1 * w2 * (t1 - t2).cos());
        let pe = -(m1 + m2) * g * l1 * t1.cos() - m2 * g * l2 * t2.cos();
        Some(ke + pe)
    }
}

impl System for DoublePendulum {
    fn dim(&self) -> usize {
        4
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let (t1, w1, t2, w2) = (y[0], y[1], y[2], y[3]);
        let (m1, m2, l1, l2, g) = (self.m1, self.m2, self.l1, self.l2, self.gravity);
        let delta = t1 - t2;
        let (sd, cd) = (delta.sin(), delta.cos());
        // Shared denominator: 2·m1 + m2 − m2·cos(2Δ) = 2·(m1 + m2·sin²Δ) > 0.
        let den = 2.0 * m1 + m2 - m2 * (2.0 * delta).cos();

        dydt[0] = w1;
        dydt[2] = w2;
        dydt[1] = (-g * (2.0 * m1 + m2) * t1.sin()
            - m2 * g * (t1 - 2.0 * t2).sin()
            - 2.0 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd))
            / (l1 * den);
        dydt[3] = (2.0
            * sd
            * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * t1.cos() + w2 * w2 * l2 * m2 * cd))
            / (l2 * den);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{simulate, simulate_adaptive};

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn underdamped_spring_matches_analytic_solution() {
        // m = 1, c = 0.4, k = 4: ωn = 2, ζ = 0.1 (underdamped).
        let sys = SpringMassDamper::new(1.0, 0.4, 4.0).unwrap();
        let (x0, v0) = (1.0, 0.0);
        let (omega_n, zeta) = (2.0f64, 0.1f64);
        let omega_d = omega_n * (1.0 - zeta * zeta).sqrt();
        let traj = simulate(&sys, &[x0, v0], 0.0, 10.0, 0.001).unwrap();
        for (t, y) in traj.t.iter().zip(traj.y.iter())
        {
            let envelope = (-zeta * omega_n * t).exp();
            let exact = envelope
                * (x0 * (omega_d * t).cos()
                    + (v0 + zeta * omega_n * x0) / omega_d * (omega_d * t).sin());
            assert!((y[0] - exact).abs() < 1e-6, "t = {t}: {} vs {exact}", y[0]);
        }
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn undamped_spring_conserves_energy() {
        let sys = SpringMassDamper::new(2.0, 0.0, 8.0).unwrap();
        let traj = simulate(&sys, &[0.5, 0.0], 0.0, 10.0, 0.001).unwrap();
        let e0 = sys.energy(&traj.y[0]).unwrap();
        for row in &traj.y
        {
            let e = sys.energy(row).unwrap();
            assert!(
                (e - e0).abs() < 1e-9 * e0,
                "energy drifted to {e} from {e0}"
            );
        }
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn pendulum_small_angle_period_is_two_pi_sqrt_l_over_g() {
        let sys = Pendulum::new(1.0, 9.81, 0.0).unwrap();
        let theta0 = 1e-3;
        let period = sys.small_angle_period();
        // After exactly one linear period a tiny-amplitude swing is back at
        // its starting angle (the nonlinear period correction is O(θ0²)).
        let traj = simulate(&sys, &[theta0, 0.0], 0.0, period, period / 4000.0).unwrap();
        let last = traj.last_state().unwrap();
        assert!(
            (last[0] - theta0).abs() < 1e-6 * theta0.max(1e-6),
            "θ(T) = {}",
            last[0]
        );
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn undamped_pendulum_conserves_energy_at_large_amplitude() {
        let sys = Pendulum::new(0.8, 9.81, 0.0).unwrap();
        let traj = simulate(&sys, &[2.0, 0.0], 0.0, 10.0, 0.001).unwrap();
        let e0 = sys.energy(&traj.y[0]).unwrap();
        for row in &traj.y
        {
            let e = sys.energy(row).unwrap();
            assert!(
                (e - e0).abs() < 1e-7 * e0,
                "energy drifted to {e} from {e0}"
            );
        }
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn damped_pendulum_amplitude_decays() {
        let sys = Pendulum::new(1.0, 9.81, 0.5).unwrap();
        let traj = simulate(&sys, &[1.0, 0.0], 0.0, 20.0, 0.001).unwrap();
        let quarter = traj.len() / 4;
        let max_abs = |rows: &[Vec<f64>]| rows.iter().map(|r| r[0].abs()).fold(0.0, f64::max);
        let early = max_abs(&traj.y[..quarter]);
        let late = max_abs(&traj.y[3 * quarter..]);
        assert!(late < 0.1 * early, "no decay: early {early}, late {late}");
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn projectile_with_linear_drag_matches_analytic_solution() {
        let (g, k) = (9.81, 0.3);
        let sys = Projectile::new(g, k).unwrap();
        let (vx0, vy0) = (12.0, 8.0);
        let traj = simulate(&sys, &[0.0, 0.0, vx0, vy0], 0.0, 1.5, 0.001).unwrap();
        let (t, y) = (traj.last_time().unwrap(), traj.last_state().unwrap());
        let decay = (-k * t).exp();
        let exact_x = vx0 / k * (1.0 - decay);
        let exact_y = (vy0 + g / k) / k * (1.0 - decay) - g * t / k;
        assert!((y[0] - exact_x).abs() < 1e-8, "x: {} vs {exact_x}", y[0]);
        assert!((y[1] - exact_y).abs() < 1e-8, "y: {} vs {exact_y}", y[1]);
        assert!((y[2] - vx0 * decay).abs() < 1e-8);
        assert!((y[3] - ((vy0 + g / k) * decay - g / k)).abs() < 1e-8);
    }

    #[test]
    fn drag_shortens_the_horizontal_distance() {
        let no_drag = Projectile::new(9.81, 0.0).unwrap();
        let with_drag = Projectile::new(9.81, 0.5).unwrap();
        let y0 = [0.0, 0.0, 10.0, 10.0];
        let a = simulate(&no_drag, &y0, 0.0, 1.0, 0.001).unwrap();
        let b = simulate(&with_drag, &y0, 0.0, 1.0, 0.001).unwrap();
        assert!(b.last_state().unwrap()[0] < a.last_state().unwrap()[0]);
    }

    #[test]
    fn constructors_reject_bad_parameters() {
        assert!(SpringMassDamper::new(0.0, 0.1, 1.0).is_err());
        assert!(SpringMassDamper::new(1.0, -0.1, 1.0).is_err());
        assert!(SpringMassDamper::new(f64::NAN, 0.1, 1.0).is_err());
        assert!(Pendulum::new(-1.0, 9.81, 0.0).is_err());
        assert!(Pendulum::new(1.0, 0.0, 0.0).is_err());
        assert!(Projectile::new(9.81, -0.5).is_err());
        assert!(Projectile::new(f64::INFINITY, 0.0).is_err());
        // Energy helpers reject malformed states instead of panicking.
        let spring = SpringMassDamper::new(1.0, 0.0, 1.0).unwrap();
        assert!(spring.energy(&[1.0]).is_none());
        let pendulum = Pendulum::new(1.0, 9.81, 0.0).unwrap();
        assert!(pendulum.energy(&[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    // Ignored under Miri: a long, transcendental-heavy chaotic run, minutes
    // slow under the interpreter and covered by the native Build & Test jobs.
    #[cfg_attr(miri, ignore)]
    fn double_pendulum_conserves_energy() {
        // A high-energy (chaotic) start: both rods raised well above horizontal.
        let sys = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81).unwrap();
        let y0 = [2.2, 0.0, 2.0, 0.0];
        // Energy is a first integral of the flow; a tight adaptive tolerance
        // keeps the integrated energy on it despite the chaotic trajectory.
        let traj = simulate_adaptive(&sys, &y0, 0.0, 15.0, 1e-11, 1e-12).unwrap();
        let e0 = sys.energy(&traj.y[0]).unwrap();
        for row in &traj.y
        {
            let e = sys.energy(row).unwrap();
            assert!(
                (e - e0).abs() < 1e-6 * e0.abs(),
                "energy drifted to {e} from {e0}"
            );
        }
    }

    #[test]
    // Ignored under Miri: see `double_pendulum_conserves_energy`.
    #[cfg_attr(miri, ignore)]
    fn double_pendulum_shows_sensitive_dependence_on_initial_conditions() {
        // Two starts differing by 1e-8 in θ1 only, integrated identically
        // (same fixed-step RK4), so any divergence is physical, not numerical.
        let sys = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81).unwrap();
        let a = simulate(&sys, &[2.0, 0.0, 2.0, 0.0], 0.0, 20.0, 1e-4).unwrap();
        let b = simulate(&sys, &[2.0 + 1e-8, 0.0, 2.0, 0.0], 0.0, 20.0, 1e-4).unwrap();

        let sep = |i: usize| -> f64 {
            let (u, v) = (&a.y[i], &b.y[i]);
            u.iter()
                .zip(v)
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt()
        };
        // Early separation is ~1e-8; by the end the trajectories are O(1)
        // apart — an enormous amplification only a chaotic system produces.
        let early = sep(a.len() / 20);
        let late = sep(a.len() - 1);
        assert!(early < 1e-5, "unexpectedly large early separation {early}");
        assert!(late > 0.1, "no chaotic divergence: late separation {late}");
        assert!(late > 1e6 * early, "separation grew only {}×", late / early);
    }

    #[test]
    fn double_pendulum_rejects_bad_parameters() {
        assert!(DoublePendulum::new(0.0, 1.0, 1.0, 1.0, 9.81).is_err());
        assert!(DoublePendulum::new(1.0, -1.0, 1.0, 1.0, 9.81).is_err());
        assert!(DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 0.0).is_err());
        assert!(DoublePendulum::new(1.0, 1.0, f64::NAN, 1.0, 9.81).is_err());
        let dp = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81).unwrap();
        assert!(dp.energy(&[1.0, 2.0, 3.0]).is_none());
    }
}
