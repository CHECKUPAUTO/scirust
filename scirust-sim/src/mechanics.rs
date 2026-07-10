//! Classical mechanics models: spring–mass–damper, pendulum, projectile with
//! linear drag. Each is validated against an analytic solution or an energy
//! argument in the tests.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

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
}
