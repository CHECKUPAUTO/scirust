//! Orbital dynamics: the planar two-body (Kepler) problem, with conservation
//! oracles and a demonstration that the symplectic integrator keeps orbits
//! closed where explicit Euler spirals outward.

use crate::engine::{SecondOrderSystem, SimError};

/// Planar two-body problem around a fixed primary of gravitational parameter
/// `μ = G·M`: `q'' = -μ·q / |q|³` with `q = [x, y]`.
///
/// Implements [`SecondOrderSystem`], so it can be integrated either
/// symplectically ([`simulate_second_order`](crate::simulate_second_order),
/// bounded energy error over many orbits) or with RK4 through
/// [`FirstOrderForm`](crate::engine::FirstOrderForm) (higher short-horizon
/// accuracy).
#[derive(Debug, Clone, PartialEq)]
pub struct TwoBody {
    mu: f64,
}

impl TwoBody {
    /// Create the model; `mu` must be finite and positive.
    pub fn new(mu: f64) -> Result<Self, SimError> {
        if !mu.is_finite() || mu <= 0.0
        {
            return Err(SimError::BadInput(format!(
                "mu = {mu} must be finite and positive"
            )));
        }
        Ok(TwoBody { mu })
    }

    /// Speed of a circular orbit of radius `r`, `√(μ/r)`, or `None` when `r`
    /// is not finite and positive.
    pub fn circular_velocity(&self, r: f64) -> Option<f64> {
        if !r.is_finite() || r <= 0.0
        {
            return None;
        }
        Some((self.mu / r).sqrt())
    }

    /// Period of a circular orbit of radius `r`, `2π·√(r³/μ)` (Kepler III),
    /// or `None` when `r` is not finite and positive.
    pub fn circular_period(&self, r: f64) -> Option<f64> {
        if !r.is_finite() || r <= 0.0
        {
            return None;
        }
        Some(2.0 * std::f64::consts::PI * (r * r * r / self.mu).sqrt())
    }

    /// Specific orbital energy `v²/2 - μ/r` of a state `[x, y]`, `[vx, vy]`,
    /// or `None` when either slice does not have length 2 or `r = 0`.
    pub fn energy(&self, q: &[f64], v: &[f64]) -> Option<f64> {
        let ([x, y], [vx, vy]) = (<[f64; 2]>::try_from(q).ok()?, <[f64; 2]>::try_from(v).ok()?);
        let r = (x * x + y * y).sqrt();
        if r == 0.0
        {
            return None;
        }
        Some(0.5 * (vx * vx + vy * vy) - self.mu / r)
    }

    /// Specific angular momentum `x·vy - y·vx` (out-of-plane component), or
    /// `None` when either slice does not have length 2.
    pub fn angular_momentum(&self, q: &[f64], v: &[f64]) -> Option<f64> {
        let ([x, y], [vx, vy]) = (<[f64; 2]>::try_from(q).ok()?, <[f64; 2]>::try_from(v).ok()?);
        Some(x * vy - y * vx)
    }
}

impl SecondOrderSystem for TwoBody {
    fn dof(&self) -> usize {
        2
    }

    fn acceleration(&self, _t: f64, q: &[f64], _v: &[f64], acc: &mut [f64]) {
        let r2 = q[0] * q[0] + q[1] * q[1];
        let r3 = r2 * r2.sqrt();
        acc[0] = -self.mu * q[0] / r3;
        acc[1] = -self.mu * q[1] / r3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FirstOrderForm, simulate, simulate_second_order};

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn circular_orbit_closes_after_one_kepler_period() {
        let sys = TwoBody::new(1.0).unwrap();
        let r = 1.0;
        let v = sys.circular_velocity(r).unwrap();
        let period = sys.circular_period(r).unwrap();
        let traj = simulate(
            &FirstOrderForm(&sys),
            &[r, 0.0, 0.0, v],
            0.0,
            period,
            period / 4000.0,
        )
        .unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - r).abs() < 1e-6, "x = {}", last[0]);
        assert!(last[1].abs() < 1e-6, "y = {}", last[1]);
        assert!(last[2].abs() < 1e-6 && (last[3] - v).abs() < 1e-6);
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn eccentric_orbit_conserves_energy_and_angular_momentum() {
        let sys = TwoBody::new(1.0).unwrap();
        // Perigee r = 1 with 1.2× circular speed: eccentricity e = 0.44.
        let y0 = [1.0, 0.0, 0.0, 1.2];
        let e0 = sys.energy(&y0[..2], &y0[2..]).unwrap();
        let h0 = sys.angular_momentum(&y0[..2], &y0[2..]).unwrap();
        let traj = simulate(&FirstOrderForm(&sys), &y0, 0.0, 20.0, 0.002).unwrap();
        for row in &traj.y
        {
            let e = sys.energy(&row[..2], &row[2..]).unwrap();
            let h = sys.angular_momentum(&row[..2], &row[2..]).unwrap();
            assert!((e - e0).abs() < 1e-9 * e0.abs(), "energy {e} vs {e0}");
            assert!((h - h0).abs() < 1e-9 * h0.abs(), "momentum {h} vs {h0}");
        }
    }

    #[test]
    fn symplectic_euler_stays_bounded_where_explicit_euler_spirals_out() {
        let sys = TwoBody::new(1.0).unwrap();
        let r0 = 1.0;
        let v0 = sys.circular_velocity(r0).unwrap();
        let period = sys.circular_period(r0).unwrap();
        let (t_end, h) = (10.0 * period, period / 200.0);

        // Symplectic (semi-implicit) Euler through the engine.
        let traj = simulate_second_order(&sys, &[r0, 0.0], &[0.0, v0], 0.0, t_end, h).unwrap();
        let sympl_radius = |row: &Vec<f64>| (row[0] * row[0] + row[1] * row[1]).sqrt();
        let max_sympl = traj.y.iter().map(sympl_radius).fold(0.0, f64::max);

        // Explicit Euler reference, same step: v += h·a(q); q += h·v_old.
        let (mut q, mut v) = ([r0, 0.0], [0.0, v0]);
        let mut acc = [0.0; 2];
        let steps = (t_end / h).round() as usize;
        let mut max_explicit = 0.0f64;
        for _ in 0..steps
        {
            sys.acceleration(0.0, &q, &v, &mut acc);
            q = [q[0] + h * v[0], q[1] + h * v[1]];
            v = [v[0] + h * acc[0], v[1] + h * acc[1]];
            max_explicit = max_explicit.max((q[0] * q[0] + q[1] * q[1]).sqrt());
        }

        // Ten orbits at 200 steps/orbit: the symplectic orbit stays near
        // r = 1 while the explicit one has visibly spiralled outward.
        assert!(max_sympl < 1.05, "symplectic radius grew to {max_sympl}");
        assert!(
            max_explicit > 1.3,
            "explicit Euler only reached {max_explicit}"
        );
    }

    #[test]
    fn helpers_reject_malformed_inputs() {
        assert!(TwoBody::new(0.0).is_err());
        assert!(TwoBody::new(f64::NAN).is_err());
        let sys = TwoBody::new(1.0).unwrap();
        assert!(sys.circular_velocity(-1.0).is_none());
        assert!(sys.circular_period(0.0).is_none());
        assert!(sys.energy(&[1.0], &[0.0, 1.0]).is_none());
        assert!(sys.energy(&[0.0, 0.0], &[0.0, 1.0]).is_none()); // r = 0
        assert!(sys.angular_momentum(&[1.0, 0.0], &[0.0]).is_none());
    }
}
