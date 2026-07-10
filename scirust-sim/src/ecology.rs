//! Population-dynamics models: Lotka–Volterra predator–prey (with its exact
//! first integral as the oracle) and logistic growth (closed-form solution).

use crate::engine::{SimError, System};

fn check_rate(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and positive"
        )))
    }
}

/// The Lotka–Volterra predator–prey model, state `y = [x, y]` with `x` the
/// prey and `y` the predator population:
///
/// `x' = α·x - β·x·y`, `y' = δ·x·y - γ·y`.
///
/// Trajectories are closed curves around the coexistence equilibrium
/// `(γ/δ, α/β)`; the quantity returned by
/// [`first_integral`](LotkaVolterra::first_integral) is exactly conserved
/// along them, which is what the tests check.
#[derive(Debug, Clone, PartialEq)]
pub struct LotkaVolterra {
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
}

impl LotkaVolterra {
    /// Create the model; all four rates must be finite and positive.
    pub fn new(alpha: f64, beta: f64, delta: f64, gamma: f64) -> Result<Self, SimError> {
        check_rate("alpha", alpha)?;
        check_rate("beta", beta)?;
        check_rate("delta", delta)?;
        check_rate("gamma", gamma)?;
        Ok(LotkaVolterra {
            alpha,
            beta,
            delta,
            gamma,
        })
    }

    /// The coexistence equilibrium `(x*, y*) = (γ/δ, α/β)`.
    pub fn equilibrium(&self) -> (f64, f64) {
        (self.gamma / self.delta, self.alpha / self.beta)
    }

    /// The conserved quantity `V = δ·x - γ·ln x + β·y - α·ln y`, or `None`
    /// when the state does not have length 2 or a population is not strictly
    /// positive (the logarithm needs the open positive quadrant).
    pub fn first_integral(&self, state: &[f64]) -> Option<f64> {
        let [x, y] = *state
        else
        {
            return None;
        };
        if x <= 0.0 || y <= 0.0
        {
            return None;
        }
        Some(self.delta * x - self.gamma * x.ln() + self.beta * y - self.alpha * y.ln())
    }
}

impl System for LotkaVolterra {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.alpha * y[0] - self.beta * y[0] * y[1];
        dydt[1] = self.delta * y[0] * y[1] - self.gamma * y[1];
    }
}

/// Logistic growth `x' = r·x·(1 - x/K)`, state `y = [x]`, with the
/// closed-form solution `x(t) = K / (1 + (K/x₀ - 1)·e^{-rt})`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogisticGrowth {
    rate: f64,
    capacity: f64,
}

impl LogisticGrowth {
    /// Create the model; growth `rate` and carrying `capacity` must be finite
    /// and positive.
    pub fn new(rate: f64, capacity: f64) -> Result<Self, SimError> {
        check_rate("rate", rate)?;
        check_rate("capacity", capacity)?;
        Ok(LogisticGrowth { rate, capacity })
    }

    /// The closed-form solution at time `t` from `x0`, or `None` when `x0`
    /// is not finite and positive.
    pub fn exact(&self, x0: f64, t: f64) -> Option<f64> {
        if !x0.is_finite() || x0 <= 0.0
        {
            return None;
        }
        Some(self.capacity / (1.0 + (self.capacity / x0 - 1.0) * (-self.rate * t).exp()))
    }
}

impl System for LogisticGrowth {
    fn dim(&self) -> usize {
        1
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = self.rate * y[0] * (1.0 - y[0] / self.capacity);
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
    fn lotka_volterra_conserves_its_first_integral() {
        let sys = LotkaVolterra::new(1.0, 0.5, 0.2, 0.8).unwrap();
        let traj = simulate(&sys, &[3.0, 1.0], 0.0, 30.0, 0.001).unwrap();
        let v0 = sys.first_integral(&traj.y[0]).unwrap();
        for row in &traj.y
        {
            let v = sys.first_integral(row).unwrap();
            assert!(
                (v - v0).abs() < 1e-6 * v0.abs(),
                "V drifted to {v} from {v0}"
            );
        }
    }

    #[test]
    fn equilibrium_is_a_fixed_point() {
        let sys = LotkaVolterra::new(1.0, 0.5, 0.2, 0.8).unwrap();
        let (x_eq, y_eq) = sys.equilibrium();
        let mut dydt = [0.0; 2];
        sys.derivatives(0.0, &[x_eq, y_eq], &mut dydt);
        assert!(dydt[0].abs() < 1e-14 && dydt[1].abs() < 1e-14);
        // Simulating from the equilibrium stays there.
        let traj = simulate(&sys, &[x_eq, y_eq], 0.0, 10.0, 0.01).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - x_eq).abs() < 1e-12 && (last[1] - y_eq).abs() < 1e-12);
    }

    #[test]
    // Ignored under Miri: a many-step accuracy/statistics run that is
    // minutes-slow under the interpreter and exercises no surface beyond
    // what the fast Miri-checked tests cover. Native Build & Test jobs
    // enforce it.
    #[cfg_attr(miri, ignore)]
    fn small_oscillations_return_after_the_linearized_period() {
        // Near the equilibrium the cycle period tends to 2π/√(α·γ); with
        // α = γ = 1 that is 2π. A 5% perturbation must come back close to
        // its starting point after one period.
        let sys = LotkaVolterra::new(1.0, 0.5, 0.5, 1.0).unwrap();
        let (x_eq, y_eq) = sys.equilibrium();
        let y0 = [x_eq * 1.05, y_eq];
        let period = 2.0 * std::f64::consts::PI;
        let traj = simulate(&sys, &y0, 0.0, period, period / 8000.0).unwrap();
        let last = traj.last_state().unwrap();
        let distance = ((last[0] - y0[0]).powi(2) + (last[1] - y0[1]).powi(2)).sqrt() / x_eq;
        assert!(
            distance < 5e-3,
            "did not return: relative distance {distance}"
        );
    }

    #[test]
    fn logistic_growth_matches_the_closed_form() {
        let sys = LogisticGrowth::new(0.7, 10.0).unwrap();
        let traj = simulate(&sys, &[0.5], 0.0, 15.0, 0.01).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = sys.exact(0.5, *t).unwrap();
            // State scale is K = 10; RK4 at h = 0.01 keeps ~1e-7 absolute.
            assert!(
                (row[0] - exact).abs() < 1e-6,
                "t = {t}: {} vs {exact}",
                row[0]
            );
        }
        // Approaching the carrying capacity from above also converges to K
        // (at t = 30 the exact solution is within 4e-9 of K).
        let from_above = simulate(&sys, &[20.0], 0.0, 30.0, 0.01).unwrap();
        assert!((from_above.last_state().unwrap()[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn constructors_and_helpers_reject_bad_inputs() {
        assert!(LotkaVolterra::new(0.0, 0.5, 0.2, 0.8).is_err());
        assert!(LotkaVolterra::new(1.0, 0.5, 0.2, f64::NAN).is_err());
        assert!(LogisticGrowth::new(0.7, 0.0).is_err());
        let sys = LotkaVolterra::new(1.0, 0.5, 0.2, 0.8).unwrap();
        assert!(sys.first_integral(&[1.0]).is_none());
        assert!(sys.first_integral(&[0.0, 1.0]).is_none());
        let logistic = LogisticGrowth::new(0.7, 10.0).unwrap();
        assert!(logistic.exact(-1.0, 1.0).is_none());
    }
}
