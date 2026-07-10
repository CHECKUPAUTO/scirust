//! Thermal models: Newton cooling (closed form) and 1-D transient heat
//! conduction by the method of lines, validated against the *discrete*
//! eigenmode decay rate and the steady-state linear profile.

use crate::engine::{SimError, System};

/// Newton's law of cooling `T' = -k·(T - T_env)`, state `y = [T]`, with the
/// closed form `T(t) = T_env + (T₀ - T_env)·e^{-kt}`.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonCooling {
    rate: f64,
    ambient: f64,
}

impl NewtonCooling {
    /// Create the model; `rate` must be finite and positive, `ambient`
    /// finite.
    pub fn new(rate: f64, ambient: f64) -> Result<Self, SimError> {
        if !rate.is_finite() || rate <= 0.0
        {
            return Err(SimError::BadInput(format!(
                "rate = {rate} must be finite and positive"
            )));
        }
        if !ambient.is_finite()
        {
            return Err(SimError::BadInput(format!(
                "ambient = {ambient} must be finite"
            )));
        }
        Ok(NewtonCooling { rate, ambient })
    }

    /// The closed-form temperature at time `t` from `t0`, or `None` when
    /// `t0` is not finite.
    pub fn exact(&self, t0: f64, t: f64) -> Option<f64> {
        if !t0.is_finite()
        {
            return None;
        }
        Some(self.ambient + (t0 - self.ambient) * (-self.rate * t).exp())
    }
}

impl System for NewtonCooling {
    fn dim(&self) -> usize {
        1
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.rate * (y[0] - self.ambient);
    }
}

/// 1-D transient heat conduction in a rod, discretized by the method of
/// lines with `n` interior nodes of spacing `dx` and fixed (Dirichlet)
/// boundary temperatures:
///
/// `Tᵢ' = α·(Tᵢ₋₁ - 2·Tᵢ + Tᵢ₊₁)/dx²`, with `T₀ = T_left`, `Tₙ₊₁ = T_right`.
///
/// The state vector holds the `n` interior temperatures. Being an explicit
/// spatial discretization, RK4 needs `h ≲ 0.7·dx²/α` for stability; the
/// semi-discrete system itself is exact about two things the tests exploit:
/// discrete sine modes decay at the discrete eigenvalue rate
/// `λ = (2α/dx²)·(1 - cos(kπ/(n+1)))`, and the steady state is the linear
/// profile between the boundary temperatures.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatRod1d {
    diffusivity: f64,
    dx: f64,
    nodes: usize,
    t_left: f64,
    t_right: f64,
}

impl HeatRod1d {
    /// Create the model. `diffusivity` and `dx` must be finite and positive,
    /// `nodes` at least 1, and both boundary temperatures finite.
    pub fn new(
        diffusivity: f64,
        dx: f64,
        nodes: usize,
        t_left: f64,
        t_right: f64,
    ) -> Result<Self, SimError> {
        if !diffusivity.is_finite() || diffusivity <= 0.0
        {
            return Err(SimError::BadInput(format!(
                "diffusivity = {diffusivity} must be finite and positive"
            )));
        }
        if !dx.is_finite() || dx <= 0.0
        {
            return Err(SimError::BadInput(format!(
                "dx = {dx} must be finite and positive"
            )));
        }
        if nodes == 0
        {
            return Err(SimError::BadInput(
                "at least one interior node is required".to_string(),
            ));
        }
        if !t_left.is_finite() || !t_right.is_finite()
        {
            return Err(SimError::BadInput(
                "boundary temperatures must be finite".to_string(),
            ));
        }
        Ok(HeatRod1d {
            diffusivity,
            dx,
            nodes,
            t_left,
            t_right,
        })
    }

    /// The steady-state temperature at interior node `i` (0-based): the
    /// linear profile between the boundary temperatures, or `None` when `i`
    /// is out of range.
    pub fn steady_state(&self, i: usize) -> Option<f64> {
        if i >= self.nodes
        {
            return None;
        }
        let fraction = (i + 1) as f64 / (self.nodes + 1) as f64;
        Some(self.t_left + (self.t_right - self.t_left) * fraction)
    }

    /// The decay rate of the `k`-th discrete sine mode (1-based),
    /// `λₖ = (2α/dx²)·(1 - cos(kπ/(n+1)))`, or `None` when `k` is 0 or
    /// above `n`.
    pub fn mode_decay_rate(&self, k: usize) -> Option<f64> {
        if k == 0 || k > self.nodes
        {
            return None;
        }
        let angle = k as f64 * std::f64::consts::PI / (self.nodes + 1) as f64;
        Some(2.0 * self.diffusivity / (self.dx * self.dx) * (1.0 - angle.cos()))
    }

    /// A stable RK4 step size, `0.7·dx²/α` (the explicit diffusion limit
    /// with a safety margin).
    pub fn stable_step(&self) -> f64 {
        0.7 * self.dx * self.dx / self.diffusivity
    }
}

impl System for HeatRod1d {
    fn dim(&self) -> usize {
        self.nodes
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let scale = self.diffusivity / (self.dx * self.dx);
        for i in 0..self.nodes
        {
            let left = if i == 0 { self.t_left } else { y[i - 1] };
            let right = if i + 1 == self.nodes
            {
                self.t_right
            }
            else
            {
                y[i + 1]
            };
            dydt[i] = scale * (left - 2.0 * y[i] + right);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    #[test]
    fn newton_cooling_matches_the_closed_form() {
        let sys = NewtonCooling::new(0.4, 20.0).unwrap();
        let traj = simulate(&sys, &[90.0], 0.0, 15.0, 0.01).unwrap();
        for (t, row) in traj.t.iter().zip(traj.y.iter())
        {
            let exact = sys.exact(90.0, *t).unwrap();
            assert!(
                (row[0] - exact).abs() < 1e-9,
                "t = {t}: {} vs {exact}",
                row[0]
            );
        }
    }

    #[test]
    fn heat_rod_reaches_the_linear_steady_state() {
        let rod = HeatRod1d::new(1.0, 0.1, 9, 100.0, 20.0).unwrap();
        // Start uniform at 0 and integrate several diffusion times.
        let traj = simulate(&rod, &[0.0; 9], 0.0, 5.0, rod.stable_step()).unwrap();
        let last = traj.last_state().unwrap();
        for (i, &temp) in last.iter().enumerate()
        {
            let exact = rod.steady_state(i).unwrap();
            assert!((temp - exact).abs() < 1e-6, "node {i}: {temp} vs {exact}");
        }
    }

    #[test]
    fn fundamental_sine_mode_decays_at_the_discrete_eigenvalue_rate() {
        // With zero boundary temperatures, the discrete sine mode is an
        // exact eigenvector of the semi-discrete system: its amplitude
        // decays as e^{-λ₁·t} with the *discrete* eigenvalue λ₁.
        let n = 9;
        let rod = HeatRod1d::new(1.0, 0.1, n, 0.0, 0.0).unwrap();
        let mode: Vec<f64> = (1..=n)
            .map(|i| (i as f64 * std::f64::consts::PI / (n + 1) as f64).sin())
            .collect();
        let lambda = rod.mode_decay_rate(1).unwrap();
        let t_end = 0.05;
        let traj = simulate(&rod, &mode, 0.0, t_end, 1e-4).unwrap();
        let decay = (-lambda * t_end).exp();
        for (i, &temp) in traj.last_state().unwrap().iter().enumerate()
        {
            let exact = mode[i] * decay;
            assert!((temp - exact).abs() < 1e-6, "node {i}: {temp} vs {exact}");
        }
    }

    #[test]
    fn heat_rod_respects_the_maximum_principle() {
        // All temperatures must stay inside the range spanned by the initial
        // condition and the boundaries — heat cannot overshoot.
        let rod = HeatRod1d::new(0.5, 0.2, 7, 10.0, 60.0).unwrap();
        let init = vec![30.0; 7];
        let traj = simulate(&rod, &init, 0.0, 2.0, rod.stable_step()).unwrap();
        for row in &traj.y
        {
            for &temp in row
            {
                assert!(
                    (10.0 - 1e-9..=60.0 + 1e-9).contains(&temp),
                    "escaped: {temp}"
                );
            }
        }
    }

    #[test]
    fn constructors_and_helpers_reject_bad_inputs() {
        assert!(NewtonCooling::new(0.0, 20.0).is_err());
        assert!(NewtonCooling::new(0.4, f64::NAN).is_err());
        assert!(HeatRod1d::new(0.0, 0.1, 5, 0.0, 0.0).is_err());
        assert!(HeatRod1d::new(1.0, -0.1, 5, 0.0, 0.0).is_err());
        assert!(HeatRod1d::new(1.0, 0.1, 0, 0.0, 0.0).is_err());
        assert!(HeatRod1d::new(1.0, 0.1, 5, f64::INFINITY, 0.0).is_err());
        let rod = HeatRod1d::new(1.0, 0.1, 5, 0.0, 1.0).unwrap();
        assert!(rod.steady_state(5).is_none());
        assert!(rod.mode_decay_rate(0).is_none());
        assert!(rod.mode_decay_rate(6).is_none());
        let cooling = NewtonCooling::new(1.0, 0.0).unwrap();
        assert!(cooling.exact(f64::NAN, 1.0).is_none());
    }
}
