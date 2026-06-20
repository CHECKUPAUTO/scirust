//! EWMA control chart — sensitive to small sustained mean shifts.

use serde::{Deserialize, Serialize};

/// Exponentially-Weighted Moving-Average control chart.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EwmaChart {
    center: f64,
    sigma: f64,
    lambda: f64,
    l: f64,
    z: f64,
    t: u32,
}

impl EwmaChart {
    /// New chart for an in-control mean/σ, smoothing `lambda ∈ (0,1]` and limit
    /// width `l` (≈ 2.7–3).
    pub fn new(center: f64, sigma: f64, lambda: f64, l: f64) -> Self {
        Self {
            center,
            sigma,
            lambda: lambda.clamp(1e-3, 1.0),
            l,
            z: center,
            t: 0,
        }
    }

    /// Current EWMA statistic.
    pub fn value(&self) -> f64 {
        self.z
    }

    /// Half-width of the (time-varying) control limit at the current step.
    pub fn limit(&self) -> f64 {
        let lam = self.lambda;
        let factor = lam / (2.0 - lam) * (1.0 - (1.0 - lam).powi(2 * self.t as i32));
        self.l * self.sigma * factor.sqrt()
    }

    /// Feed a new observation; returns `true` if it is out of control.
    pub fn update(&mut self, x: f64) -> bool {
        self.t += 1;
        self.z = self.lambda * x + (1.0 - self.lambda) * self.z;
        (self.z - self.center).abs() > self.limit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_small_sustained_shift_shewhart_would_miss() {
        // 1.5σ mean shift — never trips a 3σ Shewhart rule, but EWMA accumulates.
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 2.7);
        let shift = 1.5;
        let mut tripped = false;
        // Deterministic small zig-zag around the shifted mean.
        for k in 0..60
        {
            let noise = if k % 2 == 0 { 0.1 } else { -0.1 };
            if chart.update(shift + noise)
            {
                tripped = true;
                break;
            }
            // A single sample never exceeds 3σ, so a Shewhart 3σ test stays quiet.
            assert!((shift + noise).abs() < 3.0);
        }
        assert!(tripped, "EWMA failed to catch the 0.5σ shift");
    }

    #[test]
    fn stays_in_control_on_centered_data() {
        let mut chart = EwmaChart::new(0.0, 1.0, 0.2, 3.0);
        for k in 0..200
        {
            let x = if k % 2 == 0 { 0.3 } else { -0.3 };
            assert!(
                !chart.update(x),
                "false alarm at step {k}, z={}",
                chart.value()
            );
        }
    }
}
