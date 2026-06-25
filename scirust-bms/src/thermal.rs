//! Thermal-runaway early warning.
//!
//! The danger in a cell is not a high temperature per se but an **accelerating**
//! one. [`ThermalGuard`] tracks the (smoothed) rate of temperature rise and
//! raises a `Warning` as soon as that rate crosses a threshold — catching a
//! runaway *before* the critical temperature is reached — and `Critical` once
//! the critical temperature is hit.

use serde::{Deserialize, Serialize};

/// Thermal state verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    /// Temperature and its rate are within bounds.
    Normal,
    /// Rate of rise exceeds the threshold — early runaway warning.
    Warning,
    /// Critical temperature reached.
    Critical,
}

/// Early-warning guard on cell temperature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalGuard {
    rate_warn: f64,
    critical_temp: f64,
    last_temp: Option<f64>,
    ewma_rate: f64,
    beta: f64,
}

impl ThermalGuard {
    /// Warn when the smoothed rise rate reaches `rate_warn` (°C/s); flag
    /// `Critical` at `critical_temp` (°C).
    pub fn new(rate_warn: f64, critical_temp: f64) -> Self {
        Self {
            rate_warn,
            critical_temp,
            last_temp: None,
            ewma_rate: 0.0,
            beta: 0.5,
        }
    }

    /// Smoothed rate of temperature rise (°C/s).
    pub fn rate(&self) -> f64 {
        self.ewma_rate
    }

    /// Feed a new temperature sample `temp` (°C) taken `dt` (s) after the last,
    /// and get the verdict.
    pub fn update(&mut self, temp: f64, dt: f64) -> ThermalState {
        let rate = match self.last_temp
        {
            Some(t) if dt > 0.0 => (temp - t) / dt,
            _ => 0.0,
        };
        self.ewma_rate = self.beta * self.ewma_rate + (1.0 - self.beta) * rate;
        self.last_temp = Some(temp);

        if temp >= self.critical_temp
        {
            ThermalState::Critical
        }
        else if self.ewma_rate >= self.rate_warn
        {
            ThermalState::Warning
        }
        else
        {
            ThermalState::Normal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warns_before_critical_on_runaway() {
        // Accelerating temperature: rate grows by 0.05 °C/s each second.
        let mut g = ThermalGuard::new(1.0, 60.0);
        let dt = 1.0;
        let mut temp = 25.0;
        let mut rate = 0.0;
        let mut first_warning: Option<(usize, f64)> = None;
        let mut first_critical: Option<usize> = None;
        for k in 0..200
        {
            let st = g.update(temp, dt);
            if st == ThermalState::Warning && first_warning.is_none()
            {
                first_warning = Some((k, temp));
            }
            if st == ThermalState::Critical && first_critical.is_none()
            {
                first_critical = Some(k);
            }
            rate += 0.05;
            temp += rate * dt;
        }
        let (kw, tw) = first_warning.expect("no early warning issued");
        let kc = first_critical.expect("never reached critical");
        assert!(kw < kc, "warning ({kw}) must precede critical ({kc})");
        assert!(tw < 60.0, "warning fired at {tw} °C, not before critical");
    }

    #[test]
    fn stays_normal_on_slow_warming() {
        // Gentle 0.05 °C/s rise — well under the 1 °C/s warning rate.
        let mut g = ThermalGuard::new(1.0, 60.0);
        let mut temp = 25.0;
        for _ in 0..300
        {
            assert_eq!(g.update(temp, 1.0), ThermalState::Normal);
            temp += 0.05;
        }
    }

    #[test]
    fn ewma_rate_follows_exact_recursion() {
        // β = 0.5, so ewmaₖ = 0.5·ewmaₖ₋₁ + 0.5·rateₖ. Hand trace at dt = 1:
        //   sample 25 (no prior) → rate 0  → ewma 0
        //   sample 27           → rate 2  → ewma 0.5·0   + 0.5·2 = 1.0
        //   sample 29           → rate 2  → ewma 0.5·1.0 + 0.5·2 = 1.5
        //   sample 29.5         → rate .5 → ewma 0.5·1.5 + 0.5·0.5 = 1.0
        let mut g = ThermalGuard::new(100.0, 1000.0); // thresholds out of the way
        g.update(25.0, 1.0);
        assert!((g.rate() - 0.0).abs() < 1e-12);
        g.update(27.0, 1.0);
        assert!((g.rate() - 1.0).abs() < 1e-12, "ewma {}", g.rate());
        g.update(29.0, 1.0);
        assert!((g.rate() - 1.5).abs() < 1e-12, "ewma {}", g.rate());
        g.update(29.5, 1.0);
        assert!((g.rate() - 1.0).abs() < 1e-12, "ewma {}", g.rate());
    }

    #[test]
    fn warning_fires_exactly_at_the_rate_threshold() {
        // A single +2 °C/1 s step makes ewma = 0.5·0 + 0.5·2 = 1.0, which is
        // exactly the warning rate ⇒ Warning (the comparison is `>=`).
        let mut g = ThermalGuard::new(1.0, 60.0);
        assert_eq!(g.update(25.0, 1.0), ThermalState::Normal);
        assert_eq!(g.update(27.0, 1.0), ThermalState::Warning);
        assert!((g.rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn critical_temperature_overrides_rate() {
        // At/above the critical temperature the verdict is Critical regardless
        // of how the rate compares to the warning threshold.
        let mut g = ThermalGuard::new(1.0, 60.0);
        g.update(25.0, 1.0);
        // Jump straight to the critical temperature: rate is huge AND temp is
        // critical — Critical must win over Warning.
        assert_eq!(g.update(60.0, 1.0), ThermalState::Critical);
        // Still critical while hot even if the rate has settled to ~0.
        assert_eq!(g.update(60.0, 1.0), ThermalState::Critical);
    }

    #[test]
    fn first_sample_and_zero_dt_yield_zero_rate() {
        // No previous temperature ⇒ rate 0; and a non-positive dt is guarded so
        // it can never produce a divide-by-zero or a spurious rate spike.
        let mut g = ThermalGuard::new(1.0, 60.0);
        assert_eq!(g.update(40.0, 1.0), ThermalState::Normal); // first sample
        assert!((g.rate() - 0.0).abs() < 1e-12);
        // dt = 0 with a big temperature jump: rate contribution is 0, ewma stays 0.
        assert_eq!(g.update(55.0, 0.0), ThermalState::Normal);
        assert!((g.rate() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cooling_drives_the_smoothed_rate_negative() {
        // A falling temperature gives a negative instantaneous rate, which pulls
        // the EWMA below zero — and certainly never warns.
        let mut g = ThermalGuard::new(1.0, 200.0);
        g.update(80.0, 1.0); // prime
        let st = g.update(70.0, 1.0); // rate = -10 ⇒ ewma = 0.5·0 + 0.5·(-10) = -5
        assert_eq!(st, ThermalState::Normal);
        assert!((g.rate() - (-5.0)).abs() < 1e-12, "ewma {}", g.rate());
    }
}
