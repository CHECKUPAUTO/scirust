//! CUSUM control chart — the fastest detector of small sustained mean shifts.
//!
//! Where a Shewhart chart reacts only to the latest point, the cumulative-sum
//! chart accumulates the signed departures from target, so a small but persistent
//! drift builds up until it crosses a decision interval. The tabular form keeps
//! two one-sided sums with a **reference value** `K = k·σ` (the slack, usually
//! `k = ½` the shift to detect) and a **decision interval** `H = h·σ` (usually
//! `h ≈ 4–5`):
//!
//! ```text
//! Cᵢ⁺ = max(0, Cᵢ₋₁⁺ + (xᵢ − μ₀) − K) ,
//! Cᵢ⁻ = max(0, Cᵢ₋₁⁻ − (xᵢ − μ₀) − K) ,
//! ```
//!
//! signalling when either exceeds `H`. Together with [`crate::ewma`] it is the
//! memory-based complement to the memoryless Shewhart chart.
//!
//! ## Average run length
//!
//! Siegmund's approximation gives the one-sided ARL as a function of the actual
//! standardised shift `Δ` (in σ), with `Δ★ = Δ − k` and `b = h + 1.166`:
//!
//! ```text
//! ARL = (exp(−2Δ★b) + 2Δ★b − 1) / (2Δ★²)   (Δ★ ≠ 0) ,   b²   (Δ★ = 0) .
//! ```
//!
//! The two-sided chart combines the upper and lower one-sided ARLs as
//! `1/ARL = 1/ARL⁺ + 1/ARL⁻`.

use serde::{Deserialize, Serialize};

/// Which side of a two-sided CUSUM raised a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CusumSide {
    /// The upper sum `C⁺` exceeded `H` — an upward shift.
    Upper,
    /// The lower sum `C⁻` exceeded `H` — a downward shift.
    Lower,
}

/// A tabular two-sided CUSUM chart for an in-control mean `target` and deviation
/// `sigma`, with reference `k` and decision interval `h` (both in σ units).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CusumChart {
    target: f64,
    sigma: f64,
    k: f64,
    h: f64,
    c_plus: f64,
    c_minus: f64,
}

impl CusumChart {
    /// New chart for an in-control `target`/`sigma`, reference `k` (σ units, the
    /// half-shift to detect, typically 0.5) and decision interval `h` (σ units,
    /// typically 4–5).
    pub fn new(target: f64, sigma: f64, k: f64, h: f64) -> Self {
        CusumChart {
            target,
            sigma,
            k: k.max(0.0),
            h: h.max(0.0),
            c_plus: 0.0,
            c_minus: 0.0,
        }
    }

    /// Current upper sum `C⁺`.
    pub fn c_plus(&self) -> f64 {
        self.c_plus
    }

    /// Current lower sum `C⁻`.
    pub fn c_minus(&self) -> f64 {
        self.c_minus
    }

    /// Decision interval `H = h·σ`.
    pub fn decision_interval(&self) -> f64 {
        self.h * self.sigma
    }

    /// Reset both sums to zero (e.g. after acting on a signal).
    pub fn reset(&mut self) {
        self.c_plus = 0.0;
        self.c_minus = 0.0;
    }

    /// Feed a new observation; returns the side that signalled, if any.
    pub fn update(&mut self, x: f64) -> Option<CusumSide> {
        let big_k = self.k * self.sigma;
        let big_h = self.h * self.sigma;
        let dev = x - self.target;
        self.c_plus = (self.c_plus + dev - big_k).max(0.0);
        self.c_minus = (self.c_minus - dev - big_k).max(0.0);
        if self.c_plus > big_h
        {
            Some(CusumSide::Upper)
        }
        else if self.c_minus > big_h
        {
            Some(CusumSide::Lower)
        }
        else
        {
            None
        }
    }

    /// Two-sided average run length at a standardised mean shift `shift` (in σ),
    /// combining the upper and lower one-sided ARLs. `shift = 0` gives the
    /// in-control ARL₀.
    pub fn average_run_length(&self, shift: f64) -> f64 {
        let up = arl_one_sided(shift, self.k, self.h);
        let down = arl_one_sided(-shift, self.k, self.h);
        1.0 / (1.0 / up + 1.0 / down)
    }
}

/// One-sided upper-CUSUM ARL by Siegmund's approximation, for a standardised
/// shift `shift` (σ units), reference `k` and interval `h`.
pub fn arl_one_sided(shift: f64, k: f64, h: f64) -> f64 {
    let b = h + 1.166;
    let delta = shift - k;
    if delta.abs() < 1e-9
    {
        b * b
    }
    else
    {
        ((-2.0 * delta * b).exp() + 2.0 * delta * b - 1.0) / (2.0 * delta * delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upward_shift_is_caught_by_the_upper_sum() {
        // In control at 0 (σ=1), then a +1σ sustained shift.
        let mut c = CusumChart::new(0.0, 1.0, 0.5, 4.0);
        // A few on-target points keep both sums near zero.
        for _ in 0..5
        {
            assert!(c.update(0.0).is_none());
        }
        // A run above target accumulates C⁺ until it signals Upper.
        let mut side = None;
        for _ in 0..20
        {
            if let Some(s) = c.update(1.5)
            {
                side = Some(s);
                break;
            }
        }
        assert_eq!(side, Some(CusumSide::Upper));
        assert!(c.c_plus() > c.decision_interval());
    }

    #[test]
    fn siegmund_in_control_arl_matches_reference() {
        // k=0.5, h=4 ⇒ one-sided ARL₀ ≈ 336, two-sided ≈ 168 (Montgomery table).
        let one = arl_one_sided(0.0, 0.5, 4.0);
        assert!((one - 336.0).abs() < 10.0, "one-sided ARL0 {one}");
        let c = CusumChart::new(0.0, 1.0, 0.5, 4.0);
        let two = c.average_run_length(0.0);
        assert!((two - 168.0).abs() < 8.0, "two-sided ARL0 {two}");
        // h=5 raises ARL₀ sharply (≈ 465 two-sided).
        let c5 = CusumChart::new(0.0, 1.0, 0.5, 5.0);
        assert!(c5.average_run_length(0.0) > 400.0);
    }

    #[test]
    fn arl_drops_fast_under_a_shift() {
        // A 1σ shift should be detected in a handful of samples on average.
        let c = CusumChart::new(0.0, 1.0, 0.5, 4.0);
        let arl1 = c.average_run_length(1.0);
        assert!(arl1 < 12.0, "ARL at 1σ {arl1}");
        assert!(arl1 < c.average_run_length(0.0));
    }

    #[test]
    fn monte_carlo_false_alarm_rate_matches_arl0() {
        // Simulate in-control N(0,1) data and measure the run length to a false
        // alarm; its average should track the Siegmund two-sided ARL₀ (~168).
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut normal = || {
            let mut u01 = || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
            };
            let (a, b) = (u01().max(1e-12), u01());
            (-2.0 * a.ln()).sqrt() * (2.0 * core::f64::consts::PI * b).cos()
        };
        let runs = 4000;
        let mut total = 0u64;
        for _ in 0..runs
        {
            let mut c = CusumChart::new(0.0, 1.0, 0.5, 4.0);
            let mut n = 0u64;
            loop
            {
                n += 1;
                if c.update(normal()).is_some() || n > 5000
                {
                    break;
                }
            }
            total += n;
        }
        let emp = total as f64 / runs as f64;
        // Within ~15 % of the analytic two-sided ARL₀.
        assert!((emp - 168.0).abs() / 168.0 < 0.15, "empirical ARL0 {emp}");
    }
}
