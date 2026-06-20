//! Rest-to-rest trapezoidal-velocity motion profile (bounded velocity and
//! acceleration).
//!
//! The standard point-to-point profile: accelerate at `amax` to (at most)
//! `vmax`, cruise, then decelerate to rest. When the move is too short to reach
//! `vmax` the profile is triangular. (A jerk-limited S-curve is a future
//! refinement; this bounds velocity and acceleration.)

use serde::{Deserialize, Serialize};

/// A planned trapezoidal profile over a non-negative distance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrapezoidalProfile {
    distance: f64,
    amax: f64,
    vpeak: f64,
    t_acc: f64,
    t_flat: f64,
    d_acc: f64,
    d_flat: f64,
}

impl TrapezoidalProfile {
    /// Plan a move of `distance` (≥ 0) with limits `vmax`, `amax` (> 0).
    pub fn new(distance: f64, vmax: f64, amax: f64) -> Self {
        let distance = distance.max(0.0);
        let d_acc_full = 0.5 * vmax * vmax / amax;
        let (vpeak, t_acc, t_flat, d_acc, d_flat) = if 2.0 * d_acc_full <= distance
        {
            // Trapezoid: vmax reached.
            let t_acc = vmax / amax;
            let d_flat = distance - 2.0 * d_acc_full;
            (vmax, t_acc, d_flat / vmax, d_acc_full, d_flat)
        }
        else
        {
            // Triangle: vmax not reached.
            let vpeak = (distance * amax).sqrt();
            let t_acc = vpeak / amax;
            (vpeak, t_acc, 0.0, 0.5 * distance, 0.0)
        };
        Self {
            distance,
            amax,
            vpeak,
            t_acc,
            t_flat,
            d_acc,
            d_flat,
        }
    }

    /// Total move duration (s).
    pub fn duration(&self) -> f64 {
        2.0 * self.t_acc + self.t_flat
    }

    /// Position at time `t`.
    pub fn position(&self, t: f64) -> f64 {
        if t <= 0.0
        {
            return 0.0;
        }
        if t >= self.duration()
        {
            return self.distance;
        }
        let t2 = self.t_acc + self.t_flat;
        if t < self.t_acc
        {
            0.5 * self.amax * t * t
        }
        else if t < t2
        {
            self.d_acc + self.vpeak * (t - self.t_acc)
        }
        else
        {
            let td = t - t2;
            self.d_acc + self.d_flat + self.vpeak * td - 0.5 * self.amax * td * td
        }
    }

    /// Velocity at time `t`.
    pub fn velocity(&self, t: f64) -> f64 {
        if t <= 0.0 || t >= self.duration()
        {
            return 0.0;
        }
        let t2 = self.t_acc + self.t_flat;
        if t < self.t_acc
        {
            self.amax * t
        }
        else if t < t2
        {
            self.vpeak
        }
        else
        {
            self.vpeak - self.amax * (t - t2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_profile(distance: f64, vmax: f64, amax: f64) {
        let p = TrapezoidalProfile::new(distance, vmax, amax);
        let t_total = p.duration();
        assert!((p.position(0.0)).abs() < 1e-9);
        assert!((p.position(t_total) - distance).abs() < 1e-6, "end pos");
        assert!(
            p.velocity(0.0).abs() < 1e-9 && p.velocity(t_total).abs() < 1e-9,
            "rest"
        );

        // Sample: velocity within [0, vmax+eps], acceleration within ±amax.
        let dt = t_total / 2000.0;
        let mut prev_v = 0.0;
        let mut prev_x = 0.0;
        for k in 1..=2000
        {
            let t = k as f64 * dt;
            let v = p.velocity(t);
            let x = p.position(t);
            assert!(v <= vmax + 1e-6 && v >= -1e-9, "v {v} out of range");
            let a = (v - prev_v) / dt;
            assert!(a.abs() <= amax + 1e-3, "accel {a} exceeds amax");
            assert!(x >= prev_x - 1e-9, "position not monotone");
            prev_v = v;
            prev_x = x;
        }
    }

    #[test]
    fn long_move_is_a_trapezoid() {
        check_profile(10.0, 2.0, 1.0); // reaches vmax
    }

    #[test]
    fn short_move_is_a_triangle() {
        check_profile(0.5, 2.0, 1.0); // too short to reach vmax
    }
}
