//! ISO/TS 15066 Speed-and-Separation Monitoring (SSM) for collaborative robots.
//!
//! The protective separation distance is
//! `Sp = Sh + Sr + Ss + C`, where `Sh` is how far the human can move during the
//! robot's reaction+stop time, `Sr` is the robot's travel during reaction, `Ss`
//! its stopping distance, and `C` a static intrusion/uncertainty margin. If the
//! measured separation drops below `Sp`, the robot must slow or stop. This gives
//! a *provable* safety condition and the maximum speed that keeps it satisfied.

use serde::{Deserialize, Serialize};

/// Cell parameters for SSM.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SsmParams {
    /// Maximum human approach speed (m/s), e.g. 1.6.
    pub human_speed: f64,
    /// Robot reaction time `Tr` (s).
    pub reaction_time: f64,
    /// Robot stopping time `Ts` (s).
    pub stop_time: f64,
    /// Static margin `C` (m): intrusion distance + sensor uncertainty.
    pub margin: f64,
}

/// Robot stopping distance under constant deceleration from `robot_speed`
/// over `stop_time` (`v·Ts/2`).
fn stopping_distance(robot_speed: f64, stop_time: f64) -> f64 {
    0.5 * robot_speed * stop_time
}

/// Protective separation distance `Sp` (m) for a given robot speed.
pub fn protective_separation(p: &SsmParams, robot_speed: f64) -> f64 {
    let sh = p.human_speed * (p.reaction_time + p.stop_time);
    let sr = robot_speed * p.reaction_time;
    let ss = stopping_distance(robot_speed, p.stop_time);
    sh + sr + ss + p.margin
}

/// Whether the current `separation` (m) is safe at `robot_speed`.
pub fn is_safe(p: &SsmParams, separation: f64, robot_speed: f64) -> bool {
    separation >= protective_separation(p, robot_speed)
}

/// Maximum robot speed (m/s) that keeps `Sp ≤ separation`, clamped to
/// `[0, v_cmd]`. Solving `Sp(v) ≤ d` for `v` (Sp is affine in v):
/// `v ≤ (d − Sh − C) / (Tr + Ts/2)`.
pub fn max_safe_speed(p: &SsmParams, separation: f64, v_cmd: f64) -> f64 {
    let sh = p.human_speed * (p.reaction_time + p.stop_time);
    let denom = p.reaction_time + 0.5 * p.stop_time;
    if denom <= 0.0
    {
        return 0.0;
    }
    let v = (separation - sh - p.margin) / denom;
    v.clamp(0.0, v_cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SsmParams {
        SsmParams {
            human_speed: 1.6,
            reaction_time: 0.1,
            stop_time: 0.3,
            margin: 0.2,
        }
    }

    #[test]
    fn closer_separation_requires_lower_speed() {
        let p = params();
        let far = max_safe_speed(&p, 2.0, 5.0);
        let near = max_safe_speed(&p, 1.0, 5.0);
        assert!(
            far > near,
            "far {far} should allow more speed than near {near}"
        );
    }

    #[test]
    fn the_max_safe_speed_is_exactly_at_the_safety_boundary() {
        let p = params();
        let d = 1.5;
        let v = max_safe_speed(&p, d, 5.0);
        // At v, Sp should equal d (the binding constraint), so it is just safe.
        assert!(is_safe(&p, d, v), "v {v} not safe at d {d}");
        // A hair faster is unsafe.
        if v < 5.0
        {
            assert!(!is_safe(&p, d, v + 0.05));
        }
    }

    #[test]
    fn too_close_forces_a_stop() {
        let p = params();
        // Separation below Sh + C: even v=0 cannot be made safe by speed alone.
        let v = max_safe_speed(&p, 0.3, 5.0);
        assert_eq!(v, 0.0, "must command a stop");
    }
}
