//! Planar 2-link arm kinematics.

/// Maximum reach of a 2-link arm.
pub fn max_reach(l1: f64, l2: f64) -> f64 {
    l1 + l2
}

/// Forward kinematics: end-effector `(x, y)` from joint angles (rad).
pub fn fk_2link(l1: f64, l2: f64, th1: f64, th2: f64) -> (f64, f64) {
    (
        l1 * th1.cos() + l2 * (th1 + th2).cos(),
        l1 * th1.sin() + l2 * (th1 + th2).sin(),
    )
}

/// Inverse kinematics (elbow-down branch): joint angles reaching `(x, y)`, or
/// `None` if the target is outside the annular workspace.
pub fn ik_2link(l1: f64, l2: f64, x: f64, y: f64) -> Option<(f64, f64)> {
    let r2 = x * x + y * y;
    let c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
    if !(-1.0..=1.0).contains(&c2)
    {
        return None; // unreachable
    }
    let s2 = -(1.0 - c2 * c2).sqrt(); // elbow-down
    let th2 = s2.atan2(c2);
    let th1 = y.atan2(x) - (l2 * s2).atan2(l1 + l2 * c2);
    Some((th1, th2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn fk_of_known_angles() {
        // θ1=0, θ2=π/2: tip at (l1, l2).
        let (x, y) = fk_2link(1.0, 0.5, 0.0, PI / 2.0);
        assert!(
            (x - 1.0).abs() < 1e-9 && (y - 0.5).abs() < 1e-9,
            "({x},{y})"
        );
    }

    #[test]
    fn ik_then_fk_round_trips() {
        let (l1, l2) = (1.0, 0.7);
        for &(x, y) in &[(1.2, 0.4), (0.5, 1.0), (-0.3, 0.9), (1.5, -0.2)]
        {
            let (th1, th2) = ik_2link(l1, l2, x, y).expect("reachable");
            let (xr, yr) = fk_2link(l1, l2, th1, th2);
            assert!(
                (xr - x).abs() < 1e-9 && (yr - y).abs() < 1e-9,
                "rt ({xr},{yr})"
            );
        }
    }

    #[test]
    fn unreachable_target_returns_none() {
        // Beyond the max reach 1.7.
        assert!(ik_2link(1.0, 0.7, 2.0, 0.0).is_none());
        assert!(max_reach(1.0, 0.7) < 2.0);
    }
}
