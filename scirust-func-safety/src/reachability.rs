//! Certified finite-horizon reachability.
//!
//! Given a neural one-step state-transition map `x_{k+1} = net(x_k)` and an
//! uncertain initial state box, this propagates the **certified reachable set**
//! forward (interval/CROWN bounds at each step) and checks it stays inside the
//! safe envelope for the whole horizon. The reachable boxes are a sound
//! over-approximation, so "safe over the horizon" is a proof — not a sampling.

use scirust_core::nn::ibp::{IbpMlp, Interval};

/// Result of a finite-horizon reachability check.
#[derive(Debug, Clone)]
pub struct ReachResult {
    /// Certified reachable box at each step (`boxes[0]` is the initial set).
    pub boxes: Vec<Interval>,
    /// Whether every step stayed inside the safe envelope.
    pub safe: bool,
    /// First step (1-based) that left the envelope, if any.
    pub first_violation: Option<usize>,
}

/// Propagate the certified reachable set of `net` from `x0` for `horizon` steps,
/// checking containment in the safe envelope `[lo, hi]` at each step.
pub fn certified_reach(
    net: &IbpMlp,
    x0: &Interval,
    lo: &[f32],
    hi: &[f32],
    horizon: usize,
) -> ReachResult {
    let in_envelope = |iv: &Interval| -> bool {
        // `zip` stops at the shorter iterator, so a state dimension beyond
        // `lo.len()`/`hi.len()` would be silently ignored and could leave the
        // envelope while we still report containment. Containment is only a
        // proof when the envelope constrains *every* state coordinate, so a
        // dimension mismatch means we cannot certify safety.
        iv.lo.len() == lo.len()
            && iv.hi.len() == hi.len()
            && iv.lo.iter().zip(lo).all(|(c, l)| c >= l)
            && iv.hi.iter().zip(hi).all(|(c, h)| c <= h)
    };
    let mut cur = x0.clone();
    let mut boxes = vec![cur.clone()];
    let mut safe = in_envelope(&cur);
    let mut first_violation = if safe { None } else { Some(0) };
    for step in 1..=horizon
    {
        cur = net.certify(&cur);
        let ok = in_envelope(&cur);
        boxes.push(cur.clone());
        if !ok && first_violation.is_none()
        {
            safe = false;
            first_violation = Some(step);
        }
    }
    ReachResult {
        boxes,
        safe,
        first_violation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::ibp::IbpLinear;

    fn scalar_map(gain: f32) -> IbpMlp {
        IbpMlp::new(vec![IbpLinear::new(vec![gain], vec![0.0], 1, 1)])
    }

    #[test]
    fn contractive_system_is_safe_over_the_horizon() {
        let net = scalar_map(0.5); // x_{k+1} = 0.5 x -> shrinks
        let x0 = Interval::around(&[0.0], 1.0); // [-1, 1]
        let r = certified_reach(&net, &x0, &[-2.0], &[2.0], 30);
        assert!(
            r.safe && r.first_violation.is_none(),
            "{:?}",
            r.first_violation
        );
        // The reachable box shrinks each step.
        assert!(r.boxes.last().unwrap().max_radius() < 0.01);
    }

    #[test]
    fn expansive_system_violates_and_reports_the_step() {
        let net = scalar_map(1.5); // x_{k+1} = 1.5 x -> grows past [-2,2]
        let x0 = Interval::around(&[0.0], 1.0); // radius 1 -> 1.5 -> 2.25 at step 2
        let r = certified_reach(&net, &x0, &[-2.0], &[2.0], 10);
        assert!(!r.safe);
        assert_eq!(r.first_violation, Some(2), "boxes {:?}", r.boxes);
    }

    #[test]
    fn envelope_shorter_than_state_is_not_certified_safe() {
        // A 2-D initial box whose second coordinate [10, 12] is far outside any
        // reasonable envelope, but the envelope only specifies bounds for the
        // first coordinate. The truncating `zip` would ignore the second
        // dimension and falsely report safe; a sound check must not.
        let net = scalar_map(0.5);
        let x0 = Interval {
            lo: vec![-1.0, 10.0],
            hi: vec![1.0, 12.0],
        };
        // Envelope covers only the first coordinate.
        let r = certified_reach(&net, &x0, &[-2.0], &[2.0], 0);
        assert!(
            !r.safe,
            "envelope shorter than state must not be certified safe: {r:?}"
        );
        assert_eq!(r.first_violation, Some(0));
    }

    #[test]
    fn reachable_set_soundly_contains_true_trajectories() {
        let gain = 1.2;
        let net = scalar_map(gain);
        let x0 = Interval::around(&[0.0], 1.0);
        let r = certified_reach(&net, &x0, &[-100.0], &[100.0], 8);
        // Brute-force several initial points; each true trajectory stays boxed.
        for i in 0..=10
        {
            let mut x = -1.0 + 0.2 * i as f32;
            for b in &r.boxes
            {
                assert!(
                    x >= b.lo[0] - 1e-5 && x <= b.hi[0] + 1e-5,
                    "x {x} escaped {b:?}"
                );
                x *= gain;
            }
        }
    }
}
