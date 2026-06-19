//! Simplex-architecture safety monitor: a high-performance neural controller
//! gated by a **verified safety envelope**.
//!
//! The classic Simplex pattern pairs an unverified high-performance controller
//! with a simple verified one and a decision module that switches to the safe
//! controller before the system can leave the safe set. Here the decision is
//! made *with a proof*: for the whole input-uncertainty box `x ± ε`, the
//! network's output is **certified** with interval/CROWN bounds
//! ([`IbpMlp::certify`]). The network's action is trusted only when its certified
//! output box lies entirely inside the safe envelope `[lo, hi]`; otherwise the
//! monitor emits a verified fallback action. By construction the monitor can
//! **never** output a value outside the safe envelope, whatever the network does.
//!
//! This is the bridge between SciRust's functional-safety layer (ISO 26262
//! degraded-mode fallback) and its certified-AI layer (sound output bounds).

use scirust_core::nn::ibp::{IbpMlp, Interval};
use serde::{Deserialize, Serialize};

/// Outcome of a safety decision for one input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SafetyDecision {
    /// The network's output is certified within the envelope; use it.
    Trusted(Vec<f32>),
    /// The certified output could leave the envelope; verified fallback used.
    Fallback(Vec<f32>),
}

impl SafetyDecision {
    /// The action actually applied.
    pub fn output(&self) -> &[f32] {
        match self
        {
            SafetyDecision::Trusted(v) | SafetyDecision::Fallback(v) => v,
        }
    }

    /// Whether the network output was trusted.
    pub fn is_trusted(&self) -> bool {
        matches!(self, SafetyDecision::Trusted(_))
    }
}

/// A neural controller wrapped in a certified safety envelope.
pub struct SimplexMonitor {
    net: IbpMlp,
    lo: Vec<f32>,
    hi: Vec<f32>,
    fallback: Vec<f32>,
}

impl SimplexMonitor {
    /// Wrap `net` with the safe output envelope `[lo, hi]` and a verified
    /// `fallback` action. Panics unless `lo ≤ hi` element-wise and the fallback
    /// lies inside the envelope (the fallback must itself be safe).
    pub fn new(net: IbpMlp, lo: Vec<f32>, hi: Vec<f32>, fallback: Vec<f32>) -> Self {
        assert!(
            lo.len() == hi.len() && fallback.len() == lo.len(),
            "SimplexMonitor: dimension mismatch"
        );
        assert!(
            lo.iter().zip(&hi).all(|(l, h)| l <= h),
            "envelope needs lo ≤ hi"
        );
        assert!(
            fallback
                .iter()
                .zip(lo.iter().zip(&hi))
                .all(|(&f, (&l, &h))| f >= l && f <= h),
            "fallback action must lie inside the safe envelope"
        );
        Self {
            net,
            lo,
            hi,
            fallback,
        }
    }

    /// Whether a certified output box lies fully inside the safe envelope.
    fn envelope_contains(&self, out: &Interval) -> bool {
        out.lo.iter().zip(&self.lo).all(|(o, l)| o >= l)
            && out.hi.iter().zip(&self.hi).all(|(o, h)| o <= h)
    }

    /// Decide for input `x` under an L∞ uncertainty radius `eps`.
    ///
    /// Trusts the network's point output only if the certified output box for
    /// the whole box `x ± eps` is inside the envelope; otherwise falls back.
    pub fn decide(&self, x: &[f32], eps: f32) -> SafetyDecision {
        let certified = self.net.certify(&Interval::around(x, eps));
        if self.envelope_contains(&certified)
        {
            SafetyDecision::Trusted(self.net.forward(x))
        }
        else
        {
            SafetyDecision::Fallback(self.fallback.clone())
        }
    }

    /// The action actually applied for `x` under uncertainty `eps` (always
    /// within the safe envelope).
    pub fn action(&self, x: &[f32], eps: f32) -> Vec<f32> {
        self.decide(x, eps).output().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::ibp::IbpLinear;

    // net(x) = relu(0.5·x0) + relu(0.5·x1); for x ≥ 0 this is 0.5(x0+x1).
    fn demo_net() -> IbpMlp {
        let l1 = IbpLinear::new(vec![0.5, 0.0, 0.0, 0.5], vec![0.0, 0.0], 2, 2);
        let l2 = IbpLinear::new(vec![1.0, 1.0], vec![0.0], 2, 1);
        IbpMlp::new(vec![l1, l2])
    }

    fn monitor() -> SimplexMonitor {
        // Safe envelope on the scalar output: [-0.1, 2.0]. Fallback = 1.0.
        SimplexMonitor::new(demo_net(), vec![-0.1], vec![2.0], vec![1.0])
    }

    #[test]
    fn trusts_when_certified_safe() {
        let m = monitor();
        // x=(1,1): output ~1.0, tight box well inside [-0.1, 2.0].
        let d = m.decide(&[1.0, 1.0], 0.2);
        assert!(d.is_trusted());
        assert!((d.output()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn falls_back_when_output_could_exceed_envelope() {
        let m = monitor();
        // x=(3,3): output ~3.0 > 2.0 -> certified box exceeds envelope.
        let d = m.decide(&[3.0, 3.0], 0.2);
        assert!(!d.is_trusted());
        assert_eq!(d.output(), &[1.0]); // verified fallback, inside envelope
    }

    #[test]
    fn applied_action_is_always_inside_envelope() {
        let m = monitor();
        // Sweep a wide grid of inputs and uncertainties; the applied action
        // must never leave [-0.1, 2.0].
        let mut x0 = -2.0;
        while x0 <= 6.0
        {
            let mut x1 = -2.0;
            while x1 <= 6.0
            {
                for &eps in &[0.0_f32, 0.2, 1.0, 4.0]
                {
                    let y = m.action(&[x0, x1], eps);
                    assert!(
                        y[0] >= -0.1 && y[0] <= 2.0,
                        "unsafe action {} at x=({x0},{x1}) eps={eps}",
                        y[0]
                    );
                }
                x1 += 0.5;
            }
            x0 += 0.5;
        }
    }

    #[test]
    fn trusted_decisions_are_sound_against_brute_force() {
        let m = monitor();
        let net = demo_net();
        // Wherever the monitor trusts the net, EVERY point in the input box must
        // actually produce an in-envelope output (certification soundness).
        let (x, eps) = ([1.0_f32, 1.0], 0.2);
        let d = m.decide(&x, eps);
        assert!(d.is_trusted());
        let mut a = x[0] - eps;
        while a <= x[0] + eps
        {
            let mut b = x[1] - eps;
            while b <= x[1] + eps
            {
                let y = net.forward(&[a, b]);
                assert!(y[0] >= -0.1 && y[0] <= 2.0, "certify unsound at ({a},{b})");
                b += eps / 8.0;
            }
            a += eps / 8.0;
        }
    }
}
