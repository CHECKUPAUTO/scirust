use scirust_core::nn::conformal::conformal_quantile;

/// Conformal guard for SCIAGENT inference.
///
/// Non-conformity score = 1 − softmax_probability.
/// The conformal quantile q̂ at level α gives a threshold.
/// - Accept if score ≤ q̂ (prob ≥ 1 − q̂)
/// - Reject otherwise
///
/// With an optional abstention band:
/// - Abstain if q̂ < score ≤ q̂ + ε (close call)
pub struct ConformalGuard {
    score_threshold: f32,
    abstain_margin: f32,
    alpha: f32,
}

impl ConformalGuard {
    pub fn calibrate(nonconformity_scores: &[f32], alpha: f32) -> Self {
        let q = conformal_quantile(nonconformity_scores, alpha);
        Self {
            score_threshold: q,
            abstain_margin: q + 0.05,
            alpha,
        }
    }

    pub fn decide(&self, probability: f32) -> GuardVerdict {
        let score = 1.0 - probability;
        if score <= self.score_threshold
        {
            GuardVerdict::Accept(probability as f64)
        }
        else if score <= self.abstain_margin
        {
            GuardVerdict::Abstain
        }
        else
        {
            GuardVerdict::Reject
        }
    }

    pub fn threshold(&self) -> f32 {
        self.score_threshold
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuardVerdict {
    Accept(f64),
    Abstain,
    Reject,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_accepts_high_confidence() {
        let scores: Vec<f32> = (0..50).map(|i| 0.01 + i as f32 * 0.02).collect();
        let guard = ConformalGuard::calibrate(&scores, 0.1);
        assert!(
            guard.threshold().is_finite(),
            "threshold={:?}",
            guard.threshold()
        );
        let verdict = guard.decide(0.98);
        assert!(matches!(verdict, GuardVerdict::Accept(_)));
    }

    #[test]
    fn test_guard_rejects_or_abstains_low_confidence() {
        let scores: Vec<f32> = (0..50).map(|i| 0.01 + i as f32 * 0.02).collect();
        let guard = ConformalGuard::calibrate(&scores, 0.1);
        // Very low probability → should not be accepted
        let verdict = guard.decide(0.001);
        assert!(matches!(
            verdict,
            GuardVerdict::Reject | GuardVerdict::Abstain
        ));
    }
}
