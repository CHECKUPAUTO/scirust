//! **Statistically-guaranteed guard** — a response gate with a *distribution-free*
//! coverage guarantee, to power a CCOS-style `guard` (validate/abstain on a model's
//! output) without ad-hoc thresholds.
//!
//! Given a model's class probabilities for a decision (e.g. *is this LLM response
//! valid?*, or the next-token distribution), the guard forms the **conformal
//! prediction set** of [`ConformalClassifier`] — every class whose probability clears
//! the calibrated threshold `1 − q̂` — and turns it into a verdict:
//!
//! - exactly one class clears ⇒ [`Accept`](GuardVerdict::Accept) it;
//! - several clear ⇒ [`Abstain`](GuardVerdict::Abstain) (ambiguous, flag for review);
//! - none clears ⇒ [`Reject`](GuardVerdict::Reject) (out of distribution).
//!
//! The conformal calibration guarantees the **true class is in the set with
//! probability ≥ 1 − α** on exchangeable data, *whatever the distribution* — so the
//! guard provably does not silently drop the correct answer more than an `α` fraction
//! of the time. Deterministic. (Deep ensembles, [`crate::nn::ensemble`], give a
//! complementary epistemic-uncertainty signal for out-of-distribution flagging.)

use crate::nn::conformal::ConformalClassifier;

/// The guard's decision on a probability vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardVerdict {
    /// A single conformally-valid class cleared the threshold — accept it.
    Accept(usize),
    /// Several classes cleared — ambiguous; flag for review.
    Abstain,
    /// No class cleared — out of distribution; reject.
    Reject,
}

/// A response guard backed by a calibrated conformal classifier.
pub struct StatisticalGuard {
    clf: ConformalClassifier,
}

impl StatisticalGuard {
    /// Calibrate from held-out per-example probabilities and true labels at risk
    /// level `alpha` (coverage `1 − α`).
    pub fn calibrate(cal_probs: &[Vec<f32>], cal_labels: &[usize], alpha: f32) -> Self {
        Self {
            clf: ConformalClassifier::calibrate(cal_probs, cal_labels, alpha),
        }
    }

    /// The conformal prediction set for `probs` (every class clearing `1 − q̂`).
    pub fn prediction_set(&self, probs: &[f32]) -> Vec<usize> {
        self.clf.predict_set(probs)
    }

    /// The guard verdict for a probability vector.
    pub fn decide(&self, probs: &[f32]) -> GuardVerdict {
        let set = self.clf.predict_set(probs);
        match set.len()
        {
            0 => GuardVerdict::Reject,
            1 => GuardVerdict::Accept(set[0]),
            _ => GuardVerdict::Abstain,
        }
    }

    /// Whether the guaranteed prediction set for `probs` contains `y_true` (the event
    /// that holds with probability ≥ 1 − α).
    pub fn covers(&self, probs: &[f32], y_true: usize) -> bool {
        self.clf.covers(probs, y_true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let m = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / s).collect()
    }

    /// A decent-but-imperfect 3-class classifier: the true class gets a boosted logit
    /// plus per-class noise.
    fn sample(rng: &mut PcgEngine) -> (Vec<f32>, usize) {
        let y = (rng.next_u32() % 3) as usize;
        let mut logits = [0.0f32; 3];
        for l in logits.iter_mut()
        {
            *l = rng.float_signed();
        }
        logits[y] += 1.5;
        (softmax(&logits), y)
    }

    /// **Distribution-free coverage**: on fresh exchangeable data the true class lies
    /// in the guard's prediction set at least `1 − α` of the time. Deterministic.
    #[test]
    fn guard_has_coverage_guarantee() {
        let alpha = 0.1f32;
        let run = || -> f32 {
            let mut rng = PcgEngine::new(1);
            let (cal_probs, cal_labels): (Vec<Vec<f32>>, Vec<usize>) =
                (0..2000).map(|_| sample(&mut rng)).unzip();
            let guard = StatisticalGuard::calibrate(&cal_probs, &cal_labels, alpha);
            let mut covered = 0usize;
            let n = 5000;
            for _ in 0..n
            {
                let (p, y) = sample(&mut rng);
                if guard.covers(&p, y)
                {
                    covered += 1;
                }
            }
            covered as f32 / n as f32
        };
        let cov = run();
        assert!(cov >= 1.0 - alpha - 0.03, "coverage {cov} below 1-α");
        assert_eq!(cov.to_bits(), run().to_bits(), "non-deterministic");
    }

    /// The verdict logic: a confident output is **accepted**, a balanced two-way
    /// output is **abstained** on, and a flat (out-of-distribution) output is
    /// **rejected** — for a guard calibrated to a mid-range threshold.
    #[test]
    fn guard_accepts_confident_abstains_ambiguous_rejects_ood() {
        // Calibration engineered so 1 − q̂ ≈ 0.43 (q̂ ≈ 0.57): p_true spread 0.40…0.95.
        let cal_probs: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let p = 0.40 + 0.55 * i as f32 / 19.0;
                vec![p, (1.0 - p) * 0.5, (1.0 - p) * 0.5]
            })
            .collect();
        let cal_labels = vec![0usize; 20];
        let guard = StatisticalGuard::calibrate(&cal_probs, &cal_labels, 0.1);

        assert_eq!(guard.decide(&[0.99, 0.005, 0.005]), GuardVerdict::Accept(0));
        assert_eq!(guard.decide(&[0.5, 0.5, 0.0]), GuardVerdict::Abstain);
        assert_eq!(guard.decide(&[0.34, 0.33, 0.33]), GuardVerdict::Reject);
    }
}
