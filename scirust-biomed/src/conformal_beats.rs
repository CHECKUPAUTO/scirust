//! Conformal prediction sets for ECG beat classification.
//!
//! A beat classifier (normal, PVC, PAC, …) is wrapped with split-conformal
//! calibration — reusing the audited
//! [`scirust_core::nn::conformal::ConformalClassifier`] — so its **prediction
//! set** contains the true beat type with probability `≥ 1 − α`, regardless of
//! how well-calibrated the underlying classifier is. In a diagnostic-support
//! setting the guaranteed-coverage set is the safe object to surface to a
//! clinician.

use scirust_core::nn::conformal::ConformalClassifier;

/// Conformal prediction-set wrapper for beat-class probability vectors.
pub struct ConformalBeats {
    inner: ConformalClassifier,
}

impl ConformalBeats {
    /// Calibrate on `(probability vector, true label)` pairs at miscoverage
    /// `alpha`.
    pub fn calibrate(cal_probs: &[Vec<f32>], cal_labels: &[usize], alpha: f32) -> Self {
        Self {
            inner: ConformalClassifier::calibrate(cal_probs, cal_labels, alpha),
        }
    }

    /// The guaranteed-coverage label set for a beat's class probabilities.
    pub fn predict_set(&self, probs: &[f32]) -> Vec<usize> {
        self.inner.predict_set(probs)
    }

    /// Whether the prediction set for `probs` contains `y_true`.
    pub fn covers(&self, probs: &[f32], y_true: usize) -> bool {
        self.inner.covers(probs, y_true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn u01(&mut self) -> f32 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 + 0.5) / ((1u64 << 24) as f32)
        }
    }

    const K: usize = 4; // beat classes

    /// A noisy 4-class beat classifier: softmax of logits peaked at the true
    /// label by `margin`, with random perturbation (imperfect, like reality).
    fn sample(rng: &mut Rng, margin: f32) -> (Vec<f32>, usize) {
        let label = (rng.u01() * K as f32) as usize % K;
        let mut logits = [0.0f32; K];
        for (j, l) in logits.iter_mut().enumerate()
        {
            *l = (rng.u01() - 0.5) * 2.0;
            if j == label
            {
                *l += margin;
            }
        }
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        (exps.iter().map(|&e| e / sum).collect(), label)
    }

    #[test]
    fn prediction_sets_cover_at_least_one_minus_alpha() {
        let mut rng = Rng::new(0xEC6);
        let alpha = 0.1;
        let cal: Vec<(Vec<f32>, usize)> = (0..3000).map(|_| sample(&mut rng, 2.0)).collect();
        let cal_probs: Vec<Vec<f32>> = cal.iter().map(|(p, _)| p.clone()).collect();
        let cal_labels: Vec<usize> = cal.iter().map(|(_, l)| *l).collect();
        let cb = ConformalBeats::calibrate(&cal_probs, &cal_labels, alpha);

        let (n, mut covered) = (6000, 0usize);
        for _ in 0..n
        {
            let (probs, label) = sample(&mut rng, 2.0);
            if cb.covers(&probs, label)
            {
                covered += 1;
            }
        }
        let cov = covered as f64 / n as f64;
        assert!(
            cov >= 1.0 - alpha as f64 - 0.03,
            "coverage {cov} < 1-alpha {}",
            1.0 - alpha
        );
    }
}
