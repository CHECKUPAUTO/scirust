//! Distribution-free State-of-Health bounds.
//!
//! Wraps a point SoH estimate (0 = end-of-life, 1 = fresh) with a split-conformal
//! interval — reusing [`scirust_pdm::ConformalRul`] — so the bound covers the true
//! SoH with probability `≥ 1 − α`, clamped to the physical range `[0, 1]`.

use scirust_pdm::ConformalRul;
use serde::{Deserialize, Serialize};

/// Conformal SoH interval calibrated on `|SoH_true − SoH_pred|` residuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalSoh {
    inner: ConformalRul,
}

impl ConformalSoh {
    /// Calibrate on absolute SoH residuals (unitless, in `[0,1]`) at miscoverage
    /// `alpha`.
    pub fn calibrate(abs_residuals: &[f64], alpha: f64) -> Self {
        Self {
            inner: ConformalRul::calibrate(abs_residuals, alpha),
        }
    }

    /// Conformal half-width.
    pub fn half_width(&self) -> f64 {
        self.inner.half_width()
    }

    /// Guaranteed-coverage interval around `soh_hat`, clamped to `[0, 1]`.
    pub fn interval(&self, soh_hat: f64) -> (f64, f64) {
        let (lo, hi) = self.inner.interval(soh_hat);
        (lo.max(0.0), hi.min(1.0))
    }

    /// Whether the interval around `soh_hat` covers the realized `soh_true`.
    pub fn covers(&self, soh_hat: f64, soh_true: f64) -> bool {
        let (lo, hi) = self.interval(soh_hat);
        soh_true >= lo && soh_true <= hi
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
        fn u01(&mut self) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn soh_interval_covers_within_unit_range() {
        let mut rng = Rng::new(0xB17);
        let alpha = 0.1;
        // SoH estimation errors ~ ±0.04 (uniform), non-Gaussian.
        let cal: Vec<f64> = (0..2000).map(|_| (rng.u01() - 0.5).abs() * 0.08).collect();
        let g = ConformalSoh::calibrate(&cal, alpha);

        let (n, mut covered) = (8000, 0usize);
        for _ in 0..n
        {
            let soh_true = 0.5 + 0.5 * rng.u01(); // healthy-ish range
            let err = (rng.u01() - 0.5) * 0.08;
            let soh_hat = (soh_true - err).clamp(0.0, 1.0);
            // Interval stays within [0,1].
            let (lo, hi) = g.interval(soh_hat);
            assert!(lo >= 0.0 && hi <= 1.0);
            if g.covers(soh_hat, soh_true)
            {
                covered += 1;
            }
        }
        let cov = covered as f64 / n as f64;
        assert!(cov >= 1.0 - alpha - 0.03, "SoH coverage {cov} < 1-alpha");
    }
}
