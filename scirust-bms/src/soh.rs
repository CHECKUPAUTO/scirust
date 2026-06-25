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

    #[test]
    fn half_width_is_the_finite_sample_conformal_quantile() {
        // Split-conformal half-width = the ⌈(n+1)(1−α)⌉-th smallest |residual|.
        // residuals = [0.05, 0.10, 0.15, 0.20], n = 4, α = 0.25:
        //   k = ⌈(4+1)(1−0.25)⌉ = ⌈3.75⌉ = 4  ⇒ 4th smallest = 0.20.
        let g = ConformalSoh::calibrate(&[0.05, 0.10, 0.15, 0.20], 0.25);
        assert!(
            (g.half_width() - 0.20).abs() < 1e-6,
            "half_width = {}",
            g.half_width()
        );

        // residuals = [0.02,0.04,0.06,0.08], n = 4, α = 0.20:
        //   k = ⌈5·0.8⌉ = ⌈4⌉ = 4 ⇒ 4th smallest = 0.08.
        let g2 = ConformalSoh::calibrate(&[0.02, 0.04, 0.06, 0.08], 0.20);
        assert!(
            (g2.half_width() - 0.08).abs() < 1e-6,
            "half_width = {}",
            g2.half_width()
        );
    }

    #[test]
    fn interval_is_symmetric_and_clamped_to_unit_range() {
        // half_width = 0.08 (verified above).
        let g = ConformalSoh::calibrate(&[0.02, 0.04, 0.06, 0.08], 0.20);

        // Interior point: plain ±q, no clamping.
        let (lo, hi) = g.interval(0.50);
        assert!(
            (lo - 0.42).abs() < 1e-6 && (hi - 0.58).abs() < 1e-6,
            "({lo},{hi})"
        );

        // Near the top: upper bound saturates at 1.0.
        let (lo2, hi2) = g.interval(0.95);
        assert!((lo2 - 0.87).abs() < 1e-6, "lo {lo2}");
        assert_eq!(hi2, 1.0);

        // Near the bottom: lower bound saturates at 0.0.
        let (lo3, hi3) = g.interval(0.02);
        assert_eq!(lo3, 0.0);
        assert!((hi3 - 0.10).abs() < 1e-6, "hi {hi3}");
    }

    #[test]
    fn covers_matches_the_clamped_interval_membership() {
        // half_width = 0.20.
        let g = ConformalSoh::calibrate(&[0.05, 0.10, 0.15, 0.20], 0.25);
        let soh_hat = 0.60; // interval [0.40, 0.80]
        assert!(g.covers(soh_hat, 0.40)); // lower edge
        assert!(g.covers(soh_hat, 0.80)); // upper edge
        assert!(g.covers(soh_hat, 0.60)); // centre
        assert!(!g.covers(soh_hat, 0.39)); // just below
        assert!(!g.covers(soh_hat, 0.81)); // just above

        // Clamping shrinks the realised interval: at soh_hat = 0.90 the interval
        // is [0.70, 1.0] (upper clamped), so 1.0 is covered but 0.69 is not.
        assert!(g.covers(0.90, 1.0));
        assert!(!g.covers(0.90, 0.69));
    }

    #[test]
    fn too_few_calibration_points_give_a_vacuous_but_safe_interval() {
        // n = 3, α = 0.05 ⇒ k = ⌈4·0.95⌉ = ⌈3.8⌉ = 4 > 3 ⇒ q̂ = +∞.
        // The interval is then [0,1] after clamping and covers any valid SoH,
        // i.e. it never under-covers (it is merely uninformative).
        let g = ConformalSoh::calibrate(&[0.01, 0.02, 0.03], 0.05);
        assert!(g.half_width().is_infinite());
        let (lo, hi) = g.interval(0.5);
        assert_eq!((lo, hi), (0.0, 1.0));
        assert!(g.covers(0.5, 0.0));
        assert!(g.covers(0.5, 1.0));
    }
}
