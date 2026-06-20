//! Hotelling T² — multivariate process monitoring.
//!
//! A single statistic `T² = (x − μ)ᵀ Σ⁻¹ (x − μ)` watches several correlated
//! quality variables at once, catching shifts that per-variable charts miss
//! (because they account for the correlation). Compare against an upper control
//! limit — a χ²ₚ critical value for phase-II individuals.

use scirust_estimation::Mat;
use serde::{Deserialize, Serialize};

/// Hotelling T² monitor (mean vector + inverse covariance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotellingT2 {
    mean: Vec<f64>,
    cov_inv: Mat,
}

impl HotellingT2 {
    /// Fit on in-control multivariate samples (each row a length-p observation).
    /// Returns `None` if the covariance is singular.
    pub fn fit(data: &[Vec<f64>]) -> Option<Self> {
        let n = data.len();
        if n < 2
        {
            return None;
        }
        let p = data[0].len();
        let mut mean = vec![0.0; p];
        for row in data
        {
            for (m, &v) in mean.iter_mut().zip(row)
            {
                *m += v;
            }
        }
        for m in mean.iter_mut()
        {
            *m /= n as f64;
        }
        let mut cov = Mat::zeros(p, p);
        for row in data
        {
            for i in 0..p
            {
                for j in 0..p
                {
                    cov.data[i * p + j] += (row[i] - mean[i]) * (row[j] - mean[j]);
                }
            }
        }
        for v in cov.data.iter_mut()
        {
            *v /= (n - 1) as f64;
        }
        Some(Self {
            mean,
            cov_inv: cov.inverse()?,
        })
    }

    /// `T²` statistic for an observation.
    pub fn t2(&self, x: &[f64]) -> f64 {
        let d: Vec<f64> = x.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
        let sd = self.cov_inv.matvec(&d);
        d.iter().zip(&sd).map(|(a, b)| a * b).sum()
    }

    /// Whether the observation exceeds the upper control limit `ucl`
    /// (a χ²ₚ critical value).
    pub fn is_out_of_control(&self, x: &[f64], ucl: f64) -> bool {
        self.t2(x) > ucl
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
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.u01().max(1e-9), self.u01());
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn flags_correlated_out_of_control_point() {
        // Two correlated variables: x2 ≈ x1 + small noise.
        let mut rng = Rng::new(0x7C2);
        let data: Vec<Vec<f64>> = (0..2000)
            .map(|_| {
                let a = rng.normal();
                vec![a, a + 0.3 * rng.normal()]
            })
            .collect();
        let h = HotellingT2::fit(&data).expect("nonsingular");
        let ucl = 9.21; // χ²(2) at α = 0.01

        // T² at the mean is 0.
        assert!(h.t2(&h.mean) < 1e-6);
        // An in-control sample is (almost surely) below the UCL.
        assert!(!h.is_out_of_control(&[0.2, 0.25], ucl));
        // A point that violates the correlation (x1 high, x2 low) is far in
        // Mahalanobis distance even though each coordinate is individually modest.
        assert!(
            h.is_out_of_control(&[2.5, -2.5], ucl),
            "T² {}",
            h.t2(&[2.5, -2.5])
        );
    }
}
