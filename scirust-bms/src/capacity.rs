//! Recursive capacity (State-of-Health) estimation.
//!
//! Capacity fades as a cell ages, and SoH = present capacity / rated capacity.
//! Between two rest points the charge moved (`∫I dt`, ampere-seconds) equals
//! `Q · ΔSoC`, so each charge/discharge segment is one noisy measurement of the
//! capacity `Q`. [`RlsCapacity`] tracks `Q` online by **recursive least squares**
//! with a forgetting factor (so slow fade is followed), recovering SoH without a
//! full discharge test. Deterministic `f64`.

use serde::{Deserialize, Serialize};

/// Recursive-least-squares estimator of usable capacity (ampere-seconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlsCapacity {
    q_hat: f64,
    p: f64,
    lambda: f64,
}

impl RlsCapacity {
    /// Initialise at capacity guess `q0_as` (ampere-seconds) with forgetting
    /// factor `lambda ∈ (0, 1]` (1 = no forgetting). `p0` sets the initial
    /// uncertainty (larger = faster initial adaptation).
    pub fn new(q0_as: f64, lambda: f64, p0: f64) -> Self {
        Self {
            q_hat: q0_as,
            p: p0,
            lambda: lambda.clamp(1e-3, 1.0),
        }
    }

    /// One segment: charge moved `charge_as` (|∫I dt|, ampere-seconds) over an
    /// absolute SoC change `delta_soc` (`0..1`). Returns the updated capacity.
    pub fn update(&mut self, delta_soc: f64, charge_as: f64) -> f64 {
        let x = delta_soc;
        if x.abs() < 1e-9
        {
            return self.q_hat;
        }
        let k = self.p * x / (self.lambda + x * self.p * x);
        let err = charge_as - self.q_hat * x;
        self.q_hat += k * err;
        self.p = (self.p - k * x * self.p) / self.lambda;
        self.q_hat
    }

    /// Present capacity estimate (ampere-seconds).
    pub fn capacity_as(&self) -> f64 {
        self.q_hat
    }

    /// State of Health = present capacity / `nominal_as`, clamped to `[0, 1.5]`.
    pub fn soh(&self, nominal_as: f64) -> f64 {
        if nominal_as <= 0.0
        {
            return 0.0;
        }
        (self.q_hat / nominal_as).clamp(0.0, 1.5)
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
        fn u(&mut self, lo: f64, hi: f64) -> f64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u01 = ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
            lo + (hi - lo) * u01
        }
    }

    #[test]
    fn recovers_faded_capacity_from_segments() {
        let nominal = 2.0 * 3600.0; // 2.0 Ah rated
        let true_cap = 1.8 * 3600.0; // faded to 90% SoH
        // Start the estimator at the (wrong) rated capacity.
        let mut rls = RlsCapacity::new(nominal, 0.97, 1e6);
        let mut rng = Rng::new(0xCAFE);

        for _ in 0..60
        {
            let dsoc = rng.u(0.15, 0.5);
            let charge = true_cap * dsoc + rng.u(-50.0, 50.0); // small coulomb noise
            rls.update(dsoc, charge);
        }
        let soh = rls.soh(nominal);
        assert!((soh - 0.9).abs() < 0.02, "SoH {soh} (want ~0.90)");
        assert!((rls.capacity_as() - true_cap).abs() < 0.02 * true_cap);
    }

    #[test]
    fn zero_soc_change_is_ignored() {
        let mut rls = RlsCapacity::new(7200.0, 0.99, 1.0);
        let before = rls.capacity_as();
        assert_eq!(rls.update(0.0, 123.0), before);
    }

    #[test]
    fn single_segment_with_huge_p_recovers_charge_over_dsoc() {
        // RLS gain with P→∞ is k = 1/x, so one update sets q̂ = y/x exactly:
        // here y = 1944 As over ΔSoC = 0.3 ⇒ q̂ = 6480 As. p0 = 1e12 makes the
        // λ term negligible, so the result is exact to ~7 significant figures.
        let mut rls = RlsCapacity::new(7200.0, 1.0, 1e12);
        let q = rls.update(0.3, 6480.0 * 0.3);
        assert!((q - 6480.0).abs() < 1e-3, "q̂ = {q}, want 6480");
        assert!((rls.capacity_as() - 6480.0).abs() < 1e-3);
    }

    #[test]
    fn noiseless_segments_converge_to_true_capacity() {
        // Every segment is the exact relation charge = Q·ΔSoC, so the estimate
        // must converge to Q regardless of the (wrong) starting guess. With a
        // large p0 the initial-guess prior (weight 1/p0) is negligible against
        // the accumulated regressor energy Σx², so the fit reaches Q tightly.
        let q_true = 1.7 * 3600.0; // 6120 As
        let mut rls = RlsCapacity::new(2.4 * 3600.0, 1.0, 1e12);
        let dsocs = [0.20, 0.35, 0.15, 0.40, 0.25, 0.30, 0.18, 0.42];
        for &d in dsocs.iter().cycle().take(40)
        {
            rls.update(d, q_true * d);
        }
        assert!(
            (rls.capacity_as() - q_true).abs() < 1e-3,
            "got {}",
            rls.capacity_as()
        );
        assert!((rls.soh(2.4 * 3600.0) - 1.7 / 2.4).abs() < 1e-6);
    }

    #[test]
    fn soh_is_ratio_and_is_clamped() {
        // SoH = capacity / nominal, clamped to [0, 1.5].
        let nominal = 2.0 * 3600.0; // 7200 As
        let rls = RlsCapacity::new(0.9 * nominal, 1.0, 1.0); // 90 %
        assert!((rls.soh(nominal) - 0.9).abs() < 1e-12);

        // 300 % capacity saturates at the 1.5 ceiling.
        let over = RlsCapacity::new(3.0 * nominal, 1.0, 1.0);
        assert_eq!(over.soh(nominal), 1.5);

        // Invalid nominal ⇒ defined as 0 SoH (no divide-by-zero / sign flip).
        assert_eq!(rls.soh(0.0), 0.0);
        assert_eq!(rls.soh(-7200.0), 0.0);
    }

    #[test]
    fn negligible_dsoc_leaves_state_untouched() {
        // Below the 1e-9 guard the update is a no-op for BOTH q̂ and the internal
        // covariance, so a subsequent real segment behaves identically to one on
        // a fresh estimator.
        let mk = || RlsCapacity::new(7200.0, 0.98, 5.0);
        let mut touched = mk();
        touched.update(1e-12, 999.0); // ignored
        touched.update(1e-10, -42.0); // ignored
        let mut fresh = mk();
        let real = (0.3, 6480.0 * 0.3);
        assert_eq!(touched.update(real.0, real.1), fresh.update(real.0, real.1));
    }
}
