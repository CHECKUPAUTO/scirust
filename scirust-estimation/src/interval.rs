//! Set-membership (interval) state estimation with a **containment guarantee**.
//!
//! Where the Kalman filter gives a *probabilistic* estimate, a set-membership
//! filter gives a *guaranteed* one: for a state whose components drift by a
//! bounded rate and are measured with bounded error, it maintains a box
//! `[lo, hi]` that **provably contains the true state** at every step — as long
//! as the declared bounds actually hold.
//!
//! Invariant (by induction): if `x_k ∈ [lo, hi]`, then after
//! - `predict`: `x_{k+1} ∈ [lo − drift, hi + drift]` since `|x_{k+1} − x_k| ≤ drift`;
//! - `update(z)`: `x_{k+1} ∈ [z − meas, z + meas]` since `|z − x_{k+1}| ≤ meas`,
//!   and intersecting two sets that both contain `x_{k+1}` still contains it.
//!
//! If an update makes a component's interval empty (`lo > hi`), the declared
//! bounds were violated — a detectable inconsistency, surfaced by [`is_valid`].
//!
//! [`is_valid`]: IntervalFilter::is_valid

use serde::{Deserialize, Serialize};

/// Component-wise interval state estimator with a containment guarantee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalFilter {
    lo: Vec<f64>,
    hi: Vec<f64>,
    drift_max: Vec<f64>,
    meas_max: Vec<f64>,
}

impl IntervalFilter {
    /// Initial box `[lo, hi]` (must contain the true initial state), the
    /// per-component maximum drift per step, and the per-component maximum
    /// measurement error. All slices share the state dimension.
    pub fn new(lo: Vec<f64>, hi: Vec<f64>, drift_max: Vec<f64>, meas_max: Vec<f64>) -> Self {
        let n = lo.len();
        assert!(
            hi.len() == n && drift_max.len() == n && meas_max.len() == n,
            "IntervalFilter: dimension mismatch"
        );
        assert!(lo.iter().zip(&hi).all(|(l, h)| l <= h), "lo ≤ hi required");
        Self {
            lo,
            hi,
            drift_max,
            meas_max,
        }
    }

    /// Lower bounds of the current box.
    pub fn lower(&self) -> &[f64] {
        &self.lo
    }

    /// Upper bounds of the current box.
    pub fn upper(&self) -> &[f64] {
        &self.hi
    }

    /// Per-component box width `hi − lo`.
    pub fn width(&self) -> Vec<f64> {
        self.lo.iter().zip(&self.hi).map(|(l, h)| h - l).collect()
    }

    /// Whether every component interval is non-empty (`lo ≤ hi`). `false` means
    /// the declared bounds were violated.
    pub fn is_valid(&self) -> bool {
        self.lo.iter().zip(&self.hi).all(|(l, h)| l <= h)
    }

    /// Whether the box currently contains the point `x`.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.lo.len()
            && x.iter()
                .zip(self.lo.iter().zip(&self.hi))
                .all(|(&xi, (&l, &h))| xi >= l && xi <= h)
    }

    /// Time update: inflate each interval by the maximum drift.
    pub fn predict(&mut self) {
        for i in 0..self.lo.len()
        {
            self.lo[i] -= self.drift_max[i];
            self.hi[i] += self.drift_max[i];
        }
    }

    /// Measurement update: intersect with `[z − meas, z + meas]` per component.
    pub fn update(&mut self, z: &[f64]) {
        for (i, &zi) in z.iter().enumerate()
        {
            self.lo[i] = self.lo[i].max(zi - self.meas_max[i]);
            self.hi[i] = self.hi[i].min(zi + self.meas_max[i]);
        }
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
    fn box_always_contains_true_state() {
        let drift = 0.05;
        let meas = 0.2;
        let lo = vec![-1.0, -1.0];
        let hi = vec![1.0, 1.0];
        let mut filt = IntervalFilter::new(lo, hi, vec![drift; 2], vec![meas; 2]);
        let mut rng = Rng::new(0x5E7);
        let mut truth = [0.0_f64, 0.0];

        for _ in 0..1000
        {
            // True drift within the declared bound.
            truth[0] += rng.u(-drift, drift);
            truth[1] += rng.u(-drift, drift);
            filt.predict();
            assert!(filt.contains(&truth), "lost the true state after predict");
            // Measurement within the declared error bound.
            let z = [truth[0] + rng.u(-meas, meas), truth[1] + rng.u(-meas, meas)];
            filt.update(&z);
            assert!(filt.is_valid(), "box went empty — bounds violated?");
            assert!(filt.contains(&truth), "lost the true state after update");
        }
        // After convergence the box is tight: width ≤ 2·(drift + meas).
        for w in filt.width()
        {
            assert!(w <= 2.0 * (drift + meas) + 1e-9, "box too wide: {w}");
        }
    }

    #[test]
    fn detects_bound_violation() {
        let mut filt = IntervalFilter::new(vec![0.0], vec![0.0], vec![0.0], vec![0.1]);
        // The state is pinned to 0 (zero drift), but a measurement says 5 ± 0.1
        // — inconsistent with the box {0}. The intersection must go empty.
        filt.predict();
        filt.update(&[5.0]);
        assert!(!filt.is_valid(), "should flag inconsistent measurement");
    }
}
