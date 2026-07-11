//! Deterministic linear Kalman filter.
//!
//! Standard predict/update for `x_{k+1} = F·x_k + w` (process noise covariance
//! `Q`) and `z_k = H·x_k + v` (measurement noise covariance `R`). All arithmetic
//! is fixed-order `f64`, so a run is bit-reproducible — the determinism the rest
//! of SciRust guarantees, applied to state estimation.

use crate::linalg::Mat;
use serde::{Deserialize, Serialize};

/// Linear Kalman filter over an `n`-dim state and `m`-dim measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    x: Vec<f64>,
    p: Mat,
    f: Mat,
    q: Mat,
    h: Mat,
    r: Mat,
}

impl KalmanFilter {
    /// Build from initial state `x0`, initial covariance `p0`, and the model
    /// matrices `F` (n×n), `Q` (n×n), `H` (m×n), `R` (m×m).
    pub fn new(x0: Vec<f64>, p0: Mat, f: Mat, q: Mat, h: Mat, r: Mat) -> Self {
        let n = x0.len();
        assert!(p0.rows == n && p0.cols == n, "P0 must be n×n");
        assert!(f.rows == n && f.cols == n, "F must be n×n");
        assert!(q.rows == n && q.cols == n, "Q must be n×n");
        assert_eq!(h.cols, n, "H must be m×n");
        assert!(r.rows == h.rows && r.cols == h.rows, "R must be m×m");
        Self {
            x: x0,
            p: p0,
            f,
            q,
            h,
            r,
        }
    }

    /// Current state estimate.
    pub fn state(&self) -> &[f64] {
        &self.x
    }

    /// Current state covariance.
    pub fn covariance(&self) -> &Mat {
        &self.p
    }

    /// Time update: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
    pub fn predict(&mut self) {
        self.x = self.f.matvec(&self.x);
        self.p = self.f.matmul(&self.p).matmul(&self.f.t()).add(&self.q);
    }

    /// Measurement update with observation `z` (length `m`). Returns `false`
    /// (leaving the state unchanged) if the innovation covariance is singular.
    pub fn update(&mut self, z: &[f64]) -> bool {
        let hx = self.h.matvec(&self.x);
        let y: Vec<f64> = z.iter().zip(&hx).map(|(zi, hi)| zi - hi).collect();

        let ht = self.h.t();
        let s = self.h.matmul(&self.p).matmul(&ht).add(&self.r);
        let Some(s_inv) = s.inverse()
        else
        {
            return false;
        };
        // Kalman gain K = P Hᵀ S⁻¹.
        let k = self.p.matmul(&ht).matmul(&s_inv);

        // x ← x + K y
        let ky = k.matvec(&y);
        for (xi, kyi) in self.x.iter_mut().zip(&ky)
        {
            *xi += kyi;
        }
        // P ← (I − K H) P (I − K H)ᵀ + K R Kᵀ — the Joseph form. Algebraically
        // equivalent to the shorter (I − K H) P, but the short form only
        // stays symmetric and positive semi-definite if K is exactly the
        // optimal Kalman gain; any finite-precision drift (or a
        // slightly-off K from a caller) can otherwise produce an
        // asymmetric or indefinite P after many updates. Joseph's form is
        // symmetric and PSD by construction for any K (Bucy & Joseph 1968;
        // Grewal & Andrews, Kalman Filtering, §6.3.4) — the standard choice
        // for a filter meant to run over long, unattended horizons.
        let n = self.x.len();
        let kh = k.matmul(&self.h);
        let i_kh = Mat::identity(n).sub(&kh);
        let i_kh_t = i_kh.t();
        let krk_t = k.matmul(&self.r).matmul(&k.t());
        self.p = i_kh.matmul(&self.p).matmul(&i_kh_t).add(&krk_t);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic Gaussian-ish noise via splitmix64 + Box–Muller.
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
        fn normal(&mut self, sd: f64) -> f64 {
            let u1 = self.u01();
            let u2 = self.u01();
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    /// Constant-velocity 1-D tracker: state [pos, vel], measure position.
    fn cv_filter(dt: f64) -> KalmanFilter {
        let f = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let q = Mat::new(2, 2, vec![1e-4, 0.0, 0.0, 1e-4]);
        let h = Mat::new(1, 2, vec![1.0, 0.0]);
        let r = Mat::new(1, 1, vec![0.25]); // measurement variance (sd 0.5)
        let p0 = Mat::diag(&[1.0, 1.0]);
        KalmanFilter::new(vec![0.0, 0.0], p0, f, q, h, r)
    }

    #[test]
    fn tracks_constant_velocity_and_shrinks_covariance() {
        let dt = 1.0;
        let mut kf = cv_filter(dt);
        let mut rng = Rng::new(0xCAFE);
        let (true_v, mut true_x) = (1.0, 0.0);

        let trace0 = kf.covariance().get(0, 0) + kf.covariance().get(1, 1);
        let mut sq_err = 0.0;
        let steps = 100;
        for k in 0..steps
        {
            true_x += true_v * dt;
            let z = true_x + rng.normal(0.5);
            kf.predict();
            assert!(kf.update(&[z]));
            if k >= 50
            {
                let e = kf.state()[0] - true_x;
                sq_err += e * e;
            }
        }
        let rmse = (sq_err / 50.0).sqrt();
        let trace_end = kf.covariance().get(0, 0) + kf.covariance().get(1, 1);
        assert!(rmse < 0.5, "position RMSE {rmse} too high");
        assert!(trace_end < trace0, "covariance did not shrink");
        // Velocity learned from position-only measurements.
        assert!(
            (kf.state()[1] - true_v).abs() < 0.3,
            "vel {}",
            kf.state()[1]
        );
    }

    #[test]
    fn joseph_form_keeps_covariance_symmetric_and_psd_under_a_stiff_filter() {
        // Invariant test for a P1 audit finding (defensive fix, not a
        // reproduced failure): in double precision this specific 2-state
        // filter's short update (I − K H) P does not visibly drift within
        // 500 (or even 5e5, empirically checked) iterations — the asymmetry
        // Joseph's form guards against is a worst-case risk (imperfect or
        // non-exactly-optimal K, long unattended horizons, lower precision),
        // not something f64 rounding alone reliably exposes here. This test
        // locks in the property the Joseph form *guarantees by construction*
        // — P stays symmetric and PSD — under a stiff, near-singular
        // measurement. See Bucy & Joseph 1968; Grewal & Andrews, Kalman
        // Filtering, §6.3.4.
        let f = Mat::identity(2);
        let q = Mat::diag(&[1e-9, 1e-9]);
        let h = Mat::new(1, 2, vec![1.0, 0.0]);
        let r = Mat::new(1, 1, vec![1e-8]); // very confident measurement
        let p0 = Mat::diag(&[1e6, 1e6]);
        let mut kf = KalmanFilter::new(vec![0.0, 0.0], p0, f, q, h, r);
        for k in 0..500
        {
            kf.predict();
            assert!(kf.update(&[k as f64 * 1e-3]));
            let p = kf.covariance();
            assert!(
                p.get(0, 0) >= -1e-9 && p.get(1, 1) >= -1e-9,
                "negative variance at step {k}: diag = ({}, {})",
                p.get(0, 0),
                p.get(1, 1)
            );
            let asym = (p.get(0, 1) - p.get(1, 0)).abs();
            assert!(asym < 1e-9, "asymmetric covariance at step {k}: {asym:e}");
        }
    }

    #[test]
    fn run_is_bit_reproducible() {
        let run = || {
            let mut kf = cv_filter(1.0);
            let mut rng = Rng::new(7);
            let mut x = 0.0;
            for _ in 0..50
            {
                x += 1.0;
                let z = x + rng.normal(0.5);
                kf.predict();
                kf.update(&[z]);
            }
            kf.state().to_vec()
        };
        assert_eq!(run(), run());
    }
}
