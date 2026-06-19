//! Deterministic Extended Kalman Filter.
//!
//! Same predict/update structure as the linear filter, but the (possibly
//! nonlinear) transition `f` and measurement `h` are supplied as closures
//! together with their Jacobians, evaluated at the current estimate. Pure
//! fixed-order `f64`, so a run is bit-reproducible.

use crate::linalg::Mat;
use serde::{Deserialize, Serialize};

/// Extended Kalman filter over an `n`-dim state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ekf {
    x: Vec<f64>,
    p: Mat,
    q: Mat,
    r: Mat,
}

impl Ekf {
    /// Build from initial state/covariance and process/measurement noise.
    pub fn new(x0: Vec<f64>, p0: Mat, q: Mat, r: Mat) -> Self {
        let n = x0.len();
        assert!(p0.rows == n && p0.cols == n, "P0 must be n×n");
        assert!(q.rows == n && q.cols == n, "Q must be n×n");
        Self { x: x0, p: p0, q, r }
    }

    pub fn state(&self) -> &[f64] {
        &self.x
    }

    pub fn covariance(&self) -> &Mat {
        &self.p
    }

    /// Time update with transition `f` and its Jacobian `f_jac` (n×n), both
    /// evaluated at the current state: `x ← f(x)`, `P ← Fₓ·P·Fₓᵀ + Q`.
    pub fn predict<F, J>(&mut self, f: F, f_jac: J)
    where
        F: Fn(&[f64]) -> Vec<f64>,
        J: Fn(&[f64]) -> Mat,
    {
        let fx_jac = f_jac(&self.x);
        self.x = f(&self.x);
        self.p = fx_jac.matmul(&self.p).matmul(&fx_jac.t()).add(&self.q);
    }

    /// Measurement update with observation `z`, measurement function `h` and its
    /// Jacobian `h_jac` (m×n). Returns `false` if the innovation covariance is
    /// singular (state left unchanged).
    pub fn update<H, J>(&mut self, z: &[f64], h: H, h_jac: J) -> bool
    where
        H: Fn(&[f64]) -> Vec<f64>,
        J: Fn(&[f64]) -> Mat,
    {
        let hx = h(&self.x);
        let y: Vec<f64> = z.iter().zip(&hx).map(|(zi, hi)| zi - hi).collect();
        let hj = h_jac(&self.x);
        let hjt = hj.t();
        let s = hj.matmul(&self.p).matmul(&hjt).add(&self.r);
        let Some(s_inv) = s.inverse()
        else
        {
            return false;
        };
        let k = self.p.matmul(&hjt).matmul(&s_inv);
        let ky = k.matvec(&y);
        for (xi, kyi) in self.x.iter_mut().zip(&ky)
        {
            *xi += kyi;
        }
        let n = self.x.len();
        let kh = k.matmul(&hj);
        self.p = Mat::identity(n).sub(&kh).matmul(&self.p);
        true
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
        fn normal(&mut self, sd: f64) -> f64 {
            let u1 = self.u01();
            let u2 = self.u01();
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    // Range to a beacon and its gradient w.r.t. (px, py).
    fn range(px: f64, py: f64, bx: f64, by: f64) -> f64 {
        ((px - bx).powi(2) + (py - by).powi(2)).sqrt()
    }

    /// Constant-velocity target tracked from two noisy range beacons — linear
    /// dynamics, nonlinear (range) measurement: a textbook EKF.
    #[test]
    fn tracks_target_from_two_range_beacons() {
        let dt = 1.0;
        let (b1, b2) = ((0.0, 0.0), (20.0, 0.0));
        // State [px, py, vx, vy].
        let f_mat = Mat::new(
            4,
            4,
            vec![
                1.0, 0.0, dt, 0.0, //
                0.0, 1.0, 0.0, dt, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
        );
        let f_mat_c = f_mat.clone();
        let f = move |x: &[f64]| f_mat_c.matvec(x);
        let f_jac = move |_x: &[f64]| f_mat.clone();

        let h = |x: &[f64]| vec![range(x[0], x[1], b1.0, b1.1), range(x[0], x[1], b2.0, b2.1)];
        let h_jac = |x: &[f64]| {
            let d1 = range(x[0], x[1], b1.0, b1.1).max(1e-9);
            let d2 = range(x[0], x[1], b2.0, b2.1).max(1e-9);
            Mat::new(
                2,
                4,
                vec![
                    (x[0] - b1.0) / d1,
                    (x[1] - b1.1) / d1,
                    0.0,
                    0.0,
                    (x[0] - b2.0) / d2,
                    (x[1] - b2.1) / d2,
                    0.0,
                    0.0,
                ],
            )
        };

        let q = Mat::diag(&[1e-4, 1e-4, 1e-4, 1e-4]);
        let r = Mat::diag(&[0.04, 0.04]); // range sd 0.2
        let p0 = Mat::diag(&[4.0, 4.0, 4.0, 4.0]);
        // Initial guess offset from truth; truth starts at (5, 8) moving (0.5, 0.3).
        let mut ekf = Ekf::new(vec![6.5, 6.5, 0.0, 0.0], p0, q, r);

        let mut rng = Rng::new(0xBEEF);
        let (mut px, mut py, vx, vy) = (5.0, 8.0, 0.5, 0.3);
        let mut last_err = f64::INFINITY;
        for _ in 0..120
        {
            px += vx * dt;
            py += vy * dt;
            let z = vec![
                range(px, py, b1.0, b1.1) + rng.normal(0.2),
                range(px, py, b2.0, b2.1) + rng.normal(0.2),
            ];
            ekf.predict(&f, &f_jac);
            assert!(ekf.update(&z, h, h_jac));
            last_err = ((ekf.state()[0] - px).powi(2) + (ekf.state()[1] - py).powi(2)).sqrt();
        }
        assert!(last_err < 1.0, "final position error {last_err} too high");
    }
}
