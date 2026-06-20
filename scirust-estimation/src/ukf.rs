//! Deterministic Unscented Kalman Filter.
//!
//! Where the [`crate::ekf::Ekf`] linearizes via Jacobians, the UKF propagates a
//! deterministic set of sigma points through the true nonlinear `f`/`h` — no
//! Jacobians required, and second-order accurate. Pure fixed-order `f64`, so a
//! run is bit-reproducible.

use crate::linalg::Mat;
use serde::{Deserialize, Serialize};

/// Unscented Kalman filter over an `n`-dim state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ukf {
    x: Vec<f64>,
    p: Mat,
    q: Mat,
    r: Mat,
    alpha: f64,
    beta: f64,
    kappa: f64,
}

/// `acc += w · (a ⊗ b)` (outer product), `a` and `b` column vectors.
fn add_outer(acc: &mut Mat, w: f64, a: &[f64], b: &[f64]) {
    for (i, &ai) in a.iter().enumerate()
    {
        for (j, &bj) in b.iter().enumerate()
        {
            acc.data[i * acc.cols + j] += w * ai * bj;
        }
    }
}

impl Ukf {
    /// Build with the usual defaults (`α=1e-3`, `β=2`, `κ=0`).
    pub fn new(x0: Vec<f64>, p0: Mat, q: Mat, r: Mat) -> Self {
        Self {
            x: x0,
            p: p0,
            q,
            r,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// Override the unscented spread parameters.
    pub fn with_params(mut self, alpha: f64, beta: f64, kappa: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.kappa = kappa;
        self
    }

    pub fn state(&self) -> &[f64] {
        &self.x
    }

    pub fn covariance(&self) -> &Mat {
        &self.p
    }

    /// `(λ, mean weights, covariance weights)`.
    fn weights(&self) -> (f64, Vec<f64>, Vec<f64>) {
        let n = self.x.len();
        let nf = n as f64;
        let lambda = self.alpha * self.alpha * (nf + self.kappa) - nf;
        let c = nf + lambda;
        let mut wm = vec![1.0 / (2.0 * c); 2 * n + 1];
        let mut wc = wm.clone();
        wm[0] = lambda / c;
        wc[0] = lambda / c + (1.0 - self.alpha * self.alpha + self.beta);
        (lambda, wm, wc)
    }

    /// `2n+1` sigma points: `x0`, then `x ± column_i(√((n+λ)P))`.
    fn sigma_points(x: &[f64], p: &Mat, lambda: f64) -> Option<Vec<Vec<f64>>> {
        let n = x.len();
        let c = n as f64 + lambda;
        let scaled = Mat::new(n, n, p.data.iter().map(|v| v * c).collect());
        let l = scaled.cholesky()?;
        let mut pts = Vec::with_capacity(2 * n + 1);
        pts.push(x.to_vec());
        for i in 0..n
        {
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            for (row, (pl, mi)) in plus.iter_mut().zip(minus.iter_mut()).enumerate()
            {
                let off = l.get(row, i);
                *pl += off;
                *mi -= off;
            }
            pts.push(plus);
            pts.push(minus);
        }
        Some(pts)
    }

    /// Time update through nonlinear `f`. Returns `false` if `P` is not
    /// positive-definite.
    pub fn predict<F>(&mut self, f: F) -> bool
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = self.x.len();
        let (lambda, wm, wc) = self.weights();
        let Some(pts) = Self::sigma_points(&self.x, &self.p, lambda)
        else
        {
            return false;
        };
        let prop: Vec<Vec<f64>> = pts.iter().map(|s| f(s)).collect();

        let mut x_pred = vec![0.0; n];
        for (w, y) in wm.iter().zip(&prop)
        {
            for (xp, &yi) in x_pred.iter_mut().zip(y)
            {
                *xp += w * yi;
            }
        }
        let mut p_pred = self.q.clone();
        for (w, y) in wc.iter().zip(&prop)
        {
            let dy: Vec<f64> = y.iter().zip(&x_pred).map(|(a, b)| a - b).collect();
            add_outer(&mut p_pred, *w, &dy, &dy);
        }
        self.x = x_pred;
        self.p = p_pred;
        true
    }

    /// Measurement update through nonlinear `h` with observation `z`. Returns
    /// `false` if `P` is non-PD or the innovation covariance is singular.
    pub fn update<H>(&mut self, z: &[f64], h: H) -> bool
    where
        H: Fn(&[f64]) -> Vec<f64>,
    {
        let n = self.x.len();
        let m = z.len();
        let (lambda, wm, wc) = self.weights();
        let Some(pts) = Self::sigma_points(&self.x, &self.p, lambda)
        else
        {
            return false;
        };
        let zs: Vec<Vec<f64>> = pts.iter().map(|s| h(s)).collect();

        let mut z_pred = vec![0.0; m];
        for (w, zi) in wm.iter().zip(&zs)
        {
            for (zp, &v) in z_pred.iter_mut().zip(zi)
            {
                *zp += w * v;
            }
        }
        let mut s = self.r.clone();
        let mut pxz = Mat::zeros(n, m);
        for (idx, w) in wc.iter().enumerate()
        {
            let dz: Vec<f64> = zs[idx].iter().zip(&z_pred).map(|(a, b)| a - b).collect();
            let dx: Vec<f64> = pts[idx].iter().zip(&self.x).map(|(a, b)| a - b).collect();
            add_outer(&mut s, *w, &dz, &dz);
            add_outer(&mut pxz, *w, &dx, &dz);
        }
        let Some(s_inv) = s.inverse()
        else
        {
            return false;
        };
        let k = pxz.matmul(&s_inv);
        let innov: Vec<f64> = z.iter().zip(&z_pred).map(|(a, b)| a - b).collect();
        let dx = k.matvec(&innov);
        for (xi, d) in self.x.iter_mut().zip(&dx)
        {
            *xi += d;
        }
        // P ← P − K S Kᵀ
        self.p = self.p.sub(&k.matmul(&s).matmul(&k.t()));
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
            let (u1, u2) = (self.u01(), self.u01());
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    fn range(px: f64, py: f64, bx: f64, by: f64) -> f64 {
        ((px - bx).powi(2) + (py - by).powi(2)).sqrt()
    }

    #[test]
    fn tracks_target_from_two_range_beacons_no_jacobians() {
        let dt = 1.0;
        let (b1, b2) = ((0.0, 0.0), (20.0, 0.0));
        let f_mat = Mat::new(
            4,
            4,
            vec![
                1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
        let f = move |x: &[f64]| f_mat.matvec(x);
        let h = |x: &[f64]| vec![range(x[0], x[1], b1.0, b1.1), range(x[0], x[1], b2.0, b2.1)];

        let q = Mat::diag(&[1e-4, 1e-4, 1e-4, 1e-4]);
        let r = Mat::diag(&[0.04, 0.04]);
        let p0 = Mat::diag(&[4.0, 4.0, 4.0, 4.0]);
        let mut ukf = Ukf::new(vec![6.5, 6.5, 0.0, 0.0], p0, q, r);

        let mut rng = Rng::new(0x123);
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
            assert!(ukf.predict(&f));
            assert!(ukf.update(&z, h));
            last_err = ((ukf.state()[0] - px).powi(2) + (ukf.state()[1] - py).powi(2)).sqrt();
        }
        assert!(last_err < 1.0, "final position error {last_err}");
    }

    #[test]
    fn cholesky_factorizes_spd() {
        let a = Mat::new(3, 3, vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0]);
        let l = a.cholesky().expect("spd");
        let recon = l.matmul(&l.t());
        for (x, y) in recon.data.iter().zip(&a.data)
        {
            assert!((x - y).abs() < 1e-10);
        }
    }
}
