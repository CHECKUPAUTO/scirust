//! UD (square-root) Kalman filter — Bierman–Thornton factored covariance.
//!
//! A conventional Kalman filter stores the covariance `P` directly and can,
//! under finite precision, let it drift non-symmetric or indefinite — a silent
//! loss of the very guarantee the filter exists to provide. The UD filter never
//! forms `P`: it carries the factors `P = U D Uᵀ` (`U` unit upper-triangular,
//! `D` diagonal ≥ 0) and updates them in place, so the covariance it represents
//! is symmetric positive-semidefinite *by construction*. This is the form flown
//! in inertial navigation and spacecraft estimation for exactly that reason.
//!
//! - [`UdFilter::update`] — Bierman's scalar observational update.
//! - [`UdFilter::predict`] — Thornton's modified-weighted-Gram–Schmidt time
//!   update for `P' = Φ P Φᵀ + G Q Gᵀ`.
//!
//! Operations accumulate in a fixed order, so a run is bit-identical across
//! machines, matching the determinism the rest of SciRust upholds.

use crate::linalg::Mat;

/// A Kalman filter that carries its covariance in factored `U D Uᵀ` form.
#[derive(Debug, Clone)]
pub struct UdFilter {
    /// State estimate.
    pub x: Vec<f64>,
    /// Unit upper-triangular factor `U` (`n × n`).
    u: Mat,
    /// Diagonal factor `D` (length `n`).
    d: Vec<f64>,
}

/// Factor a symmetric positive-definite `P` as `U D Uᵀ` with `U` unit
/// upper-triangular and `D` the diagonal (modified Cholesky).
#[allow(clippy::needless_range_loop)]
fn ud_factor(p: &Mat) -> (Mat, Vec<f64>) {
    let n = p.rows;
    let mut u = Mat::identity(n);
    let mut d = vec![0.0; n];
    for j in (0..n).rev()
    {
        let mut dj = p.get(j, j);
        for k in (j + 1)..n
        {
            dj -= d[k] * u.get(j, k) * u.get(j, k);
        }
        d[j] = dj;
        u.set(j, j, 1.0);
        for i in (0..j).rev()
        {
            let mut s = p.get(i, j);
            for k in (j + 1)..n
            {
                s -= d[k] * u.get(i, k) * u.get(j, k);
            }
            u.set(i, j, if dj != 0.0 { s / dj } else { 0.0 });
        }
    }
    (u, d)
}

impl UdFilter {
    /// Initialise from state `x0` and covariance `p0` (symmetric PD).
    pub fn new(x0: Vec<f64>, p0: &Mat) -> Self {
        let (u, d) = ud_factor(p0);
        Self { x: x0, u, d }
    }

    /// Reconstruct the dense covariance `P = U D Uᵀ` (for inspection/testing).
    pub fn covariance(&self) -> Mat {
        let n = self.x.len();
        // (U D) then · Uᵀ.
        let mut ud = Mat::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                ud.data[i * n + j] = self.u.get(i, j) * self.d[j];
            }
        }
        ud.matmul(&self.u.t())
    }

    /// Bierman scalar observational update: measurement `z = h·x + noise`,
    /// `h` the `1 × n` row, `r` the measurement-noise variance (`> 0`).
    /// Returns the innovation covariance `s = h P hᵀ + r`.
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, h: &[f64], r: f64, z: f64) -> f64 {
        let n = self.x.len();
        // f = Uᵀ h    (f_j = h_j + Σ_{i<j} U_ij h_i)
        let mut f = vec![0.0; n];
        for j in 0..n
        {
            let mut acc = h[j];
            for i in 0..j
            {
                acc += self.u.get(i, j) * h[i];
            }
            f[j] = acc;
        }
        // g = D f.
        let g: Vec<f64> = (0..n).map(|j| self.d[j] * f[j]).collect();

        // Bierman recursion: accumulate the (unnormalised) gain k and update U, D.
        let mut k = vec![0.0; n];
        let mut alpha = r + f[0] * g[0];
        self.d[0] = if alpha != 0.0
        {
            r * self.d[0] / alpha
        }
        else
        {
            0.0
        };
        k[0] = g[0];
        for j in 1..n
        {
            let beta = alpha;
            alpha = beta + f[j] * g[j];
            self.d[j] = if alpha != 0.0
            {
                beta * self.d[j] / alpha
            }
            else
            {
                0.0
            };
            let lambda = if beta != 0.0 { f[j] / beta } else { 0.0 };
            for i in 0..j
            {
                let u_old = self.u.get(i, j);
                self.u.set(i, j, u_old - lambda * k[i]);
                k[i] += g[j] * u_old;
            }
            k[j] = g[j];
        }
        // alpha == h P hᵀ + r == s; Kalman gain is k / s.
        let s = alpha;
        let innov = z - h.iter().zip(&self.x).map(|(a, b)| a * b).sum::<f64>();
        if s != 0.0
        {
            for i in 0..n
            {
                self.x[i] += k[i] * innov / s;
            }
        }
        s
    }

    /// Thornton time update: propagate through `phi` (`n × n`) with process-noise
    /// gain `g` (`n × p`) and diagonal process-noise variances `qd` (length `p`),
    /// giving the factors of `P' = Φ P Φᵀ + G Q Gᵀ`.
    #[allow(clippy::needless_range_loop)]
    pub fn predict(&mut self, phi: &Mat, g: &Mat, qd: &[f64]) {
        let n = self.x.len();
        let p = qd.len();
        let np = n + p;
        // W = [Φ·U | G]  (n × (n+p)); Dw = [D ; Qd].
        let mut w = vec![0.0; n * np];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for kk in 0..n
                {
                    acc += phi.get(i, kk) * self.u.get(kk, j);
                }
                w[i * np + j] = acc;
            }
            for j in 0..p
            {
                w[i * np + n + j] = g.get(i, j);
            }
        }
        let mut dw = vec![0.0; np];
        dw[..n].copy_from_slice(&self.d);
        dw[n..(n + p)].copy_from_slice(&qd[..p]);
        // Modified weighted Gram–Schmidt, row n-1 down to 0.
        let mut u_new = Mat::identity(n);
        let mut d_new = vec![0.0; n];
        for i in (0..n).rev()
        {
            let mut sigma = 0.0;
            for kk in 0..np
            {
                sigma += w[i * np + kk] * w[i * np + kk] * dw[kk];
            }
            d_new[i] = sigma;
            for j in 0..i
            {
                let mut dot = 0.0;
                for kk in 0..np
                {
                    dot += w[j * np + kk] * dw[kk] * w[i * np + kk];
                }
                let uji = if sigma != 0.0 { dot / sigma } else { 0.0 };
                u_new.set(j, i, uji);
                for kk in 0..np
                {
                    w[j * np + kk] -= uji * w[i * np + kk];
                }
            }
        }
        self.u = u_new;
        self.d = d_new;
        // Propagate the mean.
        self.x = phi.matvec(&self.x);
    }

    /// Diagonal of the represented covariance (per-state variances), cheaply.
    pub fn variances(&self) -> Vec<f64> {
        let n = self.x.len();
        (0..n)
            .map(|i| (i..n).map(|j| self.u.get(i, j).powi(2) * self.d[j]).sum())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::KalmanFilter;

    fn cv2(dt: f64) -> Mat {
        Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0])
    }

    #[test]
    fn ud_factorization_reconstructs_the_covariance() {
        let p = Mat::new(3, 3, vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let f = UdFilter::new(vec![0.0; 3], &p);
        let r = f.covariance();
        for (a, b) in r.data.iter().zip(&p.data)
        {
            assert!((a - b).abs() < 1e-10, "{a} != {b}");
        }
    }

    #[test]
    fn matches_the_conventional_kalman_filter() {
        // Same model, same data: the UD filter must agree with a textbook KF to
        // near machine precision — while keeping its covariance factored.
        let dt = 0.5;
        let phi = cv2(dt);
        let g = Mat::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let qd = [1e-3, 1e-3];
        let q = Mat::diag(&qd);
        let h = [1.0, 0.0];
        let r = 0.2;
        let p0 = Mat::diag(&[1.0, 1.0]);

        let mut ud = UdFilter::new(vec![0.0, 0.0], &p0);
        let mut kf = KalmanFilter::new(
            vec![0.0, 0.0],
            p0.clone(),
            phi.clone(),
            q.clone(),
            Mat::new(1, 2, h.to_vec()),
            Mat::new(1, 1, vec![r]),
        );

        // Deterministic noisy ramp.
        let mut seed = 0x51A7u64;
        let mut nrand = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64) - 0.5
        };
        let mut truth = 0.0;
        for _ in 0..80
        {
            truth += dt;
            let z = truth + 0.3 * nrand();
            ud.predict(&phi, &g, &qd);
            ud.update(&h, r, z);
            kf.predict();
            kf.update(&[z]);
            for (a, b) in ud.x.iter().zip(kf.state())
            {
                assert!((a - b).abs() < 1e-9, "state {a} vs {b}");
            }
            // Covariance diagonals agree too.
            let pud = ud.covariance();
            let pkf = kf.covariance();
            for i in 0..2
            {
                assert!(
                    (pud.get(i, i) - pkf.get(i, i)).abs() < 1e-9,
                    "var[{i}] {} vs {}",
                    pud.get(i, i),
                    pkf.get(i, i)
                );
            }
        }
    }

    #[test]
    fn covariance_stays_positive_semidefinite() {
        // A stiff, near-singular measurement that can push a naive filter
        // indefinite: the UD diagonal must remain non-negative throughout.
        let phi = cv2(1.0);
        let g = Mat::identity(2);
        let qd = [1e-9, 1e-9];
        let h = [1.0, 0.0];
        let p0 = Mat::diag(&[1e6, 1e6]);
        let mut ud = UdFilter::new(vec![0.0, 0.0], &p0);
        for k in 0..50
        {
            ud.predict(&phi, &g, &qd);
            ud.update(&h, 1e-8, k as f64); // very confident measurements
            assert!(
                ud.variances().iter().all(|&v| v >= -1e-12),
                "negative variance at step {k}: {:?}",
                ud.variances()
            );
        }
    }
}
