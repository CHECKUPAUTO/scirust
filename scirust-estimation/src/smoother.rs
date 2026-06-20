//! Rauch–Tung–Striebel (RTS) fixed-interval smoother.
//!
//! The Kalman filter is causal — its estimate at time `k` uses only data up to
//! `k`. Offline (a recorded run, a maintenance log), the **smoother** adds a
//! backward pass that folds in *future* measurements, giving the minimum-
//! variance estimate at every time. Deterministic `f64`.

use crate::linalg::Mat;

/// Forward Kalman filter + backward RTS pass over a linear-Gaussian model.
///
/// Returns the smoothed state sequence (one vector per measurement). `F`/`Q` are
/// n×n, `H` is m×n, `R` is m×m, and each row of `measurements` is length m.
pub struct RtsSmoother;

impl RtsSmoother {
    /// Smooth `measurements` under the model `(F, Q, H, R)` from initial
    /// `(x0, p0)`.
    pub fn smooth(
        x0: &[f64],
        p0: &Mat,
        f: &Mat,
        q: &Mat,
        h: &Mat,
        r: &Mat,
        measurements: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = x0.len();
        let steps = measurements.len();
        if steps == 0
        {
            return Vec::new();
        }
        let ft = f.t();
        let ht = h.t();

        let mut x_pred = Vec::with_capacity(steps);
        let mut p_pred = Vec::with_capacity(steps);
        let mut x_filt = Vec::with_capacity(steps);
        let mut p_filt = Vec::with_capacity(steps);

        let mut x = x0.to_vec();
        let mut p = p0.clone();
        for z in measurements
        {
            // Predict.
            let xp = f.matvec(&x);
            let pp = f.matmul(&p).matmul(&ft).add(q);
            x_pred.push(xp.clone());
            p_pred.push(pp.clone());

            // Update.
            let hx = h.matvec(&xp);
            let y: Vec<f64> = z.iter().zip(&hx).map(|(zi, hi)| zi - hi).collect();
            let s = h.matmul(&pp).matmul(&ht).add(r);
            let (xf, pf) = match s.inverse()
            {
                Some(s_inv) =>
                {
                    let k = pp.matmul(&ht).matmul(&s_inv);
                    let ky = k.matvec(&y);
                    let xf: Vec<f64> = xp.iter().zip(&ky).map(|(a, b)| a + b).collect();
                    let pf = Mat::identity(n).sub(&k.matmul(h)).matmul(&pp);
                    (xf, pf)
                },
                None => (xp.clone(), pp.clone()),
            };
            x = xf.clone();
            p = pf.clone();
            x_filt.push(xf);
            p_filt.push(pf);
        }

        // Backward RTS pass.
        let mut x_smooth = x_filt.clone();
        let mut p_smooth = p_filt.clone();
        for k in (0..steps - 1).rev()
        {
            let Some(pp_inv) = p_pred[k + 1].inverse()
            else
            {
                continue;
            };
            let c = p_filt[k].matmul(&ft).matmul(&pp_inv);
            let dx: Vec<f64> = x_smooth[k + 1]
                .iter()
                .zip(&x_pred[k + 1])
                .map(|(a, b)| a - b)
                .collect();
            let corr = c.matvec(&dx);
            x_smooth[k] = x_filt[k].iter().zip(&corr).map(|(a, b)| a + b).collect();
            let dp = p_smooth[k + 1].sub(&p_pred[k + 1]);
            p_smooth[k] = p_filt[k].add(&c.matmul(&dp).matmul(&c.t()));
        }
        x_smooth
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

    #[test]
    fn smoother_beats_the_filter() {
        let dt = 1.0;
        let f = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
        let q = Mat::diag(&[1e-4, 1e-4]);
        let h = Mat::new(1, 2, vec![1.0, 0.0]);
        let r = Mat::new(1, 1, vec![0.25]);
        let p0 = Mat::diag(&[1.0, 1.0]);

        // True constant-velocity trajectory + noisy position measurements.
        let mut rng = Rng::new(0x5);
        let (true_v, mut true_x) = (1.0, 0.0);
        let mut truth = Vec::new();
        let mut meas = Vec::new();
        for _ in 0..100
        {
            true_x += true_v * dt;
            truth.push(true_x);
            meas.push(vec![true_x + rng.normal(0.5)]);
        }

        let smoothed = RtsSmoother::smooth(&[0.0, 0.0], &p0, &f, &q, &h, &r, &meas);

        // Filter-only RMSE (re-run the forward filter and read its position).
        let mut x = vec![0.0, 0.0];
        let mut p = p0.clone();
        let (ft, ht) = (f.t(), h.t());
        let mut filt_se = 0.0;
        let mut smooth_se = 0.0;
        for (k, z) in meas.iter().enumerate()
        {
            let xp = f.matvec(&x);
            let pp = f.matmul(&p).matmul(&ft).add(&q);
            let s = h.matmul(&pp).matmul(&ht).add(&r);
            let s_inv = s.inverse().unwrap();
            let kk = pp.matmul(&ht).matmul(&s_inv);
            let y = vec![z[0] - h.matvec(&xp)[0]];
            let ky = kk.matvec(&y);
            x = xp.iter().zip(&ky).map(|(a, b)| a + b).collect();
            p = Mat::identity(2).sub(&kk.matmul(&h)).matmul(&pp);
            filt_se += (x[0] - truth[k]).powi(2);
            smooth_se += (smoothed[k][0] - truth[k]).powi(2);
        }
        let filt_rmse = (filt_se / 100.0).sqrt();
        let smooth_rmse = (smooth_se / 100.0).sqrt();
        assert!(
            smooth_rmse < filt_rmse,
            "smoother RMSE {smooth_rmse} should beat filter {filt_rmse}"
        );
    }
}
