//! Interacting Multiple Models (IMM) filter.
//!
//! Many systems switch regimes — a target flying straight then turning, a process
//! in steady state then upset. The IMM runs a bank of Kalman filters (one per
//! model), mixes their estimates each step, and updates a probability for each
//! mode from how well it explains the measurement. It tracks across regime
//! changes that a single model lags, and exposes *which* regime is active.
//!
//! Scalar measurement (`m = 1`) — the usual tracking case.

use crate::linalg::Mat;

/// One mode's linear-Gaussian model.
#[derive(Debug, Clone)]
pub struct ImmModel {
    pub a: Mat,
    pub q: Mat,
    pub h: Mat,
    pub r: Mat,
}

/// Interacting Multiple Models filter.
pub struct Imm {
    models: Vec<ImmModel>,
    x: Vec<Vec<f64>>,
    p: Vec<Mat>,
    mu: Vec<f64>,
    pi: Vec<Vec<f64>>,
}

impl Imm {
    /// Build from per-mode `models`, a shared initial `x0`/`p0`, initial mode
    /// probabilities `mu0`, and the Markov mode-transition matrix `pi`
    /// (`pi[i][j] = P(mode i → mode j)`).
    pub fn new(
        models: Vec<ImmModel>,
        x0: Vec<f64>,
        p0: Mat,
        mu0: Vec<f64>,
        pi: Vec<Vec<f64>>,
    ) -> Self {
        let k = models.len();
        Self {
            x: vec![x0; k],
            p: vec![p0; k],
            mu: mu0,
            pi,
            models,
        }
    }

    /// Overall (mode-probability-weighted) state estimate.
    pub fn estimate(&self) -> Vec<f64> {
        let n = self.x[0].len();
        let mut out = vec![0.0; n];
        for (mu, xi) in self.mu.iter().zip(&self.x)
        {
            for (o, &v) in out.iter_mut().zip(xi)
            {
                *o += mu * v;
            }
        }
        out
    }

    /// Current mode probabilities.
    pub fn mode_probabilities(&self) -> &[f64] {
        &self.mu
    }

    /// One IMM cycle for scalar measurement `z`.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, z: f64) {
        let k = self.models.len();
        let n = self.x[0].len();

        // 1. Predicted mode probabilities and mixing weights.
        let cbar: Vec<f64> = (0..k)
            .map(|j| (0..k).map(|i| self.pi[i][j] * self.mu[i]).sum())
            .collect();

        // 2. Mixed initial conditions per mode j.
        let mut x_mix = vec![vec![0.0; n]; k];
        let mut p_mix = vec![Mat::zeros(n, n); k];
        for j in 0..k
        {
            if cbar[j] <= 0.0
            {
                x_mix[j] = self.x[j].clone();
                p_mix[j] = self.p[j].clone();
                continue;
            }
            for i in 0..k
            {
                let w = self.pi[i][j] * self.mu[i] / cbar[j];
                for d in 0..n
                {
                    x_mix[j][d] += w * self.x[i][d];
                }
            }
            for i in 0..k
            {
                let w = self.pi[i][j] * self.mu[i] / cbar[j];
                let dx: Vec<f64> = self.x[i]
                    .iter()
                    .zip(&x_mix[j])
                    .map(|(a, b)| a - b)
                    .collect();
                // P_mix += w (P_i + dx dxᵀ)
                for r0 in 0..n
                {
                    for c0 in 0..n
                    {
                        p_mix[j].data[r0 * n + c0] += w * (self.p[i].get(r0, c0) + dx[r0] * dx[c0]);
                    }
                }
            }
        }

        // 3. Mode-matched Kalman predict + update; collect likelihoods.
        let mut likelihood = vec![0.0; k];
        for j in 0..k
        {
            let m = &self.models[j];
            // Predict.
            let xp = m.a.matvec(&x_mix[j]);
            let pp = m.a.matmul(&p_mix[j]).matmul(&m.a.t()).add(&m.q);
            // Update (scalar measurement).
            let hx = m.h.matvec(&xp)[0];
            let y = z - hx;
            let ht = m.h.t();
            let s = m.h.matmul(&pp).matmul(&ht).add(&m.r).get(0, 0);
            if s <= 0.0
            {
                self.x[j] = xp;
                self.p[j] = pp;
                likelihood[j] = 1e-300;
                continue;
            }
            // Kalman gain K = P Hᵀ / s.
            let pht = pp.matmul(&ht); // n×1
            let kgain: Vec<f64> = (0..n).map(|i| pht.get(i, 0) / s).collect();
            self.x[j] = xp.iter().zip(&kgain).map(|(a, b)| a + b * y).collect();
            // P = (I - K H) P.
            let mut kh = Mat::zeros(n, n);
            for r0 in 0..n
            {
                for c0 in 0..n
                {
                    kh.data[r0 * n + c0] = kgain[r0] * m.h.get(0, c0);
                }
            }
            self.p[j] = Mat::identity(n).sub(&kh).matmul(&pp);
            // Gaussian likelihood of the innovation.
            likelihood[j] = (-0.5 * y * y / s).exp() / (2.0 * core::f64::consts::PI * s).sqrt();
        }

        // 4. Mode-probability update.
        let mut norm = 0.0;
        let mut new_mu = vec![0.0; k];
        for j in 0..k
        {
            new_mu[j] = likelihood[j] * cbar[j];
            norm += new_mu[j];
        }
        if norm > 0.0
        {
            for m in new_mu.iter_mut()
            {
                *m /= norm;
            }
            self.mu = new_mu;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cv_model(dt: f64, q: f64, r: f64) -> ImmModel {
        ImmModel {
            a: Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]),
            q: Mat::diag(&[q, q]),
            h: Mat::new(1, 2, vec![1.0, 0.0]),
            r: Mat::new(1, 1, vec![r]),
        }
    }

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn normal(&mut self, sd: f64) -> f64 {
            let u = |st: &mut u64| {
                *st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = *st;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
            };
            let (u1, u2) = (u(&mut self.s).max(1e-9), u(&mut self.s));
            sd * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn imm_single_model_step_matches_scalar_kalman() {
        // Single mode reduces IMM to a scalar KF.
        // predict: xp=0, pp = a*p*a + q = 1.
        // update z=4: y = 4-0 = 4, s = h*pp*h + r = 1+1 = 2, K = pp*h/s = 1/2,
        //   x = 0 + 0.5*4 = 2.0. estimate = mu0*x = 1*2.0 = 2.0.
        // With one mode the normalized mode prob is always 1.0.
        let m = ImmModel {
            a: Mat::new(1, 1, vec![1.0]),
            q: Mat::new(1, 1, vec![0.0]),
            h: Mat::new(1, 1, vec![1.0]),
            r: Mat::new(1, 1, vec![1.0]),
        };
        let mut imm = Imm::new(
            vec![m],
            vec![0.0],
            Mat::new(1, 1, vec![1.0]),
            vec![1.0],
            vec![vec![1.0]],
        );
        imm.step(4.0);
        assert!((imm.estimate()[0] - 2.0).abs() < 1e-12);
        assert!((imm.mode_probabilities()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mode_probability_shifts_to_the_maneuver_model() {
        let dt = 1.0;
        // Mode 0: quiet CV (low Q). Mode 1: agile (high Q) for maneuvers.
        let models = vec![cv_model(dt, 1e-4, 0.25), cv_model(dt, 1.0, 0.25)];
        let pi = vec![vec![0.95, 0.05], vec![0.1, 0.9]];
        let mut imm = Imm::new(
            models,
            vec![0.0, 1.0],
            Mat::diag(&[1.0, 1.0]),
            vec![0.5, 0.5],
            pi,
        );

        let mut rng = Rng::new(0x1117);
        let (mut pos, mut vel) = (0.0, 1.0);
        let mut quiet_mode1 = 0.0f64;
        let mut maneuver_mode1 = 0.0f64;
        for k in 0..120
        {
            // Sudden velocity reversal at k = 60 (a maneuver).
            if k == 60
            {
                vel = -2.0;
            }
            pos += vel * dt;
            imm.step(pos + rng.normal(0.5));
            if (40..60).contains(&k)
            {
                quiet_mode1 = imm.mode_probabilities()[1];
            }
            if (62..80).contains(&k)
            {
                maneuver_mode1 = maneuver_mode1.max(imm.mode_probabilities()[1]);
            }
        }
        // The agile model's probability rises after the maneuver.
        assert!(
            maneuver_mode1 > quiet_mode1 + 0.2,
            "maneuver p1 {maneuver_mode1} vs quiet p1 {quiet_mode1}"
        );
    }
}
