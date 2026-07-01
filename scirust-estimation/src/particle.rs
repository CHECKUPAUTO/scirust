//! Deterministic bootstrap particle filter (SIR).
//!
//! For strongly nonlinear or non-Gaussian problems where the (E/U)KF's single
//! Gaussian fails, a particle filter represents the posterior by a weighted
//! cloud of samples, propagated through the true dynamics and reweighted by the
//! measurement likelihood, with systematic resampling. Seeded RNG + fixed-order
//! arithmetic ⇒ bit-reproducible.

use serde::{Deserialize, Serialize};

/// Particle filter over an `n`-dim state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleFilter {
    particles: Vec<Vec<f64>>,
    weights: Vec<f64>,
    rng: u64,
}

impl ParticleFilter {
    /// `n` particles, each initialised by `init` (which receives a uniform draw
    /// per call), seeded deterministically.
    pub fn new(
        n: usize,
        mut init: impl FnMut(&mut dyn FnMut() -> f64) -> Vec<f64>,
        seed: u64,
    ) -> Self {
        let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
        let particles: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let mut draw = || next_u01(&mut rng);
                init(&mut draw)
            })
            .collect();
        Self {
            particles,
            weights: vec![1.0 / n as f64; n],
            rng,
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether the filter holds no particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Propagate each particle through `f`, then add diagonal Gaussian process
    /// noise with per-dimension standard deviations `proc_std`.
    ///
    /// If `proc_std` is shorter than the propagated state, the last entry is
    /// broadcast to every trailing dimension (so a single-element slice acts as
    /// an isotropic std); an empty `proc_std` adds no noise. This avoids
    /// silently leaving trailing dimensions noise-free on a length mismatch.
    pub fn predict(&mut self, f: impl Fn(&[f64]) -> Vec<f64>, proc_std: &[f64]) {
        for p in self.particles.iter_mut()
        {
            let mut np = f(p);
            for (i, v) in np.iter_mut().enumerate()
            {
                // Draw one deterministic normal per dimension regardless of the
                // std used, so the RNG stream stays independent of `proc_std`'s
                // length. Missing entries reuse the last provided std (or zero).
                let sd = match proc_std.len()
                {
                    0 => 0.0,
                    len => proc_std[i.min(len - 1)],
                };
                *v += sd * next_normal(&mut self.rng);
            }
            *p = np;
        }
    }

    /// Reweight by `log_likelihood(particle)`, normalize, and resample if the
    /// effective sample size drops below `len/2`.
    pub fn update(&mut self, log_likelihood: impl Fn(&[f64]) -> f64) {
        let mut max_lw = f64::NEG_INFINITY;
        let lws: Vec<f64> = self
            .particles
            .iter()
            .map(|p| {
                let lw = log_likelihood(p);
                if lw > max_lw
                {
                    max_lw = lw;
                }
                lw
            })
            .collect();
        let mut sum = 0.0;
        for (w, lw) in self.weights.iter_mut().zip(&lws)
        {
            *w *= (lw - max_lw).exp(); // stabilized
            sum += *w;
        }
        if sum <= 0.0 || !sum.is_finite()
        {
            // Degenerate: reset to uniform.
            let n = self.weights.len();
            self.weights.iter_mut().for_each(|w| *w = 1.0 / n as f64);
            return;
        }
        for w in self.weights.iter_mut()
        {
            *w /= sum;
        }
        let ess = 1.0 / self.weights.iter().map(|w| w * w).sum::<f64>();
        if ess < self.particles.len() as f64 / 2.0
        {
            self.resample();
        }
    }

    /// Systematic resampling, then reset weights to uniform.
    fn resample(&mut self) {
        let n = self.particles.len();
        let start = next_u01(&mut self.rng) / n as f64;
        let mut idx = 0;
        let mut new_particles = Vec::with_capacity(n);
        let mut c = self.weights[0];
        for j in 0..n
        {
            let u = start + j as f64 / n as f64;
            while u > c && idx + 1 < n
            {
                idx += 1;
                c += self.weights[idx];
            }
            new_particles.push(self.particles[idx].clone());
        }
        self.particles = new_particles;
        self.weights = vec![1.0 / n as f64; n];
    }

    /// Weighted-mean state estimate.
    pub fn estimate(&self) -> Vec<f64> {
        let dim = self.particles.first().map(|p| p.len()).unwrap_or(0);
        let mut out = vec![0.0; dim];
        for (p, &w) in self.particles.iter().zip(&self.weights)
        {
            for (o, &v) in out.iter_mut().zip(p)
            {
                *o += w * v;
            }
        }
        out
    }
}

fn next_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
}

fn next_normal(state: &mut u64) -> f64 {
    let u1 = next_u01(state).max(1e-12);
    let u2 = next_u01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_through_a_nonlinear_measurement() {
        // State: slow random walk. Measurement: z = x³/10 (monotone, nonlinear).
        let meas_sd = 0.3;
        let mut pf = ParticleFilter::new(2000, |u| vec![10.0 * (u() - 0.5)], 0xABCD);

        let mut rng = 0xF00Du64;
        let mut x = 1.0;
        let mut sq_err = 0.0;
        let steps = 60;
        for k in 0..steps
        {
            x += 0.3 * next_normal(&mut rng);
            let z = x.powi(3) / 10.0 + meas_sd * next_normal(&mut rng);
            pf.predict(|p| p.to_vec(), &[0.3]);
            pf.update(|p| {
                let zh = p[0].powi(3) / 10.0;
                -(z - zh).powi(2) / (2.0 * meas_sd * meas_sd)
            });
            if k >= 20
            {
                sq_err += (pf.estimate()[0] - x).powi(2);
            }
        }
        let rmse = (sq_err / 40.0).sqrt();
        assert!(rmse < 1.0, "PF RMSE {rmse} too high");
    }

    #[test]
    fn particle_len_and_is_empty() {
        let pf = ParticleFilter::new(500, |u| vec![u()], 7);
        assert_eq!(pf.len(), 500);
        assert!(!pf.is_empty());
    }

    #[test]
    fn predict_noises_trailing_dims_on_short_proc_std() {
        // 3-D state, but only a 1-element `proc_std`. Before the fix, the `zip`
        // truncated after dim 0, leaving dims 1 and 2 noise-free; the broadcast
        // now applies the (single) std to every dimension.
        let mut pf = ParticleFilter::new(400, |_| vec![0.0, 0.0, 0.0], 0x1234);
        pf.predict(|p| p.to_vec(), &[1.0]);

        // Spread across particles for each dimension: all three must be excited.
        let dims = 3;
        let n = pf.len() as f64;
        for d in 0..dims
        {
            let mean: f64 = pf.particles.iter().map(|p| p[d]).sum::<f64>() / n;
            let var: f64 = pf
                .particles
                .iter()
                .map(|p| (p[d] - mean).powi(2))
                .sum::<f64>()
                / n;
            assert!(
                var > 0.1,
                "dim {d} received no process noise (var = {var}); trailing dims silently skipped"
            );
        }
    }

    #[test]
    fn predict_empty_proc_std_adds_no_noise() {
        // Empty `proc_std` must be a no-op on the noise (not a panic).
        let mut pf = ParticleFilter::new(16, |_| vec![2.0, -1.0], 0x99);
        pf.predict(|p| p.to_vec(), &[]);
        for p in &pf.particles
        {
            assert_eq!(p, &vec![2.0, -1.0]);
        }
    }

    #[test]
    fn run_is_deterministic() {
        let run = || {
            let mut pf = ParticleFilter::new(500, |u| vec![u()], 7);
            pf.predict(|p| p.to_vec(), &[0.1]);
            pf.update(|p| -(p[0] - 0.5).powi(2));
            pf.estimate()
        };
        assert_eq!(run(), run());
    }
}
