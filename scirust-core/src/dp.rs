//! Differential Privacy — DP-SGD training.
//!
//! > ⚠️ **Experimental / no consumers**: this module is not used by any
//! > crate in the workspace. The API may change or be removed; open an
//! > issue if you depend on it.
//!
//! Implements the differentially private stochastic gradient descent
//! algorithm from Abadi et al. (2016):
//! 1. Clip **per-sample** gradients to a maximum L2 norm.
//! 2. Sum, then add Gaussian noise calibrated to the clip norm.
//! 3. Track the privacy budget with a Rényi-DP accountant.
//!
//! # ⚠️ Getting the guarantee right
//!
//! * **The noise RNG must be cryptographically secure.** All noise APIs here
//!   are bounded by `rand::CryptoRng`, so a statistical PRNG (e.g. the crate's
//!   `PcgEngine`) can no longer be passed in — an adversary who can predict the
//!   stream can subtract the noise and the DP guarantee collapses. Use
//!   [`SecureRng`] seeded from the OS ([`new_secure_rng`]); a fixed seed is for
//!   tests only.
//! * **The accountant is intentionally conservative.** It charges every step
//!   the *un-subsampled* Gaussian RDP `α/(2σ²)`. Subsampling only *amplifies*
//!   privacy, so ignoring it **over-estimates** ε — a sound upper bound that
//!   never under-reports. (The previous moments accountant used a log-moment
//!   linear in α, which made the reported ε an artifact of the α-grid and
//!   *under*-reported the true ε.) A tight subsampled-Gaussian bound is future
//!   work; until then this trades tightness for soundness.
//! * f32 Box–Muller noise has a finite output set (Mironov 2012, "On
//!   significance of the least significant bits for DP"). For a rigorous
//!   deployment prefer a discrete/rounded Gaussian; this is adequate for
//!   research use.
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::dp::{DpSgdConfig, new_secure_rng, dp_sgd_gradient};
//!
//! let config = DpSgdConfig::default();
//! let mut rng = new_secure_rng();               // OS-seeded CSPRNG
//! let batch_grad = dp_sgd_gradient(&per_sample_grads, &config, &mut rng);
//! ```

use rand::rngs::StdRng;
use rand::{CryptoRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// A cryptographically secure RNG suitable for DP noise (ChaCha-based).
pub type SecureRng = StdRng;

/// Construct a [`SecureRng`] seeded from OS entropy. Use this in production.
pub fn new_secure_rng() -> SecureRng {
    StdRng::from_entropy()
}

/// Configuration for DP-SGD training.
#[derive(Debug, Clone)]
pub struct DpSgdConfig {
    /// Maximum L2 norm for per-sample gradient clipping.
    pub l2_norm_clip: f32,
    /// Standard deviation multiplier for Gaussian noise:
    /// σ = noise_multiplier * l2_norm_clip.
    pub noise_multiplier: f32,
    /// Target δ (probability of privacy failure).
    pub delta: f64,
    /// Target ε (privacy budget). Used for accounting, not enforcement.
    pub epsilon: Option<f64>,
}

impl Default for DpSgdConfig {
    fn default() -> Self {
        Self {
            l2_norm_clip: 1.0,
            noise_multiplier: 1.1,
            delta: 1e-5,
            epsilon: None,
        }
    }
}

/// Rényi-DP accountant for tracking the privacy budget.
///
/// See the module note: accounting is **conservative** (charges the
/// un-subsampled Gaussian RDP), producing a sound upper bound on ε.
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Total ε consumed so far (upper bound).
    pub epsilon: f64,
    /// δ parameter.
    pub delta: f64,
    /// Total number of training steps.
    pub steps: usize,
    /// Sampling probability q = batch_size / dataset_size. Retained for API
    /// compatibility and a future tight (amplified) bound; the conservative
    /// accountant does not use it.
    pub q: f64,
    /// Noise multiplier σ.
    pub sigma: f64,
    /// Rényi divergence orders α used for the RDP→DP conversion.
    pub orders: Vec<f64>,
    /// Accumulated RDP at each order.
    pub log_moments: Vec<f64>,
}

impl MomentsAccountant {
    /// Create a new accountant. `sigma` must be > 0.
    pub fn new(delta: f64, q: f64, sigma: f64) -> Self {
        // Dense just above 1 (tight when many steps compose), then integers to 256.
        let mut orders: Vec<f64> = (1..20).map(|i| 1.0 + i as f64 * 0.05).collect();
        orders.extend((2..=256).map(|a| a as f64));
        let log_moments = vec![0.0; orders.len()];

        Self {
            epsilon: 0.0,
            delta,
            steps: 0,
            q,
            sigma,
            orders,
            log_moments,
        }
    }

    /// Record one training step.
    pub fn step(&mut self) {
        self.steps += 1;
        if self.sigma <= 0.0
        {
            self.epsilon = f64::INFINITY;
            return;
        }
        // Conservative: charge the un-subsampled Gaussian RDP at each order.
        for (i, &alpha) in self.orders.iter().enumerate()
        {
            self.log_moments[i] += gaussian_rdp(alpha, self.sigma);
        }
        self.epsilon = self.compute_epsilon();
    }

    /// Convert accumulated RDP to (ε, δ)-DP, taking the tightest order.
    fn compute_epsilon(&self) -> f64 {
        let mut best_eps = f64::INFINITY;
        for (i, &alpha) in self.orders.iter().enumerate()
        {
            let eps = rdp_to_dp(self.log_moments[i], alpha, self.delta);
            if eps < best_eps
            {
                best_eps = eps;
            }
        }
        best_eps.max(0.0)
    }
}

// ----- Rényi DP accountant (Mironov, CSF 2017) (#78) ---------------------------

/// **Rényi differential privacy** of the Gaussian mechanism (sensitivity 1, noise
/// multiplier `sigma`) at order `alpha > 1`: `RDP(α) = α / (2σ²)` (Mironov 2017,
/// Cor. 3). Composition is **additive** in RDP — far easier and tighter than
/// composing `(ε, δ)` pairs directly.
pub fn gaussian_rdp(alpha: f64, sigma: f64) -> f64 {
    assert!(
        alpha > 1.0 && sigma > 0.0,
        "gaussian_rdp: need alpha>1, sigma>0"
    );
    alpha / (2.0 * sigma * sigma)
}

/// Convert an RDP guarantee `rdp_eps` at order `alpha` into `(ε, δ)`-DP (Mironov
/// 2017, Prop. 3): `ε = rdp_eps + ln(1/δ)/(α − 1)`. The `α − 1` (not `α`) is what
/// makes this the *tight* conversion.
pub fn rdp_to_dp(rdp_eps: f64, alpha: f64, delta: f64) -> f64 {
    assert!(
        alpha > 1.0 && delta > 0.0 && delta < 1.0,
        "rdp_to_dp: need alpha>1, delta in (0,1)"
    );
    rdp_eps + (1.0 / delta).ln() / (alpha - 1.0)
}

/// **RDP accountant** for `steps` compositions of the Gaussian mechanism at noise
/// multiplier `sigma`: the total RDP is `steps · α/(2σ²)`; convert at every order
/// in a grid and keep the **tightest** `ε` for the target `delta`. Returns
/// `(ε, best_α)`. Much tighter than naive linear `(ε, δ)` composition (which pays
/// a `√steps`-type penalty), as the tests show.
pub fn rdp_gaussian_epsilon(steps: usize, sigma: f64, delta: f64) -> (f64, f64) {
    // A grid of orders: fine just above 1, then integers out to 256.
    let mut alphas: Vec<f64> = (1..20).map(|i| 1.0 + i as f64 * 0.05).collect();
    alphas.extend((2..=256).map(|a| a as f64));
    let mut best = (f64::INFINITY, 0.0);
    for &alpha in &alphas
    {
        let total_rdp = steps as f64 * gaussian_rdp(alpha, sigma);
        let eps = rdp_to_dp(total_rdp, alpha, delta);
        if eps < best.0
        {
            best = (eps, alpha);
        }
    }
    best
}

/// Clip gradient vector to maximum L2 norm.
///
/// If ||g||_2 > clip, scale g by clip / ||g||_2.
pub fn clip_gradients(grads: &mut [f32], l2_norm_clip: f32) {
    let norm_sq: f32 = grads.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();

    if norm > l2_norm_clip && norm > 1e-12
    {
        let scale = l2_norm_clip / norm;
        for g in grads.iter_mut()
        {
            *g *= scale;
        }
    }
}

/// Add Gaussian noise to gradients for differential privacy.
///
/// The RNG **must** be cryptographically secure (enforced by the `CryptoRng`
/// bound). σ = noise_multiplier * l2_norm_clip (should match [`clip_gradients`]).
pub fn add_noise<R: Rng + CryptoRng>(grads: &mut [f32], noise_stddev: f32, rng: &mut R) {
    if noise_stddev <= 0.0
    {
        return;
    }
    let normal = Normal::new(0.0f32, noise_stddev).expect("noise_stddev is finite and > 0");
    for g in grads.iter_mut()
    {
        *g += normal.sample(rng);
    }
}

/// Clip a **single sample's** gradient and add noise. For a real DP-SGD step
/// over a minibatch use [`dp_sgd_gradient`], which clips each sample *before*
/// summing so the sensitivity equals `l2_norm_clip`.
pub fn dp_protect<R: Rng + CryptoRng>(
    per_sample_grads: &mut [f32],
    config: &DpSgdConfig,
    rng: &mut R,
) {
    clip_gradients(per_sample_grads, config.l2_norm_clip);
    let noise_std = config.noise_multiplier * config.l2_norm_clip;
    add_noise(per_sample_grads, noise_std, rng);
}

/// One correct DP-SGD gradient step over a minibatch of per-sample gradients:
/// clip each sample to `l2_norm_clip` (bounding the sensitivity), sum, add a
/// single draw of Gaussian noise with σ = `noise_multiplier · l2_norm_clip`,
/// then average by the batch size. Returns the privatized mean gradient.
pub fn dp_sgd_gradient<R: Rng + CryptoRng>(
    per_sample_grads: &[Vec<f32>],
    config: &DpSgdConfig,
    rng: &mut R,
) -> Vec<f32> {
    assert!(
        !per_sample_grads.is_empty(),
        "dp_sgd_gradient: need at least one sample"
    );
    let dim = per_sample_grads[0].len();
    let mut summed = vec![0.0f32; dim];
    for g in per_sample_grads
    {
        assert_eq!(g.len(), dim, "dp_sgd_gradient: ragged per-sample gradients");
        let mut clipped = g.clone();
        clip_gradients(&mut clipped, config.l2_norm_clip);
        for (s, v) in summed.iter_mut().zip(clipped.iter())
        {
            *s += *v;
        }
    }
    // Sensitivity of the sum is l2_norm_clip (one sample can change it by ≤ clip).
    let noise_std = config.noise_multiplier * config.l2_norm_clip;
    add_noise(&mut summed, noise_std, rng);
    let inv = 1.0 / per_sample_grads.len() as f32;
    for s in summed.iter_mut()
    {
        *s *= inv;
    }
    summed
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixed-seed CSPRNG for reproducible tests; production uses new_secure_rng().
    fn test_rng() -> SecureRng {
        StdRng::from_seed([42u8; 32])
    }

    #[test]
    fn test_clip_below_threshold() {
        let mut grads = vec![0.1, 0.2, 0.3];
        let original = grads.clone();
        clip_gradients(&mut grads, 10.0);
        // Norm = sqrt(0.01+0.04+0.09) ≈ 0.374 < 10 → no change
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_above_threshold() {
        let mut grads = vec![3.0, 4.0]; // Norm = 5.0
        clip_gradients(&mut grads, 1.0);
        // Scale = 1/5 = 0.2
        assert!((grads[0] - 0.6).abs() < 1e-6);
        assert!((grads[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_add_noise_deterministic_with_fixed_seed() {
        let mut rng1 = StdRng::from_seed([9u8; 32]);
        let mut rng2 = StdRng::from_seed([9u8; 32]);
        let mut g1 = vec![1.0, 2.0];
        let mut g2 = vec![1.0, 2.0];

        add_noise(&mut g1, 0.1, &mut rng1);
        add_noise(&mut g2, 0.1, &mut rng2);

        // Same seed → same CSPRNG stream → same noise.
        assert!((g1[0] - g2[0]).abs() < 1e-10);
        assert!((g1[1] - g2[1]).abs() < 1e-10);
        // …and noise actually perturbed the values.
        assert_ne!(g1[0], 1.0);
    }

    #[test]
    fn test_moments_accountant() {
        let mut acc = MomentsAccountant::new(1e-5, 0.01, 1.1);
        for _ in 0..1000
        {
            acc.step();
        }
        assert!(acc.epsilon > 0.0 && acc.epsilon.is_finite());
        assert_eq!(acc.steps, 1000);
    }

    /// The conservative accountant equals the un-subsampled RDP accountant
    /// (identical order grid), is a sound upper bound, and is monotone in steps
    /// and σ. This replaces the old test that pinned the grid-artifact value.
    #[test]
    fn moments_accountant_is_sound_and_monotone() {
        let (delta, q, sigma, steps) = (1e-5, 0.01, 1.1, 1000usize);
        let mut acc = MomentsAccountant::new(delta, q, sigma);
        for _ in 0..steps
        {
            acc.step();
        }
        // Matches the verified RDP accountant exactly (same conservative bound).
        let (eps_rdp, _alpha) = rdp_gaussian_epsilon(steps, sigma, delta);
        assert!(
            (acc.epsilon - eps_rdp).abs() < 1e-9,
            "eps = {} vs rdp = {}",
            acc.epsilon,
            eps_rdp
        );
        // Monotonicity: more steps ⇒ larger ε; more noise ⇒ smaller ε.
        let mut more_steps = MomentsAccountant::new(delta, q, sigma);
        for _ in 0..2 * steps
        {
            more_steps.step();
        }
        assert!(more_steps.epsilon > acc.epsilon);
        let mut more_noise = MomentsAccountant::new(delta, q, 4.0);
        for _ in 0..steps
        {
            more_noise.step();
        }
        assert!(more_noise.epsilon < acc.epsilon);
    }

    #[test]
    fn test_dp_protect() {
        let config = DpSgdConfig {
            l2_norm_clip: 1.0,
            noise_multiplier: 0.1,
            delta: 1e-5,
            epsilon: None,
        };
        let mut rng = test_rng();
        let mut grads = vec![2.0, 2.0, 2.0]; // norm ~3.46 > 1.0
        dp_protect(&mut grads, &config, &mut rng);
        assert_ne!(grads[0], 2.0);
    }

    #[test]
    fn dp_sgd_gradient_clips_per_sample_then_noises() {
        let config = DpSgdConfig {
            l2_norm_clip: 1.0,
            noise_multiplier: 0.0, // isolate the clip+average behavior
            delta: 1e-5,
            epsilon: None,
        };
        let mut rng = test_rng();
        // Two samples, each with norm 5 → clipped to norm 1 = [0.6, 0.8].
        let samples = vec![vec![3.0, 4.0], vec![3.0, 4.0]];
        let out = dp_sgd_gradient(&samples, &config, &mut rng);
        // Mean of two identical clipped vectors is the clipped vector itself.
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    /// Gaussian RDP and the Mironov RDP→DP conversion match their closed forms.
    #[test]
    fn rdp_gaussian_and_conversion_exact() {
        // RDP(α) = α/(2σ²): (2, 1) → 1; (4, 2) → 0.5.
        assert!((gaussian_rdp(2.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((gaussian_rdp(4.0, 2.0) - 0.5).abs() < 1e-12);
        // ε = RDP + ln(1/δ)/(α−1): (1, 2, 0.01) → 1 + ln(100)/1.
        let eps = rdp_to_dp(1.0, 2.0, 0.01);
        assert!((eps - (1.0 + 100f64.ln())).abs() < 1e-9, "eps = {eps}");
    }

    /// **The RDP accountant, tested.** For composing many Gaussian steps the RDP
    /// accountant gives an `ε` far below naive linear `(ε, δ)` composition, and it
    /// behaves monotonically (more steps ⇒ larger ε, more noise ⇒ smaller ε).
    #[test]
    fn rdp_accountant_is_tighter_than_basic_composition() {
        let (steps, sigma, delta) = (100usize, 4.0f64, 1e-5f64);
        let (eps_rdp, alpha) = rdp_gaussian_epsilon(steps, sigma, delta);
        assert!(alpha > 1.0 && eps_rdp.is_finite());

        // Naive composition: each step is (ε₀, δ/steps)-DP via the analytic Gaussian
        // ε₀ = √(2·ln(1.25/δ₀))/σ, composed linearly to (steps·ε₀, δ).
        let delta0 = delta / steps as f64;
        let eps0 = (2.0 * (1.25 / delta0).ln()).sqrt() / sigma;
        let eps_basic = steps as f64 * eps0;
        assert!(
            eps_rdp < 0.5 * eps_basic,
            "RDP {eps_rdp} not much tighter than basic {eps_basic}"
        );

        // Monotonicity.
        assert!(rdp_gaussian_epsilon(200, sigma, delta).0 > eps_rdp); // more steps
        assert!(rdp_gaussian_epsilon(steps, 8.0, delta).0 < eps_rdp); // more noise
    }
}
