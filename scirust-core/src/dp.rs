//! Differential Privacy — DP-SGD training.
//!
//! Implements the differentially private stochastic gradient descent
//! algorithm from Abadi et al. (2016):
//! 1. Clip per-sample gradients to a maximum L2 norm.
//! 2. Add Gaussian noise calibrated to achieve (ε, δ)-DP.
//! 3. Track privacy budget via moments accountant (Rényi DP).
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::dp::{DpSgdConfig, clip_gradients, add_noise};
//!
//! let config = DpSgdConfig {
//!     l2_norm_clip: 1.0,
//!     noise_multiplier: 1.1,
//!     delta: 1e-5,
//! };
//!
//! clip_gradients(&mut grads, config.l2_norm_clip);
//! add_noise(&mut grads, config.noise_multiplier, &mut rng);
//! ```

use crate::nn::rng::PcgEngine;

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

/// Moments accountant for tracking privacy budget (Rényi DP).
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Total ε consumed so far.
    pub epsilon: f64,
    /// δ parameter.
    pub delta: f64,
    /// Total number of training steps.
    pub steps: usize,
    /// Sampling probability q = batch_size / dataset_size.
    pub q: f64,
    /// Noise multiplier σ.
    pub sigma: f64,
    /// Accumulated Rényi divergences at various orders α.
    pub orders: Vec<f64>,
    /// Accumulated log moments for each order.
    pub log_moments: Vec<f64>,
}

impl MomentsAccountant {
    /// Create a new moments accountant.
    pub fn new(delta: f64, q: f64, sigma: f64) -> Self {
        // Orders α for Rényi divergence computation
        let orders: Vec<f64> = (1..=100)
            .map(|i| 1.0 + (i as f64) * 0.1) // 1.1, 1.2, ..., 11.0
            .collect();
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
        // Compute Rényi divergence for each order α
        // Using the tight bound from Abadi et al. (2016)
        for (i, &alpha) in self.orders.iter().enumerate()
        {
            let moment = compute_log_moment(alpha, self.q, self.sigma);
            self.log_moments[i] += moment;
        }

        self.steps += 1;

        // Convert to (ε, δ)-DP via the optimal conversion
        self.epsilon = self.compute_epsilon();
    }

    /// Compute total ε from accumulated moments.
    fn compute_epsilon(&self) -> f64 {
        let mut best_eps = f64::INFINITY;

        for (i, &alpha) in self.orders.iter().enumerate()
        {
            // Abadi et al. (2016), Thm 2: for a target δ the tightest ε is
            // min_α (α_M(λ) − ln δ) / λ. The whole numerator is divided by α,
            // not just the ln δ term.
            let eps = (self.log_moments[i] - self.delta.ln()) / alpha;
            if eps < best_eps
            {
                best_eps = eps;
            }
        }

        best_eps.max(0.0)
    }
}

/// Compute log moment for a given α (Rényi divergence order).
fn compute_log_moment(alpha: f64, q: f64, sigma: f64) -> f64 {
    // Simplified bound for Gaussian mechanism with subsampling
    // From Abadi et al. 2016, Theorem 2 (with simplifications)
    if q == 0.0 || sigma == 0.0
    {
        return 0.0;
    }

    // Upper bound: α * q^2 / (2 * σ^2) for small α and q
    alpha * q * q / (2.0 * sigma * sigma)
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
/// σ = noise_multiplier * l2_norm_clip (should match clip_gradients).
pub fn add_noise(grads: &mut [f32], noise_stddev: f32, rng: &mut PcgEngine) {
    for g in grads.iter_mut()
    {
        *g += rng.normal(0.0, noise_stddev);
    }
}

/// Clip per-sample gradients and add noise in one step.
pub fn dp_protect(per_sample_grads: &mut [f32], config: &DpSgdConfig, rng: &mut PcgEngine) {
    clip_gradients(per_sample_grads, config.l2_norm_clip);
    let noise_std = config.noise_multiplier * config.l2_norm_clip;
    add_noise(per_sample_grads, noise_std, rng);
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_add_noise_deterministic() {
        let mut rng1 = PcgEngine::new(42);
        let mut rng2 = PcgEngine::new(42);
        let mut g1 = vec![1.0, 2.0];
        let mut g2 = vec![1.0, 2.0];

        add_noise(&mut g1, 0.1, &mut rng1);
        add_noise(&mut g2, 0.1, &mut rng2);

        // Same seed → same noise
        assert!((g1[0] - g2[0]).abs() < 1e-10);
        assert!((g1[1] - g2[1]).abs() < 1e-10);
    }

    #[test]
    fn test_moments_accountant() {
        let mut acc = MomentsAccountant::new(1e-5, 0.01, 1.1);

        // 1000 steps
        for _ in 0..1000
        {
            acc.step();
        }

        // Should have consumed some privacy budget
        assert!(acc.epsilon > 0.0);
        assert_eq!(acc.steps, 1000);
    }

    /// Regression: the moments accountant must use the Abadi (2016) conversion
    /// ε = min_α (α_M(λ) − ln δ) / λ — i.e. the *whole* numerator is divided by
    /// α, not just the ln δ term. The earlier bug (`α_M − ln δ/α`) left the
    /// log-moment undivided and over-estimated ε.
    #[test]
    fn test_moments_accountant_abadi_conversion() {
        let (delta, q, sigma, steps) = (1e-5, 0.01, 1.1, 1000usize);
        let mut acc = MomentsAccountant::new(delta, q, sigma);
        for _ in 0..steps
        {
            acc.step();
        }

        // Closed form of the correct conversion. log_moments[i] = steps·α·q²/(2σ²),
        // so ε(α) = steps·q²/(2σ²) − ln δ / α, minimized at the largest α (11.0).
        let per_step = q * q / (2.0 * sigma * sigma);
        let expected = acc
            .orders
            .iter()
            .map(|&alpha| (steps as f64 * alpha * per_step - delta.ln()) / alpha)
            .fold(f64::INFINITY, f64::min);

        assert!((acc.epsilon - expected).abs() < 1e-9, "eps = {}", acc.epsilon);
        // The buggy formula (α_M − ln δ/α) yields ≈ 1.5012 here; the correct one
        // ≈ 1.0880. Guard against a regression back to the larger value.
        assert!((acc.epsilon - 1.087_951_901_774_153).abs() < 1e-9);
        assert!(acc.epsilon < 1.2);
    }

    #[test]
    fn test_dp_protect() {
        let config = DpSgdConfig {
            l2_norm_clip: 1.0,
            noise_multiplier: 0.1,
            delta: 1e-5,
            epsilon: None,
        };
        let mut rng = PcgEngine::new(99);
        let mut grads = vec![2.0, 2.0, 2.0]; // norm ~3.46 > 1.0

        dp_protect(&mut grads, &config, &mut rng);

        // After clipping, norm should be 1.0 (before noise)
        // After noise, values differ from original
        assert_ne!(grads[0], 2.0);
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
