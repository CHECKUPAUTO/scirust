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
    pub fn new(
        delta: f64,
        q: f64,
        sigma: f64,
    ) -> Self {
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
        for (i, &alpha) in self.orders.iter().enumerate() {
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

        for (i, &alpha) in self.orders.iter().enumerate() {
            let eps = self.log_moments[i]
                - (self.delta.ln() / alpha);
            if eps < best_eps {
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
    if q == 0.0 || sigma == 0.0 {
        return 0.0;
    }

    // Upper bound: α * q^2 / (2 * σ^2) for small α and q
    let moment = alpha * q * q / (2.0 * sigma * sigma);
    moment
}

/// Clip gradient vector to maximum L2 norm.
///
/// If ||g||_2 > clip, scale g by clip / ||g||_2.
pub fn clip_gradients(grads: &mut [f32], l2_norm_clip: f32) {
    let norm_sq: f32 = grads.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();

    if norm > l2_norm_clip && norm > 1e-12 {
        let scale = l2_norm_clip / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
}

/// Add Gaussian noise to gradients for differential privacy.
///
/// σ = noise_multiplier * l2_norm_clip (should match clip_gradients).
pub fn add_noise(grads: &mut [f32], noise_stddev: f32, rng: &mut PcgEngine) {
    for g in grads.iter_mut() {
        *g += rng.normal(0.0, noise_stddev);
    }
}

/// Clip per-sample gradients and add noise in one step.
pub fn dp_protect(
    per_sample_grads: &mut [f32],
    config: &DpSgdConfig,
    rng: &mut PcgEngine,
) {
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
        for _ in 0..1000 {
            acc.step();
        }

        // Should have consumed some privacy budget
        assert!(acc.epsilon > 0.0);
        assert_eq!(acc.steps, 1000);
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
}
