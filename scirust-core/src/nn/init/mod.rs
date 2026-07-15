// scirust-core/src/nn/init/mod.rs
//
// Initialiseurs de poids pour modules NN.
//
// Quatre choix standards :
//   - KaimingNormal : N(0, sqrt(2/fan_in)). Recommandé avec ReLU.
//   - XavierUniform : Uniform(±sqrt(6/(fan_in+fan_out))). Recommandé avec sigmoid/tanh.
//   - Zeros         : tous zéros. Pour les biais.
//   - SmallNormal   : N(0, std). Pour embeddings et init custom.

use crate::autodiff::reverse::Tensor;
use crate::nn::rng::PcgEngine;

pub trait Initializer {
    /// Remplit un tenseur avec des valeurs initialisées.
    /// fan_in et fan_out sont nécessaires pour Kaiming/Xavier ;
    /// les autres initialiseurs peuvent les ignorer.
    fn fill(&self, t: &mut Tensor, fan_in: usize, fan_out: usize, rng: &mut PcgEngine);
}

// ---------- KaimingNormal ---------- //

pub struct KaimingNormal;

impl Initializer for KaimingNormal {
    fn fill(&self, t: &mut Tensor, fan_in: usize, _fan_out: usize, rng: &mut PcgEngine) {
        let std = (2.0 / fan_in as f32).sqrt();
        for x in t.data.iter_mut()
        {
            *x = rng.normal(0.0, std);
        }
    }
}

// ---------- XavierUniform ---------- //

pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill(&self, t: &mut Tensor, fan_in: usize, fan_out: usize, rng: &mut PcgEngine) {
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        for x in t.data.iter_mut()
        {
            *x = rng.float_signed() * bound;
        }
    }
}

// ---------- Zeros ---------- //

pub struct Zeros;

impl Initializer for Zeros {
    fn fill(&self, t: &mut Tensor, _fan_in: usize, _fan_out: usize, _rng: &mut PcgEngine) {
        for x in t.data.iter_mut()
        {
            *x = 0.0;
        }
    }
}

// ---------- SmallNormal ---------- //

pub struct SmallNormal {
    pub std: f32,
}

impl SmallNormal {
    pub fn new(std: f32) -> Self {
        Self { std }
    }
}

impl Initializer for SmallNormal {
    fn fill(&self, t: &mut Tensor, _fan_in: usize, _fan_out: usize, rng: &mut PcgEngine) {
        for x in t.data.iter_mut()
        {
            *x = rng.normal(0.0, self.std);
        }
    }
}

// ---------- TruncatedNormal ---------- //

/// Truncated-normal initializer: samples from `N(mean, std²)` restricted to
/// `[mean − k·std, mean + k·std]` (default `k = 2`). This is the standard
/// ViT / BERT / timm weight init — values in the far tails, which a plain
/// normal occasionally produces, are excluded so no weight starts pathologically
/// large.
///
/// Uses the deterministic **inverse-CDF transform** (one draw per element, no
/// rejection loop): draw a uniform in the truncated CDF interval and map it
/// through the standard-normal quantile `√2 · erfinv`, with the CDF bounds from
/// `erf` — both from [`scirust_special`]. The CDF math is done in `f64` for
/// accuracy, then rounded to `f32`.
pub struct TruncatedNormal {
    pub mean: f32,
    pub std: f32,
    /// Truncation half-width in units of `std` (default `2.0`).
    pub bound: f32,
}

impl TruncatedNormal {
    /// Truncated normal with the conventional `±2·std` bounds.
    pub fn new(mean: f32, std: f32) -> Self {
        Self {
            mean,
            std,
            bound: 2.0,
        }
    }

    /// Truncated normal with an explicit truncation half-width `bound` (in units
    /// of `std`).
    pub fn with_bound(mean: f32, std: f32, bound: f32) -> Self {
        Self { mean, std, bound }
    }
}

impl Initializer for TruncatedNormal {
    fn fill(&self, t: &mut Tensor, _fan_in: usize, _fan_out: usize, rng: &mut PcgEngine) {
        let (mean, std, k) = (self.mean as f64, self.std as f64, self.bound.abs() as f64);
        // Degenerate spread → every weight is the (clamped) mean.
        if std <= 0.0 || k <= 0.0
        {
            for x in t.data.iter_mut()
            {
                *x = self.mean;
            }
            return;
        }
        let a = mean - k * std;
        let b = mean + k * std;
        // Standard-normal CDF Φ(x) = ½(1 + erf(x/√2)); bounds are symmetric at ±k.
        let norm_cdf = |x: f64| 0.5 * (1.0 + scirust_special::erf(x / std::f64::consts::SQRT_2));
        let l = norm_cdf(-k);
        let u = norm_cdf(k);
        let span = 2.0 * (u - l);
        let scale = std * std::f64::consts::SQRT_2;
        for x in t.data.iter_mut()
        {
            // Uniform in [2l−1, 2u−1], then the standard-normal inverse CDF.
            let p = (2.0 * l - 1.0) + rng.float() as f64 * span;
            let mut v = mean + scale * scirust_special::erfinv(p);
            // Guard the ends against erfinv's boundary blow-up / rounding.
            if v < a
            {
                v = a;
            }
            else if v > b
            {
                v = b;
            }
            *x = v as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiming_produces_nonzero_values() {
        let mut t = Tensor::zeros(8, 16);
        let mut rng = PcgEngine::new(42);
        KaimingNormal.fill(&mut t, 16, 8, &mut rng);
        let max_abs: f32 = t.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_abs > 0.01,
            "Kaiming produced near-zero values: max_abs = {max_abs}"
        );
    }

    #[test]
    fn kaiming_std_close_to_target() {
        let fan_in = 100;
        let mut t = Tensor::zeros(fan_in, 100);
        let mut rng = PcgEngine::new(42);
        KaimingNormal.fill(&mut t, fan_in, 100, &mut rng);
        let mean: f32 = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let var: f32 = t.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.data.len() as f32;
        let expected_std = (2.0 / fan_in as f32).sqrt();
        let actual_std = var.sqrt();
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.1,
            "Kaiming std: expected {expected_std}, got {actual_std}"
        );
    }

    #[test]
    fn zeros_produces_zeros() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3);
        let mut rng = PcgEngine::new(0);
        Zeros.fill(&mut t, 1, 1, &mut rng);
        assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn xavier_in_bounded_range() {
        let fan_in = 10;
        let fan_out = 5;
        let mut t = Tensor::zeros(fan_in, fan_out);
        let mut rng = PcgEngine::new(42);
        XavierUniform.fill(&mut t, fan_in, fan_out, &mut rng);
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        for &x in &t.data
        {
            assert!(
                x.abs() <= bound + 1e-5,
                "Xavier value out of bounds: {x} > {bound}"
            );
        }
    }

    #[test]
    fn truncated_normal_stays_within_bounds() {
        let (mean, std) = (0.5f32, 0.2f32);
        let mut t = Tensor::zeros(200, 200);
        let mut rng = PcgEngine::new(7);
        TruncatedNormal::new(mean, std).fill(&mut t, 0, 0, &mut rng);
        let (lo, hi) = (mean - 2.0 * std, mean + 2.0 * std);
        for &x in &t.data
        {
            assert!(x >= lo - 1e-6 && x <= hi + 1e-6, "out of ±2σ bounds: {x}");
        }
    }

    #[test]
    fn truncated_normal_recovers_mean_and_reduced_std() {
        let (mean, std) = (0.0f32, 1.0f32);
        let mut t = Tensor::zeros(300, 300);
        let mut rng = PcgEngine::new(11);
        TruncatedNormal::new(mean, std).fill(&mut t, 0, 0, &mut rng);
        let m: f32 = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let v: f32 = t.data.iter().map(|x| (x - m).powi(2)).sum::<f32>() / t.data.len() as f32;
        // Empirical mean ≈ 0; variance of a ±2σ-truncated standard normal is
        // ≈ 0.774 (< 1, since the tails are removed).
        assert!(m.abs() < 0.02, "mean drifted: {m}");
        assert!((v - 0.774).abs() < 0.05, "truncated variance off: {v}");
    }

    #[test]
    fn truncated_normal_is_reproducible() {
        let mut a = Tensor::zeros(16, 16);
        let mut b = Tensor::zeros(16, 16);
        TruncatedNormal::new(0.0, 0.02).fill(&mut a, 0, 0, &mut PcgEngine::new(3));
        TruncatedNormal::new(0.0, 0.02).fill(&mut b, 0, 0, &mut PcgEngine::new(3));
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn truncated_normal_zero_std_is_constant() {
        let mut t = Tensor::zeros(4, 4);
        TruncatedNormal::new(1.5, 0.0).fill(&mut t, 0, 0, &mut PcgEngine::new(1));
        assert!(t.data.iter().all(|&x| x == 1.5));
    }
}
