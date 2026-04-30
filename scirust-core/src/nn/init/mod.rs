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
        for x in t.data.iter_mut() {
            *x = rng.normal(0.0, std);
        }
    }
}

// ---------- XavierUniform ---------- //

pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill(&self, t: &mut Tensor, fan_in: usize, fan_out: usize, rng: &mut PcgEngine) {
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        for x in t.data.iter_mut() {
            *x = rng.float_signed() * bound;
        }
    }
}

// ---------- Zeros ---------- //

pub struct Zeros;

impl Initializer for Zeros {
    fn fill(&self, t: &mut Tensor, _fan_in: usize, _fan_out: usize, _rng: &mut PcgEngine) {
        for x in t.data.iter_mut() { *x = 0.0; }
    }
}

// ---------- SmallNormal ---------- //

pub struct SmallNormal {
    pub std: f32,
}

impl SmallNormal {
    pub fn new(std: f32) -> Self { Self { std } }
}

impl Initializer for SmallNormal {
    fn fill(&self, t: &mut Tensor, _fan_in: usize, _fan_out: usize, rng: &mut PcgEngine) {
        for x in t.data.iter_mut() {
            *x = rng.normal(0.0, self.std);
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
        assert!(max_abs > 0.01, "Kaiming produced near-zero values: max_abs = {max_abs}");
    }

    #[test]
    fn kaiming_std_close_to_target() {
        let fan_in = 100;
        let mut t = Tensor::zeros(fan_in, 100);
        let mut rng = PcgEngine::new(42);
        KaimingNormal.fill(&mut t, fan_in, 100, &mut rng);
        let mean: f32 = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let var: f32 = t.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                       / t.data.len() as f32;
        let expected_std = (2.0 / fan_in as f32).sqrt();
        let actual_std = var.sqrt();
        assert!((actual_std - expected_std).abs() / expected_std < 0.1,
            "Kaiming std: expected {expected_std}, got {actual_std}");
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
        for &x in &t.data {
            assert!(x.abs() <= bound + 1e-5, "Xavier value out of bounds: {x} > {bound}");
        }
    }
}
