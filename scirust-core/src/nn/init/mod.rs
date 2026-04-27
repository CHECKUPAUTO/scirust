// scirust-core/src/nn/init/mod.rs
//
// Trait Initializer + implémentations.
//
// Modèle : un Initializer remplit en place un Tensor déjà alloué,
// en consommant un PcgEngine. Permet de séparer "allouer" (Linear connaît
// les dims) de "remplir" (l'utilisateur choisit la stratégie).
//
// Usage :
//   let mut rng = PcgEngine::new(42);
//   let init = KaimingNormal::default();
//   let layer = Linear::new(784, 128, &init, &mut rng);

use crate::autodiff::reverse::Tensor;
use crate::nn::rng::PcgEngine;

// ================================================================== //
//  Trait                                                              //
// ================================================================== //

pub trait Initializer {
    /// Remplit `tensor` selon la stratégie d'initialisation.
    /// Le Tensor a déjà ses dimensions définies — l'init n'écrit que
    /// dans `tensor.data`.
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine);
}

// ================================================================== //
//  Constant — utile pour les biais (zéros par défaut) ou debug        //
// ================================================================== //

pub struct Constant(pub f32);

impl Initializer for Constant {
    fn fill(&self, tensor: &mut Tensor, _rng: &mut PcgEngine) {
        for x in tensor.data.iter_mut() { *x = self.0; }
    }
}

pub struct Zeros;
impl Initializer for Zeros {
    fn fill(&self, tensor: &mut Tensor, _rng: &mut PcgEngine) {
        for x in tensor.data.iter_mut() { *x = 0.0; }
    }
}

pub struct Ones;
impl Initializer for Ones {
    fn fill(&self, tensor: &mut Tensor, _rng: &mut PcgEngine) {
        for x in tensor.data.iter_mut() { *x = 1.0; }
    }
}

// ================================================================== //
//  Uniform / Normal — paramétrables                                   //
// ================================================================== //

pub struct Uniform { pub low: f32, pub high: f32 }

impl Initializer for Uniform {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        for x in tensor.data.iter_mut() { *x = rng.uniform(self.low, self.high); }
    }
}

pub struct Normal { pub mean: f32, pub std: f32 }

impl Initializer for Normal {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        for x in tensor.data.iter_mut() { *x = self.mean + rng.normal() * self.std; }
    }
}

// ================================================================== //
//  Xavier/Glorot — adapté aux activations sigmoïde/tanh               //
// ================================================================== //
//
//   bound = √(6 / (fan_in + fan_out))
//   W ~ U(-bound, bound)
//
// Glorot & Bengio 2010 — pour préserver la variance à travers les couches
// avec des activations à pente bornée.

pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        let (fan_in, fan_out) = (tensor.rows, tensor.cols);
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        for x in tensor.data.iter_mut() { *x = rng.uniform(-bound, bound); }
    }
}

pub struct XavierNormal;

impl Initializer for XavierNormal {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        let (fan_in, fan_out) = (tensor.rows, tensor.cols);
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        for x in tensor.data.iter_mut() { *x = rng.normal() * std; }
    }
}

// ================================================================== //
//  Kaiming/He — adapté à ReLU                                         //
// ================================================================== //
//
//   std = √(2 / fan_in)        (mode "fan_in", défaut)
//   W ~ N(0, std²)
//
// He et al. 2015 — ReLU annule la moitié des activations en moyenne,
// le facteur 2 compense cette perte de variance.

pub struct KaimingNormal;

impl Initializer for KaimingNormal {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        let fan_in = tensor.rows;
        let std = (2.0 / fan_in as f32).sqrt();
        for x in tensor.data.iter_mut() { *x = rng.normal() * std; }
    }
}

pub struct KaimingUniform;

impl Initializer for KaimingUniform {
    fn fill(&self, tensor: &mut Tensor, rng: &mut PcgEngine) {
        let fan_in = tensor.rows;
        let bound = (6.0 / fan_in as f32).sqrt();
        for x in tensor.data.iter_mut() { *x = rng.uniform(-bound, bound); }
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_fills_correctly() {
        let mut t = Tensor::zeros(3, 4);
        let mut rng = PcgEngine::new(0);
        Constant(7.5).fill(&mut t, &mut rng);
        assert!(t.data.iter().all(|&x| x == 7.5));
    }

    #[test]
    fn xavier_uniform_in_bounds() {
        let mut t = Tensor::zeros(10, 20);
        let mut rng = PcgEngine::new(42);
        XavierUniform.fill(&mut t, &mut rng);
        let bound = (6.0_f32 / 30.0).sqrt();
        for &x in &t.data {
            assert!(x >= -bound && x < bound, "x = {x}, bound = {bound}");
        }
    }

    #[test]
    fn kaiming_normal_finite() {
        let mut t = Tensor::zeros(64, 128);
        let mut rng = PcgEngine::new(42);
        KaimingNormal.fill(&mut t, &mut rng);
        // Sanity : aucune valeur infinie ou NaN
        assert!(t.data.iter().all(|x| x.is_finite()));
        // Variance approximative ≈ 2/64 = 0.03125
        let mean = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let var: f32 = t.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                       / t.data.len() as f32;
        let expected = 2.0 / 64.0;
        // Tolérance 30% sur l'estimation de variance
        assert!((var - expected).abs() / expected < 0.3,
                "var = {var}, expected = {expected}");
    }

    #[test]
    fn determinism_across_invocations() {
        // Deux remplissages avec la même graine doivent produire le même tenseur
        let mut t1 = Tensor::zeros(5, 5);
        let mut t2 = Tensor::zeros(5, 5);
        let mut r1 = PcgEngine::new(99);
        let mut r2 = PcgEngine::new(99);
        XavierNormal.fill(&mut t1, &mut r1);
        XavierNormal.fill(&mut t2, &mut r2);
        assert_eq!(t1.data, t2.data);
    }
}
