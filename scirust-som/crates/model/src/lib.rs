//! SOM Model Architecture (<= 300M parameters).
//! A Graph Transformer backbone for ownership prediction.

use scirust_core::autodiff::reverse::{Tape, Var};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::Module;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::init::{KaimingNormal, Zeros};

pub struct SomModel {
    pub transformer_backbone: Vec<Linear>,
    pub ownership_head: Linear,
    pub borrow_head: Linear,
    pub lifetime_head: Linear,
    pub alias_head: Linear,
    pub escape_head: Linear,
    pub mutability_head: Linear,
    pub unsafe_head: Linear,
    pub confidence_head: Linear,
}

impl SomModel {
    pub fn new(d_model: usize, n_layers: usize, rng: &mut PcgEngine) -> Self {
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(Linear::new(d_model, d_model, &KaimingNormal, &Zeros, rng));
        }

        Self {
            transformer_backbone: layers,
            ownership_head: Linear::new(d_model, 4, &KaimingNormal, &Zeros, rng),
            borrow_head: Linear::new(d_model, 3, &KaimingNormal, &Zeros, rng),
            lifetime_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            alias_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            escape_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            mutability_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            unsafe_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
            confidence_head: Linear::new(d_model, 1, &KaimingNormal, &Zeros, rng),
        }
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>) -> SomOutput<'t> {
        let mut h = x;
        for layer in &mut self.transformer_backbone {
            h = layer.forward(tape, h).relu();
        }

        SomOutput {
            ownership: self.ownership_head.forward(tape, h),
            borrow: self.borrow_head.forward(tape, h),
            lifetime: self.lifetime_head.forward(tape, h),
            alias: self.alias_head.forward(tape, h),
            escape: self.escape_head.forward(tape, h),
            mutability: self.mutability_head.forward(tape, h),
            unsafe_prob: self.unsafe_head.forward(tape, h),
            confidence: self.confidence_head.forward(tape, h),
        }
    }
}

pub struct SomOutput<'t> {
    pub ownership: Var<'t>,
    pub borrow: Var<'t>,
    pub lifetime: Var<'t>,
    pub alias: Var<'t>,
    pub escape: Var<'t>,
    pub mutability: Var<'t>,
    pub unsafe_prob: Var<'t>,
    pub confidence: Var<'t>,
}
