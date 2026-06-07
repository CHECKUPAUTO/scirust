use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::{Initializer, Zeros};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

/// LoRA (Low-Rank Adaptation) for Linear layer.
pub struct LoRALinear {
    pub base: Linear,
    pub lora_a: Linear,
    pub lora_b: Linear,
    pub scale: f32,
}

impl LoRALinear {
    pub fn new<W: Initializer, B: Initializer>(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let base = Linear::new(in_features, out_features, w_init, b_init, rng);
        // lora_a: in -> rank (Kaiming or similar)
        let lora_a = Linear::new(in_features, rank, w_init, &Zeros, rng);
        // lora_b: rank -> out (Zeros)
        let lora_b = Linear::new(rank, out_features, &Zeros, &Zeros, rng);

        Self {
            base,
            lora_a,
            lora_b,
            scale: alpha / rank as f32,
        }
    }

    /// Convert an existing Linear layer to LoRALinear.
    pub fn from_linear(base: Linear, rank: usize, alpha: f32, rng: &mut PcgEngine) -> Self {
        let in_features = base.in_features;
        let out_features = base.out_features;
        let lora_a = Linear::new(in_features, rank, &crate::nn::init::KaimingNormal, &Zeros, rng);
        let lora_b = Linear::new(rank, out_features, &Zeros, &Zeros, rng);
        Self {
            base,
            lora_a,
            lora_b,
            scale: alpha / rank as f32,
        }
    }
}

impl Module for LoRALinear {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let base_out = self.base.forward(tape, input.clone());
        let lora_h = self.lora_a.forward(tape, input);
        let lora_out = self.lora_b.forward(tape, lora_h);

        base_out.add(lora_out.scale(self.scale))
    }

    fn parameter_indices(&self) -> Vec<usize> {
        // Typically we only train LoRA parameters
        let mut v = Vec::new();
        v.extend(self.lora_a.parameter_indices());
        v.extend(self.lora_b.parameter_indices());
        v
    }

    fn sync(&mut self, tape: &Tape) {
        self.lora_a.sync(tape);
        self.lora_b.sync(tape);
        // base remains frozen usually
    }
}
