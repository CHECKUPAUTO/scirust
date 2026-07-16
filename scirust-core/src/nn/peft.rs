use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::{Initializer, Zeros};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use std::collections::HashMap;

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
        let lora_a = Linear::new(
            in_features,
            rank,
            &crate::nn::init::KaimingNormal,
            &Zeros,
            rng,
        );
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
        let base_out = self.base.forward(tape, input);
        let lora_h = self.lora_a.forward(tape, input);
        let lora_out = self.lora_b.forward(tape, lora_h);

        base_out.try_add(lora_out.scale(self.scale)).unwrap()
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

    fn state_dict(&self) -> HashMap<String, Tensor> {
        // The base is frozen during training but still part of the checkpoint:
        // dropping it would make the saved adapter unusable on reload.
        let mut map = HashMap::new();
        map.insert("base.weight".to_string(), self.base.weight.clone());
        map.insert("base.bias".to_string(), self.base.bias.clone());
        map.insert("lora_a.weight".to_string(), self.lora_a.weight.clone());
        map.insert("lora_a.bias".to_string(), self.lora_a.bias.clone());
        map.insert("lora_b.weight".to_string(), self.lora_b.weight.clone());
        map.insert("lora_b.bias".to_string(), self.lora_b.bias.clone());
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        load_linear(&mut self.base, sd, "base")?;
        load_linear(&mut self.lora_a, sd, "lora_a")?;
        load_linear(&mut self.lora_b, sd, "lora_b")?;
        Ok(())
    }
}

/// Load `{prefix}.weight` / `{prefix}.bias` into a sub-Linear, erroring on
/// missing keys or shape mismatches (same convention as `Linear`).
fn load_linear(
    sub: &mut Linear,
    sd: &HashMap<String, Tensor>,
    prefix: &str,
) -> crate::error::Result<()> {
    let w = sd
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("missing key: {prefix}.weight"))?;
    let b = sd
        .get(&format!("{prefix}.bias"))
        .ok_or_else(|| format!("missing key: {prefix}.bias"))?;
    if w.shape() != (sub.in_features, sub.out_features)
    {
        crate::bail!(
            "{prefix}.weight shape mismatch: expected {:?}, got {:?}",
            (sub.in_features, sub.out_features),
            w.shape()
        );
    }
    if b.shape() != (1, sub.out_features)
    {
        crate::bail!(
            "{prefix}.bias shape mismatch: expected {:?}, got {:?}",
            (1, sub.out_features),
            b.shape()
        );
    }
    sub.weight = w.clone();
    sub.bias = b.clone();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::KaimingNormal;

    #[test]
    fn lora_state_dict_round_trip() {
        let mut rng = PcgEngine::new(42);
        let mut lora1 = LoRALinear::new(4, 3, 2, 4.0, &KaimingNormal, &Zeros, &mut rng);
        // Give every parameter a distinctive value (lora_b and the biases
        // start at zero, which would make the round trip vacuous).
        lora1.lora_b.weight = Tensor::from_vec((0..6).map(|i| i as f32 + 1.0).collect(), 2, 3);
        lora1.lora_a.bias = Tensor::from_vec(vec![0.5, -0.5], 1, 2);
        lora1.base.bias = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3);

        let sd = lora1.state_dict();
        assert_eq!(sd.len(), 6);

        let mut rng2 = PcgEngine::new(99);
        let mut lora2 = LoRALinear::new(4, 3, 2, 4.0, &Zeros, &Zeros, &mut rng2);
        // Missing keys must be an error, not a silent skip.
        assert!(lora2.load_state_dict(&HashMap::new()).is_err());
        lora2.load_state_dict(&sd).unwrap();

        // Compare every parameter tensor through the tape values.
        let tape1 = Tape::new();
        let tape2 = Tape::new();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let _ = lora1.forward(&tape1, tape1.input(x.clone()));
        let _ = lora2.forward(&tape2, tape2.input(x));
        // parameter_indices only covers the LoRA adapters; add the frozen base.
        let idx1 = [lora1.base.parameter_indices(), lora1.parameter_indices()].concat();
        let idx2 = [lora2.base.parameter_indices(), lora2.parameter_indices()].concat();
        assert_eq!(idx1.len(), 6);
        assert_eq!(idx1.len(), idx2.len());
        for (a, b) in idx1.iter().zip(&idx2)
        {
            assert_eq!(tape1.value(*a).data, tape2.value(*b).data);
        }
    }

    #[test]
    fn lora_load_rejects_shape_mismatch() {
        let mut rng = PcgEngine::new(42);
        let mut lora = LoRALinear::new(4, 3, 2, 4.0, &KaimingNormal, &Zeros, &mut rng);
        let mut sd = lora.state_dict();
        sd.insert("lora_a.weight".to_string(), Tensor::zeros(5, 2));
        let res = lora.load_state_dict(&sd);
        assert!(res.is_err(), "expected error on shape mismatch");
    }
}
