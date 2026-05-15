// scirust-core/src/nn/transformer/block.rs
//
// TransformerBlock — un bloc encoder Pre-LN style.
//
// Architecture (Pre-LN, recommandee pour la stabilite d'entrainement) :
//
//   x' = x + MultiHeadAttention(LayerNorm(x))
//   y  = x' + FeedForward(LayerNorm(x'))

use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::Initializer;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::attention::MultiHeadAttention;
use crate::tensor::tensor3d::Var3D;
use std::collections::HashMap;

pub struct TransformerBlock {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub ln1: LayerNorm, // pre-attention
    pub mha: MultiHeadAttention,
    pub ln2: LayerNorm, // pre-FFN
    pub ffn1: Linear,   // d_model -> d_ff
    pub ffn2: Linear,   // d_ff -> d_model
    pub name: String,
}

impl TransformerBlock {
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        causal: bool,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut s = Self {
            d_model,
            n_heads,
            d_ff,
            ln1: LayerNorm::new(d_model, 1e-5, w_init, rng),
            mha: MultiHeadAttention::new(d_model, n_heads, 0, causal, w_init, b_init, rng),
            ln2: LayerNorm::new(d_model, 1e-5, w_init, rng),
            ffn1: Linear::new(d_model, d_ff, w_init, b_init, rng),
            ffn2: Linear::new(d_ff, d_model, w_init, b_init, rng),
            name: format!("tb_d{d_model}_h{n_heads}_ff{d_ff}"),
        };
        s.ln1.name = format!("{}.ln1", s.name);
        s.ln2.name = format!("{}.ln2", s.name);
        s.mha = s.mha.with_name(&format!("{}.mha", s.name));
        s
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self.ln1.name = format!("{}.ln1", self.name);
        self.ln2.name = format!("{}.ln2", self.name);
        self.mha = self.mha.with_name(&format!("{}.mha", self.name));
        self
    }

    /// Forward pour input 3D (B, T, D).
    pub fn forward_3d<'t>(&mut self, tape: &'t Tape, x_3d: Var3D<'t>) -> Var3D<'t> {
        let (batch, seq_len, d_model) = x_3d.shape();
        assert_eq!(d_model, self.d_model);

        // Sous-couche 1 : x + Attention(LN(x))
        let ln1_out = self.ln1.forward(tape, x_3d.as_var());
        let ln1_3d = Var3D::from_var(ln1_out, batch, seq_len, d_model);
        let attn_out = self.mha.forward_3d(tape, ln1_3d);
        let x1 = x_3d.as_var().add(attn_out.as_var());

        // Sous-couche 2 : x1 + FFN(LN(x1))
        let ln2_out = self.ln2.forward(tape, x1);
        let h_ff = self.ffn1.forward(tape, ln2_out).relu();
        let ffn_out = self.ffn2.forward(tape, h_ff);
        let x2 = x1.add(ffn_out);

        Var3D::from_var(x2, batch, seq_len, d_model)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.ln1.parameter_indices());
        v.extend(self.mha.parameter_indices());
        v.extend(self.ln2.parameter_indices());
        v.extend(self.ffn1.parameter_indices());
        v.extend(self.ffn2.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.ln1.sync(tape);
        self.mha.sync(tape);
        self.ln2.sync(tape);
        self.ffn1.sync(tape);
        self.ffn2.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let p = &self.name;
        // Child modules (ln1, mha, ln2) already have globally-unique names
        for (k, v) in self.ln1.state_dict() {
            map.insert(k, v);
        }
        for (k, v) in self.mha.state_dict() {
            map.insert(k, v);
        }
        for (k, v) in self.ln2.state_dict() {
            map.insert(k, v);
        }
        // FFN linears have no names, prefix manually
        map.insert(format!("{p}.ffn1.weight"), self.ffn1.weight.clone());
        map.insert(format!("{p}.ffn1.bias"), self.ffn1.bias.clone());
        map.insert(format!("{p}.ffn2.weight"), self.ffn2.weight.clone());
        map.insert(format!("{p}.ffn2.bias"), self.ffn2.bias.clone());
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        self.ln1.load_state_dict(sd)?;
        self.mha.load_state_dict(sd)?;
        self.ln2.load_state_dict(sd)?;

        let p = &self.name;
        let ffn1_w = sd
            .get(&format!("{p}.ffn1.weight"))
            .ok_or_else(|| format!("missing key: {p}.ffn1.weight"))?;
        let ffn1_b = sd
            .get(&format!("{p}.ffn1.bias"))
            .ok_or_else(|| format!("missing key: {p}.ffn1.bias"))?;
        let ffn2_w = sd
            .get(&format!("{p}.ffn2.weight"))
            .ok_or_else(|| format!("missing key: {p}.ffn2.weight"))?;
        let ffn2_b = sd
            .get(&format!("{p}.ffn2.bias"))
            .ok_or_else(|| format!("missing key: {p}.ffn2.bias"))?;

        self.ffn1.weight = ffn1_w.clone();
        self.ffn1.bias = ffn1_b.clone();
        self.ffn2.weight = ffn2_w.clone();
        self.ffn2.bias = ffn2_b.clone();
        Ok(())
    }
}

impl Clone for TransformerBlock {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
            ln1: self.ln1.clone(),
            mha: self.mha.clone(),
            ln2: self.ln2.clone(),
            ffn1: self.ffn1.clone(),
            ffn2: self.ffn2.clone(),
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn block_forward_shape_preserved() {
        let mut rng = PcgEngine::new(0);
        let mut b = TransformerBlock::new(16, 4, 64, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (0..128).map(|x| x as f32 * 0.01).collect(),
            8,
            16,
        ));
        let x_3d = Var3D::from_var(x, 2, 4, 16);
        let out = b.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 4, 16));
    }

    #[test]
    fn block_residual_path_works() {
        let mut rng = PcgEngine::new(0);
        let mut b = TransformerBlock::new(8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        b.ffn2.weight = Tensor::zeros(b.ffn2.weight.rows, b.ffn2.weight.cols);
        b.mha.w_o.weight = Tensor::zeros(b.mha.w_o.weight.rows, b.mha.w_o.weight.cols);

        let tape = Tape::new();
        let x_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let x = tape.input(Tensor::from_vec(x_data.clone(), 2, 8));
        let x_3d = Var3D::from_var(x, 1, 2, 8);
        let out = b.forward_3d(&tape, x_3d);
        let out_t = tape.value(out.as_var().idx());
        for (i, &v) in out_t.data.iter().enumerate() {
            assert!(
                (v - x_data[i]).abs() < 1e-3,
                "residual broken at {i}: input={} output={}",
                x_data[i],
                v
            );
        }
    }

    #[test]
    fn block_gradient_flows() {
        let mut rng = PcgEngine::new(0);
        let mut b = TransformerBlock::new(4, 2, 8, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 8], 2, 4));
        let x_3d = Var3D::from_var(x, 1, 2, 4);
        let out = b.forward_3d(&tape, x_3d);
        let loss = out.as_var().sum();
        loss.backward();
        let g = tape.grad(x.idx());
        let max_abs: f32 = g.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6);
    }
}
