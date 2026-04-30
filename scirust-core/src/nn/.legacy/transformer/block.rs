// scirust-core/src/nn/transformer/block.rs
//
// TransformerBlock — un bloc encoder Pre-LN style.
//
// Architecture (Pre-LN, recommandée pour la stabilité d'entraînement) :
//
//   x' = x + MultiHeadAttention(LayerNorm(x))
//   y  = x' + FeedForward(LayerNorm(x'))
//
// Pourquoi Pre-LN au lieu du classique Post-LN :
//   - Plus stable au début de l'entraînement
//   - Évite la nécessité d'un warmup agressif
//   - C'est ce que GPT-2+ et la plupart des Transformers modernes utilisent
//
// Le FeedForward est un MLP 2-couches :
//   FFN(x) = Linear(d_ff → d_model) ∘ ReLU ∘ Linear(d_model → d_ff)
// Avec d_ff = 4 * d_model par convention.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::module::Module;
use crate::nn::transformer::attention::MultiHeadAttention;
use crate::nn::linear::Linear;
use crate::nn::layer_norm::LayerNorm;
use crate::tensor::tensor3d::Var3D;

pub struct TransformerBlock {
    pub d_model:  usize,
    pub n_heads:  usize,
    pub d_ff:     usize,
    pub ln1:      LayerNorm,    // pré-attention
    pub mha:      MultiHeadAttention,
    pub ln2:      LayerNorm,    // pré-FFN
    pub ffn1:     Linear,       // d_model → d_ff
    pub ffn2:     Linear,       // d_ff → d_model
    pub name:     String,
}

impl TransformerBlock {
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize, n_heads: usize, d_ff: usize, causal: bool,
        w_init: &W, b_init: &B, rng: &mut PcgEngine,
    ) -> Self {
        let mut s = Self {
            d_model, n_heads, d_ff,
            ln1:  LayerNorm::new(d_model, 1e-5, w_init, rng),
            mha:  MultiHeadAttention::new(d_model, n_heads, causal, w_init, b_init, rng),
            ln2:  LayerNorm::new(d_model, 1e-5, w_init, rng),
            ffn1: Linear::new(d_model, d_ff,    w_init, b_init, rng),
            ffn2: Linear::new(d_ff,    d_model, w_init, b_init, rng),
            name: format!("tb_d{d_model}_h{n_heads}_ff{d_ff}"),
        };
        s.ln1.name  = format!("{}.ln1",  s.name);
        s.ln2.name  = format!("{}.ln2",  s.name);
        s.mha       = s.mha.with_name(&format!("{}.mha", s.name));
        s.ffn1.name = format!("{}.ffn1", s.name);
        s.ffn2.name = format!("{}.ffn2", s.name);
        s
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self.ln1.name  = format!("{}.ln1",  self.name);
        self.ln2.name  = format!("{}.ln2",  self.name);
        self.mha = self.mha.with_name(&format!("{}.mha", self.name));
        self.ffn1.name = format!("{}.ffn1", self.name);
        self.ffn2.name = format!("{}.ffn2", self.name);
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
        let ln2_out = self.ln2.forward(tape, x1.clone());
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

    pub fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        v.extend(self.ln1.state_dict());
        v.extend(self.mha.state_dict());
        v.extend(self.ln2.state_dict());
        v.extend(self.ffn1.state_dict());
        v.extend(self.ffn2.state_dict());
        v
    }

    pub fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        self.ln1.load_state_dict(dict)
        + self.mha.load_state_dict(dict)
        + self.ln2.load_state_dict(dict)
        + self.ffn1.load_state_dict(dict)
        + self.ffn2.load_state_dict(dict)
    }
}

impl Clone for TransformerBlock {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model, n_heads: self.n_heads, d_ff: self.d_ff,
            ln1: self.ln1.clone(), mha: self.mha.clone(),
            ln2: self.ln2.clone(),
            ffn1: self.ffn1.clone(), ffn2: self.ffn2.clone(),
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
        let mut b = TransformerBlock::new(
            16, 4, 64, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        // (B=2, T=4, D=16)
        let x = tape.input(Tensor::from_vec(
            (0..128).map(|x| x as f32 * 0.01).collect(), 8, 16));
        let x_3d = Var3D::from_var(x, 2, 4, 16);
        let out = b.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 4, 16));
    }

    #[test]
    fn block_residual_path_works() {
        // Avec des poids quasi-zéros, l'output devrait être proche de l'input
        // (via les skip connections)
        let mut rng = PcgEngine::new(0);
        let mut b = TransformerBlock::new(
            8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        // Forcer ffn2 à zéro pour qu'il n'ajoute rien
        b.ffn2.weight = Tensor::zeros(b.ffn2.weight.rows, b.ffn2.weight.cols);
        // Et mha.w_o à zéro
        b.mha.w_o.weight = Tensor::zeros(b.mha.w_o.weight.rows, b.mha.w_o.weight.cols);

        let tape = Tape::new();
        let x_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let x = tape.input(Tensor::from_vec(x_data.clone(), 2, 8));
        let x_3d = Var3D::from_var(x, 1, 2, 8);
        let out = b.forward_3d(&tape, x_3d);
        let out_t = tape.value(out.as_var().idx());
        for (i, &v) in out_t.data.iter().enumerate() {
            assert!((v - x_data[i]).abs() < 1e-3,
                    "residual broken at {i}: input={} output={}", x_data[i], v);
        }
    }

    #[test]
    fn block_gradient_flows() {
        let mut rng = PcgEngine::new(0);
        let mut b = TransformerBlock::new(
            4, 2, 8, false, &KaimingNormal, &Zeros, &mut rng);
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
