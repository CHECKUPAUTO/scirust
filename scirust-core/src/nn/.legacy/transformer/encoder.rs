// scirust-core/src/nn/transformer/encoder.rs
//
// TransformerEncoder — empile N TransformerBlock + LayerNorm final.
//
// L'encoder est l'architecture BERT-style (sans causal mask par défaut) :
// chaque token peut attendre tous les autres tokens de la séquence.
//
// Structure :
//
//   x → [TransformerBlock × N] → LayerNorm → output
//
// Le LayerNorm final est une convention Pre-LN : il "rationalise" la sortie
// avant qu'elle soit consommée par les couches downstream (typiquement un
// classifier head).

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::module::Module;
use crate::nn::transformer::block::TransformerBlock;
use crate::tensor::tensor3d::Var3D;

pub struct TransformerEncoder {
    pub blocks:   Vec<TransformerBlock>,
    pub final_ln: LayerNorm,
    pub d_model:  usize,
    pub name:     String,
}

impl TransformerEncoder {
    pub fn new<W: Initializer, B: Initializer>(
        n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize,
        causal: bool, w_init: &W, b_init: &B, rng: &mut PcgEngine,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let block = TransformerBlock::new(
                d_model, n_heads, d_ff, causal, w_init, b_init, rng,
            ).with_name(&format!("enc_block_{i}"));
            blocks.push(block);
        }
        let mut final_ln = LayerNorm::new(d_model, 1e-5, w_init, rng);
        final_ln.name = "enc_final_ln".into();
        Self {
            blocks, final_ln, d_model,
            name: format!("encoder_n{n_layers}_d{d_model}"),
        }
    }

    pub fn forward_3d<'t>(&mut self, tape: &'t Tape, x: Var3D<'t>) -> Var3D<'t> {
        let mut h = x;
        for block in self.blocks.iter_mut() {
            h = block.forward_3d(tape, h);
        }
        // LayerNorm final sur la vue 2D
        let ln_out = self.final_ln.forward(tape, h.as_var());
        Var3D::from_var(ln_out, h.batch, h.seq_len, h.d_model)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for b in &self.blocks { v.extend(b.parameter_indices()); }
        v.extend(self.final_ln.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        for b in self.blocks.iter_mut() { b.sync(tape); }
        self.final_ln.sync(tape);
    }

    pub fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = Vec::new();
        for b in &self.blocks { v.extend(b.state_dict()); }
        v.extend(self.final_ln.state_dict());
        v
    }

    pub fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut loaded = 0;
        for b in self.blocks.iter_mut() { loaded += b.load_state_dict(dict); }
        loaded += self.final_ln.load_state_dict(dict);
        loaded
    }
}

impl Clone for TransformerEncoder {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            final_ln: self.final_ln.clone(),
            d_model: self.d_model,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn encoder_2_layers_shape() {
        let mut rng = PcgEngine::new(0);
        let mut enc = TransformerEncoder::new(
            2, 16, 4, 32, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 64], 4, 16));
        let x_3d = Var3D::from_var(x, 2, 2, 16);
        let out = enc.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 2, 16));
    }

    #[test]
    fn encoder_parameter_count_grows_with_layers() {
        let mut rng = PcgEngine::new(0);
        let mut enc1 = TransformerEncoder::new(
            1, 8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        let mut enc4 = TransformerEncoder::new(
            4, 8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        // Faire un forward pour matérialiser les parameter_indices
        let tape1 = Tape::new();
        let x1 = tape1.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let _ = enc1.forward_3d(&tape1, Var3D::from_var(x1, 1, 2, 8));
        let n1 = enc1.parameter_indices().len();

        let tape4 = Tape::new();
        let x4 = tape4.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let _ = enc4.forward_3d(&tape4, Var3D::from_var(x4, 1, 2, 8));
        let n4 = enc4.parameter_indices().len();
        // 4× plus de blocks → ~4× plus de paramètres (modulo le final_ln partagé)
        assert!(n4 > n1 * 3, "n1={n1} n4={n4}");
    }

    #[test]
    fn encoder_gradient_propagates_through_layers() {
        let mut rng = PcgEngine::new(0);
        let mut enc = TransformerEncoder::new(
            3, 4, 2, 8, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 16], 4, 4));
        let x_3d = Var3D::from_var(x, 2, 2, 4);
        let out = enc.forward_3d(&tape, x_3d);
        let loss = out.as_var().sum();
        loss.backward();
        let g = tape.grad(x.idx());
        let max_abs: f32 = g.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "grad nul après 3 layers — autograd cassé");
    }
}
