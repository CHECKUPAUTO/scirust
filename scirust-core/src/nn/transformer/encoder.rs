// scirust-core/src/nn/transformer/encoder.rs
//
// TransformerEncoder — empile N TransformerBlock + LayerNorm final.
//
// L'encoder est l'architecture BERT-style (sans causal mask par defaut) :
// chaque token peut attendre tous les autres tokens de la sequence.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::block::TransformerBlock;
use crate::tensor::tensor3d::Var3D;
use std::collections::HashMap;

pub struct TransformerEncoder {
    pub blocks: Vec<TransformerBlock>,
    pub final_ln: LayerNorm,
    pub d_model: usize,
    pub name: String,
}

impl TransformerEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        causal: bool,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers
        {
            let block = TransformerBlock::new(d_model, n_heads, d_ff, causal, w_init, b_init, rng)
                .with_name(&format!("enc_block_{i}"));
            blocks.push(block);
        }
        let mut final_ln = LayerNorm::new(d_model, 1e-5, w_init, rng);
        final_ln.name = "enc_final_ln".into();
        Self {
            blocks,
            final_ln,
            d_model,
            name: format!("encoder_n{n_layers}_d{d_model}"),
        }
    }

    pub fn forward_3d<'t>(&mut self, tape: &'t Tape, x: Var3D<'t>) -> Var3D<'t> {
        let mut h = x;
        for block in self.blocks.iter_mut()
        {
            h = block.forward_3d(tape, h);
        }
        let ln_out = self.final_ln.forward(tape, h.as_var());
        Var3D::from_var(ln_out, h.batch, h.seq_len, h.d_model)
    }

    /// Incremental single-token forward for KV-cache decoding: chain each
    /// block's [`TransformerBlock::infer_step`] then the final LayerNorm.
    pub fn infer_step<'t>(&mut self, tape: &'t Tape, x_token: Var<'t>, pos: usize) -> Var<'t> {
        let mut h = x_token;
        for block in self.blocks.iter_mut()
        {
            h = block.infer_step(tape, h, pos);
        }
        self.final_ln.forward(tape, h)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        for b in &self.blocks
        {
            v.extend(b.parameter_indices());
        }
        v.extend(self.final_ln.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        for b in self.blocks.iter_mut()
        {
            b.sync(tape);
        }
        self.final_ln.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for b in &self.blocks
        {
            for (k, v) in b.state_dict()
            {
                map.insert(k, v);
            }
        }
        for (k, v) in self.final_ln.state_dict()
        {
            map.insert(k, v);
        }
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        self.final_ln.load_state_dict(sd)?;
        for b in self.blocks.iter_mut()
        {
            b.load_state_dict(sd)?;
        }
        Ok(())
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
        let mut enc =
            TransformerEncoder::new(2, 16, 4, 32, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 64], 4, 16));
        let x_3d = Var3D::from_var(x, 2, 2, 16);
        let out = enc.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 2, 16));
    }

    #[test]
    fn encoder_parameter_count_grows_with_layers() {
        let mut rng = PcgEngine::new(0);
        let mut enc1 =
            TransformerEncoder::new(1, 8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        let mut enc4 =
            TransformerEncoder::new(4, 8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        let tape1 = Tape::new();
        let x1 = tape1.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let _ = enc1.forward_3d(&tape1, Var3D::from_var(x1, 1, 2, 8));
        let n1 = enc1.parameter_indices().len();

        let tape4 = Tape::new();
        let x4 = tape4.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let _ = enc4.forward_3d(&tape4, Var3D::from_var(x4, 1, 2, 8));
        let n4 = enc4.parameter_indices().len();
        assert!(n4 > n1 * 3, "n1={n1} n4={n4}");
    }

    #[test]
    fn encoder_gradient_propagates_through_layers() {
        let mut rng = PcgEngine::new(0);
        let mut enc = TransformerEncoder::new(3, 4, 2, 8, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 16], 4, 4));
        let x_3d = Var3D::from_var(x, 2, 2, 4);
        let out = enc.forward_3d(&tape, x_3d);
        let loss = out.as_var().sum();
        loss.backward();
        let g = tape.grad(x.idx());
        let max_abs: f32 = g.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "grad nul apres 3 layers — autograd casse");
    }

    #[test]
    fn encoder_state_dict_round_trip() {
        let mut rng = PcgEngine::new(0);
        let enc1 = TransformerEncoder::new(2, 8, 2, 16, false, &KaimingNormal, &Zeros, &mut rng);
        let sd = enc1.state_dict();

        let mut rng2 = PcgEngine::new(99);
        let mut enc2 = TransformerEncoder::new(2, 8, 2, 16, false, &Zeros, &Zeros, &mut rng2);
        enc2.load_state_dict(&sd).unwrap();

        assert_eq!(
            enc2.blocks[0].mha.w_q.weight.data,
            enc1.blocks[0].mha.w_q.weight.data
        );
        assert_eq!(enc2.final_ln.gamma.data, enc1.final_ln.gamma.data);
    }
}
