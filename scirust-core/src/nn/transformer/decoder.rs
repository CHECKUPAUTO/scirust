// scirust-core/src/nn/transformer/decoder.rs
//
// TransformerDecoder — empile N TransformerDecoderBlock + LayerNorm final.
//
// Architecture par bloc (Pre-LN) :
//   x1 = x  + SelfAttention(LayerNorm(x))
//   x2 = x1 + CrossAttention(LayerNorm(x1), encoder_output)
//   x3 = x2 + FFN(LayerNorm(x2))

use crate::autodiff::reverse::{Tape, Tensor};
use crate::nn::init::Initializer;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use crate::nn::transformer::attention::MultiHeadAttention;
use crate::tensor::tensor3d::Var3D;
use std::collections::HashMap;

pub struct TransformerDecoderBlock {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub ln1: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub ln3: LayerNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
    pub name: String,
}

impl TransformerDecoderBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut s = Self {
            d_model,
            n_heads,
            d_ff,
            ln1: LayerNorm::new(d_model, 1e-5, w_init, rng),
            self_attn: MultiHeadAttention::new(d_model, n_heads, 0, true, w_init, b_init, rng),
            ln2: LayerNorm::new(d_model, 1e-5, w_init, rng),
            cross_attn: MultiHeadAttention::new(d_model, n_heads, 0, false, w_init, b_init, rng),
            ln3: LayerNorm::new(d_model, 1e-5, w_init, rng),
            ffn1: Linear::new(d_model, d_ff, w_init, b_init, rng),
            ffn2: Linear::new(d_ff, d_model, w_init, b_init, rng),
            name: format!("tdb_d{d_model}_h{n_heads}_ff{d_ff}"),
        };
        s.ln1.name = format!("{}.ln1", s.name);
        s.ln2.name = format!("{}.ln2", s.name);
        s.ln3.name = format!("{}.ln3", s.name);
        s.self_attn = s.self_attn.with_name(&format!("{}.self_attn", s.name));
        s.cross_attn = s.cross_attn.with_name(&format!("{}.cross_attn", s.name));
        s
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self.ln1.name = format!("{}.ln1", self.name);
        self.ln2.name = format!("{}.ln2", self.name);
        self.ln3.name = format!("{}.ln3", self.name);
        self.self_attn = self
            .self_attn
            .with_name(&format!("{}.self_attn", self.name));
        self.cross_attn = self
            .cross_attn
            .with_name(&format!("{}.cross_attn", self.name));
        self
    }

    /// Forward : x_3d = entrée décodeur (B, T_dec, D)
    /// encoder_out = sortie encodeur (B, T_enc, D)
    pub fn forward_3d<'t>(
        &mut self,
        tape: &'t Tape,
        x_3d: Var3D<'t>,
        encoder_out: Var3D<'t>,
    ) -> Var3D<'t> {
        let (batch, seq_len, d_model) = x_3d.shape();
        assert_eq!(d_model, self.d_model);

        // 1) Self-attention (causal)
        let ln1_out = self.ln1.forward(tape, x_3d.as_var());
        let ln1_3d = Var3D::from_var(ln1_out, batch, seq_len, d_model);
        let self_attn_out = self.self_attn.forward_3d(tape, ln1_3d);
        let x1 = x_3d.as_var().try_add(self_attn_out.as_var()).unwrap();

        // 2) Cross-attention
        let ln2_out = self.ln2.forward(tape, x1);
        let ln2_3d = Var3D::from_var(ln2_out, batch, seq_len, d_model);
        let cross_attn_out = self.cross_attn.forward_3d_cross(tape, ln2_3d, encoder_out);
        let x2 = x1.try_add(cross_attn_out.as_var()).unwrap();

        // 3) FFN
        let ln3_out = self.ln3.forward(tape, x2);
        let h_ff = self.ffn1.forward(tape, ln3_out).relu();
        let ffn_out = self.ffn2.forward(tape, h_ff);
        let x3 = x2.try_add(ffn_out).unwrap();

        Var3D::from_var(x3, batch, seq_len, d_model)
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.ln1.parameter_indices());
        v.extend(self.self_attn.parameter_indices());
        v.extend(self.ln2.parameter_indices());
        v.extend(self.cross_attn.parameter_indices());
        v.extend(self.ln3.parameter_indices());
        v.extend(self.ffn1.parameter_indices());
        v.extend(self.ffn2.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.ln1.sync(tape);
        self.self_attn.sync(tape);
        self.ln2.sync(tape);
        self.cross_attn.sync(tape);
        self.ln3.sync(tape);
        self.ffn1.sync(tape);
        self.ffn2.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (k, v) in self.ln1.state_dict()
        {
            map.insert(k, v);
        }
        for (k, v) in self.self_attn.state_dict()
        {
            map.insert(k, v);
        }
        for (k, v) in self.ln2.state_dict()
        {
            map.insert(k, v);
        }
        for (k, v) in self.cross_attn.state_dict()
        {
            map.insert(k, v);
        }
        for (k, v) in self.ln3.state_dict()
        {
            map.insert(k, v);
        }
        map.insert(
            format!("{}.ffn1.weight", self.name),
            self.ffn1.weight.clone(),
        );
        map.insert(format!("{}.ffn1.bias", self.name), self.ffn1.bias.clone());
        map.insert(
            format!("{}.ffn2.weight", self.name),
            self.ffn2.weight.clone(),
        );
        map.insert(format!("{}.ffn2.bias", self.name), self.ffn2.bias.clone());
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        self.ln1.load_state_dict(sd)?;
        self.self_attn.load_state_dict(sd)?;
        self.ln2.load_state_dict(sd)?;
        self.cross_attn.load_state_dict(sd)?;
        self.ln3.load_state_dict(sd)?;

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

impl Clone for TransformerDecoderBlock {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
            ln1: self.ln1.clone(),
            self_attn: self.self_attn.clone(),
            ln2: self.ln2.clone(),
            cross_attn: self.cross_attn.clone(),
            ln3: self.ln3.clone(),
            ffn1: self.ffn1.clone(),
            ffn2: self.ffn2.clone(),
            name: self.name.clone(),
        }
    }
}

// ================================================================== //

pub struct TransformerDecoder {
    pub blocks: Vec<TransformerDecoderBlock>,
    pub final_ln: LayerNorm,
    pub d_model: usize,
    pub name: String,
}

impl TransformerDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new<W: Initializer, B: Initializer>(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers
        {
            let block = TransformerDecoderBlock::new(d_model, n_heads, d_ff, w_init, b_init, rng)
                .with_name(&format!("dec_block_{i}"));
            blocks.push(block);
        }
        let mut final_ln = LayerNorm::new(d_model, 1e-5, w_init, rng);
        final_ln.name = "dec_final_ln".into();
        Self {
            blocks,
            final_ln,
            d_model,
            name: format!("decoder_n{n_layers}_d{d_model}"),
        }
    }

    pub fn forward_3d<'t>(
        &mut self,
        tape: &'t Tape,
        x: Var3D<'t>,
        encoder_out: Var3D<'t>,
    ) -> Var3D<'t> {
        let mut h = x;
        for block in self.blocks.iter_mut()
        {
            h = block.forward_3d(tape, h, encoder_out);
        }
        let ln_out = self.final_ln.forward(tape, h.as_var());
        Var3D::from_var(ln_out, h.batch, h.seq_len, h.d_model)
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

impl Clone for TransformerDecoder {
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
    fn decoder_block_forward_shape_preserved() {
        let mut rng = PcgEngine::new(0);
        let mut block = TransformerDecoderBlock::new(8, 2, 16, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 24], 3, 8));
        let x_3d = Var3D::from_var(x, 1, 3, 8);
        let enc = tape.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let enc_3d = Var3D::from_var(enc, 1, 2, 8);
        let out = block.forward_3d(&tape, x_3d, enc_3d);
        assert_eq!(out.shape(), (1, 3, 8));
    }

    #[test]
    fn decoder_forward_shape_preserved() {
        let mut rng = PcgEngine::new(0);
        let mut dec = TransformerDecoder::new(2, 8, 2, 16, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 24], 3, 8));
        let x_3d = Var3D::from_var(x, 1, 3, 8);
        let enc = tape.input(Tensor::from_vec(vec![0.1; 16], 2, 8));
        let enc_3d = Var3D::from_var(enc, 1, 2, 8);
        let out = dec.forward_3d(&tape, x_3d, enc_3d);
        assert_eq!(out.shape(), (1, 3, 8));
    }

    #[test]
    fn decoder_gradient_propagates() {
        let mut rng = PcgEngine::new(0);
        let mut dec = TransformerDecoder::new(1, 4, 2, 8, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 8], 2, 4));
        let x_3d = Var3D::from_var(x, 1, 2, 4);
        let enc = tape.input(Tensor::from_vec(vec![0.1; 8], 2, 4));
        let enc_3d = Var3D::from_var(enc, 1, 2, 4);
        let out = dec.forward_3d(&tape, x_3d, enc_3d);
        let loss = out.as_var().sum();
        loss.backward();
        let g = tape.grad(x.idx());
        let max_abs: f32 = g.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "gradient nul apres decoder");
    }

    #[test]
    fn decoder_state_dict_round_trip() {
        let mut rng = PcgEngine::new(0);
        let dec1 = TransformerDecoder::new(2, 8, 2, 16, &KaimingNormal, &Zeros, &mut rng);
        let sd = dec1.state_dict();

        let mut rng2 = PcgEngine::new(99);
        let mut dec2 = TransformerDecoder::new(2, 8, 2, 16, &Zeros, &Zeros, &mut rng2);
        dec2.load_state_dict(&sd).unwrap();

        assert_eq!(
            dec2.blocks[0].self_attn.w_q.weight.data,
            dec1.blocks[0].self_attn.w_q.weight.data
        );
        assert_eq!(dec2.final_ln.gamma.data, dec1.final_ln.gamma.data);
    }
}
