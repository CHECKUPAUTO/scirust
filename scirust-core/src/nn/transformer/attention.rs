// scirust-core/src/nn/transformer/attention.rs
//
// MultiHeadAttention — implementation correcte avec les 3 primitives
// (Transpose2D, Concat, SliceCols) ajoutees en v11.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var, concat_rows};
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::module::Module;
use crate::nn::linear::Linear;
use crate::tensor::tensor3d::Var3D;

pub struct MultiHeadAttention {
    pub d_model:   usize,
    pub n_heads:   usize,
    pub d_head:    usize,
    pub w_q:       Linear,
    pub w_k:       Linear,
    pub w_v:       Linear,
    pub w_o:       Linear,
    pub causal:    bool,
    pub name:      String,
}

impl MultiHeadAttention {
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize, n_heads: usize, causal: bool,
        w_init: &W, b_init: &B, rng: &mut PcgEngine,
    ) -> Self {
        assert!(d_model.is_multiple_of(n_heads),
            "MultiHeadAttention: d_model ({d_model}) doit etre divisible par n_heads ({n_heads})");
        let d_head = d_model / n_heads;
        
        Self {
            d_model, n_heads, d_head,
            w_q: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_k: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_v: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_o: Linear::new(d_model, d_model, w_init, b_init, rng),
            causal,
            name: format!("mha_d{d_model}_h{n_heads}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn forward_3d<'t>(&mut self, tape: &'t Tape, x_3d: Var3D<'t>) -> Var3D<'t> {
        let (batch, seq_len, d_model) = x_3d.shape();
        assert_eq!(d_model, self.d_model);

        let q = self.w_q.forward(tape, x_3d.as_var());
        let k = self.w_k.forward(tape, x_3d.as_var());
        let v = self.w_v.forward(tape, x_3d.as_var());

        let attn_out = self.scaled_dot_attention(tape, q, k, v, batch, seq_len);
        let output = self.w_o.forward(tape, attn_out);
        Var3D::from_var(output, batch, seq_len, self.d_model)
    }

    fn scaled_dot_attention<'t>(
        &self, tape: &'t Tape,
        q: Var<'t>, k: Var<'t>, v: Var<'t>,
        batch: usize, seq_len: usize,
    ) -> Var<'t> {
        let h_n = self.n_heads;
        let d_h = self.d_head;
        let scale = 1.0 / (d_h as f32).sqrt();

        // Etape 1 : split par head via slice_cols
        let mut q_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut k_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut v_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        for h in 0..h_n {
            q_per_head.push(q.slice_cols(h * d_h, d_h));
            k_per_head.push(k.slice_cols(h * d_h, d_h));
            v_per_head.push(v.slice_cols(h * d_h, d_h));
        }

        // Etape 2 : pour chaque (h, b), calcule l'attention
        let mut head_outputs: Vec<Vec<Var<'t>>> =
            (0..h_n).map(|_| Vec::with_capacity(batch)).collect();
        for h in 0..h_n {
            let q_h = q_per_head[h];
            let k_h = k_per_head[h];
            let v_h = v_per_head[h];
            for b in 0..batch {
                let q_hb = q_h.slice_rows(b * seq_len, seq_len);
                let k_hb = k_h.slice_rows(b * seq_len, seq_len);
                let v_hb = v_h.slice_rows(b * seq_len, seq_len);

                let k_hb_t = k_hb.transpose_2d();
                let scores = q_hb.matmul(k_hb_t);
                let scaled = scores.scale(scale);
                let pre_softmax = if self.causal { scaled.causal_mask(seq_len) } else { scaled };
                let attn = pre_softmax.softmax(1);
                let out_hb = attn.matmul(v_hb);
                head_outputs[h].push(out_hb);
            }
        }

        // Etape 3a : concat batch-wise par head
        let mut head_full: Vec<Var<'t>> = Vec::with_capacity(h_n);
        #[allow(clippy::needless_range_loop)]
        for h in 0..h_n {
            head_full.push(concat_rows(tape, &head_outputs[h]));
        }

        // Etape 3b : padding par colonnes via matmul + somme
        let mut accumulator: Option<Var<'t>> = None;
        #[allow(clippy::needless_range_loop)]
        for h in 0..h_n {
            let pad = build_pad_matrix(tape, h, d_h, self.d_model);
            let padded = head_full[h].matmul(pad);
            accumulator = Some(match accumulator {
                None => padded,
                Some(acc) => acc.add(padded),
            });
        }
        accumulator.unwrap()
    }

    pub fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend(self.w_q.parameter_indices());
        v.extend(self.w_k.parameter_indices());
        v.extend(self.w_v.parameter_indices());
        v.extend(self.w_o.parameter_indices());
        v
    }

    pub fn sync(&mut self, tape: &Tape) {
        self.w_q.sync(tape); self.w_k.sync(tape);
        self.w_v.sync(tape); self.w_o.sync(tape);
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let p = &self.name;
        map.insert(format!("{p}.wq.weight"), self.w_q.weight.clone());
        map.insert(format!("{p}.wq.bias"),   self.w_q.bias.clone());
        map.insert(format!("{p}.wk.weight"), self.w_k.weight.clone());
        map.insert(format!("{p}.wk.bias"),   self.w_k.bias.clone());
        map.insert(format!("{p}.wv.weight"), self.w_v.weight.clone());
        map.insert(format!("{p}.wv.bias"),   self.w_v.bias.clone());
        map.insert(format!("{p}.wo.weight"), self.w_o.weight.clone());
        map.insert(format!("{p}.wo.bias"),   self.w_o.bias.clone());
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> std::result::Result<(), String> {
        let p = &self.name;
        let wq_w = sd.get(&format!("{p}.wq.weight"))
            .ok_or_else(|| format!("missing key: {p}.wq.weight"))?;
        let wq_b = sd.get(&format!("{p}.wq.bias"))
            .ok_or_else(|| format!("missing key: {p}.wq.bias"))?;
        let wk_w = sd.get(&format!("{p}.wk.weight"))
            .ok_or_else(|| format!("missing key: {p}.wk.weight"))?;
        let wk_b = sd.get(&format!("{p}.wk.bias"))
            .ok_or_else(|| format!("missing key: {p}.wk.bias"))?;
        let wv_w = sd.get(&format!("{p}.wv.weight"))
            .ok_or_else(|| format!("missing key: {p}.wv.weight"))?;
        let wv_b = sd.get(&format!("{p}.wv.bias"))
            .ok_or_else(|| format!("missing key: {p}.wv.bias"))?;
        let wo_w = sd.get(&format!("{p}.wo.weight"))
            .ok_or_else(|| format!("missing key: {p}.wo.weight"))?;
        let wo_b = sd.get(&format!("{p}.wo.bias"))
            .ok_or_else(|| format!("missing key: {p}.wo.bias"))?;

        self.w_q.weight = wq_w.clone(); self.w_q.bias = wq_b.clone();
        self.w_k.weight = wk_w.clone(); self.w_k.bias = wk_b.clone();
        self.w_v.weight = wv_w.clone(); self.w_v.bias = wv_b.clone();
        self.w_o.weight = wo_w.clone(); self.w_o.bias = wo_b.clone();
        Ok(())
    }
}

/// pad[i, j] = 1 si j == h*d_h + i, sinon 0. Shape (d_h, d_model).
fn build_pad_matrix<'t>(tape: &'t Tape, h: usize, d_h: usize, d_model: usize) -> Var<'t> {
    let mut data = vec![0.0f32; d_h * d_model];
    for i in 0..d_h {
        let j = h * d_h + i;
        data[i * d_model + j] = 1.0;
    }
    tape.input(Tensor::from_vec(data, d_h, d_model))
}

impl Clone for MultiHeadAttention {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model, n_heads: self.n_heads, d_head: self.d_head,
            w_q: self.w_q.clone(), w_k: self.w_k.clone(),
            w_v: self.w_v.clone(), w_o: self.w_o.clone(),
            causal: self.causal,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn mha_construction_validates_d_h() {
        let mut rng = PcgEngine::new(0);
        let _ = MultiHeadAttention::new(64, 4, false, &KaimingNormal, &Zeros, &mut rng);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn mha_panics_if_d_not_divisible() {
        let mut rng = PcgEngine::new(0);
        let _ = MultiHeadAttention::new(63, 4, false, &KaimingNormal, &Zeros, &mut rng);
    }

    #[test]
    fn mha_forward_shape() {
        let mut rng = PcgEngine::new(0);
        let mut mha = MultiHeadAttention::new(8, 2, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = Tensor::from_vec((0..48).map(|x| x as f32 * 0.01).collect(), 6, 8);
        let x_var = tape.input(x);
        let x_3d = Var3D::from_var(x_var, 2, 3, 8);
        let out = mha.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 3, 8));
    }

    #[test]
    fn mha_gradient_flows_to_inputs() {
        let mut rng = PcgEngine::new(42);
        let mut mha = MultiHeadAttention::new(4, 2, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x_var = tape.input(Tensor::from_vec(
            vec![0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8], 2, 4));
        let x_3d = Var3D::from_var(x_var, 1, 2, 4);
        let out = mha.forward_3d(&tape, x_3d);
        let loss = out.as_var().sum();
        loss.backward();
        let g = tape.grad(x_var.idx());
        let max_abs: f32 = g.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "gradient is zero — autograd broken");
    }

    #[test]
    fn mha_causal_mask_shape_preserved() {
        let mut rng = PcgEngine::new(0);
        let mut mha = MultiHeadAttention::new(8, 2, true, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 32], 4, 8));
        let x_3d = Var3D::from_var(x, 2, 2, 8);
        let out = mha.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 2, 8));
    }

    #[test]
    fn mha_state_dict_round_trip() {
        let mut rng = PcgEngine::new(0);
        let mha1 = MultiHeadAttention::new(8, 2, false, &KaimingNormal, &Zeros, &mut rng);
        let sd = mha1.state_dict();
        assert_eq!(sd.len(), 8);

        let mut rng2 = PcgEngine::new(99);
        let mut mha2 = MultiHeadAttention::new(8, 2, false, &Zeros, &Zeros, &mut rng2);
        mha2.load_state_dict(&sd).unwrap();

        assert_eq!(mha2.w_q.weight.data, mha1.w_q.weight.data);
        assert_eq!(mha2.w_o.bias.data,   mha1.w_o.bias.data);
    }
}
