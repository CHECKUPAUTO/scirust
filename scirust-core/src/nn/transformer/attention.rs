// scirust-core/src/nn/transformer/attention.rs
//
// MultiHeadAttention — implementation correcte avec les 3 primitives
// (Transpose2D, Concat, SliceCols) ajoutees en v11.

use crate::autodiff::reverse::{Tape, Tensor, Var, concat_rows};
use crate::nn::init::Initializer;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor3d::Var3D;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub num_kv_heads: usize,
    pub use_rope: bool,
    pub rope_theta: f32,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub causal: bool,
    pub name: String,
    pub kv_cache: RefCell<Option<(Tensor, Tensor)>>,
}

impl MultiHeadAttention {
    pub fn new<W: Initializer, B: Initializer>(
        d_model: usize,
        n_heads: usize,
        num_kv_heads: usize,
        causal: bool,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        assert!(
            d_model.is_multiple_of(n_heads),
            "MultiHeadAttention: d_model ({d_model}) doit etre divisible par n_heads ({n_heads})"
        );
        let d_head = d_model / n_heads;

        Self {
            d_model,
            n_heads,
            d_head,
            num_kv_heads: if num_kv_heads > 0
            {
                num_kv_heads
            }
            else
            {
                n_heads
            },
            use_rope: false,
            rope_theta: 10000.0,
            w_q: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_k: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_v: Linear::new(d_model, d_model, w_init, b_init, rng),
            w_o: Linear::new(d_model, d_model, w_init, b_init, rng),
            causal,
            name: format!("mha_d{d_model}_h{n_heads}"),
            kv_cache: RefCell::new(None),
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

    /// Cross-attention : Q provient de q_3d, K et V proviennent de kv_3d.
    pub fn forward_3d_cross<'t>(
        &mut self,
        tape: &'t Tape,
        q_3d: Var3D<'t>,
        kv_3d: Var3D<'t>,
    ) -> Var3D<'t> {
        let (batch, q_seq_len, d_model) = q_3d.shape();
        let (kv_batch, kv_seq_len, kv_d_model) = kv_3d.shape();
        assert_eq!(d_model, self.d_model);
        assert_eq!(kv_batch, batch);
        assert_eq!(kv_d_model, d_model);

        let q = self.w_q.forward(tape, q_3d.as_var());
        let k = self.w_k.forward(tape, kv_3d.as_var());
        let v = self.w_v.forward(tape, kv_3d.as_var());

        let attn_out = self.scaled_dot_attention_cross(tape, q, k, v, batch, q_seq_len, kv_seq_len);
        let output = self.w_o.forward(tape, attn_out);
        Var3D::from_var(output, batch, q_seq_len, self.d_model)
    }

    fn scaled_dot_attention<'t>(
        &self,
        tape: &'t Tape,
        q: Var<'t>,
        k: Var<'t>,
        v: Var<'t>,
        batch: usize,
        seq_len: usize,
    ) -> Var<'t> {
        let h_n = self.n_heads;
        let d_h = self.d_head;
        let scale = 1.0 / (d_h as f32).sqrt();

        let mut q_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut k_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut v_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        for h in 0..h_n
        {
            q_per_head.push(q.try_slice_cols(h * d_h, d_h).unwrap());
            k_per_head.push(k.try_slice_cols(h * d_h, d_h).unwrap());
            v_per_head.push(v.try_slice_cols(h * d_h, d_h).unwrap());
        }

        let mut head_outputs: Vec<Vec<Var<'t>>> =
            (0..h_n).map(|_| Vec::with_capacity(batch)).collect();
        for h in 0..h_n
        {
            let q_h = &q_per_head[h];
            let k_h = &k_per_head[h];
            let v_h = &v_per_head[h];
            for b in 0..batch
            {
                let q_hb = q_h.try_slice_rows(b * seq_len, seq_len).unwrap();
                let k_hb = k_h.try_slice_rows(b * seq_len, seq_len).unwrap();
                let v_hb = v_h.try_slice_rows(b * seq_len, seq_len).unwrap();

                let k_hb_t = k_hb.transpose_2d();
                let scores = q_hb.try_matmul(k_hb_t).unwrap();
                let scaled = scores.scale(scale);
                let pre_softmax = if self.causal
                {
                    scaled.causal_mask(seq_len)
                }
                else
                {
                    scaled
                };
                let attn = pre_softmax.try_softmax(1).unwrap();
                let out_hb = attn.try_matmul(v_hb).unwrap();
                head_outputs[h].push(out_hb);
            }
        }

        let mut head_full: Vec<Var<'t>> = Vec::with_capacity(h_n);
        for outputs in &head_outputs
        {
            head_full.push(concat_rows(tape, outputs));
        }

        combine_heads(tape, &head_full)
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_attention_cross<'t>(
        &self,
        tape: &'t Tape,
        q: Var<'t>,
        k: Var<'t>,
        v: Var<'t>,
        batch: usize,
        q_seq_len: usize,
        kv_seq_len: usize,
    ) -> Var<'t> {
        let h_n = self.n_heads;
        let d_h = self.d_head;
        let scale = 1.0 / (d_h as f32).sqrt();

        let mut q_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut k_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        let mut v_per_head: Vec<Var<'t>> = Vec::with_capacity(h_n);
        for h in 0..h_n
        {
            q_per_head.push(q.slice_cols(h * d_h, d_h));
            k_per_head.push(k.slice_cols(h * d_h, d_h));
            v_per_head.push(v.slice_cols(h * d_h, d_h));
        }

        let mut head_outputs: Vec<Vec<Var<'t>>> =
            (0..h_n).map(|_| Vec::with_capacity(batch)).collect();
        for h in 0..h_n
        {
            let q_h = q_per_head[h];
            let k_h = k_per_head[h];
            let v_h = v_per_head[h];
            for b in 0..batch
            {
                let q_hb = q_h.slice_rows(b * q_seq_len, q_seq_len);
                let k_hb = k_h.slice_rows(b * kv_seq_len, kv_seq_len);
                let v_hb = v_h.slice_rows(b * kv_seq_len, kv_seq_len);

                let k_hb_t = k_hb.transpose_2d();
                let scores = q_hb.matmul(k_hb_t);
                let scaled = scores.scale(scale);
                // Cross-attention n'est jamais causal
                let attn = scaled.softmax(1);
                let out_hb = attn.matmul(v_hb);
                head_outputs[h].push(out_hb);
            }
        }

        let mut head_full: Vec<Var<'t>> = Vec::with_capacity(h_n);
        for outputs in &head_outputs
        {
            head_full.push(concat_rows(tape, outputs));
        }

        combine_heads(tape, &head_full)
    }

    /// Inférence incrémentale avec KV-Cache (mode token unique).
    pub fn infer_step<'t>(&mut self, tape: &'t Tape, x_token: Var<'t>, _pos: usize) -> Var<'t> {
        let q = self.w_q.forward(tape, x_token);
        let k = self.w_k.forward(tape, x_token);
        let v = self.w_v.forward(tape, x_token);
        let (k_cached, v_cached) = {
            let mut cache = self.kv_cache.borrow_mut();
            match cache.as_mut()
            {
                Some((ck, cv)) =>
                {
                    let kd = tape.value(k.idx());
                    let vd = tape.value(v.idx());
                    let mut nk = ck.data.clone();
                    nk.extend(&kd.data);
                    let mut nv = cv.data.clone();
                    nv.extend(&vd.data);
                    *ck = Tensor::from_vec(nk, ck.rows + 1, ck.cols);
                    *cv = Tensor::from_vec(nv, cv.rows + 1, cv.cols);
                    (tape.input(ck.clone()), tape.input(cv.clone()))
                },
                None =>
                {
                    let kd = tape.value(k.idx());
                    let vd = tape.value(v.idx());
                    *cache = Some((kd, vd));
                    (k, v)
                },
            }
        };
        let h_n = self.n_heads;
        let d_h = self.d_head;
        let scale = 1.0 / (d_h as f32).sqrt();
        let mut heads = Vec::with_capacity(h_n);
        for h in 0..h_n
        {
            let qh = q.slice_cols(h * d_h, d_h);
            let kh = k_cached.slice_cols(h * d_h, d_h);
            let vh = v_cached.slice_cols(h * d_h, d_h);
            heads.push(
                qh.matmul(kh.transpose_2d())
                    .scale(scale)
                    .softmax(1)
                    .matmul(vh),
            );
        }
        let combined = combine_heads(tape, &heads);
        self.w_o.forward(tape, combined)
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
        self.w_q.sync(tape);
        self.w_k.sync(tape);
        self.w_v.sync(tape);
        self.w_o.sync(tape);
    }

    /// GQA: répète les têtes KV pour correspondre au nombre de têtes Q.
    /// Si num_kv_heads == num_heads, c'est un no-op (MHA standard).
    ///
    /// La concaténation au niveau `Var` (tape) est fournie par `concat_rows`
    /// (importée depuis `autodiff::reverse`). Pour le niveau `Tensor` brut,
    /// on concatène les données manuellement.
    #[allow(dead_code)]
    fn repeat_kv_heads(&self, x: Tensor, _seq_len: usize, _d_head: usize) -> Tensor {
        let repeat = self.n_heads / self.num_kv_heads;
        if repeat <= 1
        {
            return x;
        }
        let x_data = &x.data;
        let (rows, cols) = (x.rows, x.cols);
        let mut out = Vec::with_capacity(x_data.len() * repeat);
        for _ in 0..repeat
        {
            out.extend_from_slice(x_data);
        }
        Tensor::from_vec(out, rows * repeat, cols)
    }

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let p = &self.name;
        map.insert(format!("{p}.wq.weight"), self.w_q.weight.clone());
        map.insert(format!("{p}.wq.bias"), self.w_q.bias.clone());
        map.insert(format!("{p}.wk.weight"), self.w_k.weight.clone());
        map.insert(format!("{p}.wk.bias"), self.w_k.bias.clone());
        map.insert(format!("{p}.wv.weight"), self.w_v.weight.clone());
        map.insert(format!("{p}.wv.bias"), self.w_v.bias.clone());
        map.insert(format!("{p}.wo.weight"), self.w_o.weight.clone());
        map.insert(format!("{p}.wo.bias"), self.w_o.bias.clone());
        map
    }

    pub fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        let p = &self.name;
        let wq_w = sd
            .get(&format!("{p}.wq.weight"))
            .ok_or_else(|| format!("missing key: {p}.wq.weight"))?;
        let wq_b = sd
            .get(&format!("{p}.wq.bias"))
            .ok_or_else(|| format!("missing key: {p}.wq.bias"))?;
        let wk_w = sd
            .get(&format!("{p}.wk.weight"))
            .ok_or_else(|| format!("missing key: {p}.wk.weight"))?;
        let wk_b = sd
            .get(&format!("{p}.wk.bias"))
            .ok_or_else(|| format!("missing key: {p}.wk.bias"))?;
        let wv_w = sd
            .get(&format!("{p}.wv.weight"))
            .ok_or_else(|| format!("missing key: {p}.wv.weight"))?;
        let wv_b = sd
            .get(&format!("{p}.wv.bias"))
            .ok_or_else(|| format!("missing key: {p}.wv.bias"))?;
        let wo_w = sd
            .get(&format!("{p}.wo.weight"))
            .ok_or_else(|| format!("missing key: {p}.wo.weight"))?;
        let wo_b = sd
            .get(&format!("{p}.wo.bias"))
            .ok_or_else(|| format!("missing key: {p}.wo.bias"))?;

        self.w_q.weight = wq_w.clone();
        self.w_q.bias = wq_b.clone();
        self.w_k.weight = wk_w.clone();
        self.w_k.bias = wk_b.clone();
        self.w_v.weight = wv_w.clone();
        self.w_v.bias = wv_b.clone();
        self.w_o.weight = wo_w.clone();
        self.w_o.bias = wo_b.clone();
        Ok(())
    }
}

/// Recombine per-head outputs into a single `(rows, d_model)` tensor by placing
/// head `h`'s `(rows, d_head)` block into columns `[h*d_head, (h+1)*d_head)`.
///
/// This is a pure column-concatenation, expressed through the existing
/// `transpose_2d` / `concat_rows` primitives: each head is transposed to
/// `(d_head, rows)`, the heads are row-concatenated into `(d_model, rows)`, and
/// the result is transposed back to `(rows, d_model)`.
///
/// It replaces the previous per-head "pad matrix" approach, where each head was
/// multiplied by a `(d_head, d_model)` scatter matrix and the padded results
/// summed. That cost `O(rows · d_head · d_model)` FLOPs *per head* (an
/// `O(rows · d_model²)` matmul chain overall) purely to move data around. The
/// concat moves the same bytes in `O(rows · d_model)`.
///
/// The output is **bit-for-bit identical** to the pad-matrix version: the old
/// accumulator wrote each head's value into its target column exactly once and
/// added zeros everywhere else (`x + 0.0 == x` in IEEE-754), so the sum of the
/// scattered blocks equals the concatenation of those blocks.
fn combine_heads<'t>(tape: &'t Tape, heads: &[Var<'t>]) -> Var<'t> {
    let transposed: Vec<Var<'t>> = heads.iter().map(|h| h.transpose_2d()).collect();
    concat_rows(tape, &transposed).transpose_2d()
}

impl Clone for MultiHeadAttention {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_head,
            num_kv_heads: self.num_kv_heads,
            use_rope: self.use_rope,
            rope_theta: self.rope_theta,
            w_q: self.w_q.clone(),
            w_k: self.w_k.clone(),
            w_v: self.w_v.clone(),
            w_o: self.w_o.clone(),
            causal: self.causal,
            name: self.name.clone(),
            kv_cache: RefCell::new(None),
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
        let _ = MultiHeadAttention::new(64, 4, 0, false, &KaimingNormal, &Zeros, &mut rng);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn mha_panics_if_d_not_divisible() {
        let mut rng = PcgEngine::new(0);
        let _ = MultiHeadAttention::new(63, 4, 0, false, &KaimingNormal, &Zeros, &mut rng);
    }

    #[test]
    fn mha_forward_shape() {
        let mut rng = PcgEngine::new(0);
        let mut mha = MultiHeadAttention::new(8, 2, 0, false, &KaimingNormal, &Zeros, &mut rng);
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
        let mut mha = MultiHeadAttention::new(4, 2, 0, false, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x_var = tape.input(Tensor::from_vec(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            2,
            4,
        ));
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
        let mut mha = MultiHeadAttention::new(8, 2, 0, true, &KaimingNormal, &Zeros, &mut rng);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.1; 32], 4, 8));
        let x_3d = Var3D::from_var(x, 2, 2, 8);
        let out = mha.forward_3d(&tape, x_3d);
        assert_eq!(out.shape(), (2, 2, 8));
    }

    #[test]
    fn mha_state_dict_round_trip() {
        let mut rng = PcgEngine::new(0);
        let mha1 = MultiHeadAttention::new(8, 2, 0, false, &KaimingNormal, &Zeros, &mut rng);
        let sd = mha1.state_dict();
        assert_eq!(sd.len(), 8);

        let mut rng2 = PcgEngine::new(99);
        let mut mha2 = MultiHeadAttention::new(8, 2, 0, false, &Zeros, &Zeros, &mut rng2);
        mha2.load_state_dict(&sd).unwrap();

        assert_eq!(mha2.w_q.weight.data, mha1.w_q.weight.data);
        assert_eq!(mha2.w_o.bias.data, mha1.w_o.bias.data);
    }

    /// `combine_heads` must place head `h`'s block into columns
    /// `[h*d_head, (h+1)*d_head)` — the same layout the old pad-matrix
    /// scatter-and-sum produced. We build two heads of shape (rows, d_head)
    /// with distinct, easily-recognisable values and check every cell of the
    /// combined (rows, d_model) tensor, then confirm gradients flow back to
    /// both heads.
    #[test]
    fn combine_heads_places_columns_and_backprops() {
        let tape = Tape::new();
        let rows = 3;
        let d_head = 2;
        // Head 0: values 1,2 / 3,4 / 5,6 ; Head 1: 10,20 / 30,40 / 50,60.
        let h0 = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            rows,
            d_head,
        ));
        let h1 = tape.input(Tensor::from_vec(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            rows,
            d_head,
        ));
        let combined = combine_heads(&tape, &[h0, h1]);
        let val = tape.value(combined.idx());
        assert_eq!(val.shape(), (rows, 4));
        // Row-major (rows, d_model=4): [h0_col0, h0_col1, h1_col0, h1_col1].
        let expected = [
            1.0, 2.0, 10.0, 20.0, //
            3.0, 4.0, 30.0, 40.0, //
            5.0, 6.0, 50.0, 60.0,
        ];
        for (i, (&got, &exp)) in val.data.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(got, exp, "combined[{i}] = {got}, expected {exp}");
        }

        // Pure data movement ⇒ each input cell has gradient 1 under sum().
        let loss = combined.sum();
        loss.backward();
        for &g in &tape.grad(h0.idx()).data
        {
            assert_eq!(g, 1.0, "h0 grad should be 1");
        }
        for &g in &tape.grad(h1.idx()).data
        {
            assert_eq!(g, 1.0, "h1 grad should be 1");
        }
    }

    /// KV-cache correctness: feeding a sequence token-by-token through
    /// `infer_step` (incremental, O(n) decoding) must produce the same output
    /// for the final token as a full `forward_3d` over the whole sequence — the
    /// defining property of a correct KV-cache.
    #[test]
    fn kv_cache_matches_full_forward_last_position() {
        let d_model = 8;
        let n_heads = 2;
        let seq = 4;
        let mut rng = PcgEngine::new(123);
        let mut attn = MultiHeadAttention::new(
            d_model,
            n_heads,
            n_heads,
            true,
            &KaimingNormal,
            &Zeros,
            &mut rng,
        );

        let x: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();

        // Full forward → last position's output.
        let tape = Tape::new();
        let xv = tape.input(Tensor::from_vec(x.clone(), seq, d_model));
        let out3 = attn.forward_3d(&tape, Var3D::from_var(xv, 1, seq, d_model));
        let full = tape.value(out3.as_var().idx());
        let last_full = full.data[(seq - 1) * d_model..seq * d_model].to_vec();

        // Incremental decoding with the KV-cache.
        attn.kv_cache.replace(None);
        let tape2 = Tape::new();
        let mut last_inc = Vec::new();
        for t in 0..seq
        {
            let tok = tape2.input(Tensor::from_vec(
                x[t * d_model..(t + 1) * d_model].to_vec(),
                1,
                d_model,
            ));
            let o = attn.infer_step(&tape2, tok, t);
            last_inc = tape2.value(o.idx()).data.clone();
        }

        assert_eq!(last_inc.len(), last_full.len());
        let num: f32 = last_full
            .iter()
            .zip(&last_inc)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let den: f32 = last_full
            .iter()
            .map(|a| a * a)
            .sum::<f32>()
            .sqrt()
            .max(1e-30);
        assert!(
            num / den < 1e-4,
            "KV-cache mismatch: full={last_full:?} inc={last_inc:?} rel={}",
            num / den
        );
    }
}
