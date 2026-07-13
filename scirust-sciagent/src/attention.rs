use scirust_core::autodiff::reverse::{Tape, Tensor, Var, concat_rows};
use scirust_core::nn::init::{Initializer, Zeros};
use scirust_core::nn::linear::Linear;
use scirust_core::nn::module::Module;
use scirust_core::nn::rng::PcgEngine;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct GQAAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub rope_theta: f32,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub name: String,
    pub kv_cache: RefCell<Option<(Tensor, Tensor)>>,
}

impl GQAAttention {
    pub fn new<I: Initializer>(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_theta: f32,
        init: &I,
        rng: &mut PcgEngine,
    ) -> Self {
        assert!(d_model.is_multiple_of(n_heads));
        let d_head = d_model / n_heads;
        let kv_dim = n_kv_heads * d_head;
        let z = Zeros;
        Self {
            d_model,
            n_heads,
            n_kv_heads,
            d_head,
            rope_theta,
            w_q: Linear::new(d_model, d_model, init, &z, rng),
            w_k: Linear::new(d_model, kv_dim, init, &z, rng),
            w_v: Linear::new(d_model, kv_dim, init, &z, rng),
            w_o: Linear::new(d_model, d_model, init, &z, rng),
            name: format!("gqa_d{d_model}_h{n_heads}_kv{n_kv_heads}"),
            kv_cache: RefCell::new(None),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    /// RoPE as **on-tape** operations, so gradients flow back through the
    /// rotation into `w_q`/`w_k`. The previous implementation extracted the
    /// projected values (`tape.value`), rotated them outside the tape, and
    /// re-registered the result as a fresh `tape.input` — a detach: the
    /// attention-path gradient stopped there and `w_q`/`w_k` stayed frozen at
    /// their random init for the whole training run.
    ///
    /// Rotation identity used: with per-pair constants `C`/`S` (cos/sin
    /// broadcast to both lanes of each pair) and a constant pair-swap matrix
    /// `W` such that `(x·W)[2j] = −x[2j+1]`, `(x·W)[2j+1] = x[2j]`:
    /// `rope(x) = x⊙C + (x·W)⊙S` — exactly the interleaved rotation of
    /// [`rope_apply`], expressed with differentiable ops only.
    ///
    /// Positions restart at `offset` every `seq_len` rows (`pos = r % seq_len`),
    /// which also fixes batched training: the old path rotated the flattened
    /// batch with absolute row indices, so every sequence after the first got
    /// shifted positions.
    fn rope_on_tape<'t>(
        tape: &'t Tape,
        x: Var<'t>,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> Var<'t> {
        let (rows, dim) = x.shape();
        let half = dim / 2;
        let mut c = vec![0.0f32; rows * dim];
        let mut s = vec![0.0f32; rows * dim];
        for r in 0..rows
        {
            let pos = ((r % seq_len) + offset) as f32;
            for j in 0..half
            {
                let freq = theta.powf(-2.0 * j as f32 / dim as f32);
                let a = pos * freq;
                c[r * dim + 2 * j] = a.cos();
                c[r * dim + 2 * j + 1] = a.cos();
                s[r * dim + 2 * j] = a.sin();
                s[r * dim + 2 * j + 1] = a.sin();
            }
        }
        // Pair-swap-and-negate: column 2j reads −x[2j+1], column 2j+1 reads x[2j].
        let mut w = vec![0.0f32; dim * dim];
        for j in 0..half
        {
            w[(2 * j + 1) * dim + 2 * j] = -1.0;
            w[(2 * j) * dim + (2 * j + 1)] = 1.0;
        }
        let c_v = tape.input(Tensor::from_vec(c, rows, dim));
        let s_v = tape.input(Tensor::from_vec(s, rows, dim));
        let w_v = tape.input(Tensor::from_vec(w, dim, dim));
        x.hadamard(c_v).add(x.matmul(w_v).hadamard(s_v))
    }

    fn rope_apply(t: &Tensor, offset: usize, theta: f32) -> Tensor {
        let rows = t.rows;
        let dim = t.cols;
        let half = dim / 2;
        let mut cos = vec![0.0f32; rows * half];
        let mut sin = vec![0.0f32; rows * half];
        for p in 0..rows
        {
            let pos = (p + offset) as f32;
            for j in 0..half
            {
                let freq = theta.powf(-2.0 * j as f32 / dim as f32);
                let a = pos * freq;
                cos[p * half + j] = a.cos();
                sin[p * half + j] = a.sin();
            }
        }
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows
        {
            for j in 0..half
            {
                let e = t.data[r * dim + 2 * j];
                let o = t.data[r * dim + 2 * j + 1];
                let c = cos[r * half + j];
                let s = sin[r * half + j];
                out[r * dim + 2 * j] = e * c - o * s;
                out[r * dim + 2 * j + 1] = e * s + o * c;
            }
        }
        Tensor::from_vec(out, rows, dim)
    }
}

impl Clone for GQAAttention {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            d_head: self.d_head,
            rope_theta: self.rope_theta,
            w_q: self.w_q.clone(),
            w_k: self.w_k.clone(),
            w_v: self.w_v.clone(),
            w_o: self.w_o.clone(),
            name: self.name.clone(),
            kv_cache: RefCell::new(None),
        }
    }
}

impl GQAAttention {
    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>, seq_len: usize) -> Var<'t> {
        let batch = x.shape().0 / seq_len;
        let h = self.n_heads;
        let dh = self.d_head;
        let kh = self.n_kv_heads;
        let repeat = h / kh;
        let scale = 1.0 / (dh as f32).sqrt();

        let q = self.w_q.forward(tape, x);
        let k = self.w_k.forward(tape, x);
        let v = self.w_v.forward(tape, x);

        // On-tape RoPE (per-block positions): gradients flow through the
        // rotation back into w_q / w_k — see `rope_on_tape`.
        let qr = Self::rope_on_tape(tape, q, seq_len, 0, self.rope_theta);
        let kr = Self::rope_on_tape(tape, k, seq_len, 0, self.rope_theta);

        let mut head_out = Vec::with_capacity(h);
        for head in 0..h
        {
            let kv_idx = head / repeat;
            let qs = qr.slice_cols(head * dh, dh);
            let ks = kr.slice_cols(kv_idx * dh, dh);
            let vs = v.slice_cols(kv_idx * dh, dh);
            let mut pb = Vec::with_capacity(batch);
            for b in 0..batch
            {
                let qb = qs.slice_rows(b * seq_len, seq_len);
                let kb = ks.slice_rows(b * seq_len, seq_len);
                let vb = vs.slice_rows(b * seq_len, seq_len);
                let o = qb
                    .matmul(kb.transpose_2d())
                    .scale(scale)
                    .causal_mask(seq_len)
                    .softmax(1)
                    .matmul(vb);
                pb.push(o);
            }
            // Differentiable concat (Op::Concat), NOT a value copy: the old
            // hand-rolled concat re-registered the per-batch outputs as fresh
            // inputs, detaching the whole attention product from w_q/w_k/w_v.
            let cat = concat_rows(tape, &pb);
            head_out.push(cat.matmul(build_pad(tape, head, dh, self.d_model)));
        }

        let mut acc = head_out[0];
        for &ho in head_out.iter().skip(1)
        {
            acc = acc.add(ho);
        }
        self.w_o.forward(tape, acc)
    }

    pub fn infer_step<'t>(&mut self, tape: &'t Tape, x_token: Var<'t>, pos: usize) -> Var<'t> {
        let h = self.n_heads;
        let dh = self.d_head;
        let kh = self.n_kv_heads;
        let repeat = h / kh;
        let scale = 1.0 / (dh as f32).sqrt();

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

        let qv = tape.value(q.idx());
        let kv = tape.value(k_cached.idx());

        let qr = tape.input(Self::rope_apply(&qv, pos, self.rope_theta));
        let kr = tape.input(Self::rope_apply(&kv, 0, self.rope_theta));

        let mut head_out = Vec::with_capacity(h);
        for head in 0..h
        {
            let kv_idx = head / repeat;
            let qh = qr.slice_cols(head * dh, dh);
            let kh = kr.slice_cols(kv_idx * dh, dh);
            let vh = v_cached.slice_cols(kv_idx * dh, dh);
            let o = qh
                .matmul(kh.transpose_2d())
                .scale(scale)
                .softmax(1)
                .matmul(vh);
            head_out.push(o.matmul(build_pad(tape, head, dh, self.d_model)));
        }

        let mut acc = head_out[0];
        for &ho in head_out.iter().skip(1)
        {
            acc = acc.add(ho);
        }
        self.w_o.forward(tape, acc)
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

    pub fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        let p = &self.name;
        map.insert(format!("{p}.wq.weight"), self.w_q.weight.clone());
        map.insert(format!("{p}.wk.weight"), self.w_k.weight.clone());
        map.insert(format!("{p}.wv.weight"), self.w_v.weight.clone());
        map.insert(format!("{p}.wo.weight"), self.w_o.weight.clone());
        map
    }

    pub fn load_state_dict(
        &mut self,
        sd: &HashMap<String, Tensor>,
    ) -> scirust_core::error::Result<()> {
        let p = &self.name;
        self.w_q.weight = sd
            .get(&format!("{p}.wq.weight"))
            .ok_or_else(|| format!("missing {p}.wq.weight"))?
            .clone();
        self.w_k.weight = sd
            .get(&format!("{p}.wk.weight"))
            .ok_or_else(|| format!("missing {p}.wk.weight"))?
            .clone();
        self.w_v.weight = sd
            .get(&format!("{p}.wv.weight"))
            .ok_or_else(|| format!("missing {p}.wv.weight"))?
            .clone();
        self.w_o.weight = sd
            .get(&format!("{p}.wo.weight"))
            .ok_or_else(|| format!("missing {p}.wo.weight"))?
            .clone();
        Ok(())
    }
}

fn build_pad<'t>(tape: &'t Tape, h: usize, dh: usize, dm: usize) -> Var<'t> {
    let mut data = vec![0.0f32; dh * dm];
    for i in 0..dh
    {
        data[i * dm + h * dh + i] = 1.0;
    }
    tape.input(Tensor::from_vec(data, dh, dm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::init::KaimingNormal;

    fn tiny_attn() -> GQAAttention {
        let init = KaimingNormal;
        let mut rng = PcgEngine::new(7);
        GQAAttention::new(16, 4, 2, 10000.0, &init, &mut rng)
    }

    #[test]
    fn rope_on_tape_matches_the_reference_rotation() {
        // The differentiable formulation x⊙C + (x·W)⊙S must equal the scalar
        // reference `rope_apply` exactly (same interleaved pairs, same freqs).
        let tape = Tape::new();
        let rows = 6usize;
        let dim = 8usize;
        let data: Vec<f32> = (0..rows * dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let x = tape.input(Tensor::from_vec(data.clone(), rows, dim));
        let on_tape = GQAAttention::rope_on_tape(&tape, x, rows, 3, 10000.0);
        let got = tape.value(on_tape.idx());
        let want = GQAAttention::rope_apply(&Tensor::from_vec(data, rows, dim), 3, 10000.0);
        for (g, w) in got.data.iter().zip(&want.data)
        {
            assert!((g - w).abs() < 1e-5, "rope mismatch: {g} vs {w}");
        }
    }

    #[test]
    fn attention_gradients_reach_q_k_v() {
        // Regression for the detach bugs: RoPE-as-tape.input and the value-copy
        // concat cut the gradient path, so w_q/w_k/w_v trained to nothing (the
        // attention stayed at its random init for the whole 197M-token run).
        let mut attn = tiny_attn();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            (0..4 * 16).map(|i| (i as f32 * 0.13).cos()).collect(),
            4,
            16,
        ));
        let out = attn.forward(&tape, x, 4);
        let loss = out.hadamard(out).sum();
        tape.backward(loss.idx());
        for (name, lin) in [("w_q", &attn.w_q), ("w_k", &attn.w_k), ("w_v", &attn.w_v)]
        {
            let idx = lin.parameter_indices()[0];
            let g = tape.grad(idx);
            let norm: f32 = g.data.iter().map(|v| v.abs()).sum();
            assert!(norm > 1e-9, "{name} must receive gradient, got norm {norm}");
        }
    }

    #[test]
    fn batched_rows_get_per_sequence_positions() {
        // The same sequence duplicated as two batch rows must produce identical
        // outputs for both blocks: positions restart per sequence. The old code
        // gave block 1 positions seq_len..2*seq_len, so the blocks differed.
        let mut attn = tiny_attn();
        let tape = Tape::new();
        let one: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.11).sin()).collect();
        let mut two = one.clone();
        two.extend(one.iter().copied());
        let x = tape.input(Tensor::from_vec(two, 8, 16));
        let out = attn.forward(&tape, x, 4);
        let t = tape.value(out.idx());
        let (a, b) = t.data.split_at(4 * 16);
        for (va, vb) in a.iter().zip(b)
        {
            assert!(
                (va - vb).abs() < 1e-5,
                "duplicated batch rows must match: {va} vs {vb}"
            );
        }
    }
}
