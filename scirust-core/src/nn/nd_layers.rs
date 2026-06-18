//! Reusable neural-network layers over the **N-D autograd tape**
//! ([`crate::autodiff::nd`]).
//!
//! This is the step from "the N-D tape can *express* a layer" to "here is the
//! layer": [`NdLinear`] holds its own parameters, runs a forward on an
//! [`NdTape`], and applies an SGD update from the gradients — so a real N-D
//! network can be built and trained. Parameter init is seeded ([`PcgEngine`]),
//! preserving determinism.

use crate::autodiff::nd::{NdTape, NdVar};
use crate::nn::nd_optim::NdParam;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;

/// A dense layer `y = x · W + b` acting on the **last axis** of an N-D input:
/// `(…, in) → (…, out)`. The leading axes (batch, heads, sequence, …) are
/// flattened for the matmul and restored afterwards.
pub struct NdLinear {
    weight: TensorND, // (in, out)
    bias: TensorND,   // (1, out)
    in_features: usize,
    out_features: usize,
    w_idx: Option<usize>,
    b_idx: Option<usize>,
}

impl NdLinear {
    /// New layer with seeded init: `W ~ U(-s, s)` with `s = 1/√in`, `b = 0`.
    pub fn new(in_features: usize, out_features: usize, rng: &mut PcgEngine) -> Self {
        let scale = (1.0 / in_features as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.float_signed() * scale)
            .collect();
        Self {
            weight: TensorND::new(w, vec![in_features, out_features]),
            bias: TensorND::zeros(&[1, out_features]),
            in_features,
            out_features,
            w_idx: None,
            b_idx: None,
        }
    }

    /// Forward pass. Records the parameter node indices so [`Self::sgd_step`]
    /// can read their gradients after `backward`.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let xs = x.shape();
        let in_f = *xs.last().expect("NdLinear: input has no axes");
        assert_eq!(in_f, self.in_features, "NdLinear: input feature mismatch");
        let m: usize = xs[..xs.len() - 1].iter().product();

        let w = tape.input(self.weight.clone());
        let b = tape.input(self.bias.clone());
        self.w_idx = Some(w.idx());
        self.b_idx = Some(b.idx());

        let y2 = x.reshape(&[m, in_f]).matmul(w).add(b); // (m, out)
        let mut out_shape = xs[..xs.len() - 1].to_vec();
        out_shape.push(self.out_features);
        y2.reshape(&out_shape)
    }

    /// Apply `param -= lr · grad` for the weight and bias using the gradients
    /// returned by [`NdTape::backward`] (must follow a `forward` on that tape).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.w_idx
        {
            for (p, &g) in self.weight.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * g;
            }
        }
        if let Some(i) = self.b_idx
        {
            for (p, &g) in self.bias.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * g;
            }
        }
    }

    /// The weight tensor `(in, out)`.
    pub fn weight(&self) -> &TensorND {
        &self.weight
    }

    /// The bias tensor `(1, out)`.
    pub fn bias(&self) -> &TensorND {
        &self.bias
    }

    /// The trainable parameters (weight, bias) paired with their gradient
    /// indices, for an optimizer ([`NdAdam`](crate::nn::nd_optim::NdAdam)).
    /// Call after a `forward` on the tape being differentiated.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        if let Some(i) = self.w_idx
        {
            params.push(NdParam {
                value: &mut self.weight,
                grad_idx: i,
            });
        }
        if let Some(i) = self.b_idx
        {
            params.push(NdParam {
                value: &mut self.bias,
                grad_idx: i,
            });
        }
        params
    }
}

/// An embedding table `(vocab, dim)`: maps integer ids to rows via the N-D
/// tape's [`gather`](NdVar::gather). Used for both token and (learned)
/// positional embeddings. Seeded init, SGD-updatable.
pub struct NdEmbedding {
    table: TensorND, // (vocab, dim)
    idx: Option<usize>,
}

impl NdEmbedding {
    /// New table with seeded init `U(-s, s)`, `s = 1/√dim`.
    pub fn new(vocab: usize, dim: usize, rng: &mut PcgEngine) -> Self {
        let scale = (1.0 / dim as f32).sqrt();
        let data: Vec<f32> = (0..vocab * dim)
            .map(|_| rng.float_signed() * scale)
            .collect();
        Self {
            table: TensorND::new(data, vec![vocab, dim]),
            idx: None,
        }
    }

    /// Look up `ids` → `(ids.len(), dim)`, recording the table node so
    /// [`Self::sgd_step`] can read its gradient.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, ids: &[usize]) -> NdVar<'t> {
        let w = tape.input(self.table.clone());
        self.idx = Some(w.idx());
        w.gather(ids)
    }

    /// SGD-update the table from a `backward` result.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.idx
        {
            for (p, &g) in self.table.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * g;
            }
        }
    }

    /// The embedding table `(vocab, dim)`.
    pub fn table(&self) -> &TensorND {
        &self.table
    }

    /// The trainable parameter (the table) paired with its gradient index, for
    /// an optimizer. Call after a `forward` on the tape being differentiated.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        if let Some(i) = self.idx
        {
            params.push(NdParam {
                value: &mut self.table,
                grad_idx: i,
            });
        }
        params
    }
}

/// A `(seq, seq)` additive attention mask: `0` on and below the diagonal,
/// `-1e9` above it. Added to the scores before the softmax, it drives the
/// weights for "future" keys (`j > i`) to ~0 — i.e. causal/decoder attention.
/// `-1e9` (rather than `-inf`) keeps the softmax numerically safe.
fn causal_mask(seq: usize) -> TensorND {
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq
    {
        for (j, slot) in data[i * seq..(i + 1) * seq].iter_mut().enumerate()
        {
            if j > i
            {
                *slot = -1e9;
            }
        }
    }
    TensorND::new(data, vec![seq, seq])
}

/// **ALiBi** per-head slopes (Press, Smith & Lewis, *Attention with Linear
/// Biases*, ICLR 2022): the geometric sequence `mₕ = 2^(−8h/H)` for `h = 1..H`
/// (constant ratio `2^(−8/H)`; for `H` a power of two this is the paper's set,
/// from `2^(−8/H)` down to `2^(−8)`). Steeper slopes (later heads) focus on recent
/// tokens; shallow slopes attend more globally.
pub fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    (1..=n_heads)
        .map(|h| 2f32.powf(-8.0 * h as f32 / n_heads as f32))
        .collect()
}

/// **ALiBi** additive attention bias of shape `(slopes.len(), seq, seq)`: for a
/// causal query `i` and key `j ≤ i` the bias is `−slopeₕ·(i − j)` — **linear in
/// the distance**, with no learned positions — and future keys (`j > i`) get
/// `−1e9` (the causal mask). Added to the scores before the softmax it biases
/// attention toward recent tokens and extrapolates to lengths unseen in training.
pub fn alibi_bias(slopes: &[f32], seq: usize) -> TensorND {
    let h = slopes.len();
    let mut data = vec![0.0f32; h * seq * seq];
    for (head, &m) in slopes.iter().enumerate()
    {
        for i in 0..seq
        {
            for j in 0..seq
            {
                data[(head * seq + i) * seq + j] = if j > i { -1e9 } else { -m * (i - j) as f32 };
            }
        }
    }
    TensorND::new(data, vec![h, seq, seq])
}

/// Multi-head self-attention over the N-D tape, built from [`NdLinear`]
/// projections and the N-D attention math (`bmm` / `transpose_last2` /
/// `softmax`). Input and output are `(seq, d_model)`.
pub struct NdMultiHeadAttention {
    w_q: NdLinear,
    w_k: NdLinear,
    w_v: NdLinear,
    w_o: NdLinear,
    n_heads: usize,
    num_kv_heads: usize,
    d_model: usize,
    d_head: usize,
    causal: bool,
    rope: bool,
    alibi: bool,
}

/// Frequency base for [`NdMultiHeadAttention`] rotary embeddings.
const ROPE_BASE: f32 = 10000.0;

impl NdMultiHeadAttention {
    /// New layer; `d_model` must be divisible by `n_heads`. Seeded init.
    /// `causal = true` masks each position from attending to later ones
    /// (decoder/LM attention). Rotary embeddings are off by default — enable
    /// with [`Self::with_rope`].
    pub fn new(d_model: usize, n_heads: usize, causal: bool, rng: &mut PcgEngine) -> Self {
        Self::new_gqa(d_model, n_heads, n_heads, causal, rng)
    }

    /// **Grouped-query attention** (Ainslie et al. 2023): `num_kv_heads` key/value
    /// heads shared across the `n_heads` query heads (`n_heads` must be a
    /// multiple of `num_kv_heads`). `num_kv_heads == n_heads` is standard MHA;
    /// `num_kv_heads == 1` is multi-query attention. The K/V projections shrink
    /// to `num_kv_heads · d_head`.
    pub fn new_gqa(
        d_model: usize,
        n_heads: usize,
        num_kv_heads: usize,
        causal: bool,
        rng: &mut PcgEngine,
    ) -> Self {
        assert!(d_model % n_heads == 0, "d_model must divide n_heads");
        assert!(
            num_kv_heads >= 1 && n_heads % num_kv_heads == 0,
            "n_heads must be a multiple of num_kv_heads"
        );
        let d_head = d_model / n_heads;
        let kv_dim = num_kv_heads * d_head;
        Self {
            w_q: NdLinear::new(d_model, d_model, rng),
            w_k: NdLinear::new(d_model, kv_dim, rng),
            w_v: NdLinear::new(d_model, kv_dim, rng),
            w_o: NdLinear::new(d_model, d_model, rng),
            n_heads,
            num_kv_heads,
            d_model,
            d_head,
            causal,
            rope: false,
            alibi: false,
        }
    }

    /// Enable **ALiBi** (Press et al. 2022): a static per-head linear-distance bias
    /// on the attention scores instead of learned/rotary positions. Implies causal
    /// masking and is mutually exclusive with RoPE; builder-style. Standard MHA
    /// only (`num_kv_heads == n_heads`).
    pub fn with_alibi(mut self) -> Self {
        assert_eq!(
            self.n_heads, self.num_kv_heads,
            "ALiBi here is for standard MHA (num_kv_heads == n_heads)"
        );
        self.alibi = true;
        self.causal = true;
        self.rope = false;
        self
    }

    /// Enable (or disable) **rotary position embeddings** on Q and K
    /// (Su et al. 2021). Requires an even `d_head`. Builder-style, so existing
    /// call sites are unaffected.
    pub fn with_rope(mut self, enabled: bool) -> Self {
        assert!(
            !enabled || self.d_head % 2 == 0,
            "RoPE needs an even d_head (got {})",
            self.d_head
        );
        self.rope = enabled;
        self
    }

    /// `(seq, heads·d_head) → (heads, seq, d_head)`.
    fn split_heads<'t>(&self, x: NdVar<'t>, seq: usize, heads: usize) -> NdVar<'t> {
        x.reshape(&[seq, heads, self.d_head]).permute(&[1, 0, 2])
    }

    /// Self-attention `softmax(Q·Kᵀ/√d)·V`, then the output projection. When
    /// `causal`, a triangular mask is added to the scores before the softmax so
    /// position `i` cannot attend to any `j > i`. With grouped-query attention
    /// (`num_kv_heads < n_heads`) each key/value head is shared across a group of
    /// query heads via `bmm` batch broadcasting.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let seq = x.shape()[0];

        let q = self.w_q.forward(tape, x);
        let mut q = self.split_heads(q, seq, self.n_heads);
        let k = self.w_k.forward(tape, x);
        let mut k = self.split_heads(k, seq, self.num_kv_heads);
        let v = self.w_v.forward(tape, x);
        let v = self.split_heads(v, seq, self.num_kv_heads);

        if self.rope
        {
            // Rotate Q and K per head over (heads, seq, d_head); attention then
            // depends only on relative position.
            q = q.rope(ROPE_BASE);
            k = k.rope(ROPE_BASE);
        }

        let scale = tape.input(TensorND::new(
            vec![1.0 / (self.d_head as f32).sqrt()],
            vec![1],
        ));
        let group = self.n_heads / self.num_kv_heads;

        let ctx = if group == 1
        {
            // Standard multi-head path: (n_heads, seq, d_head).
            let mut scores = q.bmm(k.transpose_last2()).mul(scale);
            if self.alibi
            {
                // ALiBi: per-head linear-distance bias (already includes the mask).
                scores = scores.add(tape.input(alibi_bias(&alibi_slopes(self.n_heads), seq)));
            }
            else if self.causal
            {
                scores = scores.add(tape.input(causal_mask(seq)));
            }
            scores.softmax().bmm(v)
        }
        else
        {
            // GQA: q (kv_heads, group, seq, d_head) vs k/v (kv_heads, 1, …) — the
            // size-1 group axis broadcasts, sharing each kv head across `group`
            // query heads. Then fold the group axis back into the head axis.
            let kvh = self.num_kv_heads;
            let qg = q.reshape(&[kvh, group, seq, self.d_head]);
            let kg = k.reshape(&[kvh, 1, seq, self.d_head]);
            let vg = v.reshape(&[kvh, 1, seq, self.d_head]);
            let mut scores = qg.bmm(kg.transpose_last2()).mul(scale); // (kvh, group, seq, seq)
            if self.causal
            {
                scores = scores.add(tape.input(causal_mask(seq)));
            }
            scores
                .softmax()
                .bmm(vg)
                .reshape(&[self.n_heads, seq, self.d_head])
        };

        // Merge heads: (h, seq, d_head) → (seq, h, d_head) → (seq, d_model).
        let merged = ctx.permute(&[1, 0, 2]).reshape(&[seq, self.d_model]);
        self.w_o.forward(tape, merged)
    }

    /// SGD-update every projection from a `backward` result.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.w_q.sgd_step(grads, lr);
        self.w_k.sgd_step(grads, lr);
        self.w_v.sgd_step(grads, lr);
        self.w_o.sgd_step(grads, lr);
    }

    /// Trainable parameters of all four projections, in q/k/v/o order.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

/// Layer normalisation over the last axis with a learnable affine
/// (`y = γ·layernorm(x) + β`).
pub struct NdLayerNorm {
    gamma: TensorND, // (d,)
    beta: TensorND,  // (d,)
    eps: f32,
    g_idx: Option<usize>,
    b_idx: Option<usize>,
}

impl NdLayerNorm {
    /// New layer over the last axis of width `d`. `γ = 1`, `β = 0`.
    pub fn new(d: usize, eps: f32) -> Self {
        Self {
            gamma: TensorND::ones(&[d]),
            beta: TensorND::zeros(&[d]),
            eps,
            g_idx: None,
            b_idx: None,
        }
    }

    /// Forward: normalise the last axis then apply the broadcast affine.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let g = tape.input(self.gamma.clone());
        let b = tape.input(self.beta.clone());
        self.g_idx = Some(g.idx());
        self.b_idx = Some(b.idx());
        x.layernorm(self.eps).mul(g).add(b)
    }

    /// SGD-update `γ` and `β`.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.g_idx
        {
            for (p, &gv) in self.gamma.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * gv;
            }
        }
        if let Some(i) = self.b_idx
        {
            for (p, &gv) in self.beta.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * gv;
            }
        }
    }

    /// Trainable parameters (gamma, beta) with their gradient indices.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        if let Some(i) = self.g_idx
        {
            params.push(NdParam {
                value: &mut self.gamma,
                grad_idx: i,
            });
        }
        if let Some(i) = self.b_idx
        {
            params.push(NdParam {
                value: &mut self.beta,
                grad_idx: i,
            });
        }
        params
    }
}

/// RMS normalisation over the last axis with a learnable scale
/// (`y = γ · rmsnorm(x)`) — the LLaMA-family normalisation (no centring, no
/// bias). Cheaper than [`NdLayerNorm`].
pub struct NdRmsNorm {
    gamma: TensorND, // (d,)
    eps: f32,
    g_idx: Option<usize>,
}

impl NdRmsNorm {
    /// New layer over the last axis of width `d`. `γ = 1`.
    pub fn new(d: usize, eps: f32) -> Self {
        Self {
            gamma: TensorND::ones(&[d]),
            eps,
            g_idx: None,
        }
    }

    /// Forward: RMS-normalise the last axis then apply the broadcast scale.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let g = tape.input(self.gamma.clone());
        self.g_idx = Some(g.idx());
        x.rmsnorm(self.eps).mul(g)
    }

    /// SGD-update `γ`.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        if let Some(i) = self.g_idx
        {
            for (p, &gv) in self.gamma.data.iter_mut().zip(&grads[i].data)
            {
                *p -= lr * gv;
            }
        }
    }

    /// Trainable parameter (`γ`) with its gradient index.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = Vec::new();
        if let Some(i) = self.g_idx
        {
            params.push(NdParam {
                value: &mut self.gamma,
                grad_idx: i,
            });
        }
        params
    }
}

/// A **SwiGLU** feed-forward block (Shazeer 2020): `SiLU(x·Wg) ⊙ (x·Wu)` then a
/// down-projection, where `SiLU(z) = z · σ(z)`. The gated-FFN used by LLaMA/PaLM
/// in place of the two-matrix ReLU FFN. Input/output `(…, d_model)`; the gate
/// and up projections widen to `d_ff`.
pub struct NdSwiGLU {
    w_gate: NdLinear, // d_model → d_ff
    w_up: NdLinear,   // d_model → d_ff
    w_down: NdLinear, // d_ff → d_model
}

impl NdSwiGLU {
    /// New block. Seeded init.
    pub fn new(d_model: usize, d_ff: usize, rng: &mut PcgEngine) -> Self {
        Self {
            w_gate: NdLinear::new(d_model, d_ff, rng),
            w_up: NdLinear::new(d_model, d_ff, rng),
            w_down: NdLinear::new(d_ff, d_model, rng),
        }
    }

    /// Forward `down( SiLU(gate(x)) ⊙ up(x) )`.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let gate = self.w_gate.forward(tape, x);
        let silu = gate.mul(gate.sigmoid()); // SiLU(z) = z·σ(z)
        let up = self.w_up.forward(tape, x);
        self.w_down.forward(tape, silu.mul(up))
    }

    /// SGD-update the three projections.
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.w_gate.sgd_step(grads, lr);
        self.w_up.sgd_step(grads, lr);
        self.w_down.sgd_step(grads, lr);
    }

    /// Trainable parameters of all three projections (gate, up, down).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.w_gate.parameters();
        params.extend(self.w_up.parameters());
        params.extend(self.w_down.parameters());
        params
    }
}

/// A full **Pre-LN transformer block** over the N-D tape:
/// `x₁ = x + Attn(LN₁(x))`, `y = x₁ + FFN(LN₂(x₁))`. Input/output `(seq, d_model)`.
pub struct NdTransformerBlock {
    ln1: NdLayerNorm,
    attn: NdMultiHeadAttention,
    ln2: NdLayerNorm,
    ffn1: NdLinear,
    ffn2: NdLinear,
}

impl NdTransformerBlock {
    /// New block. Seeded init. `causal` selects masked (decoder/LM) attention.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        causal: bool,
        rng: &mut PcgEngine,
    ) -> Self {
        Self {
            ln1: NdLayerNorm::new(d_model, 1e-5),
            attn: NdMultiHeadAttention::new(d_model, n_heads, causal, rng),
            ln2: NdLayerNorm::new(d_model, 1e-5),
            ffn1: NdLinear::new(d_model, d_ff, rng),
            ffn2: NdLinear::new(d_ff, d_model, rng),
        }
    }

    /// Pre-LN forward with residual connections.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let a = self.ln1.forward(tape, x);
        let a = self.attn.forward(tape, a);
        let x1 = x.add(a); // residual 1
        let f = self.ln2.forward(tape, x1);
        let f = self.ffn1.forward(tape, f).relu();
        let f = self.ffn2.forward(tape, f);
        x1.add(f) // residual 2
    }

    /// SGD-update every parameter (both LayerNorms, attention, both FFN linears).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.ln1.sgd_step(grads, lr);
        self.attn.sgd_step(grads, lr);
        self.ln2.sgd_step(grads, lr);
        self.ffn1.sgd_step(grads, lr);
        self.ffn2.sgd_step(grads, lr);
    }

    /// Trainable parameters of the whole block, in a fixed order
    /// (ln1, attention, ln2, ffn1, ffn2).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.ln1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ffn1.parameters());
        params.extend(self.ffn2.parameters());
        params
    }
}

/// A **LLaMA-style transformer block**: Pre-**RMSNorm**, causal attention, and a
/// **SwiGLU** feed-forward, with residuals —
/// `x₁ = x + Attn(RMS₁(x))`, `y = x₁ + SwiGLU(RMS₂(x₁))`. The modern decoder
/// block (vs the LayerNorm + ReLU-FFN [`NdTransformerBlock`]). `(seq, d_model)`.
pub struct NdLlamaBlock {
    norm1: NdRmsNorm,
    attn: NdMultiHeadAttention,
    norm2: NdRmsNorm,
    ffn: NdSwiGLU,
}

impl NdLlamaBlock {
    /// New block. Seeded init. `causal` selects masked (decoder/LM) attention.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        causal: bool,
        rng: &mut PcgEngine,
    ) -> Self {
        Self {
            norm1: NdRmsNorm::new(d_model, 1e-5),
            attn: NdMultiHeadAttention::new(d_model, n_heads, causal, rng),
            norm2: NdRmsNorm::new(d_model, 1e-5),
            ffn: NdSwiGLU::new(d_model, d_ff, rng),
        }
    }

    /// Pre-RMSNorm forward with residual connections.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let a = self.norm1.forward(tape, x);
        let a = self.attn.forward(tape, a);
        let x1 = x.add(a); // residual 1
        let f = self.norm2.forward(tape, x1);
        let f = self.ffn.forward(tape, f);
        x1.add(f) // residual 2
    }

    /// SGD-update every parameter (both RMSNorms, attention, the SwiGLU FFN).
    pub fn sgd_step(&mut self, grads: &[TensorND], lr: f32) {
        self.norm1.sgd_step(grads, lr);
        self.attn.sgd_step(grads, lr);
        self.norm2.sgd_step(grads, lr);
        self.ffn.sgd_step(grads, lr);
    }

    /// Trainable parameters in a fixed order (norm1, attention, norm2, ffn).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.norm1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ffn.parameters());
        params
    }
}

/// **DeltaNet delta rule** (Yang et al. 2024) — linear-attention recurrence with
/// a fast-weight memory `S` (`d×d`) updated by the *delta rule*:
///
/// ```text
/// S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ ,   o_t = S_t q_t
/// ```
///
/// i.e. each step writes the *prediction error* `v_t − S_{t-1}k_t` into memory,
/// gated by `β_t ∈ (0,1)`. Linear-time, **causal**, and fully differentiable: the
/// recurrence is unrolled on the tape (per-timestep `gather`/`matmul`, outputs
/// reassembled with [`cat0`](NdVar::cat0)), so gradients are exact and
/// finite-difference-checked. `q`, `k`, `v` are `(seq, d)`, `beta` is `(seq, 1)`;
/// returns `(seq, d)`.
pub fn delta_rule<'t>(
    tape: &'t NdTape,
    q: NdVar<'t>,
    k: NdVar<'t>,
    v: NdVar<'t>,
    beta: NdVar<'t>,
) -> NdVar<'t> {
    let qs = q.shape();
    let (seq, d) = (qs[0], qs[1]);
    let mut s = tape.input(TensorND::zeros(&[d, d])); // fast-weight memory S_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let k_col = k.gather(&[t]).reshape(&[d, 1]);
        let k_row = k.gather(&[t]).reshape(&[1, d]);
        let v_col = v.gather(&[t]).reshape(&[d, 1]);
        let q_col = q.gather(&[t]).reshape(&[d, 1]);
        let b_t = beta.gather(&[t]); // (1,1), broadcasts over (d,d)
        let sk = s.matmul(k_col); // S k_t                (d,1)
        let delta = v_col.matmul(k_row).sub(sk.matmul(k_row)); // (v_t − S k_t) k_tᵀ
        s = s.add(delta.mul(b_t)); // S_t                  (d,d)
        outs.push(s.matmul(q_col).reshape(&[1, d])); // o_t (1,d)
    }
    outs[0].cat0(&outs[1..])
}

/// **DeltaNet** single-head linear-attention layer: project the input to
/// `q, k, v` and a per-step gate `β = σ(·)`, then run the [`delta_rule`]
/// recurrence. Deterministic; trainable through the N-D tape like the other
/// layers. `forward` maps `(seq, d_model) → (seq, d_model)`.
pub struct NdDeltaNet {
    q_proj: NdLinear,
    k_proj: NdLinear,
    v_proj: NdLinear,
    beta_proj: NdLinear,
}

impl NdDeltaNet {
    /// New layer with seeded projections (`q,k,v: d_model→d_model`, `β: d_model→1`).
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            q_proj: NdLinear::new(d_model, d_model, rng),
            k_proj: NdLinear::new(d_model, d_model, rng),
            v_proj: NdLinear::new(d_model, d_model, rng),
            beta_proj: NdLinear::new(d_model, 1, rng),
        }
    }

    /// Forward pass over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let q = self.q_proj.forward(tape, x);
        let k = self.k_proj.forward(tape, x);
        let v = self.v_proj.forward(tape, x);
        let beta = self.beta_proj.forward(tape, x).sigmoid(); // (seq,1) ∈ (0,1)
        delta_rule(tape, q, k, v, beta)
    }

    /// Trainable parameters in a fixed order (q, k, v, β projections).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.beta_proj.parameters());
        params
    }
}

/// **Mamba selective scan** (Gu & Dao 2023) — the S6 input-dependent
/// state-space recurrence with a diagonal state matrix. For each channel `i` of
/// `d` and state index `j` of `n`, with continuous `A = −exp(a_log)` and a
/// **selective** (input-dependent) timestep `Δ`, input matrix `B`, output matrix
/// `C`, the zero-order-hold discretisation gives:
///
/// ```text
/// h_t[i,j] = exp(Δ_t[i]·A[i,j])·h_{t-1}[i,j] + Δ_t[i]·B_t[j]·x_t[i]
/// y_t[i]   = Σ_j h_t[i,j]·C_t[j]
/// ```
///
/// Linear-time, causal, deterministic; unrolled on the tape (so gradients are
/// exact and finite-difference-checked). `x`/`delta` are `(seq, d)`, `a_log` is
/// `(d, n)`, `b`/`c` are `(seq, n)`; returns `(seq, d)`. `delta` must already be
/// positive (the layer uses `Δ = exp(·)`).
pub fn selective_scan<'t>(
    tape: &'t NdTape,
    x: NdVar<'t>,
    delta: NdVar<'t>,
    a_log: NdVar<'t>,
    b: NdVar<'t>,
    c: NdVar<'t>,
) -> NdVar<'t> {
    let xs = x.shape();
    let (seq, d) = (xs[0], xs[1]);
    let n = a_log.shape()[1];
    let ea = a_log.exp(); // exp(a_log) = −A  (d, n)
    let neg1 = tape.input(TensorND::new(vec![-1.0f32], vec![1, 1]));
    let mut h = tape.input(TensorND::zeros(&[d, n])); // h_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let dt_col = delta.gather(&[t]).reshape(&[d, 1]); // Δ_t   (d,1)
        let x_col = x.gather(&[t]).reshape(&[d, 1]); //      x_t   (d,1)
        let b_row = b.gather(&[t]).reshape(&[1, n]); //      B_t   (1,n)
        let c_col = c.gather(&[t]).reshape(&[n, 1]); //      C_t   (n,1)
        // Ā = exp(Δ ⊙ A) = exp(−Δ ⊙ exp(a_log))
        let da = dt_col.mul(ea).mul(neg1).exp(); // (d,n)
        // B̄x = (Δ ⊙ B) ⊙ x
        let dbx = dt_col.mul(b_row).mul(x_col); // (d,n)
        h = da.mul(h).add(dbx); // (d,n)
        outs.push(h.matmul(c_col).reshape(&[1, d])); // y_t (1,d)
    }
    outs[0].cat0(&outs[1..])
}

/// **Mamba** selective state-space layer: project the input to the SSM input
/// `x`, the selective timestep `Δ = exp(·)`, and the input/output matrices
/// `B, C`, run the [`selective_scan`], add the gated skip `D ⊙ x`, and project
/// back. Diagonal real `A` initialised S4D-style (`A[:,j] = −(j+1)`).
/// Deterministic; trainable through the N-D tape. `(seq, d_model) → (seq, d_model)`.
pub struct NdMamba {
    in_proj: NdLinear,
    delta_proj: NdLinear,
    b_proj: NdLinear,
    c_proj: NdLinear,
    out_proj: NdLinear,
    a_log: TensorND,  // (d_inner, n)
    d_skip: TensorND, // (1, d_inner)
    a_idx: Option<usize>,
    d_idx: Option<usize>,
}

impl NdMamba {
    /// New layer; `d_inner` is the SSM channel count, `n` the state size.
    pub fn new(d_model: usize, d_inner: usize, n: usize, rng: &mut PcgEngine) -> Self {
        // S4D-real init: A[:,j] = −(j+1) ⇒ a_log[:,j] = ln(j+1).
        let mut a_log = vec![0f32; d_inner * n];
        for i in 0..d_inner
        {
            for j in 0..n
            {
                a_log[i * n + j] = ((j + 1) as f32).ln();
            }
        }
        Self {
            in_proj: NdLinear::new(d_model, d_inner, rng),
            delta_proj: NdLinear::new(d_model, d_inner, rng),
            b_proj: NdLinear::new(d_model, n, rng),
            c_proj: NdLinear::new(d_model, n, rng),
            out_proj: NdLinear::new(d_inner, d_model, rng),
            a_log: TensorND::new(a_log, vec![d_inner, n]),
            d_skip: TensorND::zeros(&[1, d_inner]),
            a_idx: None,
            d_idx: None,
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let xi = self.in_proj.forward(tape, x); // (seq, d_inner)
        let delta = self.delta_proj.forward(tape, x).exp(); // Δ > 0
        let b = self.b_proj.forward(tape, x); // (seq, n)
        let c = self.c_proj.forward(tape, x); // (seq, n)
        let a_log_v = tape.input(self.a_log.clone());
        self.a_idx = Some(a_log_v.idx());
        let scan = selective_scan(tape, xi, delta, a_log_v, b, c); // (seq, d_inner)
        let d_v = tape.input(self.d_skip.clone());
        self.d_idx = Some(d_v.idx());
        let y = scan.add(d_v.mul(xi)); // gated skip D ⊙ x
        self.out_proj.forward(tape, y) // (seq, d_model)
    }

    /// Trainable parameters in a fixed order.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let (a_idx, d_idx) = (self.a_idx, self.d_idx);
        let mut params = self.in_proj.parameters();
        params.extend(self.delta_proj.parameters());
        params.extend(self.b_proj.parameters());
        params.extend(self.c_proj.parameters());
        params.extend(self.out_proj.parameters());
        if let Some(i) = a_idx
        {
            params.push(NdParam {
                value: &mut self.a_log,
                grad_idx: i,
            });
        }
        if let Some(i) = d_idx
        {
            params.push(NdParam {
                value: &mut self.d_skip,
                grad_idx: i,
            });
        }
        params
    }
}

/// **S4D** diagonal structured state-space scan (Gu et al., *S4*, ICLR 2022; Gu,
/// Goel & Ré, *S4D*) — a **linear time-invariant** SSM. Unlike Mamba's
/// input-dependent [`selective_scan`], the diagonal state matrix `A`, the
/// input/output vectors `B`/`C` and the per-channel step `Δ` are **fixed
/// parameters**: discretise `Ā = exp(Δ⊙A)` and `B̄ = Δ⊙B`, run the recurrence
/// `h_t = Ā⊙h_{t−1} + B̄⊙x_t` (state `(d, n)`) and read out `y_t = Σ_n C⊙h_t`.
/// `x` is `(seq, d)`; `a_log`, `b`, `c` are `(d, n)`; `log_dt` is `(d, 1)`; returns
/// `(seq, d)`. `A = −exp(a_log) < 0` keeps the recurrence contractive. Gradients
/// are exact and finite-difference-checked.
pub fn s4_scan<'t>(
    tape: &'t NdTape,
    x: NdVar<'t>,
    a_log: NdVar<'t>,
    b: NdVar<'t>,
    c: NdVar<'t>,
    log_dt: NdVar<'t>,
) -> NdVar<'t> {
    let xs = x.shape();
    let (seq, d) = (xs[0], xs[1]);
    let n = a_log.shape()[1];
    let neg1 = tape.input(TensorND::new(vec![-1.0f32], vec![1, 1]));
    let ones_n = tape.input(TensorND::new(vec![1.0f32; n], vec![n, 1])); // sum over n
    let dt = log_dt.exp(); // Δ > 0   (d,1)
    let a = a_log.exp().mul(neg1); // A = −exp(a_log)  (d,n)
    let abar = dt.mul(a).exp(); // exp(Δ⊙A)         (d,n)
    let bbar = dt.mul(b); // Δ⊙B              (d,n)
    let mut h = tape.input(TensorND::zeros(&[d, n])); // h_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let x_col = x.gather(&[t]).reshape(&[d, 1]); // x_t   (d,1)
        let bx = bbar.mul(x_col); // B̄⊙x_t (d,n)
        h = abar.mul(h).add(bx); // (d,n)
        let y = c.mul(h).matmul(ones_n).reshape(&[1, d]); // Σ_n C⊙h  (1,d)
        outs.push(y);
    }
    outs[0].cat0(&outs[1..])
}

/// **S4** layer (diagonal, S4D): project the input to the SSM channels, run a
/// fixed-parameter [`s4_scan`] with HiPPO-style diagonal init `A[:,j] = −(j+1)`,
/// add a gated skip `D⊙x` and project back. Deterministic; trainable through the
/// N-D tape. `(seq, d_model) → (seq, d_model)`.
pub struct NdS4 {
    in_proj: NdLinear,
    out_proj: NdLinear,
    a_log: TensorND,  // (d, n)
    b: TensorND,      // (d, n)
    c: TensorND,      // (d, n)
    log_dt: TensorND, // (d, 1)
    d_skip: TensorND, // (1, d)
    a_idx: Option<usize>,
    b_idx: Option<usize>,
    c_idx: Option<usize>,
    dt_idx: Option<usize>,
    skip_idx: Option<usize>,
}

impl NdS4 {
    /// New layer; `d` is the SSM channel count, `n` the state size.
    pub fn new(d_model: usize, d: usize, n: usize, rng: &mut PcgEngine) -> Self {
        // HiPPO-style diagonal (S4D-Lin) init: A[:,j] = −(j+1) ⇒ a_log = ln(j+1).
        let mut a_log = vec![0f32; d * n];
        for i in 0..d
        {
            for j in 0..n
            {
                a_log[i * n + j] = ((j + 1) as f32).ln();
            }
        }
        // B = 1 (standard S4D), C seeded, Δ ≈ 0.1 (log_dt = ln 0.1).
        let b = vec![1.0f32; d * n];
        let c: Vec<f32> = (0..d * n)
            .map(|_| rng.float_signed() * (1.0 / n as f32).sqrt())
            .collect();
        let log_dt = vec![(0.1f32).ln(); d];
        Self {
            in_proj: NdLinear::new(d_model, d, rng),
            out_proj: NdLinear::new(d, d_model, rng),
            a_log: TensorND::new(a_log, vec![d, n]),
            b: TensorND::new(b, vec![d, n]),
            c: TensorND::new(c, vec![d, n]),
            log_dt: TensorND::new(log_dt, vec![d, 1]),
            d_skip: TensorND::zeros(&[1, d]),
            a_idx: None,
            b_idx: None,
            c_idx: None,
            dt_idx: None,
            skip_idx: None,
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let xi = self.in_proj.forward(tape, x); // (seq, d)
        let a_v = tape.input(self.a_log.clone());
        self.a_idx = Some(a_v.idx());
        let b_v = tape.input(self.b.clone());
        self.b_idx = Some(b_v.idx());
        let c_v = tape.input(self.c.clone());
        self.c_idx = Some(c_v.idx());
        let dt_v = tape.input(self.log_dt.clone());
        self.dt_idx = Some(dt_v.idx());
        let scan = s4_scan(tape, xi, a_v, b_v, c_v, dt_v); // (seq, d)
        let skip_v = tape.input(self.d_skip.clone());
        self.skip_idx = Some(skip_v.idx());
        let y = scan.add(skip_v.mul(xi)); // gated skip D⊙x
        self.out_proj.forward(tape, y) // (seq, d_model)
    }

    /// Trainable parameters in a fixed order.
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let (a_idx, b_idx, c_idx, dt_idx, skip_idx) = (
            self.a_idx,
            self.b_idx,
            self.c_idx,
            self.dt_idx,
            self.skip_idx,
        );
        let mut params = self.in_proj.parameters();
        params.extend(self.out_proj.parameters());
        for (idx, value) in [
            (a_idx, &mut self.a_log),
            (b_idx, &mut self.b),
            (c_idx, &mut self.c),
            (dt_idx, &mut self.log_dt),
            (skip_idx, &mut self.d_skip),
        ]
        {
            if let Some(i) = idx
            {
                params.push(NdParam { value, grad_idx: i });
            }
        }
        params
    }
}

/// **LoRA** — Low-Rank Adaptation (Hu et al., ICLR 2022). A frozen base weight
/// `W` (`in × out`) is adapted by a trainable low-rank update `ΔW = (α/r)·A·B`,
/// with `A` (`in × r`) and `B` (`r × out`), so only `r·(in+out)` parameters are
/// learned instead of `in·out`. `B` starts at **zero** (so the layer is exactly
/// the base map at init) and `A` is seeded; **only `A` and `B`** are returned by
/// [`parameters()`](Self::parameters) — the base `W` never moves. Acts on the
/// last axis like [`NdLinear`]; `y = x·W + (α/r)·(x·A)·B`. Gradient-checked.
pub struct LoraLinear {
    w: TensorND,  // (in, out) — frozen base
    a: TensorND,  // (in, r)   — trainable
    b: TensorND,  // (r, out)  — trainable
    scaling: f32, // α / r
    in_features: usize,
    out_features: usize,
    a_idx: Option<usize>,
    b_idx: Option<usize>,
}

impl LoraLinear {
    /// New adapter over a given frozen base weight `w` (`in × out`, row-major),
    /// with rank `r` and LoRA `alpha`. `A ~ U(-s, s)` (`s = 1/√in`), `B = 0`.
    pub fn new(
        w: Vec<f32>,
        in_features: usize,
        out_features: usize,
        r: usize,
        alpha: f32,
        rng: &mut PcgEngine,
    ) -> Self {
        assert_eq!(w.len(), in_features * out_features, "LoraLinear: base size");
        assert!(r >= 1, "LoraLinear: rank must be ≥ 1");
        let s = (1.0 / in_features as f32).sqrt();
        let a: Vec<f32> = (0..in_features * r)
            .map(|_| rng.float_signed() * s)
            .collect();
        Self {
            w: TensorND::new(w, vec![in_features, out_features]),
            a: TensorND::new(a, vec![in_features, r]),
            b: TensorND::zeros(&[r, out_features]),
            scaling: alpha / r as f32,
            in_features,
            out_features,
            a_idx: None,
            b_idx: None,
        }
    }

    /// Forward `(…, in) → (…, out)` (leading axes flattened then restored).
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let xs = x.shape();
        let in_f = *xs.last().expect("LoraLinear: input has no axes");
        assert_eq!(in_f, self.in_features, "LoraLinear: input feature mismatch");
        let m: usize = xs[..xs.len() - 1].iter().product();

        let w = tape.input(self.w.clone());
        let a = tape.input(self.a.clone());
        let b = tape.input(self.b.clone());
        self.a_idx = Some(a.idx());
        self.b_idx = Some(b.idx());
        let scale = tape.input(TensorND::new(vec![self.scaling], vec![1, 1]));

        let x2 = x.reshape(&[m, in_f]); // (m, in)
        let base = x2.matmul(w); // (m, out)
        let delta = x2.matmul(a).matmul(b).mul(scale); // (m, out)
        let y2 = base.add(delta);

        let mut out_shape = xs[..xs.len() - 1].to_vec();
        out_shape.push(self.out_features);
        y2.reshape(&out_shape)
    }

    /// The two trainable LoRA factors `A` and `B` (the base `W` is frozen).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let (a_idx, b_idx) = (self.a_idx, self.b_idx);
        let mut params = Vec::new();
        if let Some(i) = a_idx
        {
            params.push(NdParam {
                value: &mut self.a,
                grad_idx: i,
            });
        }
        if let Some(i) = b_idx
        {
            params.push(NdParam {
                value: &mut self.b,
                grad_idx: i,
            });
        }
        params
    }
}

/// **Retention** (RetNet, Sun et al. 2023) — a linear-attention recurrence with a
/// fixed per-head decay `γ`: a state matrix `S` (`d × d`) accumulates the outer
/// products `kₜᵀvₜ` with exponential decay, and the output reads it out with the
/// query:
///
/// ```text
/// S_t = γ·S_{t-1} + kₜᵀ·vₜ ,   o_t = q_t·S_t
/// ```
///
/// This recurrent form is mathematically **equal to** the parallel form
/// `(QKᵀ ⊙ D)V` with `D_{nm} = γ^{n-m}` (causal) — the RetNet duality, used as the
/// test oracle. Linear-time, causal, deterministic; unrolled on the tape (so
/// gradients are exact and finite-difference-checked). `q`/`k`/`v` are
/// `(seq, d)`; returns `(seq, d)`.
pub fn retention<'t>(
    tape: &'t NdTape,
    q: NdVar<'t>,
    k: NdVar<'t>,
    v: NdVar<'t>,
    gamma: f32,
) -> NdVar<'t> {
    let qs = q.shape();
    let (seq, d) = (qs[0], qs[1]);
    let g = tape.input(TensorND::new(vec![gamma], vec![1, 1]));
    let mut s = tape.input(TensorND::zeros(&[d, d])); // S_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let k_col = k.gather(&[t]).reshape(&[d, 1]); // (d,1)
        let v_row = v.gather(&[t]).reshape(&[1, d]); // (1,d)
        let q_row = q.gather(&[t]).reshape(&[1, d]); // (1,d)
        let kv = k_col.matmul(v_row); // kₜᵀvₜ  (d,d)
        s = s.mul(g).add(kv); // γS + kᵀv
        outs.push(q_row.matmul(s)); // o_t  (1,d)
    }
    outs[0].cat0(&outs[1..])
}

/// **RetNet** single-head retention layer: project the input to `q, k, v` and run
/// the [`retention`] recurrence with a fixed decay `γ`. Deterministic; trainable
/// through the N-D tape. `forward` maps `(seq, d_model) → (seq, d_model)`.
pub struct NdRetention {
    q_proj: NdLinear,
    k_proj: NdLinear,
    v_proj: NdLinear,
    gamma: f32,
}

impl NdRetention {
    /// New layer with seeded projections and decay `gamma ∈ (0, 1)`.
    pub fn new(d_model: usize, gamma: f32, rng: &mut PcgEngine) -> Self {
        Self {
            q_proj: NdLinear::new(d_model, d_model, rng),
            k_proj: NdLinear::new(d_model, d_model, rng),
            v_proj: NdLinear::new(d_model, d_model, rng),
            gamma,
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let q = self.q_proj.forward(tape, x);
        let k = self.k_proj.forward(tape, x);
        let v = self.v_proj.forward(tape, x);
        retention(tape, q, k, v, self.gamma)
    }

    /// Trainable parameters (q, k, v projections).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params
    }
}

/// **Gated Linear Attention** (GLA, Yang et al., ICML 2024) — a linear-attention
/// recurrence with a **data-dependent** per-key-channel forget gate `αₜ ∈ (0,1)`
/// (instead of RetNet's fixed scalar decay):
///
/// ```text
/// S_t = diag(αₜ)·S_{t-1} + kₜᵀ·vₜ ,   o_t = q_t·S_t
/// ```
///
/// Linear-time, causal, deterministic; unrolled on the tape. `q`/`k`/`v` and the
/// gate `alpha` are `(seq, d)` (`alpha` already in `(0,1)`); returns `(seq, d)`.
pub fn gated_linear_attention<'t>(
    tape: &'t NdTape,
    q: NdVar<'t>,
    k: NdVar<'t>,
    v: NdVar<'t>,
    alpha: NdVar<'t>,
) -> NdVar<'t> {
    let qs = q.shape();
    let (seq, d) = (qs[0], qs[1]);
    let mut s = tape.input(TensorND::zeros(&[d, d])); // S_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let a_col = alpha.gather(&[t]).reshape(&[d, 1]); // (d,1) per-key-channel gate
        let k_col = k.gather(&[t]).reshape(&[d, 1]);
        let v_row = v.gather(&[t]).reshape(&[1, d]);
        let q_row = q.gather(&[t]).reshape(&[1, d]);
        let kv = k_col.matmul(v_row); // (d,d)
        s = s.mul(a_col).add(kv); // diag(α)S + kᵀv  (α_col broadcasts over columns)
        outs.push(q_row.matmul(s)); // o_t (1,d)
    }
    outs[0].cat0(&outs[1..])
}

/// **GLA** single-head layer: project the input to `q, k, v` and a data-dependent
/// forget gate `α = σ(·)`, then run [`gated_linear_attention`]. Deterministic;
/// trainable through the N-D tape. `(seq, d_model) → (seq, d_model)`.
pub struct NdGla {
    q_proj: NdLinear,
    k_proj: NdLinear,
    v_proj: NdLinear,
    g_proj: NdLinear,
}

impl NdGla {
    /// New layer with seeded projections (`q,k,v` and the gate).
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            q_proj: NdLinear::new(d_model, d_model, rng),
            k_proj: NdLinear::new(d_model, d_model, rng),
            v_proj: NdLinear::new(d_model, d_model, rng),
            g_proj: NdLinear::new(d_model, d_model, rng),
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let q = self.q_proj.forward(tape, x);
        let k = self.k_proj.forward(tape, x);
        let v = self.v_proj.forward(tape, x);
        let alpha = self.g_proj.forward(tape, x).sigmoid(); // gate ∈ (0,1)
        gated_linear_attention(tape, q, k, v, alpha)
    }

    /// Trainable parameters (q, k, v, gate projections).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.g_proj.parameters());
        params
    }
}

/// **HGRN gated linear recurrence** (Qin et al., NeurIPS 2023) — a per-channel
/// (elementwise) leaky integrator with a data-dependent forget gate
/// `fₜ ∈ (0,1)`:
///
/// ```text
/// hₜ = fₜ ⊙ h_{t-1} + (1 − fₜ) ⊙ cₜ ,   oₜ = hₜ
/// ```
///
/// No matrix state — linear-time, causal, deterministic; unrolled on the tape.
/// `c` (candidate) and `f` (gate, already in `(0,1)`) are `(seq, d)`; returns
/// `(seq, d)`.
pub fn hgrn<'t>(tape: &'t NdTape, c: NdVar<'t>, f: NdVar<'t>) -> NdVar<'t> {
    let cs = c.shape();
    let (seq, d) = (cs[0], cs[1]);
    let ones = tape.input(TensorND::new(vec![1.0f32; d], vec![1, d]));
    let mut h = tape.input(TensorND::zeros(&[1, d])); // h_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let f_t = f.gather(&[t]); // (1,d)
        let c_t = c.gather(&[t]); // (1,d)
        let one_minus_f = ones.sub(f_t); // (1 − fₜ)
        h = f_t.mul(h).add(one_minus_f.mul(c_t)); // fₜ⊙h + (1−fₜ)⊙cₜ
        outs.push(h);
    }
    outs[0].cat0(&outs[1..])
}

/// **HGRN** single-layer token mixer: a candidate `c = W_c·x` is leaked into a
/// running state through a **lower-bounded** forget gate
/// `f = lb + (1 − lb)·σ(W_f·x)` (the lower bound `lb ∈ [0,1)` controls the
/// minimum memory horizon — deeper layers use a larger `lb`). Deterministic;
/// trainable through the N-D tape. `(seq, d_model) → (seq, d_model)`.
pub struct NdHgrn {
    c_proj: NdLinear,
    f_proj: NdLinear,
    lower_bound: f32,
}

impl NdHgrn {
    /// New layer with seeded projections and forget-gate lower bound `lb ∈ [0,1)`.
    pub fn new(d_model: usize, lb: f32, rng: &mut PcgEngine) -> Self {
        Self {
            c_proj: NdLinear::new(d_model, d_model, rng),
            f_proj: NdLinear::new(d_model, d_model, rng),
            lower_bound: lb,
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let c = self.c_proj.forward(tape, x);
        let lb = tape.input(TensorND::new(vec![self.lower_bound], vec![1, 1]));
        let scale = tape.input(TensorND::new(vec![1.0 - self.lower_bound], vec![1, 1]));
        // f = lb + (1−lb)·σ(W_f·x)  ∈ [lb, 1)
        let f = self.f_proj.forward(tape, x).sigmoid().mul(scale).add(lb);
        hgrn(tape, c, f)
    }

    /// Trainable parameters (candidate and forget-gate projections).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.c_proj.parameters();
        params.extend(self.f_proj.parameters());
        params
    }
}

/// **RWKV time-mixing** — the WKV operator (Peng et al., *RWKV: Reinventing RNNs
/// for the Transformer Era*, 2023). A linear-attention recurrence with a
/// **per-channel exponential time decay** `decay ∈ (0,1)` and a **bonus** for the
/// current token:
///
/// ```text
/// wkv_t = ( Σ_{i<t} decay^{t-1-i}·e^{k_i}·v_i + bonus·e^{k_t}·v_t )
///         / ( Σ_{i<t} decay^{t-1-i}·e^{k_i}     + bonus·e^{k_t} )
/// ```
///
/// computed in linear time by carrying numerator/denominator states `(a, b)`:
/// `wkv_t = (a + bonus·e^{k_t}·v_t)/(b + bonus·e^{k_t})`, then
/// `a ← decay·a + e^{k_t}·v_t`, `b ← decay·b + e^{k_t}`. Causal and deterministic;
/// unrolled on the tape (uses the `exp` and `div` ops). `k`/`v` are `(seq, d)`;
/// `decay` (in `(0,1)`) and `bonus` (`> 0`) are `(1, d)`; returns `(seq, d)`.
pub fn rwkv_wkv<'t>(
    tape: &'t NdTape,
    k: NdVar<'t>,
    v: NdVar<'t>,
    decay: NdVar<'t>,
    bonus: NdVar<'t>,
) -> NdVar<'t> {
    let ks = k.shape();
    let (seq, d) = (ks[0], ks[1]);
    let mut a = tape.input(TensorND::zeros(&[1, d])); // numerator state
    let mut b = tape.input(TensorND::zeros(&[1, d])); // denominator state
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let k_t = k.gather(&[t]); // (1,d)
        let v_t = v.gather(&[t]); // (1,d)
        let ek = k_t.exp(); // e^{k_t}
        let euk = bonus.mul(ek); // bonus·e^{k_t} (= e^{u+k_t})
        let num = a.add(euk.mul(v_t)); // a + bonus·e^{k}·v
        let den = b.add(euk); // b + bonus·e^{k}
        outs.push(num.div(den)); // wkv_t
        // Carry the (decayed) running sums to the next step.
        let ekv = ek.mul(v_t);
        a = decay.mul(a).add(ekv); // decay·a + e^{k}·v
        b = decay.mul(b).add(ek); // decay·b + e^{k}
    }
    outs[0].cat0(&outs[1..])
}

/// **RWKV** time-mixing block: receptance-gated WKV. From the input `x` it
/// projects a **receptance** gate `r = σ(W_r·x) ∈ (0,1)`, a key `k = W_k·x` and a
/// value `v = W_v·x`, runs the [`rwkv_wkv`] recurrence with **learnable**
/// per-channel decay `σ(w_decay) ∈ (0,1)` and bonus `e^{u_bonus} > 0`, and gates
/// the projected output: `out = r ⊙ (W_o·wkv)`. Deterministic; trainable through
/// the N-D tape. `(seq, d_model) → (seq, d_model)`. (Token-shift is omitted; this
/// is the core time-mixing operator.)
pub struct NdRwkv {
    r_proj: NdLinear,
    k_proj: NdLinear,
    v_proj: NdLinear,
    o_proj: NdLinear,
    w_decay: TensorND, // raw (1,d); decay = σ(w_decay) ∈ (0,1)
    u_bonus: TensorND, // raw (1,d); bonus = e^{u_bonus} > 0
    wd_idx: Option<usize>,
    ub_idx: Option<usize>,
}

impl NdRwkv {
    /// New layer with seeded projections; decay initialised slow (`σ(2)≈0.88`) and
    /// bonus initialised to 1 (`u = 0`).
    pub fn new(d_model: usize, rng: &mut PcgEngine) -> Self {
        Self {
            r_proj: NdLinear::new(d_model, d_model, rng),
            k_proj: NdLinear::new(d_model, d_model, rng),
            v_proj: NdLinear::new(d_model, d_model, rng),
            o_proj: NdLinear::new(d_model, d_model, rng),
            w_decay: TensorND::new(vec![2.0f32; d_model], vec![1, d_model]),
            u_bonus: TensorND::new(vec![0.0f32; d_model], vec![1, d_model]),
            wd_idx: None,
            ub_idx: None,
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let r = self.r_proj.forward(tape, x).sigmoid(); // receptance gate ∈ (0,1)
        let k = self.k_proj.forward(tape, x);
        let v = self.v_proj.forward(tape, x);
        let wd = tape.input(self.w_decay.clone());
        self.wd_idx = Some(wd.idx());
        let ub = tape.input(self.u_bonus.clone());
        self.ub_idx = Some(ub.idx());
        let decay = wd.sigmoid(); // ∈ (0,1)
        let bonus = ub.exp(); // > 0
        let wkv = rwkv_wkv(tape, k, v, decay, bonus);
        let out = self.o_proj.forward(tape, wkv);
        r.mul(out) // receptance-gated output
    }

    /// Trainable parameters (r/k/v/o projections + per-channel decay & bonus).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.r_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        if let Some(i) = self.wd_idx
        {
            params.push(NdParam {
                value: &mut self.w_decay,
                grad_idx: i,
            });
        }
        if let Some(i) = self.ub_idx
        {
            params.push(NdParam {
                value: &mut self.u_bonus,
                grad_idx: i,
            });
        }
        params
    }
}

/// **sLSTM cell** — the scalar-memory xLSTM recurrence (Beck et al., *xLSTM:
/// Extended Long Short-Term Memory*, NeurIPS 2024, arXiv:2405.04517). It extends
/// the LSTM with an **exponential input gate** and a **normaliser state**:
///
/// ```text
/// iₜ = exp(ĩₜ)            (exponential input gate)
/// fₜ = σ(f̃ₜ)             (forget gate ∈ (0,1))
/// zₜ = tanh(z̃ₜ)          (cell input)
/// oₜ = σ(õₜ)             (output gate ∈ (0,1))
/// cₜ = fₜ·cₜ₋₁ + iₜ·zₜ    (cell state)
/// nₜ = fₜ·nₜ₋₁ + iₜ       (normaliser state)
/// hₜ = oₜ ⊙ (cₜ / nₜ)
/// ```
///
/// `tanh` is built from the available `sigmoid` op via the exact identity
/// `tanh(x) = 2σ(2x) − 1`. The log-space stabiliser state `mₜ` is **omitted**: it
/// cancels exactly in the ratio `cₜ/nₜ` (a pure numerical device) and the bounded
/// inputs used here keep `exp` finite; the normaliser `nₜ ≥ iₜ = exp(ĩₜ) > 0` is
/// provably positive so the division is always well-defined. Because `cₜ/nₜ` is a
/// positive-weighted average of `zₜ ∈ (−1,1)`, the output is intrinsically bounded
/// in `(−1,1)` (stable without the stabiliser). Recurrent gate connections (memory
/// mixing) are omitted — gates come from the (projected) input, and each channel
/// runs an independent scalar cell. Pre-activations `i_pre`/`f_pre`/`z_pre`/`o_pre`
/// are `(seq, d)`; returns `(seq, d)`. Unrolled on the tape; gradient-checked.
pub fn slstm_scan<'t>(
    tape: &'t NdTape,
    i_pre: NdVar<'t>,
    f_pre: NdVar<'t>,
    z_pre: NdVar<'t>,
    o_pre: NdVar<'t>,
) -> NdVar<'t> {
    let xs = i_pre.shape();
    let (seq, d) = (xs[0], xs[1]);
    let two = tape.input(TensorND::new(vec![2.0f32], vec![1, 1]));
    let one = tape.input(TensorND::new(vec![1.0f32], vec![1, 1]));
    let mut c = tape.input(TensorND::zeros(&[1, d])); // cell state c_0 = 0
    let mut n = tape.input(TensorND::zeros(&[1, d])); // normaliser n_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let i_t = i_pre.gather(&[t]).exp(); // exponential input gate
        let f_t = f_pre.gather(&[t]).sigmoid(); // forget gate ∈ (0,1)
        let z_t = z_pre.gather(&[t]).mul(two).sigmoid().mul(two).sub(one); // tanh
        let o_t = o_pre.gather(&[t]).sigmoid(); // output gate ∈ (0,1)
        c = f_t.mul(c).add(i_t.mul(z_t)); // cₜ = fₜcₜ₋₁ + iₜzₜ
        n = f_t.mul(n).add(i_t); // nₜ = fₜnₜ₋₁ + iₜ
        outs.push(o_t.mul(c.div(n))); // hₜ = oₜ⊙(cₜ/nₜ)
    }
    outs[0].cat0(&outs[1..])
}

/// **mLSTM cell** — the matrix-memory xLSTM recurrence (Beck et al., *xLSTM*).
/// Replaces the scalar cell of [`slstm_scan`] with a `d×d` covariance memory
/// updated by key/value **outer products**, with a query read-out:
///
/// ```text
/// iₜ = exp(ĩₜ),  fₜ = σ(f̃ₜ)             (scalar gates)
/// Cₜ = fₜ·Cₜ₋₁ + iₜ·(vₜᵀ kₜ)            (d×d matrix state)
/// nₜ = fₜ·nₜ₋₁ + iₜ·kₜ                   (normaliser, 1×d)
/// hₜ = (Cₜ qₜᵀ) / max(|nₜ·qₜ|, 1)
/// ```
///
/// The stabilising denominator is built from the **exact** identities
/// `|a| = relu(a) + relu(−a)` and `max(a,1) = relu(a−1) + 1`, so no new op is
/// needed and the guard is faithful (not omitted). `q`/`k`/`v` are `(seq, d)`; the
/// scalar pre-gates `i_pre`/`f_pre` are `(seq, 1)`; returns `(seq, d)`. Unrolled on
/// the tape; forward-checked against a reference recurrence and gradient-checked.
pub fn mlstm_scan<'t>(
    tape: &'t NdTape,
    q: NdVar<'t>,
    k: NdVar<'t>,
    v: NdVar<'t>,
    i_pre: NdVar<'t>,
    f_pre: NdVar<'t>,
) -> NdVar<'t> {
    let qs = q.shape();
    let (seq, d) = (qs[0], qs[1]);
    let one = tape.input(TensorND::new(vec![1.0f32], vec![1, 1]));
    let neg1 = tape.input(TensorND::new(vec![-1.0f32], vec![1, 1]));
    let mut cmat = tape.input(TensorND::zeros(&[d, d])); // C_0 = 0
    let mut n = tape.input(TensorND::zeros(&[1, d])); // n_0 = 0
    let mut outs: Vec<NdVar<'t>> = Vec::with_capacity(seq);
    for t in 0..seq
    {
        let q_t = q.gather(&[t]); // (1,d)
        let k_t = k.gather(&[t]); // (1,d)
        let v_t = v.gather(&[t]); // (1,d)
        let i_t = i_pre.gather(&[t]).exp(); // (1,1) exp gate
        let f_t = f_pre.gather(&[t]).sigmoid(); // (1,1) forget gate
        let outer = v_t.reshape(&[d, 1]).matmul(k_t); // vₜᵀkₜ  (d,d)
        cmat = f_t.mul(cmat).add(i_t.mul(outer)); // Cₜ
        n = f_t.mul(n).add(i_t.mul(k_t)); // nₜ
        let hraw = cmat.matmul(q_t.reshape(&[d, 1])); // Cₜqₜᵀ  (d,1)
        let dot = n.mul(q_t).sum().reshape(&[1, 1]); // nₜ·qₜ   (1,1)
        let abs = dot.relu().add(dot.mul(neg1).relu()); // |nₜ·qₜ|
        let denom = abs.sub(one).relu().add(one); // max(|·|,1)
        outs.push(hraw.div(denom).reshape(&[1, d])); // hₜ
    }
    outs[0].cat0(&outs[1..])
}

/// **xLSTM block** (sLSTM variant): from the input `x` it projects the four sLSTM
/// pre-activations (input/forget/cell/output), runs the [`slstm_scan`] scalar
/// recurrence and projects the hidden state back. Deterministic; trainable through
/// the N-D tape. `(seq, d_model) → (seq, d_model)`. (The matrix-memory variant is
/// available as the standalone [`mlstm_scan`].)
pub struct NdXlstm {
    i_proj: NdLinear,
    f_proj: NdLinear,
    z_proj: NdLinear,
    o_proj: NdLinear,
    out_proj: NdLinear,
}

impl NdXlstm {
    /// New layer with seeded projections; `d` is the sLSTM cell width.
    pub fn new(d_model: usize, d: usize, rng: &mut PcgEngine) -> Self {
        Self {
            i_proj: NdLinear::new(d_model, d, rng),
            f_proj: NdLinear::new(d_model, d, rng),
            z_proj: NdLinear::new(d_model, d, rng),
            o_proj: NdLinear::new(d_model, d, rng),
            out_proj: NdLinear::new(d, d_model, rng),
        }
    }

    /// Forward over a `(seq, d_model)` sequence.
    pub fn forward<'t>(&mut self, tape: &'t NdTape, x: NdVar<'t>) -> NdVar<'t> {
        let i_pre = self.i_proj.forward(tape, x);
        let f_pre = self.f_proj.forward(tape, x);
        let z_pre = self.z_proj.forward(tape, x);
        let o_pre = self.o_proj.forward(tape, x);
        let h = slstm_scan(tape, i_pre, f_pre, z_pre, o_pre);
        self.out_proj.forward(tape, h)
    }

    /// Trainable parameters (four gate projections + output projection).
    pub fn parameters(&mut self) -> Vec<NdParam<'_>> {
        let mut params = self.i_proj.parameters();
        params.extend(self.f_proj.parameters());
        params.extend(self.z_proj.parameters());
        params.extend(self.o_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mse<'t>(pred: NdVar<'t>, target: NdVar<'t>) -> NdVar<'t> {
        let diff = pred.sub(target);
        diff.mul(diff).sum()
    }

    /// A plain-`Vec` reference for the delta-rule recurrence, for the forward test.
    fn delta_rule_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        seq: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut s = vec![0f32; d * d];
        let mut out = vec![0f32; seq * d];
        for t in 0..seq
        {
            // S_{t-1} k_t
            let mut sk = vec![0f32; d];
            for i in 0..d
            {
                for j in 0..d
                {
                    sk[i] += s[i * d + j] * k[t * d + j];
                }
            }
            // S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ
            for i in 0..d
            {
                for j in 0..d
                {
                    s[i * d + j] += beta[t] * (v[t * d + i] - sk[i]) * k[t * d + j];
                }
            }
            // o_t = S_t q_t
            for i in 0..d
            {
                let mut acc = 0f32;
                for j in 0..d
                {
                    acc += s[i * d + j] * q[t * d + j];
                }
                out[t * d + i] = acc;
            }
        }
        out
    }

    /// The tape-unrolled `delta_rule` matches the hand-written recurrence.
    #[test]
    fn delta_rule_matches_reference() {
        let (seq, d) = (4usize, 3usize);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 + 0.4).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 - 0.3).sin())
            .collect();
        let beta: Vec<f32> = (0..seq).map(|i| 0.3 + 0.1 * i as f32).collect();

        let want = delta_rule_reference(&q, &k, &v, &beta, seq, d);
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q, vec![seq, d]));
        let kv = tape.input(TensorND::new(k, vec![seq, d]));
        let vv = tape.input(TensorND::new(v, vec![seq, d]));
        let bv = tape.input(TensorND::new(beta, vec![seq, 1]));
        let out = tape.value(delta_rule(&tape, qv, kv, vv, bv));
        assert_eq!(out.shape, vec![seq, d]);
        for (got, w) in out.data.iter().zip(&want)
        {
            assert!((got - w).abs() < 1e-5, "delta_rule mismatch: {got} vs {w}");
        }
    }

    /// `delta_rule` gradients (w.r.t. q, k, v, β) match finite differences.
    #[test]
    fn delta_rule_gradient_check() {
        let (seq, d) = (3usize, 2usize);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 + 0.2).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.25 - 0.1).sin())
            .collect();
        let beta: Vec<f32> = vec![0.3, 0.6, 0.5];

        // loss = Σ out²
        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32], bb: &[f32]| -> f32 {
            let t = NdTape::new();
            let qv = t.input(TensorND::new(qq.to_vec(), vec![seq, d]));
            let kv = t.input(TensorND::new(kk.to_vec(), vec![seq, d]));
            let vv2 = t.input(TensorND::new(vv.to_vec(), vec![seq, d]));
            let bv = t.input(TensorND::new(bb.to_vec(), vec![seq, 1]));
            let o = delta_rule(&t, qv, kv, vv2, bv);
            t.value(o.mul(o).sum()).data[0]
        };
        let t = NdTape::new();
        let qv = t.input(TensorND::new(q.clone(), vec![seq, d]));
        let kv = t.input(TensorND::new(k.clone(), vec![seq, d]));
        let vv = t.input(TensorND::new(v.clone(), vec![seq, d]));
        let bv = t.input(TensorND::new(beta.clone(), vec![seq, 1]));
        let o = delta_rule(&t, qv, kv, vv, bv);
        let grads = t.backward(o.mul(o).sum());
        let (gq, gk, gv, gb) = (
            grads[qv.idx()].clone(),
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
            grads[bv.idx()].clone(),
        );

        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "delta_rule grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gq, &q, &|p| loss_of(p, &k, &v, &beta));
        check(&gk, &k, &|p| loss_of(&q, p, &v, &beta));
        check(&gv, &v, &|p| loss_of(&q, &k, p, &beta));
        check(&gb, &beta, &|p| loss_of(&q, &k, &v, p));
    }

    /// The `NdDeltaNet` layer is deterministic and can drive a loss down: training
    /// its projections with Adam reduces the MSE to a fixed target sequence.
    #[test]
    fn nd_deltanet_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(7);
            let mut layer = NdDeltaNet::new(d, &mut rng);
            let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..120
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(
            last < first * 0.5,
            "DeltaNet did not learn: {first} -> {last}"
        );
        // Determinism: a second identical run gives bit-identical endpoints.
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// Plain-`Vec` reference for the Mamba selective scan, for the forward test.
    #[allow(clippy::too_many_arguments)]
    fn selective_scan_reference(
        x: &[f32],
        delta: &[f32],
        a_log: &[f32],
        b: &[f32],
        c: &[f32],
        seq: usize,
        d: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut h = vec![0f32; d * n];
        let mut out = vec![0f32; seq * d];
        for t in 0..seq
        {
            for i in 0..d
            {
                for j in 0..n
                {
                    let a = -(a_log[i * n + j].exp()); // A = −exp(a_log)
                    let da = (delta[t * d + i] * a).exp();
                    let dbx = delta[t * d + i] * b[t * n + j] * x[t * d + i];
                    h[i * n + j] = da * h[i * n + j] + dbx;
                }
            }
            for i in 0..d
            {
                let mut acc = 0f32;
                for j in 0..n
                {
                    acc += h[i * n + j] * c[t * n + j];
                }
                out[t * d + i] = acc;
            }
        }
        out
    }

    /// The tape-unrolled `selective_scan` matches the hand-written SSM recurrence.
    #[test]
    fn selective_scan_matches_reference() {
        let (seq, d, n) = (4usize, 3usize, 2usize);
        let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let delta: Vec<f32> = (0..seq * d).map(|i| 0.2 + 0.05 * i as f32).collect();
        let a_log: Vec<f32> = (0..d * n).map(|j| ((j % n) as f32 + 1.0).ln()).collect();
        let b: Vec<f32> = (0..seq * n).map(|i| (i as f32 * 0.2 + 0.3).cos()).collect();
        let c: Vec<f32> = (0..seq * n)
            .map(|i| (i as f32 * 0.17 - 0.2).sin())
            .collect();

        let want = selective_scan_reference(&x, &delta, &a_log, &b, &c, seq, d, n);
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x, vec![seq, d]));
        let dv = tape.input(TensorND::new(delta, vec![seq, d]));
        let av = tape.input(TensorND::new(a_log, vec![d, n]));
        let bv = tape.input(TensorND::new(b, vec![seq, n]));
        let cv = tape.input(TensorND::new(c, vec![seq, n]));
        let y = tape.value(selective_scan(&tape, xv, dv, av, bv, cv));
        assert_eq!(y.shape, vec![seq, d]);
        for (got, w) in y.data.iter().zip(&want)
        {
            assert!(
                (got - w).abs() < 1e-5,
                "selective_scan mismatch: {got} vs {w}"
            );
        }
    }

    /// `selective_scan` gradients (x, Δ, a_log, B, C) match finite differences.
    #[test]
    fn selective_scan_gradient_check() {
        let (seq, d, n) = (3usize, 2usize, 2usize);
        let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let delta: Vec<f32> = (0..seq * d).map(|i| 0.2 + 0.1 * i as f32).collect();
        let a_log: Vec<f32> = (0..d * n).map(|j| ((j % n) as f32 + 1.0).ln()).collect();
        let b: Vec<f32> = (0..seq * n).map(|i| (i as f32 * 0.3 + 0.2).cos()).collect();
        let c: Vec<f32> = (0..seq * n)
            .map(|i| (i as f32 * 0.25 - 0.1).sin())
            .collect();

        let loss_of = |xx: &[f32], dd: &[f32], aa: &[f32], bb: &[f32], cc: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xx.to_vec(), vec![seq, d]));
            let dv = t.input(TensorND::new(dd.to_vec(), vec![seq, d]));
            let av = t.input(TensorND::new(aa.to_vec(), vec![d, n]));
            let bv = t.input(TensorND::new(bb.to_vec(), vec![seq, n]));
            let cv = t.input(TensorND::new(cc.to_vec(), vec![seq, n]));
            let y = selective_scan(&t, xv, dv, av, bv, cv);
            t.value(y.mul(y).sum()).data[0]
        };
        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d]));
        let dv = t.input(TensorND::new(delta.clone(), vec![seq, d]));
        let av = t.input(TensorND::new(a_log.clone(), vec![d, n]));
        let bv = t.input(TensorND::new(b.clone(), vec![seq, n]));
        let cv = t.input(TensorND::new(c.clone(), vec![seq, n]));
        let y = selective_scan(&t, xv, dv, av, bv, cv);
        let grads = t.backward(y.mul(y).sum());
        let (gx, gd, ga, gb, gc) = (
            grads[xv.idx()].clone(),
            grads[dv.idx()].clone(),
            grads[av.idx()].clone(),
            grads[bv.idx()].clone(),
            grads[cv.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "selective_scan grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gx, &x, &|p| loss_of(p, &delta, &a_log, &b, &c));
        check(&gd, &delta, &|p| loss_of(&x, p, &a_log, &b, &c));
        check(&ga, &a_log, &|p| loss_of(&x, &delta, p, &b, &c));
        check(&gb, &b, &|p| loss_of(&x, &delta, &a_log, p, &c));
        check(&gc, &c, &|p| loss_of(&x, &delta, &a_log, &b, p));
    }

    /// The `NdMamba` layer trains (MSE↓ to a target sequence) and is
    /// bit-for-bit deterministic across identical runs.
    #[test]
    fn nd_mamba_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d_model) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(5);
            let mut layer = NdMamba::new(d_model, 6, 4, &mut rng);
            let x: Vec<f32> = (0..seq * d_model)
                .map(|i| (i as f32 * 0.3 - 1.0).sin())
                .collect();
            let target: Vec<f32> = (0..seq * d_model).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..120
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d_model]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d_model]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.6, "Mamba did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// `s4_scan` gradients (w.r.t. x, a_log, B, C, log_dt) match finite differences.
    #[test]
    fn s4_scan_gradient_check() {
        let (seq, d, n) = (4usize, 2usize, 3usize);
        let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let a_log: Vec<f32> = (0..d * n).map(|j| ((j % n) as f32 + 1.0).ln()).collect();
        let b: Vec<f32> = (0..d * n).map(|i| 0.5 + 0.3 * (i as f32).cos()).collect();
        let c: Vec<f32> = (0..d * n).map(|i| (i as f32 * 0.25 - 0.1).sin()).collect();
        let log_dt: Vec<f32> = (0..d).map(|i| (0.2 + 0.05 * i as f32).ln()).collect();

        let loss_of = |xx: &[f32], aa: &[f32], bb: &[f32], cc: &[f32], dd: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xx.to_vec(), vec![seq, d]));
            let av = t.input(TensorND::new(aa.to_vec(), vec![d, n]));
            let bv = t.input(TensorND::new(bb.to_vec(), vec![d, n]));
            let cv = t.input(TensorND::new(cc.to_vec(), vec![d, n]));
            let dv = t.input(TensorND::new(dd.to_vec(), vec![d, 1]));
            let y = s4_scan(&t, xv, av, bv, cv, dv);
            t.value(y.mul(y).sum()).data[0]
        };
        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d]));
        let av = t.input(TensorND::new(a_log.clone(), vec![d, n]));
        let bv = t.input(TensorND::new(b.clone(), vec![d, n]));
        let cv = t.input(TensorND::new(c.clone(), vec![d, n]));
        let dv = t.input(TensorND::new(log_dt.clone(), vec![d, 1]));
        let y = s4_scan(&t, xv, av, bv, cv, dv);
        let grads = t.backward(y.mul(y).sum());
        let (gx, ga, gb, gc, gd) = (
            grads[xv.idx()].clone(),
            grads[av.idx()].clone(),
            grads[bv.idx()].clone(),
            grads[cv.idx()].clone(),
            grads[dv.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "s4_scan grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gx, &x, &|p| loss_of(p, &a_log, &b, &c, &log_dt));
        check(&ga, &a_log, &|p| loss_of(&x, p, &b, &c, &log_dt));
        check(&gb, &b, &|p| loss_of(&x, &a_log, p, &c, &log_dt));
        check(&gc, &c, &|p| loss_of(&x, &a_log, &b, p, &log_dt));
        check(&gd, &log_dt, &|p| loss_of(&x, &a_log, &b, &c, p));
    }

    /// The `NdS4` layer trains (MSE↓ to a target sequence) and is bit-for-bit
    /// deterministic across identical runs.
    #[test]
    fn nd_s4_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d_model) = (5usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(9);
            let mut layer = NdS4::new(d_model, 6, 4, &mut rng);
            let x: Vec<f32> = (0..seq * d_model)
                .map(|i| (i as f32 * 0.3 - 1.0).sin())
                .collect();
            let target: Vec<f32> = (0..seq * d_model).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..120
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d_model]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d_model]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.6, "S4 did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// `slstm_scan` gradients (w.r.t. the four gate pre-activations) match finite
    /// differences — the exp/forget/cell-input/output recurrence is smooth.
    #[test]
    fn slstm_scan_gradient_check() {
        let (seq, d) = (4usize, 3usize);
        let ip: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 0.4).sin()).collect();
        let fp: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 + 0.5).cos()).collect();
        let zp: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.25 - 0.2).sin())
            .collect();
        let op: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 + 0.1).cos())
            .collect();

        let loss_of = |a: &[f32], b: &[f32], c: &[f32], e: &[f32]| -> f32 {
            let t = NdTape::new();
            let iv = t.input(TensorND::new(a.to_vec(), vec![seq, d]));
            let fv = t.input(TensorND::new(b.to_vec(), vec![seq, d]));
            let zv = t.input(TensorND::new(c.to_vec(), vec![seq, d]));
            let ov = t.input(TensorND::new(e.to_vec(), vec![seq, d]));
            let y = slstm_scan(&t, iv, fv, zv, ov);
            t.value(y.mul(y).sum()).data[0]
        };
        let t = NdTape::new();
        let iv = t.input(TensorND::new(ip.clone(), vec![seq, d]));
        let fv = t.input(TensorND::new(fp.clone(), vec![seq, d]));
        let zv = t.input(TensorND::new(zp.clone(), vec![seq, d]));
        let ov = t.input(TensorND::new(op.clone(), vec![seq, d]));
        let y = slstm_scan(&t, iv, fv, zv, ov);
        let grads = t.backward(y.mul(y).sum());
        let (gi, gf, gz, go) = (
            grads[iv.idx()].clone(),
            grads[fv.idx()].clone(),
            grads[zv.idx()].clone(),
            grads[ov.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "slstm grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gi, &ip, &|p| loss_of(p, &fp, &zp, &op));
        check(&gf, &fp, &|p| loss_of(&ip, p, &zp, &op));
        check(&gz, &zp, &|p| loss_of(&ip, &fp, p, &op));
        check(&go, &op, &|p| loss_of(&ip, &fp, &zp, p));
    }

    /// A plain-`Vec` reference for the mLSTM matrix-memory recurrence — proves the
    /// tape forward (outer products, query read-out, `max(|nₜ·qₜ|,1)` denominator).
    #[allow(clippy::too_many_arguments)]
    fn mlstm_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        ip: &[f32],
        fp: &[f32],
        seq: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut cmat = vec![0f32; d * d];
        let mut n = vec![0f32; d];
        let mut out = vec![0f32; seq * d];
        for t in 0..seq
        {
            let it = ip[t].exp();
            let ft = 1.0 / (1.0 + (-fp[t]).exp());
            for a in 0..d
            {
                for b in 0..d
                {
                    cmat[a * d + b] = ft * cmat[a * d + b] + it * v[t * d + a] * k[t * d + b];
                }
            }
            for (b, nb) in n.iter_mut().enumerate()
            {
                *nb = ft * *nb + it * k[t * d + b];
            }
            let mut dot = 0f32;
            for b in 0..d
            {
                dot += n[b] * q[t * d + b];
            }
            let denom = dot.abs().max(1.0);
            for a in 0..d
            {
                let mut hr = 0f32;
                for b in 0..d
                {
                    hr += cmat[a * d + b] * q[t * d + b];
                }
                out[t * d + a] = hr / denom;
            }
        }
        out
    }

    /// The tape-unrolled `mlstm_scan` matches the hand-written matrix recurrence
    /// (with the stabilising denominator active for these moderate inputs).
    #[test]
    fn mlstm_scan_matches_reference() {
        let (seq, d) = (4usize, 3usize);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 + 0.4).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 - 0.3).sin())
            .collect();
        let ip: Vec<f32> = (0..seq).map(|i| 0.2 + 0.1 * i as f32).collect();
        let fp: Vec<f32> = (0..seq).map(|i| 0.5 - 0.1 * i as f32).collect();
        let want = mlstm_reference(&q, &k, &v, &ip, &fp, seq, d);
        let t = NdTape::new();
        let qv = t.input(TensorND::new(q, vec![seq, d]));
        let kv = t.input(TensorND::new(k, vec![seq, d]));
        let vv = t.input(TensorND::new(v, vec![seq, d]));
        let iv = t.input(TensorND::new(ip, vec![seq, 1]));
        let fv = t.input(TensorND::new(fp, vec![seq, 1]));
        let got = t.value(mlstm_scan(&t, qv, kv, vv, iv, fv));
        for (g, w) in got.data.iter().zip(&want)
        {
            assert!((g - w).abs() < 1e-4, "mlstm forward: got {g}, want {w}");
        }
    }

    /// `mlstm_scan` gradients match finite differences. Keys/queries are kept small
    /// so `|nₜ·qₜ| < 1` and the denominator sits on its flat `max(·,1)=1` branch:
    /// the recurrence is smooth (the `relu`-built guard is independently tested).
    #[test]
    fn mlstm_scan_gradient_check() {
        let (seq, d) = (4usize, 3usize);
        let q: Vec<f32> = (0..seq * d)
            .map(|i| 0.15 * (i as f32 * 0.3 - 0.5).sin())
            .collect();
        let k: Vec<f32> = (0..seq * d)
            .map(|i| 0.15 * (i as f32 * 0.2 + 0.4).cos())
            .collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 - 0.3).sin())
            .collect();
        let ip: Vec<f32> = (0..seq).map(|i| 0.2 + 0.1 * i as f32).collect();
        let fp: Vec<f32> = (0..seq).map(|i| 0.5 - 0.1 * i as f32).collect();

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32], ii: &[f32], ff: &[f32]| -> f32 {
            let t = NdTape::new();
            let qv = t.input(TensorND::new(qq.to_vec(), vec![seq, d]));
            let kv = t.input(TensorND::new(kk.to_vec(), vec![seq, d]));
            let vv = t.input(TensorND::new(vv.to_vec(), vec![seq, d]));
            let iv = t.input(TensorND::new(ii.to_vec(), vec![seq, 1]));
            let fv = t.input(TensorND::new(ff.to_vec(), vec![seq, 1]));
            let y = mlstm_scan(&t, qv, kv, vv, iv, fv);
            t.value(y.mul(y).sum()).data[0]
        };
        let t = NdTape::new();
        let qv = t.input(TensorND::new(q.clone(), vec![seq, d]));
        let kv = t.input(TensorND::new(k.clone(), vec![seq, d]));
        let vv = t.input(TensorND::new(v.clone(), vec![seq, d]));
        let iv = t.input(TensorND::new(ip.clone(), vec![seq, 1]));
        let fv = t.input(TensorND::new(fp.clone(), vec![seq, 1]));
        let y = mlstm_scan(&t, qv, kv, vv, iv, fv);
        let grads = t.backward(y.mul(y).sum());
        let (gq, gk, gv, gi, gf) = (
            grads[qv.idx()].clone(),
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
            grads[iv.idx()].clone(),
            grads[fv.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "mlstm grad {i}: numeric {num}, analytic {}",
                    analytic.data[i]
                );
            }
        };
        check(&gq, &q, &|p| loss_of(p, &k, &v, &ip, &fp));
        check(&gk, &k, &|p| loss_of(&q, p, &v, &ip, &fp));
        check(&gv, &v, &|p| loss_of(&q, &k, p, &ip, &fp));
        check(&gi, &ip, &|p| loss_of(&q, &k, &v, p, &fp));
        check(&gf, &fp, &|p| loss_of(&q, &k, &v, &ip, p));
    }

    /// The `NdXlstm` layer trains (MSE↓ to a target sequence) and is bit-for-bit
    /// deterministic across identical runs.
    #[test]
    fn nd_xlstm_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d_model) = (5usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(11);
            let mut layer = NdXlstm::new(d_model, 6, &mut rng);
            let x: Vec<f32> = (0..seq * d_model)
                .map(|i| (i as f32 * 0.3 - 1.0).sin())
                .collect();
            let target: Vec<f32> = (0..seq * d_model).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..150
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d_model]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d_model]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.6, "xLSTM did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// Parallel-form reference for RetNet retention: `o[n] = Σ_{m≤n} γ^{n−m}
    /// (q[n]·k[m]) v[m]` — the oracle the tape recurrence must match.
    fn retention_parallel(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gamma: f32,
        seq: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; seq * d];
        for n in 0..seq
        {
            for m in 0..=n
            {
                let mut qk = 0f32;
                for i in 0..d
                {
                    qk += q[n * d + i] * k[m * d + i];
                }
                let w = gamma.powi((n - m) as i32) * qk;
                for j in 0..d
                {
                    out[n * d + j] += w * v[m * d + j];
                }
            }
        }
        out
    }

    /// The tape-unrolled `retention` equals the parallel form (RetNet duality).
    #[test]
    fn retention_matches_parallel_form() {
        let (seq, d, gamma) = (5usize, 3usize, 0.9f32);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 + 0.4).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 - 0.3).sin())
            .collect();

        let want = retention_parallel(&q, &k, &v, gamma, seq, d);
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q, vec![seq, d]));
        let kv = tape.input(TensorND::new(k, vec![seq, d]));
        let vv = tape.input(TensorND::new(v, vec![seq, d]));
        let out = tape.value(retention(&tape, qv, kv, vv, gamma));
        assert_eq!(out.shape, vec![seq, d]);
        for (got, w) in out.data.iter().zip(&want)
        {
            assert!((got - w).abs() < 1e-4, "retention mismatch: {got} vs {w}");
        }
    }

    /// `retention` gradients (q, k, v) match finite differences.
    #[test]
    fn retention_gradient_check() {
        let (seq, d, gamma) = (3usize, 2usize, 0.8f32);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 + 0.2).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.25 - 0.1).sin())
            .collect();

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let t = NdTape::new();
            let qv = t.input(TensorND::new(qq.to_vec(), vec![seq, d]));
            let kv = t.input(TensorND::new(kk.to_vec(), vec![seq, d]));
            let vv2 = t.input(TensorND::new(vv.to_vec(), vec![seq, d]));
            let o = retention(&t, qv, kv, vv2, gamma);
            t.value(o.mul(o).sum()).data[0]
        };
        let t = NdTape::new();
        let qv = t.input(TensorND::new(q.clone(), vec![seq, d]));
        let kv = t.input(TensorND::new(k.clone(), vec![seq, d]));
        let vv = t.input(TensorND::new(v.clone(), vec![seq, d]));
        let o = retention(&t, qv, kv, vv, gamma);
        let grads = t.backward(o.mul(o).sum());
        let (gq, gk, gv) = (
            grads[qv.idx()].clone(),
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "retention grad {i}: {num} vs {}",
                    analytic.data[i]
                );
            }
        };
        check(&gq, &q, &|p| loss_of(p, &k, &v));
        check(&gk, &k, &|p| loss_of(&q, p, &v));
        check(&gv, &v, &|p| loss_of(&q, &k, p));
    }

    /// The `NdRetention` layer trains (MSE↓) and is bit-for-bit deterministic.
    #[test]
    fn nd_retention_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(6);
            let mut layer = NdRetention::new(d, 0.9, &mut rng);
            let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..120
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(
            last < first * 0.6,
            "RetNet did not learn: {first} -> {last}"
        );
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// Plain-`Vec` reference for the GLA recurrence (the definition oracle).
    fn gla_reference(q: &[f32], k: &[f32], v: &[f32], a: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let mut s = vec![0f32; d * d];
        let mut out = vec![0f32; seq * d];
        for t in 0..seq
        {
            for i in 0..d
            {
                for j in 0..d
                {
                    s[i * d + j] = a[t * d + i] * s[i * d + j] + k[t * d + i] * v[t * d + j];
                }
            }
            for j in 0..d
            {
                let mut acc = 0f32;
                for i in 0..d
                {
                    acc += q[t * d + i] * s[i * d + j];
                }
                out[t * d + j] = acc;
            }
        }
        out
    }

    /// The tape-unrolled `gated_linear_attention` matches the recurrence reference.
    #[test]
    fn gla_matches_reference() {
        let (seq, d) = (4usize, 3usize);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 + 0.4).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.17 - 0.3).sin())
            .collect();
        let a: Vec<f32> = (0..seq * d)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.1).sin())
            .collect();

        let want = gla_reference(&q, &k, &v, &a, seq, d);
        let tape = NdTape::new();
        let qv = tape.input(TensorND::new(q, vec![seq, d]));
        let kv = tape.input(TensorND::new(k, vec![seq, d]));
        let vv = tape.input(TensorND::new(v, vec![seq, d]));
        let av = tape.input(TensorND::new(a, vec![seq, d]));
        let out = tape.value(gated_linear_attention(&tape, qv, kv, vv, av));
        assert_eq!(out.shape, vec![seq, d]);
        for (got, w) in out.data.iter().zip(&want)
        {
            assert!((got - w).abs() < 1e-4, "GLA mismatch: {got} vs {w}");
        }
    }

    /// `gated_linear_attention` gradients (q, k, v, α) match finite differences.
    #[test]
    fn gla_gradient_check() {
        let (seq, d) = (3usize, 2usize);
        let q: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 + 0.2).cos()).collect();
        let v: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.25 - 0.1).sin())
            .collect();
        let a: Vec<f32> = (0..seq * d).map(|i| 0.4 + 0.1 * i as f32 % 0.5).collect();

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32], aa: &[f32]| -> f32 {
            let t = NdTape::new();
            let qv = t.input(TensorND::new(qq.to_vec(), vec![seq, d]));
            let kv = t.input(TensorND::new(kk.to_vec(), vec![seq, d]));
            let vv2 = t.input(TensorND::new(vv.to_vec(), vec![seq, d]));
            let av = t.input(TensorND::new(aa.to_vec(), vec![seq, d]));
            let o = gated_linear_attention(&t, qv, kv, vv2, av);
            t.value(o.mul(o).sum()).data[0]
        };
        let t = NdTape::new();
        let qv = t.input(TensorND::new(q.clone(), vec![seq, d]));
        let kv = t.input(TensorND::new(k.clone(), vec![seq, d]));
        let vv = t.input(TensorND::new(v.clone(), vec![seq, d]));
        let av = t.input(TensorND::new(a.clone(), vec![seq, d]));
        let o = gated_linear_attention(&t, qv, kv, vv, av);
        let grads = t.backward(o.mul(o).sum());
        let (gq, gk, gv, ga) = (
            grads[qv.idx()].clone(),
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
            grads[av.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "GLA grad {i}: {num} vs {}",
                    analytic.data[i]
                );
            }
        };
        check(&gq, &q, &|p| loss_of(p, &k, &v, &a));
        check(&gk, &k, &|p| loss_of(&q, p, &v, &a));
        check(&gv, &v, &|p| loss_of(&q, &k, p, &a));
        check(&ga, &a, &|p| loss_of(&q, &k, &v, p));
    }

    /// The `NdGla` layer trains (MSE↓) and is bit-for-bit deterministic.
    #[test]
    fn nd_gla_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(8);
            let mut layer = NdGla::new(d, &mut rng);
            let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..150
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.6, "GLA did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// Plain-`Vec` reference for the HGRN gated linear recurrence.
    fn hgrn_reference(c: &[f32], f: &[f32], seq: usize, d: usize) -> Vec<f32> {
        let mut h = vec![0f32; d];
        let mut out = vec![0f32; seq * d];
        for t in 0..seq
        {
            for j in 0..d
            {
                h[j] = f[t * d + j] * h[j] + (1.0 - f[t * d + j]) * c[t * d + j];
                out[t * d + j] = h[j];
            }
        }
        out
    }

    /// The tape-unrolled `hgrn` matches the recurrence reference.
    #[test]
    fn hgrn_matches_reference() {
        let (seq, d) = (5usize, 3usize);
        let c: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let f: Vec<f32> = (0..seq * d)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.1).cos())
            .collect();

        let want = hgrn_reference(&c, &f, seq, d);
        let tape = NdTape::new();
        let cv = tape.input(TensorND::new(c, vec![seq, d]));
        let fv = tape.input(TensorND::new(f, vec![seq, d]));
        let out = tape.value(hgrn(&tape, cv, fv));
        assert_eq!(out.shape, vec![seq, d]);
        for (got, w) in out.data.iter().zip(&want)
        {
            assert!((got - w).abs() < 1e-5, "HGRN mismatch: {got} vs {w}");
        }
    }

    /// `hgrn` gradients (c, f) match finite differences.
    #[test]
    fn hgrn_gradient_check() {
        let (seq, d) = (4usize, 2usize);
        let c: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let f: Vec<f32> = (0..seq * d).map(|i| 0.4 + 0.05 * i as f32 % 0.5).collect();

        let loss_of = |cc: &[f32], ff: &[f32]| -> f32 {
            let t = NdTape::new();
            let cv = t.input(TensorND::new(cc.to_vec(), vec![seq, d]));
            let fv = t.input(TensorND::new(ff.to_vec(), vec![seq, d]));
            let o = hgrn(&t, cv, fv);
            t.value(o.mul(o).sum()).data[0]
        };
        let t = NdTape::new();
        let cv = t.input(TensorND::new(c.clone(), vec![seq, d]));
        let fv = t.input(TensorND::new(f.clone(), vec![seq, d]));
        let o = hgrn(&t, cv, fv);
        let grads = t.backward(o.mul(o).sum());
        let (gc, gf) = (grads[cv.idx()].clone(), grads[fv.idx()].clone());
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 2e-2,
                    "HGRN grad {i}: {num} vs {}",
                    analytic.data[i]
                );
            }
        };
        check(&gc, &c, &|p| loss_of(p, &f));
        check(&gf, &f, &|p| loss_of(&c, p));
    }

    /// The `NdHgrn` layer trains (MSE↓) and is bit-for-bit deterministic.
    #[test]
    fn nd_hgrn_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(9);
            let mut layer = NdHgrn::new(d, 0.0, &mut rng);
            let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..150
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.7, "HGRN did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// Explicit RWKV WKV reference (the weighted-sum formula, per channel).
    fn rwkv_reference(
        k: &[f32],
        v: &[f32],
        decay: &[f32],
        bonus: &[f32],
        seq: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; seq * d];
        for j in 0..d
        {
            for t in 0..seq
            {
                let (mut num, mut den) = (0f32, 0f32);
                for i in 0..t
                {
                    let wgt = decay[j].powi((t - 1 - i) as i32) * k[i * d + j].exp();
                    num += wgt * v[i * d + j];
                    den += wgt;
                }
                let wc = bonus[j] * k[t * d + j].exp();
                num += wc * v[t * d + j];
                den += wc;
                out[t * d + j] = num / den;
            }
        }
        out
    }

    /// The tape-unrolled `rwkv_wkv` matches the explicit weighted-sum formula.
    #[test]
    fn rwkv_wkv_matches_reference() {
        let (seq, d) = (5usize, 3usize);
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2 - 0.5).sin()).collect();
        let v: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3).cos()).collect();
        let decay: Vec<f32> = (0..d).map(|j| 0.8 + 0.05 * j as f32).collect(); // ∈(0,1)
        let bonus: Vec<f32> = (0..d).map(|j| 1.0 + 0.2 * j as f32).collect(); // >0

        let want = rwkv_reference(&k, &v, &decay, &bonus, seq, d);
        let tape = NdTape::new();
        let kv = tape.input(TensorND::new(k, vec![seq, d]));
        let vv = tape.input(TensorND::new(v, vec![seq, d]));
        let dv = tape.input(TensorND::new(decay, vec![1, d]));
        let bv = tape.input(TensorND::new(bonus, vec![1, d]));
        let out = tape.value(rwkv_wkv(&tape, kv, vv, dv, bv));
        assert_eq!(out.shape, vec![seq, d]);
        for (got, w) in out.data.iter().zip(&want)
        {
            assert!((got - w).abs() < 1e-5, "RWKV mismatch: {got} vs {w}");
        }
    }

    /// `rwkv_wkv` gradients (k, v, decay, bonus) match finite differences.
    #[test]
    fn rwkv_wkv_gradient_check() {
        let (seq, d) = (4usize, 2usize);
        let k: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.4 - 0.5).sin()).collect();
        let v: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.25).cos()).collect();
        let decay: Vec<f32> = (0..d).map(|j| 0.85 + 0.05 * j as f32).collect();
        let bonus: Vec<f32> = (0..d).map(|j| 1.0 + 0.1 * j as f32).collect();

        let loss_of = |kk: &[f32], vv: &[f32], dd: &[f32], bb: &[f32]| -> f32 {
            let t = NdTape::new();
            let kv = t.input(TensorND::new(kk.to_vec(), vec![seq, d]));
            let vv2 = t.input(TensorND::new(vv.to_vec(), vec![seq, d]));
            let dv = t.input(TensorND::new(dd.to_vec(), vec![1, d]));
            let bv = t.input(TensorND::new(bb.to_vec(), vec![1, d]));
            let o = rwkv_wkv(&t, kv, vv2, dv, bv);
            t.value(o.mul(o).sum()).data[0]
        };
        let t = NdTape::new();
        let kv = t.input(TensorND::new(k.clone(), vec![seq, d]));
        let vv = t.input(TensorND::new(v.clone(), vec![seq, d]));
        let dv = t.input(TensorND::new(decay.clone(), vec![1, d]));
        let bv = t.input(TensorND::new(bonus.clone(), vec![1, d]));
        let o = rwkv_wkv(&t, kv, vv, dv, bv);
        let grads = t.backward(o.mul(o).sum());
        let (gk, gv, gd, gb) = (
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
            grads[dv.idx()].clone(),
            grads[bv.idx()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for i in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[i] += eps;
                dn[i] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[i]).abs() < 3e-2,
                    "RWKV grad {i}: {num} vs {}",
                    analytic.data[i]
                );
            }
        };
        check(&gk, &k, &|p| loss_of(p, &v, &decay, &bonus));
        check(&gv, &v, &|p| loss_of(&k, p, &decay, &bonus));
        check(&gd, &decay, &|p| loss_of(&k, &v, p, &bonus));
        check(&gb, &bonus, &|p| loss_of(&k, &v, &decay, p));
    }

    /// The `NdRwkv` layer trains (MSE↓) and is bit-for-bit deterministic.
    #[test]
    fn nd_rwkv_trains_and_is_deterministic() {
        use crate::nn::nd_optim::NdAdam;
        let (seq, d) = (4usize, 4usize);
        let run = || -> (f32, f32) {
            let mut rng = PcgEngine::new(11);
            let mut layer = NdRwkv::new(d, &mut rng);
            let x: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
            let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.2).cos()).collect();
            let mut opt = NdAdam::with_lr(0.05);
            let (mut first, mut last) = (0f32, 0f32);
            for step in 0..150
            {
                let tape = NdTape::new();
                let xv = tape.input(TensorND::new(x.clone(), vec![seq, d]));
                let tv = tape.input(TensorND::new(target.clone(), vec![seq, d]));
                let out = layer.forward(&tape, xv);
                let loss = mse(out, tv);
                let lval = tape.value(loss).data[0];
                if step == 0
                {
                    first = lval;
                }
                last = lval;
                let grads = tape.backward(loss);
                opt.step(&mut layer.parameters(), &grads);
            }
            (first, last)
        };
        let (first, last) = run();
        assert!(last < first * 0.7, "RWKV did not learn: {first} -> {last}");
        let (first2, last2) = run();
        assert_eq!(first.to_bits(), first2.to_bits());
        assert_eq!(last.to_bits(), last2.to_bits());
    }

    /// ALiBi slopes are the geometric sequence `2^(−8h/H)`: decreasing, constant
    /// ratio `2^(−8/H)`, last slope (h=H) is `2^(−8) = 1/256`.
    #[test]
    fn alibi_slopes_are_geometric() {
        let h = 8usize;
        let s = alibi_slopes(h);
        let ratio = 2f32.powf(-8.0 / h as f32);
        assert!((s[0] - ratio).abs() < 1e-6);
        for k in 1..h
        {
            assert!((s[k] / s[k - 1] - ratio).abs() < 1e-5, "ratio at {k}");
            assert!(s[k] < s[k - 1], "not decreasing at {k}");
        }
        assert!(
            (s[h - 1] - 1.0 / 256.0).abs() < 1e-6,
            "last slope {}",
            s[h - 1]
        );
    }

    /// ALiBi bias is linear in distance, causal, and shift-invariant (Toeplitz).
    #[test]
    fn alibi_bias_is_linear_causal_and_toeplitz() {
        let seq = 6usize;
        let m = 0.5f32;
        let bias = alibi_bias(&[m], seq);
        let at = |i: usize, j: usize| bias.data[i * seq + j]; // single head
        for i in 0..seq
        {
            for j in 0..=i
            {
                assert!(
                    (at(i, j) - (-m * (i - j) as f32)).abs() < 1e-6,
                    "bias({i},{j})"
                );
            }
            for j in (i + 1)..seq
            {
                assert!(at(i, j) < -1e8, "not masked ({i},{j})");
            }
        }
        assert!((at(5, 2) - at(4, 1)).abs() < 1e-6, "not Toeplitz"); // shift-invariant
        assert!(
            at(5, 5) > at(5, 4) && at(5, 4) > at(5, 0),
            "not recency-ordered"
        );
    }

    /// Applied to uniform scores then softmax, ALiBi makes the attention weights
    /// **decay with distance** — recency bias exactly `∝ exp(−slope·dist)`.
    #[test]
    fn alibi_softmax_weights_decay_with_distance() {
        let seq = 8usize;
        let m = 0.3f32;
        let bias = alibi_bias(&[m], seq);
        let i = seq - 1; // query = last position (all keys are causal/past)
        let row: Vec<f32> = (0..seq).map(|j| bias.data[i * seq + j]).collect();
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&z| (z - mx).exp()).collect();
        let z: f32 = exps.iter().sum();
        let w: Vec<f32> = exps.iter().map(|&e| e / z).collect();
        for j in 0..i
        {
            assert!(w[j] < w[j + 1], "weight not decaying at j={j}");
        }
        assert!(
            (w[i - 1] / w[i] - (-m).exp()).abs() < 1e-4,
            "decay ratio not exp(−m)"
        );
    }

    /// `NdMultiHeadAttention::with_alibi` runs end-to-end (right output shape) and
    /// is bit-for-bit deterministic — ALiBi tested on the real attention layer.
    #[test]
    fn nd_attention_with_alibi_runs_and_is_deterministic() {
        let (seq, d_model, heads) = (5usize, 8usize, 4usize);
        let run = || -> Vec<u32> {
            let mut rng = PcgEngine::new(6);
            let mut attn = NdMultiHeadAttention::new(d_model, heads, true, &mut rng).with_alibi();
            let x: Vec<f32> = (0..seq * d_model)
                .map(|i| (i as f32 * 0.2 - 1.0).sin())
                .collect();
            let tape = NdTape::new();
            let xv = tape.input(TensorND::new(x, vec![seq, d_model]));
            let out = tape.value(attn.forward(&tape, xv));
            assert_eq!(out.shape, vec![seq, d_model]);
            out.data.iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(run(), run());
    }

    /// LoRA at init equals the frozen base map (`B = 0` ⇒ `ΔW = 0`), and its
    /// `A`/`B` gradients match finite differences; `parameters()` exposes only A,B.
    #[test]
    fn lora_starts_as_base_and_gradient_checks() {
        let (in_f, out_f, r) = (4usize, 3usize, 2usize);
        let mut rng = PcgEngine::new(2);
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng.float_signed()).collect();
        let x: Vec<f32> = (0..2 * in_f)
            .map(|i| (i as f32 * 0.3 - 1.0).sin())
            .collect();

        // At init (B = 0), LoRA forward == x · W exactly.
        let mut lora = LoraLinear::new(w.clone(), in_f, out_f, r, 8.0, &mut rng);
        let t0 = NdTape::new();
        let xv = t0.input(TensorND::new(x.clone(), vec![2, in_f]));
        let y = t0.value(lora.forward(&t0, xv));
        for b in 0..2
        {
            for o in 0..out_f
            {
                let mut base = 0f32;
                for i in 0..in_f
                {
                    base += x[b * in_f + i] * w[i * out_f + o];
                }
                assert!(
                    (y.data[b * out_f + o] - base).abs() < 1e-5,
                    "LoRA init ≠ base"
                );
            }
        }
        assert_eq!(lora.parameters().len(), 2, "LoRA exposes only A and B");

        // Gradient check on A and B (perturb after a few updates so B ≠ 0).
        let a0 = lora.a.data.clone();
        let mut b0 = lora.b.data.clone();
        for v in b0.iter_mut()
        {
            *v = 0.1; // make B non-trivial for the check
        }
        let loss_of = |aa: &[f32], bb: &[f32]| -> f32 {
            let mut lr = LoraLinear::new(w.clone(), in_f, out_f, r, 8.0, &mut PcgEngine::new(2));
            lr.a = TensorND::new(aa.to_vec(), vec![in_f, r]);
            lr.b = TensorND::new(bb.to_vec(), vec![r, out_f]);
            let t = NdTape::new();
            let xv = t.input(TensorND::new(x.clone(), vec![2, in_f]));
            let o = lr.forward(&t, xv);
            t.value(o.mul(o).sum()).data[0]
        };
        let mut lr = LoraLinear::new(w.clone(), in_f, out_f, r, 8.0, &mut PcgEngine::new(2));
        lr.a = TensorND::new(a0.clone(), vec![in_f, r]);
        lr.b = TensorND::new(b0.clone(), vec![r, out_f]);
        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![2, in_f]));
        let o = lr.forward(&t, xv);
        let grads = t.backward(o.mul(o).sum());
        let (ga, gb) = (
            grads[lr.a_idx.unwrap()].clone(),
            grads[lr.b_idx.unwrap()].clone(),
        );
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for k in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[k] += eps;
                dn[k] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[k]).abs() < 2e-2,
                    "LoRA grad {k}: {num} vs {}",
                    analytic.data[k]
                );
            }
        };
        check(&ga, &a0, &|p| loss_of(p, &b0));
        check(&gb, &b0, &|p| loss_of(&a0, p));
    }

    /// Gradient of the loss w.r.t. the layer **input** matches finite
    /// differences (exercises reshape + matmul + bias-broadcast backward).
    #[test]
    fn nd_linear_input_gradient_check() {
        let mut rng = PcgEngine::new(1);
        let mut lin = NdLinear::new(3, 2, &mut rng);
        // A 3-D input (batch=2, seq=2, in=3) to exercise the flatten/restore.
        let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let target: Vec<f32> = (0..8).map(|i| (i as f32 * 0.1).cos()).collect();

        let loss_of = |xd: &[f32], lin: &mut NdLinear| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![2, 2, 3]));
            let tv = t.input(TensorND::new(target.clone(), vec![2, 2, 2]));
            let y = lin.forward(&t, xv);
            t.value(mse(y, tv)).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![2, 2, 3]));
        let tv = t.input(TensorND::new(target.clone(), vec![2, 2, 2]));
        let y = lin.forward(&t, xv);
        let grads = t.backward(mse(y, tv));
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut lin) - loss_of(&dn, &mut lin)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "input grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// A real training loop on a 2-layer N-D MLP: the regression loss strictly
    /// decreases — proving the weight gradients and the SGD step are correct.
    #[test]
    fn nd_mlp_trains_and_loss_decreases() {
        let mut rng = PcgEngine::new(7);
        let mut l1 = NdLinear::new(4, 8, &mut rng);
        let mut l2 = NdLinear::new(8, 3, &mut rng);

        // Deterministic synthetic data X (6×4) and a fixed teacher target.
        let xs: Vec<f32> = (0..24).map(|i| (i as f32 * 0.17 - 1.0).sin()).collect();
        let ts: Vec<f32> = (0..18).map(|i| (i as f32 * 0.09).cos() * 0.5).collect();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..40
        {
            let t = NdTape::new();
            let x = t.input(TensorND::new(xs.clone(), vec![6, 4]));
            let target = t.input(TensorND::new(ts.clone(), vec![6, 3]));
            let h = l1.forward(&t, x).relu();
            let y = l2.forward(&t, h);
            let loss_v = mse(y, target);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            l1.sgd_step(&grads, 0.02);
            l2.sgd_step(&grads, 0.02);
        }
        assert!(
            last < first * 0.7,
            "MLP did not learn: first {first}, last {last}"
        );
    }

    /// [`NdEmbedding`] selects the right rows and its SGD update touches only
    /// the rows that were looked up (the rest keep their values).
    #[test]
    fn nd_embedding_forward_and_update() {
        let mut rng = PcgEngine::new(3);
        let mut emb = NdEmbedding::new(4, 3, &mut rng);
        let before = emb.table().data.clone();

        let t = NdTape::new();
        let e = emb.forward(&t, &[1, 1]); // row 1, twice
        assert_eq!(e.shape(), vec![2, 3]);
        let ev = t.value(e);
        assert_eq!(&ev.data[0..3], &before[3..6]); // gathered row == table row 1
        let grads = t.backward(e.sum());
        emb.sgd_step(&grads, 0.1);

        let after = emb.table().data.clone();
        assert_eq!(&after[0..3], &before[0..3]); // row 0 untouched
        assert_ne!(&after[3..6], &before[3..6]); // row 1 moved
        assert_eq!(&after[6..12], &before[6..12]); // rows 2,3 untouched
    }

    /// The full multi-head attention **layer** (q/k/v/o projections + the
    /// attention block) is correct: its gradient w.r.t. the input matches finite
    /// differences, and the output keeps the `(seq, d_model)` shape.
    #[test]
    fn nd_attention_layer_gradient_check() {
        let (d_model, n_heads, seq) = (8usize, 2, 3);
        let mut rng = PcgEngine::new(2);
        let mut attn = NdMultiHeadAttention::new(d_model, n_heads, false, &mut rng);
        let x: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.4).sin())
            .collect();

        let loss_of = |xd: &[f32], attn: &mut NdMultiHeadAttention| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
            let out = attn.forward(&t, xv);
            t.value(out.sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let out = attn.forward(&t, xv);
        assert_eq!(out.shape(), vec![seq, d_model]);
        let grads = t.backward(out.sum());
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut attn) - loss_of(&dn, &mut attn)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "attention layer grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// **Causality**: with the mask on, changing the input at the *last*
    /// position must leave every *earlier* output position bit-for-bit
    /// unchanged — each position attends only to itself and the past. The
    /// perturbed position's own output *does* move, proving the change really
    /// propagated (so the invariance above is causality, not a dead forward).
    #[test]
    fn nd_causal_attention_is_causal() {
        let (d_model, n_heads, seq) = (8usize, 2, 4);
        let mut rng = PcgEngine::new(5);
        let mut attn = NdMultiHeadAttention::new(d_model, n_heads, true, &mut rng);

        let base: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.21 - 0.7).sin())
            .collect();
        let mut perturbed = base.clone();
        for v in perturbed[(seq - 1) * d_model..].iter_mut()
        {
            *v += 0.5; // move only the last position's features
        }

        let run = |xd: &[f32], attn: &mut NdMultiHeadAttention| -> Vec<f32> {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
            let out = attn.forward(&t, xv);
            t.value(out).data.clone()
        };

        let a = run(&base, &mut attn);
        let b = run(&perturbed, &mut attn);

        for i in 0..seq - 1
        {
            for c in 0..d_model
            {
                let k = i * d_model + c;
                assert_eq!(
                    a[k], b[k],
                    "causal leak: output position {i} changed when only the last input moved"
                );
            }
        }
        let last = (seq - 1) * d_model;
        let moved: f32 = a[last..]
            .iter()
            .zip(&b[last..])
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(
            moved > 1e-4,
            "perturbation did not propagate (moved {moved})"
        );
    }

    /// Attention **with rotary embeddings** (`with_rope(true)`) keeps its
    /// `(seq, d_model)` shape and its input gradient matches finite differences
    /// — RoPE is wired into the attention path correctly.
    #[test]
    fn nd_attention_with_rope_gradient_check() {
        let (d_model, n_heads, seq) = (8usize, 2, 3);
        let mut rng = PcgEngine::new(6);
        let mut attn = NdMultiHeadAttention::new(d_model, n_heads, true, &mut rng).with_rope(true);
        let x: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.4).sin())
            .collect();

        let loss_of = |xd: &[f32], attn: &mut NdMultiHeadAttention| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
            t.value(attn.forward(&t, xv).sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let out = attn.forward(&t, xv);
        assert_eq!(out.shape(), vec![seq, d_model]);
        let grads = t.backward(out.sum());
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut attn) - loss_of(&dn, &mut attn)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "rope attention grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// **Grouped-query attention**: the K/V projections shrink to
    /// `num_kv_heads · d_head`, the output keeps shape `(seq, d_model)`, and the
    /// input gradient matches finite differences (covers GQA *and* the
    /// multi-query `num_kv_heads = 1` case via the `bmm`-broadcast path).
    #[test]
    fn nd_gqa_gradient_check() {
        let (d_model, n_heads, seq) = (8usize, 4, 3);
        for num_kv_heads in [2usize, 1]
        {
            let mut rng = PcgEngine::new(8);
            let mut attn =
                NdMultiHeadAttention::new_gqa(d_model, n_heads, num_kv_heads, true, &mut rng);
            // K/V projections are narrower than d_model: num_kv_heads · d_head.
            let d_head = d_model / n_heads;
            assert_eq!(
                attn.w_k.weight().shape,
                vec![d_model, num_kv_heads * d_head]
            );

            let x: Vec<f32> = (0..seq * d_model)
                .map(|i| (i as f32 * 0.17 - 0.3).sin())
                .collect();
            let loss_of = |xd: &[f32], attn: &mut NdMultiHeadAttention| -> f32 {
                let t = NdTape::new();
                let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
                t.value(attn.forward(&t, xv).sum()).data[0]
            };

            let t = NdTape::new();
            let xv = t.input(TensorND::new(x.clone(), vec![seq, d_model]));
            let out = attn.forward(&t, xv);
            assert_eq!(out.shape(), vec![seq, d_model]);
            let grads = t.backward(out.sum());
            let gx = grads[xv.idx()].clone();

            let eps = 1e-3f32;
            for k in 0..x.len()
            {
                let mut up = x.clone();
                let mut dn = x.clone();
                up[k] += eps;
                dn[k] -= eps;
                let num = (loss_of(&up, &mut attn) - loss_of(&dn, &mut attn)) / (2.0 * eps);
                assert!(
                    (num - gx.data[k]).abs() < 2e-2,
                    "gqa(kv={num_kv_heads}) grad {k}: numeric {num}, analytic {}",
                    gx.data[k]
                );
            }
        }
    }

    /// A **full Pre-LN transformer block** (LayerNorm + attention + residual +
    /// FFN + residual) trains end to end on the N-D tape: the regression loss
    /// drops well below its initial value. The milestone: "here is the
    /// transformer block, and it learns".
    #[test]
    fn nd_transformer_block_trains() {
        let (d_model, n_heads, d_ff, seq) = (8usize, 2, 16, 4);
        let mut rng = PcgEngine::new(11);
        let mut block = NdTransformerBlock::new(d_model, n_heads, d_ff, false, &mut rng);

        let xs: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.5).sin())
            .collect();
        let ts: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.07).cos() * 0.3)
            .collect();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..80
        {
            let t = NdTape::new();
            let x = t.input(TensorND::new(xs.clone(), vec![seq, d_model]));
            let target = t.input(TensorND::new(ts.clone(), vec![seq, d_model]));
            let y = block.forward(&t, x);
            let loss_v = mse(y, target);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            block.sgd_step(&grads, 0.01);
        }
        assert!(
            last < first * 0.7,
            "transformer block did not learn: first {first}, last {last}"
        );
    }

    /// [`NdRmsNorm`] layer: gradient w.r.t. the input matches finite differences
    /// (exercises the rmsnorm op + the broadcast `γ` scale).
    #[test]
    fn nd_rmsnorm_layer_input_gradient_check() {
        let (d, seq) = (5usize, 3usize);
        let mut norm = NdRmsNorm::new(d, 1e-6);
        let x: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.2 - 0.6).sin() + 0.3)
            .collect();
        let target: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.1).cos() * 0.4).collect();

        let loss_of = |xd: &[f32], norm: &mut NdRmsNorm| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d]));
            let tv = t.input(TensorND::new(target.clone(), vec![seq, d]));
            let y = norm.forward(&t, xv);
            t.value(mse(y, tv)).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d]));
        let tv = t.input(TensorND::new(target.clone(), vec![seq, d]));
        let y = norm.forward(&t, xv);
        let grads = t.backward(mse(y, tv));
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut norm) - loss_of(&dn, &mut norm)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "rmsnorm layer grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// [`NdSwiGLU`] FFN: gradient w.r.t. the input matches finite differences
    /// (exercises gate/up/down projections + the SiLU gate).
    #[test]
    fn nd_swiglu_gradient_check() {
        let (d_model, d_ff, seq) = (4usize, 8usize, 2usize);
        let mut rng = PcgEngine::new(9);
        let mut ffn = NdSwiGLU::new(d_model, d_ff, &mut rng);
        let x: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.23 - 0.5).sin())
            .collect();
        let target: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.11).cos() * 0.3)
            .collect();

        let loss_of = |xd: &[f32], ffn: &mut NdSwiGLU| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![seq, d_model]));
            let tv = t.input(TensorND::new(target.clone(), vec![seq, d_model]));
            let y = ffn.forward(&t, xv);
            t.value(mse(y, tv)).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d_model]));
        let tv = t.input(TensorND::new(target.clone(), vec![seq, d_model]));
        let y = ffn.forward(&t, xv);
        assert_eq!(y.shape(), vec![seq, d_model]);
        let grads = t.backward(mse(y, tv));
        let gx = grads[xv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up, &mut ffn) - loss_of(&dn, &mut ffn)) / (2.0 * eps);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "swiglu grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// A **LLaMA-style block** (Pre-RMSNorm + causal attention + SwiGLU) trains
    /// end to end on the N-D tape: the regression loss drops well below its
    /// initial value — proof RMSNorm and SwiGLU compose and learn.
    #[test]
    fn nd_llama_block_trains() {
        let (d_model, n_heads, d_ff, seq) = (8usize, 2, 16, 4);
        let mut rng = PcgEngine::new(13);
        let mut block = NdLlamaBlock::new(d_model, n_heads, d_ff, true, &mut rng);

        let xs: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.13 - 0.5).sin())
            .collect();
        let ts: Vec<f32> = (0..seq * d_model)
            .map(|i| (i as f32 * 0.07).cos() * 0.3)
            .collect();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..80
        {
            let t = NdTape::new();
            let x = t.input(TensorND::new(xs.clone(), vec![seq, d_model]));
            let target = t.input(TensorND::new(ts.clone(), vec![seq, d_model]));
            let y = block.forward(&t, x);
            let loss_v = mse(y, target);
            let loss = t.value(loss_v).data[0];
            if step == 0
            {
                first = loss;
            }
            last = loss;
            let grads = t.backward(loss_v);
            block.sgd_step(&grads, 0.01);
        }
        assert!(
            last < first * 0.7,
            "LLaMA block did not learn: first {first}, last {last}"
        );
    }
}
