//! VRAM-resident matmul chains (feature `wgpu`).
//!
//! [`GpuChain`] keeps intermediate activations **in GPU memory across a
//! sequence of matmuls** — the result of one GEMM feeds the next without a CPU
//! round-trip. Upload the inputs once, chain `matmul`s on [`GpuMatrix`] handles,
//! and download only the final result.
//!
//! This is the device-residency mechanism: on real GPU hardware it removes the
//! per-op upload/download traffic that otherwise dominates. (On a software
//! Vulkan adapter such as Mesa lavapipe it is functionally correct but offers
//! no speedup — the value is the mechanism and its oracle-checked correctness.)
//!
//! Scope: residency here covers **GEMM chains**. Wiring it transparently into
//! the autograd tape would require the tape's value storage (`DeviceTensor`,
//! currently a CPU `Tensor`) to become lazily-materialised GPU storage and the
//! whole forward op-set (bias, activations, im2col) to be device-resident —
//! tracked as future work in `docs/GPU.md` (P2.2).

use crate::wgpu_backend::{GpuMatrix, WgpuContext};
use crate::{BackendError, BackendResult};

/// A handle to a wgpu device for building VRAM-resident matmul chains.
pub struct GpuChain {
    ctx: WgpuContext,
}

/// The resident weights of one pre-norm, single-head transformer block, as
/// [`GpuMatrix`] handles already uploaded to VRAM. Shapes for a `d`-wide model
/// with MLP hidden size `h`: the two RMSNorm gains are `d`-length, the Q/K/V/O
/// projections are `d×d`, and the MLP `w_gate`/`w_up` are `d×h` with `w_down`
/// `h×d`. Consumed by [`GpuChain::transformer_block`].
pub struct BlockWeights<'a> {
    /// Pre-attention RMSNorm gain (`d`).
    pub norm1: &'a GpuMatrix,
    /// Query projection (`d×d`).
    pub wq: &'a GpuMatrix,
    /// Key projection (`d×d`).
    pub wk: &'a GpuMatrix,
    /// Value projection (`d×d`).
    pub wv: &'a GpuMatrix,
    /// Attention output projection (`d×d`).
    pub wo: &'a GpuMatrix,
    /// Pre-MLP RMSNorm gain (`d`).
    pub norm2: &'a GpuMatrix,
    /// SwiGLU gate projection (`d×h`).
    pub wg: &'a GpuMatrix,
    /// SwiGLU up projection (`d×h`).
    pub wu: &'a GpuMatrix,
    /// SwiGLU down projection (`h×d`).
    pub wd: &'a GpuMatrix,
}

/// Weights of one **GQA transformer block** — multi-head grouped-query attention
/// with RoPE, then SwiGLU MLP — matching the sciagent `SciAgentBlock`. Unlike
/// [`BlockWeights`] (single head, square q/k/v), the key/value projections are
/// `d×(n_kv_heads·dh)` where `dh = d/n_heads`.
pub struct GqaBlockWeights<'a> {
    /// Pre-attention RMSNorm gain (`d`).
    pub norm1: &'a GpuMatrix,
    /// Query projection (`d×d`, `d = n_heads·dh`).
    pub wq: &'a GpuMatrix,
    /// Key projection (`d×kv_dim`, `kv_dim = n_kv_heads·dh`).
    pub wk: &'a GpuMatrix,
    /// Value projection (`d×kv_dim`).
    pub wv: &'a GpuMatrix,
    /// Attention output projection (`d×d`).
    pub wo: &'a GpuMatrix,
    /// Pre-MLP RMSNorm gain (`d`).
    pub norm2: &'a GpuMatrix,
    /// SwiGLU gate projection (`d×h`).
    pub wg: &'a GpuMatrix,
    /// SwiGLU up projection (`d×h`).
    pub wu: &'a GpuMatrix,
    /// SwiGLU down projection (`h×d`).
    pub wd: &'a GpuMatrix,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (`n_heads % n_kv_heads == 0`).
    pub n_kv_heads: usize,
    /// RoPE base frequency.
    pub theta: f32,
}

/// The weight gradients of one **GQA transformer block**, produced by
/// [`GpuChain::gqa_transformer_block_backward_full`] — same shapes as the
/// corresponding [`GqaBlockWeights`] fields (`dwk`/`dwv` are `d×kv_dim`,
/// `dnorm1`/`dnorm2` are `1×d`).
pub struct GqaBlockGrads {
    /// `∂L/∂Wq` (`d×d`).
    pub dwq: GpuMatrix,
    /// `∂L/∂Wk` (`d×kv_dim`).
    pub dwk: GpuMatrix,
    /// `∂L/∂Wv` (`d×kv_dim`).
    pub dwv: GpuMatrix,
    /// `∂L/∂Wo` (`d×d`).
    pub dwo: GpuMatrix,
    /// `∂L/∂Wg` (`d×h`).
    pub dwg: GpuMatrix,
    /// `∂L/∂Wu` (`d×h`).
    pub dwu: GpuMatrix,
    /// `∂L/∂Wd` (`h×d`).
    pub dwd: GpuMatrix,
    /// `∂L/∂norm1` — the pre-attention RMSNorm gain gradient (`1×d`).
    pub dnorm1: GpuMatrix,
    /// `∂L/∂norm2` — the pre-MLP RMSNorm gain gradient (`1×d`).
    pub dnorm2: GpuMatrix,
}

/// Forward activations of one transformer block that the backward pass needs to
/// read — produced by [`GpuChain::transformer_block_forward_cached`] and
/// consumed by [`GpuChain::transformer_block_backward`]. All resident.
pub struct BlockCache {
    q: GpuMatrix,       // xn·Wq   (t×d)
    k: GpuMatrix,       // xn·Wk   (t×d)
    v: GpuMatrix,       // xn·Wv   (t×d)
    weights: GpuMatrix, // softmax scores (t×t)
    h: GpuMatrix,       // x + attn·Wo  (residual 1, t×d)
    gate: GpuMatrix,    // hn·Wg   (t×h)
    up: GpuMatrix,      // hn·Wu   (t×h)
}

/// The projection weight gradients of one transformer block, produced by
/// [`GpuChain::transformer_block_backward_full`] — one resident matrix per
/// projection, same shapes as the corresponding [`BlockWeights`] fields. (The
/// two RMSNorm gain gradients are not included here; freeze the norms or handle
/// their small `d`-vectors separately.)
pub struct BlockGrads {
    /// `∂L/∂Wq` (`d×d`).
    pub dwq: GpuMatrix,
    /// `∂L/∂Wk` (`d×d`).
    pub dwk: GpuMatrix,
    /// `∂L/∂Wv` (`d×d`).
    pub dwv: GpuMatrix,
    /// `∂L/∂Wo` (`d×d`).
    pub dwo: GpuMatrix,
    /// `∂L/∂Wg` (`d×h`).
    pub dwg: GpuMatrix,
    /// `∂L/∂Wu` (`d×h`).
    pub dwu: GpuMatrix,
    /// `∂L/∂Wd` (`h×d`).
    pub dwd: GpuMatrix,
}

/// The resident weights of a full **tied-embedding decoder**: a shared
/// `vocab × d` embedding table (which is also the LM head), the `N`
/// transformer [`BlockWeights`], and a final `d`-length RMSNorm gain. Consumed
/// by [`GpuChain::model_forward_tied`].
pub struct ModelWeights<'a> {
    /// Token embedding table (`vocab × d`), tied to the output LM head.
    pub embedding: &'a GpuMatrix,
    /// The transformer blocks, applied in order.
    pub blocks: &'a [BlockWeights<'a>],
    /// Final pre-logits RMSNorm gain (`d`).
    pub final_norm: &'a GpuMatrix,
}

/// The resident weights of the full **GQA tied-embedding decoder** — the real
/// `scirust-sciagent` model: a shared `vocab × d` embedding (also the LM head),
/// `N` [`GqaBlockWeights`] (multi-head grouped-query attention + RoPE + SwiGLU),
/// and a final RMSNorm gain. Consumed by [`GpuChain::gqa_model_forward`] /
/// [`GpuChain::gqa_model_backward`].
pub struct GqaModelWeights<'a> {
    /// Token embedding table (`vocab × d`), tied to the output LM head.
    pub embedding: &'a GpuMatrix,
    /// The GQA transformer blocks, applied in order.
    pub blocks: &'a [GqaBlockWeights<'a>],
    /// Final pre-logits RMSNorm gain (`d`).
    pub final_norm: &'a GpuMatrix,
}

/// The gradients of a full [`GqaModelWeights`], produced by
/// [`GpuChain::gqa_model_backward`]: one `vocab × d` **tied** embedding gradient
/// (accumulating both the input-embedding and the output-head paths), one
/// [`GqaBlockGrads`] per block (in block order), and the final pre-logits RMSNorm
/// gain gradient.
pub struct GqaModelGrads {
    /// `∂L/∂E` for the tied embedding/LM-head table (`vocab × d`).
    pub d_embedding: GpuMatrix,
    /// Per-block weight gradients, in block order.
    pub blocks: Vec<GqaBlockGrads>,
    /// `∂L/∂final_norm` — the final pre-logits RMSNorm gain gradient (`1×d`).
    pub d_final_norm: GpuMatrix,
}

impl GpuChain {
    /// Acquire a GPU device. Returns `None` if no adapter is available.
    pub fn new() -> Option<Self> {
        WgpuContext::new().ok().map(|ctx| Self { ctx })
    }

    /// Name of the underlying adapter.
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }

    /// Upload a row-major `rows×cols` matrix; it stays resident in VRAM.
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> GpuMatrix {
        self.ctx.upload(data, rows, cols)
    }

    /// `C = A·B`, keeping the result resident (no download).
    pub fn matmul(&self, a: &GpuMatrix, b: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.gemm_resident(a, b, false, false)
    }

    /// `C = op(A)·op(B)` with optional transposes, result resident.
    pub fn matmul_t(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        transpose_a: bool,
        transpose_b: bool,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.gemm_resident(a, b, transpose_a, transpose_b)
    }

    /// Elementwise `a + b` (same shape), result resident.
    pub fn add(&self, a: &GpuMatrix, b: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.ew_resident(a, b, 0)
    }

    /// Elementwise `a * b` (same shape), result resident.
    pub fn mul(&self, a: &GpuMatrix, b: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.ew_resident(a, b, 1)
    }

    /// Elementwise `relu(a)`, result resident.
    pub fn relu(&self, a: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.ew_resident(a, a, 2)
    }

    /// SwiGLU gate `silu(gate) ⊙ up` (same shape), result resident — the
    /// nonlinearity of the SwiGLU MLP.
    pub fn swiglu(&self, gate: &GpuMatrix, up: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.ew_resident(gate, up, 3)
    }

    /// Row-wise RMSNorm `x / sqrt(mean(x²) + eps) · weight`, result resident.
    /// `weight` is a resident `cols`-length gain vector.
    pub fn rms_norm(
        &self,
        x: &GpuMatrix,
        weight: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.rms_norm_resident(x, weight, eps)
    }

    /// The SwiGLU MLP forward, **fully resident**:
    /// `(silu(x·W_gate) ⊙ (x·W_up)) · W_down`.
    ///
    /// `x` is `t×d`, `w_gate`/`w_up` are `d×h`, `w_down` is `h×d`; returns the
    /// `t×d` output. Every intermediate (`t×h` gate/up/activation) stays in
    /// VRAM — the on-device MLP the SML's transformer block will call. (Apply
    /// [`Self::rms_norm`] to `x` first for the pre-norm block.)
    pub fn swiglu_mlp(
        &self,
        x: &GpuMatrix,
        w_gate: &GpuMatrix,
        w_up: &GpuMatrix,
        w_down: &GpuMatrix,
    ) -> BackendResult<GpuMatrix> {
        let gate = self.matmul(x, w_gate)?; // x·W_gate  (t×h)
        let up = self.matmul(x, w_up)?; // x·W_up    (t×h)
        let act = self.swiglu(&gate, &up)?; // silu(gate)⊙up (t×h)
        self.matmul(&act, w_down) // ·W_down    (t×d)
    }

    /// Row-wise softmax of a resident matrix, result resident.
    pub fn softmax(&self, x: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.softmax_resident(x)
    }

    /// Scale a resident score matrix by `scale` and (optionally) apply the
    /// causal mask, result resident.
    pub fn scale_causal_mask(
        &self,
        x: &GpuMatrix,
        scale: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.scale_causal_mask_resident(x, scale, causal)
    }

    /// Single-head scaled dot-product attention, **fully resident**:
    /// `softmax( (Q·Kᵀ)/√d  [+ causal mask] ) · V`.
    ///
    /// `q` and `k` are `t×d` (queries/keys, `d` = head dim), `v` is `t×dv`;
    /// returns the `t×dv` context. The `t×t` score matrix never leaves VRAM
    /// between the score GEMM, the scale/mask, the softmax and the value GEMM —
    /// this is the on-device attention block the SML's forward pass will call.
    pub fn attention(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let scale = 1.0 / (q.cols() as f32).sqrt();
        let scores = self.matmul_t(q, k, false, true)?; // S = Q·Kᵀ   (t×t)
        let scaled = self.scale_causal_mask(&scores, scale, causal)?; // /√d [+ mask]
        let weights = self.softmax(&scaled)?; // row softmax (t×t)
        self.matmul(&weights, v) // context = W·V (t×dv)
    }

    /// Rotary position embedding of a resident `rows × dim` matrix (rows =
    /// positions, `dim` = head dim), result resident. Interleaved-pair rotation
    /// with `pos = (row mod seq_len) + offset` and `freqⱼ = theta^(-2j/dim)` —
    /// matching the sciagent model's RoPE. Apply to Q and K before the score
    /// GEMM. `dim` must be even.
    pub fn rope(
        &self,
        x: &GpuMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.rope_resident(x, seq_len, offset, theta)
    }

    /// Backward of [`Self::rope`]: the transpose rotation of the upstream grad
    /// `dy` (`rows × dim`, resident), result resident.
    pub fn rope_backward(
        &self,
        dy: &GpuMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.rope_backward_resident(dy, seq_len, offset, theta)
    }

    /// Gather columns `[col_start, col_start+ncols)` of a resident matrix into a
    /// resident `rows × ncols` matrix — e.g. one head's `d_head` slice of a
    /// full-width projection. Backward is [`Self::place_cols`].
    pub fn slice_cols(
        &self,
        x: &GpuMatrix,
        col_start: usize,
        ncols: usize,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.slice_cols_resident(x, col_start, ncols)
    }

    /// Scatter a resident narrow block into a zero-padded `rows × dst_cols`
    /// matrix at `col_start` — e.g. place a head's context back into its
    /// `d_model` slot before summing heads. Adjoint of [`Self::slice_cols`].
    pub fn place_cols(
        &self,
        x: &GpuMatrix,
        col_start: usize,
        dst_cols: usize,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.place_cols_resident(x, col_start, dst_cols)
    }

    /// **Resident multi-head grouped-query attention**, single sequence
    /// (`rows = seq_len`), matching the sciagent model's attention:
    ///
    /// ```text
    /// qr = rope(q)         kr = rope(k)                  # full width, per model
    /// for head in 0..n_heads:                            # kv = head / (n_heads/n_kv_heads)
    ///     ctx_head = attention( slice(qr,head), slice(kr,kv), slice(v,kv) )
    ///     out += place(ctx_head into its d_model slot)
    /// ```
    ///
    /// `q` is `t×(n_heads·dh)`, `k`/`v` are `t×(n_kv_heads·dh)`; returns the
    /// `t×(n_heads·dh)` concatenated context (the caller applies `w_o`). RoPE is
    /// applied to the *full-width* `q`/`k` exactly as `rope_on_tape` does — each
    /// uses its own width in the frequency schedule — so the result matches the
    /// CPU model. Every intermediate stays in VRAM. `v` is not rotated. This
    /// composes brick-17 RoPE, brick-18a slice/place and the single-head
    /// `attention`, all already gradient-checked.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        theta: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let d_model = q.cols();
        if n_heads == 0 || n_kv_heads == 0 || !d_model.is_multiple_of(n_heads)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention: q.cols {d_model} not divisible by n_heads {n_heads}"
            )));
        }
        let dh = d_model / n_heads;
        if !n_heads.is_multiple_of(n_kv_heads)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention: n_heads {n_heads} not a multiple of n_kv_heads {n_kv_heads}"
            )));
        }
        if k.cols() != n_kv_heads * dh || v.cols() != n_kv_heads * dh
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention: expected k/v cols = n_kv_heads·dh = {}, got k {}, v {}",
                n_kv_heads * dh,
                k.cols(),
                v.cols()
            )));
        }
        if q.rows() != seq_len || k.rows() != seq_len || v.rows() != seq_len
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention: single sequence only — rows must equal seq_len {seq_len} \
                 (q {}, k {}, v {})",
                q.rows(),
                k.rows(),
                v.rows()
            )));
        }

        let qr = self.rope(q, seq_len, 0, theta)?;
        let kr = self.rope(k, seq_len, 0, theta)?;
        let repeat = n_heads / n_kv_heads;
        let mut out: Option<GpuMatrix> = None;
        for head in 0..n_heads
        {
            let kv = head / repeat;
            let qs = self.slice_cols(&qr, head * dh, dh)?;
            let ks = self.slice_cols(&kr, kv * dh, dh)?;
            let vs = self.slice_cols(v, kv * dh, dh)?;
            let ctx = self.attention(&qs, &ks, &vs, causal)?; // scale = 1/√dh
            let padded = self.place_cols(&ctx, head * dh, d_model)?;
            out = Some(match out
            {
                None => padded,
                Some(acc) => self.add(&acc, &padded)?,
            });
        }
        // n_heads ≥ 1 is guaranteed above, so `out` is always Some.
        out.ok_or_else(|| BackendError::ShapeMismatch("gqa_attention: n_heads must be ≥ 1".into()))
    }

    /// Backward of [`Self::gqa_attention`]. Given the upstream grad `dout`
    /// (`t×(n_heads·dh)`, grad of the concatenated context), returns
    /// `(dq, dk, dv)` — grads of the pre-RoPE projections `q` (`t×(n_heads·dh)`)
    /// and `k`/`v` (`t×(n_kv_heads·dh)`), all resident.
    ///
    /// The per-head forward intermediates (`weights`) are recomputed here, then
    /// each head's single-head attention adjoint is taken (the same reverse chain
    /// as [`Self::transformer_block_backward`]). Because a grouped-query key/value
    /// head is shared by `repeat = n_heads/n_kv_heads` query heads, `dk`/`dv`
    /// **accumulate** over those heads (place + add), while `dq` slots are
    /// disjoint. Finally RoPE's adjoint maps `dqr→dq` and `dkr→dk` (each at its
    /// own width); `v` is not rotated so `dv` passes straight through.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_backward(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        dout: &GpuMatrix,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        theta: f32,
        causal: bool,
    ) -> BackendResult<(GpuMatrix, GpuMatrix, GpuMatrix)> {
        let d_model = q.cols();
        if n_heads == 0 || n_kv_heads == 0 || !d_model.is_multiple_of(n_heads)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention_backward: q.cols {d_model} not divisible by n_heads {n_heads}"
            )));
        }
        let dh = d_model / n_heads;
        if !n_heads.is_multiple_of(n_kv_heads)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention_backward: n_heads {n_heads} not a multiple of n_kv_heads {n_kv_heads}"
            )));
        }
        let kv_dim = n_kv_heads * dh;
        if k.cols() != kv_dim || v.cols() != kv_dim
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention_backward: expected k/v cols = {kv_dim}, got k {}, v {}",
                k.cols(),
                v.cols()
            )));
        }
        if q.rows() != seq_len
            || k.rows() != seq_len
            || v.rows() != seq_len
            || dout.rows() != seq_len
        {
            return Err(BackendError::ShapeMismatch(format!(
                "gqa_attention_backward: single sequence only — rows must equal seq_len {seq_len}"
            )));
        }

        let qr = self.rope(q, seq_len, 0, theta)?;
        let kr = self.rope(k, seq_len, 0, theta)?;
        let repeat = n_heads / n_kv_heads;
        let scale = 1.0 / (dh as f32).sqrt();

        // Grad accumulators for the (post-RoPE) qr/kr and the (un-rotated) v.
        let mut dqr: Option<GpuMatrix> = None; // disjoint per head
        let mut dkr: Option<GpuMatrix> = None; // shared kv heads accumulate
        let mut dvv: Option<GpuMatrix> = None;

        for head in 0..n_heads
        {
            let kv = head / repeat;
            let qs = self.slice_cols(&qr, head * dh, dh)?;
            let ks = self.slice_cols(&kr, kv * dh, dh)?;
            let vs = self.slice_cols(v, kv * dh, dh)?;

            // Recompute this head's forward softmax weights.
            let scores = self.matmul_t(&qs, &ks, false, true)?;
            let scaled = self.scale_causal_mask(&scores, scale, causal)?;
            let weights = self.softmax(&scaled)?;

            // Grad of this head's context = adjoint of place_cols = slice of dout.
            let d_ctx = self.slice_cols(dout, head * dh, dh)?;

            // Single-head attention adjoint (see transformer_block_backward).
            let dweights = self.matmul_t(&d_ctx, &vs, false, true)?; // d_ctx·vsᵀ
            let dvs = self.matmul_t(&weights, &d_ctx, true, false)?; // weightsᵀ·d_ctx
            let dscaled = self.softmax_backward(&weights, &dweights)?;
            let dscores = self.scale_causal_mask_backward(&dscaled, scale, causal)?;
            let dqs = self.matmul(&dscores, &ks)?; // dscores·ks
            let dks = self.matmul_t(&dscores, &qs, true, false)?; // dscoresᵀ·qs

            // Scatter each head's grads back to full width and accumulate.
            let dqs_full = self.place_cols(&dqs, head * dh, d_model)?;
            let dks_full = self.place_cols(&dks, kv * dh, kv_dim)?;
            let dvs_full = self.place_cols(&dvs, kv * dh, kv_dim)?;
            dqr = Some(match dqr
            {
                None => dqs_full,
                Some(acc) => self.add(&acc, &dqs_full)?,
            });
            dkr = Some(match dkr
            {
                None => dks_full,
                Some(acc) => self.add(&acc, &dks_full)?,
            });
            dvv = Some(match dvv
            {
                None => dvs_full,
                Some(acc) => self.add(&acc, &dvs_full)?,
            });
        }

        let dqr = dqr.expect("n_heads ≥ 1");
        let dkr = dkr.expect("n_heads ≥ 1");
        let dv = dvv.expect("n_heads ≥ 1");

        // RoPE adjoint: qr = rope(q), kr = rope(k); v was not rotated.
        let dq = self.rope_backward(&dqr, seq_len, 0, theta)?;
        let dk = self.rope_backward(&dkr, seq_len, 0, theta)?;
        Ok((dq, dk, dv))
    }

    /// A complete **pre-norm residual GQA transformer block**, fully resident —
    /// the real `scirust-sciagent` `SciAgentBlock`, on the GPU:
    ///
    /// ```text
    /// h   = x + gqa_attention( rms_norm(x)·Wq, ·Wk, ·Wv ) · Wo
    /// out = h + swiglu_mlp( rms_norm(h) )
    /// ```
    ///
    /// `x` is `t×d` (single sequence, `t = seq_len`); the block preserves that
    /// shape. Multi-head grouped-query attention with RoPE (via
    /// [`Self::gqa_attention`]) — `w.wq`/`w.wo` are `d×d`, `w.wk`/`w.wv` are
    /// `d×kv_dim`. Everything stays in VRAM. One whole layer of the 350M forward.
    pub fn gqa_transformer_block(
        &self,
        x: &GpuMatrix,
        w: &GqaBlockWeights,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let seq_len = x.rows();
        // Attention sub-block (pre-norm + residual).
        let xn = self.rms_norm(x, w.norm1, eps)?;
        let q = self.matmul(&xn, w.wq)?; // t×d
        let k = self.matmul(&xn, w.wk)?; // t×kv_dim
        let v = self.matmul(&xn, w.wv)?; // t×kv_dim
        let ctx = self.gqa_attention(
            &q,
            &k,
            &v,
            w.n_heads,
            w.n_kv_heads,
            seq_len,
            w.theta,
            causal,
        )?;
        let attn_out = self.matmul(&ctx, w.wo)?; // t×d
        let h = self.add(x, &attn_out)?;
        // MLP sub-block (pre-norm + residual).
        let hn = self.rms_norm(&h, w.norm2, eps)?;
        let mlp = self.swiglu_mlp(&hn, w.wg, w.wu, w.wd)?;
        self.add(&h, &mlp)
    }

    /// Backward of [`Self::gqa_transformer_block`], returning `dx` **and** all nine
    /// weight gradients ([`GqaBlockGrads`]: the seven projections plus the two
    /// RMSNorm gains `dnorm1`/`dnorm2`) — the GQA analogue of
    /// [`Self::transformer_block_backward_full`], so a GQA block can be trained
    /// on the device with an AdamW step. The forward activations are recomputed
    /// here (cheap resident ops); the attention adjoint goes through
    /// [`Self::gqa_attention_backward`] (multi-head, grouped-query, RoPE). The two
    /// RMSNorm gain gradients are emitted via [`Self::rms_norm_gain_backward`].
    pub fn gqa_transformer_block_backward_full(
        &self,
        x: &GpuMatrix,
        w: &GqaBlockWeights,
        dout: &GpuMatrix,
        eps: f32,
        causal: bool,
    ) -> BackendResult<(GpuMatrix, GqaBlockGrads)> {
        let seq_len = x.rows();
        // --- recompute the forward activations the backward contracts with ---
        let xn = self.rms_norm(x, w.norm1, eps)?;
        let q = self.matmul(&xn, w.wq)?;
        let k = self.matmul(&xn, w.wk)?;
        let v = self.matmul(&xn, w.wv)?;
        let ctx = self.gqa_attention(
            &q,
            &k,
            &v,
            w.n_heads,
            w.n_kv_heads,
            seq_len,
            w.theta,
            causal,
        )?;
        let h = self.add(x, &self.matmul(&ctx, w.wo)?)?;
        let hn = self.rms_norm(&h, w.norm2, eps)?;
        let gate = self.matmul(&hn, w.wg)?;
        let up = self.matmul(&hn, w.wu)?;
        let act = self.swiglu(&gate, &up)?;

        // --- MLP path ---
        let dact = self.matmul_t(dout, w.wd, false, true)?;
        let dwd = self.matmul_t(&act, dout, true, false)?; // actᵀ·dout   (h×d)
        let (dgate, dup) = self.swiglu_backward(&gate, &up, &dact)?;
        let dwg = self.matmul_t(&hn, &dgate, true, false)?; // hnᵀ·dgate  (d×h)
        let dwu = self.matmul_t(&hn, &dup, true, false)?; // hnᵀ·dup    (d×h)
        let dhn = self.add(
            &self.matmul_t(&dgate, w.wg, false, true)?,
            &self.matmul_t(&dup, w.wu, false, true)?,
        )?;
        // hn = rms_norm(h, norm2) ⇒ the norm2 gain grad accumulates dhn ⊙ (h/rms).
        let dnorm2 = self.rms_norm_gain_backward(&h, &dhn, eps)?;
        let dh = self.add(dout, &self.rms_norm_backward(&h, w.norm2, &dhn, eps)?)?;

        // --- attention path ---
        let dwo = self.matmul_t(&ctx, &dh, true, false)?; // ctxᵀ·dh     (d×d)
        let d_ctx = self.matmul_t(&dh, w.wo, false, true)?; // dh·Woᵀ     (t×d)
        let (dq, dk, dv) = self.gqa_attention_backward(
            &q,
            &k,
            &v,
            &d_ctx,
            w.n_heads,
            w.n_kv_heads,
            seq_len,
            w.theta,
            causal,
        )?;
        let dwq = self.matmul_t(&xn, &dq, true, false)?; // xnᵀ·dq  (d×d)
        let dwk = self.matmul_t(&xn, &dk, true, false)?; // xnᵀ·dk  (d×kv_dim)
        let dwv = self.matmul_t(&xn, &dv, true, false)?; // xnᵀ·dv  (d×kv_dim)
        let dxn = self.add(
            &self.add(
                &self.matmul_t(&dq, w.wq, false, true)?,
                &self.matmul_t(&dk, w.wk, false, true)?,
            )?,
            &self.matmul_t(&dv, w.wv, false, true)?,
        )?;
        // xn = rms_norm(x, norm1) ⇒ the norm1 gain grad accumulates dxn ⊙ (x/rms).
        let dnorm1 = self.rms_norm_gain_backward(x, &dxn, eps)?;
        let dx = self.add(&dh, &self.rms_norm_backward(x, w.norm1, &dxn, eps)?)?;

        Ok((
            dx,
            GqaBlockGrads {
                dwq,
                dwk,
                dwv,
                dwo,
                dwg,
                dwu,
                dwd,
                dnorm1,
                dnorm2,
            },
        ))
    }

    /// A complete **pre-norm residual transformer block**, fully resident:
    ///
    /// ```text
    /// h   = x + attention( rms_norm(x)·Wq, ·Wk, ·Wv ) · Wo
    /// out = h + swiglu_mlp( rms_norm(h) )
    /// ```
    ///
    /// `x` is `t×d`; the block preserves that shape. Single head (`d` = head
    /// dim). Everything — the two norms, the four attention projections, the
    /// score matrix, the two residual adds and the MLP — stays in VRAM from the
    /// input upload to the output download. This is **one full layer of the
    /// 350M forward** as a single resident call.
    pub fn transformer_block(
        &self,
        x: &GpuMatrix,
        w: &BlockWeights,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        // Attention sub-block (pre-norm + residual).
        let xn = self.rms_norm(x, w.norm1, eps)?;
        let q = self.matmul(&xn, w.wq)?;
        let k = self.matmul(&xn, w.wk)?;
        let v = self.matmul(&xn, w.wv)?;
        let attn = self.attention(&q, &k, &v, causal)?;
        let attn_out = self.matmul(&attn, w.wo)?;
        let h = self.add(x, &attn_out)?; // residual 1

        // MLP sub-block (pre-norm + residual).
        let hn = self.rms_norm(&h, w.norm2, eps)?;
        let mlp = self.swiglu_mlp(&hn, w.wg, w.wu, w.wd)?;
        self.add(&h, &mlp) // residual 2
    }

    /// [`Self::transformer_block`] that also returns a [`BlockCache`] of the
    /// forward activations the backward pass reads. Same math as the plain
    /// forward; use this when you intend to call
    /// [`Self::transformer_block_backward`].
    pub fn transformer_block_forward_cached(
        &self,
        x: &GpuMatrix,
        w: &BlockWeights,
        eps: f32,
        causal: bool,
    ) -> BackendResult<(GpuMatrix, BlockCache)> {
        let xn = self.rms_norm(x, w.norm1, eps)?;
        let q = self.matmul(&xn, w.wq)?;
        let k = self.matmul(&xn, w.wk)?;
        let v = self.matmul(&xn, w.wv)?;
        let scale = 1.0 / (q.cols() as f32).sqrt();
        let scaled = self.scale_causal_mask(&self.matmul_t(&q, &k, false, true)?, scale, causal)?;
        let weights = self.softmax(&scaled)?;
        let a = self.matmul(&weights, &v)?;
        let h = self.add(x, &self.matmul(&a, w.wo)?)?; // residual 1
        let hn = self.rms_norm(&h, w.norm2, eps)?;
        let gate = self.matmul(&hn, w.wg)?;
        let up = self.matmul(&hn, w.wu)?;
        let act = self.swiglu(&gate, &up)?;
        let out = self.add(&h, &self.matmul(&act, w.wd)?)?; // residual 2
        Ok((
            out,
            BlockCache {
                q,
                k,
                v,
                weights,
                h,
                gate,
                up,
            },
        ))
    }

    /// Backward of the transformer block — the **input gradient** `dx` given the
    /// upstream grad `dout`, the block weights, and the forward [`BlockCache`].
    /// Chains every adjoint (residual → MLP GEMMs → SwiGLU → norm2 → residual →
    /// Wo → attention: value/softmax/scale-mask/scores → QKV → norm1) in reverse.
    /// Gradients that reach `x` by two paths (the two residual skips) are summed.
    pub fn transformer_block_backward(
        &self,
        x: &GpuMatrix,
        w: &BlockWeights,
        cache: &BlockCache,
        dout: &GpuMatrix,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        // out = h + mlp  ⇒  dmlp = dout, and dh gets a residual contribution below.
        // mlp = act·Wd  ⇒  dact = dout·Wdᵀ
        let dact = self.matmul_t(dout, w.wd, false, true)?;
        // act = silu(gate)⊙up  ⇒  dgate, dup
        let (dgate, dup) = self.swiglu_backward(&cache.gate, &cache.up, &dact)?;
        // gate = hn·Wg, up = hn·Wu  ⇒  dhn = dgate·Wgᵀ + dup·Wuᵀ
        let dhn = self.add(
            &self.matmul_t(&dgate, w.wg, false, true)?,
            &self.matmul_t(&dup, w.wu, false, true)?,
        )?;
        // hn = rms_norm(h, norm2)  ⇒  dh (from MLP path)
        let dh_mlp = self.rms_norm_backward(&cache.h, w.norm2, &dhn, eps)?;
        // h feeds both the MLP norm and the residual add of out ⇒ dh = dout + dh_mlp
        let dh = self.add(dout, &dh_mlp)?;
        // h = x + a·Wo  ⇒  dao = dh ; da = dh·Woᵀ
        let da = self.matmul_t(&dh, w.wo, false, true)?;
        // a = weights·v  ⇒  dweights = da·vᵀ ; dv = weightsᵀ·da
        let dweights = self.matmul_t(&da, &cache.v, false, true)?;
        let dv = self.matmul_t(&cache.weights, &da, true, false)?;
        // weights = softmax(scaled)  ⇒  dscaled
        let dscaled = self.softmax_backward(&cache.weights, &dweights)?;
        // scaled = scale_mask(q·kᵀ)  ⇒  dscores
        let scale = 1.0 / (cache.q.cols() as f32).sqrt();
        let dscores = self.scale_causal_mask_backward(&dscaled, scale, causal)?;
        // scores = q·kᵀ  ⇒  dq = dscores·k ; dk = dscoresᵀ·q
        let dq = self.matmul(&dscores, &cache.k)?;
        let dk = self.matmul_t(&dscores, &cache.q, true, false)?;
        // q,k,v = xn·{Wq,Wk,Wv}  ⇒  dxn = dq·Wqᵀ + dk·Wkᵀ + dv·Wvᵀ
        let dxn = self.add(
            &self.add(
                &self.matmul_t(&dq, w.wq, false, true)?,
                &self.matmul_t(&dk, w.wk, false, true)?,
            )?,
            &self.matmul_t(&dv, w.wv, false, true)?,
        )?;
        // xn = rms_norm(x, norm1)  ⇒  dx (from attention path)
        let dx_attn = self.rms_norm_backward(x, w.norm1, &dxn, eps)?;
        // x feeds both norm1 and the residual add of h ⇒ dx = dh + dx_attn
        self.add(&dh, &dx_attn)
    }

    /// Like [`Self::transformer_block_backward`] but also returns the **projection
    /// weight gradients** ([`BlockGrads`]) alongside `dx`. The forward
    /// activations that the weight grads need but the cache doesn't hold (`xn`,
    /// `a`, `hn`, `act`) are recomputed here — cheap forward ops. Enables an
    /// AdamW update of every projection: a transformer block that trains on the
    /// GPU. (RMSNorm gain gradients are not produced; freeze the norms.)
    pub fn transformer_block_backward_full(
        &self,
        x: &GpuMatrix,
        w: &BlockWeights,
        cache: &BlockCache,
        dout: &GpuMatrix,
        eps: f32,
        causal: bool,
    ) -> BackendResult<(GpuMatrix, BlockGrads)> {
        // Recompute the forward activations the weight gradients contract with.
        let xn = self.rms_norm(x, w.norm1, eps)?;
        let a = self.matmul(&cache.weights, &cache.v)?; // attention output (t×d)
        let h = &cache.h;
        let hn = self.rms_norm(h, w.norm2, eps)?;
        let act = self.swiglu(&cache.gate, &cache.up)?;

        // --- MLP path ---
        let dact = self.matmul_t(dout, w.wd, false, true)?;
        let dwd = self.matmul_t(&act, dout, true, false)?; // actᵀ·dout   (h×d)
        let (dgate, dup) = self.swiglu_backward(&cache.gate, &cache.up, &dact)?;
        let dwg = self.matmul_t(&hn, &dgate, true, false)?; // hnᵀ·dgate  (d×h)
        let dwu = self.matmul_t(&hn, &dup, true, false)?; // hnᵀ·dup    (d×h)
        let dhn = self.add(
            &self.matmul_t(&dgate, w.wg, false, true)?,
            &self.matmul_t(&dup, w.wu, false, true)?,
        )?;
        let dh = self.add(dout, &self.rms_norm_backward(h, w.norm2, &dhn, eps)?)?;

        // --- attention path ---
        let dwo = self.matmul_t(&a, &dh, true, false)?; // aᵀ·dh      (d×d)
        let da = self.matmul_t(&dh, w.wo, false, true)?;
        let dweights = self.matmul_t(&da, &cache.v, false, true)?;
        let dv = self.matmul_t(&cache.weights, &da, true, false)?;
        let dscaled = self.softmax_backward(&cache.weights, &dweights)?;
        let scale = 1.0 / (cache.q.cols() as f32).sqrt();
        let dscores = self.scale_causal_mask_backward(&dscaled, scale, causal)?;
        let dq = self.matmul(&dscores, &cache.k)?;
        let dk = self.matmul_t(&dscores, &cache.q, true, false)?;
        let dwq = self.matmul_t(&xn, &dq, true, false)?; // xnᵀ·dq     (d×d)
        let dwk = self.matmul_t(&xn, &dk, true, false)?; // xnᵀ·dk     (d×d)
        let dwv = self.matmul_t(&xn, &dv, true, false)?; // xnᵀ·dv     (d×d)
        let dxn = self.add(
            &self.add(
                &self.matmul_t(&dq, w.wq, false, true)?,
                &self.matmul_t(&dk, w.wk, false, true)?,
            )?,
            &self.matmul_t(&dv, w.wv, false, true)?,
        )?;
        let dx = self.add(&dh, &self.rms_norm_backward(x, w.norm1, &dxn, eps)?)?;

        Ok((
            dx,
            BlockGrads {
                dwq,
                dwk,
                dwv,
                dwo,
                dwg,
                dwu,
                dwd,
            },
        ))
    }

    /// Apply a **stack of transformer blocks** in sequence, fully resident:
    /// block `i`'s output feeds block `i+1` without ever leaving VRAM. `x` is
    /// `t×d`; every `BlockWeights` must be `d`-consistent. Returns the `t×d`
    /// output of the last block — the resident trunk of the 350M forward
    /// (`N` layers). With no blocks it returns a copy of `x`.
    pub fn transformer_stack(
        &self,
        x: &GpuMatrix,
        blocks: &[BlockWeights],
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let mut cur: Option<GpuMatrix> = None;
        for b in blocks
        {
            let input = cur.as_ref().unwrap_or(x);
            cur = Some(self.transformer_block(input, b, eps, causal)?);
        }
        match cur
        {
            Some(m) => Ok(m),
            // No layers: identity. Round-trip to return an owned resident copy.
            None => Ok(self.upload(&self.download(x)?, x.rows(), x.cols())),
        }
    }

    /// Token embedding gather: build a resident `tokens.len() × d` matrix whose
    /// row `i` is row `tokens[i]` of the `vocab × d` `table`.
    pub fn embed(&self, tokens: &[u32], table: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.embed_resident(tokens, table)
    }

    /// The **full tied-embedding decoder forward**, `tokens → logits`, fully
    /// resident:
    ///
    /// ```text
    /// emb    = embed(tokens, E)              // t×d
    /// trunk  = transformer_stack(emb, blocks) // t×d
    /// logits = rms_norm(trunk, final) · Eᵀ    // t×vocab   (tied LM head)
    /// ```
    ///
    /// Returns the `t × vocab` logit matrix (`t = tokens.len()`). Nothing leaves
    /// VRAM between the embedding gather and the final logit GEMM — a whole
    /// 350M forward pass as one resident call.
    pub fn model_forward_tied(
        &self,
        tokens: &[u32],
        w: &ModelWeights,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let emb = self.embed(tokens, w.embedding)?;
        let trunk = self.transformer_stack(&emb, w.blocks, eps, causal)?;
        let normed = self.rms_norm(&trunk, w.final_norm, eps)?;
        // Tied head: logits = normed · Eᵀ  (E is vocab×d ⇒ Eᵀ is d×vocab).
        self.matmul_t(&normed, w.embedding, false, true)
    }

    /// Apply a **stack of GQA transformer blocks** in sequence, fully resident —
    /// the multi-head grouped-query analogue of [`Self::transformer_stack`].
    /// Returns the `t×d` output of the last block. With no blocks, a resident
    /// copy of `x`.
    pub fn gqa_transformer_stack(
        &self,
        x: &GpuMatrix,
        blocks: &[GqaBlockWeights],
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let mut cur: Option<GpuMatrix> = None;
        for b in blocks
        {
            let input = cur.as_ref().unwrap_or(x);
            cur = Some(self.gqa_transformer_block(input, b, eps, causal)?);
        }
        match cur
        {
            Some(m) => Ok(m),
            None => Ok(self.upload(&self.download(x)?, x.rows(), x.cols())),
        }
    }

    /// The **full GQA tied-embedding decoder forward**, `tokens → logits`, fully
    /// resident — the real `scirust-sciagent` model on the GPU:
    ///
    /// ```text
    /// emb    = embed(tokens, E)                    // t×d
    /// trunk  = gqa_transformer_stack(emb, blocks)  // t×d  (N GQA layers)
    /// logits = rms_norm(trunk, final) · Eᵀ         // t×vocab  (tied LM head)
    /// ```
    ///
    /// Returns the `t × vocab` logits (`t = tokens.len()`). Nothing leaves VRAM
    /// between the embedding gather and the final logit GEMM.
    pub fn gqa_model_forward(
        &self,
        tokens: &[u32],
        w: &GqaModelWeights,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let emb = self.embed(tokens, w.embedding)?;
        let trunk = self.gqa_transformer_stack(&emb, w.blocks, eps, causal)?;
        let normed = self.rms_norm(&trunk, w.final_norm, eps)?;
        self.matmul_t(&normed, w.embedding, false, true)
    }

    /// The **full GQA model backward**, given `dlogits = ∂L/∂logits` (`t×vocab`,
    /// e.g. from [`Self::cross_entropy_grad`]): returns the **tied** embedding
    /// gradient and every block's weight gradients ([`GqaModelGrads`]), all
    /// resident. The forward activations are recomputed here (the block-boundary
    /// inputs the backward contracts with).
    ///
    /// The embedding `E` is used twice — the input lookup **and** the output head
    /// `logits = normed·Eᵀ` — so `dE` accumulates both paths:
    /// `dE = dlogitsᵀ·normed  (head)  +  embed_backward(tokens, d_emb)  (lookup)`.
    /// The head also feeds `d_normed = dlogits·E`, which flows back through the
    /// final RMSNorm and the `N` blocks (reverse order), each via
    /// [`Self::gqa_transformer_block_backward_full`]. The final RMSNorm gain
    /// gradient is emitted via [`Self::rms_norm_gain_backward`].
    pub fn gqa_model_backward(
        &self,
        tokens: &[u32],
        w: &GqaModelWeights,
        dlogits: &GpuMatrix,
        eps: f32,
        causal: bool,
    ) -> BackendResult<GqaModelGrads> {
        // Recompute the block-boundary activations: xs[i] is the input to block i.
        let emb = self.embed(tokens, w.embedding)?;
        let mut xs = Vec::with_capacity(w.blocks.len() + 1);
        xs.push(emb);
        for b in w.blocks
        {
            let out = self.gqa_transformer_block(xs.last().unwrap(), b, eps, causal)?;
            xs.push(out);
        }
        let trunk = xs.last().unwrap();
        let normed = self.rms_norm(trunk, w.final_norm, eps)?;

        // Tied head: logits = normed · Eᵀ.
        let d_normed = self.matmul(dlogits, w.embedding)?; // dlogits · E     (t×d)
        let de_head = self.matmul_t(dlogits, &normed, true, false)?; // dlogitsᵀ·normed (vocab×d)

        // Final RMSNorm — its gain grad, then the input grad flowing to the trunk.
        let d_final_norm = self.rms_norm_gain_backward(trunk, &d_normed, eps)?;
        let mut d_cur = self.rms_norm_backward(trunk, w.final_norm, &d_normed, eps)?;
        let mut block_grads: Vec<GqaBlockGrads> = Vec::with_capacity(w.blocks.len());
        for i in (0..w.blocks.len()).rev()
        {
            let (dx, grads) = self.gqa_transformer_block_backward_full(
                &xs[i],
                &w.blocks[i],
                &d_cur,
                eps,
                causal,
            )?;
            d_cur = dx;
            block_grads.push(grads);
        }
        block_grads.reverse();

        // d_cur is now d(emb); add the embedding-lookup path into the tied grad.
        let vocab = w.embedding.rows();
        let de_embed = self.embed_backward(tokens, &d_cur, vocab)?;
        let d_embedding = self.add(&de_head, &de_embed)?;
        Ok(GqaModelGrads {
            d_embedding,
            blocks: block_grads,
            d_final_norm,
        })
    }

    // ---- Backward (vjp) primitives ----------------------------------------

    /// Backward of `C = A·B`: given `grad_c = ∂L/∂C`, return
    /// `(grad_a, grad_b)` with `grad_a = grad_c·Bᵀ` (`m×k`) and
    /// `grad_b = Aᵀ·grad_c` (`k×n`), both resident. `a` is `m×k`, `b` is `k×n`,
    /// `grad_c` is `m×n`. Every matmul in the backward pass reduces to these two
    /// transposed GEMMs — the foundation the rest of the backward builds on.
    pub fn matmul_backward(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        grad_c: &GpuMatrix,
    ) -> BackendResult<(GpuMatrix, GpuMatrix)> {
        let grad_a = self.matmul_t(grad_c, b, false, true)?; // grad_c · Bᵀ  (m×k)
        let grad_b = self.matmul_t(a, grad_c, true, false)?; // Aᵀ · grad_c  (k×n)
        Ok((grad_a, grad_b))
    }

    /// Backward of row-wise softmax: `dx = y ⊙ (dy − Σⱼ dyⱼyⱼ)`, given the
    /// forward output `y` and upstream grad `dy`. Result resident.
    pub fn softmax_backward(&self, y: &GpuMatrix, dy: &GpuMatrix) -> BackendResult<GpuMatrix> {
        self.ctx.softmax_backward_resident(y, dy)
    }

    /// Backward of the SwiGLU gate `c = silu(a) ⊙ b`: returns `(da, db)`
    /// resident, `da = dc·silu'(a)·b`, `db = dc·silu(a)`.
    pub fn swiglu_backward(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        dc: &GpuMatrix,
    ) -> BackendResult<(GpuMatrix, GpuMatrix)> {
        self.ctx.swiglu_backward_resident(a, b, dc)
    }

    /// Backward of RMSNorm (input gradient `dx`), given the input `x`, the gain
    /// `weight`, upstream grad `dy` and `eps`. Result resident.
    pub fn rms_norm_backward(
        &self,
        x: &GpuMatrix,
        weight: &GpuMatrix,
        dy: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.rms_norm_backward_resident(x, weight, dy, eps)
    }

    /// **Gain gradient** of RMSNorm: given the input `x` (`rows × d`) and the
    /// upstream grad `dy` (`rows × d`), returns `dweight` (`1 × d`) — the gradient
    /// of the per-channel gain. Since `y[r,j] = (x[r,j]/rms[r])·weight[j]`, the
    /// gain grad is `dweight[j] = Σ_r dy[r,j]·(x[r,j]/rms[r])`. Composed from
    /// existing resident ops (no new kernel): `normalized = rms_norm(x, ones)`
    /// gives `x/rms`, then a `ones[1×rows]·(dy ⊙ normalized)` GEMM sums over rows.
    /// The CPU contract is [`crate::ops::cpu_rms_norm_gain_backward`]. This is what
    /// lets the RMSNorm gains train on the resident path.
    pub fn rms_norm_gain_backward(
        &self,
        x: &GpuMatrix,
        dy: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        let (rows, d) = (x.rows(), x.cols());
        let ones_d = self.upload(&vec![1.0f32; d], 1, d);
        let normalized = self.rms_norm(x, &ones_d, eps)?; // x / rms  (rows×d)
        let prod = self.mul(dy, &normalized)?; // dy ⊙ (x/rms)
        let ones_row = self.upload(&vec![1.0f32; rows], 1, rows);
        self.matmul(&ones_row, &prod) // [1×rows]·[rows×d] = column sums (1×d)
    }

    /// Backward of scale + causal mask: `din = scale·dout` at kept positions,
    /// `0` above the diagonal. Result resident.
    pub fn scale_causal_mask_backward(
        &self,
        dout: &GpuMatrix,
        scale: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        self.ctx
            .scale_causal_mask_backward_resident(dout, scale, causal)
    }

    /// Backward of the embedding gather: accumulate `dout` (`t×d`) into a
    /// resident `vocab × d` table gradient (row `v` = sum of `dout` rows whose
    /// token is `v`). Deterministic, no atomics. Result resident.
    pub fn embed_backward(
        &self,
        tokens: &[u32],
        dout: &GpuMatrix,
        vocab: usize,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.embed_backward_resident(tokens, dout, vocab)
    }

    /// Cross-entropy loss gradient w.r.t. the `logits` (`t × vocab`) for the
    /// per-row `targets`: `dlogits = (softmax(logits) − onehot(target)) / t`,
    /// resident. The seed of the training backward — feed it as the upstream
    /// grad of the LM head.
    pub fn cross_entropy_grad(
        &self,
        logits: &GpuMatrix,
        targets: &[u32],
    ) -> BackendResult<GpuMatrix> {
        self.ctx.cross_entropy_grad_resident(logits, targets)
    }

    /// One SGD parameter update `param − lr·grad`, resident. Feed the result
    /// back as the next iteration's parameter — the optimizer step that closes
    /// the on-device training loop.
    pub fn sgd_step(
        &self,
        param: &GpuMatrix,
        grad: &GpuMatrix,
        lr: f32,
    ) -> BackendResult<GpuMatrix> {
        self.ctx.sgd_step_resident(param, grad, lr)
    }

    /// One AdamW step at optimizer step `step` (1-based), updating `param`, `m`
    /// and `v` **in place**. Start `m`/`v` at zero and reuse them across steps.
    /// Decoupled weight decay; bias-corrected. Nothing to return — the resident
    /// `param` buffer is updated on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &self,
        param: &GpuMatrix,
        grad: &GpuMatrix,
        m: &GpuMatrix,
        v: &GpuMatrix,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> BackendResult<()> {
        self.ctx
            .adamw_step_resident(param, grad, m, v, lr, betas, eps, weight_decay, step)
    }

    /// Download a resident matrix back to a CPU `Vec<f32>` (row-major).
    pub fn download(&self, mat: &GpuMatrix) -> BackendResult<Vec<f32>> {
        self.ctx.download(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, RawComputeBackend};

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-30);
        num / den
    }

    /// CPU reference for one pre-norm residual transformer block, used as the
    /// oracle for both the single-block and the stack tests. Weights match
    /// [`BlockWeights`]: `n1`/`n2` are `d`, `wq..wo` are `d×d`, `wg`/`wu` are
    /// `d×h`, `wd` is `h×d`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_block(
        x: &[f32],
        (n1, n2): (&[f32], &[f32]),
        (wq, wk, wv, wo): (&[f32], &[f32], &[f32], &[f32]),
        (wg, wu, wd): (&[f32], &[f32], &[f32]),
        (t, d, h): (usize, usize, usize),
        eps: f32,
        causal: bool,
    ) -> Vec<f32> {
        use crate::ops::{cpu_rms_norm, cpu_scale_causal_mask, cpu_softmax};
        let gemm = |a: &[f32], b: &[f32], m, k, n| CpuBackend.gemm_f32(a, b, m, k, n).unwrap();
        let transpose = |a: &[f32], r: usize, c: usize| {
            let mut o = vec![0.0f32; r * c];
            for i in 0..r
            {
                for j in 0..c
                {
                    o[j * r + i] = a[i * c + j];
                }
            }
            o
        };
        let xn = cpu_rms_norm(x, n1, eps, t, d);
        let q = gemm(&xn, wq, t, d, d);
        let k = gemm(&xn, wk, t, d, d);
        let v = gemm(&xn, wv, t, d, d);
        let s = cpu_scale_causal_mask(
            &gemm(&q, &transpose(&k, t, d), t, d, t),
            t,
            t,
            1.0 / (d as f32).sqrt(),
            causal,
        );
        let a = gemm(&cpu_softmax(&s, t, t), &v, t, t, d);
        let ao = gemm(&a, wo, t, d, d);
        let hh: Vec<f32> = x.iter().zip(&ao).map(|(a, b)| a + b).collect();
        let hn = cpu_rms_norm(&hh, n2, eps, t, d);
        let gate = gemm(&hn, wg, t, d, h);
        let up = gemm(&hn, wu, t, d, h);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let m = gemm(&act, wd, t, h, d);
        hh.iter().zip(&m).map(|(a, b)| a + b).collect()
    }

    /// A two-GEMM chain `(A·B)·C` keeps the intermediate `T = A·B` in VRAM and
    /// feeds it straight into the second matmul — only the final result is
    /// downloaded. Must match the CPU oracle. Skips if no adapter.
    #[test]
    fn resident_chain_keeps_intermediate_in_vram() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // A: 2×3, B: 3×2, C: 2×4.
        let a: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 * 0.4 + 0.5).cos()).collect();
        let c: Vec<f32> = (0..8).map(|i| (i as f32 * 0.2 - 0.7).sin()).collect();

        let ga = chain.upload(&a, 2, 3);
        let gb = chain.upload(&b, 3, 2);
        let gc = chain.upload(&c, 2, 4);

        let gt = chain.matmul(&ga, &gb).unwrap(); // T = A·B, resident 2×2
        assert_eq!((gt.rows(), gt.cols()), (2, 2));
        // gt is consumed by the next matmul WITHOUT ever being downloaded.
        let gout = chain.matmul(&gt, &gc).unwrap(); // OUT = T·C, resident 2×4
        assert_eq!((gout.rows(), gout.cols()), (2, 4));
        let out = chain.download(&gout).unwrap();

        // CPU oracle: (A·B)·C.
        let t = CpuBackend.gemm_f32(&a, &b, 2, 3, 2).unwrap();
        let expected = CpuBackend.gemm_f32(&t, &c, 2, 2, 4).unwrap();
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// Resident transpose path: `Aᵀ·B` matches the CPU oracle.
    #[test]
    fn resident_transpose() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // stored a is 3×2 (= op(A)ᵀ, op(A) is 2×3); b is 3×4 → op(A)ᵀ?? use ta.
        // op(A) = aᵀ is 2×3 (a stored 3×2), op(B)=b 3×4 → C 2×4. Wait k must match:
        // op(A) m×k = 2×3, op(B) k×n = 3×4. a stored k×m = 3×2, b stored 3×4.
        let a: Vec<f32> = (0..6).map(|i| i as f32 - 3.0).collect(); // 3×2
        let b: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect(); // 3×4
        let ga = chain.upload(&a, 3, 2);
        let gb = chain.upload(&b, 3, 4);
        let gout = chain.matmul_t(&ga, &gb, true, false).unwrap();
        assert_eq!((gout.rows(), gout.cols()), (2, 4));
        let out = chain.download(&gout).unwrap();

        // CPU oracle: op(A)=aᵀ (2×3) · b (3×4). Build aᵀ then gemm.
        let mut at = vec![0.0f32; 6];
        for i in 0..2
        {
            for q in 0..3
            {
                at[i * 3 + q] = a[q * 2 + i];
            }
        }
        let expected = CpuBackend.gemm_f32(&at, &b, 2, 3, 4).unwrap();
        assert!(rel_err(&out, &expected) < 1e-4);
    }

    /// Degenerate dimensions must not panic (wgpu rejects zero-sized buffers):
    /// `k == 0` yields an all-zeros result, `m == 0` yields an empty matrix.
    #[test]
    fn resident_degenerate_dims_are_handled() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // k == 0: (2×0)·(0×3) → 2×3 of zeros.
        let a = chain.upload(&[], 2, 0);
        let b = chain.upload(&[], 0, 3);
        let c = chain.matmul(&a, &b).unwrap();
        assert_eq!((c.rows(), c.cols()), (2, 3));
        assert_eq!(chain.download(&c).unwrap(), vec![0.0; 6]);

        // m == 0: (0×2)·(2×3) → 0×3, an empty download.
        let e = chain.upload(&[], 0, 2);
        let f = chain.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let g = chain.matmul(&e, &f).unwrap();
        assert_eq!((g.rows(), g.cols()), (0, 3));
        assert!(chain.download(&g).unwrap().is_empty());
    }

    /// A full resident layer chain GEMM → +bias → ReLU stays in VRAM and
    /// matches the CPU oracle on lavapipe.
    #[test]
    fn resident_layer_chain_gemm_bias_relu() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // X(2×3) · W(3×2) = H(2×2); H + B; relu. All resident.
        let x = vec![0.5, -0.4, 0.3, -0.2, 0.6, 0.1];
        let w = vec![0.2, -0.5, 0.4, 0.1, -0.3, 0.7];
        let bias = vec![-0.6, 0.05, -0.6, 0.05]; // 2×2, pushes some cells < 0

        let gx = chain.upload(&x, 2, 3);
        let gw = chain.upload(&w, 3, 2);
        let gb = chain.upload(&bias, 2, 2);
        let h = chain.matmul(&gx, &gw).unwrap();
        let hb = chain.add(&h, &gb).unwrap();
        let out = chain.download(&chain.relu(&hb).unwrap()).unwrap();

        // CPU oracle.
        let cpu_h = CpuBackend.gemm_f32(&x, &w, 2, 3, 2).unwrap();
        let expected: Vec<f32> = cpu_h
            .iter()
            .zip(&bias)
            .map(|(h, b)| (h + b).max(0.0))
            .collect();
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
        // ReLU actually clamped something (so the test is meaningful).
        assert!(expected.contains(&0.0));
    }

    /// The **fully resident** single-head attention block —
    /// `softmax((Q·Kᵀ)/√d + causal mask)·V` with the `t×t` scores never leaving
    /// VRAM — must match a step-by-step CPU oracle. Skips if no adapter; asserts
    /// on lavapipe (CI) and a real GPU (Thor). Also checks causality: no query
    /// attends to a future key, so masking a future V row leaves output intact.
    #[test]
    fn resident_attention_matches_cpu_oracle() {
        use crate::ops::{cpu_scale_causal_mask, cpu_softmax};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, dv) = (6usize, 8usize, 5usize);
        let q: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.13 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.09 + 0.4).cos()).collect();
        let v: Vec<f32> = (0..t * dv).map(|i| (i as f32 * 0.17 - 0.6).sin()).collect();

        // GPU: fully resident forward.
        let gq = chain.upload(&q, t, d);
        let gk = chain.upload(&k, t, d);
        let gv = chain.upload(&v, t, dv);
        let gout = chain.attention(&gq, &gk, &gv, true).unwrap();
        assert_eq!((gout.rows(), gout.cols()), (t, dv));
        let out = chain.download(&gout).unwrap();

        // CPU oracle, step by step: S = Q·Kᵀ → /√d + mask → softmax → ·V.
        let mut kt = vec![0.0f32; d * t];
        for r in 0..t
        {
            for c in 0..d
            {
                kt[c * t + r] = k[r * d + c];
            }
        }
        let s = CpuBackend.gemm_f32(&q, &kt, t, d, t).unwrap();
        let s = cpu_scale_causal_mask(&s, t, t, 1.0 / (d as f32).sqrt(), true);
        let w = cpu_softmax(&s, t, t);
        let expected = CpuBackend.gemm_f32(&w, &v, t, t, dv).unwrap();

        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// Resident row-wise RMSNorm matches the CPU oracle. Skips if no adapter.
    #[test]
    fn resident_rms_norm_matches_cpu_oracle() {
        use crate::ops::cpu_rms_norm;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, cols) = (4usize, 6usize);
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.23 - 1.5).sin() * 2.0)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.5 + 0.1 * i as f32).collect();

        let gx = chain.upload(&x, rows, cols);
        let gw = chain.upload(&w, 1, cols);
        let out = chain
            .download(&chain.rms_norm(&gx, &gw, eps).unwrap())
            .unwrap();

        let expected = cpu_rms_norm(&x, &w, eps, rows, cols);
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// The resident RoPE forward matches the CPU oracle (the model's exact
    /// interleaved rotation). Rows are token positions; here 6 rows span two
    /// sequences of `seq_len = 3`, so the `pos = row mod seq_len` restart is
    /// exercised. Skips if no adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn resident_rope_matches_cpu_oracle() {
        use crate::ops::cpu_rope;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, dim, seq_len, offset, theta) = (6usize, 8usize, 3usize, 0usize, 10_000.0f32);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.19 - 1.1).sin() * 1.7)
            .collect();

        let gx = chain.upload(&x, rows, dim);
        let out = chain
            .download(&chain.rope(&gx, seq_len, offset, theta).unwrap())
            .unwrap();

        let expected = cpu_rope(&x, rows, dim, seq_len, offset, theta);
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// RoPE backward must match numerical gradients. RoPE is linear, so for
    /// `L = Σ rope(X)⊙G` the input gradient is `dx = rope_backward(G)` (the
    /// transpose rotation applied to `G`); checked against the CPU adjoint and
    /// against central finite differences of `L` over `X`. `offset = 1` and a
    /// mid-range `seq_len` exercise the position arithmetic. Skips if no adapter.
    #[test]
    fn rope_backward_matches_finite_differences() {
        use crate::ops::{cpu_rope, cpu_rope_backward};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, dim, seq_len, offset, theta) = (4usize, 6usize, 4usize, 1usize, 10_000.0f32);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.27 - 0.9).sin() * 1.3)
            .collect();
        let g: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.4 + 0.15).cos())
            .collect(); // dL/dY

        let gg = chain.upload(&g, rows, dim);
        let dx_gpu = chain
            .download(&chain.rope_backward(&gg, seq_len, offset, theta).unwrap())
            .unwrap();
        // GPU must match the CPU adjoint formula (same arithmetic).
        let dx_cpu = cpu_rope_backward(&g, rows, dim, seq_len, offset, theta);
        assert!(rel_err(&dx_gpu, &dx_cpu) < 1e-4);

        // Gold standard: central finite differences of L = Σ rope(X)⊙G.
        let loss = |xx: &[f32]| -> f32 {
            cpu_rope(xx, rows, dim, seq_len, offset, theta)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..rows * dim
        {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += eps;
            xm[idx] -= eps;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * eps);
            assert!(
                (fd - dx_gpu[idx]).abs() < 1e-2,
                "dx[{idx}]: fd={fd} gpu={}",
                dx_gpu[idx]
            );
        }
    }

    /// Column slice/place are pure copies, so the GPU must be **bit-exact** to
    /// the CPU oracle (no arithmetic → no accumulation-order slack). Slices a
    /// column block out, then scatters it back into a wider zero-padded matrix.
    /// Skips if no adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn resident_slice_place_cols_match_cpu_oracle() {
        use crate::ops::{cpu_place_cols, cpu_slice_cols};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, src_cols) = (5usize, 12usize);
        let (col_start, ncols) = (4usize, 3usize); // a "head" at columns [4, 7)
        let x: Vec<f32> = (0..rows * src_cols)
            .map(|i| (i as f32 * 0.17 - 1.0).sin())
            .collect();

        let gx = chain.upload(&x, rows, src_cols);
        let sliced = chain
            .download(&chain.slice_cols(&gx, col_start, ncols).unwrap())
            .unwrap();
        assert_eq!(sliced, cpu_slice_cols(&x, rows, src_cols, col_start, ncols));

        let gs = chain.upload(&sliced, rows, ncols);
        let placed = chain
            .download(&chain.place_cols(&gs, col_start, src_cols).unwrap())
            .unwrap();
        assert_eq!(
            placed,
            cpu_place_cols(&sliced, rows, ncols, col_start, src_cols)
        );
    }

    /// `slice_cols` backward: RoPE-style gradient check. For `L = Σ slice(X)⊙G`
    /// the input gradient is `dx = place_cols(G)` — the adjoint scatter — checked
    /// bit-exactly against the CPU adjoint AND against central finite differences
    /// of `L` over `X`. Confirms `place_cols` really is `slice_cols`'s adjoint.
    #[test]
    fn slice_cols_backward_matches_finite_differences() {
        use crate::ops::{cpu_place_cols, cpu_slice_cols};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, src_cols) = (4usize, 10usize);
        let (col_start, ncols) = (3usize, 4usize);
        let x: Vec<f32> = (0..rows * src_cols)
            .map(|i| (i as f32 * 0.3 - 0.7).sin())
            .collect();
        let g: Vec<f32> = (0..rows * ncols)
            .map(|i| (i as f32 * 0.45 + 0.2).cos())
            .collect(); // dL/dY

        // Analytic adjoint on the GPU: dx = place_cols(G).
        let gg = chain.upload(&g, rows, ncols);
        let dx_gpu = chain
            .download(&chain.place_cols(&gg, col_start, src_cols).unwrap())
            .unwrap();
        assert_eq!(dx_gpu, cpu_place_cols(&g, rows, ncols, col_start, src_cols));

        // Gold standard: central finite differences of L = Σ slice(X)⊙G.
        let loss = |xx: &[f32]| -> f32 {
            cpu_slice_cols(xx, rows, src_cols, col_start, ncols)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..rows * src_cols
        {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += eps;
            xm[idx] -= eps;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * eps);
            assert!(
                (fd - dx_gpu[idx]).abs() < 1e-2,
                "dx[{idx}]: fd={fd} gpu={}",
                dx_gpu[idx]
            );
        }
    }

    /// **Resident multi-head GQA attention** matches the CPU oracle (the model's
    /// exact math): full-width RoPE on q/k, grouped-query head mapping
    /// (`4 heads / 2 kv-heads`), causal, heads concatenated. Skips if no adapter;
    /// asserts on lavapipe / a real GPU.
    #[test]
    fn resident_gqa_attention_matches_cpu_oracle() {
        use crate::ops::cpu_gqa_attention;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (n_heads, n_kv_heads, dh, seq_len) = (4usize, 2usize, 8usize, 6usize);
        let d_model = n_heads * dh; // 32
        let kv_dim = n_kv_heads * dh; // 16
        let theta = 10_000.0f32;
        let q: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.11 - 1.0).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| (i as f32 * 0.07 + 0.3).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| (i as f32 * 0.05 - 0.2).sin() * 0.5)
            .collect();

        let gq = chain.upload(&q, seq_len, d_model);
        let gk = chain.upload(&k, seq_len, kv_dim);
        let gv = chain.upload(&v, seq_len, kv_dim);
        let out = chain
            .download(
                &chain
                    .gqa_attention(&gq, &gk, &gv, n_heads, n_kv_heads, seq_len, theta, true)
                    .unwrap(),
            )
            .unwrap();

        let expected = cpu_gqa_attention(
            &q, &k, &v, seq_len, n_heads, n_kv_heads, dh, seq_len, theta, true,
        );
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "rel_err {}",
            rel_err(&out, &expected)
        );
    }

    /// **Multi-head GQA attention backward** gradient-checked end-to-end. For
    /// `L = Σ gqa_attention(Q,K,V) ⊙ G`, `(dQ,dK,dV) = gqa_attention_backward(…,G)`;
    /// each is checked against central finite differences of `L` over that input,
    /// via the CPU oracle. Uses `2 heads / 1 kv-head` so the two query heads share
    /// the single key/value head — this specifically exercises the `dK`/`dV`
    /// accumulation across grouped-query heads. Causal, RoPE on. Skips if no
    /// adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn gqa_attention_backward_matches_finite_differences() {
        use crate::ops::cpu_gqa_attention;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (n_heads, n_kv_heads, dh, seq_len) = (2usize, 1usize, 4usize, 4usize);
        let d_model = n_heads * dh; // 8
        let kv_dim = n_kv_heads * dh; // 4
        let theta = 10_000.0f32;
        let q: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.13 - 0.8).sin() * 0.5)
            .collect();
        let k: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| (i as f32 * 0.09 + 0.4).cos() * 0.5)
            .collect();
        let v: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| (i as f32 * 0.06 - 0.1).sin() * 0.5)
            .collect();
        let g: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.21 + 0.2).cos())
            .collect(); // dL/dOut

        let gq = chain.upload(&q, seq_len, d_model);
        let gk = chain.upload(&k, seq_len, kv_dim);
        let gv = chain.upload(&v, seq_len, kv_dim);
        let gg = chain.upload(&g, seq_len, d_model);
        let (dq_g, dk_g, dv_g) = chain
            .gqa_attention_backward(
                &gq, &gk, &gv, &gg, n_heads, n_kv_heads, seq_len, theta, true,
            )
            .unwrap();
        let dq = chain.download(&dq_g).unwrap();
        let dk = chain.download(&dk_g).unwrap();
        let dv = chain.download(&dv_g).unwrap();

        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            cpu_gqa_attention(
                qq, kk, vv, seq_len, n_heads, n_kv_heads, dh, seq_len, theta, true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..q.len()
        {
            let (mut p, mut m) = (q.clone(), q.clone());
            p[idx] += eps;
            m[idx] -= eps;
            let fd = (loss(&p, &k, &v) - loss(&m, &k, &v)) / (2.0 * eps);
            assert!(
                (fd - dq[idx]).abs() < 2e-2,
                "dq[{idx}]: fd={fd} gpu={}",
                dq[idx]
            );
        }
        for idx in 0..k.len()
        {
            let (mut p, mut m) = (k.clone(), k.clone());
            p[idx] += eps;
            m[idx] -= eps;
            let fd = (loss(&q, &p, &v) - loss(&q, &m, &v)) / (2.0 * eps);
            assert!(
                (fd - dk[idx]).abs() < 2e-2,
                "dk[{idx}]: fd={fd} gpu={}",
                dk[idx]
            );
        }
        for idx in 0..v.len()
        {
            let (mut p, mut m) = (v.clone(), v.clone());
            p[idx] += eps;
            m[idx] -= eps;
            let fd = (loss(&q, &k, &p) - loss(&q, &k, &m)) / (2.0 * eps);
            assert!(
                (fd - dv[idx]).abs() < 2e-2,
                "dv[{idx}]: fd={fd} gpu={}",
                dv[idx]
            );
        }
    }

    /// The **resident GQA transformer block** — the real `SciAgentBlock`
    /// (pre-norm multi-head grouped-query attention with RoPE + residual, then
    /// pre-norm SwiGLU MLP + residual) — matches a step-by-step CPU oracle.
    /// `4 heads / 2 kv-heads`, causal. Skips if no adapter; asserts on lavapipe /
    /// a real GPU. This is one whole 350M layer forward on the resident path.
    #[test]
    fn resident_gqa_transformer_block_matches_cpu_oracle() {
        use crate::ops::cpu_gqa_transformer_block;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, n_heads, n_kv_heads, dh, ff) = (6usize, 4usize, 2usize, 4usize, 16usize);
        let d = n_heads * dh; // 16
        let kv_dim = n_kv_heads * dh; // 8
        let (theta, eps) = (10_000.0f32, 1e-5f32);
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.031 + phase).sin() * amp)
                .collect()
        };
        let x = gen(t * d, 0.0, 1.0);
        let n1: Vec<f32> = (0..d).map(|i| 0.7 + 0.03 * i as f32).collect();
        let wq = gen(d * d, 0.5, 0.3);
        let wk = gen(d * kv_dim, 1.1, 0.3);
        let wv = gen(d * kv_dim, 1.7, 0.3);
        let wo = gen(d * d, 2.3, 0.3);
        let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.02 * i as f32).collect();
        let wg = gen(d * ff, 2.9, 0.25);
        let wu = gen(d * ff, 3.5, 0.25);
        let wd = gen(ff * d, 4.1, 0.25);

        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let gx = up(&x, t, d);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        let (gwq, gwk, gwv, gwo) = (
            up(&wq, d, d),
            up(&wk, d, kv_dim),
            up(&wv, d, kv_dim),
            up(&wo, d, d),
        );
        let (gwg, gwu, gwd) = (up(&wg, d, ff), up(&wu, d, ff), up(&wd, ff, d));
        let weights = GqaBlockWeights {
            norm1: &gn1,
            wq: &gwq,
            wk: &gwk,
            wv: &gwv,
            wo: &gwo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
            n_heads,
            n_kv_heads,
            theta,
        };
        let out = chain
            .download(
                &chain
                    .gqa_transformer_block(&gx, &weights, eps, true)
                    .unwrap(),
            )
            .unwrap();

        let expected = cpu_gqa_transformer_block(
            &x, &n1, &wq, &wk, &wv, &wo, &n2, &wg, &wu, &wd, t, d, kv_dim, ff, n_heads, n_kv_heads,
            dh, theta, eps, true,
        );
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "rel_err {}",
            rel_err(&out, &expected)
        );
    }

    /// The **GQA transformer block backward** gradient-checked end-to-end: `dx`
    /// and all nine weight gradients (seven projections + the two RMSNorm gains
    /// `dnorm1`/`dnorm2`) checked against central finite differences of
    /// `L = Σ gqa_transformer_block(x; W) ⊙ G` via the CPU oracle.
    /// `2 heads / 1 kv-head` so the query heads share the single kv head — the
    /// weight grads `dWk`/`dWv` then flow through the grouped-query accumulation.
    /// Skips if no adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn gqa_transformer_block_backward_matches_finite_differences() {
        use crate::ops::cpu_gqa_transformer_block;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, n_heads, n_kv_heads, dh, ff) = (4usize, 2usize, 1usize, 2usize, 6usize);
        let d = n_heads * dh; // 4
        let kv_dim = n_kv_heads * dh; // 2
        let (theta, eps) = (10_000.0f32, 1e-5f32);
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.037 + phase).sin() * amp)
                .collect()
        };
        let x = gen(t * d, 0.0, 0.7);
        let n1: Vec<f32> = (0..d).map(|i| 0.8 + 0.05 * i as f32).collect();
        let wq = gen(d * d, 0.5, 0.4);
        let wk = gen(d * kv_dim, 1.1, 0.4);
        let wv = gen(d * kv_dim, 1.7, 0.4);
        let wo = gen(d * d, 2.3, 0.4);
        let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.03 * i as f32).collect();
        let wg = gen(d * ff, 2.9, 0.3);
        let wu = gen(d * ff, 3.5, 0.3);
        let wd = gen(ff * d, 4.1, 0.3);
        let g = gen(t * d, 5.0, 1.0); // dL/dOut

        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let gx = up(&x, t, d);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        let (gwq, gwk, gwv, gwo) = (
            up(&wq, d, d),
            up(&wk, d, kv_dim),
            up(&wv, d, kv_dim),
            up(&wo, d, d),
        );
        let (gwg, gwu, gwd) = (up(&wg, d, ff), up(&wu, d, ff), up(&wd, ff, d));
        let gg = up(&g, t, d);
        let weights = GqaBlockWeights {
            norm1: &gn1,
            wq: &gwq,
            wk: &gwk,
            wv: &gwv,
            wo: &gwo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
            n_heads,
            n_kv_heads,
            theta,
        };
        let (dx_g, grads) = chain
            .gqa_transformer_block_backward_full(&gx, &weights, &gg, eps, true)
            .unwrap();
        let dl = |m: &GpuMatrix| chain.download(m).unwrap();
        let (dx, dwq, dwk, dwv, dwo, dwg, dwu, dwd) = (
            dl(&dx_g),
            dl(&grads.dwq),
            dl(&grads.dwk),
            dl(&grads.dwv),
            dl(&grads.dwo),
            dl(&grads.dwg),
            dl(&grads.dwu),
            dl(&grads.dwd),
        );

        // L = Σ block(x; W) ⊙ G, on the CPU oracle.
        #[allow(clippy::too_many_arguments)]
        let loss = |x_: &[f32],
                    wq_: &[f32],
                    wk_: &[f32],
                    wv_: &[f32],
                    wo_: &[f32],
                    wg_: &[f32],
                    wu_: &[f32],
                    wd_: &[f32]|
         -> f32 {
            cpu_gqa_transformer_block(
                x_, &n1, wq_, wk_, wv_, wo_, &n2, wg_, wu_, wd_, t, d, kv_dim, ff, n_heads,
                n_kv_heads, dh, theta, eps, true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        let hh = 1e-3f32;
        let check = |name: &str, analytic: &[f32], base: &[f32], f: &dyn Fn(&[f32]) -> f32| {
            for idx in 0..base.len()
            {
                let (mut p, mut m) = (base.to_vec(), base.to_vec());
                p[idx] += hh;
                m[idx] -= hh;
                let fd = (f(&p) - f(&m)) / (2.0 * hh);
                assert!(
                    (fd - analytic[idx]).abs() < 2e-2,
                    "{name}[{idx}]: fd={fd} gpu={}",
                    analytic[idx]
                );
            }
        };
        check("dx", &dx, &x, &|z| {
            loss(z, &wq, &wk, &wv, &wo, &wg, &wu, &wd)
        });
        check("dwq", &dwq, &wq, &|z| {
            loss(&x, z, &wk, &wv, &wo, &wg, &wu, &wd)
        });
        check("dwk", &dwk, &wk, &|z| {
            loss(&x, &wq, z, &wv, &wo, &wg, &wu, &wd)
        });
        check("dwv", &dwv, &wv, &|z| {
            loss(&x, &wq, &wk, z, &wo, &wg, &wu, &wd)
        });
        check("dwo", &dwo, &wo, &|z| {
            loss(&x, &wq, &wk, &wv, z, &wg, &wu, &wd)
        });
        check("dwg", &dwg, &wg, &|z| {
            loss(&x, &wq, &wk, &wv, &wo, z, &wu, &wd)
        });
        check("dwu", &dwu, &wu, &|z| {
            loss(&x, &wq, &wk, &wv, &wo, &wg, z, &wd)
        });
        check("dwd", &dwd, &wd, &|z| {
            loss(&x, &wq, &wk, &wv, &wo, &wg, &wu, z)
        });

        // RMSNorm gain grads: perturb n1 / n2 (which the `loss` above holds fixed).
        let (dnorm1, dnorm2) = (dl(&grads.dnorm1), dl(&grads.dnorm2));
        let loss_n1 = |nn: &[f32]| -> f32 {
            cpu_gqa_transformer_block(
                &x, nn, &wq, &wk, &wv, &wo, &n2, &wg, &wu, &wd, t, d, kv_dim, ff, n_heads,
                n_kv_heads, dh, theta, eps, true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        let loss_n2 = |nn: &[f32]| -> f32 {
            cpu_gqa_transformer_block(
                &x, &n1, &wq, &wk, &wv, &wo, nn, &wg, &wu, &wd, t, d, kv_dim, ff, n_heads,
                n_kv_heads, dh, theta, eps, true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        check("dnorm1", &dnorm1, &n1, &loss_n1);
        check("dnorm2", &dnorm2, &n2, &loss_n2);
    }

    /// The **full resident GQA model forward** — the real `scirust-sciagent`
    /// decoder (`tokens → embed → N × GQA block → final RMSNorm → tied LM head`) —
    /// matches a step-by-step CPU reference over a 2-layer tied-embedding config.
    /// Skips if no adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn resident_gqa_model_forward_matches_cpu_oracle() {
        use crate::ops::{cpu_gqa_transformer_block, cpu_rms_norm};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (vocab, n_layers, n_heads, n_kv_heads, dh, ff) =
            (6usize, 2usize, 2usize, 1usize, 4usize, 12usize);
        let d = n_heads * dh; // 8
        let kv_dim = n_kv_heads * dh; // 4
        let (theta, eps) = (10_000.0f32, 1e-5f32);
        let tokens: Vec<u32> = vec![1, 3, 0, 5];
        let t = tokens.len();
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.029 + phase).sin() * amp)
                .collect()
        };
        let embedding = gen(vocab * d, 0.2, 0.6);
        let final_norm: Vec<f32> = (0..d).map(|i| 0.85 + 0.02 * i as f32).collect();

        // Per-block raw weights.
        struct Bw {
            n1: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            n2: Vec<f32>,
            wg: Vec<f32>,
            wu: Vec<f32>,
            wd: Vec<f32>,
        }
        let blocks_raw: Vec<Bw> = (0..n_layers)
            .map(|l| {
                let s = l as f32 * 10.0;
                Bw {
                    n1: (0..d).map(|i| 0.8 + 0.03 * i as f32).collect(),
                    wq: gen(d * d, s + 0.5, 0.3),
                    wk: gen(d * kv_dim, s + 1.1, 0.3),
                    wv: gen(d * kv_dim, s + 1.7, 0.3),
                    wo: gen(d * d, s + 2.3, 0.3),
                    n2: (0..d).map(|i| 0.9 - 0.02 * i as f32).collect(),
                    wg: gen(d * ff, s + 2.9, 0.25),
                    wu: gen(d * ff, s + 3.5, 0.25),
                    wd: gen(ff * d, s + 4.1, 0.25),
                }
            })
            .collect();

        // Upload everything; the per-block matrices must outlive the weight refs.
        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let gemb = up(&embedding, vocab, d);
        let gfn = up(&final_norm, 1, d);
        let gblocks: Vec<[GpuMatrix; 9]> = blocks_raw
            .iter()
            .map(|b| {
                [
                    up(&b.n1, 1, d),
                    up(&b.wq, d, d),
                    up(&b.wk, d, kv_dim),
                    up(&b.wv, d, kv_dim),
                    up(&b.wo, d, d),
                    up(&b.n2, 1, d),
                    up(&b.wg, d, ff),
                    up(&b.wu, d, ff),
                    up(&b.wd, ff, d),
                ]
            })
            .collect();
        let weights_blocks: Vec<GqaBlockWeights> = gblocks
            .iter()
            .map(|g| GqaBlockWeights {
                norm1: &g[0],
                wq: &g[1],
                wk: &g[2],
                wv: &g[3],
                wo: &g[4],
                norm2: &g[5],
                wg: &g[6],
                wu: &g[7],
                wd: &g[8],
                n_heads,
                n_kv_heads,
                theta,
            })
            .collect();
        let mw = GqaModelWeights {
            embedding: &gemb,
            blocks: &weights_blocks,
            final_norm: &gfn,
        };
        let logits_gpu = chain
            .download(&chain.gqa_model_forward(&tokens, &mw, eps, true).unwrap())
            .unwrap();

        // CPU reference: embed → blocks → final norm → tied head.
        let mut x: Vec<f32> = tokens
            .iter()
            .flat_map(|&tk| embedding[(tk as usize) * d..(tk as usize) * d + d].to_vec())
            .collect();
        for b in &blocks_raw
        {
            x = cpu_gqa_transformer_block(
                &x, &b.n1, &b.wq, &b.wk, &b.wv, &b.wo, &b.n2, &b.wg, &b.wu, &b.wd, t, d, kv_dim,
                ff, n_heads, n_kv_heads, dh, theta, eps, true,
            );
        }
        let normed = cpu_rms_norm(&x, &final_norm, eps, t, d);
        let mut logits_cpu = vec![0.0f32; t * vocab];
        for i in 0..t
        {
            for vv in 0..vocab
            {
                let mut acc = 0.0f32;
                for dd in 0..d
                {
                    acc += normed[i * d + dd] * embedding[vv * d + dd];
                }
                logits_cpu[i * vocab + vv] = acc;
            }
        }
        assert!(
            rel_err(&logits_gpu, &logits_cpu) < 1e-4,
            "rel_err {}",
            rel_err(&logits_gpu, &logits_cpu)
        );
    }

    /// The **full resident GQA model backward** gradient-checked end-to-end. For
    /// `L = Σ gqa_model_forward(tokens; E, blocks) ⊙ G` (so `dlogits = G`),
    /// `gqa_model_backward(...,G)` yields the tied embedding grad, every block's
    /// weight grads, and the final RMSNorm gain grad. The **tied `dE`** (both the
    /// input-lookup and the output-head paths through the shared `E`) is checked
    /// against central finite differences of `L` over `E`, each block's `dWq` over
    /// its `Wq`, and `d_final_norm` over `final_norm` — a 2-layer / 2-heads / 1-kv
    /// config, so the block chain and the grouped-query accumulation are both
    /// exercised. Skips if no adapter.
    #[test]
    fn gqa_model_backward_matches_finite_differences() {
        use crate::ops::{cpu_gqa_transformer_block, cpu_rms_norm};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (vocab, n_layers, n_heads, n_kv_heads, dh, ff) =
            (5usize, 2usize, 2usize, 1usize, 2usize, 8usize);
        let d = n_heads * dh; // 4
        let kv_dim = n_kv_heads * dh; // 2
        let (theta, eps) = (10_000.0f32, 1e-5f32);
        let tokens: Vec<u32> = vec![1, 4, 2];
        let t = tokens.len();
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.041 + phase).sin() * amp)
                .collect()
        };
        let embedding = gen(vocab * d, 0.2, 0.5);
        let final_norm: Vec<f32> = (0..d).map(|i| 0.85 + 0.02 * i as f32).collect();

        struct Bw {
            n1: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            n2: Vec<f32>,
            wg: Vec<f32>,
            wu: Vec<f32>,
            wd: Vec<f32>,
        }
        let mut blocks_raw: Vec<Bw> = (0..n_layers)
            .map(|l| {
                let s = l as f32 * 7.0;
                Bw {
                    n1: (0..d).map(|i| 0.8 + 0.03 * i as f32).collect(),
                    wq: gen(d * d, s + 0.5, 0.25),
                    wk: gen(d * kv_dim, s + 1.1, 0.25),
                    wv: gen(d * kv_dim, s + 1.7, 0.25),
                    wo: gen(d * d, s + 2.3, 0.25),
                    n2: (0..d).map(|i| 0.9 - 0.02 * i as f32).collect(),
                    wg: gen(d * ff, s + 2.9, 0.2),
                    wu: gen(d * ff, s + 3.5, 0.2),
                    wd: gen(ff * d, s + 4.1, 0.2),
                }
            })
            .collect();
        let g = gen(t * vocab, 5.0, 1.0); // dL/dlogits

        // Upload weights → GqaModelWeights, and run the resident backward.
        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let gemb = up(&embedding, vocab, d);
        let gfn = up(&final_norm, 1, d);
        let gblocks: Vec<[GpuMatrix; 9]> = blocks_raw
            .iter()
            .map(|b| {
                [
                    up(&b.n1, 1, d),
                    up(&b.wq, d, d),
                    up(&b.wk, d, kv_dim),
                    up(&b.wv, d, kv_dim),
                    up(&b.wo, d, d),
                    up(&b.n2, 1, d),
                    up(&b.wg, d, ff),
                    up(&b.wu, d, ff),
                    up(&b.wd, ff, d),
                ]
            })
            .collect();
        let weights_blocks: Vec<GqaBlockWeights> = gblocks
            .iter()
            .map(|gb| GqaBlockWeights {
                norm1: &gb[0],
                wq: &gb[1],
                wk: &gb[2],
                wv: &gb[3],
                wo: &gb[4],
                norm2: &gb[5],
                wg: &gb[6],
                wu: &gb[7],
                wd: &gb[8],
                n_heads,
                n_kv_heads,
                theta,
            })
            .collect();
        let mw = GqaModelWeights {
            embedding: &gemb,
            blocks: &weights_blocks,
            final_norm: &gfn,
        };
        let gg = up(&g, t, vocab);
        let grads = chain
            .gqa_model_backward(&tokens, &mw, &gg, eps, true)
            .unwrap();
        let de = chain.download(&grads.d_embedding).unwrap();
        let dwq_l: Vec<Vec<f32>> = grads
            .blocks
            .iter()
            .map(|bg| chain.download(&bg.dwq).unwrap())
            .collect();

        // CPU reference loss L = Σ logits ⊙ G over (E, blocks).
        let loss = |emb_: &[f32], blocks_: &[Bw]| -> f32 {
            let mut x: Vec<f32> = tokens
                .iter()
                .flat_map(|&tk| emb_[(tk as usize) * d..(tk as usize) * d + d].to_vec())
                .collect();
            for b in blocks_
            {
                x = cpu_gqa_transformer_block(
                    &x, &b.n1, &b.wq, &b.wk, &b.wv, &b.wo, &b.n2, &b.wg, &b.wu, &b.wd, t, d,
                    kv_dim, ff, n_heads, n_kv_heads, dh, theta, eps, true,
                );
            }
            let normed = cpu_rms_norm(&x, &final_norm, eps, t, d);
            let mut acc = 0.0f32;
            for i in 0..t
            {
                for vv in 0..vocab
                {
                    let mut lg = 0.0f32;
                    for dd in 0..d
                    {
                        lg += normed[i * d + dd] * emb_[vv * d + dd];
                    }
                    acc += lg * g[i * vocab + vv];
                }
            }
            acc
        };
        let hh = 1e-3f32;

        // Tied embedding gradient: perturbing E moves both the lookup and the head.
        let mut emb = embedding.clone();
        for idx in 0..emb.len()
        {
            let orig = emb[idx];
            emb[idx] = orig + hh;
            let lp = loss(&emb, &blocks_raw);
            emb[idx] = orig - hh;
            let lm = loss(&emb, &blocks_raw);
            emb[idx] = orig;
            let fd = (lp - lm) / (2.0 * hh);
            assert!(
                (fd - de[idx]).abs() < 2e-2,
                "dE[{idx}]: fd={fd} gpu={}",
                de[idx]
            );
        }

        // Each block's dWq, in place (validates the multi-block backprop chain).
        #[allow(clippy::needless_range_loop)]
        for l in 0..n_layers
        {
            for idx in 0..blocks_raw[l].wq.len()
            {
                let orig = blocks_raw[l].wq[idx];
                blocks_raw[l].wq[idx] = orig + hh;
                let lp = loss(&embedding, &blocks_raw);
                blocks_raw[l].wq[idx] = orig - hh;
                let lm = loss(&embedding, &blocks_raw);
                blocks_raw[l].wq[idx] = orig;
                let fd = (lp - lm) / (2.0 * hh);
                assert!(
                    (fd - dwq_l[l][idx]).abs() < 2e-2,
                    "block{l}.dWq[{idx}]: fd={fd} gpu={}",
                    dwq_l[l][idx]
                );
            }
        }

        // Final pre-logits RMSNorm gain: perturbing `final_norm` moves only the
        // head's `normed = rms_norm(trunk, final_norm)` (the block trunk is
        // independent of it). Validates `grads.d_final_norm`.
        let dfn = chain.download(&grads.d_final_norm).unwrap();
        let loss_fn = |fnorm: &[f32]| -> f32 {
            let mut x: Vec<f32> = tokens
                .iter()
                .flat_map(|&tk| embedding[(tk as usize) * d..(tk as usize) * d + d].to_vec())
                .collect();
            for b in &blocks_raw
            {
                x = cpu_gqa_transformer_block(
                    &x, &b.n1, &b.wq, &b.wk, &b.wv, &b.wo, &b.n2, &b.wg, &b.wu, &b.wd, t, d,
                    kv_dim, ff, n_heads, n_kv_heads, dh, theta, eps, true,
                );
            }
            let normed = cpu_rms_norm(&x, fnorm, eps, t, d);
            let mut acc = 0.0f32;
            for i in 0..t
            {
                for vv in 0..vocab
                {
                    let mut lg = 0.0f32;
                    for dd in 0..d
                    {
                        lg += normed[i * d + dd] * embedding[vv * d + dd];
                    }
                    acc += lg * g[i * vocab + vv];
                }
            }
            acc
        };
        let mut fnv = final_norm.clone();
        for idx in 0..fnv.len()
        {
            let orig = fnv[idx];
            fnv[idx] = orig + hh;
            let lp = loss_fn(&fnv);
            fnv[idx] = orig - hh;
            let lm = loss_fn(&fnv);
            fnv[idx] = orig;
            let fd = (lp - lm) / (2.0 * hh);
            assert!(
                (fd - dfn[idx]).abs() < 2e-2,
                "d_final_norm[{idx}]: fd={fd} gpu={}",
                dfn[idx]
            );
        }
    }

    /// The **fully resident** SwiGLU MLP — `(silu(x·W_gate) ⊙ (x·W_up))·W_down`
    /// with every `t×h` intermediate kept in VRAM — matches a step-by-step CPU
    /// oracle. Skips if no adapter; asserts on lavapipe / a real GPU.
    #[test]
    fn resident_swiglu_mlp_matches_cpu_oracle() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h) = (5usize, 4usize, 8usize);
        let x: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.15 - 0.9).sin()).collect();
        let wg: Vec<f32> = (0..d * h)
            .map(|i| (i as f32 * 0.07 + 0.2).cos() * 0.5)
            .collect();
        let wu: Vec<f32> = (0..d * h)
            .map(|i| (i as f32 * 0.05 - 0.4).sin() * 0.5)
            .collect();
        let wd: Vec<f32> = (0..h * d)
            .map(|i| (i as f32 * 0.09 + 0.1).cos() * 0.5)
            .collect();

        let gx = chain.upload(&x, t, d);
        let gwg = chain.upload(&wg, d, h);
        let gwu = chain.upload(&wu, d, h);
        let gwd = chain.upload(&wd, h, d);
        let out = chain
            .download(&chain.swiglu_mlp(&gx, &gwg, &gwu, &gwd).unwrap())
            .unwrap();
        assert_eq!(out.len(), t * d);

        // CPU oracle: gate = x·Wg, up = x·Wu, act = silu(gate)⊙up, out = act·Wd.
        let gate = CpuBackend.gemm_f32(&x, &wg, t, d, h).unwrap();
        let up = CpuBackend.gemm_f32(&x, &wu, t, d, h).unwrap();
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let expected = CpuBackend.gemm_f32(&act, &wd, t, h, d).unwrap();
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// The **complete residual transformer block** — pre-norm attention with
    /// Q/K/V/O projections + residual, then pre-norm SwiGLU MLP + residual, all
    /// resident — matches a full step-by-step CPU oracle. Skips if no adapter;
    /// asserts on lavapipe / a real GPU. This is one whole 350M layer forward.
    #[test]
    fn resident_transformer_block_matches_cpu_oracle() {
        use crate::ops::{cpu_rms_norm, cpu_scale_causal_mask, cpu_softmax};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h) = (6usize, 8usize, 16usize);
        let eps = 1e-5f32;
        // Deterministic pseudo-random weights (distinct phase per matrix).
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.031 + phase).sin() * amp)
                .collect()
        };
        let x = gen(t * d, 0.0, 1.0);
        let n1 = (0..d).map(|i| 0.7 + 0.03 * i as f32).collect::<Vec<_>>();
        let wq = gen(d * d, 0.5, 0.3);
        let wk = gen(d * d, 1.1, 0.3);
        let wv = gen(d * d, 1.7, 0.3);
        let wo = gen(d * d, 2.3, 0.3);
        let n2 = (0..d).map(|i| 0.9 - 0.02 * i as f32).collect::<Vec<_>>();
        let wg = gen(d * h, 2.9, 0.25);
        let wu = gen(d * h, 3.5, 0.25);
        let wd = gen(h * d, 4.1, 0.25);

        // GPU: one resident call.
        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let gx = up(&x, t, d);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        let (gwq, gwk, gwv, gwo) = (up(&wq, d, d), up(&wk, d, d), up(&wv, d, d), up(&wo, d, d));
        let (gwg, gwu, gwd) = (up(&wg, d, h), up(&wu, d, h), up(&wd, h, d));
        let weights = BlockWeights {
            norm1: &gn1,
            wq: &gwq,
            wk: &gwk,
            wv: &gwv,
            wo: &gwo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        };
        let out = chain
            .download(&chain.transformer_block(&gx, &weights, eps, true).unwrap())
            .unwrap();

        // CPU oracle, step by step.
        let gemm = |a: &[f32], b: &[f32], m, k, n| CpuBackend.gemm_f32(a, b, m, k, n).unwrap();
        let transpose = |a: &[f32], r: usize, c: usize| {
            let mut o = vec![0.0f32; r * c];
            for i in 0..r
            {
                for j in 0..c
                {
                    o[j * r + i] = a[i * c + j];
                }
            }
            o
        };
        let xn = cpu_rms_norm(&x, &n1, eps, t, d);
        let q = gemm(&xn, &wq, t, d, d);
        let k = gemm(&xn, &wk, t, d, d);
        let v = gemm(&xn, &wv, t, d, d);
        let s = gemm(&q, &transpose(&k, t, d), t, d, t); // Q·Kᵀ
        let s = cpu_scale_causal_mask(&s, t, t, 1.0 / (d as f32).sqrt(), true);
        let aw = cpu_softmax(&s, t, t);
        let a = gemm(&aw, &v, t, t, d);
        let ao = gemm(&a, &wo, t, d, d);
        let hh: Vec<f32> = x.iter().zip(&ao).map(|(a, b)| a + b).collect(); // residual 1
        let hn = cpu_rms_norm(&hh, &n2, eps, t, d);
        let gate = gemm(&hn, &wg, t, d, h);
        let upm = gemm(&hn, &wu, t, d, h);
        let act: Vec<f32> = gate
            .iter()
            .zip(&upm)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let m = gemm(&act, &wd, t, h, d);
        let expected: Vec<f32> = hh.iter().zip(&m).map(|(a, b)| a + b).collect(); // residual 2

        assert_eq!(out.len(), t * d);
        assert!(
            rel_err(&out, &expected) < 1e-4,
            "out={out:?} exp={expected:?}"
        );
    }

    /// A **stack of transformer blocks** run resident (each block's output
    /// feeds the next in VRAM) must match the CPU oracle applied layer by layer.
    /// This is the resident trunk of the 350M forward. Skips if no adapter.
    #[test]
    fn resident_transformer_stack_matches_cpu_oracle() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h, layers) = (5usize, 8usize, 16usize, 3usize);
        let eps = 1e-5f32;
        // Distinct deterministic weights per layer (phase offset by layer index).
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.029 + phase).sin() * amp)
                .collect()
        };
        let x = gen(t * d, 0.0, 1.0);

        // Per-layer CPU weight sets, kept alive to back the GPU uploads.
        struct W {
            n1: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            n2: Vec<f32>,
            wg: Vec<f32>,
            wu: Vec<f32>,
            wd: Vec<f32>,
        }
        let ws: Vec<W> = (0..layers)
            .map(|l| {
                let p = l as f32 * 10.0;
                W {
                    n1: (0..d).map(|i| 0.7 + 0.02 * i as f32).collect(),
                    wq: gen(d * d, p + 0.5, 0.3),
                    wk: gen(d * d, p + 1.1, 0.3),
                    wv: gen(d * d, p + 1.7, 0.3),
                    wo: gen(d * d, p + 2.3, 0.3),
                    n2: (0..d).map(|i| 0.9 - 0.01 * i as f32).collect(),
                    wg: gen(d * h, p + 2.9, 0.25),
                    wu: gen(d * h, p + 3.5, 0.25),
                    wd: gen(h * d, p + 4.1, 0.25),
                }
            })
            .collect();

        // Upload every layer's weights; keep the GpuMatrix handles alive.
        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let g: Vec<[GpuMatrix; 9]> = ws
            .iter()
            .map(|w| {
                [
                    up(&w.n1, 1, d),
                    up(&w.wq, d, d),
                    up(&w.wk, d, d),
                    up(&w.wv, d, d),
                    up(&w.wo, d, d),
                    up(&w.n2, 1, d),
                    up(&w.wg, d, h),
                    up(&w.wu, d, h),
                    up(&w.wd, h, d),
                ]
            })
            .collect();
        let blocks: Vec<BlockWeights> = g
            .iter()
            .map(|m| BlockWeights {
                norm1: &m[0],
                wq: &m[1],
                wk: &m[2],
                wv: &m[3],
                wo: &m[4],
                norm2: &m[5],
                wg: &m[6],
                wu: &m[7],
                wd: &m[8],
            })
            .collect();

        let gx = up(&x, t, d);
        let out = chain
            .download(&chain.transformer_stack(&gx, &blocks, eps, true).unwrap())
            .unwrap();

        // CPU oracle: fold x through each layer's block.
        let mut cur = x.clone();
        for w in &ws
        {
            cur = cpu_block(
                &cur,
                (&w.n1, &w.n2),
                (&w.wq, &w.wk, &w.wv, &w.wo),
                (&w.wg, &w.wu, &w.wd),
                (t, d, h),
                eps,
                true,
            );
        }
        assert_eq!(out.len(), t * d);
        assert!(rel_err(&out, &cur) < 1e-4, "out={out:?} exp={cur:?}");
    }

    /// Resident embedding gather matches the CPU oracle (row `i` = table row
    /// `tokens[i]`). Skips if no adapter.
    #[test]
    fn resident_embed_matches_cpu_oracle() {
        use crate::ops::cpu_embed;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (vocab, d) = (11usize, 6usize);
        let table: Vec<f32> = (0..vocab * d)
            .map(|i| (i as f32 * 0.1 - 1.0).sin())
            .collect();
        let tokens: Vec<u32> = vec![0, 5, 10, 3, 3, 7];
        let gt = chain.upload(&table, vocab, d);
        let out = chain.download(&chain.embed(&tokens, &gt).unwrap()).unwrap();
        let expected = cpu_embed(&tokens, &table, d, vocab);
        assert_eq!(out, expected); // gather is an exact copy — bit-identical.
    }

    /// The **full tied-embedding decoder forward** — embed → 2 blocks → final
    /// RMSNorm → tied LM head — run resident (`tokens → logits`), must match a
    /// step-by-step CPU oracle. Skips if no adapter; asserts on lavapipe / a real
    /// GPU. This is a whole 350M forward pass in miniature.
    #[test]
    fn resident_model_forward_matches_cpu_oracle() {
        use crate::ops::cpu_embed;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (vocab, d, h, layers) = (13usize, 8usize, 16usize, 2usize);
        let eps = 1e-5f32;
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.027 + phase).sin() * amp)
                .collect()
        };
        let embedding = gen(vocab * d, 0.3, 1.0);
        let final_norm: Vec<f32> = (0..d).map(|i| 0.8 + 0.01 * i as f32).collect();
        let tokens: Vec<u32> = vec![1, 4, 9, 2, 7];
        let t = tokens.len();

        struct W {
            n1: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            n2: Vec<f32>,
            wg: Vec<f32>,
            wu: Vec<f32>,
            wd: Vec<f32>,
        }
        let ws: Vec<W> = (0..layers)
            .map(|l| {
                let p = l as f32 * 10.0;
                W {
                    n1: (0..d).map(|i| 0.7 + 0.02 * i as f32).collect(),
                    wq: gen(d * d, p + 0.5, 0.3),
                    wk: gen(d * d, p + 1.1, 0.3),
                    wv: gen(d * d, p + 1.7, 0.3),
                    wo: gen(d * d, p + 2.3, 0.3),
                    n2: (0..d).map(|i| 0.9 - 0.01 * i as f32).collect(),
                    wg: gen(d * h, p + 2.9, 0.25),
                    wu: gen(d * h, p + 3.5, 0.25),
                    wd: gen(h * d, p + 4.1, 0.25),
                }
            })
            .collect();

        // Upload everything resident.
        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let g_emb = up(&embedding, vocab, d);
        let g_fn = up(&final_norm, 1, d);
        let g: Vec<[GpuMatrix; 9]> = ws
            .iter()
            .map(|w| {
                [
                    up(&w.n1, 1, d),
                    up(&w.wq, d, d),
                    up(&w.wk, d, d),
                    up(&w.wv, d, d),
                    up(&w.wo, d, d),
                    up(&w.n2, 1, d),
                    up(&w.wg, d, h),
                    up(&w.wu, d, h),
                    up(&w.wd, h, d),
                ]
            })
            .collect();
        let blocks: Vec<BlockWeights> = g
            .iter()
            .map(|m| BlockWeights {
                norm1: &m[0],
                wq: &m[1],
                wk: &m[2],
                wv: &m[3],
                wo: &m[4],
                norm2: &m[5],
                wg: &m[6],
                wu: &m[7],
                wd: &m[8],
            })
            .collect();
        let mw = ModelWeights {
            embedding: &g_emb,
            blocks: &blocks,
            final_norm: &g_fn,
        };
        let logits = chain
            .download(&chain.model_forward_tied(&tokens, &mw, eps, true).unwrap())
            .unwrap();

        // CPU oracle: embed → blocks → final norm → h·Eᵀ.
        let mut cur = cpu_embed(&tokens, &embedding, d, vocab);
        for w in &ws
        {
            cur = cpu_block(
                &cur,
                (&w.n1, &w.n2),
                (&w.wq, &w.wk, &w.wv, &w.wo),
                (&w.wg, &w.wu, &w.wd),
                (t, d, h),
                eps,
                true,
            );
        }
        let normed = crate::ops::cpu_rms_norm(&cur, &final_norm, eps, t, d);
        // logits = normed(t×d) · Eᵀ(d×vocab); build Eᵀ then GEMM.
        let mut et = vec![0.0f32; d * vocab];
        for r in 0..vocab
        {
            for c in 0..d
            {
                et[c * vocab + r] = embedding[r * d + c];
            }
        }
        let expected = CpuBackend.gemm_f32(&normed, &et, t, d, vocab).unwrap();
        assert_eq!(logits.len(), t * vocab);
        assert!(rel_err(&logits, &expected) < 1e-4, "logits mismatch");
    }

    /// The GEMM backward (vjp) must match numerical gradients. For the scalar
    /// loss `L = Σ (A·B) ⊙ G`, the analytic gradients are `grad_a = G·Bᵀ` and
    /// `grad_b = Aᵀ·G`; this checks the GPU's gradients against central finite
    /// differences of `L` computed on the CPU — the gold-standard correctness
    /// test for an adjoint. Skips if no adapter.
    #[test]
    fn matmul_backward_matches_finite_differences() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.3 - 0.5).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.4 + 0.2).cos()).collect();
        let g: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.7 - 1.0).sin()).collect(); // dL/dC

        let (ga, gb, gg) = (
            chain.upload(&a, m, k),
            chain.upload(&b, k, n),
            chain.upload(&g, m, n),
        );
        let (da_m, db_m) = chain.matmul_backward(&ga, &gb, &gg).unwrap();
        assert_eq!((da_m.rows(), da_m.cols()), (m, k));
        assert_eq!((db_m.rows(), db_m.cols()), (k, n));
        let grad_a = chain.download(&da_m).unwrap();
        let grad_b = chain.download(&db_m).unwrap();

        // Scalar loss L(A,B) = Σ (A·B) ⊙ G, evaluated on the CPU.
        let loss = |aa: &[f32], bb: &[f32]| -> f32 {
            CpuBackend
                .gemm_f32(aa, bb, m, k, n)
                .unwrap()
                .iter()
                .zip(&g)
                .map(|(c, gg)| c * gg)
                .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..m * k
        {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[idx] += eps;
            am[idx] -= eps;
            let fd = (loss(&ap, &b) - loss(&am, &b)) / (2.0 * eps);
            assert!(
                (fd - grad_a[idx]).abs() < 1e-2,
                "grad_a[{idx}]: fd={fd} gpu={}",
                grad_a[idx]
            );
        }
        for idx in 0..k * n
        {
            let (mut bp, mut bm) = (b.clone(), b.clone());
            bp[idx] += eps;
            bm[idx] -= eps;
            let fd = (loss(&a, &bp) - loss(&a, &bm)) / (2.0 * eps);
            assert!(
                (fd - grad_b[idx]).abs() < 1e-2,
                "grad_b[{idx}]: fd={fd} gpu={}",
                grad_b[idx]
            );
        }
    }

    /// Softmax backward must match numerical gradients. For `L = Σ softmax(X)⊙G`
    /// the input gradient is `dx = softmax_backward(Y, G)`; checked against
    /// central finite differences of `L` over `X` on the CPU. Skips if no adapter.
    #[test]
    fn softmax_backward_matches_finite_differences() {
        use crate::ops::{cpu_softmax, cpu_softmax_backward};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, cols) = (3usize, 5usize);
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
            .collect();
        let g: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.5 + 0.2).cos())
            .collect(); // dL/dY

        let y = cpu_softmax(&x, rows, cols);
        let gy = chain.upload(&y, rows, cols);
        let gg = chain.upload(&g, rows, cols);
        let dx_gpu = chain
            .download(&chain.softmax_backward(&gy, &gg).unwrap())
            .unwrap();
        // GPU must also match the CPU adjoint formula exactly (same arithmetic).
        let dx_cpu = cpu_softmax_backward(&y, &g, rows, cols);
        assert!(rel_err(&dx_gpu, &dx_cpu) < 1e-4);

        // Gold standard: central finite differences of L = Σ softmax(X)⊙G.
        let loss = |xx: &[f32]| -> f32 {
            cpu_softmax(xx, rows, cols)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..rows * cols
        {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += eps;
            xm[idx] -= eps;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * eps);
            assert!(
                (fd - dx_gpu[idx]).abs() < 1e-2,
                "dx[{idx}]: fd={fd} gpu={}",
                dx_gpu[idx]
            );
        }
    }

    /// SwiGLU-gate backward must match numerical gradients. For
    /// `L = Σ (silu(A)⊙B) ⊙ G`, `(da, db) = swiglu_backward(A, B, G)`; each is
    /// checked against central finite differences of `L`. Skips if no adapter.
    #[test]
    fn swiglu_backward_matches_finite_differences() {
        use crate::ops::cpu_swiglu_backward;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let n = 12usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4 - 1.5).sin() * 2.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 0.5).cos()).collect();
        let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 - 0.3).sin()).collect(); // dL/dC

        let (ga, gb, gg) = (
            chain.upload(&a, 1, n),
            chain.upload(&b, 1, n),
            chain.upload(&g, 1, n),
        );
        let (da_m, db_m) = chain.swiglu_backward(&ga, &gb, &gg).unwrap();
        let da_gpu = chain.download(&da_m).unwrap();
        let db_gpu = chain.download(&db_m).unwrap();
        let (da_cpu, db_cpu) = cpu_swiglu_backward(&a, &b, &g);
        assert!(rel_err(&da_gpu, &da_cpu) < 1e-4 && rel_err(&db_gpu, &db_cpu) < 1e-4);

        // Gold standard: finite differences of L = Σ (silu(A)⊙B) ⊙ G.
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let loss =
            |aa: &[f32], bb: &[f32]| -> f32 { (0..n).map(|i| silu(aa[i]) * bb[i] * g[i]).sum() };
        let eps = 1e-3f32;
        for idx in 0..n
        {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[idx] += eps;
            am[idx] -= eps;
            let fd = (loss(&ap, &b) - loss(&am, &b)) / (2.0 * eps);
            assert!(
                (fd - da_gpu[idx]).abs() < 1e-2,
                "da[{idx}]: fd={fd} gpu={}",
                da_gpu[idx]
            );
            let (mut bp, mut bm) = (b.clone(), b.clone());
            bp[idx] += eps;
            bm[idx] -= eps;
            let fd = (loss(&a, &bp) - loss(&a, &bm)) / (2.0 * eps);
            assert!(
                (fd - db_gpu[idx]).abs() < 1e-2,
                "db[{idx}]: fd={fd} gpu={}",
                db_gpu[idx]
            );
        }
    }

    /// RMSNorm backward must match numerical gradients. For `L = Σ rmsnorm(X,w)⊙G`
    /// the input gradient `dx = rms_norm_backward(X, w, G)` is checked against
    /// central finite differences of `L` over `X` on the CPU — this exercises the
    /// mean-coupling term of the jacobian. Skips if no adapter.
    #[test]
    fn rms_norm_backward_matches_finite_differences() {
        use crate::ops::{cpu_rms_norm, cpu_rms_norm_backward};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, cols) = (3usize, 5usize);
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.6 + 0.1 * i as f32).collect();
        let g: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.5 + 0.2).cos())
            .collect(); // dL/dY

        let dx_gpu = chain
            .download(
                &chain
                    .rms_norm_backward(
                        &chain.upload(&x, rows, cols),
                        &chain.upload(&w, 1, cols),
                        &chain.upload(&g, rows, cols),
                        eps,
                    )
                    .unwrap(),
            )
            .unwrap();
        assert!(rel_err(&dx_gpu, &cpu_rms_norm_backward(&x, &w, &g, eps, rows, cols)) < 1e-4);

        // Gold standard: central finite differences of L = Σ rmsnorm(X,w)⊙G.
        let loss = |xx: &[f32]| -> f32 {
            cpu_rms_norm(xx, &w, eps, rows, cols)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let step = 1e-3f32;
        for idx in 0..rows * cols
        {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += step;
            xm[idx] -= step;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * step);
            assert!(
                (fd - dx_gpu[idx]).abs() < 1e-2,
                "dx[{idx}]: fd={fd} gpu={}",
                dx_gpu[idx]
            );
        }
    }

    /// RMSNorm **gain** gradient, gradient-checked. For `L = Σ rms_norm(X,w)⊙G`,
    /// `dw = rms_norm_gain_backward(X, G)` — checked against the CPU adjoint AND
    /// central finite differences of `L` over `w`. Skips if no adapter.
    #[test]
    fn rms_norm_gain_backward_matches_finite_differences() {
        use crate::ops::{cpu_rms_norm, cpu_rms_norm_gain_backward};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, cols) = (4usize, 6usize);
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.23 - 1.1).sin() * 1.5)
            .collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.6 + 0.1 * i as f32).collect();
        let g: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.37 + 0.2).cos())
            .collect(); // dL/dY

        let gx = chain.upload(&x, rows, cols);
        let gg = chain.upload(&g, rows, cols);
        let dw_gpu = chain
            .download(&chain.rms_norm_gain_backward(&gx, &gg, eps).unwrap())
            .unwrap();
        // GPU matches the CPU adjoint.
        let dw_cpu = cpu_rms_norm_gain_backward(&x, &g, eps, rows, cols);
        assert!(rel_err(&dw_gpu, &dw_cpu) < 1e-4);

        // Gold standard: central finite differences of L = Σ rms_norm(X,w)⊙G over w.
        let loss = |ww: &[f32]| -> f32 {
            cpu_rms_norm(&x, ww, eps, rows, cols)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let step = 1e-3f32;
        for j in 0..cols
        {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp[j] += step;
            wm[j] -= step;
            let fd = (loss(&wp) - loss(&wm)) / (2.0 * step);
            assert!(
                (fd - dw_gpu[j]).abs() < 1e-2,
                "dw[{j}]: fd={fd} gpu={}",
                dw_gpu[j]
            );
        }
    }

    /// Scale + causal-mask backward: `din = scale·dout` below/on the diagonal,
    /// `0` above. Exact match to the CPU oracle (no accumulation). Skips if no
    /// adapter.
    #[test]
    fn scale_causal_mask_backward_matches_cpu_oracle() {
        use crate::ops::cpu_scale_causal_mask_backward;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let n = 6usize;
        let scale = 0.125f32;
        let dout: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let din_gpu = chain
            .download(
                &chain
                    .scale_causal_mask_backward(&chain.upload(&dout, n, n), scale, true)
                    .unwrap(),
            )
            .unwrap();
        let din_cpu = cpu_scale_causal_mask_backward(&dout, n, n, scale, true);
        assert!(rel_err(&din_gpu, &din_cpu) < 1e-6);
        // Above the diagonal must be exactly zero.
        for i in 0..n
        {
            for j in i + 1..n
            {
                assert_eq!(din_gpu[i * n + j], 0.0, "({i},{j}) not zeroed");
            }
        }
    }

    /// Embedding backward: the scatter-sum must match the CPU oracle exactly
    /// (a pure accumulation of gathered rows) and match finite differences of
    /// `L = Σ embed(tokens, E)⊙G` over the table `E`. Skips if no adapter.
    #[test]
    fn embed_backward_matches_cpu_and_finite_differences() {
        use crate::ops::{cpu_embed, cpu_embed_backward};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (vocab, d) = (7usize, 4usize);
        let tokens: Vec<u32> = vec![0, 3, 6, 3, 1, 3]; // token 3 repeats → accumulation
        let t = tokens.len();
        let g: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.3 - 0.5).sin()).collect(); // dL/dOut

        let dtable_gpu = chain
            .download(
                &chain
                    .embed_backward(&tokens, &chain.upload(&g, t, d), vocab)
                    .unwrap(),
            )
            .unwrap();
        let dtable_cpu = cpu_embed_backward(&tokens, &g, d, vocab);
        assert!(rel_err(&dtable_gpu, &dtable_cpu) < 1e-5);

        // Finite differences of L = Σ embed(tokens, E)⊙G over the table E.
        let table: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.11).cos()).collect();
        let loss = |tab: &[f32]| -> f32 {
            cpu_embed(&tokens, tab, d, vocab)
                .iter()
                .zip(&g)
                .map(|(a, b)| a * b)
                .sum()
        };
        let eps = 1e-3f32;
        for idx in 0..vocab * d
        {
            let (mut ep, mut em) = (table.clone(), table.clone());
            ep[idx] += eps;
            em[idx] -= eps;
            let fd = (loss(&ep) - loss(&em)) / (2.0 * eps);
            assert!(
                (fd - dtable_gpu[idx]).abs() < 1e-2,
                "dE[{idx}]: fd={fd} gpu={}",
                dtable_gpu[idx]
            );
        }
    }

    /// The **composed block backward** — `dx` through the whole transformer block
    /// (attention + MLP + both residuals + both norms) — must match numerical
    /// gradients. Forward-with-cache then backward on the GPU; the loss
    /// `L = Σ block(X)⊙G` and its central finite differences over `X` are
    /// computed on the CPU (via `cpu_block`), so this validates that every
    /// adjoint composes correctly. The end-to-end integration test. Skips if no
    /// adapter.
    #[test]
    fn transformer_block_backward_matches_finite_differences() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h) = (4usize, 6usize, 12usize);
        let eps = 1e-5f32;
        let gen = |n: usize, phase: f32, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.037 + phase).sin() * amp)
                .collect()
        };
        let x = gen(t * d, 0.0, 1.0);
        let n1: Vec<f32> = (0..d).map(|i| 0.7 + 0.02 * i as f32).collect();
        let wq = gen(d * d, 0.5, 0.3);
        let wk = gen(d * d, 1.1, 0.3);
        let wv = gen(d * d, 1.7, 0.3);
        let wo = gen(d * d, 2.3, 0.3);
        let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.01 * i as f32).collect();
        let wg = gen(d * h, 2.9, 0.25);
        let wu = gen(d * h, 3.5, 0.25);
        let wd = gen(h * d, 4.1, 0.25);
        let g = gen(t * d, 5.0, 1.0); // dL/dout

        let up = |data: &[f32], r: usize, c: usize| chain.upload(data, r, c);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        let (gwq, gwk, gwv, gwo) = (up(&wq, d, d), up(&wk, d, d), up(&wv, d, d), up(&wo, d, d));
        let (gwg, gwu, gwd) = (up(&wg, d, h), up(&wu, d, h), up(&wd, h, d));
        let weights = BlockWeights {
            norm1: &gn1,
            wq: &gwq,
            wk: &gwk,
            wv: &gwv,
            wo: &gwo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        };
        let gx = up(&x, t, d);
        let (out_m, cache) = chain
            .transformer_block_forward_cached(&gx, &weights, eps, true)
            .unwrap();
        // Forward parity against the CPU block oracle.
        let out_gpu = chain.download(&out_m).unwrap();
        let out_cpu = cpu_block(
            &x,
            (&n1, &n2),
            (&wq, &wk, &wv, &wo),
            (&wg, &wu, &wd),
            (t, d, h),
            eps,
            true,
        );
        assert!(rel_err(&out_gpu, &out_cpu) < 1e-4, "forward mismatch");

        let dx_gpu = chain
            .download(
                &chain
                    .transformer_block_backward(&gx, &weights, &cache, &up(&g, t, d), eps, true)
                    .unwrap(),
            )
            .unwrap();

        // Gold standard: central finite differences of L = Σ block(X)⊙G over X.
        let loss = |xx: &[f32]| -> f32 {
            cpu_block(
                xx,
                (&n1, &n2),
                (&wq, &wk, &wv, &wo),
                (&wg, &wu, &wd),
                (t, d, h),
                eps,
                true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        let step = 1e-3f32;
        for idx in 0..t * d
        {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += step;
            xm[idx] -= step;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * step);
            assert!(
                (fd - dx_gpu[idx]).abs() < 2e-2,
                "dx[{idx}]: fd={fd} gpu={}",
                dx_gpu[idx]
            );
        }
    }

    /// Cross-entropy gradient must match numerical gradients. `dlogits =
    /// cross_entropy_grad(logits, targets)` is checked against the CPU analytic
    /// `(softmax − onehot)/rows` and against central finite differences of the
    /// mean cross-entropy loss over the logits. The seed of the training
    /// backward. Skips if no adapter.
    #[test]
    fn cross_entropy_grad_matches_finite_differences() {
        use crate::ops::{cpu_cross_entropy, cpu_cross_entropy_grad};

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (rows, vocab) = (4usize, 7usize);
        let logits: Vec<f32> = (0..rows * vocab)
            .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
            .collect();
        let targets: Vec<u32> = vec![2, 5, 0, 6];

        let dl_gpu = chain
            .download(
                &chain
                    .cross_entropy_grad(&chain.upload(&logits, rows, vocab), &targets)
                    .unwrap(),
            )
            .unwrap();
        assert!(
            rel_err(
                &dl_gpu,
                &cpu_cross_entropy_grad(&logits, &targets, rows, vocab)
            ) < 1e-4
        );

        // Gold standard: central finite differences of the mean loss over logits.
        let step = 1e-3f32;
        for idx in 0..rows * vocab
        {
            let (mut lp, mut lm) = (logits.clone(), logits.clone());
            lp[idx] += step;
            lm[idx] -= step;
            let fd = (cpu_cross_entropy(&lp, &targets, rows, vocab)
                - cpu_cross_entropy(&lm, &targets, rows, vocab))
                / (2.0 * step);
            assert!(
                (fd - dl_gpu[idx]).abs() < 1e-2,
                "dlogits[{idx}]: fd={fd} gpu={}",
                dl_gpu[idx]
            );
        }
    }

    /// The **capstone**: a real on-device training loop actually reduces the
    /// loss. A linear model `logits = x·W` with cross-entropy targets; each step
    /// runs `xent_grad → matmul_backward (dW) → sgd_step(W)` entirely on the GPU.
    /// Asserts the loss decreases monotonically and ends well below the start —
    /// the whole loop (forward → loss → grads → update) works. Skips if no
    /// adapter.
    #[test]
    fn sgd_step_reduces_cross_entropy_loss() {
        use crate::ops::cpu_cross_entropy;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, vocab) = (6usize, 5usize, 8usize);
        let x: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.21 - 0.7).sin()).collect();
        let w0: Vec<f32> = (0..d * vocab)
            .map(|i| (i as f32 * 0.13 + 0.2).cos() * 0.3)
            .collect();
        let targets: Vec<u32> = (0..t as u32).map(|i| (i * 5 + 1) % vocab as u32).collect();
        let lr = 0.5f32;

        let gx = chain.upload(&x, t, d);
        let mut gw = chain.upload(&w0, d, vocab);
        let mut losses = Vec::new();
        for _ in 0..12
        {
            // Forward: logits = x·W.
            let logits = chain.matmul(&gx, &gw).unwrap();
            let logits_cpu = chain.download(&logits).unwrap();
            losses.push(cpu_cross_entropy(&logits_cpu, &targets, t, vocab));
            // Backward: dlogits = xent grad; dW = xᵀ·dlogits (matmul_backward.1).
            let dlogits = chain.cross_entropy_grad(&logits, &targets).unwrap();
            let (_dx, dw) = chain.matmul_backward(&gx, &gw, &dlogits).unwrap();
            // Optimizer step: W ← W − lr·dW.
            gw = chain.sgd_step(&gw, &dw, lr).unwrap();
        }

        // Loss decreases monotonically and ends far below the start.
        for pair in losses.windows(2)
        {
            assert!(pair[1] < pair[0] + 1e-6, "loss went up: {pair:?}");
        }
        assert!(
            *losses.last().unwrap() < losses[0] * 0.7,
            "loss barely moved: {} → {}",
            losses[0],
            losses.last().unwrap()
        );
    }

    /// One AdamW step matches the CPU oracle exactly (param, m, v) — the moments
    /// update and the bias-corrected, weight-decayed parameter update. Skips if
    /// no adapter.
    #[test]
    fn adamw_step_matches_cpu_oracle() {
        use crate::ops::cpu_adamw_step;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let n = 16usize;
        let mut param: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 0.1).cos() * 0.5).collect();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let (lr, betas, eps, wd) = (0.01f32, (0.9f32, 0.999f32), 1e-8f32, 0.01f32);

        let gp = chain.upload(&param, 1, n);
        let gg = chain.upload(&grad, 1, n);
        let gm = chain.upload(&m, 1, n);
        let gv = chain.upload(&v, 1, n);
        chain
            .adamw_step(&gp, &gg, &gm, &gv, lr, betas, eps, wd, 1)
            .unwrap();
        cpu_adamw_step(&mut param, &grad, &mut m, &mut v, lr, betas, eps, wd, 1);

        assert!(
            rel_err(&chain.download(&gp).unwrap(), &param) < 1e-5,
            "param mismatch"
        );
        assert!(
            rel_err(&chain.download(&gm).unwrap(), &m) < 1e-5,
            "m mismatch"
        );
        assert!(
            rel_err(&chain.download(&gv).unwrap(), &v) < 1e-5,
            "v mismatch"
        );
    }

    /// An AdamW training loop reduces the loss on-device. Same linear + cross-
    /// entropy model as the SGD capstone, but the update is the resident AdamW
    /// step with persistent `m`/`v` moments. Asserts the loss ends well below the
    /// start. Skips if no adapter.
    #[test]
    fn adamw_training_loop_reduces_loss() {
        use crate::ops::cpu_cross_entropy;

        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, vocab) = (6usize, 5usize, 8usize);
        let x: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.21 - 0.7).sin()).collect();
        let w0: Vec<f32> = (0..d * vocab)
            .map(|i| (i as f32 * 0.13 + 0.2).cos() * 0.3)
            .collect();
        let targets: Vec<u32> = (0..t as u32).map(|i| (i * 5 + 1) % vocab as u32).collect();

        let gx = chain.upload(&x, t, d);
        let gw = chain.upload(&w0, d, vocab);
        let gm = chain.upload(&vec![0.0f32; d * vocab], d, vocab);
        let gv = chain.upload(&vec![0.0f32; d * vocab], d, vocab);
        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 1..=30u32
        {
            let logits = chain.matmul(&gx, &gw).unwrap();
            let l = cpu_cross_entropy(&chain.download(&logits).unwrap(), &targets, t, vocab);
            if step == 1
            {
                first = l;
            }
            last = l;
            let dlogits = chain.cross_entropy_grad(&logits, &targets).unwrap();
            let (_dx, dw) = chain.matmul_backward(&gx, &gw, &dlogits).unwrap();
            chain
                .adamw_step(&gw, &dw, &gm, &gv, 0.1, (0.9, 0.999), 1e-8, 0.0, step)
                .unwrap();
        }
        assert!(last < first * 0.5, "AdamW barely moved: {first} → {last}");
    }

    /// The block weight gradient `dWq` must match numerical gradients: for
    /// `L = Σ block(x; W)⊙G`, `transformer_block_backward_full` gives `dWq`,
    /// checked against central finite differences of `L` over `Wq` (via the CPU
    /// block oracle). Validates the weight-grad plumbing. Skips if no adapter.
    #[test]
    fn block_weight_grad_dwq_matches_finite_differences() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h) = (4usize, 6usize, 10usize);
        let eps = 1e-5f32;
        let gen = |n: usize, p: f32, a: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.037 + p).sin() * a).collect()
        };
        let x = gen(t * d, 0.0, 1.0);
        let n1: Vec<f32> = (0..d).map(|i| 0.7 + 0.02 * i as f32).collect();
        let (wq, wk, wv, wo) = (
            gen(d * d, 0.5, 0.3),
            gen(d * d, 1.1, 0.3),
            gen(d * d, 1.7, 0.3),
            gen(d * d, 2.3, 0.3),
        );
        let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.01 * i as f32).collect();
        let (wg, wu, wd) = (
            gen(d * h, 2.9, 0.25),
            gen(d * h, 3.5, 0.25),
            gen(h * d, 4.1, 0.25),
        );
        let g = gen(t * d, 5.0, 1.0);

        let up = |data: &[f32], r, c| chain.upload(data, r, c);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        let (gwq, gwk, gwv, gwo) = (up(&wq, d, d), up(&wk, d, d), up(&wv, d, d), up(&wo, d, d));
        let (gwg, gwu, gwd) = (up(&wg, d, h), up(&wu, d, h), up(&wd, h, d));
        let weights = BlockWeights {
            norm1: &gn1,
            wq: &gwq,
            wk: &gwk,
            wv: &gwv,
            wo: &gwo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        };
        let gx = up(&x, t, d);
        let (_out, cache) = chain
            .transformer_block_forward_cached(&gx, &weights, eps, true)
            .unwrap();
        let (_dx, grads) = chain
            .transformer_block_backward_full(&gx, &weights, &cache, &up(&g, t, d), eps, true)
            .unwrap();
        let dwq_gpu = chain.download(&grads.dwq).unwrap();

        // Finite differences of L = Σ block(x; Wq)⊙G over Wq.
        let loss = |wqp: &[f32]| -> f32 {
            cpu_block(
                &x,
                (&n1, &n2),
                (wqp, &wk, &wv, &wo),
                (&wg, &wu, &wd),
                (t, d, h),
                eps,
                true,
            )
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum()
        };
        let step = 1e-3f32;
        for idx in 0..d * d
        {
            let (mut wp, mut wm) = (wq.clone(), wq.clone());
            wp[idx] += step;
            wm[idx] -= step;
            let fd = (loss(&wp) - loss(&wm)) / (2.0 * step);
            assert!(
                (fd - dwq_gpu[idx]).abs() < 2e-2,
                "dWq[{idx}]: fd={fd} gpu={}",
                dwq_gpu[idx]
            );
        }
    }

    /// **A transformer block trains on the GPU.** Fit a fixed target with an MSE
    /// loss `L = ½‖block(x) − Y‖²` (so `dout = block(x) − Y`); each step runs the
    /// forward, the full backward (all 7 projection grads), and an AdamW update
    /// of every projection — entirely on the device. Asserts the loss falls well
    /// below the start. Skips if no adapter.
    #[test]
    fn transformer_block_trains_with_adamw() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (t, d, h) = (4usize, 6usize, 10usize);
        let eps = 1e-5f32;
        let gen = |n: usize, p: f32, a: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.041 + p).sin() * a).collect()
        };
        let x = gen(t * d, 0.0, 1.0);
        let y = gen(t * d, 7.0, 0.5); // target
        let n1: Vec<f32> = (0..d).map(|i| 0.7 + 0.02 * i as f32).collect();
        let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.01 * i as f32).collect();
        let up = |data: &[f32], r, c| chain.upload(data, r, c);
        let gx = up(&x, t, d);
        let gy = up(&y, t, d);
        let (gn1, gn2) = (up(&n1, 1, d), up(&n2, 1, d));
        // Trainable projections + their AdamW moments (start at zero).
        let mk = |data: Vec<f32>, r, c| {
            (
                up(&data, r, c),
                up(&vec![0.0; r * c], r, c),
                up(&vec![0.0; r * c], r, c),
            )
        };
        let (gwq, mq, vq) = mk(gen(d * d, 0.5, 0.3), d, d);
        let (gwk, mk_, vk) = mk(gen(d * d, 1.1, 0.3), d, d);
        let (gwv, mv, vv) = mk(gen(d * d, 1.7, 0.3), d, d);
        let (gwo, mo, vo) = mk(gen(d * d, 2.3, 0.3), d, d);
        let (gwg, mg, vg) = mk(gen(d * h, 2.9, 0.25), d, h);
        let (gwu, mu, vu) = mk(gen(d * h, 3.5, 0.25), d, h);
        let (gwd, md, vd) = mk(gen(h * d, 4.1, 0.25), h, d);
        let betas = (0.9f32, 0.999f32);

        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 1..=40u32
        {
            let weights = BlockWeights {
                norm1: &gn1,
                wq: &gwq,
                wk: &gwk,
                wv: &gwv,
                wo: &gwo,
                norm2: &gn2,
                wg: &gwg,
                wu: &gwu,
                wd: &gwd,
            };
            let (out, cache) = chain
                .transformer_block_forward_cached(&gx, &weights, eps, true)
                .unwrap();
            let out_cpu = chain.download(&out).unwrap();
            let l: f32 = out_cpu
                .iter()
                .zip(&y)
                .map(|(o, t)| 0.5 * (o - t) * (o - t))
                .sum();
            if step == 1
            {
                first = l;
            }
            last = l;
            // dL/dout = out − Y  (computed as a resident subtract via add of −Y).
            let neg_y = chain.sgd_step(&gy, &gy, 2.0).unwrap(); // gy − 2·gy = −gy
            let dout = chain.add(&out, &neg_y).unwrap();
            let (_dx, grads) = chain
                .transformer_block_backward_full(&gx, &weights, &cache, &dout, eps, true)
                .unwrap();
            let opt = |p: &GpuMatrix, gr: &GpuMatrix, m: &GpuMatrix, v: &GpuMatrix| {
                chain
                    .adamw_step(p, gr, m, v, 0.02, betas, 1e-8, 0.0, step)
                    .unwrap();
            };
            opt(&gwq, &grads.dwq, &mq, &vq);
            opt(&gwk, &grads.dwk, &mk_, &vk);
            opt(&gwv, &grads.dwv, &mv, &vv);
            opt(&gwo, &grads.dwo, &mo, &vo);
            opt(&gwg, &grads.dwg, &mg, &vg);
            opt(&gwu, &grads.dwu, &mu, &vu);
            opt(&gwd, &grads.dwd, &md, &vd);
        }
        assert!(last < first * 0.5, "block did not train: {first} → {last}");
    }

    /// Resident elementwise mul matches the CPU product; shape mismatch errors.
    #[test]
    fn resident_elementwise_mul() {
        let Some(chain) = GpuChain::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, -1.0, 0.5, 2.0];
        let ga = chain.upload(&a, 2, 2);
        let gb = chain.upload(&b, 2, 2);
        let out = chain.download(&chain.mul(&ga, &gb).unwrap()).unwrap();
        assert_eq!(out, vec![5.0, -2.0, 1.5, 8.0]);
        // Shape mismatch is an error, not a panic.
        let gc = chain.upload(&[1.0, 2.0], 1, 2);
        assert!(chain.add(&ga, &gc).is_err());
    }
}
