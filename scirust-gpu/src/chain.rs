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

use crate::BackendResult;
use crate::wgpu_backend::{GpuMatrix, WgpuContext};

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
