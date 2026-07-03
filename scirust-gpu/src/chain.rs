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
