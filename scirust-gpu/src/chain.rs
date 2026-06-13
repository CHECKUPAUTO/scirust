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
}
