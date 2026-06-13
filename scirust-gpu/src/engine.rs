//! Autograd-tape GPU engine (feature `wgpu`).
//!
//! [`WgpuEngine`] implements [`scirust_core::autodiff::reverse::GpuEngine`], the
//! hook the `Tape` calls for GPU-accelerated matmul forward/backward. Attach it
//! with `Tape::with_gpu_engine` / `set_gpu_engine`; `Var::matmul_gpu` then runs
//! its forward and backward GEMMs on the GPU.
//!
//! The device + pipeline are created once (in [`WgpuEngine::new`]) and reused
//! across the many GEMMs of a backward pass. If a particular dispatch fails
//! after the device was acquired, the engine falls back to a CPU GEMM with the
//! identical contract, so it never poisons the tape with wrong results.

use scirust_core::autodiff::reverse::GpuEngine;

use crate::wgpu_backend::WgpuContext;

/// A wgpu-backed [`GpuEngine`] for the autograd tape.
pub struct WgpuEngine {
    ctx: WgpuContext,
}

impl WgpuEngine {
    /// Acquire a GPU device and compile the GEMM pipeline. Returns `None` if no
    /// adapter is available (e.g. no Vulkan driver) — callers then keep the
    /// CPU-only tape rather than getting a fake engine.
    pub fn new() -> Option<Self> {
        WgpuContext::new().ok().map(|ctx| Self { ctx })
    }

    /// Name of the underlying adapter (e.g. `"llvmpipe (LLVM 20, 256 bits)"`).
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }
}

impl core::fmt::Debug for WgpuEngine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "WgpuEngine({})", self.ctx.adapter_name())
    }
}

impl GpuEngine for WgpuEngine {
    fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) {
        if self
            .ctx
            .gemm(alpha, a, b, beta, c, m, k, n, transpose_a, transpose_b)
            .is_err()
        {
            cpu_gemm(alpha, a, b, beta, c, m, k, n, transpose_a, transpose_b);
        }
    }
}

/// CPU reference GEMM with the exact [`GpuEngine::gemm`] contract — the safety
/// net used only if a GPU dispatch fails mid-flight.
#[allow(clippy::too_many_arguments)]
fn cpu_gemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) {
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f32;
            for q in 0..k
            {
                let av = if ta { a[q * m + i] } else { a[i * k + q] };
                let bv = if tb { b[j * k + q] } else { b[q * n + j] };
                acc += av * bv;
            }
            c[i * n + j] = alpha * acc + beta * c[i * n + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::{Tape, Tensor};

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

    /// `cpu_gemm` matches the GpuEngine contract (transpose + alpha/beta).
    #[test]
    fn cpu_fallback_contract() {
        // op(A)=Aᵀ (stored a is k×m=2×2), op(B)=I, alpha=2, beta=0.5.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 20.0, 30.0, 40.0];
        cpu_gemm(2.0, &a, &b, 0.5, &mut c, 2, 2, 2, true, false);
        assert_eq!(c, [7.0, 16.0, 19.0, 28.0]);
    }

    /// End-to-end: `matmul_gpu` forward + backward on the GPU engine must match
    /// the plain CPU `matmul` tape within tolerance. Skips if no adapter.
    #[test]
    fn tape_matmul_gpu_matches_cpu_forward_and_backward() {
        let Some(engine) = WgpuEngine::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        eprintln!("wgpu engine on: {}", engine.adapter_name());

        // A: 2×3, B: 3×2.
        let a_data = vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6];
        let b_data = vec![0.7f32, 0.8, -0.9, 1.0, 1.1, -1.2];

        // Reference: CPU tape.
        let cpu_tape = Tape::new();
        let a0 = cpu_tape.input(Tensor::from_vec(a_data.clone(), 2, 3));
        let b0 = cpu_tape.input(Tensor::from_vec(b_data.clone(), 3, 2));
        let c0 = a0.matmul(b0);
        let loss0 = c0.sum();
        cpu_tape.backward(loss0.idx());
        let cpu_c = cpu_tape.value(c0.idx());
        let cpu_ga = cpu_tape.grad(a0.idx());
        let cpu_gb = cpu_tape.grad(b0.idx());

        // GPU tape.
        let gpu_tape = Tape::new().with_gpu_engine(engine);
        let a1 = gpu_tape.input(Tensor::from_vec(a_data, 2, 3));
        let b1 = gpu_tape.input(Tensor::from_vec(b_data, 3, 2));
        let c1 = a1.matmul_gpu(b1);
        let loss1 = c1.sum();
        gpu_tape.backward(loss1.idx());
        let gpu_c = gpu_tape.value(c1.idx());
        let gpu_ga = gpu_tape.grad(a1.idx());
        let gpu_gb = gpu_tape.grad(b1.idx());

        assert!(rel_err(&gpu_c.data, &cpu_c.data) < 1e-4, "forward mismatch");
        assert!(rel_err(&gpu_ga.data, &cpu_ga.data) < 1e-4, "dA mismatch");
        assert!(rel_err(&gpu_gb.data, &cpu_gb.data) < 1e-4, "dB mismatch");
    }

    /// Conv2d forward + backward through the GPU engine (im2col GEMMs) must
    /// match the CPU path within tolerance. Skips if no adapter.
    #[test]
    fn conv2d_gpu_matches_cpu() {
        let Some(engine) = WgpuEngine::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let (batch, in_c, h, w, out_c, k, stride, pad) = (2usize, 2, 3, 3, 3, 2, 1, 0);
        let in_feats = in_c * h * w;
        let w_cols = in_c * k * k;
        let xs: Vec<f32> = (0..batch * in_feats)
            .map(|i| (i as f32 * 0.1 - 1.0).sin())
            .collect();
        let ws: Vec<f32> = (0..out_c * w_cols)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();

        // CPU reference tape.
        let ct = Tape::new();
        let cx = ct.input(Tensor::from_vec(xs.clone(), batch, in_feats));
        let cw = ct.input(Tensor::from_vec(ws.clone(), out_c, w_cols));
        let co = cx
            .try_conv2d_forward(cw, None, batch, in_c, h, w, out_c, k, stride, pad)
            .unwrap();
        ct.backward(co.sum().idx());
        let (c_out, c_gx, c_gw) = (ct.value(co.idx()), ct.grad(cx.idx()), ct.grad(cw.idx()));

        // GPU tape (im2col GEMMs routed through the engine).
        let gt = Tape::new().with_gpu_engine(engine);
        let gx = gt.input(Tensor::from_vec(xs, batch, in_feats));
        let gw = gt.input(Tensor::from_vec(ws, out_c, w_cols));
        let go = gx
            .try_conv2d_forward(gw, None, batch, in_c, h, w, out_c, k, stride, pad)
            .unwrap();
        gt.backward(go.sum().idx());
        let (g_out, g_gx, g_gw) = (gt.value(go.idx()), gt.grad(gx.idx()), gt.grad(gw.idx()));

        assert!(
            rel_err(&g_out.data, &c_out.data) < 1e-4,
            "conv forward mismatch"
        );
        assert!(
            rel_err(&g_gx.data, &c_gx.data) < 1e-4,
            "conv dInput mismatch"
        );
        assert!(
            rel_err(&g_gw.data, &c_gw.data) < 1e-4,
            "conv dWeight mismatch"
        );
    }
}
