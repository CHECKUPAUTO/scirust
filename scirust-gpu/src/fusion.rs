//! Kernel fusion engine — combines multiple ops into a single GPU dispatch.
//!
//! Fusion eliminates intermediate VRAM allocations and kernel launch overhead
//! by compiling sequences like `GEMM → bias → activation` into a single WGSL
//! compute shader dispatch.
//!
//! The fusion planner takes a linear sequence of ops and merges compatible
//! adjacent operations into the fused GEMM kernel.

use crate::BackendResult;
use crate::kernels;
use crate::kernels::FusedAct;
use crate::wgpu_backend::WgpuContext;
use wgpu::util::DeviceExt;

/// A node in a fusion graph. GEMM nodes are fused with their downstream
/// bias-add and activation.
#[derive(Debug, Clone)]
pub enum FusionNode {
    /// GEMM: A · B, op(A) is m×k, op(B) is k×n
    Gemm {
        a_data: Vec<f32>,
        b_data: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        transpose_a: bool,
        transpose_b: bool,
    },
    /// Bias add (must follow a GEMM on the same shape)
    Bias(Vec<f32>),
    /// Activation
    Act(FusedAct),
}

/// A fused GEMM + bias + activation sequence, compiled into a single dispatch.
pub struct FusedLayer {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub act: FusedAct,
}

impl FusedLayer {
    /// Execute the fused layer on GPU. Returns the result in CPU memory.
    pub fn execute(
        &self,
        ctx: &WgpuContext,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        ta: bool,
        tb: bool,
    ) -> BackendResult<Vec<f32>> {
        let m = self.m;
        let n = self.n;
        let k = self.k;
        let elems = m * n;

        if elems == 0
        {
            return Ok(Vec::new());
        }

        let mut c = if let Some(bi) = bias
        {
            bi.to_vec()
        }
        else
        {
            vec![0.0f32; elems]
        };

        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;

        let a_buf = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused-a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused-b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buf = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused-c"),
                contents: bytemuck::cast_slice(&c),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            1.0f32.to_bits(), // alpha
            if bias.is_some()
            {
                1.0f32.to_bits()
            }
            else
            {
                0.0f32.to_bits()
            }, // beta
            self.act as u32,
        ];
        let p_buf = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused-p"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fused-gemm"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    kernels::FUSED_GEMM_WGSL,
                )),
            });
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fused-gemm"),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });

        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(16), (n as u32).div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes);
        ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| crate::BackendError::Unavailable("wgpu"))?
            .map_err(|_| crate::BackendError::Unavailable("wgpu"))?;

        let data = slice.get_mapped_range();
        c.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(c)
    }
}

/// Plan fusion for a linear layer: GEMM + optional bias + optional activation.
///
/// Returns a `FusedLayer` if fusion is possible, or `None` if the sequence
/// cannot be fused (e.g., shapes incompatible).
pub fn plan_fusion(m: usize, k: usize, n: usize, act: Option<FusedAct>) -> FusedLayer {
    FusedLayer {
        m,
        k,
        n,
        act: act.unwrap_or(FusedAct::None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_backend::WgpuContext;

    /// Independent CPU oracle for the fused kernel, derived directly from the
    /// `FUSED_GEMM_WGSL` source's index arithmetic (not from the tile-loading
    /// logic under test): `C = act(op(A)·op(B) + bias)`. This is exactly the
    /// GPU-vs-CPU-oracle validation the audit found missing — the previous
    /// tests short-circuited before ever dispatching the kernel, which is why
    /// a P0 indexing bug in the tiled accumulation went unnoticed.
    #[allow(clippy::too_many_arguments)]
    fn cpu_fused_oracle(
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
        act: FusedAct,
    ) -> Vec<f32> {
        let apply_act = |x: f32| -> f32 {
            match act
            {
                FusedAct::None => x,
                FusedAct::Relu => x.max(0.0),
                FusedAct::Gelu =>
                {
                    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
                },
                FusedAct::Silu => x * (1.0 / (1.0 + (-x).exp())),
                FusedAct::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                FusedAct::Tanh => x.tanh(),
            }
        };
        let mut out = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    // Matches FUSED_GEMM_WGSL's `a_idx`/`b_idx` `select(...)`
                    // exactly: ta/tb pick between a row-major (m,k)/(k,n)
                    // layout and its transposed (k,m)/(n,k) storage.
                    let a_val = if ta { a[p * m + i] } else { a[i * k + p] };
                    let b_val = if tb { b[j * k + p] } else { b[p * n + j] };
                    acc += a_val * b_val;
                }
                let beta_term = bias.map(|bi| bi[i * n + j]).unwrap_or(0.0);
                out[i * n + j] = apply_act(acc + beta_term);
            }
        }
        out
    }

    fn rel_err(gpu: &[f32], cpu: &[f32]) -> f32 {
        let num: f32 = gpu
            .iter()
            .zip(cpu)
            .map(|(g, c)| (g - c) * (g - c))
            .sum::<f32>()
            .sqrt();
        let den: f32 = cpu.iter().map(|c| c * c).sum::<f32>().sqrt().max(1e-30);
        num / den
    }

    #[test]
    fn test_fused_gemm_relu() {
        // Deliberately not a multiple of 16 (the tile size): this is exactly
        // the size class the audit found untested, and where a boundary
        // indexing bug would show up alongside the core accumulation bug.
        let (m, k, n) = (13usize, 19usize, 7usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31 - 3.0).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.17 + 1.0).cos()).collect();

        let Ok(ctx) = WgpuContext::new()
        else
        {
            eprintln!("wgpu: no adapter available, skipping fused GEMM+ReLU parity");
            return;
        };
        let layer = plan_fusion(m, k, n, Some(FusedAct::Relu));
        let gpu = layer.execute(&ctx, &a, &b, None, false, false).unwrap();
        let cpu = cpu_fused_oracle(&a, &b, None, m, k, n, false, false, FusedAct::Relu);
        assert_eq!(gpu.len(), cpu.len());
        assert!(rel_err(&gpu, &cpu) < 1e-3, "gpu={gpu:?} cpu={cpu:?}");
        // ReLU output must be non-negative on both paths.
        assert!(gpu.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_fused_gemm_gelu() {
        // With bias, and again a non-multiple-of-16 shape.
        let (m, k, n) = (20usize, 16usize, 11usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.23 - 2.0).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.11 + 0.5).cos()).collect();
        let bias: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();

        let Ok(ctx) = WgpuContext::new()
        else
        {
            eprintln!("wgpu: no adapter available, skipping fused GEMM+GELU parity");
            return;
        };
        let layer = plan_fusion(m, k, n, Some(FusedAct::Gelu));
        let gpu = layer
            .execute(&ctx, &a, &b, Some(&bias), false, false)
            .unwrap();
        let cpu = cpu_fused_oracle(&a, &b, Some(&bias), m, k, n, false, false, FusedAct::Gelu);
        assert_eq!(gpu.len(), cpu.len());
        assert!(rel_err(&gpu, &cpu) < 1e-3, "gpu={gpu:?} cpu={cpu:?}");
    }

    #[test]
    fn test_fused_gemm_transpose_and_bias() {
        // Exercises op(A) = Aᵀ together with bias and a third activation
        // (Silu), on a non-multiple-of-16 shape.
        let (m, k, n) = (9usize, 14usize, 17usize);
        // Stored transposed: A is (k, m) so that op(A) = Aᵀ is (m, k).
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.19 - 1.0).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13 + 0.2).cos()).collect();
        let bias: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.05).cos() * 0.2).collect();

        let Ok(ctx) = WgpuContext::new()
        else
        {
            eprintln!("wgpu: no adapter available, skipping fused transpose+bias parity");
            return;
        };
        let layer = plan_fusion(m, k, n, Some(FusedAct::Silu));
        let gpu = layer
            .execute(&ctx, &a, &b, Some(&bias), true, false)
            .unwrap();
        let cpu = cpu_fused_oracle(&a, &b, Some(&bias), m, k, n, true, false, FusedAct::Silu);
        assert_eq!(gpu.len(), cpu.len());
        assert!(rel_err(&gpu, &cpu) < 1e-3, "gpu={gpu:?} cpu={cpu:?}");
    }
}
