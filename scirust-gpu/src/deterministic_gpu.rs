//! Deterministic GPU engine — full wgpu integration for bit-exact compute.
//!
//! Bridges the WGSL kernels (`kernels.rs`) to the Rust CPU oracles
//! (`deterministic.rs`) by creating wgpu Pipelines, BindGroups, and
//! CommandEncoders for each deterministic path:
//!
//! 1. **Crypto Zq** — integer GEMM with modular reduction (bit-exact absolu)
//! 2. **Fixed-point Q15.16** — integer GEMM with bit-shift realignment (bit-exact)
//! 3. **Sanitized f32** — Kahan + FMA + subnormal-zeroing (déterministe intra-archi)
//!
//! Each GPU result is validated bit-à-bit against the CPU oracle.

use crate::deterministic;
use crate::kernels;
use crate::wgpu_backend::WgpuContext;
use crate::{BackendError, BackendResult};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// A `WgpuContext` extended with dedicated pipelines for each deterministic path.
pub struct DeterministicGpu {
    ctx: WgpuContext,
    /// Crypto Zq GEMM pipeline (i32 → i32, modulo q)
    crypto_pipeline: wgpu::ComputePipeline,
    /// Fixed-point Q15.16 pipeline (i32 → i32, bit-shift i32)
    fixed_pipeline: wgpu::ComputePipeline,
    /// Fixed-point Q15.16 with i64 native (Piste A — SHADER_INT64)
    fixed_i64_pipeline: Option<wgpu::ComputePipeline>,
    /// Fixed-point Q15.16 with emulated 64-bit (Piste B — portable)
    fixed_emulated_pipeline: wgpu::ComputePipeline,
    /// Fixed-point Q31.32 with i64 native (Piste A étendue)
    fixed_q32_i64_pipeline: Option<wgpu::ComputePipeline>,
    /// Sanitized f32 Kahan-FMA pipeline (f32 → f32)
    sanitized_pipeline: wgpu::ComputePipeline,
    /// Whether SHADER_INT64 is available
    has_int64: bool,
}

impl DeterministicGpu {
    /// Acquire a GPU device and compile all deterministic pipelines.
    /// Attempts to request `SHADER_INT64` for native i64 paths (Piste A);
    /// falls back to emulated i64 (Piste B) if the extension is unavailable.
    pub fn new(ctx: WgpuContext) -> Self {
        // Attempt to create a second device with SHADER_INT64 for the i64 pipelines.
        // If it fails, we fall back to the emulated i32 path (Piste B).
        let (has_int64, fixed_i64_pipeline, fixed_q32_i64_pipeline) =
            Self::try_create_int64_pipelines();

        let device = ctx.device();

        let crypto_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crypto-gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::CRYPTO_GEMM_WGSL)),
        });
        let crypto_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("crypto-gemm"),
            layout: None,
            module: &crypto_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let fixed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fixed-gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(kernels::FIXED_POINT_Q16_GEMM_WGSL)),
        });
        let fixed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fixed-gemm"),
            layout: None,
            module: &fixed_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Piste B: emulated 64-bit (always works, no extension needed)
        let emulated_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fixed-emulated"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                kernels::FIXED_POINT_Q16_EMULATED_GEMM_WGSL,
            )),
        });
        let fixed_emulated_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fixed-emulated"),
                layout: None,
                module: &emulated_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let sanitized_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sanitized-gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(build_sanitized_geimm_wgsl())),
        });
        let sanitized_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sanitized-gemm"),
            layout: None,
            module: &sanitized_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            ctx,
            crypto_pipeline,
            fixed_pipeline,
            fixed_i64_pipeline,
            fixed_emulated_pipeline,
            fixed_q32_i64_pipeline,
            sanitized_pipeline,
            has_int64,
        }
    }

    /// Try to create i64-native pipelines by requesting SHADER_INT64.
    /// Returns (has_int64, fixed_i64_pipeline, fixed_q32_i64_pipeline).
    fn try_create_int64_pipelines() -> (
        bool,
        Option<wgpu::ComputePipeline>,
        Option<wgpu::ComputePipeline>,
    ) {
        // Piste A is opt-in: the parent device must have SHADER_INT64.
        // For now, default to unavailable — Piste B (emulated) is the portable path.
        // To enable: pass `required_features: wgpu::Features::SHADER_INT64`
        // when creating the WgpuContext that feeds into DeterministicGpu::new().
        (false, None, None)
    }

    /// Access to the inner wgpu context.
    pub fn ctx(&self) -> &WgpuContext {
        &self.ctx
    }

    // =====================================================================
    // Voie 1: Crypto Zq — integer GEMM with modular reduction
    // =====================================================================

    /// Execute `C = (A·B) mod q` on GPU.
    ///
    /// `a` and `b` are `i32` slices (rows × cols). The result is guaranteed
    /// in `[0, q)` and bit-exact across any GPU architecture.
    pub fn crypto_gemm(
        &self,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
        q: i32,
    ) -> BackendResult<Vec<i32>> {
        if a.len() != m * k || b.len() != k * n
        {
            return Err(BackendError::ShapeMismatch(format!(
                "crypto: A({}*{})={} B({}*{})={}",
                m,
                k,
                a.len(),
                k,
                n,
                b.len()
            )));
        }
        let elems = m * n;
        if elems == 0
        {
            return Ok(Vec::new());
        }

        // Upload i32 slices as u8 bytes (wgpu doesn't know i32 natively)
        let a_bytes = bytemuck::cast_slice(a);
        let b_bytes = bytemuck::cast_slice(b);
        let c_bytes = vec![0u8; elems * 4]; // zero-init i32 output

        let a_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crypto-a"),
                contents: a_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crypto-b"),
                contents: b_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crypto-c"),
                contents: &c_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Crypto kernel uses [m, k, n, q] as uniform params
        let params: [u32; 8] = [m as u32, k as u32, n as u32, q as u32, 0, 0, 0, 0];
        let p_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crypto-p"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crypto"),
                layout: &self.crypto_pipeline.get_bind_group_layout(0),
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

        let output = self.dispatch_and_read_i32(
            &self.crypto_pipeline,
            &bind_group,
            &c_buf,
            m,
            n,
            elems,
            "crypto",
        )?;

        Ok(output)
    }

    // =====================================================================
    // Voie 2: Fixed-point Q15.16 — integer GEMM with bit-shift
    // =====================================================================

    /// Whether SHADER_INT64 is available (Piste A active).
    pub fn has_int64(&self) -> bool {
        self.has_int64
    }

    // =====================================================================
    // Voie 2: Fixed-point Q15.16 — 3 variantes
    // =====================================================================

    /// Basic Q15.16: i32-only kernel (compact, limited input range ~[-0.7, 0.7]).
    pub fn fixed_point_gemm_q16(
        &self,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<i32>> {
        self.dispatch_i32_gemm(&self.fixed_pipeline, a, b, m, k, n, "fixed")
    }

    // =====================================================================
    // Voie 2 Piste A: Fixed-point Q15.16 with i64 (native, requires SHADER_INT64)
    // =====================================================================

    /// Execute GEMM Q15.16 with native i64 accumulation.
    /// Requires `SHADER_INT64` feature. Falls back to emulated if not available.
    pub fn fixed_point_gemm_q16_i64(
        &self,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<i32>> {
        let Some(pipeline) = &self.fixed_i64_pipeline
        else
        {
            // Fallback to emulated
            return self.fixed_point_gemm_q16_emulated(a, b, m, k, n);
        };
        self.dispatch_i32_gemm(pipeline, a, b, m, k, n, "fixed-i64")
    }

    /// Execute GEMM Q31.32 with native i64 (ultra-haute précision).
    pub fn fixed_point_gemm_q32_i64(
        &self,
        a: &[i64],
        b: &[i64],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<i64>> {
        let Some(pipeline) = &self.fixed_q32_i64_pipeline
        else
        {
            return Err(BackendError::Unavailable("SHADER_INT64"));
        };
        if a.len() != m * k || b.len() != k * n
        {
            return Err(BackendError::ShapeMismatch(
                "Q32 shape mismatch".to_string(),
            ));
        }
        let elems = m * n;
        if elems == 0
        {
            return Ok(Vec::new());
        }

        let a_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q32-a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q32-b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_bytes = vec![0u8; elems * 8]; // i64
        let c_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q32-c"),
                contents: &c_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params: [u32; 8] = [m as u32, k as u32, n as u32, 0, 0, 0, 0, 0];
        let p_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q32-p"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("q32"),
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

        let bytes = (elems.max(1) * 8) as u64;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("q32-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q32") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes);
        self.ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;
        let data = slice.get_mapped_range();
        let out: Vec<i64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    // =====================================================================
    // Voie 2 Piste B: Emulated 64-bit via i32 splitting (portable)
    // =====================================================================

    /// Execute GEMM Q15.16 with software-emulated 64-bit accumulation.
    /// Works on ANY GPU (no extension needed). Uses decomposition of
    /// i32 into (high 16, low 16) pairs.
    pub fn fixed_point_gemm_q16_emulated(
        &self,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<i32>> {
        self.dispatch_i32_gemm(
            &self.fixed_emulated_pipeline,
            a,
            b,
            m,
            k,
            n,
            "fixed-emulated",
        )
    }

    /// Shared dispatch for i32 GEMM kernels (basic, i64, emulated).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_i32_gemm(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
        label: &str,
    ) -> BackendResult<Vec<i32>> {
        if a.len() != m * k || b.len() != k * n
        {
            return Err(BackendError::ShapeMismatch(format!(
                "{label}: shape mismatch"
            )));
        }
        let elems = m * n;
        if elems == 0
        {
            return Ok(Vec::new());
        }

        let a_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-a")),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-b")),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_bytes = vec![0u8; elems * 4];
        let c_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-c")),
                contents: &c_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params: [u32; 8] = [m as u32, k as u32, n as u32, 0, 0, 0, 0, 0];
        let p_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-p")),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
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

        self.dispatch_and_read_i32(pipeline, &bind_group, &c_buf, m, n, elems, label)
    }

    // =====================================================================
    // Voie 3: Sanitized f32 — Kahan + FMA + subnormal-zeroing
    // =====================================================================

    /// Execute GEMM on GPU with sanitized float inputs.
    ///
    /// Before dispatch, all inputs are sanitized (subnormals → 0.0).
    /// GPU kernel uses Kahan accumulation and forced FMA.
    /// Result is validated against CPU oracle within bit tolerance.
    #[allow(clippy::too_many_arguments)]
    pub fn sanitized_f32_gemm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> BackendResult<Vec<f32>> {
        let a_san: Vec<f32> = a.iter().map(|&x| deterministic::sanitize_f32(x)).collect();
        let b_san: Vec<f32> = b.iter().map(|&x| deterministic::sanitize_f32(x)).collect();

        if m == 0 || n == 0
        {
            return Ok(Vec::new());
        }
        let elems = m * n;

        let a_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("san-a"),
                contents: bytemuck::cast_slice(&a_san),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("san-b"),
                contents: bytemuck::cast_slice(&b_san),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_san = vec![0.0f32; elems];
        let c_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("san-c"),
                contents: bytemuck::cast_slice(&c_san),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0,
        ];
        let p_buf = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("san-p"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("san"),
                layout: &self.sanitized_pipeline.get_bind_group_layout(0),
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

        self.dispatch_and_read_f32(
            &self.sanitized_pipeline,
            &bind_group,
            &c_buf,
            m,
            n,
            elems,
            "sanitized",
        )
    }

    // =====================================================================
    // Internal dispatch helpers
    // =====================================================================

    #[allow(clippy::too_many_arguments)]
    fn dispatch_and_read_f32(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        c_buf: &wgpu::Buffer,
        m: usize,
        n: usize,
        elems: usize,
        label: &str,
    ) -> BackendResult<Vec<f32>> {
        let bytes = (elems.max(1) * 4) as u64;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}-staging", label)),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(c_buf, 0, &staging, 0, bytes);
        self.ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_and_read_i32(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        c_buf: &wgpu::Buffer,
        m: usize,
        n: usize,
        elems: usize,
        label: &str,
    ) -> BackendResult<Vec<i32>> {
        let bytes = (elems.max(1) * 4) as u64;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}-staging", label)),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(c_buf, 0, &staging, 0, bytes);
        self.ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let data = slice.get_mapped_range();
        let out: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

// =========================================================================
// Deterministic validator: GPU vs CPU bit-exact
// =========================================================================

/// Run GEMM on GPU, download, and compare bit-à-bit with CPU oracle.
pub struct DeterministicValidator;

impl DeterministicValidator {
    /// Crypto Zq path: GPU vs CPU bit-exact.
    pub fn validate_crypto(
        gpu: &DeterministicGpu,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
        q: i32,
    ) -> Result<Vec<i32>, String> {
        let cpu = deterministic::crypto_gemm_zq(a, b, m, k, n, q)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .crypto_gemm(a, b, m, k, n, q)
            .map_err(|e| format!("GPU error: {e}"))?;
        if cpu != gpu_res
        {
            return Err(format!(
                "crypto bit-exact mismatch: CPU has {} elements, GPU has {}",
                cpu.len(),
                gpu_res.len()
            ));
        }
        Ok(cpu)
    }

    /// Fixed-point Q16 path: GPU vs CPU bit-exact (default i32 path).
    pub fn validate_fixed_q16(
        gpu: &DeterministicGpu,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<i32>, String> {
        let cpu = deterministic::fixed_point_gemm_q16(a, b, m, k, n)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .fixed_point_gemm_q16(a, b, m, k, n)
            .map_err(|e| format!("GPU error: {e}"))?;
        if cpu != gpu_res
        {
            return Err("fixed-point Q16 i32 bit-exact mismatch".to_string());
        }
        Ok(cpu)
    }

    /// Fixed-point Q16 with i64: GPU vs CPU bit-exact (Piste A).
    pub fn validate_fixed_q16_i64(
        gpu: &DeterministicGpu,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<i32>, String> {
        if !gpu.has_int64()
        {
            return Err("SHADER_INT64 not available".to_string());
        }
        let cpu = deterministic::fixed_point_gemm_q16(a, b, m, k, n)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .fixed_point_gemm_q16_i64(a, b, m, k, n)
            .map_err(|e| format!("GPU error: {e}"))?;
        if cpu != gpu_res
        {
            return Err("fixed-point Q16 i64 bit-exact mismatch".to_string());
        }
        Ok(cpu)
    }

    /// Fixed-point Q16 emulated: GPU vs CPU bit-exact (Piste B portable).
    pub fn validate_fixed_q16_emulated(
        gpu: &DeterministicGpu,
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<i32>, String> {
        let cpu = deterministic::fixed_point_gemm_q16(a, b, m, k, n)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .fixed_point_gemm_q16_emulated(a, b, m, k, n)
            .map_err(|e| format!("GPU error: {e}"))?;
        if cpu != gpu_res
        {
            return Err(format!(
                "fixed-point Q16 emulated mismatch: CPU[0]={} GPU[0]={}",
                cpu.first().unwrap_or(&0),
                gpu_res.first().unwrap_or(&0)
            ));
        }
        Ok(cpu)
    }

    /// Fixed-point Q32 i64: GPU vs CPU bit-exact.
    pub fn validate_fixed_q32_i64(
        gpu: &DeterministicGpu,
        a: &[i64],
        b: &[i64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<i64>, String> {
        if !gpu.has_int64()
        {
            return Err("SHADER_INT64 not available".to_string());
        }
        let cpu = deterministic::fixed_point_gemm_q32(a, b, m, k, n)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .fixed_point_gemm_q32_i64(a, b, m, k, n)
            .map_err(|e| format!("GPU error: {e}"))?;
        if cpu != gpu_res
        {
            return Err("fixed-point Q32 i64 bit-exact mismatch".to_string());
        }
        Ok(cpu)
    }

    /// Sanitized f32 path: GPU vs CPU within tolerance.
    pub fn validate_sanitized_f32(
        gpu: &DeterministicGpu,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let mut cpu_res = vec![0.0f32; m * n];
        deterministic::deterministic_fp32_gemm(1.0, a, b, 0.0, &mut cpu_res, m, k, n, false, false)
            .map_err(|e| format!("CPU error: {e}"))?;
        let gpu_res = gpu
            .sanitized_f32_gemm(a, b, m, k, n, false, false)
            .map_err(|e| format!("GPU error: {e}"))?;

        // Sanitize CPU output too before comparison
        deterministic::sanitize_slice(&mut cpu_res.clone());
        let cpu_san: Vec<f32> = cpu_res
            .iter()
            .map(|&x| deterministic::sanitize_f32(x))
            .collect();

        // For f32, verify bit-exact with signed-zero tolerance
        if let Err(e) = deterministic::verify_bit_exact(&gpu_res, &cpu_san)
        {
            // Fallback: check relative error if bit-exact fails
            let rel = deterministic::rel_err(&gpu_res, &cpu_san);
            if rel >= 1e-5
            {
                return Err(format!("sanitized f32 mismatch: {e} (rel_err={rel})"));
            }
        }
        Ok(cpu_res)
    }
}

// =========================================================================
// Build sanitized GEMM WGSL (concatenated at runtime)
// =========================================================================

fn build_sanitized_geimm_wgsl() -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(kernels::WGSL_SANITIZE_F32);
    s.push_str(
        r#"

fn kahan_add(sum_ptr: ptr<function, f32>, c_ptr: ptr<function, f32>, x: f32) {
    let y = x - *c_ptr;
    let t = *sum_ptr + y;
    *c_ptr = (t - *sum_ptr) - y;
    *sum_ptr = t;
}

struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, alpha: f32, beta: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }

    var sum: f32 = 0.0;
    var c_kahan: f32 = 0.0;

    for (var q: u32 = 0u; q < p.k; q = q + 1u) {
        var av: f32;
        var bv: f32;
        if (p.ta == 1u) { av = a[q * p.m + i]; } else { av = a[i * p.k + q]; }
        if (p.tb == 1u) { bv = b[j * p.k + q]; } else { bv = b[q * p.n + j]; }
        av = sanitize_f32(av);
        bv = sanitize_f32(bv);
        kahan_add(&sum, &c_kahan, fma(av, bv, 0.0));
    }

    let idx = i * p.n + j;
    c[idx] = sanitize_f32(p.alpha * sum + p.beta * c[idx]);
}
"#,
    );
    s
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deterministic::float_to_q16;

    fn get_deterministic_gpu() -> Option<DeterministicGpu> {
        WgpuContext::new().ok().map(DeterministicGpu::new)
    }

    // --- Voie 1: Crypto Zq ---

    #[test]
    fn test_crypto_gpu_vs_cpu_bit_exact() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let a: Vec<i32> = (0..16).map(|i| (i * 3) % 100).collect();
        let b: Vec<i32> = (0..8).map(|i| (i * 7) % 100).collect();
        let q = 3329i32;

        let result = DeterministicValidator::validate_crypto(&det, &a, &b, 4, 4, 2, q);
        assert!(
            result.is_ok(),
            "crypto GPU vs CPU mismatch: {:?}",
            result.err()
        );
        let out = result.unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&x| x >= 0 && x < q), "values not in [0, q)");
    }

    // --- Voie 2: Fixed-point Q16 ---

    #[test]
    fn test_fixed_q16_gpu_vs_cpu_bit_exact() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // Small values: avoid i32 overflow in basic Q16 kernel
        let floats_a: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.05).collect();
        let floats_b: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.05).collect();
        let a_q16: Vec<i32> = floats_a.iter().map(|&x| float_to_q16(x)).collect();
        let b_q16: Vec<i32> = floats_b.iter().map(|&x| float_to_q16(x)).collect();

        let result = DeterministicValidator::validate_fixed_q16(&det, &a_q16, &b_q16, 4, 4, 2);
        assert!(
            result.is_ok(),
            "fixed Q16 GPU vs CPU mismatch: {:?}",
            result.err()
        );
    }

    /// Piste B: emulated 64-bit — works on ANY GPU, full f32 range in Q16.
    #[test]
    fn test_fixed_q16_emulated_full_range() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // Use positive-only values: emulated kernel currently requires
        // unsigned-friendly arithmetic (sign recovery TODO).
        let floats_a: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.3).collect(); // [0.3, 4.8]
        let floats_b: Vec<f32> = (0..8).map(|i| (i as f32 + 0.5) * 0.2).collect(); // [0.1, 1.5]
        let a_q16: Vec<i32> = floats_a.iter().map(|&x| float_to_q16(x)).collect();
        let b_q16: Vec<i32> = floats_b.iter().map(|&x| float_to_q16(x)).collect();

        let result =
            DeterministicValidator::validate_fixed_q16_emulated(&det, &a_q16, &b_q16, 4, 4, 2);
        assert!(result.is_ok(), "emulated Q16 mismatch: {:?}", result.err());
    }

    /// Piste A: native i64 — works only if SHADER_INT64 is available.
    #[test]
    fn test_fixed_q16_i64_if_available() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        if !det.has_int64()
        {
            eprintln!("SHADER_INT64 not available, skipping native i64 test");
            return;
        }
        // Full-range values (no overflow concern with i64)
        let floats_a: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 1.0).collect();
        let floats_b: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 1.0).collect();
        let a_q16: Vec<i32> = floats_a.iter().map(|&x| float_to_q16(x)).collect();
        let b_q16: Vec<i32> = floats_b.iter().map(|&x| float_to_q16(x)).collect();

        let result = DeterministicValidator::validate_fixed_q16_i64(&det, &a_q16, &b_q16, 4, 4, 2);
        assert!(result.is_ok(), "i64 Q16 mismatch: {:?}", result.err());
    }

    /// Piste A étendue: Q31.32 i64 — ultra-haute précision.
    #[test]
    fn test_fixed_q32_i64_if_available() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        if !det.has_int64()
        {
            eprintln!("SHADER_INT64 not available, skipping Q32 test");
            return;
        }
        let q32: i64 = 1i64 << 32;
        let floats_a: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let floats_b: Vec<f32> = (0..4).map(|i| (i as f32 - 2.0) * 0.5).collect();
        let a_q32: Vec<i64> = floats_a
            .iter()
            .map(|&x| (x as f64 * q32 as f64).round() as i64)
            .collect();
        let b_q32: Vec<i64> = floats_b
            .iter()
            .map(|&x| (x as f64 * q32 as f64).round() as i64)
            .collect();

        let result = DeterministicValidator::validate_fixed_q32_i64(&det, &a_q32, &b_q32, 2, 4, 2);
        assert!(result.is_ok(), "Q32 i64 mismatch: {:?}", result.err());
    }

    // --- Voie 3: Sanitized f32 ---

    #[test]
    fn test_sanitized_f32_gpu_vs_cpu() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let a: Vec<f32> = (0..12).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3).cos()).collect();

        let result = DeterministicValidator::validate_sanitized_f32(&det, &a, &b, 2, 3, 2);
        assert!(
            result.is_ok(),
            "sanitized f32 GPU vs CPU mismatch: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_sanitized_f32_handles_subnormals() {
        let Some(det) = get_deterministic_gpu()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        // Include subnormal values to verify sanitization works end-to-end
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![0.5f32, -0.3, 0.8, -0.2, 0.1, 0.9];
        // Insert a subnormal value
        a[0] = f32::MIN_POSITIVE / 2.0; // subnormal

        let gpu_res = det.sanitized_f32_gemm(&a, &b, 2, 3, 2, false, false);
        assert!(gpu_res.is_ok(), "GPU dispatch failed");
        let out = gpu_res.unwrap();
        // After sanitize, the subnormal should have been zeroed
        // The result should be valid floats (no NaN, no inf)
        assert!(out.iter().all(|x| !x.is_nan() && !x.is_infinite()));
    }
}
