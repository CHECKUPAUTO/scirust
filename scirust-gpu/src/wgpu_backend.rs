//! Real wgpu compute path (feature `wgpu`).
//!
//! Provides a general `f32` GEMM as a WGSL compute shader executed through wgpu
//! (Vulkan/Metal/DX12/GL):
//!
//! ```text
//! C = alpha * op(A) * op(B) + beta * C
//! ```
//!
//! with optional transposes — the exact contract of
//! [`scirust_core::autodiff::reverse::GpuEngine`], so the same kernel powers
//! both the standalone [`crate::WgpuBackend`] and the autograd-tape engine
//! ([`crate::WgpuEngine`], see `engine.rs`).
//!
//! Results are validated against the deterministic [`crate::CpuBackend`] oracle
//! within a documented floating-point tolerance (GPU accumulation order is not
//! bit-identical to the scalar CPU path). The path is exercised in CI on a
//! software Vulkan adapter (Mesa *lavapipe*), satisfying "no claim without a
//! test" without physical GPU hardware. See `docs/GPU.md` (P2.2).

use std::borrow::Cow;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::{BackendError, BackendResult};

/// General WGSL GEMM: `C = alpha·op(A)·op(B) + beta·C`, row-major, one
/// invocation per output cell. `op(A)` is `m×k`, `op(B)` is `k×n`, `C` is `m×n`;
/// `ta`/`tb` flag whether the *stored* `a`/`b` is the transpose of `op`.
const GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, alpha: f32, beta: f32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var acc: f32 = 0.0;
    for (var q: u32 = 0u; q < p.k; q = q + 1u) {
        var av: f32;
        var bv: f32;
        if (p.ta == 1u) { av = a[q * p.m + i]; } else { av = a[i * p.k + q]; }
        if (p.tb == 1u) { bv = b[j * p.k + q]; } else { bv = b[q * p.n + j]; }
        acc = acc + av * bv;
    }
    let idx = i * p.n + j;
    c[idx] = p.alpha * acc + p.beta * c[idx];
}
"#;

/// Elementwise kernel: `op` selects `0=add`, `1=mul` (binary, `a` and `b`),
/// `2=relu` (unary, `b` ignored), or `3=swiglu` (binary: `silu(a)·b`, the
/// SwiGLU gate — `silu(x) = x·σ(x)`). One invocation per element.
const EW_WGSL: &str = r#"
struct P { n: u32, op: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    if (p.op == 0u) { c[i] = a[i] + b[i]; }
    else if (p.op == 1u) { c[i] = a[i] * b[i]; }
    else if (p.op == 2u) { c[i] = max(a[i], 0.0); }
    else { c[i] = (a[i] / (1.0 + exp(-a[i]))) * b[i]; }
}
"#;

/// Token embedding gather: row `i` of the output is row `tokens[i]` of the
/// `vocab × d` `table`. One invocation per output element. Out-of-range tokens
/// are clamped to the last row (defensive; callers pass valid ids). The CPU
/// contract is [`crate::ops::cpu_embed`].
const EMBED_WGSL: &str = r#"
struct P { rows: u32, d: u32, vocab: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       table: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.rows * p.d) { return; }
    let row = i / p.d;
    let col = i % p.d;
    let tok = min(tokens[row], p.vocab - 1u);
    out[i] = table[tok * p.d + col];
}
"#;

/// Row-wise RMSNorm: `x / sqrt(mean(x²) + eps) · weight`, one invocation per
/// row over the row's `cols` elements; `weight` is a `cols`-length gain vector.
/// `eps` rides through the `u32` uniform as raw bits. The CPU contract is
/// [`crate::ops::cpu_rms_norm`].
const RMSNORM_WGSL: &str = r#"
struct P { rows: u32, cols: u32, eps_bits: u32, _p0: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var ss = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { let x = inp[base + j]; ss = ss + x * x; }
    let rms = sqrt(ss / f32(p.cols) + bitcast<f32>(p.eps_bits));
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { out[base + j] = inp[base + j] / rms * weight[j]; }
}
"#;

/// Row-wise softmax: one invocation per row computes
/// `exp(x - rowmax) / sum(exp(x - rowmax))` over the row's `cols` elements
/// (max-subtracted for stability). The missing transformer-attention primitive;
/// the CPU contract is [`crate::ops::cpu_softmax`].
const SOFTMAX_WGSL: &str = r#"
struct P { rows: u32, cols: u32, _p0: u32, _p1: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= p.rows) { return; }
    if (p.cols == 0u) { return; }
    let base = row * p.cols;
    var m = inp[base];
    for (var j: u32 = 1u; j < p.cols; j = j + 1u) { m = max(m, inp[base + j]); }
    var s = 0.0;
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { s = s + exp(inp[base + j] - m); }
    for (var j: u32 = 0u; j < p.cols; j = j + 1u) { out[base + j] = exp(inp[base + j] - m) / s; }
}
"#;

/// Pre-softmax attention step: scale a `rows × cols` score matrix by `scale`,
/// and — when `causal == 1` — overwrite every above-diagonal entry (key
/// `j > i` for query `i`) with the `-1e30` mask sentinel. 2D dispatch: `gid.x`
/// is the key column `j`, `gid.y` the query row `i`. `scale` is smuggled
/// through the `u32` uniform as raw bits and reconstructed with `bitcast`. The
/// CPU contract is [`crate::ops::cpu_scale_causal_mask`].
const MASK_WGSL: &str = r#"
struct P { rows: u32, cols: u32, causal: u32, scale_bits: u32, };

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;
    if (i >= p.rows || j >= p.cols) { return; }
    let idx = i * p.cols + j;
    if (p.causal == 1u && j > i) {
        out[idx] = -1.0e30;
    } else {
        out[idx] = inp[idx] * bitcast<f32>(p.scale_bits);
    }
}
"#;

/// A wgpu device + compiled compute pipelines, created once and reused across
/// calls (adapter/device acquisition and shader compilation are expensive).
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    ew_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    embed_pipeline: wgpu::ComputePipeline,
    adapter_name: String,
}

/// A row-major `f32` matrix resident in GPU memory (a storage buffer + shape).
///
/// Produced by [`crate::GpuChain`] (`upload` / `matmul`); an intermediate stays
/// in VRAM and feeds the next GEMM without a CPU round-trip.
pub struct GpuMatrix {
    buf: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

impl GpuMatrix {
    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl WgpuContext {
    /// Acquire an adapter/device and compile the GEMM pipeline. Returns
    /// [`BackendError::Unavailable`] if no adapter is available (e.g. no Vulkan
    /// driver) — never a silent fake.
    pub fn new() -> BackendResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(BackendError::Unavailable("wgpu"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scirust-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let ew_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ew"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EW_WGSL)),
        });
        let ew_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ew"),
            layout: None,
            module: &ew_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SOFTMAX_WGSL)),
        });
        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax"),
            layout: None,
            module: &softmax_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scale_causal_mask"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MASK_WGSL)),
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale_causal_mask"),
            layout: None,
            module: &mask_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMSNORM_WGSL)),
        });
        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rmsnorm"),
            layout: None,
            module: &rmsnorm_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let embed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embed"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EMBED_WGSL)),
        });
        let embed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embed"),
            layout: None,
            module: &embed_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            ew_pipeline,
            softmax_pipeline,
            mask_pipeline,
            rmsnorm_pipeline,
            embed_pipeline,
            adapter_name,
        })
    }

    /// Row-wise softmax of a row-major `rows × cols` matrix, matching
    /// [`crate::ops::cpu_softmax`]. Uploads the input, dispatches one thread per
    /// row, downloads the result.
    pub fn softmax_rows(&self, data: &[f32], rows: usize, cols: usize) -> BackendResult<Vec<f32>> {
        if data.len() != rows * cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "softmax: {} elems != {rows}×{cols}",
                data.len()
            )));
        }
        if data.is_empty()
        {
            return Ok(Vec::new());
        }
        let bytes = std::mem::size_of_val(data) as u64;
        let in_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax-in"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self._encode_softmax(&in_buf, &out_buf, rows, cols);
        self.download_buffer(&out_buf, data.len(), bytes)
    }

    /// Row-wise softmax of a **resident** `rows × cols` matrix, result kept in
    /// VRAM (no download) — the resident counterpart of [`Self::softmax_rows`]
    /// used to keep attention weights on-device across the score → weight →
    /// context chain.
    pub fn softmax_resident(&self, x: &GpuMatrix) -> BackendResult<GpuMatrix> {
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("softmax-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0 && x.cols > 0
        {
            self._encode_softmax(&x.buf, &out_buf, x.rows, x.cols);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one row-wise-softmax dispatch reading `in_buf`
    /// (`rows × cols`, row-major) and writing `out_buf`. Shared by the
    /// upload/download and resident entry points.
    fn _encode_softmax(
        &self,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, 0, 0];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax"),
            layout: &self.softmax_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Pre-softmax attention step on a row-major `rows × cols` score matrix:
    /// multiply by `scale`, and — when `causal` — replace every entry above the
    /// diagonal (key `j > i`) with the `-1e30` mask sentinel. Matches
    /// [`crate::ops::cpu_scale_causal_mask`]. One thread per score cell.
    pub fn scale_causal_mask(
        &self,
        scores: &[f32],
        rows: usize,
        cols: usize,
        scale: f32,
        causal: bool,
    ) -> BackendResult<Vec<f32>> {
        if scores.len() != rows * cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "scale_causal_mask: {} elems != {rows}×{cols}",
                scores.len()
            )));
        }
        if scores.is_empty()
        {
            return Ok(Vec::new());
        }
        let bytes = std::mem::size_of_val(scores) as u64;
        let in_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mask-in"),
                contents: bytemuck::cast_slice(scores),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self._encode_scale_causal_mask(&in_buf, &out_buf, rows, cols, scale, causal);
        self.download_buffer(&out_buf, scores.len(), bytes)
    }

    /// Scale + causal mask of a **resident** `rows × cols` score matrix, result
    /// kept in VRAM (no download) — the resident counterpart of
    /// [`Self::scale_causal_mask`]. `rows`/`cols` are `x.rows`/`x.cols`; the
    /// masking convention (query = row, key = column) is the same.
    pub fn scale_causal_mask_resident(
        &self,
        x: &GpuMatrix,
        scale: f32,
        causal: bool,
    ) -> BackendResult<GpuMatrix> {
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_scale_causal_mask(&x.buf, &out_buf, x.rows, x.cols, scale, causal);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one scale + causal-mask dispatch reading `in_buf`
    /// (`rows × cols`, row-major) and writing `out_buf`. Shared by the
    /// upload/download and resident entry points.
    fn _encode_scale_causal_mask(
        &self,
        in_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        scale: f32,
        causal: bool,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, causal as u32, scale.to_bits()];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mask-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask"),
            layout: &self.mask_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mask"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mask"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((cols as u32).div_ceil(8), (rows as u32).div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Row-wise RMSNorm of a **resident** `rows × cols` matrix, result kept in
    /// VRAM — `x / sqrt(mean(x²) + eps) · weight`, matching
    /// [`crate::ops::cpu_rms_norm`]. `weight` is a resident `cols`-length gain
    /// vector (any shape whose element count is `x.cols`).
    pub fn rms_norm_resident(
        &self,
        x: &GpuMatrix,
        weight: &GpuMatrix,
        eps: f32,
    ) -> BackendResult<GpuMatrix> {
        let weight_len = weight.rows * weight.cols;
        if weight_len != x.cols
        {
            return Err(BackendError::ShapeMismatch(format!(
                "rms_norm: weight has {weight_len} elems, expected cols = {}",
                x.cols
            )));
        }
        let elems = x.rows * x.cols;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmsnorm-res"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            self._encode_rms_norm(&x.buf, &weight.buf, &out_buf, x.rows, x.cols, eps);
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows: x.rows,
            cols: x.cols,
        })
    }

    /// Encode + submit one RMSNorm dispatch: `in_buf` (`rows × cols`) normalised
    /// row-wise and scaled by the `cols`-length `weight_buf`, written to
    /// `out_buf`. `eps` is passed as raw bits and reconstructed with `bitcast`.
    fn _encode_rms_norm(
        &self,
        in_buf: &wgpu::Buffer,
        weight_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) {
        let params: [u32; 4] = [rows as u32, cols as u32, eps.to_bits(), 0];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsnorm-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm"),
            layout: &self.rmsnorm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rmsnorm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rmsnorm_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Token embedding gather: build a resident `tokens.len() × d` matrix whose
    /// row `i` is row `tokens[i]` of the resident `vocab × d` `table`. Matches
    /// [`crate::ops::cpu_embed`]. The token ids are uploaded as a `u32` buffer;
    /// the gathered rows stay in VRAM to feed the transformer stack.
    pub fn embed_resident(&self, tokens: &[u32], table: &GpuMatrix) -> BackendResult<GpuMatrix> {
        let rows = tokens.len();
        let d = table.cols;
        let vocab = table.rows;
        let elems = rows * d;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embed-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if elems > 0
        {
            let tok_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-tokens"),
                    contents: bytemuck::cast_slice(tokens),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let params: [u32; 4] = [rows as u32, d as u32, vocab as u32, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embed-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed"),
                layout: &self.embed_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tok_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("embed"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.embed_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((elems as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: out_buf,
            rows,
            cols: d,
        })
    }

    /// Resident elementwise op: `op` is `0=add`, `1=mul` (binary), `2=relu`
    /// (unary), `3=swiglu` (binary: `silu(a)·b`). For binary ops `a` and `b`
    /// must share a shape; the result stays in VRAM. For relu, pass `b = a`
    /// (it is ignored).
    pub fn ew_resident(&self, a: &GpuMatrix, b: &GpuMatrix, op: u32) -> BackendResult<GpuMatrix> {
        if op != 2 && (a.rows != b.rows || a.cols != b.cols)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "elementwise: {}×{} vs {}×{}",
                a.rows, a.cols, b.rows, b.cols
            )));
        }
        let n = a.rows * a.cols;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if n > 0
        {
            let params: [u32; 4] = [n as u32, op, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ew-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ew"),
                layout: &self.ew_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
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
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ew") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ew"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ew_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(GpuMatrix {
            buf: c_buf,
            rows: a.rows,
            cols: a.cols,
        })
    }

    /// Access to the raw wgpu Device (for tensor, fusion, conv_gpu modules).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access to the raw wgpu Queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Name of the underlying adapter (e.g. `"llvmpipe (LLVM 20, 256 bits)"`).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Download a storage buffer back to CPU.
    pub fn download_buffer(
        &self,
        buf: &wgpu::Buffer,
        elems: usize,
        bytes: u64,
    ) -> BackendResult<Vec<f32>> {
        if elems == 0
        {
            return Ok(Vec::new());
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Encode + submit one GEMM dispatch (public for tensor.rs / fusion.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gemm(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
        alpha: f32,
        beta: f32,
    ) {
        // Reuse the private encode_gemm logic
        self._encode_gemm(a_buf, b_buf, c_buf, m, k, n, ta, tb, alpha, beta)
    }

    /// Elementwise resident op that takes GpuTensor-compatible buffers.
    pub fn ew_resident_tensor(
        &self,
        a: &crate::tensor::GpuTensor,
        b: &crate::tensor::GpuTensor,
        op: u32,
    ) -> BackendResult<crate::tensor::GpuTensor> {
        if op != 2 && (a.rows != b.rows || a.cols != b.cols)
        {
            return Err(BackendError::ShapeMismatch(format!(
                "elementwise shape mismatch: {}×{} vs {}×{}",
                a.rows, a.cols, b.rows, b.cols
            )));
        }
        let n = a.elems;
        let bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if n > 0
        {
            let params: [u32; 4] = [n as u32, op, 0, 0];
            let p_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ew-params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ew"),
                layout: &self.ew_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
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
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ew") });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ew"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ew_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(crate::tensor::GpuTensor {
            buf: std::sync::Arc::new(c_buf),
            rows: a.rows,
            cols: a.cols,
            elems: n,
        })
    }

    /// `C = alpha·op(A)·op(B) + beta·C`, writing the result back into `c`.
    ///
    /// `op(A)` is `m×k`, `op(B)` is `k×n`, `C` is `m×n`. When `ta`/`tb` is set,
    /// the stored `a`/`b` buffer is the transpose of the corresponding operand.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
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
    ) -> BackendResult<()> {
        if m == 0 || n == 0
        {
            return Ok(());
        }
        if k == 0
        {
            // No contraction: C = beta·C (handled on the host, no GPU work).
            for v in c.iter_mut()
            {
                *v *= beta;
            }
            return Ok(());
        }

        let bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c"),
                contents: bytemuck::cast_slice(c),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            alpha.to_bits(),
            beta.to_bits(),
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.pipeline.get_bind_group_layout(0),
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;

        let data = slice.get_mapped_range();
        c.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(())
    }

    /// Upload a row-major `rows×cols` matrix to a resident GPU storage buffer.
    pub fn upload(&self, data: &[f32], rows: usize, cols: usize) -> GpuMatrix {
        // wgpu rejects zero-sized buffers; back an empty matrix with a 4-byte
        // placeholder so the handle stays valid (`download` short-circuits empties).
        let buf = if data.is_empty()
        {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resident-empty"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }
        else
        {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("resident"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        GpuMatrix { buf, rows, cols }
    }

    /// `C = op(A)·op(B)` with both operands already resident; the result **stays
    /// in VRAM** (no download). `ta`/`tb` request a transpose of `a`/`b`. This
    /// is what keeps activations device-resident across a chain of GEMMs.
    pub fn gemm_resident(
        &self,
        a: &GpuMatrix,
        b: &GpuMatrix,
        ta: bool,
        tb: bool,
    ) -> BackendResult<GpuMatrix> {
        let m = if ta { a.cols } else { a.rows };
        let k = if ta { a.rows } else { a.cols };
        let n = if tb { b.rows } else { b.cols };
        let kb = if tb { b.cols } else { b.rows };
        if k != kb
        {
            return Err(BackendError::ShapeMismatch(format!(
                "inner dims disagree: op(A) is {m}×{k}, op(B) is {kb}×{n}"
            )));
        }
        // Never create a zero-sized buffer (wgpu rejects it). For a degenerate
        // result (`m`/`n`/`k == 0`) the zero-initialised buffer already holds
        // the correct empty/all-zeros matrix, so we skip the dispatch entirely.
        let elems = m * n;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Fresh result: alpha=1, beta=0. wgpu zero-initialises c_buf, so the
        // `beta·C` term reads valid zeros.
        if m != 0 && n != 0 && k != 0
        {
            self._encode_gemm(&a.buf, &b.buf, &c_buf, m, k, n, ta, tb, 1.0, 0.0);
        }
        Ok(GpuMatrix {
            buf: c_buf,
            rows: m,
            cols: n,
        })
    }

    /// Download a resident matrix back to a CPU `Vec<f32>` (row-major).
    pub fn download(&self, mat: &GpuMatrix) -> BackendResult<Vec<f32>> {
        let elems = mat.rows * mat.cols;
        if elems == 0
        {
            return Ok(Vec::new());
        }
        let bytes = (elems * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        encoder.copy_buffer_to_buffer(&mat.buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Unavailable("wgpu"))?
            .map_err(|_| BackendError::Unavailable("wgpu"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Encode + submit one GEMM dispatch into the given buffers (no download).
    #[allow(clippy::too_many_arguments)]
    fn _encode_gemm(
        &self,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        c_buf: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
        alpha: f32,
        beta: f32,
    ) {
        let params: [u32; 8] = [
            m as u32,
            k as u32,
            n as u32,
            ta as u32,
            tb as u32,
            alpha.to_bits(),
            beta.to_bits(),
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemm"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

/// One-shot row-major `C = A·B`. Acquires a fresh [`WgpuContext`]; for repeated
/// calls (e.g. an autograd backward pass) prefer a cached [`crate::WgpuEngine`].
pub fn wgpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> BackendResult<Vec<f32>> {
    if m == 0 || n == 0
    {
        return Ok(Vec::new());
    }
    let mut c = vec![0.0f32; m * n];
    if k == 0
    {
        return Ok(c);
    }
    WgpuContext::new()?.gemm(1.0, a, b, 0.0, &mut c, m, k, n, false, false)?;
    Ok(c)
}

/// One-shot row-wise softmax over a `rows × cols` matrix. Acquires a fresh
/// [`WgpuContext`]; for repeated calls prefer holding a context.
pub fn wgpu_softmax(data: &[f32], rows: usize, cols: usize) -> BackendResult<Vec<f32>> {
    WgpuContext::new()?.softmax_rows(data, rows, cols)
}

/// One-shot scale + causal mask over a `rows × cols` score matrix. Acquires a
/// fresh [`WgpuContext`]; for repeated calls prefer holding a context.
pub fn wgpu_scale_causal_mask(
    scores: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
    causal: bool,
) -> BackendResult<Vec<f32>> {
    WgpuContext::new()?.scale_causal_mask(scores, rows, cols, scale, causal)
}

#[cfg(test)]
mod tests {
    use crate::{CpuBackend, GpuAccelerator, RawComputeBackend, WgpuBackend};

    /// Maximum |gpu - cpu| relative to the CPU Frobenius norm. GPU accumulation
    /// is not bit-identical to the scalar oracle, so we assert a tolerance.
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

    /// If no adapter is available in this environment, skip rather than fail —
    /// CI provides a software Vulkan adapter (lavapipe) so the assertion path
    /// is actually exercised there.
    fn run_or_skip(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
        match super::wgpu_gemm(a, b, m, k, n)
        {
            Ok(v) => Some(v),
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping");
                None
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn wgpu_gemm_matches_cpu_oracle() {
        // A (3×4) · B (4×2), values chosen to be non-trivial.
        let a: Vec<f32> = (0..12).map(|i| (i as f32 * 0.5 - 2.0).sin()).collect();
        let b: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3 + 1.0).cos()).collect();
        if let Some(gpu) = run_or_skip(&a, &b, 3, 4, 2)
        {
            let cpu = CpuBackend.gemm_f32(&a, &b, 3, 4, 2).unwrap();
            assert_eq!(gpu.len(), cpu.len());
            assert!(rel_err(&gpu, &cpu) < 1e-4, "gpu={gpu:?} cpu={cpu:?}");
        }
    }

    #[test]
    fn wgpu_gemm_identity_roundtrip() {
        let a = [1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let id = [1.0f32, 0.0, 0.0, 1.0];
        if let Some(gpu) = run_or_skip(&a, &id, 2, 2, 2)
        {
            assert!(rel_err(&gpu, &a) < 1e-5);
        }
    }

    #[test]
    fn wgpu_backend_wired_under_feature() {
        // With the feature on, the WgpuBackend dispatches to the real path:
        // either a correct result (adapter present) or an honest Unavailable.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let id = [1.0f32, 0.0, 0.0, 1.0];
        match WgpuBackend.gemm_f32(&a, &id, 2, 2, 2)
        {
            Ok(v) =>
            {
                let cpu = CpuBackend.gemm_f32(&a, &id, 2, 2, 2).unwrap();
                assert!(rel_err(&v, &cpu) < 1e-5);
                assert_eq!(GpuAccelerator::Wgpu(WgpuBackend).device_name(), "wgpu");
            },
            Err(crate::BackendError::Unavailable("wgpu")) =>
            {},
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }

    /// GPU row-wise softmax must match the CPU oracle. Exercised against
    /// lavapipe in CI and the real GPU on-device; skipped where no adapter is
    /// present (this dev container has no Vulkan ICD).
    #[test]
    fn wgpu_softmax_matches_cpu_oracle() {
        let (rows, cols) = (4usize, 7usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.37 - 3.0).sin() * 2.0)
            .collect();
        match super::wgpu_softmax(&data, rows, cols)
        {
            Ok(gpu) =>
            {
                let cpu = crate::ops::cpu_softmax(&data, rows, cols);
                assert_eq!(gpu.len(), cpu.len());
                assert!(rel_err(&gpu, &cpu) < 1e-4, "gpu={gpu:?} cpu={cpu:?}");
                // Each row is a probability distribution.
                for r in 0..rows
                {
                    let s: f32 = gpu[r * cols..r * cols + cols].iter().sum();
                    assert!((s - 1.0).abs() < 1e-3, "row {r} sums to {s}");
                }
            },
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping softmax parity");
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    /// GPU scale + causal mask must match the CPU oracle, including the exact
    /// `-1e30` sentinel written above the diagonal. Exercised on lavapipe in CI
    /// and the real GPU on-device; skipped where no adapter is present.
    #[test]
    fn wgpu_scale_causal_mask_matches_cpu_oracle() {
        let (rows, cols) = (5usize, 5usize);
        let scores: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.21 - 2.0).cos() * 3.0)
            .collect();
        let scale = 0.125_f32; // 1/sqrt(64), a realistic 1/sqrt(head_dim).
        match super::wgpu_scale_causal_mask(&scores, rows, cols, scale, true)
        {
            Ok(gpu) =>
            {
                let cpu = crate::ops::cpu_scale_causal_mask(&scores, rows, cols, scale, true);
                assert_eq!(gpu.len(), cpu.len());
                // Above-diagonal entries are the exact sentinel on both paths.
                for i in 0..rows
                {
                    for j in 0..cols
                    {
                        let idx = i * cols + j;
                        if j > i
                        {
                            assert_eq!(gpu[idx], crate::ops::MASK_NEG, "masked ({i},{j})");
                        }
                        else
                        {
                            assert!(
                                (gpu[idx] - cpu[idx]).abs() < 1e-5,
                                "kept ({i},{j}): gpu={} cpu={}",
                                gpu[idx],
                                cpu[idx]
                            );
                        }
                    }
                }
                // Non-causal path is a pure scale.
                let gpu_ns =
                    super::wgpu_scale_causal_mask(&scores, rows, cols, scale, false).unwrap();
                let cpu_ns = crate::ops::cpu_scale_causal_mask(&scores, rows, cols, scale, false);
                assert!(rel_err(&gpu_ns, &cpu_ns) < 1e-5);
            },
            Err(crate::BackendError::Unavailable(_)) =>
            {
                eprintln!("wgpu: no adapter available, skipping mask parity");
            },
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    /// The general kernel must honour transpose + alpha/beta, matching a hand
    /// computation. C = 2·Aᵀ·B + 0.5·C0.
    #[test]
    fn wgpu_gemm_transpose_alpha_beta() {
        let ctx = match super::WgpuContext::new()
        {
            Ok(c) => c,
            Err(_) =>
            {
                eprintln!("wgpu: no adapter, skipping");
                return;
            },
        };
        // op(A) = Aᵀ where stored A is k×m. Take A stored as 2×2 → op(A) 2×2.
        // stored a (k×m = 2×2) = [[1,2],[3,4]] → op(A)=Aᵀ=[[1,3],[2,4]]
        // b (k×n = 2×2) = [[1,0],[0,1]] (identity) → op(A)·op(B) = op(A)
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [10.0f32, 20.0, 30.0, 40.0];
        ctx.gemm(2.0, &a, &b, 0.5, &mut c, 2, 2, 2, true, false)
            .unwrap();
        // 2·[[1,3],[2,4]] + 0.5·[[10,20],[30,40]] = [[7,16],[19,28]]
        let expected = [7.0f32, 16.0, 19.0, 28.0];
        let err: f32 = c
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-3, "got {c:?}");
    }
}
