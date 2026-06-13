//! Real wgpu compute path (feature `wgpu`).
//!
//! Implements a row-major `f32` GEMM as a WGSL compute shader executed through
//! wgpu (Vulkan/Metal/DX12/GL). It is validated against the deterministic
//! [`crate::CpuBackend`] oracle with a documented floating-point tolerance —
//! GPU accumulation order is not bit-identical to the scalar CPU path, so the
//! oracle comparison is *bit-tolerant* by design (see `docs/GPU.md`, P2.2).
//!
//! Portability: this path is exercised in CI on a software Vulkan adapter
//! (Mesa *lavapipe*), so the "no claim without a test" rule is satisfied
//! without requiring physical GPU hardware.

use std::borrow::Cow;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::{BackendError, BackendResult};

/// WGSL: `C(m×n) = A(m×k) · B(k×n)`, row-major. One invocation per output cell.
const GEMM_WGSL: &str = r#"
struct Dims { m: u32, k: u32, n: u32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= dims.m || col >= dims.n) { return; }
    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < dims.k; p = p + 1u) {
        acc = acc + a[row * dims.k + p] * b[p * dims.n + col];
    }
    c[row * dims.n + col] = acc;
}
"#;

/// Run the GEMM on a wgpu device. Returns [`BackendError::Unavailable`] if no
/// adapter/device can be acquired (e.g. no Vulkan driver), never fabricated
/// output. Dimensions are assumed pre-validated by the caller.
pub(crate) fn wgpu_gemm(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> BackendResult<Vec<f32>> {
    // Degenerate shapes never touch the GPU (wgpu rejects zero-sized buffers).
    if m == 0 || n == 0
    {
        return Ok(Vec::new());
    }
    if k == 0
    {
        return Ok(vec![0.0f32; m * n]);
    }

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

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("scirust-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .map_err(|_| BackendError::Unavailable("wgpu"))?;

    let out_bytes = (m * n * std::mem::size_of::<f32>()) as u64;

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims = [m as u32, k as u32, n as u32, 0u32];
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm"),
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
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let gx = (m as u32).div_ceil(8);
        let gy = (n as u32).div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, out_bytes);
    queue.submit(Some(encoder.finish()));

    // Read back: map, block until the GPU + mapping complete, then copy out.
    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| BackendError::Unavailable("wgpu"))?
        .map_err(|_| BackendError::Unavailable("wgpu"))?;

    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
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
}
