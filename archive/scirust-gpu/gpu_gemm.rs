// scirust-gpu/src/gpu_gemm.rs
//
// SGEMM tuilée sur GPU via wgpu, exploitant la mémoire partagée
// (var<workgroup>) pour réduire les accès à la mémoire globale.
//
// Algorithme (référence CUTLASS / cuBLAS-lite) :
//
//   - Chaque workgroup calcule un bloc TILE_M × TILE_N de C
//   - Au sein du workgroup, on charge en mémoire partagée :
//       - un sous-bloc TILE_M × TILE_K de A
//       - un sous-bloc TILE_K × TILE_N de B
//   - Synchronisation (workgroupBarrier), puis chaque thread accumule
//     son output dans un registre local
//   - On itère sur la dimension K par paquets de TILE_K
//
// Constantes choisies pour adapter à wgpu (limites communes) :
//   TILE_M = TILE_N = 16, TILE_K = 16
//   workgroup_size = 16x16 = 256 threads
//   shared memory = 2 * 16 * 16 * 4 = 2 Ko (largement sous les 16 Ko mini)

#![cfg(feature = "wgpu")]

use crate::gpu_tensor::{GpuContext, GpuTensor};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ================================================================== //
//  Shader WGSL — SGEMM tuilée                                         //
// ================================================================== //

pub const SGEMM_TILED_WGSL: &str = r#"
// Constantes de tuilage
const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a: array<f32>;  // (m, k) row-major
@group(0) @binding(1) var<storage, read>       b: array<f32>;  // (k, n) row-major
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // (m, n) row-major
@group(0) @binding(3) var<uniform>             dims: Dims;

// Mémoire partagée par workgroup — partagée entre les 256 threads
var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

@compute @workgroup_size(16, 16)
fn sgemm_tiled(
    @builtin(workgroup_id)        wg_id:    vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = wg_id.y * TILE_M + local_id.y;  // ligne dans C
    let col = wg_id.x * TILE_N + local_id.x;  // colonne dans C

    let local_row = local_id.y;
    let local_col = local_id.x;

    // Accumulateur en registre (privé au thread)
    var acc: f32 = 0.0;

    // Itération sur K par tuiles de TILE_K
    let n_tiles = (dims.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {

        // ---- Phase 1 : chargement coopératif en mémoire partagée ----

        let a_col = t * TILE_K + local_col;
        if (row < dims.m && a_col < dims.k) {
            tile_a[local_row * TILE_K + local_col] = a[row * dims.k + a_col];
        } else {
            tile_a[local_row * TILE_K + local_col] = 0.0;
        }

        let b_row = t * TILE_K + local_row;
        if (b_row < dims.k && col < dims.n) {
            tile_b[local_row * TILE_N + local_col] = b[b_row * dims.n + col];
        } else {
            tile_b[local_row * TILE_N + local_col] = 0.0;
        }

        // Barrière : tous les threads ont fini de charger
        workgroupBarrier();

        // ---- Phase 2 : multiplication bloc × bloc en shared memory ----
        // Boucle déroulée par le compilateur (TILE_K = 16 connu)
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            acc = acc + tile_a[local_row * TILE_K + k]
                      * tile_b[k * TILE_N + local_col];
        }

        // Barrière avant de réécrire dans tile_a/tile_b à la prochaine itération
        workgroupBarrier();
    }

    // ---- Écriture dans C ----
    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = acc;
    }
}
"#;

// ================================================================== //
//  Pipeline + bind group layout                                       //
// ================================================================== //

pub struct SgemmPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl SgemmPipeline {
    pub fn build(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sgemm_tiled_module"),
            source: wgpu::ShaderSource::Wgsl(SGEMM_TILED_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sgemm_bgl"),
            entries: &[
                // a (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // c (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // dims (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sgemm_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sgemm_tiled"),
            layout: Some(&pl_layout),
            module: &module,
            entry_point: "sgemm_tiled",
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }
}

// ================================================================== //
//  API publique                                                       //
// ================================================================== //

/// C = A × B, sur GPU, A:(m,k), B:(k,n), C:(m,n).
/// Toutes les matrices doivent être en row-major contigu.
pub fn sgemm_gpu(
    ctx: &GpuContext,
    pipeline: &SgemmPipeline,
    a: &GpuTensor,
    b: &GpuTensor,
    c: &GpuTensor,
) {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    assert_eq!(k_a, k_b, "sgemm_gpu: K mismatch ({k_a} vs {k_b})");
    assert_eq!(c.shape(), (m, n), "sgemm_gpu: C shape mismatch");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Dims {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }
    let dims = Dims {
        m: m as u32,
        n: n as u32,
        k: k_a as u32,
        _pad: 0,
    };

    let dims_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sgemm_dims"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sgemm_bg"),
        layout: &pipeline.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        // Nombre de workgroups : un workgroup couvre 16×16 de C
        let groups_x = ((n + 15) / 16) as u32;
        let groups_y = ((m + 15) / 16) as u32;
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::Tensor;

    #[test]
    fn sgemm_gpu_correctness_or_skip() {
        let ctx = match GpuContext::try_init()
        {
            Some(c) => c,
            None =>
            {
                eprintln!("[skip] aucun GPU compatible wgpu");
                return;
            },
        };

        let pipeline = SgemmPipeline::build(&ctx.device);

        // A (3×4) × B (4×2) = C (3×2)
        let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        let a_cpu = Tensor::from_vec(a_data.clone(), 3, 4);
        let b_cpu = Tensor::from_vec(b_data.clone(), 4, 2);
        let c_cpu = Tensor::zeros(3, 2);

        let a_gpu = GpuTensor::from_cpu(&ctx, &a_cpu);
        let b_gpu = GpuTensor::from_cpu(&ctx, &b_cpu);
        let c_gpu = GpuTensor::from_cpu(&ctx, &c_cpu);

        sgemm_gpu(&ctx, &pipeline, &a_gpu, &b_gpu, &c_gpu);

        let c_result = c_gpu.to_cpu(&ctx);

        // Référence CPU
        let mut expected = vec![0.0f32; 6];
        for i in 0..3
        {
            for j in 0..2
            {
                let mut acc = 0.0;
                for p in 0..4
                {
                    acc += a_data[i * 4 + p] * b_data[p * 2 + j];
                }
                expected[i * 2 + j] = acc;
            }
        }

        for i in 0..6
        {
            assert!(
                (c_result.data[i] - expected[i]).abs() < 1e-3,
                "mismatch at {i}: gpu={} cpu={}",
                c_result.data[i],
                expected[i]
            );
        }
    }
}
