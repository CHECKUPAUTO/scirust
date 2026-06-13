// scirust-gpu/src/wgpu_backend.rs
//
// WgpuBackend — backend portable via wgpu (Vulkan/Metal/DX12/WebGL).
// Tourne sur :
//   - GPU dédiés NVIDIA/AMD/Intel
//   - GPU intégrés Apple Silicon (Metal)
//   - Navigateur (WebGPU)
//
// Activation : --features wgpu
//
// Comme cuda_backend.rs : structure correcte + un kernel saxpy fonctionnel.
// Les autres opérations délèguent au CPU pour l'instant — implémentation
// progressive en suivant le même pattern.

#![cfg(feature = "wgpu")]

use std::sync::OnceLock;
use wgpu::util::DeviceExt;

use scirust_core::matrix::backend::SimdBackend;
use scirust_core::matrix::view::{MatrixShape, MatrixView, MatrixViewMut};

// ------------------------------------------------------------------ //
//  Shader WGSL — saxpy                                                //
// ------------------------------------------------------------------ //

const SAXPY_WGSL: &str = r#"
struct Params {
    alpha: f32,
    n:     u32,
}

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read_write> y:      array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn saxpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = params.alpha * x[i] + y[i];
}
"#;

const RELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) { return; }
    data[i] = max(data[i], 0.0);
}
"#;

// ------------------------------------------------------------------ //
//  Contexte wgpu                                                       //
// ------------------------------------------------------------------ //

struct WgpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    saxpy_pipeline: wgpu::ComputePipeline,
    saxpy_layout: wgpu::BindGroupLayout,
    relu_pipeline: wgpu::ComputePipeline,
    relu_layout: wgpu::BindGroupLayout,
}

static CTX: OnceLock<Option<WgpuCtx>> = OnceLock::new();

fn try_ctx() -> Option<&'static WgpuCtx> {
    CTX.get_or_init(|| {
        // Initialisation bloquante via pollster pour rester sync
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;

            // Pipeline saxpy
            let saxpy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("saxpy"),
                source: wgpu::ShaderSource::Wgsl(SAXPY_WGSL.into()),
            });
            let saxpy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("saxpy_bgl"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
            let saxpy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("saxpy_pl"),
                bind_group_layouts: &[&saxpy_layout],
                push_constant_ranges: &[],
            });
            let saxpy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("saxpy_pipe"),
                layout: Some(&saxpy_pl),
                module: &saxpy_module,
                entry_point: "saxpy",
                compilation_options: Default::default(),
                cache: None,
            });

            // Pipeline relu
            let relu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("relu"),
                source: wgpu::ShaderSource::Wgsl(RELU_WGSL.into()),
            });
            let relu_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("relu_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let relu_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("relu_pl"),
                bind_group_layouts: &[&relu_layout],
                push_constant_ranges: &[],
            });
            let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("relu_pipe"),
                layout: Some(&relu_pl),
                module: &relu_module,
                entry_point: "relu",
                compilation_options: Default::default(),
                cache: None,
            });

            Some(WgpuCtx {
                device,
                queue,
                saxpy_pipeline,
                saxpy_layout,
                relu_pipeline,
                relu_layout,
            })
        })
    })
    .as_ref()
}

// ------------------------------------------------------------------ //
//  WgpuBackend                                                        //
// ------------------------------------------------------------------ //

pub struct WgpuBackend;

impl WgpuBackend {
    pub fn try_init() -> Option<Self> {
        try_ctx().map(|_| Self)
    }
}

impl SimdBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        let ctx = match try_ctx()
        {
            Some(c) => c,
            None =>
            {
                // Fallback CPU si init wgpu a échoué
                for (yi, xi) in y.iter_mut().zip(x.iter())
                {
                    *yi += alpha * xi;
                }
                return;
            },
        };

        let n = x.len() as u32;

        // Buffers
        let x_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x_buf"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let y_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("y_buf"),
                contents: bytemuck::cast_slice(y),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            alpha: f32,
            n: u32,
            _pad: [u32; 2],
        }
        let params = Params {
            alpha,
            n,
            _pad: [0; 2],
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("saxpy_bg"),
            layout: &ctx.saxpy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Encode + submit
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.saxpy_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (n + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Read back
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (y.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, (y.len() * 4) as u64);
        ctx.queue.submit(Some(enc.finish()));

        // Map + lecture (bloquant via pollster)
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map_async");

        let data = slice.get_mapped_range();
        y.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
    }

    // Les méthodes restantes : fallback CPU pour l'instant.
    // À implémenter en suivant le pattern saxpy ci-dessus.
    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        for (yi, xi) in y.iter_mut().zip(x.iter())
        {
            *yi += alpha * xi;
        }
    }
    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        let (m, _) = a.shape();
        for i in 0..m
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            y[i] = alpha * dot + beta * y[i];
        }
    }
    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        let (m, k) = a.shape();
        let (_, n) = b.shape();
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[(i, p)] * b[(p, j)];
                }
                c[(i, j)] = alpha * acc + beta * c[(i, j)];
            }
        }
    }
    fn relu_f32(&self, v: &mut [f32]) {
        // CPU fallback — TODO: dispatch via relu_pipeline (already created at init)
        for x in v.iter_mut()
        {
            *x = x.max(0.0);
        }
    }
}

#[cfg(not(feature = "wgpu"))]
pub struct WgpuBackend;
#[cfg(not(feature = "wgpu"))]
impl WgpuBackend {
    pub fn try_init() -> Option<Self> {
        None
    }
}
