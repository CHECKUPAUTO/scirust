// scirust-gpu/src/gpu_tensor.rs
//
// GpuTensor — tenseur résident sur GPU.
//
// Problème résolu : dans v3, chaque appel à un backend GPU faisait
// un round-trip CPU→GPU→CPU des données, ce qui annule les gains GPU.
// Avec GpuTensor, les données restent en VRAM entre les opérations ;
// on n'effectue le download que lorsque l'utilisateur appelle to_cpu().
//
// API :
//   let g = GpuTensor::from_cpu(&backend, &cpu_tensor)?;
//   let h = g.relu_inplace(&backend)?;        // pas de transfert
//   let cpu = h.to_cpu(&backend)?;            // download final
//
// Cette première version cible wgpu (portabilité) ; le pattern est
// identique pour CUDA en remplaçant Buffer par DeviceBuffer.

#![cfg(feature = "wgpu")]

use std::sync::Arc;
use wgpu::util::DeviceExt;
use scirust_core::autodiff::reverse::Tensor;

// ================================================================== //
//  GpuTensor                                                          //
// ================================================================== //

pub struct GpuTensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub rows:   usize,
    pub cols:   usize,
}

impl GpuTensor {
    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
    pub fn len(&self) -> usize { self.rows * self.cols }
    pub fn byte_size(&self) -> u64 { (self.len() * 4) as u64 }
}

// ================================================================== //
//  GpuContext — partagé entre tous les GpuTensor                      //
// ================================================================== //

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    // Pipelines compilés une fois et réutilisés
    pub relu_pipeline: wgpu::ComputePipeline,
    pub relu_bgl:      wgpu::BindGroupLayout,
    pub axpy_pipeline: wgpu::ComputePipeline,
    pub axpy_bgl:      wgpu::BindGroupLayout,
}

impl GpuContext {
    pub fn try_init() -> Option<Arc<Self>> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await?;

            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor::default(), None,
            ).await.ok()?;

            // ----- Pipeline ReLU ----- //
            let relu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("relu_module"),
                source: wgpu::ShaderSource::Wgsl(RELU_WGSL.into()),
            });
            let relu_bgl = make_storage_bgl(&device, "relu_bgl", 1, false);
            let relu_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("relu_pl"),
                bind_group_layouts: &[&relu_bgl],
                push_constant_ranges: &[],
            });
            let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("relu_pipe"),
                layout: Some(&relu_pl),
                module: &relu_module,
                entry_point: "relu_main",
                compilation_options: Default::default(),
                cache: None,
            });

            // ----- Pipeline AXPY ----- //
            let axpy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("axpy_module"),
                source: wgpu::ShaderSource::Wgsl(AXPY_WGSL.into()),
            });
            let axpy_bgl = make_axpy_bgl(&device);
            let axpy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("axpy_pl"),
                bind_group_layouts: &[&axpy_bgl],
                push_constant_ranges: &[],
            });
            let axpy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("axpy_pipe"),
                layout: Some(&axpy_pl),
                module: &axpy_module,
                entry_point: "axpy_main",
                compilation_options: Default::default(),
                cache: None,
            });

            Some(Arc::new(GpuContext {
                device, queue,
                relu_pipeline, relu_bgl,
                axpy_pipeline, axpy_bgl,
            }))
        })
    }
}

fn make_storage_bgl(device: &wgpu::Device, label: &str, n: u32, read_only: bool)
    -> wgpu::BindGroupLayout
{
    let entries: Vec<_> = (0..n).map(|i| wgpu::BindGroupLayoutEntry {
        binding: i, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label), entries: &entries,
    })
}

fn make_axpy_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("axpy_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// ================================================================== //
//  Shaders WGSL                                                        //
// ================================================================== //

const RELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn relu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) { return; }
    data[i] = max(data[i], 0.0);
}
"#;

const AXPY_WGSL: &str = r#"
struct Params { alpha: f32, n: u32, _pad: vec2<u32> };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform>             p: Params;

@compute @workgroup_size(64)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = p.alpha * x[i] + y[i];
}
"#;

// ================================================================== //
//  Opérations GpuTensor — pas de transfert !                          //
// ================================================================== //

impl GpuTensor {
    /// Upload CPU → GPU (allocation + copy unique).
    pub fn from_cpu(ctx: &GpuContext, t: &Tensor) -> Self {
        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_tensor"),
            contents: bytemuck::cast_slice(&t.data),
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
        });
        Self { buffer: Arc::new(buf), rows: t.rows, cols: t.cols }
    }

    /// Download GPU → CPU. Bloquant.
    pub fn to_cpu(&self, ctx: &GpuContext) -> Tensor {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_dl"),
            size: self.byte_size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.byte_size());
        ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map_async");

        let view = slice.get_mapped_range();
        let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.unmap();

        Tensor::from_vec(data, self.rows, self.cols)
    }

    /// ReLU in-place — aucune copie CPU.
    pub fn relu_inplace(&self, ctx: &GpuContext) {
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relu_bg"),
            layout: &ctx.relu_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.relu_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.len() as u32 + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }

    /// AXPY in-place : self += alpha * other  (pas de transfert)
    pub fn axpy_inplace(&self, ctx: &GpuContext, alpha: f32, other: &GpuTensor) {
        assert_eq!(self.shape(), other.shape());
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { alpha: f32, n: u32, _pad: [u32; 2] }
        let params = Params { alpha, n: self.len() as u32, _pad: [0; 2] };
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axpy_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axpy_bg"),
            layout: &ctx.axpy_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: other.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.axpy_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (self.len() as u32 + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }

    /// Clone GPU→GPU (pas de transfert CPU)
    pub fn clone_gpu(&self, ctx: &GpuContext) -> Self {
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_clone"),
            size: self.byte_size(),
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &buf, 0, self.byte_size());
        ctx.queue.submit(Some(enc.finish()));
        Self { buffer: Arc::new(buf), rows: self.rows, cols: self.cols }
    }
}

// ================================================================== //
//  Stub sans wgpu                                                     //
// ================================================================== //

#[cfg(not(feature = "wgpu"))]
pub struct GpuTensor;
#[cfg(not(feature = "wgpu"))]
pub struct GpuContext;
#[cfg(not(feature = "wgpu"))]
impl GpuContext {
    pub fn try_init() -> Option<std::sync::Arc<Self>> { None }
}
