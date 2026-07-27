use core::fmt;
use std::{borrow::Cow, num::NonZeroU64, sync::mpsc};

use crate::{BackendError, BackendResult, RawComputeBackend, WgpuContext, check_gemm_dims};

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType,
    DeviceCapabilities, DeviceId, DeviceKind, KernelFormat, KernelModule, LaunchConfig,
    MemorySpace,
};

/// Adapter exposing a persistent [`WgpuContext`] through
/// `scirust_compute::ComputeBackend`.
///
/// The context owns one shared WGPU device and queue. Cloning the context does
/// not create another GPU device.
pub struct WgpuComputeAdapter {
    ctx: WgpuContext,
    capabilities: DeviceCapabilities,
    binding_alignment: usize,
    max_dispatch_size: u32,
}

impl WgpuComputeAdapter {
    /// Acquire a WGPU device and construct its compute-contract adapter.
    pub fn new() -> ComputeResult<Self> {
        let ctx =
            WgpuContext::new().map_err(|_| ComputeError::BackendUnavailable(DeviceKind::Wgpu))?;

        Ok(Self::from_context(ctx))
    }

    /// Wrap an already acquired context without creating another WGPU device.
    pub fn from_context(ctx: WgpuContext) -> Self {
        let limits = ctx.device().limits();

        let capabilities = DeviceCapabilities {
            device: DeviceId::new(DeviceKind::Wgpu, 0),
            name: format!("scirust-gpu-wgpu: {}", ctx.adapter_name()),
            supported_dtypes: vec![DType::U32, DType::I32, DType::F32],
            max_buffer_bytes: usize::try_from(limits.max_buffer_size).ok(),
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            supports_async_execution: true,
        };

        Self {
            ctx,
            capabilities,
            binding_alignment: limits.min_storage_buffer_offset_alignment as usize,
            max_dispatch_size: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Shared underlying WGPU context.
    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    /// Capabilities reported by the acquired device.
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

impl fmt::Debug for WgpuComputeAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuComputeAdapter")
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// Device-resident buffer owned by the WGPU adapter.
pub struct WgpuComputeBuffer {
    pub(crate) raw: wgpu::Buffer,
    pub(crate) bytes: usize,
    alignment: usize,
    memory_space: MemorySpace,
}

impl WgpuComputeBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn memory_space(&self) -> MemorySpace {
        self.memory_space
    }
}

impl fmt::Debug for WgpuComputeBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuComputeBuffer")
            .field("bytes", &self.bytes)
            .field("alignment", &self.alignment)
            .field("memory_space", &self.memory_space)
            .finish_non_exhaustive()
    }
}

/// Dynamically compiled WGSL compute pipeline.
pub struct WgpuComputeKernel {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) entry_point: String,
}

impl WgpuComputeKernel {
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

impl fmt::Debug for WgpuComputeKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuComputeKernel")
            .field("entry_point", &self.entry_point)
            .finish_non_exhaustive()
    }
}

/// Logical stream mapped to the adapter's single WGPU queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuComputeStream(());

/// Completion token returned by a WGPU queue submission.
#[derive(Debug, Clone)]
pub struct WgpuComputeEvent {
    pub(crate) submission: wgpu::SubmissionIndex,
}

impl RawComputeBackend for WgpuComputeAdapter {
    fn device_name(&self) -> &'static str {
        "wgpu"
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        let output_len = check_gemm_dims(a, b, m, k, n)?;
        let mut output = vec![0.0; output_len];

        if output_len == 0 || k == 0
        {
            return Ok(output);
        }

        self.ctx
            .gemm(1.0, a, b, 0.0, &mut output, m, k, n, false, false)
            .map_err(|error| match error
            {
                BackendError::Unavailable(name) => BackendError::Unavailable(name),
                BackendError::ShapeMismatch(message) => BackendError::ShapeMismatch(message),
                BackendError::Execution(message) => BackendError::Execution(message),
                BackendError::Unlicensed(message) => BackendError::Unlicensed(message),
            })?;

        Ok(output)
    }
}

fn checked_range(
    total_bytes: usize,
    offset_bytes: usize,
    length_bytes: usize,
    overflow_message: &'static str,
    bounds_message: &'static str,
) -> ComputeResult<usize> {
    let end = offset_bytes
        .checked_add(length_bytes)
        .ok_or_else(|| ComputeError::Transfer(overflow_message.to_string()))?;

    if end > total_bytes
    {
        return Err(ComputeError::Transfer(bounds_message.to_string()));
    }

    Ok(end)
}

fn require_copy_alignment(offset_bytes: usize, length_bytes: usize) -> ComputeResult<()> {
    if !offset_bytes.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
        || !length_bytes.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
    {
        return Err(ComputeError::Unsupported(
            "WGPU transfers require 4-byte-aligned offsets and lengths",
        ));
    }

    Ok(())
}

impl ComputeBackend for WgpuComputeAdapter {
    type Buffer = WgpuComputeBuffer;
    type Kernel = WgpuComputeKernel;
    type Stream = WgpuComputeStream;
    type Event = WgpuComputeEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer> {
        if alignment == 0
        {
            return Err(ComputeError::InvalidArgument(
                "buffer alignment must be non-zero",
            ));
        }
        if alignment != wgpu::COPY_BUFFER_ALIGNMENT as usize
        {
            return Err(ComputeError::Unsupported(
                "WGPU adapter currently supports 4-byte buffer alignment only",
            ));
        }
        if memory_space != MemorySpace::Device
        {
            return Err(ComputeError::Unsupported(
                "WGPU adapter supports device memory only",
            ));
        }
        if !bytes.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
        {
            return Err(ComputeError::Unsupported(
                "WGPU buffer sizes must be multiples of 4 bytes",
            ));
        }
        if let Some(maximum) = self.capabilities.max_buffer_bytes
        {
            if bytes > maximum
            {
                return Err(ComputeError::Allocation(format!(
                    "requested {bytes} bytes exceeds device limit {maximum}"
                )));
            }
        }

        let physical_bytes = bytes.max(wgpu::COPY_BUFFER_ALIGNMENT as usize);
        let size = u64::try_from(physical_bytes)
            .map_err(|error| ComputeError::Allocation(error.to_string()))?;

        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let raw = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("scirust-compute-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let validation = pollster::block_on(self.ctx.device().pop_error_scope());
        let out_of_memory = pollster::block_on(self.ctx.device().pop_error_scope());

        if let Some(error) = validation.or(out_of_memory)
        {
            return Err(ComputeError::Allocation(error.to_string()));
        }

        Ok(WgpuComputeBuffer {
            raw,
            bytes,
            alignment,
            memory_space,
        })
    }

    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
        checked_range(
            destination.bytes,
            offset_bytes,
            data.len(),
            "write range overflow",
            "write exceeds buffer bounds",
        )?;

        if data.is_empty()
        {
            return Ok(());
        }

        require_copy_alignment(offset_bytes, data.len())?;

        let offset = u64::try_from(offset_bytes)
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        self.ctx
            .queue()
            .write_buffer(&destination.raw, offset, data);

        if let Some(error) = pollster::block_on(self.ctx.device().pop_error_scope())
        {
            return Err(ComputeError::Transfer(error.to_string()));
        }

        Ok(())
    }

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        checked_range(
            source.bytes,
            offset_bytes,
            destination.len(),
            "read range overflow",
            "read exceeds buffer bounds",
        )?;

        if destination.is_empty()
        {
            return Ok(());
        }

        require_copy_alignment(offset_bytes, destination.len())?;

        let offset = u64::try_from(offset_bytes)
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;
        let length = u64::try_from(destination.len())
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("scirust-compute-readback"),
            size: length,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-compute-read"),
                });

        encoder.copy_buffer_to_buffer(&source.raw, offset, &staging, 0, length);
        let submission = self.ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self
            .ctx
            .device()
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        receiver
            .recv()
            .map_err(|error| ComputeError::Transfer(error.to_string()))?
            .map_err(|error| ComputeError::Transfer(error.to_string()))?;

        if let Some(error) = pollster::block_on(self.ctx.device().pop_error_scope())
        {
            return Err(ComputeError::Transfer(error.to_string()));
        }

        let mapped = slice.get_mapped_range();
        destination.copy_from_slice(&mapped);
        drop(mapped);
        staging.unmap();

        Ok(())
    }

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Wgsl
        {
            return Err(ComputeError::Unsupported(
                "WGPU adapter accepts WGSL kernels only",
            ));
        }

        let source = core::str::from_utf8(&module.code)
            .map_err(|error| ComputeError::Compilation(error.to_string()))?;

        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let shader = self
            .ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(module.entry_point.as_str()),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });

        let pipeline =
            self.ctx
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(module.entry_point.as_str()),
                    layout: None,
                    module: &shader,
                    entry_point: module.entry_point.as_str(),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        if let Some(error) = pollster::block_on(self.ctx.device().pop_error_scope())
        {
            return Err(ComputeError::Compilation(error.to_string()));
        }

        Ok(WgpuComputeKernel {
            pipeline,
            entry_point: module.entry_point.clone(),
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(WgpuComputeStream(()))
    }

    fn launch(
        &self,
        kernel: &Self::Kernel,
        _stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        if config.shared_memory_bytes != 0
        {
            return Err(ComputeError::Unsupported(
                "WGSL workgroup memory is declared statically in the shader",
            ));
        }
        if config.block != [1, 1, 1]
        {
            return Err(ComputeError::Unsupported(
                "WGSL workgroup size is encoded in the shader; use block [1, 1, 1]",
            ));
        }
        if config
            .grid
            .iter()
            .any(|dimension| *dimension > self.max_dispatch_size)
        {
            return Err(ComputeError::Launch(
                "dispatch grid exceeds the WGPU device limit".to_string(),
            ));
        }

        let mut ordered: Vec<_> = bindings.iter().collect();
        ordered.sort_by_key(|binding| binding.slot);

        if ordered.windows(2).any(|pair| pair[0].slot == pair[1].slot)
        {
            return Err(ComputeError::InvalidArgument(
                "buffer binding slots must be unique",
            ));
        }

        let mut entries = Vec::with_capacity(ordered.len());
        for binding in ordered
        {
            if binding.access == BufferAccess::WriteOnly
            {
                return Err(ComputeError::Unsupported(
                    "WGSL storage buffers do not support write-only bindings",
                ));
            }

            checked_range(
                binding.buffer.bytes,
                binding.offset_bytes,
                binding.length_bytes,
                "binding range overflow",
                "binding exceeds buffer bounds",
            )?;

            if binding.length_bytes == 0
            {
                return Err(ComputeError::InvalidArgument(
                    "buffer binding length must be non-zero",
                ));
            }
            if binding.offset_bytes % self.binding_alignment != 0
            {
                return Err(ComputeError::Unsupported(
                    "WGPU storage-buffer offsets must satisfy device alignment",
                ));
            }
            if !binding
                .length_bytes
                .is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
            {
                return Err(ComputeError::Unsupported(
                    "WGPU storage-buffer lengths must be multiples of 4 bytes",
                ));
            }

            let offset = u64::try_from(binding.offset_bytes)
                .map_err(|error| ComputeError::Launch(error.to_string()))?;
            let length = u64::try_from(binding.length_bytes)
                .map_err(|error| ComputeError::Launch(error.to_string()))?;
            let size = NonZeroU64::new(length).ok_or(ComputeError::InvalidArgument(
                "buffer binding length must be non-zero",
            ))?;

            entries.push(wgpu::BindGroupEntry {
                binding: binding.slot,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &binding.buffer.raw,
                    offset,
                    size: Some(size),
                }),
            });
        }

        self.ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let bind_group = if entries.is_empty()
        {
            None
        }
        else
        {
            Some(
                self.ctx
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(kernel.entry_point()),
                        layout: &kernel.pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    }),
            )
        };

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(kernel.entry_point()),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(kernel.entry_point()),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            if let Some(bind_group) = &bind_group
            {
                pass.set_bind_group(0, bind_group, &[]);
            }
            pass.dispatch_workgroups(config.grid[0], config.grid[1], config.grid[2]);
        }

        let submission = self.ctx.queue().submit(Some(encoder.finish()));

        if let Some(error) = pollster::block_on(self.ctx.device().pop_error_scope())
        {
            return Err(ComputeError::Launch(error.to_string()));
        }

        Ok(WgpuComputeEvent { submission })
    }

    fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
        let _ = self
            .ctx
            .device()
            .poll(wgpu::Maintain::WaitForSubmissionIndex(
                event.submission.clone(),
            ));
        Ok(())
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        let _ = self.ctx.device().poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use scirust_compute::DeviceKind;

    fn adapter_or_skip() -> Option<WgpuComputeAdapter> {
        WgpuComputeAdapter::new().ok()
    }

    #[test]
    fn adapter_reports_honest_wgpu_capabilities() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        let capabilities = adapter.capabilities();

        assert_eq!(capabilities.device.kind(), DeviceKind::Wgpu);
        assert!(capabilities.supports_dtype(DType::F32));
        assert!(capabilities.supports_dtype(DType::U32));
        assert!(capabilities.supports_async_execution);
        assert!(
            capabilities
                .max_workgroup_size
                .iter()
                .all(|value| *value > 0)
        );
    }

    #[test]
    fn device_buffer_round_trip_is_checked() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        let buffer = adapter
            .allocate(16, 4, MemorySpace::Device)
            .expect("device allocation");

        let input = [1_u32, 2, 3, 4];
        adapter
            .write(&buffer, 0, bytemuck::cast_slice(&input))
            .expect("device upload");

        let mut output = [0_u32; 4];
        adapter
            .read(&buffer, 0, bytemuck::cast_slice_mut(&mut output))
            .expect("device readback");

        assert_eq!(output, input);
        assert_eq!(buffer.len(), 16);
        assert_eq!(buffer.alignment(), 4);
        assert_eq!(buffer.memory_space(), MemorySpace::Device);

        assert!(adapter.write(&buffer, 2, &[0_u8; 4]).is_err());
        assert!(adapter.read(&buffer, 16, &mut [0_u8; 4]).is_err());
    }

    #[test]
    fn wgsl_kernel_executes_through_compute_contract() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        const WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    values[id.x] = values[id.x] + 1u;
}
"#;

        let buffer = adapter
            .allocate(16, 4, MemorySpace::Device)
            .expect("device allocation");

        let input = [10_u32, 20, 30, 40];
        adapter
            .write(&buffer, 0, bytemuck::cast_slice(&input))
            .expect("device upload");

        let module = KernelModule::new(KernelFormat::Wgsl, "main", WGSL.as_bytes().to_vec())
            .expect("valid kernel module");
        let kernel = adapter.compile(&module).expect("WGSL compilation");
        let stream = adapter.create_stream().expect("logical stream");
        let config =
            LaunchConfig::new([4, 1, 1], [1, 1, 1], 0).expect("valid launch configuration");

        let bindings = [BufferBinding {
            slot: 0,
            buffer: &buffer,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::ReadWrite,
        }];

        let event = adapter
            .launch(&kernel, &stream, config, &bindings)
            .expect("WGSL dispatch");

        adapter.wait(&event).expect("dispatch completion");

        let mut output = [0_u32; 4];
        adapter
            .read(&buffer, 0, bytemuck::cast_slice_mut(&mut output))
            .expect("device readback");

        assert_eq!(output, [11, 21, 31, 41]);
        adapter.synchronize(&stream).expect("queue synchronization");
    }

    #[test]
    fn write_only_bindings_are_rejected_honestly() {
        let Some(adapter) = adapter_or_skip()
        else
        {
            return;
        };

        const WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read_write> value: array<u32>;

@compute @workgroup_size(1)
fn main() {
    value[0] = 1u;
}
"#;

        let buffer = adapter
            .allocate(4, 4, MemorySpace::Device)
            .expect("device allocation");

        let module = KernelModule::new(KernelFormat::Wgsl, "main", WGSL.as_bytes().to_vec())
            .expect("valid kernel module");

        let kernel = adapter.compile(&module).expect("WGSL compilation");
        let stream = adapter.create_stream().expect("logical stream");
        let config =
            LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).expect("valid launch configuration");

        let bindings = [BufferBinding {
            slot: 0,
            buffer: &buffer,
            offset_bytes: 0,
            length_bytes: 4,
            access: BufferAccess::WriteOnly,
        }];

        assert!(matches!(
            adapter.launch(&kernel, &stream, config, &bindings),
            Err(ComputeError::Unsupported(
                "WGSL storage buffers do not support write-only bindings"
            ))
        ));
    }
}
