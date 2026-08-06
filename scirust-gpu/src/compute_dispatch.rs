extern crate alloc;

use alloc::{sync::Arc, vec::Vec};

use scirust_compute::{
    BufferBinding, ComputeBackend, ComputeError, ComputeResult, DeviceCapabilities, DeviceKind,
    KernelModule, LaunchConfig, MemorySpace,
};

use crate::compute_adapter::{CpuBuffer, CpuComputeAdapter, CpuEvent, CpuKernel, CpuStream};

#[cfg(feature = "cuda")]
use crate::cuda_compute_adapter::{
    CudaComputeAdapter, CudaComputeBuffer, CudaComputeEvent, CudaComputeKernel, CudaComputeStream,
};

#[cfg(feature = "wgpu")]
use crate::wgpu_compute_adapter::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuComputeStream,
};

/// Backend-selection policy applied once, when the dispatcher is created.
///
/// `Auto` probes in the deterministic order CUDA, WGPU, CPU. `Exact` never
/// falls back to another backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePreference {
    Auto,
    Exact(DeviceKind),
}

/// Outcome of one backend-selection attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendSelectionOutcome {
    Selected,
    Rejected(ComputeError),
    NotCompiled,
}

/// Observable record of one backend considered during construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelectionAttempt {
    pub kind: DeviceKind,
    pub outcome: BackendSelectionOutcome,
}

impl BackendSelectionAttempt {
    fn selected(kind: DeviceKind) -> Self {
        Self {
            kind,
            outcome: BackendSelectionOutcome::Selected,
        }
    }

    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    fn rejected(kind: DeviceKind, error: ComputeError) -> Self {
        Self {
            kind,
            outcome: BackendSelectionOutcome::Rejected(error),
        }
    }

    #[cfg(any(not(feature = "cuda"), not(feature = "wgpu")))]
    fn not_compiled(kind: DeviceKind) -> Self {
        Self {
            kind,
            outcome: BackendSelectionOutcome::NotCompiled,
        }
    }
}

enum SelectedBackend {
    Cpu(CpuComputeAdapter),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuComputeAdapter),
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeAdapter),
}

/// Runtime-selected implementation of the backend-neutral compute contract.
///
/// Selection happens only at construction. Buffers, kernels, streams and
/// events remain bound to that selected backend for their complete lifetime.
///
/// Automatic fallback is observable through [`ComputeDispatcher::selection_attempts`].
pub struct ComputeDispatcher {
    preference: ComputePreference,
    backend: SelectedBackend,
    origin: Arc<()>,
    selection_attempts: Vec<BackendSelectionAttempt>,
}

impl ComputeDispatcher {
    pub fn new(preference: ComputePreference) -> ComputeResult<Self> {
        let (backend, selection_attempts) = match preference
        {
            ComputePreference::Auto => Self::select_automatic(),
            ComputePreference::Exact(kind) => (
                Self::select_exact(kind)?,
                vec![BackendSelectionAttempt::selected(kind)],
            ),
        };

        Ok(Self {
            preference,
            backend,
            origin: Arc::new(()),
            selection_attempts,
        })
    }

    pub fn automatic() -> Self {
        let (backend, selection_attempts) = Self::select_automatic();

        Self {
            preference: ComputePreference::Auto,
            backend,
            origin: Arc::new(()),
            selection_attempts,
        }
    }

    pub fn cpu() -> Self {
        Self {
            preference: ComputePreference::Exact(DeviceKind::Cpu),
            backend: SelectedBackend::Cpu(CpuComputeAdapter::new()),
            origin: Arc::new(()),
            selection_attempts: vec![BackendSelectionAttempt::selected(DeviceKind::Cpu)],
        }
    }

    pub const fn preference(&self) -> ComputePreference {
        self.preference
    }

    pub fn selected_kind(&self) -> DeviceKind {
        match &self.backend
        {
            SelectedBackend::Cpu(_) => DeviceKind::Cpu,
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(_) => DeviceKind::Wgpu,
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(_) => DeviceKind::Cuda,
        }
    }

    pub fn selection_attempts(&self) -> &[BackendSelectionAttempt] {
        &self.selection_attempts
    }

    fn select_automatic() -> (SelectedBackend, Vec<BackendSelectionAttempt>) {
        let mut attempts = vec![
            #[cfg(not(feature = "cuda"))]
            BackendSelectionAttempt::not_compiled(DeviceKind::Cuda),
            #[cfg(not(feature = "wgpu"))]
            BackendSelectionAttempt::not_compiled(DeviceKind::Wgpu),
        ];

        #[cfg(feature = "cuda")]
        match CudaComputeAdapter::new()
        {
            Ok(backend) =>
            {
                attempts.push(BackendSelectionAttempt::selected(DeviceKind::Cuda));
                return (SelectedBackend::Cuda(backend), attempts);
            },
            Err(error) =>
            {
                attempts.push(BackendSelectionAttempt::rejected(DeviceKind::Cuda, error));
            },
        }

        #[cfg(feature = "wgpu")]
        match WgpuComputeAdapter::new()
        {
            Ok(backend) =>
            {
                attempts.push(BackendSelectionAttempt::selected(DeviceKind::Wgpu));
                return (SelectedBackend::Wgpu(backend), attempts);
            },
            Err(error) =>
            {
                attempts.push(BackendSelectionAttempt::rejected(DeviceKind::Wgpu, error));
            },
        }

        attempts.extend([BackendSelectionAttempt::selected(DeviceKind::Cpu)]);

        (SelectedBackend::Cpu(CpuComputeAdapter::new()), attempts)
    }

    fn select_exact(kind: DeviceKind) -> ComputeResult<SelectedBackend> {
        match kind
        {
            DeviceKind::Cpu => Ok(SelectedBackend::Cpu(CpuComputeAdapter::new())),
            DeviceKind::Wgpu =>
            {
                #[cfg(feature = "wgpu")]
                {
                    WgpuComputeAdapter::new().map(SelectedBackend::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(ComputeError::BackendUnavailable(DeviceKind::Wgpu))
                }
            },
            DeviceKind::Cuda =>
            {
                #[cfg(feature = "cuda")]
                {
                    CudaComputeAdapter::new().map(SelectedBackend::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(ComputeError::BackendUnavailable(DeviceKind::Cuda))
                }
            },
            _ => Err(ComputeError::BackendUnavailable(kind)),
        }
    }
}

impl Default for ComputeDispatcher {
    fn default() -> Self {
        Self::automatic()
    }
}

enum DispatchBufferInner {
    Cpu(CpuBuffer),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuComputeBuffer),
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeBuffer),
}

pub struct DispatchBuffer {
    origin: Arc<()>,
    inner: DispatchBufferInner,
}

impl DispatchBuffer {
    pub fn kind(&self) -> DeviceKind {
        match &self.inner
        {
            DispatchBufferInner::Cpu(_) => DeviceKind::Cpu,
            #[cfg(feature = "wgpu")]
            DispatchBufferInner::Wgpu(_) => DeviceKind::Wgpu,
            #[cfg(feature = "cuda")]
            DispatchBufferInner::Cuda(_) => DeviceKind::Cuda,
        }
    }

    fn belongs_to(&self, origin: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.origin, origin)
    }
}

enum DispatchKernelInner {
    Cpu(CpuKernel),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuComputeKernel),
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeKernel),
}

pub struct DispatchKernel {
    origin: Arc<()>,
    inner: DispatchKernelInner,
}

impl DispatchKernel {
    pub fn kind(&self) -> DeviceKind {
        match &self.inner
        {
            DispatchKernelInner::Cpu(_) => DeviceKind::Cpu,
            #[cfg(feature = "wgpu")]
            DispatchKernelInner::Wgpu(_) => DeviceKind::Wgpu,
            #[cfg(feature = "cuda")]
            DispatchKernelInner::Cuda(_) => DeviceKind::Cuda,
        }
    }

    fn belongs_to(&self, origin: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.origin, origin)
    }
}

enum DispatchStreamInner {
    Cpu(CpuStream),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuComputeStream),
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeStream),
}

pub struct DispatchStream {
    origin: Arc<()>,
    inner: DispatchStreamInner,
}

impl DispatchStream {
    pub fn kind(&self) -> DeviceKind {
        match &self.inner
        {
            DispatchStreamInner::Cpu(_) => DeviceKind::Cpu,
            #[cfg(feature = "wgpu")]
            DispatchStreamInner::Wgpu(_) => DeviceKind::Wgpu,
            #[cfg(feature = "cuda")]
            DispatchStreamInner::Cuda(_) => DeviceKind::Cuda,
        }
    }

    fn belongs_to(&self, origin: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.origin, origin)
    }
}

enum DispatchEventInner {
    Cpu(CpuEvent),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuComputeEvent),
    #[cfg(feature = "cuda")]
    Cuda(CudaComputeEvent),
}

pub struct DispatchEvent {
    origin: Arc<()>,
    inner: DispatchEventInner,
}

impl DispatchEvent {
    pub fn kind(&self) -> DeviceKind {
        match &self.inner
        {
            DispatchEventInner::Cpu(_) => DeviceKind::Cpu,
            #[cfg(feature = "wgpu")]
            DispatchEventInner::Wgpu(_) => DeviceKind::Wgpu,
            #[cfg(feature = "cuda")]
            DispatchEventInner::Cuda(_) => DeviceKind::Cuda,
        }
    }

    fn belongs_to(&self, origin: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.origin, origin)
    }
}

fn resource_mismatch<T>() -> ComputeResult<T> {
    Err(ComputeError::InvalidArgument(
        "compute resource belongs to a different backend",
    ))
}

impl ComputeBackend for ComputeDispatcher {
    type Buffer = DispatchBuffer;
    type Kernel = DispatchKernel;
    type Stream = DispatchStream;
    type Event = DispatchEvent;

    fn capabilities(&self) -> &DeviceCapabilities {
        match &self.backend
        {
            SelectedBackend::Cpu(backend) => backend.capabilities(),
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(backend) => backend.capabilities(),
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(backend) => backend.capabilities(),
        }
    }

    fn allocate(
        &self,
        bytes: usize,
        alignment: usize,
        memory_space: MemorySpace,
    ) -> ComputeResult<Self::Buffer> {
        match &self.backend
        {
            SelectedBackend::Cpu(backend) =>
            {
                backend
                    .allocate(bytes, alignment, memory_space)
                    .map(|inner| DispatchBuffer {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchBufferInner::Cpu(inner),
                    })
            },
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(backend) =>
            {
                backend
                    .allocate(bytes, alignment, memory_space)
                    .map(|inner| DispatchBuffer {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchBufferInner::Wgpu(inner),
                    })
            },
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(backend) =>
            {
                backend
                    .allocate(bytes, alignment, memory_space)
                    .map(|inner| DispatchBuffer {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchBufferInner::Cuda(inner),
                    })
            },
        }
    }

    #[allow(unreachable_patterns)]
    fn write(
        &self,
        destination: &Self::Buffer,
        offset_bytes: usize,
        data: &[u8],
    ) -> ComputeResult<()> {
        if !destination.belongs_to(&self.origin)
        {
            return resource_mismatch();
        }

        match (&self.backend, &destination.inner)
        {
            (SelectedBackend::Cpu(backend), DispatchBufferInner::Cpu(buffer)) =>
            {
                backend.write(buffer, offset_bytes, data)
            },
            #[cfg(feature = "wgpu")]
            (SelectedBackend::Wgpu(backend), DispatchBufferInner::Wgpu(buffer)) =>
            {
                backend.write(buffer, offset_bytes, data)
            },
            #[cfg(feature = "cuda")]
            (SelectedBackend::Cuda(backend), DispatchBufferInner::Cuda(buffer)) =>
            {
                backend.write(buffer, offset_bytes, data)
            },
            _ => resource_mismatch(),
        }
    }

    #[allow(unreachable_patterns)]
    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        if !source.belongs_to(&self.origin)
        {
            return resource_mismatch();
        }

        match (&self.backend, &source.inner)
        {
            (SelectedBackend::Cpu(backend), DispatchBufferInner::Cpu(buffer)) =>
            {
                backend.read(buffer, offset_bytes, destination)
            },
            #[cfg(feature = "wgpu")]
            (SelectedBackend::Wgpu(backend), DispatchBufferInner::Wgpu(buffer)) =>
            {
                backend.read(buffer, offset_bytes, destination)
            },
            #[cfg(feature = "cuda")]
            (SelectedBackend::Cuda(backend), DispatchBufferInner::Cuda(buffer)) =>
            {
                backend.read(buffer, offset_bytes, destination)
            },
            _ => resource_mismatch(),
        }
    }

    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        match &self.backend
        {
            SelectedBackend::Cpu(backend) => backend.compile(module).map(|inner| DispatchKernel {
                origin: Arc::clone(&self.origin),
                inner: DispatchKernelInner::Cpu(inner),
            }),
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(backend) => backend.compile(module).map(|inner| DispatchKernel {
                origin: Arc::clone(&self.origin),
                inner: DispatchKernelInner::Wgpu(inner),
            }),
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(backend) => backend.compile(module).map(|inner| DispatchKernel {
                origin: Arc::clone(&self.origin),
                inner: DispatchKernelInner::Cuda(inner),
            }),
        }
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        match &self.backend
        {
            SelectedBackend::Cpu(backend) => backend.create_stream().map(|inner| DispatchStream {
                origin: Arc::clone(&self.origin),
                inner: DispatchStreamInner::Cpu(inner),
            }),
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(backend) => backend.create_stream().map(|inner| DispatchStream {
                origin: Arc::clone(&self.origin),
                inner: DispatchStreamInner::Wgpu(inner),
            }),
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(backend) => backend.create_stream().map(|inner| DispatchStream {
                origin: Arc::clone(&self.origin),
                inner: DispatchStreamInner::Cuda(inner),
            }),
        }
    }

    #[allow(irrefutable_let_patterns)]
    fn launch(
        &self,
        kernel: &Self::Kernel,
        stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        if !kernel.belongs_to(&self.origin) || !stream.belongs_to(&self.origin)
        {
            return resource_mismatch();
        }

        if bindings
            .iter()
            .any(|binding| !binding.buffer.belongs_to(&self.origin))
        {
            return resource_mismatch();
        }

        match &self.backend
        {
            SelectedBackend::Cpu(backend) =>
            {
                let DispatchKernelInner::Cpu(kernel) = &kernel.inner
                else
                {
                    return resource_mismatch();
                };
                let DispatchStreamInner::Cpu(stream) = &stream.inner
                else
                {
                    return resource_mismatch();
                };

                let mut translated = Vec::with_capacity(bindings.len());
                for binding in bindings
                {
                    let DispatchBufferInner::Cpu(buffer) = &binding.buffer.inner
                    else
                    {
                        return resource_mismatch();
                    };
                    translated.push(BufferBinding {
                        slot: binding.slot,
                        buffer,
                        offset_bytes: binding.offset_bytes,
                        length_bytes: binding.length_bytes,
                        access: binding.access,
                    });
                }

                backend
                    .launch(kernel, stream, config, &translated)
                    .map(|inner| DispatchEvent {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchEventInner::Cpu(inner),
                    })
            },
            #[cfg(feature = "wgpu")]
            SelectedBackend::Wgpu(backend) =>
            {
                let DispatchKernelInner::Wgpu(kernel) = &kernel.inner
                else
                {
                    return resource_mismatch();
                };
                let DispatchStreamInner::Wgpu(stream) = &stream.inner
                else
                {
                    return resource_mismatch();
                };

                let mut translated = Vec::with_capacity(bindings.len());
                for binding in bindings
                {
                    let DispatchBufferInner::Wgpu(buffer) = &binding.buffer.inner
                    else
                    {
                        return resource_mismatch();
                    };
                    translated.push(BufferBinding {
                        slot: binding.slot,
                        buffer,
                        offset_bytes: binding.offset_bytes,
                        length_bytes: binding.length_bytes,
                        access: binding.access,
                    });
                }

                backend
                    .launch(kernel, stream, config, &translated)
                    .map(|inner| DispatchEvent {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchEventInner::Wgpu(inner),
                    })
            },
            #[cfg(feature = "cuda")]
            SelectedBackend::Cuda(backend) =>
            {
                let DispatchKernelInner::Cuda(kernel) = &kernel.inner
                else
                {
                    return resource_mismatch();
                };
                let DispatchStreamInner::Cuda(stream) = &stream.inner
                else
                {
                    return resource_mismatch();
                };

                let mut translated = Vec::with_capacity(bindings.len());
                for binding in bindings
                {
                    let DispatchBufferInner::Cuda(buffer) = &binding.buffer.inner
                    else
                    {
                        return resource_mismatch();
                    };
                    translated.push(BufferBinding {
                        slot: binding.slot,
                        buffer,
                        offset_bytes: binding.offset_bytes,
                        length_bytes: binding.length_bytes,
                        access: binding.access,
                    });
                }

                backend
                    .launch(kernel, stream, config, &translated)
                    .map(|inner| DispatchEvent {
                        origin: Arc::clone(&self.origin),
                        inner: DispatchEventInner::Cuda(inner),
                    })
            },
        }
    }

    #[allow(unreachable_patterns)]
    fn wait(&self, event: &Self::Event) -> ComputeResult<()> {
        if !event.belongs_to(&self.origin)
        {
            return resource_mismatch();
        }

        match (&self.backend, &event.inner)
        {
            (SelectedBackend::Cpu(backend), DispatchEventInner::Cpu(event)) => backend.wait(event),
            #[cfg(feature = "wgpu")]
            (SelectedBackend::Wgpu(backend), DispatchEventInner::Wgpu(event)) =>
            {
                backend.wait(event)
            },
            #[cfg(feature = "cuda")]
            (SelectedBackend::Cuda(backend), DispatchEventInner::Cuda(event)) =>
            {
                backend.wait(event)
            },
            _ => resource_mismatch(),
        }
    }

    #[allow(unreachable_patterns)]
    fn synchronize(&self, stream: &Self::Stream) -> ComputeResult<()> {
        if !stream.belongs_to(&self.origin)
        {
            return resource_mismatch();
        }

        match (&self.backend, &stream.inner)
        {
            (SelectedBackend::Cpu(backend), DispatchStreamInner::Cpu(stream)) =>
            {
                backend.synchronize(stream)
            },
            #[cfg(feature = "wgpu")]
            (SelectedBackend::Wgpu(backend), DispatchStreamInner::Wgpu(stream)) =>
            {
                backend.synchronize(stream)
            },
            #[cfg(feature = "cuda")]
            (SelectedBackend::Cuda(backend), DispatchStreamInner::Cuda(stream)) =>
            {
                backend.synchronize(stream)
            },
            _ => resource_mismatch(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_cpu_selection_has_one_observable_attempt() {
        let dispatcher = ComputeDispatcher::cpu();

        assert_eq!(
            dispatcher.selection_attempts(),
            &[BackendSelectionAttempt {
                kind: DeviceKind::Cpu,
                outcome: BackendSelectionOutcome::Selected,
            }]
        );
    }

    #[test]
    fn automatic_trace_ends_with_the_selected_backend() {
        let dispatcher = ComputeDispatcher::automatic();
        let attempts = dispatcher.selection_attempts();
        let selected = attempts
            .last()
            .expect("automatic selection records a result");

        assert_eq!(selected.kind, dispatcher.selected_kind());
        assert_eq!(selected.outcome, BackendSelectionOutcome::Selected);
        assert_eq!(
            attempts
                .iter()
                .filter(|attempt| attempt.outcome == BackendSelectionOutcome::Selected)
                .count(),
            1
        );
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
    #[test]
    fn cpu_only_trace_reports_compiled_out_device_backends() {
        let dispatcher = ComputeDispatcher::automatic();

        assert_eq!(
            dispatcher.selection_attempts(),
            &[
                BackendSelectionAttempt {
                    kind: DeviceKind::Cuda,
                    outcome: BackendSelectionOutcome::NotCompiled,
                },
                BackendSelectionAttempt {
                    kind: DeviceKind::Wgpu,
                    outcome: BackendSelectionOutcome::NotCompiled,
                },
                BackendSelectionAttempt {
                    kind: DeviceKind::Cpu,
                    outcome: BackendSelectionOutcome::Selected,
                },
            ]
        );
    }

    #[test]
    fn exact_cpu_selection_is_stable() {
        let dispatcher = ComputeDispatcher::new(ComputePreference::Exact(DeviceKind::Cpu)).unwrap();

        assert_eq!(
            dispatcher.preference(),
            ComputePreference::Exact(DeviceKind::Cpu)
        );
        assert_eq!(dispatcher.selected_kind(), DeviceKind::Cpu);
        assert_eq!(dispatcher.capabilities().device.kind(), DeviceKind::Cpu);
    }

    #[test]
    fn cpu_dispatch_roundtrip_preserves_offsets() {
        let dispatcher = ComputeDispatcher::cpu();
        let buffer = dispatcher.allocate(8, 1, MemorySpace::Host).unwrap();

        dispatcher.write(&buffer, 2, &[1, 2, 3, 4]).unwrap();

        let mut output = [0u8; 4];
        dispatcher.read(&buffer, 2, &mut output).unwrap();

        assert_eq!(output, [1, 2, 3, 4]);
        assert_eq!(buffer.kind(), DeviceKind::Cpu);
    }

    #[test]
    fn resources_from_another_dispatcher_are_rejected() {
        let owner = ComputeDispatcher::cpu();
        let foreign = ComputeDispatcher::cpu();
        let buffer = owner.allocate(4, 1, MemorySpace::Host).unwrap();

        assert!(matches!(
            foreign.write(&buffer, 0, &[1]),
            Err(ComputeError::InvalidArgument(
                "compute resource belongs to a different backend"
            ))
        ));
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
    #[test]
    fn automatic_selection_uses_cpu_when_no_device_backend_is_compiled() {
        let dispatcher = ComputeDispatcher::automatic();
        assert_eq!(dispatcher.selected_kind(), DeviceKind::Cpu);
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn exact_wgpu_selection_never_falls_back() {
        assert!(matches!(
            ComputeDispatcher::new(ComputePreference::Exact(DeviceKind::Wgpu)),
            Err(ComputeError::BackendUnavailable(DeviceKind::Wgpu))
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn exact_cuda_selection_never_falls_back() {
        assert!(matches!(
            ComputeDispatcher::new(ComputePreference::Exact(DeviceKind::Cuda)),
            Err(ComputeError::BackendUnavailable(DeviceKind::Cuda))
        ));
    }

    #[test]
    fn reference_backend_is_not_fabricated() {
        assert!(matches!(
            ComputeDispatcher::new(ComputePreference::Exact(DeviceKind::Reference)),
            Err(ComputeError::BackendUnavailable(DeviceKind::Reference))
        ));
    }
}
