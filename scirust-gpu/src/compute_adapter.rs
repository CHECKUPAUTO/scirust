extern crate alloc;

use alloc::{string::ToString, vec, vec::Vec};
use core::cell::RefCell;

use crate::{BackendResult, CpuBackend, RawComputeBackend};

use scirust_compute::{
    BufferBinding, ComputeBackend, ComputeError, ComputeResult, DType, DeviceCapabilities,
    DeviceId, KernelFormat, KernelModule, LaunchConfig, MemorySpace,
};
use scirust_tensor_reference::PreparedReferenceKernel;

use crate::cpu_reference;

/// Adapter exposing the deterministic SciRust CPU path through
/// `scirust_compute::ComputeBackend`.
#[derive(Debug)]
pub struct CpuComputeAdapter {
    capabilities: DeviceCapabilities,
}

impl CpuComputeAdapter {
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities {
                device: DeviceId::cpu(),
                name: "scirust-gpu-cpu".to_string(),
                supported_dtypes: vec![DType::F32],
                max_buffer_bytes: None,
                max_workgroup_size: [1, 1, 1],
                supports_async_execution: false,
            },
        }
    }

    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
}

impl Default for CpuComputeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct CpuBuffer {
    pub(crate) bytes: RefCell<Vec<u8>>,
    alignment: usize,
    memory_space: MemorySpace,
}

impl CpuBuffer {
    pub fn len(&self) -> usize {
        self.bytes.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn memory_space(&self) -> MemorySpace {
        self.memory_space
    }
}

/// A Reference kernel compiled for the CPU backend.
///
/// Holds the original [`KernelModule`] and the [`PreparedReferenceKernel`]
/// decoded from it. Decoding happens **once**, in
/// [`ComputeBackend::compile`]; `launch` never re-decodes.
///
/// `PartialEq`/`Eq` compare the module alone. That is exact rather than partial:
/// `prepared` is derived deterministically from `module`, so equal modules
/// always yield equal prepared kernels. They are written by hand instead of
/// derived because `PreparedReferenceKernel` deliberately implements neither —
/// it carries an `f32` scale factor, and a NaN factor would make a kernel
/// unequal to itself. Hand-writing them keeps this type's public API exactly as
/// it was before the Reference interpreter was wired in.
#[derive(Debug, Clone)]
pub struct CpuKernel {
    pub(crate) module: KernelModule,
    pub(crate) prepared: PreparedReferenceKernel,
}

impl PartialEq for CpuKernel {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module
    }
}

impl Eq for CpuKernel {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStream(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuEvent(());

impl RawComputeBackend for CpuComputeAdapter {
    fn device_name(&self) -> &'static str {
        CpuBackend.device_name()
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        CpuBackend.gemm_f32(a, b, m, k, n)
    }
}

impl ComputeBackend for CpuComputeAdapter {
    type Buffer = CpuBuffer;
    type Kernel = CpuKernel;
    type Stream = CpuStream;
    type Event = CpuEvent;

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
        if alignment != 1
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter currently supports byte alignment only",
            ));
        }
        if memory_space != MemorySpace::Host
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter supports host memory only",
            ));
        }

        let mut storage = Vec::new();
        storage
            .try_reserve_exact(bytes)
            .map_err(|error| ComputeError::Allocation(error.to_string()))?;
        storage.resize(bytes, 0);

        Ok(CpuBuffer {
            bytes: RefCell::new(storage),
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
        let end = offset_bytes
            .checked_add(data.len())
            .ok_or_else(|| ComputeError::Transfer("write range overflow".to_string()))?;

        let mut bytes = destination
            .bytes
            .try_borrow_mut()
            .map_err(|_| ComputeError::Transfer("buffer is already borrowed".to_string()))?;

        if end > bytes.len()
        {
            return Err(ComputeError::Transfer(
                "write exceeds buffer bounds".to_string(),
            ));
        }

        bytes[offset_bytes..end].copy_from_slice(data);
        Ok(())
    }

    fn read(
        &self,
        source: &Self::Buffer,
        offset_bytes: usize,
        destination: &mut [u8],
    ) -> ComputeResult<()> {
        let end = offset_bytes
            .checked_add(destination.len())
            .ok_or_else(|| ComputeError::Transfer("read range overflow".to_string()))?;

        let bytes = source
            .bytes
            .try_borrow()
            .map_err(|_| ComputeError::Transfer("buffer is mutably borrowed".to_string()))?;

        if end > bytes.len()
        {
            return Err(ComputeError::Transfer(
                "read exceeds buffer bounds".to_string(),
            ));
        }

        destination.copy_from_slice(&bytes[offset_bytes..end]);
        Ok(())
    }

    /// Decodes and validates a Reference module once, up front.
    ///
    /// A wrong [`KernelFormat`] keeps its historical
    /// [`ComputeError::Unsupported`] mapping, since that is the contract the
    /// shared conformance suite asserts for every backend. Every other
    /// preparation failure — bad magic, unsupported version, mismatched entry
    /// point, non-`F32` element type, `Exp`, `Log`, unknown opcode — becomes
    /// [`ComputeError::Compilation`], carrying the typed reason rendered once
    /// at this boundary.
    fn compile(&self, module: &KernelModule) -> ComputeResult<Self::Kernel> {
        if module.format != KernelFormat::Reference
        {
            return Err(ComputeError::Unsupported(
                "CPU adapter accepts reference kernels only",
            ));
        }

        let prepared = PreparedReferenceKernel::from_kernel_module(module)
            .map_err(|error| ComputeError::Compilation(error.to_string()))?;

        Ok(CpuKernel {
            module: module.clone(),
            prepared,
        })
    }

    fn create_stream(&self) -> ComputeResult<Self::Stream> {
        Ok(CpuStream(()))
    }

    /// Executes a compiled Reference kernel synchronously.
    ///
    /// The binding ABI, access rules, byte layout and transactional ordering are
    /// documented in [`crate::cpu_reference`]. Execution is fully serial and
    /// completes before this returns, so the [`CpuEvent`] handed back is already
    /// finished — `wait` and `synchronize` therefore stay trivially correct. No
    /// thread and no asynchronous task is created.
    ///
    /// `stream` needs no ownership check: [`CpuStream`] has a private field, so
    /// only this crate can construct one and the type system already guarantees
    /// any `&CpuStream` reaching here belongs to the CPU backend.
    fn launch(
        &self,
        kernel: &Self::Kernel,
        _stream: &Self::Stream,
        config: LaunchConfig,
        bindings: &[BufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::Event> {
        cpu_reference::launch_reference_kernel(&kernel.prepared, config, bindings)
            .map_err(|error| ComputeError::Launch(cpu_reference::describe(&error)))?;

        Ok(CpuEvent(()))
    }

    fn wait(&self, _event: &Self::Event) -> ComputeResult<()> {
        Ok(())
    }

    fn synchronize(&self, _stream: &Self::Stream) -> ComputeResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use scirust_compute::BufferAccess;
    use scirust_tensor_ir::Operation;

    use crate::cpu_reference::test_support;

    #[test]
    fn adapter_reports_honest_cpu_capabilities() {
        let adapter = CpuComputeAdapter::new();

        assert_eq!(adapter.capabilities().device, DeviceId::cpu());
        assert!(adapter.capabilities().supports_dtype(DType::F32));
        assert!(!adapter.capabilities().supports_async_execution);
    }

    #[test]
    fn adapter_preserves_the_existing_cpu_gemm_contract() {
        let adapter = CpuComputeAdapter::new();
        let a = [1.0, 2.0, 3.0, 4.0];
        let identity = [1.0, 0.0, 0.0, 1.0];

        assert_eq!(adapter.device_name(), "cpu");
        assert_eq!(
            adapter.gemm_f32(&a, &identity, 2, 2, 2).unwrap(),
            a.to_vec()
        );
    }

    #[test]
    fn host_buffer_round_trip_is_checked() {
        let adapter = CpuComputeAdapter::new();
        let buffer = adapter.allocate(8, 1, MemorySpace::Host).unwrap();

        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.alignment(), 1);
        assert_eq!(buffer.memory_space(), MemorySpace::Host);

        adapter.write(&buffer, 2, &[10, 20, 30]).unwrap();

        let mut output = [0u8; 8];
        adapter.read(&buffer, 0, &mut output).unwrap();

        assert_eq!(output, [0, 0, 10, 20, 30, 0, 0, 0]);
    }

    #[test]
    fn invalid_buffer_requests_are_rejected() {
        let adapter = CpuComputeAdapter::new();

        assert!(matches!(
            adapter.allocate(8, 0, MemorySpace::Host),
            Err(ComputeError::InvalidArgument(_))
        ));

        assert!(matches!(
            adapter.allocate(8, 1, MemorySpace::Device),
            Err(ComputeError::Unsupported(_))
        ));

        let buffer = adapter.allocate(4, 1, MemorySpace::Host).unwrap();

        assert!(matches!(
            adapter.write(&buffer, 3, &[1, 2]),
            Err(ComputeError::Transfer(_))
        ));
    }

    #[test]
    fn reference_kernel_compilation_is_explicit() {
        let adapter = CpuComputeAdapter::new();

        // A genuine artefact, not the arbitrary `b"reference"` bytes this test
        // used before `compile` validated its payload.
        let reference = test_support::canonical_reference_module();
        let kernel = adapter.compile(&reference).unwrap();

        assert_eq!(kernel.module, reference);

        let wgsl = KernelModule::new(KernelFormat::Wgsl, "main", b"wgsl".to_vec()).unwrap();

        assert!(matches!(
            adapter.compile(&wgsl),
            Err(ComputeError::Unsupported(_))
        ));
    }

    #[test]
    fn compilation_rejects_an_invalid_reference_payload() {
        let adapter = CpuComputeAdapter::new();

        // Right format, corrupt magic: no longer accepted, and reported as a
        // compilation failure rather than an unsupported format.
        assert!(matches!(
            adapter.compile(&test_support::corrupted_reference_module()),
            Err(ComputeError::Compilation(_))
        ));
    }

    #[test]
    fn compilation_rejects_a_mismatched_entry_point() {
        let adapter = CpuComputeAdapter::new();

        let artifact = test_support::unary_artifact(Operation::Relu, vec![2, 2]);
        let mislabelled = KernelModule::new(
            KernelFormat::Reference,
            "not_the_derived_name",
            artifact.encode(),
        )
        .unwrap();

        assert!(matches!(
            adapter.compile(&mislabelled),
            Err(ComputeError::Compilation(_))
        ));
    }

    #[test]
    fn compilation_rejects_exp_and_log() {
        let adapter = CpuComputeAdapter::new();

        for operation in [Operation::Exp, Operation::Log]
        {
            let module = test_support::unary_module(operation, vec![4]);

            assert!(
                matches!(adapter.compile(&module), Err(ComputeError::Compilation(_))),
                "Exp and Log have no bit-reproducible CPU implementation and must be rejected"
            );
        }
    }

    #[test]
    fn compilation_decodes_once_and_launch_reuses_the_prepared_kernel() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![4]))
            .unwrap();

        // Observable consequence of decoding once: the same compiled kernel
        // serves repeated launches, and each produces the same result.
        for _ in 0..3
        {
            let input = float_buffer(&adapter, &[-1.0, 2.0, -3.0, 4.0]);
            let output = float_buffer(&adapter, &[0.0; 4]);

            adapter
                .launch(
                    &kernel,
                    &stream,
                    canonical_config(),
                    &[
                        read_binding(0, &input, 0, 4),
                        write_binding(1, &output, 0, 4),
                    ],
                )
                .unwrap();

            assert_eq!(
                read_floats(&adapter, &output, 0, 4),
                vec![0.0, 2.0, 0.0, 4.0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Launch: element-wise opcodes
    // -----------------------------------------------------------------------

    #[test]
    fn launches_relu() {
        let output = launch_unary(Operation::Relu, vec![4], &[-1.5, 0.0, 2.5, -0.0]);
        assert_eq!(bits(&output), bits(&[0.0, 0.0, 2.5, 0.0]));
    }

    #[test]
    fn launches_scale() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::module_of(&test_support::scale_artifact(
                0.5,
                vec![3],
            )))
            .unwrap();

        let input = float_buffer(&adapter, &[1.0, -2.0, 3.0]);
        let output = float_buffer(&adapter, &[0.0; 3]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &input, 0, 3),
                    write_binding(1, &output, 0, 3),
                ],
            )
            .unwrap();

        assert_eq!(read_floats(&adapter, &output, 0, 3), vec![0.5, -1.0, 1.5]);
    }

    #[test]
    fn launches_add_sub_mul_and_div() {
        let cases = [
            (Operation::Add, vec![5.0, 7.0, 9.0]),
            (Operation::Sub, vec![-3.0, -3.0, -3.0]),
            (Operation::Mul, vec![4.0, 10.0, 18.0]),
            (Operation::Div, vec![0.25, 0.4, 0.5]),
        ];

        for (operation, expected) in cases
        {
            let label = alloc::format!("{operation:?}");
            let output = launch_binary(operation, vec![3], &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
            assert_eq!(output, expected, "unexpected result for {label}");
        }
    }

    #[test]
    fn launch_preserves_signed_zeros_and_ieee_division() {
        // 1.0 * -0.0 = -0.0 and -1.0 * -0.0 = +0.0: only the bits distinguish
        // them, so compare bit patterns.
        let scaled = {
            let adapter = CpuComputeAdapter::new();
            let stream = adapter.create_stream().unwrap();
            let kernel = adapter
                .compile(&test_support::module_of(&test_support::scale_artifact(
                    -0.0,
                    vec![2],
                )))
                .unwrap();
            let input = float_buffer(&adapter, &[1.0, -1.0]);
            let output = float_buffer(&adapter, &[0.0; 2]);
            adapter
                .launch(
                    &kernel,
                    &stream,
                    canonical_config(),
                    &[
                        read_binding(0, &input, 0, 2),
                        write_binding(1, &output, 0, 2),
                    ],
                )
                .unwrap();
            read_floats(&adapter, &output, 0, 2)
        };
        assert_eq!(bits(&scaled), bits(&[-0.0, 0.0]));

        let divided = launch_binary(Operation::Div, vec![3], &[1.0, -1.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(divided[0].to_bits(), f32::INFINITY.to_bits());
        assert_eq!(divided[1].to_bits(), f32::NEG_INFINITY.to_bits());
        // 0.0 / 0.0 is a NaN; its payload is deliberately not asserted.
        assert!(divided[2].is_nan());
    }

    #[test]
    fn launch_propagates_nan_without_a_payload_guarantee() {
        let payload_nan = f32::from_bits(0x7FC0_1234);

        // Arithmetic: a NaN result is required, its payload is not specified.
        let sum = launch_binary(Operation::Add, vec![2], &[payload_nan, 1.0], &[1.0, 1.0]);
        assert!(sum[0].is_nan());
        assert!(!sum[1].is_nan());

        // Relu returns its input untouched, so the payload survives exactly.
        let relu = launch_unary(Operation::Relu, vec![1], &[payload_nan]);
        assert_eq!(relu[0].to_bits(), 0x7FC0_1234);
    }

    #[test]
    fn launch_writes_at_a_non_zero_offset_inside_a_larger_buffer() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        // Eight floats of room; the kernel binds two of them, at element 3.
        let input = float_buffer(&adapter, &[0.0, 0.0, 0.0, -5.0, 6.0, 0.0, 0.0, 0.0]);
        let output = float_buffer(&adapter, &[9.0; 8]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &input, 3, 2),
                    write_binding(1, &output, 5, 2),
                ],
            )
            .unwrap();

        // Only elements 5 and 6 changed; the rest of the buffer is untouched.
        assert_eq!(
            read_floats(&adapter, &output, 0, 8),
            vec![9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 6.0, 9.0]
        );
    }

    // -----------------------------------------------------------------------
    // Launch: structural opcodes
    // -----------------------------------------------------------------------

    #[test]
    fn launches_shape_copy() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::module_of(&test_support::reshape_artifact(
                vec![2, 3],
                vec![6],
            )))
            .unwrap();

        let input = float_buffer(&adapter, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = float_buffer(&adapter, &[0.0; 6]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &input, 0, 6),
                    write_binding(1, &output, 0, 6),
                ],
            )
            .unwrap();

        assert_eq!(
            read_floats(&adapter, &output, 0, 6),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn launches_a_two_dimensional_permute() {
        let output = launch_transpose(
            vec![2, 3],
            vec![3, 2],
            vec![1, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn launches_a_three_dimensional_permute() {
        // [2, 3, 4] under permutation [2, 0, 1] becomes [4, 2, 3].
        let mut input = vec![0.0f32; 24];
        for i in 0..2
        {
            for j in 0..3
            {
                for k in 0..4
                {
                    input[i * 12 + j * 4 + k] = (i * 100 + j * 10 + k) as f32;
                }
            }
        }

        let output = launch_transpose(vec![2, 3, 4], vec![4, 2, 3], vec![2, 0, 1], &input);

        let mut expected = vec![0.0f32; 24];
        for k in 0..4
        {
            for i in 0..2
            {
                for j in 0..3
                {
                    expected[k * 6 + i * 3 + j] = (i * 100 + j * 10 + k) as f32;
                }
            }
        }

        assert_eq!(output, expected);
    }

    #[test]
    fn launches_a_scalar_tensor() {
        // Rank 0 holds exactly one element.
        let output = launch_transpose(vec![], vec![], vec![], &[42.0]);
        assert_eq!(output, vec![42.0]);
    }

    #[test]
    fn launches_a_zero_element_tensor() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::module_of(&test_support::transpose_artifact(
                vec![0, 3],
                vec![3, 0],
                vec![1, 0],
            )))
            .unwrap();

        // Non-empty buffers, zero-length bindings: must succeed and write
        // nothing.
        let input = float_buffer(&adapter, &[7.0; 2]);
        let output = float_buffer(&adapter, &[7.0; 2]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &input, 0, 0),
                    write_binding(1, &output, 0, 0),
                ],
            )
            .unwrap();

        assert_eq!(read_floats(&adapter, &output, 0, 2), vec![7.0, 7.0]);
    }

    // -----------------------------------------------------------------------
    // Binding ABI
    // -----------------------------------------------------------------------

    #[test]
    fn binding_slice_order_is_not_significant() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::binary_module(Operation::Sub, vec![2]))
            .unwrap();

        let lhs = float_buffer(&adapter, &[10.0, 20.0]);
        let rhs = float_buffer(&adapter, &[1.0, 2.0]);
        let output = float_buffer(&adapter, &[0.0; 2]);

        // Slots 2, 0, 1 handed over in reverse: resolution is by slot number.
        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    write_binding(2, &output, 0, 2),
                    read_binding(0, &lhs, 0, 2),
                    read_binding(1, &rhs, 0, 2),
                ],
            )
            .unwrap();

        assert_eq!(read_floats(&adapter, &output, 0, 2), vec![9.0, 18.0]);
    }

    #[test]
    fn one_buffer_may_serve_as_both_operand_and_result() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        // Same buffer, overlapping ranges: copy-in/copy-out reads everything
        // before writing anything, so this is well defined.
        let shared = float_buffer(&adapter, &[-1.0, 5.0, 0.0, 0.0]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &shared, 0, 2),
                    write_binding(1, &shared, 0, 2),
                ],
            )
            .unwrap();

        assert_eq!(
            read_floats(&adapter, &shared, 0, 4),
            vec![0.0, 5.0, 0.0, 0.0]
        );
    }

    /// Each case: a description, the bindings to submit, and nothing else —
    /// every one must fail and leave the output buffer untouched.
    #[test]
    fn invalid_bindings_are_rejected_and_leave_the_output_untouched() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        let sentinel = [111.0f32, 222.0, 333.0, 444.0];

        // (name, build the binding list)
        type BuildBindings =
            for<'a> fn(&'a CpuBuffer, &'a CpuBuffer) -> Vec<BufferBinding<'a, CpuBuffer>>;
        type Case = (&'static str, BuildBindings);

        let cases: [Case; 12] = [
            ("too few bindings", |input, _output| {
                vec![read_binding(0, input, 0, 2)]
            }),
            ("too many bindings", |input, output| {
                vec![
                    read_binding(0, input, 0, 2),
                    write_binding(1, output, 0, 2),
                    read_binding(0, input, 0, 2),
                ]
            }),
            ("duplicate slot", |input, _output| {
                vec![read_binding(0, input, 0, 2), read_binding(0, input, 0, 2)]
            }),
            ("slot beyond the kernel's range", |input, output| {
                vec![read_binding(0, input, 0, 2), write_binding(7, output, 0, 2)]
            }),
            ("operand declared WriteOnly", |input, output| {
                vec![
                    write_binding(0, input, 0, 2),
                    write_binding(1, output, 0, 2),
                ]
            }),
            ("operand declared ReadWrite", |input, output| {
                vec![
                    binding(0, input, 0, 2, BufferAccess::ReadWrite),
                    write_binding(1, output, 0, 2),
                ]
            }),
            ("result declared ReadOnly", |input, output| {
                vec![read_binding(0, input, 0, 2), read_binding(1, output, 0, 2)]
            }),
            ("result declared ReadWrite", |input, output| {
                vec![
                    read_binding(0, input, 0, 2),
                    binding(1, output, 0, 2, BufferAccess::ReadWrite),
                ]
            }),
            ("offset not a multiple of 4", |input, output| {
                vec![
                    raw_binding(0, input, 2, 8, BufferAccess::ReadOnly),
                    write_binding(1, output, 0, 2),
                ]
            }),
            ("length not a multiple of 4", |input, output| {
                vec![
                    raw_binding(0, input, 0, 7, BufferAccess::ReadOnly),
                    write_binding(1, output, 0, 2),
                ]
            }),
            ("length too short", |input, output| {
                vec![read_binding(0, input, 0, 1), write_binding(1, output, 0, 2)]
            }),
            ("range past the end of the buffer", |input, output| {
                vec![read_binding(0, input, 3, 2), write_binding(1, output, 0, 2)]
            }),
        ];

        for (name, build) in cases
        {
            let input = float_buffer(&adapter, &[-1.0, 5.0, 0.0, 0.0]);
            let output = float_buffer(&adapter, &sentinel);

            let result = adapter.launch(
                &kernel,
                &stream,
                canonical_config(),
                &build(&input, &output),
            );

            assert!(
                matches!(result, Err(ComputeError::Launch(_))),
                "{name} must be rejected as a launch failure"
            );
            assert_eq!(
                bits(&read_floats(&adapter, &output, 0, 4)),
                bits(&sentinel),
                "{name} must leave the output buffer untouched"
            );
        }
    }

    #[test]
    fn a_result_length_that_is_too_long_is_rejected_and_leaves_the_output_untouched() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        let sentinel = [111.0f32, 222.0, 333.0, 444.0];
        let input = float_buffer(&adapter, &[-1.0, 5.0, 0.0, 0.0]);
        let output = float_buffer(&adapter, &sentinel);

        let result = adapter.launch(
            &kernel,
            &stream,
            canonical_config(),
            &[
                read_binding(0, &input, 0, 2),
                write_binding(1, &output, 0, 3),
            ],
        );

        assert!(matches!(result, Err(ComputeError::Launch(_))));
        assert_eq!(bits(&read_floats(&adapter, &output, 0, 4)), bits(&sentinel));
    }

    // -----------------------------------------------------------------------
    // Backend contract
    // -----------------------------------------------------------------------

    #[test]
    fn non_canonical_launch_configurations_are_rejected() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        let sentinel = [111.0f32, 222.0];
        let configs = [
            LaunchConfig::new([2, 1, 1], [1, 1, 1], 0).unwrap(),
            LaunchConfig::new([1, 1, 1], [64, 1, 1], 0).unwrap(),
            LaunchConfig::new([1, 1, 1], [1, 1, 1], 16).unwrap(),
        ];

        for config in configs
        {
            let input = float_buffer(&adapter, &[-1.0, 5.0]);
            let output = float_buffer(&adapter, &sentinel);

            let result = adapter.launch(
                &kernel,
                &stream,
                config,
                &[
                    read_binding(0, &input, 0, 2),
                    write_binding(1, &output, 0, 2),
                ],
            );

            assert!(
                matches!(result, Err(ComputeError::Launch(_))),
                "{config:?} is not the canonical CPU reference configuration"
            );
            assert_eq!(bits(&read_floats(&adapter, &output, 0, 2)), bits(&sentinel));
        }
    }

    #[test]
    fn execution_contract_is_synchronous_and_complete_on_return() {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(Operation::Relu, vec![2]))
            .unwrap();

        let input = float_buffer(&adapter, &[-1.0, 5.0]);
        let output = float_buffer(&adapter, &[0.0; 2]);

        let event = adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &input, 0, 2),
                    write_binding(1, &output, 0, 2),
                ],
            )
            .unwrap();

        // The result is already visible before waiting: execution completed
        // before `launch` returned.
        assert_eq!(read_floats(&adapter, &output, 0, 2), vec![0.0, 5.0]);

        adapter.wait(&event).unwrap();
        adapter.synchronize(&stream).unwrap();
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn canonical_config() -> LaunchConfig {
        LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Allocates a host buffer holding `values` as little-endian `f32`s.
    fn float_buffer(adapter: &CpuComputeAdapter, values: &[f32]) -> CpuBuffer {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let buffer = adapter
            .allocate(bytes.len().max(1), 1, MemorySpace::Host)
            .unwrap();
        adapter.write(&buffer, 0, &bytes).unwrap();
        buffer
    }

    fn read_floats(
        adapter: &CpuComputeAdapter,
        buffer: &CpuBuffer,
        offset_elements: usize,
        elements: usize,
    ) -> Vec<f32> {
        let mut bytes = vec![0u8; elements * 4];
        adapter
            .read(buffer, offset_elements * 4, &mut bytes)
            .unwrap();

        let (groups, _) = bytes.as_chunks::<4>();
        groups
            .iter()
            .map(|group| f32::from_le_bytes(*group))
            .collect()
    }

    fn binding<'a>(
        slot: u32,
        buffer: &'a CpuBuffer,
        offset_elements: usize,
        elements: usize,
        access: BufferAccess,
    ) -> BufferBinding<'a, CpuBuffer> {
        raw_binding(slot, buffer, offset_elements * 4, elements * 4, access)
    }

    fn raw_binding<'a>(
        slot: u32,
        buffer: &'a CpuBuffer,
        offset_bytes: usize,
        length_bytes: usize,
        access: BufferAccess,
    ) -> BufferBinding<'a, CpuBuffer> {
        BufferBinding {
            slot,
            buffer,
            offset_bytes,
            length_bytes,
            access,
        }
    }

    fn read_binding<'a>(
        slot: u32,
        buffer: &'a CpuBuffer,
        offset_elements: usize,
        elements: usize,
    ) -> BufferBinding<'a, CpuBuffer> {
        binding(
            slot,
            buffer,
            offset_elements,
            elements,
            BufferAccess::ReadOnly,
        )
    }

    fn write_binding<'a>(
        slot: u32,
        buffer: &'a CpuBuffer,
        offset_elements: usize,
        elements: usize,
    ) -> BufferBinding<'a, CpuBuffer> {
        binding(
            slot,
            buffer,
            offset_elements,
            elements,
            BufferAccess::WriteOnly,
        )
    }

    fn launch_unary(operation: Operation, dims: Vec<usize>, input: &[f32]) -> Vec<f32> {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::unary_module(operation, dims))
            .unwrap();

        let source = float_buffer(&adapter, input);
        let output = float_buffer(&adapter, &vec![0.0f32; input.len()]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &source, 0, input.len()),
                    write_binding(1, &output, 0, input.len()),
                ],
            )
            .unwrap();

        read_floats(&adapter, &output, 0, input.len())
    }

    fn launch_binary(
        operation: Operation,
        dims: Vec<usize>,
        left: &[f32],
        right: &[f32],
    ) -> Vec<f32> {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::binary_module(operation, dims))
            .unwrap();

        let lhs = float_buffer(&adapter, left);
        let rhs = float_buffer(&adapter, right);
        let output = float_buffer(&adapter, &vec![0.0f32; left.len()]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &lhs, 0, left.len()),
                    read_binding(1, &rhs, 0, right.len()),
                    write_binding(2, &output, 0, left.len()),
                ],
            )
            .unwrap();

        read_floats(&adapter, &output, 0, left.len())
    }

    fn launch_transpose(
        input_dims: Vec<usize>,
        output_dims: Vec<usize>,
        permutation: Vec<usize>,
        input: &[f32],
    ) -> Vec<f32> {
        let adapter = CpuComputeAdapter::new();
        let stream = adapter.create_stream().unwrap();
        let kernel = adapter
            .compile(&test_support::module_of(&test_support::transpose_artifact(
                input_dims,
                output_dims,
                permutation,
            )))
            .unwrap();

        let source = float_buffer(&adapter, input);
        let output = float_buffer(&adapter, &vec![0.0f32; input.len()]);

        adapter
            .launch(
                &kernel,
                &stream,
                canonical_config(),
                &[
                    read_binding(0, &source, 0, input.len()),
                    write_binding(1, &output, 0, input.len()),
                ],
            )
            .unwrap();

        read_floats(&adapter, &output, 0, input.len())
    }
}
