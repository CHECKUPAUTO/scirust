//! The canonical Reference pipeline executed for real on a CUDA device.
//!
//! ```text
//! Graph -> CanonicalCompiler -> KernelLowerer -> LoweredPlan
//!       -> ReferencePlanRuntime<CudaReferenceAdapter>
//!       -> generated CUDA C -> NVRTC -> PTX -> CUDA module
//!       -> real kernel launch -> event -> device-to-host readback
//! ```
//!
//! Nothing here is mocked and nothing falls back: every value below crossed a
//! real CUDA stream, or the test did not run at all. The adapter's own
//! launch counter is asserted where that claim matters, so "it ran on the GPU"
//! rests on something observable rather than on the absence of an error.
//!
//! # Running these
//!
//! A machine without the CUDA driver, without NVRTC or without a device has no
//! adapter, and each test then **skips**. That is convenient locally and
//! dangerous in CI, so the skip is opt-out: set `SCIRUST_REQUIRE_CUDA=1` and a
//! missing device becomes a failure instead of a silent pass. The self-hosted
//! Jetson job sets it.
//!
//! # What is compared bitwise, and what is not
//!
//! `ShapeCopy`, `Permute` and `Relu` move or select 32-bit words and never
//! touch the float unit, so they are compared against the CPU interpreter
//! **bit for bit**, NaN payloads included. `Scale`, `Add`, `Sub`, `Mul` and
//! `Div` are `f32` arithmetic and are compared within a stated tolerance —
//! this crate does not promise cross-architecture bit identity for them.

#![cfg(feature = "cuda")]

use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, DType, DeviceKind, KernelFormat,
    KernelModule, LaunchConfig, MemorySpace, Shape,
};
use scirust_gpu::{CpuComputeAdapter, CudaReferenceAdapter};
use scirust_tensor_compile::{
    CanonicalCompiler, ExternalBindings, KernelLowerer, LogicalBindingId, LoweredPlan,
};
use scirust_tensor_ir::{Graph, NodeId, Operation, Scalar, TensorType};
use scirust_tensor_reference::ReferenceKernelGenerator;
use scirust_tensor_runtime::{PlanExternalValues, ReferencePlanRuntime};

/// The device these tests run on. Explicit on purpose: this adapter has no
/// default ordinal, never falls back to device zero, and never selects a device
/// implicitly.
const DEVICE_ORDINAL: usize = 0;

// ---------------------------------------------------------------------------
// Device acquisition
// ---------------------------------------------------------------------------

/// Acquire an adapter, or skip — unless the caller demanded a real device.
fn adapter_or_skip() -> Option<CudaReferenceAdapter> {
    match CudaReferenceAdapter::new(DEVICE_ORDINAL)
    {
        Ok(adapter) =>
        {
            let info = adapter.device_info();
            eprintln!(
                "cuda device {}: {} sm_{}{} memory={}B max_threads_per_block={} block_size={}",
                info.ordinal,
                info.name,
                info.compute_capability.0,
                info.compute_capability.1,
                info.total_memory_bytes,
                info.max_threads_per_block,
                adapter.block_size()
            );
            Some(adapter)
        },
        Err(error) =>
        {
            assert!(
                std::env::var_os("SCIRUST_REQUIRE_CUDA").is_none(),
                "SCIRUST_REQUIRE_CUDA is set, so a real CUDA device is mandatory, but none could \
                 be acquired: {error}"
            );
            eprintln!("skipping: no CUDA device available ({error})");
            None
        },
    }
}

// ---------------------------------------------------------------------------
// Plan fixtures
// ---------------------------------------------------------------------------

fn f32_type(dims: Vec<usize>) -> TensorType {
    TensorType::new(DType::F32, Shape::new(dims))
}

fn lower(graph: &Graph) -> LoweredPlan {
    let plan = CanonicalCompiler::new()
        .compile(graph)
        .expect("valid graph");
    let bindings = ExternalBindings::derive(&plan);
    KernelLowerer::new()
        .lower(&plan, &bindings)
        .expect("valid lowering")
}

/// `x -> op -> output`.
fn unary_plan(op: Operation, dims: Vec<usize>, output_dims: Vec<usize>) -> LoweredPlan {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(dims)).expect("input");
    let result = graph
        .add_node(op, vec![input], f32_type(output_dims))
        .expect("operation");
    graph.set_outputs(vec![result]).expect("outputs");
    lower(&graph)
}

/// `a, b -> op -> output`.
fn binary_plan(op: Operation, dims: Vec<usize>) -> LoweredPlan {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let lhs = graph.add_input("a", ty.clone()).expect("input");
    let rhs = graph.add_input("b", ty.clone()).expect("input");
    let result = graph.add_node(op, vec![lhs, rhs], ty).expect("operation");
    graph.set_outputs(vec![result]).expect("outputs");
    lower(&graph)
}

/// `x -> op(x, x) -> output`: one external value bound to both operands, which
/// is what makes the same device allocation appear at two kernel arguments.
fn self_binary_plan(op: Operation, dims: Vec<usize>) -> LoweredPlan {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let x = graph.add_input("x", ty.clone()).expect("input");
    let result = graph.add_node(op, vec![x, x], ty).expect("operation");
    graph.set_outputs(vec![result]).expect("outputs");
    lower(&graph)
}

/// Prepares and runs one plan on an existing runtime, returning the single
/// output's values.
///
/// Taking the runtime by reference matters: acquiring a CUDA context is not
/// free, so each test builds at most one.
fn run_on<B: ComputeBackend>(
    runtime: &ReferencePlanRuntime<B>,
    plan: &LoweredPlan,
    values: &[&[f32]],
) -> Vec<f32> {
    let prepared = runtime.prepare(plan).expect("preparable plan");

    let mut external = PlanExternalValues::new();
    for (index, value) in values.iter().enumerate()
    {
        let binding = LogicalBindingId::new(u32::try_from(index).expect("binding fits u32"));
        external.bind(binding, value);
    }

    let outputs = runtime.execute(&prepared, &external).expect("execution");
    assert_eq!(outputs.len(), 1);
    outputs.into_values()[0].values.clone()
}

fn run_cpu(plan: &LoweredPlan, values: &[&[f32]]) -> Vec<f32> {
    run_on(
        &ReferencePlanRuntime::new(CpuComputeAdapter::new()),
        plan,
        values,
    )
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Relative comparison for `f32` arithmetic, whose cross-architecture behaviour
/// this crate does not promise to be bit-identical. The tolerance is stated
/// rather than assumed: one scalar operation per element, no accumulation, so
/// anything beyond a last-place difference would be a real defect.
fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");

    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate()
    {
        let tolerance = 1e-6_f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "{label}[{index}]: got {got}, expected {want}"
        );
    }
}

/// Builds the Reference module of a single-instruction plan.
fn single_kernel_module(plan: &LoweredPlan) -> KernelModule {
    let generated = ReferenceKernelGenerator::new()
        .generate(plan)
        .expect("artefact generation");
    let artifact = generated.artifacts().first().expect("at least one kernel");
    artifact.to_kernel_module().expect("kernel module")
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values
    {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn from_bytes(bytes: &[u8]) -> Vec<f32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "whole 4-byte words only");
    words.iter().copied().map(f32::from_ne_bytes).collect()
}

/// The canonical placeholder geometry `ReferencePlanRuntime` issues.
const CANONICAL_LAUNCH: LaunchConfig = LaunchConfig {
    grid: [1, 1, 1],
    block: [1, 1, 1],
    shared_memory_bytes: 0,
};

// ---------------------------------------------------------------------------
// The eight operations
// ---------------------------------------------------------------------------

#[test]
fn the_eight_operations_execute_on_cuda() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    assert_eq!(
        adapter.capabilities().device.kind(),
        DeviceKind::Cuda,
        "the results below must come from a CUDA device, not a host interpreter"
    );

    let runtime = ReferencePlanRuntime::new(adapter);

    let unary_input = [-2.0_f32, -0.5, 0.0, 3.0];
    let left = [6.0_f32, -4.0, 2.5, 8.0];
    let right = [2.0_f32, 8.0, 0.5, -4.0];

    // Relu — bitwise, because it copies or zeroes a word without arithmetic.
    let plan = unary_plan(Operation::Relu, vec![4], vec![4]);
    let relu = run_on(&runtime, &plan, &[&unary_input]);
    assert_eq!(bits(&relu), bits(&[0.0, 0.0, 0.0, 3.0]), "relu");

    // Scale
    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(-2.0),
        },
        vec![4],
        vec![4],
    );
    assert_close(
        &run_on(&runtime, &plan, &[&unary_input]),
        &[4.0, 1.0, -0.0, -6.0],
        "scale",
    );

    // Add / Sub / Mul / Div
    for (operation, expected, label) in [
        (Operation::Add, [8.0_f32, 4.0, 3.0, 4.0], "add"),
        (Operation::Sub, [4.0, -12.0, 2.0, 12.0], "sub"),
        (Operation::Mul, [12.0, -32.0, 1.25, -32.0], "mul"),
        (Operation::Div, [3.0, -0.5, 5.0, -2.0], "div"),
    ]
    {
        let plan = binary_plan(operation, vec![4]);
        assert_close(&run_on(&runtime, &plan, &[&left, &right]), &expected, label);
    }

    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // ShapeCopy (Reshape) — bitwise.
    let plan = unary_plan(
        Operation::Reshape {
            shape: Shape::new(vec![6]),
        },
        vec![2, 3],
        vec![6],
    );
    assert_eq!(run_on(&runtime, &plan, &[&source]), source.to_vec());

    // Permute (Transpose) — bitwise.
    let plan = unary_plan(
        Operation::Transpose {
            permutation: vec![1, 0],
        },
        vec![2, 3],
        vec![3, 2],
    );
    assert_eq!(
        run_on(&runtime, &plan, &[&source]),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    // Eight programs, eight compilations, eight launches — all on the device.
    let counters = runtime.backend().counters();
    assert_eq!(counters.kernels_compiled, 8, "counters: {counters:?}");
    assert_eq!(counters.kernels_launched, 8, "counters: {counters:?}");
}

// ---------------------------------------------------------------------------
// Bitwise agreement with the CPU interpreter
// ---------------------------------------------------------------------------

#[test]
fn word_moving_opcodes_are_bit_identical_to_the_cpu() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);

    // A NaN with a distinctive payload, both zeros, both infinities and a
    // subnormal: everything a float unit is entitled to mangle and a word move
    // is not.
    let awkward = [
        f32::from_bits(0x7fc0_1234),
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x0000_0001),
    ];

    let copy = unary_plan(
        Operation::Reshape {
            shape: Shape::new(vec![6]),
        },
        vec![2, 3],
        vec![6],
    );
    assert_eq!(
        bits(&run_on(&runtime, &copy, &[&awkward])),
        bits(&run_cpu(&copy, &[&awkward])),
        "ShapeCopy must move words verbatim"
    );

    let permute = unary_plan(
        Operation::Transpose {
            permutation: vec![1, 0],
        },
        vec![2, 3],
        vec![3, 2],
    );
    assert_eq!(
        bits(&run_on(&runtime, &permute, &[&awkward])),
        bits(&run_cpu(&permute, &[&awkward])),
        "Permute must move words verbatim"
    );

    // Relu returns its input untouched for NaN and for strictly positive
    // values, and `+0.0` for everything else — including `-0.0` and `-inf`.
    let relu = unary_plan(Operation::Relu, vec![6], vec![6]);
    let cuda = run_on(&runtime, &relu, &[&awkward]);
    assert_eq!(
        bits(&cuda),
        bits(&run_cpu(&relu, &[&awkward])),
        "Relu must agree with the CPU bit for bit"
    );
    assert_eq!(
        cuda[0].to_bits(),
        0x7fc0_1234,
        "the NaN payload must survive"
    );
    assert_eq!(cuda[1].to_bits(), 0, "-0.0 must become +0.0");
}

#[test]
fn arithmetic_opcodes_agree_with_the_cpu_within_tolerance() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);

    let left = [1.0_f32, -3.5, 1e-8, 6.25, 1e20, -0.0];
    let right = [3.0_f32, 7.0, 3.0, 0.5, 3.0, 4.0];

    for (operation, label) in [
        (Operation::Add, "add"),
        (Operation::Sub, "sub"),
        (Operation::Mul, "mul"),
        (Operation::Div, "div"),
    ]
    {
        let plan = binary_plan(operation, vec![6]);
        let cpu = run_cpu(&plan, &[&left, &right]);
        assert_close(&run_on(&runtime, &plan, &[&left, &right]), &cpu, label);
    }

    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(0.1),
        },
        vec![6],
        vec![6],
    );
    let cpu = run_cpu(&plan, &[&left]);
    assert_close(&run_on(&runtime, &plan, &[&left]), &cpu, "scale");
}

// ---------------------------------------------------------------------------
// Shapes, sizes and geometry
// ---------------------------------------------------------------------------

#[test]
fn scalar_and_zero_element_tensors_round_trip() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);

    // Rank 0: one element, no index arithmetic at all.
    let scalar = unary_plan(Operation::Relu, vec![], vec![]);
    assert_eq!(run_on(&runtime, &scalar, &[&[-4.0]]), vec![0.0]);
    assert_eq!(run_on(&runtime, &scalar, &[&[4.0]]), vec![4.0]);

    let launched_before = runtime.backend().counters().kernels_launched;

    // Zero elements: compiled like any other kernel, launched not at all.
    let empty = unary_plan(Operation::Relu, vec![0, 3], vec![0, 3]);
    assert_eq!(run_on(&runtime, &empty, &[&[]]), Vec::<f32>::new());

    assert_eq!(
        runtime.backend().counters().kernels_launched,
        launched_before,
        "an empty tensor must not be counted as a launched kernel"
    );
}

#[test]
fn a_tensor_larger_than_one_block_is_fully_covered() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // Well past the 256-thread block this adapter launches with, and not a
    // multiple of it, so the tail is exercised too.
    let count = 1000;
    let input: Vec<f32> = (0..count).map(|index| index as f32 - 500.0).collect();
    let expected: Vec<f32> = input.iter().map(|value| value * 0.5).collect();

    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(0.5),
        },
        vec![count],
        vec![count],
    );

    let runtime = ReferencePlanRuntime::new(adapter);
    let cuda = run_on(&runtime, &plan, &[&input]);

    assert_eq!(cuda.len(), count);
    assert_close(&cuda, &expected, "1000-element scale");
}

#[test]
fn a_rank_three_permutation_matches_the_cpu() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // [2, 3, 4] -> [4, 2, 3] under axes [2, 0, 1]: every axis moves, and no
    // stride is 1 except the innermost, so a wrong index formula cannot pass.
    let input: Vec<f32> = (0..24).map(|index| index as f32).collect();
    let plan = unary_plan(
        Operation::Transpose {
            permutation: vec![2, 0, 1],
        },
        vec![2, 3, 4],
        vec![4, 2, 3],
    );

    let runtime = ReferencePlanRuntime::new(adapter);

    assert_eq!(
        bits(&run_on(&runtime, &plan, &[&input])),
        bits(&run_cpu(&plan, &[&input])),
        "a rank-3 permutation must be bit-identical to the CPU"
    );
}

#[test]
fn the_launch_geometry_is_derived_from_the_element_count() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let block = adapter.block_size();
    assert!(block > 0);
    assert!(
        block <= adapter.device_info().max_threads_per_block,
        "the block must respect the device limit"
    );

    for elements in [1_usize, 255, 256, 257, 1000]
    {
        let plan = unary_plan(Operation::Relu, vec![elements], vec![elements]);
        let kernel = adapter
            .compile(&single_kernel_module(&plan))
            .expect("compiles");

        assert_eq!(kernel.elements(), elements);
        assert_eq!(kernel.block_size(), block);

        let expected = u32::try_from(elements.div_ceil(block as usize)).expect("grid fits u32");
        assert_eq!(
            kernel.grid_size(),
            expected.min(adapter.device_info().max_grid_size[0]),
            "grid for {elements} element(s)"
        );
    }

    // Zero elements: no block at all, not one empty block.
    let empty = unary_plan(Operation::Relu, vec![0], vec![0]);
    let kernel = adapter
        .compile(&single_kernel_module(&empty))
        .expect("compiles");
    assert_eq!(kernel.grid_size(), 0);
}

// ---------------------------------------------------------------------------
// Aliasing: one allocation at several read-only arguments
// ---------------------------------------------------------------------------

#[test]
fn a_tensor_added_to_itself_executes() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);
    let input = [1.5_f32, -2.25, 0.0, 8.0];

    // `add(x, x)`: one external value, one device allocation, two kernel
    // arguments. Before the read-only aliasing rule this was refused outright.
    let plan = self_binary_plan(Operation::Add, vec![4]);
    assert_close(
        &run_on(&runtime, &plan, &[&input]),
        &[3.0, -4.5, 0.0, 16.0],
        "add(x, x)",
    );

    let plan = self_binary_plan(Operation::Mul, vec![4]);
    assert_close(
        &run_on(&runtime, &plan, &[&input]),
        &[2.25, 5.0625, 0.0, 64.0],
        "mul(x, x)",
    );

    let plan = self_binary_plan(Operation::Sub, vec![4]);
    assert_close(
        &run_on(&runtime, &plan, &[&input]),
        &[0.0, 0.0, 0.0, 0.0],
        "sub(x, x)",
    );
}

#[test]
fn two_read_only_windows_of_one_allocation_execute() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // Driven through the compute contract directly, because the plan runtime
    // gives every value its own slot and so never produces two offsets into one
    // allocation. The adapter must still accept it.
    let plan = binary_plan(Operation::Add, vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&plan))
        .expect("compiles");
    let stream = adapter.create_stream().expect("stream");

    let operands = adapter
        .allocate(32, 1, MemorySpace::Host)
        .expect("operand allocation");
    let result = adapter
        .allocate(16, 1, MemorySpace::Host)
        .expect("result allocation");

    adapter
        .write(&operands, 0, &to_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .expect("first window");
    adapter
        .write(&operands, 16, &to_bytes(&[10.0, 20.0, 30.0, 40.0]))
        .expect("second window");

    let event = adapter
        .launch(
            &kernel,
            &stream,
            CANONICAL_LAUNCH,
            &[
                BufferBinding {
                    slot: 0,
                    buffer: &operands,
                    offset_bytes: 0,
                    length_bytes: 16,
                    access: BufferAccess::ReadOnly,
                },
                BufferBinding {
                    slot: 1,
                    buffer: &operands,
                    offset_bytes: 16,
                    length_bytes: 16,
                    access: BufferAccess::ReadOnly,
                },
                BufferBinding {
                    slot: 2,
                    buffer: &result,
                    offset_bytes: 0,
                    length_bytes: 16,
                    access: BufferAccess::WriteOnly,
                },
            ],
        )
        .expect("two read-only windows of one allocation must be accepted");

    assert!(event.launched(), "a four-element kernel must be submitted");
    adapter.wait(&event).expect("completion");

    let mut bytes = [0_u8; 16];
    adapter.read(&result, 0, &mut bytes).expect("readback");
    assert_eq!(from_bytes(&bytes), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn a_binding_at_a_non_zero_offset_executes() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let plan = unary_plan(Operation::Relu, vec![4], vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&plan))
        .expect("compiles");
    let stream = adapter.create_stream().expect("stream");

    let operand = adapter
        .allocate(64, 1, MemorySpace::Host)
        .expect("operand allocation");
    let result = adapter
        .allocate(64, 1, MemorySpace::Host)
        .expect("result allocation");

    adapter
        .write(&operand, 16, &to_bytes(&[-1.0, 0.0, 2.0, -3.0]))
        .expect("upload at an offset");

    let event = adapter
        .launch(
            &kernel,
            &stream,
            CANONICAL_LAUNCH,
            &[
                BufferBinding {
                    slot: 0,
                    buffer: &operand,
                    offset_bytes: 16,
                    length_bytes: 16,
                    access: BufferAccess::ReadOnly,
                },
                BufferBinding {
                    slot: 1,
                    buffer: &result,
                    offset_bytes: 32,
                    length_bytes: 16,
                    access: BufferAccess::WriteOnly,
                },
            ],
        )
        .expect("offset bindings must be accepted");

    adapter.wait(&event).expect("completion");

    let mut bytes = [0_u8; 16];
    adapter.read(&result, 32, &mut bytes).expect("readback");
    assert_eq!(from_bytes(&bytes), vec![0.0, 0.0, 2.0, 0.0]);

    // Nothing was written outside the bound window.
    let mut untouched = [0_u8; 32];
    adapter.read(&result, 0, &mut untouched).expect("readback");
    assert_eq!(from_bytes(&untouched), vec![0.0; 8]);
}

// ---------------------------------------------------------------------------
// Multi-node plans and repeated execution
// ---------------------------------------------------------------------------

#[test]
fn a_multi_node_plan_matches_the_cpu_backend() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // a, b -> add -> relu -> scale -> transpose -> reshape -> output.
    let mut graph = Graph::new();
    let ty = f32_type(vec![2, 3]);
    let a = graph.add_input("a", ty.clone()).expect("input");
    let b = graph.add_input("b", ty.clone()).expect("input");
    let sum = graph
        .add_node(Operation::Add, vec![a, b], ty.clone())
        .expect("add");
    let activated = graph
        .add_node(Operation::Relu, vec![sum], ty.clone())
        .expect("relu");
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(0.5),
            },
            vec![activated],
            ty,
        )
        .expect("scale");
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![scaled],
            f32_type(vec![3, 2]),
        )
        .expect("transpose");
    let flattened = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            vec![transposed],
            f32_type(vec![6]),
        )
        .expect("reshape");
    graph.set_outputs(vec![flattened]).expect("outputs");

    let plan = lower(&graph);
    assert!(
        plan.instructions().len() >= 5,
        "the plan must really carry five dispatches"
    );

    let left = [-3.0_f32, -1.0, 0.0, 1.0, 4.0, 9.0];
    let right = [1.0_f32; 6];

    let runtime = ReferencePlanRuntime::new(adapter);
    let cuda = run_on(&runtime, &plan, &[&left, &right]);
    let cpu = run_cpu(&plan, &[&left, &right]);

    assert_eq!(cuda.len(), 6);
    assert_close(&cuda, &cpu, "multi-node plan");

    let counters = runtime.backend().counters();
    assert!(
        counters.kernels_launched >= 5,
        "every dispatch must reach the device: {counters:?}"
    );
}

#[test]
fn one_prepared_plan_serves_several_executions() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);
    let plan = unary_plan(Operation::Relu, vec![4], vec![4]);
    let prepared = runtime.prepare(&plan).expect("preparable plan");

    let compiled_after_prepare = runtime.backend().counters().kernels_compiled;

    let first = [-1.0_f32, 2.0, -3.0, 4.0];
    let second = [5.0_f32, -6.0, 7.0, -8.0];

    for _ in 0..3
    {
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &first);
        let outputs = runtime.execute(&prepared, &values).expect("first run");
        assert_eq!(outputs.into_values()[0].values, vec![0.0, 2.0, 0.0, 4.0]);

        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &second);
        let outputs = runtime.execute(&prepared, &values).expect("second run");
        assert_eq!(outputs.into_values()[0].values, vec![5.0, 0.0, 7.0, 0.0]);
    }

    let counters = runtime.backend().counters();
    assert_eq!(
        counters.kernels_compiled, compiled_after_prepare,
        "nothing may be compiled during execution: {counters:?}"
    );
    assert_eq!(
        counters.kernels_launched, 6,
        "six executions, six launches: {counters:?}"
    );
}

// ---------------------------------------------------------------------------
// Refusals — never a silent CPU fallback
// ---------------------------------------------------------------------------

#[test]
fn exp_and_log_are_refused_by_the_adapter_itself() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    for operation in [Operation::Exp, Operation::Log]
    {
        let label = format!("{operation:?}");
        let plan = unary_plan(operation, vec![4], vec![4]);
        let module = single_kernel_module(&plan);

        let error = adapter
            .compile(&module)
            .expect_err("Exp and Log must be refused");
        assert!(
            matches!(error, ComputeError::Unsupported(_)),
            "{label} must be refused as unsupported, got {error:?}"
        );
    }

    // Refused before NVRTC was ever invoked.
    assert_eq!(adapter.counters().kernels_compiled, 0);
}

#[test]
fn a_non_reference_module_is_refused() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let ptx = KernelModule::new(
        KernelFormat::Ptx,
        "kernel",
        b".version 8.0\n.target sm_80\n".to_vec(),
    )
    .expect("valid module");
    assert!(matches!(
        adapter.compile(&ptx),
        Err(ComputeError::Unsupported(_))
    ));

    let wgsl = KernelModule::new(
        KernelFormat::Wgsl,
        "main",
        b"@compute @workgroup_size(1) fn main() {}".to_vec(),
    )
    .expect("valid module");
    assert!(matches!(
        adapter.compile(&wgsl),
        Err(ComputeError::Unsupported(_))
    ));

    // A Reference module whose bytes are not a valid artefact.
    let corrupt = KernelModule::new(
        KernelFormat::Reference,
        "scirust_reference_kernel_0",
        vec![0xff; 8],
    )
    .expect("valid module");
    assert!(matches!(
        adapter.compile(&corrupt),
        Err(ComputeError::Compilation(_))
    ));

    assert_eq!(adapter.counters().kernels_compiled, 0);
}

#[test]
fn a_non_canonical_launch_configuration_is_refused() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let plan = unary_plan(Operation::Relu, vec![4], vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&plan))
        .expect("compiles");
    let stream = adapter.create_stream().expect("stream");

    let operand = adapter.allocate(16, 1, MemorySpace::Host).expect("operand");
    let result = adapter.allocate(16, 1, MemorySpace::Host).expect("result");

    let bindings = [
        BufferBinding {
            slot: 0,
            buffer: &operand,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::ReadOnly,
        },
        BufferBinding {
            slot: 1,
            buffer: &result,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::WriteOnly,
        },
    ];

    let wrong = LaunchConfig {
        grid: [4, 1, 1],
        block: [64, 1, 1],
        shared_memory_bytes: 0,
    };
    assert!(matches!(
        adapter.launch(&kernel, &stream, wrong, &bindings),
        Err(ComputeError::Unsupported(_))
    ));

    // A write bound where the contract expects a read is refused too.
    let swapped = [
        BufferBinding {
            slot: 0,
            buffer: &operand,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::ReadWrite,
        },
        BufferBinding {
            slot: 1,
            buffer: &result,
            offset_bytes: 0,
            length_bytes: 16,
            access: BufferAccess::WriteOnly,
        },
    ];
    assert!(matches!(
        adapter.launch(&kernel, &stream, CANONICAL_LAUNCH, &swapped),
        Err(ComputeError::Unsupported(_))
    ));

    assert_eq!(
        adapter.counters().kernels_launched,
        0,
        "a refused launch must not be counted"
    );
}

// ---------------------------------------------------------------------------
// Generated source
// ---------------------------------------------------------------------------

#[test]
fn the_generated_cuda_c_is_deterministic_and_specialised() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(0.25),
        },
        vec![7],
        vec![7],
    );
    let module = single_kernel_module(&plan);

    let first = adapter.compile(&module).expect("compiles");
    let second = adapter.compile(&module).expect("compiles again");

    assert_eq!(
        first.source(),
        second.source(),
        "the same artefact must produce byte-identical source"
    );
    println!("{}", first.source());

    // The element count is baked in, and the factor comes from its raw bits
    // rather than a formatted decimal.
    assert!(
        first.source().contains("total = 7ull"),
        "source: {}",
        first.source()
    );
    assert!(
        first.source().contains("__uint_as_float(0x3e800000u)"),
        "source: {}",
        first.source()
    );
    // The generated function's name carries no kernel id, no address and no
    // counter, so two structurally identical artefacts produce the same bytes.
    assert!(
        !first.source().contains("scirust_reference_kernel_"),
        "source: {}",
        first.source()
    );

    assert_eq!(first.elements(), 7);
    assert_eq!(first.entry_point(), "scirust_reference_kernel_0");
    assert_eq!(first.cuda_entry_point(), "scirust_reference_kernel");
    assert_eq!(first.kernel_id(), 0);
}

#[test]
fn the_parameter_order_is_operands_then_result() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let plan = binary_plan(Operation::Sub, vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&plan))
        .expect("compiles");
    let source = kernel.source();

    let operand_0 = source.find("const float* operand_0").expect("operand 0");
    let operand_1 = source.find("const float* operand_1").expect("operand 1");
    let result = source.find("float* result").expect("result");

    assert!(operand_0 < operand_1, "source: {source}");
    assert!(operand_1 < result, "source: {source}");
    assert!(
        source.contains("result[index] = operand_0[index] - operand_1[index];"),
        "source: {source}"
    );
}

#[test]
fn word_moving_kernels_address_words_and_arithmetic_kernels_address_floats() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let copy = unary_plan(
        Operation::Reshape {
            shape: Shape::new(vec![6]),
        },
        vec![2, 3],
        vec![6],
    );
    let kernel = adapter
        .compile(&single_kernel_module(&copy))
        .expect("compiles");
    assert!(
        kernel.source().contains("const unsigned int* operand_0"),
        "ShapeCopy must move 32-bit words: {}",
        kernel.source()
    );

    let add = binary_plan(Operation::Add, vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&add))
        .expect("compiles");
    assert!(
        kernel.source().contains("const float* operand_0"),
        "Add must operate on f32: {}",
        kernel.source()
    );
    assert!(!kernel.source().contains("unsigned int*"));
}

// ---------------------------------------------------------------------------
// The device really is a CUDA device
// ---------------------------------------------------------------------------

#[test]
fn the_adapter_reports_an_honest_cuda_identity() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let capabilities = adapter.capabilities();
    assert_eq!(capabilities.device.kind(), DeviceKind::Cuda);
    assert!(capabilities.supports_dtype(DType::F32));
    assert!(
        !capabilities.supports_dtype(DType::F64),
        "this adapter executes F32 only and must not claim otherwise"
    );
    assert!(!capabilities.supports_dtype(DType::Bf16));
    assert!(capabilities.max_workgroup_size.iter().all(|size| *size > 0));

    let info = adapter.device_info();
    assert_eq!(info.ordinal, DEVICE_ORDINAL, "no implicit device selection");
    assert!(!info.name.is_empty());
    assert!(info.total_memory_bytes > 0);
    assert!(info.compute_capability.0 > 0);

    // A fresh adapter has done nothing, and says so.
    assert_eq!(adapter.counters(), Default::default());
}

#[test]
fn an_ordinal_beyond_the_device_count_is_refused_without_falling_back() {
    if adapter_or_skip().is_none()
    {
        return;
    }

    // Far beyond any plausible device count: the request must fail rather than
    // quietly open device zero.
    let error = CudaReferenceAdapter::new(4096).expect_err("no such device");
    assert!(
        matches!(error, ComputeError::InvalidArgument(_)),
        "got {error:?}"
    );
}

// ---------------------------------------------------------------------------
// The fixtures themselves
// ---------------------------------------------------------------------------

#[test]
fn plans_are_built_through_the_real_canonical_pipeline() {
    // Guards the fixtures: a hand-built plan would not exercise the compiler or
    // the lowerer, and the tests above would prove much less.
    let plan = binary_plan(Operation::Add, vec![2, 2]);

    assert_eq!(plan.instructions().len(), 1);
    assert_eq!(plan.bindings().len(), 2);
    assert_eq!(plan.outputs().len(), 1);
    assert_eq!(plan.outputs()[0].node, NodeId::new(2));

    // `add(x, x)` really does reduce to one external binding, which is what
    // makes one allocation appear at two kernel arguments.
    let aliased = self_binary_plan(Operation::Add, vec![4]);
    assert_eq!(aliased.bindings().len(), 1);
}
