//! The canonical Reference pipeline executed for real on a WGPU device.
//!
//! ```text
//! Graph -> CanonicalCompiler -> KernelLowerer -> LoweredPlan
//!       -> ReferencePlanRuntime<WgpuReferenceAdapter>
//!       -> generated WGSL -> compute pipeline -> dispatch -> readback
//! ```
//!
//! Nothing here is mocked and nothing falls back: every value below crossed a
//! real WGPU queue, or the test did not run at all.
//!
//! # Running these
//!
//! A machine without a Vulkan/Metal/DX12/GL driver has no adapter, and each
//! test then **skips**. That is convenient locally and dangerous in CI, so the
//! skip is opt-out: set `SCIRUST_REQUIRE_WGPU=1` and a missing adapter becomes
//! a failure instead of a silent pass. The CI lavapipe job sets it.

#![cfg(feature = "wgpu")]

use scirust_compute::{DType, DeviceKind, KernelFormat, Shape};
use scirust_gpu::{CpuComputeAdapter, WgpuReferenceAdapter};
use scirust_tensor_compile::{
    CanonicalCompiler, ExternalBindings, KernelLowerer, LogicalBindingId, LoweredPlan,
};
use scirust_tensor_ir::{Graph, NodeId, Operation, Scalar, TensorType};
use scirust_tensor_reference::ReferenceKernelGenerator;
use scirust_tensor_runtime::{PlanExternalValues, ReferencePlanRuntime};

// ---------------------------------------------------------------------------
// Device acquisition
// ---------------------------------------------------------------------------

/// Acquire an adapter, or skip — unless the caller demanded a real device.
fn adapter_or_skip() -> Option<WgpuReferenceAdapter> {
    match WgpuReferenceAdapter::new()
    {
        Ok(adapter) =>
        {
            let info = adapter.adapter_info();
            eprintln!(
                "wgpu adapter: {} [{}] driver='{}' class={:?} hardware={}",
                info.name,
                info.backend,
                info.driver,
                info.class,
                info.class.is_hardware()
            );
            Some(adapter)
        },
        Err(error) =>
        {
            assert!(
                std::env::var_os("SCIRUST_REQUIRE_WGPU").is_none(),
                "SCIRUST_REQUIRE_WGPU is set, so a real WGPU device is mandatory, but none \
                 could be acquired: {error}"
            );
            eprintln!("skipping: no WGPU adapter available ({error})");
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

/// Prepares and runs one plan on an existing runtime, returning the single
/// output's values.
///
/// Taking the runtime by reference matters: acquiring a WGPU device is not free
/// and some drivers dislike dozens of them in one process, so each test builds
/// at most one.
fn run_on<B: scirust_compute::ComputeBackend>(
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

/// Relative comparison for `f32` arithmetic, whose cross-backend behaviour this
/// crate does not promise to be bit-identical.
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

// ---------------------------------------------------------------------------
// The eight operations
// ---------------------------------------------------------------------------

#[test]
fn the_eight_operations_execute_on_wgpu() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    assert_eq!(
        adapter.capabilities().device.kind(),
        DeviceKind::Wgpu,
        "the results below must come from a WGPU device, not a host interpreter"
    );

    let runtime = ReferencePlanRuntime::new(adapter);

    let unary_input = [-2.0_f32, -0.5, 0.0, 3.0];
    let left = [6.0_f32, -4.0, 2.5, 8.0];
    let right = [2.0_f32, 8.0, 0.5, -4.0];

    // Relu
    let plan = unary_plan(Operation::Relu, vec![4], vec![4]);
    let prepared = runtime.prepare(&plan).expect("relu prepares");
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &unary_input);
    let outputs = runtime.execute(&prepared, &values).expect("relu runs");
    assert_close(
        &outputs.into_values()[0].values,
        &[0.0, 0.0, 0.0, 3.0],
        "relu",
    );

    // Scale
    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(-2.0),
        },
        vec![4],
        vec![4],
    );
    let prepared = runtime.prepare(&plan).expect("scale prepares");
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &unary_input);
    let outputs = runtime.execute(&prepared, &values).expect("scale runs");
    assert_close(
        &outputs.into_values()[0].values,
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
        let prepared = runtime.prepare(&plan).expect("binary prepares");
        let mut values = PlanExternalValues::new();
        values
            .bind(LogicalBindingId::new(0), &left)
            .bind(LogicalBindingId::new(1), &right);
        let outputs = runtime.execute(&prepared, &values).expect("binary runs");
        assert_close(&outputs.into_values()[0].values, &expected, label);
    }

    // ShapeCopy (Reshape)
    let plan = unary_plan(
        Operation::Reshape {
            shape: Shape::new(vec![6]),
        },
        vec![2, 3],
        vec![6],
    );
    let prepared = runtime.prepare(&plan).expect("reshape prepares");
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &source);
    let outputs = runtime.execute(&prepared, &values).expect("reshape runs");
    assert_eq!(outputs.into_values()[0].values, source.to_vec());

    // Permute (Transpose)
    let plan = unary_plan(
        Operation::Transpose {
            permutation: vec![1, 0],
        },
        vec![2, 3],
        vec![3, 2],
    );
    let prepared = runtime.prepare(&plan).expect("transpose prepares");
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &source);
    let outputs = runtime.execute(&prepared, &values).expect("transpose runs");
    assert_eq!(
        outputs.into_values()[0].values,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// Bit-exactness, where it is actually promised
// ---------------------------------------------------------------------------

#[test]
fn structural_opcodes_are_bit_identical_to_the_cpu() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let nan = f32::from_bits(0x7fc0_dead);
    let other_nan = f32::from_bits(0xffc0_0001);
    let payload = [
        -0.0_f32,
        nan,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 4.0,
        other_nan,
    ];

    // ShapeCopy: [2, 3] -> [6].
    let runtime = ReferencePlanRuntime::new(adapter);

    let copy = unary_plan(
        Operation::Reshape {
            shape: Shape::new(vec![6]),
        },
        vec![2, 3],
        vec![6],
    );
    let gpu = run_on(&runtime, &copy, &[&payload]);
    let cpu = run_cpu(&copy, &[&payload]);

    assert_eq!(bits(&gpu), bits(&payload), "ShapeCopy must move exact bits");
    assert_eq!(
        bits(&gpu),
        bits(&cpu),
        "ShapeCopy must match the CPU exactly"
    );

    // Permute: [2, 3] -> [3, 2].
    let permute = unary_plan(
        Operation::Transpose {
            permutation: vec![1, 0],
        },
        vec![2, 3],
        vec![3, 2],
    );
    let gpu = run_on(&runtime, &permute, &[&payload]);
    let cpu = run_cpu(&permute, &[&payload]);

    assert_eq!(bits(&gpu), bits(&cpu), "Permute must match the CPU exactly");
    // Row-major [[a,b,c],[d,e,f]] transposed is [a,d,b,e,c,f].
    assert_eq!(
        bits(&gpu),
        bits(&[
            payload[0], payload[3], payload[1], payload[4], payload[2], payload[5]
        ])
    );
}

// ---------------------------------------------------------------------------
// Multi-node plans, against the CPU oracle
// ---------------------------------------------------------------------------

#[test]
fn a_multi_node_plan_matches_the_cpu_backend() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // a, b -> Add -> Scale -> Relu -> Permute -> Reshape -> output
    let ty = f32_type(vec![2, 3]);
    let mut graph = Graph::new();
    let lhs = graph.add_input("a", ty.clone()).expect("input");
    let rhs = graph.add_input("b", ty.clone()).expect("input");
    let sum = graph
        .add_node(Operation::Add, vec![lhs, rhs], ty.clone())
        .expect("add");
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(0.5),
            },
            vec![sum],
            ty.clone(),
        )
        .expect("scale");
    let activated = graph
        .add_node(Operation::Relu, vec![scaled], ty)
        .expect("relu");
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![activated],
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
    let left = [1.0_f32, -3.0, 5.0, -7.0, 9.0, -11.0];
    let right = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0];

    let gpu = run_on(&ReferencePlanRuntime::new(adapter), &plan, &[&left, &right]);
    let cpu = run_cpu(&plan, &[&left, &right]);

    assert_close(&gpu, &cpu, "multi-node plan versus the CPU oracle");
    assert_eq!(gpu.len(), 6);
}

// ---------------------------------------------------------------------------
// Preparation, reuse and shape edge cases
// ---------------------------------------------------------------------------

#[test]
fn one_prepared_plan_serves_several_executions() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // x -> relu -> relu -> output: two dispatches sharing one logical kernel,
    // so the shader is generated and compiled exactly once.
    let ty = f32_type(vec![4]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).expect("input");
    let first = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .expect("relu");
    let second = graph
        .add_node(Operation::Relu, vec![first], ty)
        .expect("relu");
    graph.set_outputs(vec![second]).expect("outputs");

    let plan = lower(&graph);
    let runtime = ReferencePlanRuntime::new(adapter);
    let prepared = runtime.prepare(&plan).expect("preparable plan");

    assert_eq!(prepared.dispatch_count(), 2);
    assert_eq!(
        prepared.kernel_count(),
        1,
        "two dispatches sharing a signature compile one shader"
    );

    for (input_values, expected) in [
        ([-1.0_f32, 2.0, -3.0, 4.0], [0.0_f32, 2.0, 0.0, 4.0]),
        ([5.0, -6.0, 7.0, -8.0], [5.0, 0.0, 7.0, 0.0]),
    ]
    {
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &input_values);
        let outputs = runtime.execute(&prepared, &values).expect("execution");
        assert_close(&outputs.into_values()[0].values, &expected, "reused plan");
    }
}

#[test]
fn identical_inputs_reproduce_identical_bits_on_one_device() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(0.1),
        },
        vec![5],
        vec![5],
    );
    let runtime = ReferencePlanRuntime::new(adapter);
    let prepared = runtime.prepare(&plan).expect("preparable plan");

    let input = [1.0_f32, -0.0, 3.3, -7.7, 1e-8];
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &input);

    let first = runtime.execute(&prepared, &values).expect("first run");
    let second = runtime.execute(&prepared, &values).expect("second run");

    assert_eq!(
        bits(&first.into_values()[0].values),
        bits(&second.into_values()[0].values),
        "the same device and the same inputs must reproduce exactly"
    );
}

#[test]
fn scalar_and_zero_element_tensors_round_trip() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let runtime = ReferencePlanRuntime::new(adapter);

    let scalar = unary_plan(Operation::Relu, Vec::new(), Vec::new());
    assert_close(
        &run_on(&runtime, &scalar, &[&[-4.0]]),
        &[0.0],
        "scalar relu",
    );

    // Zero elements: nothing to dispatch, and nothing to read back.
    let empty = unary_plan(Operation::Relu, vec![0, 3], vec![0, 3]);
    let empty_input: [f32; 0] = [];
    assert!(run_on(&runtime, &empty, &[&empty_input]).is_empty());
}

#[test]
fn a_tensor_larger_than_one_workgroup_is_fully_covered() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    // 1000 elements is far more than the 64 invocations a single workgroup
    // provides, so this fails loudly if the dispatch grid were taken literally
    // from the runtime's canonical [1, 1, 1].
    let count = 1000;
    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(3.0),
        },
        vec![count],
        vec![count],
    );

    let input: Vec<f32> = (0..count).map(|index| index as f32).collect();
    let expected: Vec<f32> = input.iter().map(|value| value * 3.0).collect();

    let gpu = run_on(&ReferencePlanRuntime::new(adapter), &plan, &[&input]);
    assert_eq!(gpu.len(), count);
    assert_close(&gpu, &expected, "1000-element scale");
}

// ---------------------------------------------------------------------------
// Refusals — never a silent CPU fallback
// ---------------------------------------------------------------------------

/// Builds the Reference module of a single-instruction plan.
fn single_kernel_module(plan: &LoweredPlan) -> scirust_compute::KernelModule {
    let generated = ReferenceKernelGenerator::new()
        .generate(plan)
        .expect("artefact generation");
    let artifact = generated.artifacts().first().expect("at least one kernel");
    artifact.to_kernel_module().expect("kernel module")
}

#[test]
fn exp_and_log_are_refused_by_the_adapter_itself() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    use scirust_compute::{ComputeBackend, ComputeError};

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
}

#[test]
fn a_non_reference_module_is_refused() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    use scirust_compute::{ComputeBackend, ComputeError, KernelModule};

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
}

// ---------------------------------------------------------------------------
// Generated source
// ---------------------------------------------------------------------------

#[test]
fn the_generated_wgsl_is_deterministic_and_specialised() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    use scirust_compute::ComputeBackend;

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
        first.wgsl(),
        second.wgsl(),
        "the same artefact must produce byte-identical source"
    );
    assert!(
        first.wgsl().contains("@compute @workgroup_size(64)"),
        "source: {}",
        first.wgsl()
    );
    // The element count is baked in, and the factor comes from its raw bits
    // rather than a formatted decimal.
    assert!(first.wgsl().contains("7u"), "source: {}", first.wgsl());
    assert!(
        first.wgsl().contains("bitcast<f32>(0x3e800000u)"),
        "source: {}",
        first.wgsl()
    );
    assert_eq!(first.elements(), 7);
    assert_eq!(first.workgroups(), 1);
    assert_eq!(first.entry_point(), "scirust_reference_kernel_0");
    assert_eq!(first.kernel_id(), 0);
}

#[test]
fn structural_kernels_move_words_and_arithmetic_kernels_do_not() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    use scirust_compute::ComputeBackend;

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
        kernel.wgsl().contains("array<u32>"),
        "ShapeCopy must move u32 words: {}",
        kernel.wgsl()
    );

    let add = binary_plan(Operation::Add, vec![4]);
    let kernel = adapter
        .compile(&single_kernel_module(&add))
        .expect("compiles");
    assert!(
        kernel.wgsl().contains("array<f32>"),
        "Add must operate on f32: {}",
        kernel.wgsl()
    );
    assert!(!kernel.wgsl().contains("array<u32>"));
}

// ---------------------------------------------------------------------------
// The device really is a GPU device
// ---------------------------------------------------------------------------

#[test]
fn the_adapter_reports_an_honest_wgpu_identity() {
    let Some(adapter) = adapter_or_skip()
    else
    {
        return;
    };

    let capabilities = adapter.capabilities();
    assert_eq!(capabilities.device.kind(), DeviceKind::Wgpu);
    assert!(capabilities.supports_dtype(DType::F32));
    assert!(
        !capabilities.supports_dtype(DType::F64),
        "this adapter executes F32 only and must not claim otherwise"
    );
    assert!(capabilities.max_workgroup_size.iter().all(|size| *size > 0));

    let info = adapter.adapter_info();
    assert!(!info.name.is_empty());
    assert!(!info.backend.is_empty());
    // Whether the device is hardware or a software rasteriser is reported, not
    // guessed at, and never hidden.
    assert_eq!(
        info.class.is_hardware(),
        info.class != scirust_gpu::WgpuDeviceClass::SoftwareCpu
            && info.class != scirust_gpu::WgpuDeviceClass::Other
    );
}

// ---------------------------------------------------------------------------
// A node identifier is used above only through the public plan API.
// ---------------------------------------------------------------------------

#[test]
fn plans_are_built_through_the_real_canonical_pipeline() {
    // Guards the fixtures themselves: a hand-built plan would not exercise the
    // compiler or the lowerer, and the tests above would prove much less.
    let plan = binary_plan(Operation::Add, vec![2, 2]);

    assert_eq!(plan.instructions().len(), 1);
    assert_eq!(plan.bindings().len(), 2);
    assert_eq!(plan.outputs().len(), 1);
    assert_eq!(plan.outputs()[0].node, NodeId::new(2));
}
