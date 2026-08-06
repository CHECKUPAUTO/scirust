//! End-to-end tests of the Reference plan runtime on the real CPU backend.
//!
//! Every test drives the genuine pipeline — no mock, no hand-built plan:
//!
//! ```text
//! Graph -> CanonicalCompiler -> ExecutionPlan -> MemoryPlan
//!       -> KernelLowerer -> LoweredPlan
//!       -> ReferencePlanRuntime::prepare / execute
//!       -> CpuComputeAdapter -> ReferenceInterpreter
//! ```
//!
//! Bit-level assertions use `to_bits()` wherever the semantics demand it:
//! signed zeros and NaN payloads are indistinguishable under `==` (or compare
//! unequal, for NaN), so value equality would silently pass where the contract
//! requires an exact bit pattern.

use scirust_compute::{DType, Shape};
use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_compile::{
    CanonicalCompiler, ExternalBindings, ExternalValueKind, KernelLowerer, LogicalBindingId,
    LoweredPlan,
};
use scirust_tensor_ir::{ConstantId, Graph, NodeId, Operation, Scalar, TensorType};
use scirust_tensor_runtime::{
    PlanExecutionError, PlanExternalValues, PlanPreparationError, PreparedReferencePlan,
    ReferencePlanRuntime,
};

// ---------------------------------------------------------------------------
// Fixtures
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

fn runtime() -> ReferencePlanRuntime<CpuComputeAdapter> {
    ReferencePlanRuntime::new(CpuComputeAdapter::new())
}

fn prepare(plan: &LoweredPlan) -> PreparedReferencePlan<scirust_gpu::CpuKernel> {
    runtime().prepare(plan).expect("preparable plan")
}

/// `input -> op -> output`.
fn unary_plan(op: Operation, dims: Vec<usize>) -> LoweredPlan {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![input], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

/// `a, b -> op -> output`.
fn binary_plan(op: Operation, dims: Vec<usize>) -> LoweredPlan {
    let ty = f32_type(dims);
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", ty.clone()).unwrap();
    let rhs = graph.add_input("rhs", ty.clone()).unwrap();
    let result = graph.add_node(op, vec![lhs, rhs], ty).unwrap();
    graph.set_outputs(vec![result]).unwrap();
    lower(&graph)
}

fn run_unary(plan: &LoweredPlan, input: &[f32]) -> Vec<f32> {
    let runtime = runtime();
    let prepared = runtime.prepare(plan).expect("preparable plan");
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), input);

    let outputs = runtime.execute(&prepared, &values).expect("successful run");
    assert_eq!(outputs.len(), 1);
    outputs.into_values()[0].values.clone()
}

fn run_binary(plan: &LoweredPlan, left: &[f32], right: &[f32]) -> Vec<f32> {
    let runtime = runtime();
    let prepared = runtime.prepare(plan).expect("preparable plan");
    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), left)
        .bind(LogicalBindingId::new(1), right);

    let outputs = runtime.execute(&prepared, &values).expect("successful run");
    assert_eq!(outputs.len(), 1);
    outputs.into_values()[0].values.clone()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// The eight supported opcodes
// ---------------------------------------------------------------------------

#[test]
fn executes_relu() {
    let plan = unary_plan(Operation::Relu, vec![4]);
    assert_eq!(
        bits(&run_unary(&plan, &[-1.5, 0.0, 2.5, -0.0])),
        bits(&[0.0, 0.0, 2.5, 0.0])
    );
}

#[test]
fn executes_scale() {
    let plan = unary_plan(
        Operation::Scale {
            factor: Scalar::f32(0.5),
        },
        vec![3],
    );
    assert_eq!(run_unary(&plan, &[1.0, -2.0, 3.0]), vec![0.5, -1.0, 1.5]);
}

#[test]
fn executes_the_four_binary_operations() {
    let cases = [
        (Operation::Add, vec![5.0f32, 7.0, 9.0]),
        (Operation::Sub, vec![-3.0, -3.0, -3.0]),
        (Operation::Mul, vec![4.0, 10.0, 18.0]),
        (Operation::Div, vec![0.25, 0.4, 0.5]),
    ];

    for (operation, expected) in cases
    {
        let label = format!("{operation:?}");
        let plan = binary_plan(operation, vec![3]);
        assert_eq!(
            run_binary(&plan, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            expected,
            "unexpected result for {label}"
        );
    }
}

#[test]
fn executes_shape_copy() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let reshaped = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            vec![input],
            f32_type(vec![6]),
        )
        .unwrap();
    graph.set_outputs(vec![reshaped]).unwrap();
    let plan = lower(&graph);

    assert_eq!(
        run_unary(&plan, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn executes_permute() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![input],
            f32_type(vec![3, 2]),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();
    let plan = lower(&graph);

    assert_eq!(
        run_unary(&plan, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// Multi-kernel plans
// ---------------------------------------------------------------------------

#[test]
fn executes_scale_then_relu() {
    let ty = f32_type(vec![4]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(-2.0),
            },
            vec![input],
            ty.clone(),
        )
        .unwrap();
    let activated = graph.add_node(Operation::Relu, vec![scaled], ty).unwrap();
    graph.set_outputs(vec![activated]).unwrap();
    let plan = lower(&graph);

    let prepared = prepare(&plan);
    assert_eq!(prepared.dispatch_count(), 2);

    // -2 * [1, -1, 3, -4] = [-2, 2, -6, 8]; relu -> [0, 2, 0, 8].
    assert_eq!(
        run_unary(&plan, &[1.0, -1.0, 3.0, -4.0]),
        vec![0.0, 2.0, 0.0, 8.0]
    );
}

#[test]
fn executes_add_then_scale() {
    let ty = f32_type(vec![3]);
    let mut graph = Graph::new();
    let lhs = graph.add_input("a", ty.clone()).unwrap();
    let rhs = graph.add_input("b", ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![lhs, rhs], ty.clone())
        .unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(10.0),
            },
            vec![sum],
            ty,
        )
        .unwrap();
    graph.set_outputs(vec![scaled]).unwrap();
    let plan = lower(&graph);

    assert_eq!(
        run_binary(&plan, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
        vec![50.0, 70.0, 90.0]
    );
}

#[test]
fn executes_permute_then_shape_copy() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 3])).unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![input],
            f32_type(vec![3, 2]),
        )
        .unwrap();
    let flattened = graph
        .add_node(
            Operation::Reshape {
                shape: Shape::new(vec![6]),
            },
            vec![transposed],
            f32_type(vec![6]),
        )
        .unwrap();
    graph.set_outputs(vec![flattened]).unwrap();
    let plan = lower(&graph);

    assert_eq!(
        run_unary(&plan, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

// ---------------------------------------------------------------------------
// Kernel reuse and plan reuse
// ---------------------------------------------------------------------------

#[test]
fn a_kernel_shared_by_several_dispatches_is_compiled_once() {
    // Two Relu dispatches over the same tensor type share one logical kernel,
    // because the lowerer deduplicates identical signatures.
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let first = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .unwrap();
    let second = graph.add_node(Operation::Relu, vec![first], ty).unwrap();
    graph.set_outputs(vec![second]).unwrap();
    let plan = lower(&graph);

    let prepared = prepare(&plan);

    assert_eq!(prepared.dispatch_count(), 2);
    assert_eq!(
        prepared.kernel_count(),
        1,
        "one compiled kernel must back both dispatches"
    );
}

#[test]
fn one_prepared_plan_serves_several_executions_with_different_values() {
    let plan = unary_plan(Operation::Relu, vec![3]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let first_input = [-1.0f32, 2.0, -3.0];
    let mut first_values = PlanExternalValues::new();
    first_values.bind(LogicalBindingId::new(0), &first_input);
    let first = runtime.execute(&prepared, &first_values).unwrap();

    let second_input = [5.0f32, -6.0, 7.0];
    let mut second_values = PlanExternalValues::new();
    second_values.bind(LogicalBindingId::new(0), &second_input);
    let second = runtime.execute(&prepared, &second_values).unwrap();

    assert_eq!(first.values()[0].values, vec![0.0, 2.0, 0.0]);
    assert_eq!(second.values()[0].values, vec![5.0, 0.0, 7.0]);
}

#[test]
fn identical_inputs_reproduce_identical_bits() {
    let plan = binary_plan(Operation::Div, vec![4]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let left = [1.0f32, 3.0, -7.0, 1e30];
    let right = [3.0f32, 7.0, 11.0, 7.0];

    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), &left)
        .bind(LogicalBindingId::new(1), &right);

    let first = runtime.execute(&prepared, &values).unwrap();
    let second = runtime.execute(&prepared, &values).unwrap();

    assert_eq!(
        bits(&first.values()[0].values),
        bits(&second.values()[0].values)
    );
}

#[test]
fn preparing_the_same_plan_twice_yields_the_same_logical_layout() {
    let plan = unary_plan(Operation::Relu, vec![2, 2]);
    let runtime = runtime();

    let first = runtime.prepare(&plan).unwrap();
    let second = runtime.prepare(&plan).unwrap();

    assert_eq!(first.kernel_count(), second.kernel_count());
    assert_eq!(first.dispatch_count(), second.dispatch_count());
    assert_eq!(first.buffer_slots(), second.buffer_slots());
    assert_eq!(first.external_values(), second.external_values());
    assert_eq!(first.outputs(), second.outputs());
    assert_eq!(first.launch_config(), second.launch_config());
}

// ---------------------------------------------------------------------------
// Memory layout
// ---------------------------------------------------------------------------

#[test]
fn a_slot_reused_across_disjoint_lifetimes_backs_one_physical_buffer() {
    // x -> relu -> exp-free chain of three same-typed values: the planner
    // recycles the first slot for the third value.
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let first = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .unwrap();
    let second = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(2.0),
            },
            vec![first],
            ty.clone(),
        )
        .unwrap();
    let third = graph.add_node(Operation::Relu, vec![second], ty).unwrap();
    graph.set_outputs(vec![third]).unwrap();
    let plan = lower(&graph);

    let prepared = prepare(&plan);

    // Three dispatches, but the memory planner needs fewer slots than values.
    assert_eq!(prepared.dispatch_count(), 3);
    assert!(
        prepared.buffer_slots().len() < 3,
        "slot reuse must survive into the runtime's physical layout"
    );

    // The final output is still correct despite that reuse.
    assert_eq!(run_unary(&plan, &[-1.0, 3.0]), vec![0.0, 6.0]);
}

#[test]
fn executes_a_scalar_tensor() {
    let plan = unary_plan(Operation::Relu, vec![]);
    assert_eq!(run_unary(&plan, &[-4.0]), vec![0.0]);
}

#[test]
fn executes_a_zero_element_tensor() {
    let plan = unary_plan(Operation::Relu, vec![0, 3]);
    let result = run_unary(&plan, &[]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// External values
// ---------------------------------------------------------------------------

#[test]
fn external_inputs_and_constants_are_both_supplied_by_the_caller() {
    let ty = f32_type(vec![3]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let constant = graph.add_constant(ConstantId::new(7), ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![input, constant], ty)
        .unwrap();
    graph.set_outputs(vec![sum]).unwrap();
    let plan = lower(&graph);

    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let kinds: Vec<ExternalValueKind> = prepared
        .external_values()
        .iter()
        .map(|spec| spec.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![ExternalValueKind::Input, ExternalValueKind::Constant]
    );

    let input_values = [1.0f32, 2.0, 3.0];
    let constant_values = [10.0f32, 20.0, 30.0];
    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), &input_values)
        .bind(LogicalBindingId::new(1), &constant_values);

    let outputs = runtime.execute(&prepared, &values).unwrap();
    assert_eq!(outputs.values()[0].values, vec![11.0, 22.0, 33.0]);

    // The caller's constant slice is untouched.
    assert_eq!(constant_values, [10.0, 20.0, 30.0]);
}

#[test]
fn the_order_values_are_supplied_in_has_no_effect() {
    let plan = binary_plan(Operation::Sub, vec![2]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let left = [10.0f32, 20.0];
    let right = [1.0f32, 2.0];

    let mut forward = PlanExternalValues::new();
    forward
        .bind(LogicalBindingId::new(0), &left)
        .bind(LogicalBindingId::new(1), &right);

    let mut reversed = PlanExternalValues::new();
    reversed
        .bind(LogicalBindingId::new(1), &right)
        .bind(LogicalBindingId::new(0), &left);

    assert_eq!(
        runtime.execute(&prepared, &forward).unwrap(),
        runtime.execute(&prepared, &reversed).unwrap()
    );
}

#[test]
fn a_missing_value_is_rejected() {
    let plan = binary_plan(Operation::Add, vec![2]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let left = [1.0f32, 2.0];
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &left);

    assert_eq!(
        runtime.execute(&prepared, &values),
        Err(PlanExecutionError::MissingExternalValue {
            binding: LogicalBindingId::new(1)
        })
    );
}

#[test]
fn an_unexpected_value_is_rejected() {
    let plan = unary_plan(Operation::Relu, vec![2]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let data = [1.0f32, 2.0];
    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), &data)
        .bind(LogicalBindingId::new(9), &data);

    assert_eq!(
        runtime.execute(&prepared, &values),
        Err(PlanExecutionError::UnexpectedExternalValue {
            binding: LogicalBindingId::new(9)
        })
    );
}

#[test]
fn a_duplicate_value_is_rejected() {
    let plan = unary_plan(Operation::Relu, vec![2]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let data = [1.0f32, 2.0];
    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), &data)
        .bind(LogicalBindingId::new(0), &data);

    assert_eq!(
        runtime.execute(&prepared, &values),
        Err(PlanExecutionError::DuplicateExternalValue {
            binding: LogicalBindingId::new(0)
        })
    );
}

#[test]
fn a_value_of_the_wrong_length_is_rejected() {
    let plan = unary_plan(Operation::Relu, vec![3]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    for (data, actual) in [(vec![1.0f32, 2.0], 2usize), (vec![1.0, 2.0, 3.0, 4.0], 4)]
    {
        let mut values = PlanExternalValues::new();
        values.bind(LogicalBindingId::new(0), &data);

        assert_eq!(
            runtime.execute(&prepared, &values),
            Err(PlanExecutionError::ExternalValueLengthMismatch {
                binding: LogicalBindingId::new(0),
                expected: 3,
                actual,
            })
        );
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

#[test]
fn several_outputs_are_returned_in_plan_order() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let relu = graph
        .add_node(Operation::Relu, vec![input], ty.clone())
        .unwrap();
    let scaled = graph
        .add_node(
            Operation::Scale {
                factor: Scalar::f32(3.0),
            },
            vec![input],
            ty,
        )
        .unwrap();
    // Declared in reverse construction order, and an external input is an
    // output too.
    graph.set_outputs(vec![scaled, relu, input]).unwrap();
    let plan = lower(&graph);

    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let data = [-2.0f32, 4.0];
    let mut values = PlanExternalValues::new();
    values.bind(LogicalBindingId::new(0), &data);
    let outputs = runtime.execute(&prepared, &values).unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs.values().iter().map(|o| o.node).collect::<Vec<_>>(),
        vec![scaled, relu, input]
    );
    assert_eq!(outputs.values()[0].values, vec![-6.0, 12.0]); // scale
    assert_eq!(outputs.values()[1].values, vec![0.0, 4.0]); // relu
    assert_eq!(outputs.values()[2].values, vec![-2.0, 4.0]); // the input itself
}

#[test]
fn an_output_can_come_straight_from_a_consumed_constant() {
    let ty = f32_type(vec![2]);
    let mut graph = Graph::new();
    let input = graph.add_input("x", ty.clone()).unwrap();
    let constant = graph.add_constant(ConstantId::new(1), ty.clone()).unwrap();
    let sum = graph
        .add_node(Operation::Add, vec![input, constant], ty)
        .unwrap();
    graph.set_outputs(vec![sum, constant]).unwrap();
    let plan = lower(&graph);

    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    let input_values = [1.0f32, 2.0];
    let constant_values = [10.0f32, 20.0];
    let mut values = PlanExternalValues::new();
    values
        .bind(LogicalBindingId::new(0), &input_values)
        .bind(LogicalBindingId::new(1), &constant_values);

    let outputs = runtime.execute(&prepared, &values).unwrap();
    assert_eq!(outputs.values()[0].values, vec![11.0, 22.0]);
    assert_eq!(outputs.values()[1].values, vec![10.0, 20.0]);
}

#[test]
fn structural_opcodes_copy_nan_payloads_and_signed_zeros_bit_for_bit() {
    let payload_nan = f32::from_bits(0x7FC0_1234);

    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![2, 2])).unwrap();
    let transposed = graph
        .add_node(
            Operation::Transpose {
                permutation: vec![1, 0],
            },
            vec![input],
            f32_type(vec![2, 2]),
        )
        .unwrap();
    graph.set_outputs(vec![transposed]).unwrap();
    let plan = lower(&graph);

    // [[nan, -0.0], [inf, 1.0]] transposes to [[nan, inf], [-0.0, 1.0]].
    let result = run_unary(&plan, &[payload_nan, -0.0, f32::INFINITY, 1.0]);

    assert_eq!(
        bits(&result),
        vec![
            0x7FC0_1234,
            f32::INFINITY.to_bits(),
            (-0.0f32).to_bits(),
            1.0f32.to_bits(),
        ]
    );
}

#[test]
fn an_output_whose_type_cannot_be_determined_is_rejected() {
    // A lone input declared as the only output: it is never a kernel argument,
    // so no tensor type for it appears anywhere in the lowered plan.
    let mut graph = Graph::new();
    let input = graph.add_input("x", f32_type(vec![4])).unwrap();
    graph.set_outputs(vec![input]).unwrap();
    let plan = lower(&graph);

    assert_eq!(
        runtime().prepare(&plan).err(),
        Some(PlanPreparationError::UndeterminedValueType {
            node: NodeId::new(0)
        })
    );
}

// ---------------------------------------------------------------------------
// Exp and Log
// ---------------------------------------------------------------------------

#[test]
fn exp_and_log_are_rejected_by_the_runtime_itself() {
    for operation in [Operation::Exp, Operation::Log]
    {
        let label = format!("{operation:?}");
        let plan = unary_plan(operation, vec![4]);

        let error = runtime().prepare(&plan).expect_err("must be rejected");

        assert!(
            matches!(
                error,
                PlanPreparationError::DeterministicMathUnavailable { .. }
            ),
            "{label} must be rejected by the runtime, not left to the backend: got {error:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Transactionality
// ---------------------------------------------------------------------------

#[test]
fn a_failed_execution_returns_no_outputs_and_leaves_the_plan_reusable() {
    let plan = unary_plan(Operation::Relu, vec![3]);
    let runtime = runtime();
    let prepared = runtime.prepare(&plan).unwrap();

    // Fails during value validation, before the first dispatch.
    let wrong = [1.0f32];
    let mut bad_values = PlanExternalValues::new();
    bad_values.bind(LogicalBindingId::new(0), &wrong);
    assert!(runtime.execute(&prepared, &bad_values).is_err());

    // The same prepared plan still runs afterwards.
    let good = [-1.0f32, 2.0, -3.0];
    let mut good_values = PlanExternalValues::new();
    good_values.bind(LogicalBindingId::new(0), &good);

    let outputs = runtime.execute(&prepared, &good_values).unwrap();
    assert_eq!(outputs.values()[0].values, vec![0.0, 2.0, 0.0]);

    // And the caller's slices were never mutated.
    assert_eq!(good, [-1.0, 2.0, -3.0]);
}

// ---------------------------------------------------------------------------
// Backend contract
// ---------------------------------------------------------------------------

#[test]
fn the_prepared_plan_uses_the_canonical_launch_configuration() {
    let plan = unary_plan(Operation::Relu, vec![2]);
    let prepared = prepare(&plan);
    let config = prepared.launch_config();

    assert_eq!(config.grid, [1, 1, 1]);
    assert_eq!(config.block, [1, 1, 1]);
    assert_eq!(config.shared_memory_bytes, 0);
}

#[test]
fn prepared_metadata_describes_the_plan() {
    let plan = binary_plan(Operation::Mul, vec![2, 2]);
    let prepared = prepare(&plan);

    assert_eq!(prepared.external_values().len(), 2);
    assert_eq!(
        prepared.external_values()[0].tensor_type,
        f32_type(vec![2, 2])
    );
    assert_eq!(prepared.buffer_slots().len(), 1);
    assert_eq!(prepared.outputs().len(), 1);
    assert_eq!(prepared.outputs()[0].tensor_type, f32_type(vec![2, 2]));
}
