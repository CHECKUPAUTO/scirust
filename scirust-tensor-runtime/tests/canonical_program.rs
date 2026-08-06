//! End-to-end tests of the canonical façade on the real CPU backend.
//!
//! Every test drives the genuine pipeline — no mock, no hand-built graph:
//!
//! ```text
//! CanonicalProgram
//!   -> CanonicalSession (compile, lower, prepare)
//!   -> CanonicalInputs
//!   -> CpuComputeAdapter -> ReferenceInterpreter
//!   -> CanonicalOutputs
//! ```
//!
//! Nothing here names a `Graph`, a `NodeId`, a `TensorType`, a `DType`, a
//! `ConstantId`, a `LogicalBindingId`, a `GraphInputs`, a `GraphConstants` or a
//! `PlanOutputs` — which is the point of the layer under test.
//!
//! Bit-level assertions use `to_bits()` wherever the semantics demand it:
//! signed zeros and NaN payloads are indistinguishable under `==` (or compare
//! unequal, for NaN), so value equality would silently pass where the contract
//! requires an exact bit pattern.

use scirust_gpu::CpuComputeAdapter;
use scirust_tensor_core::TensorND;
use scirust_tensor_runtime::{
    CanonicalBuildError, CanonicalExecutionError, CanonicalInput, CanonicalInputs,
    CanonicalOutputs, CanonicalPreparationError, CanonicalProgram, CanonicalSession,
    ReferencePlanRuntime,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

type Session = CanonicalSession<CpuComputeAdapter>;

fn runtime() -> ReferencePlanRuntime<CpuComputeAdapter> {
    ReferencePlanRuntime::new(CpuComputeAdapter::new())
}

fn try_prepare(program: CanonicalProgram) -> Result<Session, CanonicalPreparationError> {
    program.prepare(runtime())
}

fn prepare(program: CanonicalProgram) -> Session {
    try_prepare(program).expect("preparable program")
}

fn tensor(values: &[f32], shape: &[usize]) -> TensorND {
    TensorND::new(values.to_vec(), shape.to_vec())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Builds one binary operation and declares it as the program's output.
type BuildBinary =
    fn(&mut CanonicalProgram, CanonicalInput, CanonicalInput) -> Result<(), CanonicalBuildError>;

fn run(session: &Session, bindings: &[(CanonicalInput, &TensorND)]) -> CanonicalOutputs {
    let mut inputs = CanonicalInputs::new();
    for &(input, value) in bindings
    {
        inputs.bind(input, value);
    }
    session.execute(&inputs).expect("successful run")
}

/// `x -> op -> output`, driven by a closure so each test names its own operation.
fn run_unary<F>(shape: &[usize], values: &[f32], build: F) -> TensorND
where
    F: FnOnce(&mut CanonicalProgram, CanonicalInput) -> Result<(), CanonicalBuildError>,
{
    let mut program = CanonicalProgram::new();
    let x = program.input("x", shape).expect("input");
    build(&mut program, x).expect("valid operation");

    let session = prepare(program);
    let value = tensor(values, shape);
    let mut outputs = run(&session, &[(x, &value)]).into_values();

    assert_eq!(outputs.len(), 1);
    outputs.remove(0)
}

// ---------------------------------------------------------------------------
// The eight exposed operations
// ---------------------------------------------------------------------------

#[test]
fn executes_relu() {
    let result = run_unary(&[4], &[-2.0, -0.5, 0.0, 3.0], |program, x| {
        let y = program.relu(x)?;
        program.set_outputs([y])
    });

    assert_eq!(result.data, vec![0.0, 0.0, 0.0, 3.0]);
    assert_eq!(result.shape, vec![4]);
}

#[test]
fn executes_scale() {
    let result = run_unary(&[3], &[1.0, -2.0, 0.5], |program, x| {
        let y = program.scale(x, -2.0)?;
        program.set_outputs([y])
    });

    assert_eq!(result.data, vec![-2.0, 4.0, -1.0]);
}

#[test]
fn executes_the_four_binary_operations() {
    let left = tensor(&[6.0, -4.0], &[2]);
    let right = tensor(&[2.0, 8.0], &[2]);

    let cases: [(&str, BuildBinary, Vec<f32>); 4] = [
        (
            "add",
            |program, a, b| {
                let y = program.add(a, b)?;
                program.set_outputs([y])
            },
            vec![8.0, 4.0],
        ),
        (
            "sub",
            |program, a, b| {
                let y = program.sub(a, b)?;
                program.set_outputs([y])
            },
            vec![4.0, -12.0],
        ),
        (
            "mul",
            |program, a, b| {
                let y = program.mul(a, b)?;
                program.set_outputs([y])
            },
            vec![12.0, -32.0],
        ),
        (
            "div",
            |program, a, b| {
                let y = program.div(a, b)?;
                program.set_outputs([y])
            },
            vec![3.0, -0.5],
        ),
    ];

    for (label, build, expected) in cases
    {
        let mut program = CanonicalProgram::new();
        let a = program.input("a", &[2]).expect("input");
        let b = program.input("b", &[2]).expect("input");
        build(&mut program, a, b).expect("valid operation");

        let session = prepare(program);
        let outputs = run(&session, &[(a, &left), (b, &right)]).into_values();
        assert_eq!(outputs[0].data, expected, "{label}");
    }
}

#[test]
fn executes_reshape() {
    let result = run_unary(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |program, x| {
        let y = program.reshape(x, &[6])?;
        program.set_outputs([y])
    });

    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(result.shape, vec![6]);
}

#[test]
fn executes_permute() {
    let result = run_unary(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |program, x| {
        let y = program.permute(x, &[1, 0])?;
        program.set_outputs([y])
    });

    assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(result.shape, vec![3, 2]);
}

// ---------------------------------------------------------------------------
// Multi-operation programs
// ---------------------------------------------------------------------------

#[test]
fn executes_scale_then_relu() {
    let result = run_unary(&[4], &[1.0, -2.0, 3.0, -4.0], |program, x| {
        let scaled = program.scale(x, -1.0)?;
        let activated = program.relu(scaled)?;
        program.set_outputs([activated])
    });

    assert_eq!(result.data, vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn executes_add_then_scale() {
    let mut program = CanonicalProgram::new();
    let a = program.input("a", &[3]).expect("input");
    let b = program.input("b", &[3]).expect("input");
    let sum = program.add(a, b).expect("add");
    let scaled = program.scale(sum, 0.5).expect("scale");
    program.set_outputs([scaled]).expect("outputs");

    let session = prepare(program);
    let left = tensor(&[1.0, 3.0, 5.0], &[3]);
    let right = tensor(&[1.0, 1.0, 1.0], &[3]);

    let outputs = run(&session, &[(a, &left), (b, &right)]).into_values();
    assert_eq!(outputs[0].data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn executes_permute_then_reshape() {
    let result = run_unary(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |program, x| {
        let transposed = program.permute(x, &[1, 0])?;
        let flattened = program.reshape(transposed, &[6])?;
        program.set_outputs([flattened])
    });

    assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(result.shape, vec![6]);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn a_constant_is_an_operand_of_a_unary_operation() {
    let mut program = CanonicalProgram::new();
    let c = program
        .constant(tensor(&[1.0, 2.0, -1.0], &[3]))
        .expect("constant");
    let scaled = program.scale(c, 3.0).expect("scale");
    program.set_outputs([scaled]).expect("outputs");

    let session = prepare(program);
    assert!(
        session.inputs().is_empty(),
        "nothing is asked of the caller"
    );

    let outputs = run(&session, &[]).into_values();
    assert_eq!(outputs[0].data, vec![3.0, 6.0, -3.0]);
}

#[test]
fn a_constant_is_an_operand_of_a_binary_operation() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[3]).expect("input");
    let weights = program
        .constant(tensor(&[2.0, 4.0, 8.0], &[3]))
        .expect("constant");
    let product = program.mul(x, weights).expect("mul");
    program.set_outputs([product]).expect("outputs");

    let session = prepare(program);
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.inputs()[0].name, "x");

    let value = tensor(&[1.0, 1.0, 1.0], &[3]);
    let outputs = run(&session, &[(x, &value)]).into_values();
    assert_eq!(outputs[0].data, vec![2.0, 4.0, 8.0]);
}

#[test]
fn a_scalar_constant_is_supported() {
    let mut program = CanonicalProgram::new();
    let c = program.constant(TensorND::scalar(-3.0)).expect("constant");
    let activated = program.relu(c).expect("relu");
    program.set_outputs([activated]).expect("outputs");

    let outputs = run(&prepare(program), &[]).into_values();
    assert_eq!(outputs[0].data, vec![0.0]);
    assert!(outputs[0].shape.is_empty(), "a scalar has no dimension");
}

#[test]
fn a_zero_element_constant_is_supported() {
    let mut program = CanonicalProgram::new();
    let c = program.constant(tensor(&[], &[0, 3])).expect("constant");
    let activated = program.relu(c).expect("relu");
    program.set_outputs([activated]).expect("outputs");

    let outputs = run(&prepare(program), &[]).into_values();
    assert!(outputs[0].data.is_empty());
    assert_eq!(outputs[0].shape, vec![0, 3]);
}

#[test]
fn a_constant_keeps_its_exact_bits_through_a_reshape() {
    let nan = f32::from_bits(0x7fc0_dead);
    let payload = [-0.0f32, nan, f32::INFINITY, f32::MIN_POSITIVE / 2.0];

    let mut program = CanonicalProgram::new();
    let c = program
        .constant(tensor(&payload, &[2, 2]))
        .expect("constant");
    let flattened = program.reshape(c, &[4]).expect("reshape");
    program.set_outputs([flattened]).expect("outputs");

    let outputs = run(&prepare(program), &[]).into_values();
    assert_eq!(bits(&outputs[0].data), bits(&payload));
    assert!(
        payload[3] != 0.0 && payload[3] < f32::MIN_POSITIVE,
        "the fourth value must be subnormal for this test to mean anything"
    );
}

#[test]
fn a_constant_keeps_its_exact_bits_through_a_permutation() {
    let nan = f32::from_bits(0xffc0_0001);
    let payload = [-0.0f32, nan, f32::NEG_INFINITY, 2.0];

    let mut program = CanonicalProgram::new();
    let c = program
        .constant(tensor(&payload, &[2, 2]))
        .expect("constant");
    let transposed = program.permute(c, &[1, 0]).expect("permute");
    program.set_outputs([transposed]).expect("outputs");

    let outputs = run(&prepare(program), &[]).into_values();
    assert_eq!(
        bits(&outputs[0].data),
        bits(&[-0.0, f32::NEG_INFINITY, nan, 2.0])
    );
}

#[test]
fn several_constants_receive_distinct_deterministic_identities() {
    let mut program = CanonicalProgram::new();
    let first = program
        .constant(tensor(&[1.0, 2.0], &[2]))
        .expect("constant");
    let second = program
        .constant(tensor(&[10.0, 20.0], &[2]))
        .expect("constant");
    let third = program
        .constant(tensor(&[100.0, 200.0], &[2]))
        .expect("constant");
    program
        .set_outputs([first, second, third])
        .expect("outputs");

    let outputs = run(&prepare(program), &[]).into_values();
    assert_eq!(outputs[0].data, vec![1.0, 2.0]);
    assert_eq!(outputs[1].data, vec![10.0, 20.0]);
    assert_eq!(outputs[2].data, vec![100.0, 200.0]);
}

#[test]
fn a_constant_eliminated_as_dead_is_neither_required_nor_executed() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let live = program.relu(x).expect("relu");
    let dead = program
        .constant(tensor(&[7.0, 9.0], &[2]))
        .expect("constant");
    program.scale(dead, 2.0).expect("dead scale");
    program.set_outputs([live]).expect("outputs");

    let session = prepare(program);
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs().len(), 1);

    let value = tensor(&[-3.0, 4.0], &[2]);
    let outputs = run(&session, &[(x, &value)]).into_values();
    assert_eq!(outputs[0].data, vec![0.0, 4.0]);
}

#[test]
fn a_non_dense_constant_is_rejected_at_build_time() {
    let mut broken = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    broken.strides[0] = 1;

    let mut program = CanonicalProgram::new();
    assert_eq!(
        program.constant(broken).err(),
        Some(CanonicalBuildError::NonContiguousTensor {
            shape: vec![2, 2],
            elements: 4,
        })
    );
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

#[test]
fn several_inputs_are_supported_in_any_binding_order() {
    let mut program = CanonicalProgram::new();
    let a = program.input("a", &[2]).expect("input");
    let b = program.input("b", &[2]).expect("input");
    let c = program.input("c", &[2]).expect("input");
    let sum = program.add(a, b).expect("add");
    let total = program.add(sum, c).expect("add");
    program.set_outputs([total]).expect("outputs");

    let session = prepare(program);
    let one = tensor(&[1.0, 1.0], &[2]);
    let two = tensor(&[2.0, 2.0], &[2]);
    let four = tensor(&[4.0, 4.0], &[2]);

    let forward = run(&session, &[(a, &one), (b, &two), (c, &four)]);
    let reversed = run(&session, &[(c, &four), (b, &two), (a, &one)]);

    assert_eq!(forward, reversed);
    assert_eq!(forward.values()[0].data, vec![7.0, 7.0]);
}

#[test]
fn a_missing_input_is_rejected() {
    let mut program = CanonicalProgram::new();
    let a = program.input("a", &[2]).expect("input");
    let b = program.input("b", &[2]).expect("input");
    let sum = program.add(a, b).expect("add");
    program.set_outputs([sum]).expect("outputs");

    let session = prepare(program);
    let value = tensor(&[1.0, 2.0], &[2]);

    let mut inputs = CanonicalInputs::new();
    inputs.bind(a, &value);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(CanonicalExecutionError::MissingInput {
            input: b,
            name: "b".to_string(),
        })
    );
}

#[test]
fn a_duplicated_input_is_rejected() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let y = program.relu(x).expect("relu");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);
    let value = tensor(&[1.0, 2.0], &[2]);

    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &value).bind(x, &value);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(CanonicalExecutionError::DuplicateInput { input: x })
    );
}

#[test]
fn an_input_of_the_wrong_shape_is_rejected_even_at_the_right_element_count() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[6]).expect("input");
    let y = program.relu(x).expect("relu");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);

    // Same six values, different shape: the layer below would only compare
    // element counts, so this check belongs to the façade.
    let reshaped = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &reshaped);
    assert_eq!(
        session.execute(&inputs).err(),
        Some(CanonicalExecutionError::InputShapeMismatch {
            input: x,
            name: "x".to_string(),
            expected: vec![6],
            actual: vec![2, 3],
        })
    );

    let short = tensor(&[1.0, 2.0], &[2]);
    let mut too_short = CanonicalInputs::new();
    too_short.bind(x, &short);
    assert_eq!(
        session.execute(&too_short).err(),
        Some(CanonicalExecutionError::InputShapeMismatch {
            input: x,
            name: "x".to_string(),
            expected: vec![6],
            actual: vec![2],
        })
    );
}

#[test]
fn a_non_dense_input_is_rejected() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2, 2]).expect("input");
    let y = program.relu(x).expect("relu");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);
    let mut broken = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    broken.strides[0] = 1;

    let mut inputs = CanonicalInputs::new();
    inputs.bind(x, &broken);

    assert_eq!(
        session.execute(&inputs).err(),
        Some(CanonicalExecutionError::NonContiguousInput {
            input: x,
            name: "x".to_string(),
            shape: vec![2, 2],
            elements: 4,
        })
    );
}

#[test]
fn an_input_eliminated_as_dead_is_neither_required_nor_accepted() {
    let mut program = CanonicalProgram::new();
    let live = program.input("live", &[2]).expect("input");
    let dead = program.input("dead", &[2]).expect("input");
    let kept = program.relu(live).expect("relu");
    program.relu(dead).expect("dead relu");
    program.set_outputs([kept]).expect("outputs");

    let session = prepare(program);
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.inputs()[0].input, live);

    let value = tensor(&[-1.0, 2.0], &[2]);

    let mut with_dead = CanonicalInputs::new();
    with_dead.bind(live, &value).bind(dead, &value);
    assert_eq!(
        session.execute(&with_dead).err(),
        Some(CanonicalExecutionError::UnexpectedInput { input: dead })
    );

    let outputs = run(&session, &[(live, &value)]).into_values();
    assert_eq!(outputs[0].data, vec![0.0, 2.0]);
}

#[test]
fn an_input_keeps_its_exact_bits_through_a_structural_copy() {
    let nan = f32::from_bits(0x7fc0_1234);
    let payload = [-0.0f32, nan, f32::INFINITY, f32::MIN_POSITIVE / 4.0];

    let result = run_unary(&[2, 2], &payload, |program, x| {
        let flattened = program.reshape(x, &[4])?;
        program.set_outputs([flattened])
    });

    assert_eq!(bits(&result.data), bits(&payload));
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

#[test]
fn several_outputs_are_returned_in_the_declared_order() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let activated = program.relu(x).expect("relu");
    let scaled = program.scale(x, 10.0).expect("scale");
    // Deliberately not the construction order.
    program.set_outputs([scaled, activated]).expect("outputs");

    let session = prepare(program);
    assert_eq!(session.outputs().len(), 2);
    assert_eq!(session.outputs()[0].value, scaled);
    assert_eq!(session.outputs()[1].value, activated);

    let value = tensor(&[-1.0, 2.0], &[2]);
    let outputs = run(&session, &[(x, &value)]).into_values();
    assert_eq!(outputs[0].data, vec![-10.0, 20.0]);
    assert_eq!(outputs[1].data, vec![0.0, 2.0]);
}

#[test]
fn a_duplicated_output_is_preserved_as_is() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let activated = program.relu(x).expect("relu");
    program
        .set_outputs([activated, activated])
        .expect("outputs");

    let session = prepare(program);
    assert_eq!(session.outputs().len(), 2);

    let value = tensor(&[-4.0, 5.0], &[2]);
    let outputs = run(&session, &[(x, &value)]).into_values();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].data, vec![0.0, 5.0]);
    assert_eq!(outputs[1].data, vec![0.0, 5.0]);
}

#[test]
fn an_output_can_be_an_input_directly() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[4]).expect("input");
    program.set_outputs([x.value()]).expect("outputs");

    let session = prepare(program);
    assert_eq!(session.inputs().len(), 1);
    assert_eq!(session.outputs()[0].shape, vec![4]);

    let nan = f32::from_bits(0x7fc0_0abc);
    let payload = tensor(&[-0.0, nan, f32::INFINITY, 1.5], &[4]);
    let outputs = run(&session, &[(x, &payload)]).into_values();

    assert_eq!(bits(&outputs[0].data), bits(&payload.data));
}

#[test]
fn an_output_can_be_a_constant_directly() {
    let nan = f32::from_bits(0xffc0_5678);
    let payload = [-0.0f32, nan, f32::NEG_INFINITY];

    let mut program = CanonicalProgram::new();
    let c = program.constant(tensor(&payload, &[3])).expect("constant");
    program.set_outputs([c]).expect("outputs");

    let session = prepare(program);
    assert!(session.inputs().is_empty());

    let outputs = run(&session, &[]).into_values();
    assert_eq!(bits(&outputs[0].data), bits(&payload));
}

#[test]
fn a_program_with_no_operation_at_all_still_runs() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2, 2]).expect("input");
    let c = program
        .constant(tensor(&[9.0, 9.0], &[2]))
        .expect("constant");
    program.set_outputs([x.value(), c]).expect("outputs");

    let session = prepare(program);
    let value = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let outputs = run(&session, &[(x, &value)]).into_values();

    assert_eq!(outputs[0].data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(outputs[0].shape, vec![2, 2]);
    assert_eq!(outputs[1].data, vec![9.0, 9.0]);
}

#[test]
fn a_scalar_input_round_trips() {
    let result = run_unary(&[], &[-4.0], |program, x| {
        let y = program.relu(x)?;
        program.set_outputs([y])
    });

    assert_eq!(result.data, vec![0.0]);
    assert!(result.shape.is_empty());
}

#[test]
fn a_zero_dimension_input_round_trips() {
    let result = run_unary(&[0, 3], &[], |program, x| {
        let y = program.relu(x)?;
        program.set_outputs([y])
    });

    assert!(result.data.is_empty());
    assert_eq!(result.shape, vec![0, 3]);
}

#[test]
fn a_program_without_an_output_cannot_be_prepared() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    program.relu(x).expect("relu");

    assert_eq!(
        try_prepare(program).err(),
        Some(CanonicalPreparationError::NoOutputs)
    );
}

// ---------------------------------------------------------------------------
// Reuse and determinism
// ---------------------------------------------------------------------------

#[test]
fn one_session_serves_several_executions_with_different_values() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[3]).expect("input");
    let y = program.relu(x).expect("relu");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);

    for (values, expected) in [
        ([-1.0f32, 2.0, -3.0], vec![0.0f32, 2.0, 0.0]),
        ([4.0, -5.0, 6.0], vec![4.0, 0.0, 6.0]),
        ([0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
    ]
    {
        let value = tensor(&values, &[3]);
        let outputs = run(&session, &[(x, &value)]).into_values();
        assert_eq!(outputs[0].data, expected);
    }
}

#[test]
fn identical_inputs_reproduce_identical_bits() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[4]).expect("input");
    let y = program.scale(x, 0.1).expect("scale");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);
    let value = tensor(&[1.0, -0.0, 3.3, -7.7], &[4]);

    let first = run(&session, &[(x, &value)]).into_values();
    let second = run(&session, &[(x, &value)]).into_values();

    assert_eq!(bits(&first[0].data), bits(&second[0].data));
}

#[test]
fn a_session_stays_usable_after_a_rejected_input() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let y = program.relu(x).expect("relu");
    program.set_outputs([y]).expect("outputs");

    let session = prepare(program);

    let wrong = tensor(&[1.0], &[1]);
    let mut rejected = CanonicalInputs::new();
    rejected.bind(x, &wrong);
    assert!(session.execute(&rejected).is_err());

    let value = tensor(&[-1.0, 2.0], &[2]);
    let outputs = run(&session, &[(x, &value)]).into_values();
    assert_eq!(outputs[0].data, vec![0.0, 2.0]);
}

#[test]
fn constants_stay_identical_across_executions() {
    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");
    let bias = program
        .constant(tensor(&[100.0, 200.0], &[2]))
        .expect("constant");
    let sum = program.add(x, bias).expect("add");
    program.set_outputs([sum]).expect("outputs");

    let session = prepare(program);

    let first = tensor(&[1.0, 2.0], &[2]);
    let second = tensor(&[3.0, 4.0], &[2]);

    assert_eq!(
        run(&session, &[(x, &first)]).into_values()[0].data,
        vec![101.0, 202.0]
    );
    assert_eq!(
        run(&session, &[(x, &second)]).into_values()[0].data,
        vec![103.0, 204.0]
    );
}

#[test]
fn preparing_the_same_program_twice_yields_the_same_public_metadata() {
    let describe = || {
        let mut program = CanonicalProgram::new();
        let x = program.input("x", &[2]).expect("input");
        let c = program
            .constant(tensor(&[1.0, 1.0], &[2]))
            .expect("constant");
        let sum = program.add(x, c).expect("add");
        program.set_outputs([sum, sum]).expect("outputs");

        let session = prepare(program);
        (session.inputs().to_vec(), session.outputs().to_vec())
    };

    assert_eq!(describe(), describe());
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

/// `exp`, `log` and `matmul` have no method on `CanonicalProgram`, so a program
/// cannot even name them: the layers below reject them, and this façade does not
/// advertise them.
///
/// The absence is enforced by the compiler, not by this test — the test only
/// records the contract and checks the operations that *are* offered.
#[test]
fn the_facade_offers_exactly_the_eight_executable_operations() {
    let mut program = CanonicalProgram::new();
    let a = program.input("a", &[2]).expect("input");
    let b = program.input("b", &[2]).expect("input");

    program.add(a, b).expect("add");
    program.sub(a, b).expect("sub");
    program.mul(a, b).expect("mul");
    program.div(a, b).expect("div");
    program.relu(a).expect("relu");
    program.scale(a, 2.0).expect("scale");
    let flat = program.reshape(a, &[1, 2]).expect("reshape");
    program.permute(flat, &[1, 0]).expect("permute");

    program.set_outputs([a.value()]).expect("outputs");
    let session = prepare(program);
    assert_eq!(session.outputs().len(), 1);
}

#[test]
fn a_duplicate_input_name_is_rejected() {
    let mut program = CanonicalProgram::new();
    program.input("weights", &[2]).expect("input");

    assert_eq!(
        program.input("weights", &[3]).err(),
        Some(CanonicalBuildError::DuplicateInputName {
            name: "weights".to_string(),
        })
    );
}

#[test]
fn a_handle_from_another_program_is_rejected() {
    let mut donor = CanonicalProgram::new();
    donor.input("a", &[2]).expect("input");
    donor.input("b", &[2]).expect("input");
    let stranger = donor.input("c", &[2]).expect("input");

    let mut program = CanonicalProgram::new();
    let x = program.input("x", &[2]).expect("input");

    assert_eq!(
        program.add(x, stranger).err(),
        Some(CanonicalBuildError::ForeignValue {
            value: stranger.value(),
        })
    );
}
