//! Integration tests for the V2 IR: verification rules, interpreter oracles,
//! non-finite policy, recurrence execution and multi-output programs.

use super::compat::from_v1;
use super::interpret::{
    ExecutionError, ExecutionPolicy, FloatPolicy, TensorDataError, ValueTensor, execute_program,
};
use super::ir::{
    AxisOp, Bin, Narrow, Op, Permute, Reduce, Ref, ResearchProgram, Section, ShapeTo, Ter, Un,
};
use super::semantics::NumericalSemantics;
use super::types::{DType, ScalarValue, ValueType};
use super::verify::{ProgramError, SectionKind, VerificationLimits, verify_program};

fn f32_type(shape: &[usize]) -> ValueType {
    ValueType::new(DType::F32, shape.to_vec())
}

fn f64_type(shape: &[usize]) -> ValueType {
    ValueType::new(DType::F64, shape.to_vec())
}

fn tensor_f32(data: &[f32], shape: &[usize]) -> ValueTensor {
    ValueTensor::new(
        DType::F32,
        shape.to_vec(),
        data.iter().map(|&value| value as f64).collect(),
    )
    .unwrap()
}

fn tensor_f64(data: &[f64], shape: &[usize]) -> ValueTensor {
    ValueTensor::new(DType::F64, shape.to_vec(), data.to_vec()).unwrap()
}

fn default_limits() -> VerificationLimits {
    VerificationLimits::default()
}

// ---------------------------------------------------------------------------
// Verification: positive cases
// ---------------------------------------------------------------------------

#[test]
fn verifies_a_broadcast_arithmetic_expression() {
    // out = x * [1.0] where the scalar constant broadcasts over x.
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![
            Op::Const(ScalarValue::F32(1.0)),
            Op::Mul(Bin::new(Ref::Input(0), Ref::Local(0))),
        ]),
        vec![1],
    );
    let verified = verify_program(&program, default_limits()).unwrap();
    assert_eq!(verified.output_types, vec![f32_type(&[2, 3])]);
    assert_eq!(verified.finalize_active, vec![true, true]);
}

/// Online-softmax state update over scalar items (m = running max,
/// l = running normaliser), used by several tests below.
fn online_softmax_step() -> (Section, Vec<usize>) {
    let section = Section::new(vec![
        Op::Max(Bin::new(Ref::StatePrev(0), Ref::Item(0))), // 0: m'
        Op::Sub(Bin::new(Ref::StatePrev(0), Ref::Local(0))), // 1: m_prev - m'
        Op::Exp(Un::new(Ref::Local(1))),                    // 2: alpha
        Op::Sub(Bin::new(Ref::Item(0), Ref::Local(0))),     // 3: x - m'
        Op::Exp(Un::new(Ref::Local(3))),                    // 4: e
        Op::Mul(Bin::new(Ref::StatePrev(1), Ref::Local(2))), // 5: l_prev * alpha
        Op::Add(Bin::new(Ref::Local(5), Ref::Local(4))),    // 6: l'
    ]);
    (section, vec![0, 6])
}

#[test]
fn verifies_scalar_recurrence_with_two_state_components() {
    let (step, next_state) = online_softmax_step();
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64); 2],
        steps: 3,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(f64::NEG_INFINITY)),
            Op::Const(ScalarValue::F64(0.0)),
        ]),
        init_state: vec![0, 1],
        step,
        next_state,
        finalize: Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(0),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(1),
                shape: vec![],
            }),
        ]),
        outputs: vec![0, 1],
    };
    let verified = verify_program(&program, default_limits()).unwrap();
    assert_eq!(verified.output_types.len(), 2);
    assert_eq!(verified.step_active, vec![true; 7]);
}

// ---------------------------------------------------------------------------
// Verification: negative cases (one per rule family)
// ---------------------------------------------------------------------------

fn expect_error(program: &ResearchProgram, limits: VerificationLimits) -> ProgramError {
    verify_program(program, limits).expect_err("program must fail verification")
}

#[test]
fn rejects_use_before_definition() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Local(1)))]),
        vec![0],
    );
    assert_eq!(
        expect_error(&program, default_limits()),
        ProgramError::NonCausalDependency {
            section: SectionKind::Finalize,
            node: 0,
            source: 1,
        }
    );
}

#[test]
fn rejects_item_reference_outside_step_section() {
    let program = ResearchProgram::expression(
        vec![],
        Section::new(vec![Op::Abs(Un::new(Ref::Item(0)))]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::RefIllegalInSection { .. }
    ));
}

#[test]
fn rejects_input_reference_in_step_section() {
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![ValueType::scalar(DType::F64)],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64)],
        steps: 1,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        step: Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        outputs: vec![0],
    };
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::RefIllegalInSection {
            section: SectionKind::Step,
            ..
        }
    ));
}

#[test]
fn rejects_state_final_reference_in_init_section() {
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F32)],
        state: vec![ValueType::scalar(DType::F32)],
        steps: 1,
        init: Section::new(vec![Op::Abs(Un::new(Ref::StateFinal(0)))]),
        init_state: vec![0],
        step: Section::new(vec![Op::Abs(Un::new(Ref::Item(0)))]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F32(0.0))]),
        outputs: vec![0],
    };
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::RefIllegalInSection {
            section: SectionKind::Init,
            ..
        }
    ));
}

#[test]
fn rejects_input_out_of_bounds() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(3)))]),
        vec![0],
    );
    assert_eq!(
        expect_error(&program, default_limits()),
        ProgramError::InputOutOfBounds {
            section: SectionKind::Finalize,
            node: 0,
            input: 3,
            available: 1,
        }
    );
}

#[test]
fn rejects_item_out_of_bounds() {
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64)],
        steps: 1,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        step: Section::new(vec![Op::Abs(Un::new(Ref::Item(5)))]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        outputs: vec![0],
    };
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::ItemOutOfBounds {
            item: 5,
            available: 1,
            ..
        }
    ));
}

#[test]
fn rejects_nan_constant_but_allows_infinity_identities() {
    for value in [ScalarValue::F32(f32::NAN), ScalarValue::F64(f64::NAN)]
    {
        let program =
            ResearchProgram::expression(vec![], Section::new(vec![Op::Const(value)]), vec![0]);
        assert!(matches!(
            expect_error(&program, default_limits()),
            ProgramError::NonNanConstant { .. }
        ));
    }

    // ±Infinity constants are admissible (running-max initialisers).
    for value in [
        ScalarValue::F64(f64::NEG_INFINITY),
        ScalarValue::F64(f64::INFINITY),
        ScalarValue::F32(f32::NEG_INFINITY),
    ]
    {
        let program =
            ResearchProgram::expression(vec![], Section::new(vec![Op::Const(value)]), vec![0]);
        assert!(verify_program(&program, default_limits()).is_ok());
    }
}

#[test]
fn rejects_dtype_mismatch_in_add() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2]), f64_type(&[2])],
        Section::new(vec![Op::Add(Bin::new(Ref::Input(0), Ref::Input(1)))]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::DTypeMismatch { .. }
    ));
}

#[test]
fn rejects_broadcast_incompatible_add() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3]), f32_type(&[3, 2])],
        Section::new(vec![Op::Add(Bin::new(Ref::Input(0), Ref::Input(1)))]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::BroadcastIncompatible { .. }
    ));
}

#[test]
fn rejects_bool_operand_to_float_unary() {
    let program = ResearchProgram::expression(
        vec![ValueType::new(DType::Bool, vec![4])],
        Section::new(vec![Op::Sqrt(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::DTypeMismatch {
            expected: DType::F64,
            found: DType::Bool,
            ..
        }
    ));
}

#[test]
fn rejects_select_branch_type_mismatch() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![
            Op::Lt(Bin::new(Ref::Input(0), Ref::Input(0))),
            Op::Const(ScalarValue::Bool(true)),
            Op::Select(Ter::new(Ref::Local(1), Ref::Local(0), Ref::Input(0))),
        ]),
        vec![2],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::DTypeMismatch { op: "select", .. }
    ));
}

#[test]
fn rejects_select_mask_not_broadcastable() {
    // A reshaped [6] mask cannot broadcast onto the [2, 3] branches.
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![
            Op::Lt(Bin::new(Ref::Input(0), Ref::Input(0))),
            Op::Reshape(ShapeTo {
                src: Ref::Local(0),
                shape: vec![6],
            }),
            Op::Select(Ter::new(Ref::Local(1), Ref::Input(0), Ref::Input(0))),
        ]),
        vec![2],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::MaskNotBroadcastable { .. }
    ));
}

#[test]
fn rejects_reduction_axis_out_of_range() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![Op::ReduceSum(Reduce {
            src: Ref::Input(0),
            axis: Some(2),
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::ReductionAxisOutOfRange {
            axis: 2,
            rank: 2,
            ..
        }
    ));
}

#[test]
fn rejects_mean_and_max_over_empty() {
    for op in [
        Op::ReduceMean(Reduce {
            src: Ref::Input(0),
            axis: None,
        }),
        Op::ReduceMax(Reduce {
            src: Ref::Input(0),
            axis: Some(0),
        }),
    ]
    .into_iter()
    {
        let program =
            ResearchProgram::expression(vec![f32_type(&[0])], Section::new(vec![op]), vec![0]);
        assert!(matches!(
            expect_error(&program, default_limits()),
            ProgramError::ReductionOverEmptyForbidden { .. }
        ));
    }

    // Sum over empty is defined and allowed.
    let program = ResearchProgram::expression(
        vec![f32_type(&[0])],
        Section::new(vec![Op::ReduceSum(Reduce {
            src: Ref::Input(0),
            axis: None,
        })]),
        vec![0],
    );
    assert!(verify_program(&program, default_limits()).is_ok());
}

#[test]
fn rejects_reshape_with_wrong_element_count() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![Op::Reshape(ShapeTo {
            src: Ref::Input(0),
            shape: vec![4],
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::ReshapeElementMismatch {
            source_elements: 6,
            target_elements: 4,
            ..
        }
    ));
}

#[test]
fn rejects_squeeze_of_non_unit_axis() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![Op::Squeeze(AxisOp {
            src: Ref::Input(0),
            axis: 1,
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::SqueezeAxisNotOne { dimension: 3, .. }
    ));
}

#[test]
fn rejects_invalid_transpose_permutation() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![Op::Transpose(Permute {
            src: Ref::Input(0),
            perm: vec![0, 0],
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::TransposePermutationInvalid { .. }
    ));
}

#[test]
fn rejects_narrow_beyond_dimension() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[4])],
        Section::new(vec![Op::Narrow(Narrow {
            src: Ref::Input(0),
            axis: 0,
            start: 2,
            len: 3,
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::NarrowRangeInvalid { .. }
    ));
}

#[test]
fn rejects_concat_off_axis_mismatch() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3]), f32_type(&[2, 4])],
        Section::new(vec![super::ir::Op::Concat {
            lhs: Ref::Input(0),
            rhs: Ref::Input(1),
            axis: 0,
        }]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::ConcatAxisShapeMismatch { .. }
    ));
}

#[test]
fn rejects_steps_without_state_and_state_without_steps() {
    let mut stepped = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    stepped.steps = 4;
    assert_eq!(
        expect_error(&stepped, default_limits()),
        ProgramError::StepsWithoutState
    );

    let mut stateful = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    stateful.state = vec![ValueType::scalar(DType::F32)];
    stateful.init_state = vec![0];
    stateful.next_state = vec![0];
    assert_eq!(
        expect_error(&stateful, default_limits()),
        ProgramError::StateWithoutSteps { components: 1 }
    );

    let mut sections = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    sections.init = Section::new(vec![Op::Const(ScalarValue::F32(0.0))]);
    assert!(matches!(
        expect_error(&sections, default_limits()),
        ProgramError::RecurrenceSectionsWithoutState { .. }
    ));
}

#[test]
fn rejects_init_or_next_state_count_mismatch() {
    let build = || ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64); 2],
        steps: 1,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        step: Section::new(vec![Op::Exp(Un::new(Ref::Item(0)))]),
        next_state: vec![0, 0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        outputs: vec![0],
    };
    assert_eq!(
        expect_error(&build(), default_limits()),
        ProgramError::InitStateCountMismatch {
            expected: 2,
            found: 1,
        }
    );

    let mut swapped = build();
    swapped.init_state = vec![0, 0];
    swapped.next_state = vec![0];
    assert_eq!(
        expect_error(&swapped, default_limits()),
        ProgramError::NextStateCountMismatch {
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn rejects_state_binding_type_mismatch() {
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64)],
        steps: 1,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        // Step produces a vector while the declared state is scalar.
        step: Section::new(vec![Op::BroadcastTo(ShapeTo {
            src: Ref::Item(0),
            shape: vec![4],
        })]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        outputs: vec![0],
    };
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::ShapeMismatchExact {
            op: "state_update",
            ..
        }
    ));
}

#[test]
fn rejects_duplicate_outputs() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[1])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0, 0],
    );
    assert_eq!(
        expect_error(&program, default_limits()),
        ProgramError::OutputDuplicate { output: 0 }
    );
}

#[test]
fn rejects_no_outputs() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[1])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![],
    );
    assert_eq!(
        expect_error(&program, default_limits()),
        ProgramError::NoOutputs
    );
}

#[test]
fn enforces_rank_and_node_budgets() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 2])],
        Section::new(vec![Op::BroadcastTo(ShapeTo {
            src: Ref::Input(0),
            shape: vec![2, 2, 2, 2, 2, 2, 2, 2, 2],
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&program, default_limits()),
        ProgramError::RankLimitExceeded {
            rank: 9,
            maximum: 8,
            ..
        }
    ));

    let limits = VerificationLimits {
        max_nodes_per_section: 2,
        ..VerificationLimits::default()
    };
    let wide = ResearchProgram::expression(
        vec![f32_type(&[2])],
        Section::new(vec![
            Op::Const(ScalarValue::F32(0.0)),
            Op::Const(ScalarValue::F32(1.0)),
            Op::Sqrt(Un::new(Ref::Input(0))),
        ]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&wide, limits),
        ProgramError::TooManyNodesInSection {
            nodes: 3,
            maximum: 2,
            ..
        }
    ));

    let tight_total = VerificationLimits {
        max_nodes_total: 2,
        ..VerificationLimits::default()
    };
    assert!(matches!(
        expect_error(&wide, tight_total),
        ProgramError::TooManyNodesTotal {
            nodes: 3,
            maximum: 2
        }
    ));
}

#[test]
fn enforces_tensor_and_total_element_budgets() {
    let limits = VerificationLimits {
        max_elements_per_tensor: 100,
        max_total_register_elements: 150,
        ..VerificationLimits::default()
    };
    let program = ResearchProgram::expression(
        vec![f32_type(&[10, 10])],
        Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::Input(0),
                shape: vec![10, 10],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::Input(0),
                shape: vec![10, 10],
            }),
        ]),
        vec![1],
    );
    assert!(matches!(
        expect_error(&program, limits),
        ProgramError::TotalRegisterElementsExceeded {
            elements: 200,
            maximum: 150
        }
    ));

    let big = VerificationLimits {
        max_elements_per_tensor: 50,
        ..VerificationLimits::default()
    };
    let single = ResearchProgram::expression(
        vec![f32_type(&[10, 10])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&single, big),
        ProgramError::SignatureTensorTooLarge {
            elements: 100,
            maximum: 50,
            ..
        }
    ));

    let produced = ResearchProgram::expression(
        vec![ValueType::scalar(DType::F32)],
        Section::new(vec![Op::BroadcastTo(ShapeTo {
            src: Ref::Input(0),
            shape: vec![100],
        })]),
        vec![0],
    );
    assert!(matches!(
        expect_error(&produced, big),
        ProgramError::TensorTooLarge {
            elements: 100,
            maximum: 50,
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// Interpreter: exact oracles
// ---------------------------------------------------------------------------

#[test]
fn interprets_affine_expression_with_exact_oracle() {
    // out = (x * 3) - y computed entirely in f32.
    let program = ResearchProgram::expression(
        vec![f32_type(&[3]), f32_type(&[3])],
        Section::new(vec![
            Op::Const(ScalarValue::F32(3.0)),
            Op::Mul(Bin::new(Ref::Input(0), Ref::Local(0))),
            Op::Sub(Bin::new(Ref::Local(1), Ref::Input(1))),
        ]),
        vec![2],
    );
    let x = [1.0f32, -2.0, 3.5];
    let y = [0.5f32, 0.25, -0.75];
    let expected: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(&a, b)| ((a * 3.0f32) - b) as f64)
        .collect();
    let result = execute_program(
        &program,
        &[tensor_f32(&x, &[3]), tensor_f32(&y, &[3])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].data, expected);
}

#[test]
fn interprets_reduction_mean_over_axis() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[2, 3])],
        Section::new(vec![Op::ReduceMean(Reduce {
            src: Ref::Input(0),
            axis: Some(1),
        })]),
        vec![0],
    );
    let input = [1.0f64, 2.0, 3.0, 4.0, 6.0, 8.0];
    let result = execute_program(
        &program,
        &[tensor_f64(&input, &[2, 3])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].shape, vec![2]);
    assert_eq!(result.outputs[0].data, vec![2.0, 6.0]);
}

#[test]
fn interprets_masked_select_with_boolean_logic() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[4])],
        Section::new(vec![
            Op::Const(ScalarValue::F32(0.0)),               // 0
            Op::Gt(Bin::new(Ref::Input(0), Ref::Local(0))), // 1: mask
            Op::Not(Un::new(Ref::Local(1))),                // 2
            Op::Const(ScalarValue::F32(1.0)),               // 3
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(3),
                shape: vec![4],
            }), // 4: +ones
            Op::Const(ScalarValue::F32(-1.0)),              // 5
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(5),
                shape: vec![4],
            }), // 6: -ones
            Op::Select(Ter::new(Ref::Local(1), Ref::Local(4), Ref::Local(6))), // 7
            Op::And(Bin::new(Ref::Local(1), Ref::Local(2))), // 8
            Op::Select(Ter::new(Ref::Local(8), Ref::Local(4), Ref::Local(6))), // 9
        ]),
        vec![7, 9],
    );
    let input = [-2.0f32, 5.0, 0.0, 3.0];
    let result = execute_program(
        &program,
        &[tensor_f32(&input, &[4])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data, vec![-1.0, 1.0, -1.0, 1.0]);
    // mask AND NOT(mask) is always false -> -1 everywhere.
    assert_eq!(result.outputs[1].data, vec![-1.0; 4]);
}

#[test]
fn executes_online_softmax_scan_with_exact_oracle() {
    let (step, next_state) = online_softmax_step();
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64); 2],
        steps: 4,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(f64::NEG_INFINITY)), // m0
            Op::Const(ScalarValue::F64(0.0)),               // l0
        ]),
        init_state: vec![0, 1],
        step,
        next_state,
        finalize: Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(0),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(1),
                shape: vec![],
            }),
            Op::Div(Bin::new(Ref::Local(1), Ref::Local(1))),
        ]),
        outputs: vec![0, 1, 2],
    };

    let items: Vec<ValueTensor> = [1.0f64, 2.0, 3.0, 1.5]
        .iter()
        .map(|&value| tensor_f64(&[value], &[]))
        .collect();
    let result = execute_program(
        &program,
        &[],
        &items,
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();

    // Independent oracle.
    let mut oracle_m = f64::NEG_INFINITY;
    let mut oracle_l = 0.0;
    for &x in &[1.0f64, 2.0, 3.0, 1.5]
    {
        let m_new = oracle_m.max(x);
        oracle_l = oracle_l * (oracle_m - m_new).exp() + (x - m_new).exp();
        oracle_m = m_new;
    }
    assert_eq!(result.outputs[0].data, vec![oracle_m]);
    assert!((result.outputs[1].data[0] - oracle_l).abs() < 1e-12);
    // Self-ratio output is exactly 1.
    assert_eq!(result.outputs[2].data, vec![1.0]);
    // Executed nodes: init 2 + 7 live step nodes x4 + finalize 3.
    assert_eq!(result.executed_nodes, 2 + 28 + 3);
}

#[test]
fn dead_sections_are_not_executed() {
    // A dead Exp that would overflow must never run under liveness.
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64)],
        steps: 2,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        step: Section::new(vec![
            Op::Exp(Un::new(Ref::Item(0))), // dead on huge items
            Op::Reshape(ShapeTo {
                src: Ref::StatePrev(0),
                shape: vec![],
            }), // live identity
        ]),
        next_state: vec![1],
        finalize: Section::new(vec![Op::Reshape(ShapeTo {
            src: Ref::StateFinal(0),
            shape: vec![],
        })]),
        outputs: vec![0],
    };
    let items = [tensor_f64(&[1e308], &[]), tensor_f64(&[1e308], &[])];
    let result = execute_program(
        &program,
        &[],
        &items,
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data, vec![0.0]);
    assert_eq!(result.executed_nodes, 1 + 2 + 1);
}

#[test]
fn executes_linear_algebra_operators() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3]), f32_type(&[3]), f32_type(&[2])],
        Section::new(vec![
            Op::MatVec(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Dot(Bin::new(Ref::Input(1), Ref::Input(1))),
            Op::Outer(Bin::new(Ref::Input(1), Ref::Input(1))),
            Op::VecMat(Bin::new(Ref::Input(2), Ref::Input(0))),
        ]),
        vec![0, 1, 2, 3],
    );
    let matrix = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector = [1.0f32, -1.0, 2.0];
    let short = [1.0f32, -1.0];
    let result = execute_program(
        &program,
        &[
            tensor_f32(&matrix, &[2, 3]),
            tensor_f32(&vector, &[3]),
            tensor_f32(&short, &[2]),
        ],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();

    // matvec: [1*1+2*(-1)+3*2, 4*1+5*(-1)+6*2] = [5, 11]
    assert_eq!(result.outputs[0].data, vec![5.0, 11.0]);
    // dot(v, v) = 1 + 1 + 4 = 6
    assert_eq!(result.outputs[1].data, vec![6.0]);
    assert_eq!(result.outputs[2].shape, vec![3, 3]);
    assert_eq!(result.outputs[2].data[0], 1.0);
    assert_eq!(result.outputs[2].data[3], -1.0);
    // vecmat: [1,-1] x [[1,2,3],[4,5,6]] = [-3, -3, -3]
    assert_eq!(result.outputs[3].data, vec![-3.0, -3.0, -3.0]);
}

#[test]
fn executes_batched_matmul() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[2, 1, 2]), f64_type(&[2, 2, 1])],
        Section::new(vec![Op::BatchedMatMul(Bin::new(
            Ref::Input(0),
            Ref::Input(1),
        ))]),
        vec![0],
    );
    // Batch b: [[1,2]] x [[3],[4]] = [11]; batch 1: [[5,6]] x [[7],[8]] = [83].
    let lhs = [1.0, 2.0, 5.0, 6.0];
    let rhs = [3.0, 4.0, 7.0, 8.0];
    let result = execute_program(
        &program,
        &[tensor_f64(&lhs, &[2, 1, 2]), tensor_f64(&rhs, &[2, 2, 1])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].shape, vec![2, 1, 1]);
    assert_eq!(result.outputs[0].data, vec![11.0, 83.0]);
}

#[test]
fn executes_shape_algebra() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 3])],
        Section::new(vec![
            Op::Transpose(Permute {
                src: Ref::Input(0),
                perm: vec![1, 0],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::Local(0),
                shape: vec![6],
            }),
            Op::Narrow(Narrow {
                src: Ref::Local(1),
                axis: 0,
                start: 2,
                len: 3,
            }),
            Op::Concat {
                lhs: Ref::Local(2),
                rhs: Ref::Local(2),
                axis: 0,
            },
            Op::Unsqueeze(AxisOp {
                src: Ref::Local(3),
                axis: 0,
            }),
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(4),
                shape: vec![2, 6],
            }),
            Op::ReduceSum(Reduce {
                src: Ref::Local(5),
                axis: None,
            }),
        ]),
        vec![1, 2, 3],
    );
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = execute_program(
        &program,
        &[tensor_f32(&input, &[2, 3])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    // Transpose of [1..6] row-major is [1,4,2,5,3,6].
    assert_eq!(result.outputs[0].data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    // Narrow keeps elements 2..5 = [2,5,3].
    assert_eq!(result.outputs[1].data, vec![2.0, 5.0, 3.0]);
    // Concat with itself doubles it.
    assert_eq!(result.outputs[2].data, vec![2.0, 5.0, 3.0, 2.0, 5.0, 3.0]);
}

#[test]
fn interprets_clamp_pow_and_muladd() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[])],
        Section::new(vec![
            Op::Const(ScalarValue::F64(0.0)),  // 0
            Op::Const(ScalarValue::F64(10.0)), // 1
            Op::Clamp(Ter::new(Ref::Input(0), Ref::Local(0), Ref::Local(1))), // 2
            Op::Pow(Bin::new(Ref::Local(2), Ref::Local(1))), // 3
            Op::MulAdd(Ter::new(Ref::Local(2), Ref::Local(1), Ref::Local(2))), // 4
            Op::Log1p(Un::new(Ref::Local(3))), // 5
            Op::Rsqrt(Un::new(Ref::Local(4))), // 6
        ]),
        vec![3, 5, 6],
    );
    let result = execute_program(
        &program,
        &[tensor_f64(&[2.0], &[])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data, vec![1024.0]); // pow(clamp(2), 10)
    assert_eq!(result.outputs[1].data, vec![(1024.0f64).ln_1p()]);
    assert_eq!(result.outputs[2].data, vec![1.0 / (22.0f64).sqrt()]); // rsqrt(mul_add)
}

#[test]
fn transcendental_oracle_matches_native_semantics() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[3])],
        Section::new(vec![
            Op::Exp(Un::new(Ref::Input(0))),
            Op::Log(Un::new(Ref::Input(0))),
            Op::Tanh(Un::new(Ref::Input(0))),
            Op::Sin(Un::new(Ref::Input(0))),
        ]),
        vec![0, 1, 2, 3],
    );
    let input = [0.25f64, 1.5, 2.0];
    let result = execute_program(
        &program,
        &[tensor_f64(&input, &[3])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    for (index, oracle) in [
        input.iter().copied().map(f64::exp).collect::<Vec<_>>(),
        input.iter().copied().map(f64::ln).collect::<Vec<_>>(),
        input.iter().copied().map(f64::tanh).collect::<Vec<_>>(),
        input.iter().copied().map(f64::sin).collect::<Vec<_>>(),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(result.outputs[index].data, oracle, "output {index}");
    }
}

// ---------------------------------------------------------------------------
// Non-finite policy
// ---------------------------------------------------------------------------

#[test]
fn reject_policy_catches_overflowing_intermediate() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[1])],
        Section::new(vec![Op::Exp(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    // exp(1000) overflows to +Infinity which flows through IEEE ops but the
    // observable output must remain finite under the default regime.
    let error = execute_program(
        &program,
        &[tensor_f64(&[1e3], &[1])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        ExecutionError::NonFiniteOutput {
            output: 0,
            element: 0,
        }
    );

    let allowed = execute_program(
        &program,
        &[tensor_f64(&[1e3], &[1])],
        &[],
        ExecutionPolicy {
            floats: FloatPolicy::AllowNonFinite,
        },
        default_limits(),
    )
    .unwrap();
    assert_eq!(allowed.outputs[0].data, vec![f64::INFINITY]);
}

#[test]
fn log_of_negative_is_rejected_under_default_policy() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[1])],
        Section::new(vec![Op::Log(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let error = execute_program(
        &program,
        &[tensor_f64(&[-1.0], &[1])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ExecutionError::NanResult { element: 0, .. }
    ));
}

#[test]
fn division_by_zero_is_rejected_under_default_policy() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[1])],
        Section::new(vec![
            Op::Const(ScalarValue::F64(0.0)),
            Op::Div(Bin::new(Ref::Input(0), Ref::Local(0))),
        ]),
        vec![1],
    );
    let error = execute_program(
        &program,
        &[tensor_f64(&[1.0], &[1])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        ExecutionError::NonFiniteOutput {
            output: 0,
            element: 0,
        }
    );
}

#[test]
fn signed_zero_flows_through_subtraction() {
    // x - x == +0 for finite x (IEEE round-to-nearest), never NaN.
    let program = ResearchProgram::expression(
        vec![f64_type(&[2])],
        Section::new(vec![Op::Sub(Bin::new(Ref::Input(0), Ref::Input(0)))]),
        vec![0],
    );
    let result = execute_program(
        &program,
        &[tensor_f64(&[0.0, -3.0], &[2])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(result.outputs[0].data[1].to_bits(), 0.0f64.to_bits());
}

#[test]
fn rejects_wrong_arity_and_types() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[1])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    assert_eq!(
        execute_program(
            &program,
            &[],
            &[],
            ExecutionPolicy::default(),
            default_limits()
        )
        .unwrap_err(),
        ExecutionError::InputArity {
            expected: 1,
            found: 0,
        }
    );
    assert!(matches!(
        execute_program(
            &program,
            &[tensor_f32(&[1.0], &[1])],
            &[],
            ExecutionPolicy::default(),
            default_limits()
        )
        .unwrap_err(),
        ExecutionError::InputTypeMismatch { .. }
    ));

    // Item arity: two steps, one item slot each => exactly 2 item tensors.
    let scan = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![ValueType::scalar(DType::F64)],
        state: vec![ValueType::scalar(DType::F64)],
        steps: 2,
        init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        init_state: vec![0],
        step: Section::new(vec![Op::Abs(Un::new(Ref::Item(0)))]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
        outputs: vec![0],
    };
    assert_eq!(
        execute_program(
            &scan,
            &[],
            &[],
            ExecutionPolicy::default(),
            default_limits()
        )
        .unwrap_err(),
        ExecutionError::ItemArity {
            expected: 2,
            found: 0,
        }
    );
}

// ---------------------------------------------------------------------------
// Determinism and compat
// ---------------------------------------------------------------------------

#[test]
fn repeated_execution_is_bit_identical() {
    let program = ResearchProgram::expression(
        vec![f32_type(&[2, 2])],
        Section::new(vec![
            Op::Tanh(Un::new(Ref::Input(0))),
            Op::ReduceSum(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
        ]),
        vec![1],
    );
    let input = [0.3f32, -1.7, 2.2, 0.9];
    let first = execute_program(
        &program,
        &[tensor_f32(&input, &[2, 2])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    let second = execute_program(
        &program,
        &[tensor_f32(&input, &[2, 2])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(first.outputs, second.outputs);
    assert_eq!(first.executed_nodes, second.executed_nodes);
    for (a, b) in first.outputs[0].data.iter().zip(&second.outputs[0].data)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn v1_programs_lift_to_bit_identical_v2_execution() {
    let v1 = crate::tensor::ir::TensorProgram::new(
        vec![
            crate::tensor::ir::TensorInstruction::Input { input: 0 },
            crate::tensor::ir::TensorInstruction::Scale {
                src: 0,
                factor: -1.5,
            },
            crate::tensor::ir::TensorInstruction::Relu { src: 1 },
            crate::tensor::ir::TensorInstruction::Transpose2d { src: 2 },
            crate::tensor::ir::TensorInstruction::MatMul { lhs: 3, rhs: 0 },
        ],
        4,
    );
    let shapes = [vec![2, 3]];
    let lifted = from_v1(&v1, &shapes);

    let input =
        scirust_tensor_core::TensorND::new(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0], vec![2, 3]);
    let v1_result = crate::tensor::execute_program(
        &v1,
        std::slice::from_ref(&input),
        crate::tensor::VerificationLimits::default(),
    )
    .unwrap();

    let v2_result = execute_program(
        &lifted,
        &[ValueTensor::from_f32(&input)],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();

    assert_eq!(v1_result.output.shape, v2_result.outputs[0].shape);
    for (a, b) in v1_result.output.data.iter().zip(&v2_result.outputs[0].data)
    {
        assert_eq!(a.to_bits(), (*b as f32).to_bits());
    }
}

// ---------------------------------------------------------------------------
// Hardened trust-boundary and resource-budget regressions
// ---------------------------------------------------------------------------

#[test]
fn init_section_locals_are_legal_and_type_checked() {
    let scalar = ValueType::scalar(DType::F64);
    let program = ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![],
        state: vec![scalar],
        steps: 1,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(-2.0)),
            Op::Abs(Un::new(Ref::Local(0))),
        ]),
        init_state: vec![1],
        step: Section::new(vec![Op::Abs(Un::new(Ref::StatePrev(0)))]),
        next_state: vec![0],
        finalize: Section::new(vec![Op::Abs(Un::new(Ref::StateFinal(0)))]),
        outputs: vec![0],
    };
    let verified = verify_program(&program, default_limits()).unwrap();
    assert_eq!(verified.init_types[1], ValueType::scalar(DType::F64));
}

#[test]
fn signature_count_rank_and_resident_byte_budgets_are_enforced() {
    let two_inputs = ResearchProgram::expression(
        vec![f64_type(&[]), f64_type(&[])],
        Section::new(vec![Op::Add(Bin::new(Ref::Input(0), Ref::Input(1)))]),
        vec![0],
    );
    let limits = VerificationLimits {
        max_inputs: 1,
        ..default_limits()
    };
    assert!(matches!(
        verify_program(&two_inputs, limits),
        Err(ProgramError::TooManyInputs { .. })
    ));

    let excessive_rank = ResearchProgram::expression(
        vec![f64_type(&[1; 9])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    assert!(matches!(
        verify_program(&excessive_rank, default_limits()),
        Err(ProgramError::SignatureRankLimitExceeded { .. })
    ));

    let resident = ResearchProgram::expression(
        vec![f64_type(&[4])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let limits = VerificationLimits {
        max_temporary_bytes: 95,
        ..default_limits()
    };
    assert_eq!(
        verify_program(&resident, limits),
        Err(ProgramError::TemporaryBytesExceeded {
            bytes: 96,
            maximum: 95,
        })
    );
}

#[test]
fn zero_element_shapes_cannot_hide_stride_overflow() {
    let shape = vec![0, usize::MAX, usize::MAX];
    let program = ResearchProgram::expression(
        vec![ValueType::new(DType::F64, shape.clone())],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    assert_eq!(
        verify_program(&program, default_limits()),
        Err(ProgramError::SignatureStrideOverflow {
            kind: super::verify::SignatureKind::Input,
            index: 0,
            shape,
        })
    );
}

#[test]
fn complete_materialized_item_sequence_is_bounded() {
    let program = super::reference::online_softmax_recurrence(3);
    let limits = VerificationLimits {
        max_stream_input_elements: 2,
        ..default_limits()
    };
    assert_eq!(
        verify_program(&program, limits),
        Err(ProgramError::StreamInputElementsExceeded {
            elements: 3,
            maximum: 2,
        })
    );
}

#[test]
fn finite_only_rejects_infinite_constants_at_verification() {
    let program = ResearchProgram::expression(
        vec![],
        Section::new(vec![Op::Const(ScalarValue::F64(f64::NEG_INFINITY))]),
        vec![0],
    )
    .with_semantics(NumericalSemantics::FiniteOnly);
    assert_eq!(
        verify_program(&program, default_limits()),
        Err(ProgramError::NonFiniteConstantInFiniteOnly {
            section: SectionKind::Finalize,
            node: 0,
        })
    );
}

#[test]
fn externally_constructed_tensor_payloads_cannot_bypass_layout_validation() {
    let bool_program = ResearchProgram::expression(
        vec![ValueType::scalar(DType::Bool)],
        Section::new(vec![Op::Not(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let invalid_bool = ValueTensor {
        dtype: DType::Bool,
        shape: vec![],
        data: vec![2.0],
    };
    assert!(matches!(
        execute_program(
            &bool_program,
            &[invalid_bool],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        ),
        Err(ExecutionError::InvalidInputLayout { input: 0, .. })
    ));

    let vector_program = ResearchProgram::expression(
        vec![f64_type(&[2])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let wrong_length = ValueTensor {
        dtype: DType::F64,
        shape: vec![2],
        data: vec![1.0],
    };
    assert!(matches!(
        execute_program(
            &vector_program,
            &[wrong_length],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        ),
        Err(ExecutionError::InvalidInputLayout { input: 0, .. })
    ));

    let f32_program = ResearchProgram::expression(
        vec![f32_type(&[])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let not_binary32 = ValueTensor {
        dtype: DType::F32,
        shape: vec![],
        data: vec![0.1f64],
    };
    assert!(matches!(
        execute_program(
            &f32_program,
            &[not_binary32],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        ),
        Err(ExecutionError::InvalidInputLayout { input: 0, .. })
    ));
}

#[test]
fn reject_non_finite_policy_stops_at_the_first_overflowing_intermediate() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[])],
        Section::new(vec![
            Op::Exp(Un::new(Ref::Input(0))),
            Op::Abs(Un::new(Ref::Local(0))),
        ]),
        vec![1],
    );
    assert_eq!(
        execute_program(
            &program,
            &[ValueTensor::scalar_f64(1000.0)],
            &[],
            ExecutionPolicy {
                floats: FloatPolicy::RejectNonFinite,
            },
            default_limits(),
        ),
        Err(ExecutionError::NonFiniteResult {
            section: SectionKind::Finalize,
            node: 0,
            element: 0,
        })
    );
}

#[test]
fn fused_multiply_add_is_observably_distinct_from_mul_then_add() {
    let a = 1.000_000_000_000_000_2f64;
    let b = 1.000_000_000_000_000_2f64;
    let c = -1.000_000_000_000_000_4f64;
    let program = ResearchProgram::expression(
        vec![],
        Section::new(vec![
            Op::Const(ScalarValue::F64(a)),
            Op::Const(ScalarValue::F64(b)),
            Op::Const(ScalarValue::F64(c)),
            Op::MulAdd(Ter::new(Ref::Local(0), Ref::Local(1), Ref::Local(2))),
            Op::Mul(Bin::new(Ref::Local(0), Ref::Local(1))),
            Op::Add(Bin::new(Ref::Local(4), Ref::Local(2))),
        ]),
        vec![3, 5],
    );
    let result = execute_program(
        &program,
        &[],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(
        result.outputs[0].data[0].to_bits(),
        a.mul_add(b, c).to_bits()
    );
    assert_eq!(result.outputs[1].data[0].to_bits(), (a * b + c).to_bits());
    assert_ne!(
        result.outputs[0].data[0].to_bits(),
        result.outputs[1].data[0].to_bits()
    );
}

#[test]
fn min_max_follow_the_documented_deterministic_extrema_contract() {
    // Normative contract pinned with INDEPENDENT literal bit expectations
    // (not by mirroring the implementation helpers): negative zero wins min,
    // positive zero wins max, a lone NaN defers to the numeric operand and
    // two NaNs collapse onto the canonical quiet NaN 0x7ff8_0000_0000_0000.
    const QNAN: u64 = 0x7ff8_0000_0000_0000;
    let program = ResearchProgram::expression(
        vec![f64_type(&[]), f64_type(&[])],
        Section::new(vec![
            Op::Min(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Max(Bin::new(Ref::Input(0), Ref::Input(1))),
        ]),
        vec![0, 1],
    );
    for (left, right, expected_min, expected_max) in [
        (0.0f64, -0.0f64, (-0.0f64).to_bits(), 0.0f64.to_bits()),
        (-0.0f64, 0.0f64, (-0.0f64).to_bits(), 0.0f64.to_bits()),
        (1.5, -2.5, (-2.5f64).to_bits(), 1.5f64.to_bits()),
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY.to_bits(),
            f64::INFINITY.to_bits(),
        ),
        (f64::NAN, 3.0, 3.0f64.to_bits(), 3.0f64.to_bits()),
        (3.0, f64::NAN, 3.0f64.to_bits(), 3.0f64.to_bits()),
        (f64::NAN, f64::NAN, QNAN, QNAN),
    ]
    {
        for (a, b) in [(left, right), (right, left)]
        {
            let result = execute_program(
                &program,
                &[ValueTensor::scalar_f64(a), ValueTensor::scalar_f64(b)],
                &[],
                ExecutionPolicy {
                    floats: FloatPolicy::AllowNonFinite,
                },
                default_limits(),
            )
            .unwrap();
            assert_eq!(
                result.outputs[0].data[0].to_bits(),
                expected_min,
                "min({a:?},{b:?})"
            );
            assert_eq!(
                result.outputs[1].data[0].to_bits(),
                expected_max,
                "max({a:?},{b:?})"
            );
        }
    }
    // Swap-symmetry: operand order must never be observable.
    for (left, right) in [
        (f64::NAN, 1.0),
        (0.0, -0.0),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::NAN, f64::NAN),
    ]
    {
        let forward = execute_program(
            &program,
            &[
                ValueTensor::scalar_f64(left),
                ValueTensor::scalar_f64(right),
            ],
            &[],
            ExecutionPolicy {
                floats: FloatPolicy::AllowNonFinite,
            },
            default_limits(),
        )
        .unwrap();
        let swapped = execute_program(
            &program,
            &[
                ValueTensor::scalar_f64(right),
                ValueTensor::scalar_f64(left),
            ],
            &[],
            ExecutionPolicy {
                floats: FloatPolicy::AllowNonFinite,
            },
            default_limits(),
        )
        .unwrap();
        assert_eq!(
            forward.outputs[0].data[0].to_bits(),
            swapped.outputs[0].data[0].to_bits()
        );
        assert_eq!(
            forward.outputs[1].data[0].to_bits(),
            swapped.outputs[1].data[0].to_bits()
        );
    }
}

#[test]
fn empty_sum_and_product_use_explicit_identities() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[0])],
        Section::new(vec![
            Op::ReduceSum(Reduce {
                src: Ref::Input(0),
                axis: None,
            }),
            Op::ReduceProd(Reduce {
                src: Ref::Input(0),
                axis: None,
            }),
        ]),
        vec![0, 1],
    );
    let result = execute_program(
        &program,
        &[tensor_f64(&[], &[0])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(result.outputs[1].data, vec![1.0]);
}

#[test]
fn subnormal_values_are_not_normalized_by_the_ir() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(subnormal.is_subnormal());
    let program = ResearchProgram::expression(
        vec![f64_type(&[])],
        Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
        vec![0],
    );
    let result = execute_program(
        &program,
        &[ValueTensor::scalar_f64(-subnormal)],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(result.outputs[0].data[0].to_bits(), subnormal.to_bits());
}

// ---------------------------------------------------------------------------
// Adversarial numerical-semantics coverage (signed zero / extrema / folding)
// ---------------------------------------------------------------------------

/// Comparisons keep native IEEE semantics: NaN makes every ordered comparison
/// and equality false and inequality true; `+0 == -0`.
#[test]
fn float_comparisons_follow_ieee_on_nan_and_signed_zero() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[]), f64_type(&[])],
        Section::new(vec![
            Op::Eq(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Ne(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Lt(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Ge(Bin::new(Ref::Input(0), Ref::Input(1))),
        ]),
        vec![0, 1, 2, 3],
    );
    for (left, right) in [
        (f64::NAN, 1.0),
        (0.0, -0.0),
        (f64::INFINITY, f64::NEG_INFINITY),
    ]
    {
        let result = execute_program(
            &program,
            &[
                ValueTensor::scalar_f64(left),
                ValueTensor::scalar_f64(right),
            ],
            &[],
            ExecutionPolicy {
                floats: FloatPolicy::AllowNonFinite,
            },
            default_limits(),
        )
        .unwrap();
        assert_eq!(result.outputs[0].data[0] != 0.0, left == right);
        assert_eq!(result.outputs[1].data[0] != 0.0, left != right);
        assert_eq!(result.outputs[2].data[0] != 0.0, left < right);
        assert_eq!(result.outputs[3].data[0] != 0.0, left >= right);
    }
}

/// Elementwise extrema resolve opposite-signed zeros identically in f32 and
/// f64, in both operand orders, including through `Clamp` bounds.
#[test]
fn signed_zero_extrema_are_order_independent_in_both_dtypes() {
    let build = |dtype: DType| {
        ResearchProgram::expression(
            vec![ValueType::scalar(dtype), ValueType::scalar(dtype)],
            Section::new(vec![
                Op::Min(Bin::new(Ref::Input(0), Ref::Input(1))),
                Op::Max(Bin::new(Ref::Input(0), Ref::Input(1))),
                Op::Const(match dtype
                {
                    DType::F32 => ScalarValue::F32(-5.0),
                    _ => ScalarValue::F64(-5.0),
                }),
                Op::Const(match dtype
                {
                    DType::F32 => ScalarValue::F32(0.0),
                    _ => ScalarValue::F64(0.0),
                }),
                Op::Clamp(Ter::new(Ref::Input(0), Ref::Local(2), Ref::Input(1))),
            ]),
            vec![0, 1, 4],
        )
    };
    for dtype in [DType::F32, DType::F64]
    {
        let scalar = |value: f64| ValueTensor {
            dtype,
            shape: vec![],
            data: vec![value],
        };
        for (left, right) in [(0.0f64, -0.0f64), (-0.0f64, 0.0f64)]
        {
            // validate_layout requires exact binary32 payloads for F32.
            if dtype == DType::F32
            {
                assert!((left as f32) as f64 == left && (right as f32) as f64 == right);
            }
            let result = execute_program(
                &build(dtype),
                &[scalar(left), scalar(right)],
                &[],
                ExecutionPolicy::default(),
                default_limits(),
            )
            .unwrap();
            // Literal oracles: min resolves to -0, max to +0, and
            // clamp(+0|-0, lo=-5, hi=∓0) inherits the -0 tie-break.
            assert_eq!(
                result.outputs[0].data[0].to_bits(),
                (-0.0f64).to_bits(),
                "{dtype:?} min({left},{right})"
            );
            assert_eq!(
                result.outputs[1].data[0].to_bits(),
                0.0f64.to_bits(),
                "{dtype:?} max({left},{right})"
            );
            assert_eq!(
                result.outputs[2].data[0].to_bits(),
                (-0.0f64).to_bits(),
                "{dtype:?} clamp"
            );
        }
    }
}

/// Constant folding of extrema must reproduce interpreter bits exactly,
/// including the `-0` outcome of `min(+0,-0)` and its `+0` maximum.
#[test]
fn folded_extrema_constants_match_interpreted_extrema_bit_for_bit() {
    let build = |min_first: bool| {
        ResearchProgram::expression(
            vec![],
            Section::new(vec![
                Op::Const(ScalarValue::F64(if min_first { 0.0 } else { -0.0 })),
                Op::Const(ScalarValue::F64(if min_first { -0.0 } else { 0.0 })),
                Op::Min(Bin::new(Ref::Local(0), Ref::Local(1))),
                Op::Max(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2, 3],
        )
    };
    for program in [build(true), build(false)]
    {
        let executed = execute_program(
            &program,
            &[],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        )
        .unwrap();
        let canonical =
            super::simplify::canonicalize(&program, VerificationLimits::default()).unwrap();
        // Both Min and Max fold into plain constants.
        assert!(canonical.stats.constants_folded >= 2);
        let refolded = execute_program(
            &canonical.program,
            &[],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        )
        .unwrap();
        assert_eq!(
            executed.outputs[0].data[0].to_bits(),
            refolded.outputs[0].data[0].to_bits()
        );
        assert_eq!(
            executed.outputs[1].data[0].to_bits(),
            refolded.outputs[1].data[0].to_bits()
        );
        assert_eq!(refolded.outputs[0].data[0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(refolded.outputs[1].data[0].to_bits(), 0.0f64.to_bits());
        // Folding is idempotent on identity: digest stable after a second run.
        let twice =
            super::simplify::canonicalize(&canonical.program, VerificationLimits::default())
                .unwrap();
        assert_eq!(
            super::program_digest(&canonical.program),
            super::program_digest(&twice.program)
        );
    }
}

/// Canonicalization may normalize `Min`/`Max` operands even under
/// `StrictIeee` because the kernels are swap-symmetric by contract; it must
/// NOT normalize `Add` there, and must do so under `FiniteOnly`.
#[test]
fn commutative_normalization_respects_regime_validity_domains() {
    let build = |swap: bool, op: fn(Bin) -> Op| {
        ResearchProgram::expression(
            vec![f64_type(&[4]), f64_type(&[4])],
            Section::new(vec![
                Op::Tanh(Un::new(Ref::Input(0))),
                Op::Exp(Un::new(Ref::Input(1))),
                op(
                    if swap
                    {
                        Bin::new(Ref::Local(1), Ref::Local(0))
                    }
                    else
                    {
                        Bin::new(Ref::Local(0), Ref::Local(1))
                    },
                ),
            ]),
            vec![2],
        )
    };
    // Min: both operand orders converge onto one canonical form under every
    // regime, and execution over adversarial inputs stays bit-identical.
    for semantics in [
        NumericalSemantics::StrictIeee,
        NumericalSemantics::FiniteOnly,
        NumericalSemantics::RealAlgebraicExperimental,
    ]
    {
        let left = build(false, Op::Min).with_semantics(semantics);
        let right = build(true, Op::Min).with_semantics(semantics);
        let canonical_left =
            super::simplify::canonicalize(&left, VerificationLimits::default()).unwrap();
        let canonical_right =
            super::simplify::canonicalize(&right, VerificationLimits::default()).unwrap();
        assert!(
            super::canonical_equal(&canonical_left.program, &canonical_right.program),
            "{semantics:?}: Min operands must normalize"
        );

        // Differential execution across zeros/infinities under the permissive
        // research policy: the ±Inf rows would be rejected as outputs
        // otherwise, which is the default policy's job, not this test's.
        let policy = ExecutionPolicy {
            floats: FloatPolicy::AllowNonFinite,
        };
        let inputs = || {
            vec![
                tensor_f64(&[0.0, -0.0, f64::INFINITY, 3.5], &[4]),
                tensor_f64(&[-0.0, 0.0, -1.0, f64::NEG_INFINITY], &[4]),
            ]
        };
        let a = execute_program(
            &canonical_left.program,
            &inputs(),
            &[],
            policy,
            default_limits(),
        )
        .unwrap();
        let b = execute_program(
            &canonical_right.program,
            &inputs(),
            &[],
            policy,
            default_limits(),
        )
        .unwrap();
        assert_eq!(
            a.outputs[0].data[0].to_bits(),
            b.outputs[0].data[0].to_bits()
        );
    }

    // Add: normalization only where finite rewrites are admitted.
    let strict_pair = (
        build(false, Op::Add).with_semantics(NumericalSemantics::StrictIeee),
        build(true, Op::Add).with_semantics(NumericalSemantics::StrictIeee),
    );
    let strict_left =
        super::simplify::canonicalize(&strict_pair.0, VerificationLimits::default()).unwrap();
    let strict_right =
        super::simplify::canonicalize(&strict_pair.1, VerificationLimits::default()).unwrap();
    assert!(
        !super::canonical_equal(&strict_left.program, &strict_right.program),
        "StrictIeee must not reorder floating Add operands"
    );
    let finite_pair = (
        build(false, Op::Add).with_semantics(NumericalSemantics::FiniteOnly),
        build(true, Op::Add).with_semantics(NumericalSemantics::FiniteOnly),
    );
    let finite_left =
        super::simplify::canonicalize(&finite_pair.0, VerificationLimits::default()).unwrap();
    let finite_right =
        super::simplify::canonicalize(&finite_pair.1, VerificationLimits::default()).unwrap();
    assert!(super::canonical_equal(
        &finite_left.program,
        &finite_right.program
    ));
}

/// Reduce-max/min are encounter-order independent for signed zeros and defer
/// to numeric elements when NaN appears (permissive policy).
#[test]
fn reduction_extrema_are_canonical_over_zeros_and_nan() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[2])],
        Section::new(vec![
            Op::ReduceMax(Reduce {
                src: Ref::Input(0),
                axis: None,
            }),
            Op::ReduceMin(Reduce {
                src: Ref::Input(0),
                axis: None,
            }),
        ]),
        vec![0, 1],
    );
    // Opposite orders of the same multiset must agree bit-for-bit.
    for values in [
        ([0.0f64, -0.0f64], [-0.0f64, 0.0f64]),
        ([5.0, 5.0], [5.0, 5.0]),
    ]
    {
        let forward = execute_program(
            &program,
            &[tensor_f64(&values.0, &[2])],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        )
        .unwrap();
        let backward = execute_program(
            &program,
            &[tensor_f64(&values.1, &[2])],
            &[],
            ExecutionPolicy::default(),
            default_limits(),
        )
        .unwrap();
        assert_eq!(
            forward.outputs[0].data[0].to_bits(),
            backward.outputs[0].data[0].to_bits()
        );
        assert_eq!(
            forward.outputs[1].data[0].to_bits(),
            backward.outputs[1].data[0].to_bits()
        );
    }
    // Canonical outcomes themselves.
    let zeros = execute_program(
        &program,
        &[tensor_f64(&[0.0, -0.0], &[2])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    )
    .unwrap();
    assert_eq!(zeros.outputs[0].data[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(zeros.outputs[1].data[0].to_bits(), (-0.0f64).to_bits());

    // NaN defers to numeric operands; a reduction whose elements are all NaN
    // evaluates to the canonical quiet NaN — never to a synthetic ±Infinity
    // identity leaking into a non-empty domain.
    let mixed = execute_program(
        &program,
        &[tensor_f64(&[5.0, f64::NAN], &[2])],
        &[],
        ExecutionPolicy {
            floats: FloatPolicy::AllowNonFinite,
        },
        default_limits(),
    )
    .unwrap();
    assert_eq!(mixed.outputs[0].data[0], 5.0);
    let all_nan = execute_program(
        &program,
        &[tensor_f64(&[f64::NAN, f64::NAN], &[2])],
        &[],
        ExecutionPolicy {
            floats: FloatPolicy::AllowNonFinite,
        },
        default_limits(),
    )
    .unwrap();
    assert_eq!(
        all_nan.outputs[0].data[0].to_bits(),
        0x7ff8_0000_0000_0000u64
    );
    // Under the default policy the same program aborts at the producer node.
    let rejected = execute_program(
        &program,
        &[tensor_f64(&[f64::NAN, f64::NAN], &[2])],
        &[],
        ExecutionPolicy::default(),
        default_limits(),
    );
    assert!(
        matches!(
            rejected,
            Err(ExecutionError::NonFiniteInput { input: 0, .. })
        ),
        "NaN external inputs stay rejected before execution"
    );
}

/// Full adversarial extrema-reduction matrix over f32 and f64 under
/// `AllowNonFinite`, with independent literal bit expectations.
///
/// Contract: extrema reductions are statically non-empty and seed with the
/// canonical quiet NaN, so a lone numeric element is returned exactly, a NaN
/// defers to any numeric element, and an all-NaN domain evaluates to the
/// canonical quiet NaN (0x7fc0_0000 / 0x7ff8_0000_0000_0000) instead of
/// leaking a synthetic ±Infinity identity.
#[test]
fn extrema_reduction_nan_matrix_matches_literal_contract() {
    const QNAN_F64: u64 = 0x7ff8_0000_0000_0000;
    const QNAN_F32: u32 = 0x7fc0_0000;
    let build = |dtype: DType| {
        ResearchProgram::expression(
            vec![ValueType::new(dtype, vec![3])],
            Section::new(vec![
                Op::ReduceMax(Reduce {
                    src: Ref::Input(0),
                    axis: None,
                }),
                Op::ReduceMin(Reduce {
                    src: Ref::Input(0),
                    axis: None,
                }),
            ]),
            vec![0, 1],
        )
    };
    // (multiset, expected max, expected min) — expectations hold for every
    // permutation of the multiset (order independence).
    let cases: &[(&[f64], u64, u64)] = &[
        (&[f64::NAN], QNAN_F64, QNAN_F64),
        (&[f64::NAN, f64::NAN], QNAN_F64, QNAN_F64),
        (&[f64::NAN, 5.0], 5.0f64.to_bits(), 5.0f64.to_bits()),
        (&[5.0, f64::NAN], 5.0f64.to_bits(), 5.0f64.to_bits()),
        (
            &[f64::NAN, 5.0, f64::NAN],
            5.0f64.to_bits(),
            5.0f64.to_bits(),
        ),
        (
            &[7.0, -3.0, f64::NAN],
            7.0f64.to_bits(),
            (-3.0f64).to_bits(),
        ),
        (&[0.0, -0.0], 0.0f64.to_bits(), (-0.0f64).to_bits()),
        (&[-0.0, 0.0], 0.0f64.to_bits(), (-0.0f64).to_bits()),
        (
            &[f64::INFINITY],
            f64::INFINITY.to_bits(),
            f64::INFINITY.to_bits(),
        ),
        (
            &[f64::NEG_INFINITY],
            f64::NEG_INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
        ),
        (
            &[f64::INFINITY, f64::NAN],
            f64::INFINITY.to_bits(),
            f64::INFINITY.to_bits(),
        ),
        (
            &[f64::NEG_INFINITY, f64::NAN],
            f64::NEG_INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
        ),
    ];
    for dtype in [DType::F32, DType::F64]
    {
        for (values, expected_max, expected_min) in cases
        {
            // Every permutation of the multiset must give identical bits:
            // the deterministic extrema kernels are associative, commutative
            // and idempotent, so encounter order is unobservable. This does
            // NOT generalize to ReduceSum/Prod/Mean, which keep fixed order.
            for permutation in [
                values.to_vec(),
                {
                    let mut v = values.to_vec();
                    v.reverse();
                    v
                },
                {
                    let mut v = values.to_vec();
                    let last = v.len() - 1;
                    v.swap(0, last);
                    if v.len() > 1
                    {
                        v.swap(0, 1);
                    }
                    v
                },
            ]
            {
                // The f64 carrier stores every binary32 value exactly, so the
                // same literal u64 expectations hold for both dtypes.
                let mut tensor_data = permutation.clone();
                while tensor_data.len() < 3
                {
                    // Pad by repeating the last element: idempotent kernels
                    // make duplicates semantically inert for extrema.
                    tensor_data.push(*tensor_data.last().unwrap());
                }
                let tensor = ValueTensor::new(dtype, vec![3], tensor_data).unwrap();
                let result = execute_program(
                    &build(dtype),
                    std::slice::from_ref(&tensor),
                    &[],
                    ExecutionPolicy {
                        floats: FloatPolicy::AllowNonFinite,
                    },
                    default_limits(),
                )
                .unwrap();
                assert_eq!(
                    result.outputs[0].data[0].to_bits(),
                    *expected_max,
                    "{dtype:?} max({values:?})"
                );
                assert_eq!(
                    result.outputs[1].data[0].to_bits(),
                    *expected_min,
                    "{dtype:?} min({values:?})"
                );
            }
        }
    }
    // The canonical quiet-NaN bit patterns are owned constants, not assumed
    // from the representation of the source-level NAN constants.
    assert_eq!(super::interpret::canonical_nan_f64().to_bits(), QNAN_F64);
    assert_eq!(super::interpret::canonical_nan_f32().to_bits(), QNAN_F32);
}

/// `Min(x,x)`/`Max(x,x)` collapse only where finite rewrites are admitted;
/// under `StrictIeee` the self-extrema nodes survive.
#[test]
fn min_max_self_rewrites_stay_finite_domain_gated() {
    let build = || {
        ResearchProgram::expression(
            vec![f64_type(&[4])],
            Section::new(vec![
                Op::Abs(Un::new(Ref::Input(0))),
                Op::Min(Bin::new(Ref::Local(0), Ref::Local(0))),
                Op::Max(Bin::new(Ref::Local(0), Ref::Local(0))),
                Op::Add(Bin::new(Ref::Local(1), Ref::Local(2))),
            ]),
            vec![3],
        )
    };
    let strict = super::simplify::canonicalize(
        &build().with_semantics(NumericalSemantics::StrictIeee),
        VerificationLimits::default(),
    )
    .unwrap();
    assert!(
        strict
            .program
            .finalize
            .ops
            .iter()
            .any(|op| matches!(op, Op::Min(_) | Op::Max(_))),
        "StrictIeee keeps self-extrema nodes"
    );
    let finite = super::simplify::canonicalize(
        &build().with_semantics(NumericalSemantics::FiniteOnly),
        VerificationLimits::default(),
    )
    .unwrap();
    assert!(
        !finite
            .program
            .finalize
            .ops
            .iter()
            .any(|op| matches!(op, Op::Min(_) | Op::Max(_))),
        "FiniteOnly collapses self-extrema nodes"
    );
}

/// The public conversion boundary returns a structured error instead of
/// panicking on a hostile dtype mismatch.
#[test]
fn to_f32_tensor_reports_a_structured_error_for_wrong_dtype() {
    let tensor = ValueTensor::scalar_f64(1.0);
    let error = tensor
        .to_f32_tensor()
        .expect_err("f64 must not convert as f32");
    assert_eq!(
        error,
        TensorDataError::DTypeMismatch {
            expected: DType::F32,
            found: DType::F64,
        }
    );
    let boolean = ValueTensor::scalar_bool(true);
    assert!(matches!(
        boolean.to_f32_tensor(),
        Err(TensorDataError::DTypeMismatch { .. })
    ));
}

/// A hostile `Narrow{start: usize::MAX}` must be rejected without the error
/// formatter itself overflowing (`start + len` used to panic in debug).
#[test]
fn overflowing_narrow_range_is_rejected_and_displayable() {
    let program = ResearchProgram::expression(
        vec![f64_type(&[4])],
        Section::new(vec![Op::Narrow(Narrow {
            src: Ref::Input(0),
            axis: 0,
            start: usize::MAX,
            len: 2,
        })]),
        vec![0],
    );
    let error = expect_error(&program, default_limits());
    assert!(matches!(error, ProgramError::NarrowRangeInvalid { .. }));
    // Formatting must not panic (debug builds check integer overflow).
    let rendered = error.to_string();
    assert!(rendered.contains("exceeds axis"));
}
