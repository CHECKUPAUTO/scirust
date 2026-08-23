//! Auditable reference constructions made exclusively from general V2 ops.
//!
//! These are representation fixtures and integration examples, not generator
//! shortcuts and not target-specific IR primitives. Search grammars do not
//! call this module.

use super::ir::{Bin, Narrow, Op, Reduce, Ref, ResearchProgram, Section, ShapeTo, Ter, Un};
use super::semantics::NumericalSemantics;
use super::types::{DType, ScalarValue, ValueType};

fn f64_scalar() -> ValueType {
    ValueType::scalar(DType::F64)
}

fn f64_vector(length: usize) -> ValueType {
    ValueType::new(DType::F64, vec![length])
}

fn bool_vector(length: usize) -> ValueType {
    ValueType::new(DType::Bool, vec![length])
}

/// `y = a*x + b` as two distinct roundings (`Mul`, then `Add`), not FMA.
#[must_use]
pub fn affine_scalar_program() -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_scalar(), f64_scalar(), f64_scalar()],
        Section::new(vec![
            Op::Mul(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Add(Bin::new(Ref::Local(0), Ref::Input(2))),
        ]),
        vec![1],
    )
}

/// Sum all elements of a vector using the fixed row-major reduction order.
#[must_use]
pub fn reduction_sum_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length)],
        Section::new(vec![Op::ReduceSum(Reduce {
            src: Ref::Input(0),
            axis: None,
        })]),
        vec![0],
    )
}

/// Maximum of all vector elements.
#[must_use]
pub fn reduction_max_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length)],
        Section::new(vec![Op::ReduceMax(Reduce {
            src: Ref::Input(0),
            axis: None,
        })]),
        vec![0],
    )
}

/// Stable two-pass softmax building blocks, with ordered outputs `(m, e, l)`:
/// `m=max(x)`, `e=exp(x-m)`, `l=sum(e)`. There is no Softmax opcode.
#[must_use]
pub fn two_pass_softmax_building_blocks(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length)],
        Section::new(vec![
            Op::ReduceMax(Reduce {
                src: Ref::Input(0),
                axis: None,
            }),
            Op::Sub(Bin::new(Ref::Input(0), Ref::Local(0))),
            Op::Exp(Un { src: Ref::Local(1) }),
            Op::ReduceSum(Reduce {
                src: Ref::Local(2),
                axis: None,
            }),
        ]),
        vec![0, 2, 3],
    )
}

/// Stable online-softmax `(m, l)` fold over one scalar item per step.
#[must_use]
pub fn online_softmax_recurrence(steps: u32) -> ResearchProgram {
    ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![f64_scalar()],
        state: vec![f64_scalar(), f64_scalar()],
        steps,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(f64::NEG_INFINITY)),
            Op::Const(ScalarValue::F64(0.0)),
        ]),
        init_state: vec![0, 1],
        step: Section::new(vec![
            // m_new = max(m_old, x)
            Op::Max(Bin::new(Ref::StatePrev(0), Ref::Item(0))),
            // old contribution: l_old * exp(m_old - m_new)
            Op::Sub(Bin::new(Ref::StatePrev(0), Ref::Local(0))),
            Op::Exp(Un { src: Ref::Local(1) }),
            Op::Mul(Bin::new(Ref::StatePrev(1), Ref::Local(2))),
            // current contribution: exp(x - m_new)
            Op::Sub(Bin::new(Ref::Item(0), Ref::Local(0))),
            Op::Exp(Un { src: Ref::Local(4) }),
            Op::Add(Bin::new(Ref::Local(3), Ref::Local(5))),
        ]),
        next_state: vec![0, 6],
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
    }
}

/// Welford fold with ordered state/output `(count, mean, M2)`.
#[must_use]
pub fn welford_recurrence(steps: u32) -> ResearchProgram {
    ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![f64_scalar()],
        state: vec![f64_scalar(), f64_scalar(), f64_scalar()],
        steps,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(0.0)),
            Op::Const(ScalarValue::F64(0.0)),
            Op::Const(ScalarValue::F64(0.0)),
        ]),
        init_state: vec![0, 1, 2],
        step: Section::new(vec![
            Op::Const(ScalarValue::F64(1.0)),
            Op::Add(Bin::new(Ref::StatePrev(0), Ref::Local(0))),
            Op::Sub(Bin::new(Ref::Item(0), Ref::StatePrev(1))),
            Op::Div(Bin::new(Ref::Local(2), Ref::Local(1))),
            Op::Add(Bin::new(Ref::StatePrev(1), Ref::Local(3))),
            Op::Sub(Bin::new(Ref::Item(0), Ref::Local(4))),
            Op::Mul(Bin::new(Ref::Local(2), Ref::Local(5))),
            Op::Add(Bin::new(Ref::StatePrev(2), Ref::Local(6))),
        ]),
        next_state: vec![1, 4, 7],
        finalize: Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(0),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(1),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(2),
                shape: vec![],
            }),
        ]),
        outputs: vec![0, 1, 2],
    }
}

/// Kahan-style `(sum, compensation)` fold.
#[must_use]
pub fn compensated_sum_recurrence(steps: u32) -> ResearchProgram {
    ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![f64_scalar()],
        state: vec![f64_scalar(), f64_scalar()],
        steps,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(0.0)),
            Op::Const(ScalarValue::F64(0.0)),
        ]),
        init_state: vec![0, 1],
        step: Section::new(vec![
            Op::Sub(Bin::new(Ref::Item(0), Ref::StatePrev(1))),
            Op::Add(Bin::new(Ref::StatePrev(0), Ref::Local(0))),
            Op::Sub(Bin::new(Ref::Local(1), Ref::StatePrev(0))),
            Op::Sub(Bin::new(Ref::Local(2), Ref::Local(0))),
        ]),
        next_state: vec![1, 3],
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
    }
}

/// Generic matrix multiplication fixture.
#[must_use]
pub fn matrix_multiplication_program(m: usize, k: usize, n: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![
            ValueType::new(DType::F64, vec![m, k]),
            ValueType::new(DType::F64, vec![k, n]),
        ],
        Section::new(vec![Op::MatMul(Bin::new(Ref::Input(0), Ref::Input(1)))]),
        vec![0],
    )
}

/// Elementwise masked conditional update.
#[must_use]
pub fn masked_update_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length), f64_vector(length), bool_vector(length)],
        Section::new(vec![Op::Select(Ter::new(
            Ref::Input(2),
            Ref::Input(1),
            Ref::Input(0),
        ))]),
        vec![0],
    )
}

/// Explicit scalar-to-vector broadcast followed by elementwise addition.
#[must_use]
pub fn shape_broadcast_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length), f64_scalar()],
        Section::new(vec![
            Op::BroadcastTo(ShapeTo {
                src: Ref::Input(1),
                shape: vec![length],
            }),
            Op::Add(Bin::new(Ref::Input(0), Ref::Local(0))),
        ]),
        vec![1],
    )
}

/// Static narrow + mask/select + reduction building blocks. Ordered outputs:
/// `(selected, selected_sum, prefix)`.
#[must_use]
pub fn indexed_masked_accumulation_program(length: usize) -> ResearchProgram {
    let prefix = length.min(2);
    ResearchProgram::expression(
        vec![f64_vector(length), bool_vector(length)],
        Section::new(vec![
            Op::Const(ScalarValue::F64(0.0)),
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(0),
                shape: vec![length],
            }),
            Op::Select(Ter::new(Ref::Input(1), Ref::Input(0), Ref::Local(1))),
            Op::ReduceSum(Reduce {
                src: Ref::Local(2),
                axis: None,
            }),
            Op::Narrow(Narrow {
                src: Ref::Input(0),
                axis: 0,
                start: 0,
                len: prefix,
            }),
        ]),
        vec![2, 3, 4],
    )
}

/// Error-budget algebra: ordered outputs `(abs_error, L1_error, max_error)`.
#[must_use]
pub fn error_budget_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length), f64_vector(length)],
        Section::new(vec![
            Op::Sub(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Abs(Un { src: Ref::Local(0) }),
            Op::ReduceSum(Reduce {
                src: Ref::Local(1),
                axis: None,
            }),
            Op::ReduceMax(Reduce {
                src: Ref::Local(1),
                axis: None,
            }),
        ]),
        vec![1, 2, 3],
    )
}

/// Threshold/support mask and support-driven value selection.
#[must_use]
pub fn threshold_support_program(length: usize) -> ResearchProgram {
    ResearchProgram::expression(
        vec![f64_vector(length), f64_scalar()],
        Section::new(vec![
            Op::Gt(Bin::new(Ref::Input(0), Ref::Input(1))),
            Op::Const(ScalarValue::F64(0.0)),
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(1),
                shape: vec![length],
            }),
            Op::Select(Ter::new(Ref::Local(0), Ref::Input(0), Ref::Local(2))),
        ]),
        vec![0, 3],
    )
}

/// Sum/max/min/mean bound/statistic building blocks.
#[must_use]
pub fn reduction_statistics_program(length: usize) -> ResearchProgram {
    let source = Ref::Input(0);
    ResearchProgram::expression(
        vec![f64_vector(length)],
        Section::new(vec![
            Op::ReduceSum(Reduce {
                src: source,
                axis: None,
            }),
            Op::ReduceMax(Reduce {
                src: source,
                axis: None,
            }),
            Op::ReduceMin(Reduce {
                src: source,
                axis: None,
            }),
            Op::ReduceMean(Reduce {
                src: source,
                axis: None,
            }),
        ]),
        vec![0, 1, 2, 3],
    )
}

/// Bounded Newton update for `sqrt(a)` with state `(a, y)` and outer inputs
/// `(a, initial_guess)`. No dynamic termination or unrestricted loop exists.
#[must_use]
pub fn bounded_root_recurrence(steps: u32) -> ResearchProgram {
    ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![f64_scalar(), f64_scalar()],
        items: vec![],
        state: vec![f64_scalar(), f64_scalar()],
        steps,
        init: Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::Input(0),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::Input(1),
                shape: vec![],
            }),
        ]),
        init_state: vec![0, 1],
        step: Section::new(vec![
            Op::Const(ScalarValue::F64(0.5)),
            Op::Div(Bin::new(Ref::StatePrev(0), Ref::StatePrev(1))),
            Op::Add(Bin::new(Ref::StatePrev(1), Ref::Local(1))),
            Op::Mul(Bin::new(Ref::Local(0), Ref::Local(2))),
            Op::Reshape(ShapeTo {
                src: Ref::StatePrev(0),
                shape: vec![],
            }),
        ]),
        next_state: vec![4, 3],
        finalize: Section::new(vec![Op::Reshape(ShapeTo {
            src: Ref::StateFinal(1),
            shape: vec![],
        })]),
        outputs: vec![0],
    }
}

/// Attention-like stable fold with state `(m, l, o)` and step items `(score,
/// value_vector)`. The vector accumulator update is built from broadcast
/// arithmetic; no Attention or OnlineSoftmax opcode exists.
#[must_use]
pub fn attention_recurrence(steps: u32, value_width: usize) -> ResearchProgram {
    ResearchProgram {
        semantics: NumericalSemantics::StrictIeee,
        inputs: vec![],
        items: vec![f64_scalar(), f64_vector(value_width)],
        state: vec![f64_scalar(), f64_scalar(), f64_vector(value_width)],
        steps,
        init: Section::new(vec![
            Op::Const(ScalarValue::F64(f64::NEG_INFINITY)),
            Op::Const(ScalarValue::F64(0.0)),
            Op::Const(ScalarValue::F64(0.0)),
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(2),
                shape: vec![value_width],
            }),
        ]),
        init_state: vec![0, 1, 3],
        step: Section::new(vec![
            Op::Max(Bin::new(Ref::StatePrev(0), Ref::Item(0))),
            Op::Sub(Bin::new(Ref::StatePrev(0), Ref::Local(0))),
            Op::Exp(Un { src: Ref::Local(1) }),
            Op::Sub(Bin::new(Ref::Item(0), Ref::Local(0))),
            Op::Exp(Un { src: Ref::Local(3) }),
            Op::Mul(Bin::new(Ref::StatePrev(1), Ref::Local(2))),
            Op::Add(Bin::new(Ref::Local(5), Ref::Local(4))),
            Op::Mul(Bin::new(Ref::StatePrev(2), Ref::Local(2))),
            Op::Mul(Bin::new(Ref::Item(1), Ref::Local(4))),
            Op::Add(Bin::new(Ref::Local(7), Ref::Local(8))),
        ]),
        next_state: vec![0, 6, 9],
        finalize: Section::new(vec![
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(0),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(1),
                shape: vec![],
            }),
            Op::Reshape(ShapeTo {
                src: Ref::StateFinal(2),
                shape: vec![value_width],
            }),
            Op::Div(Bin::new(Ref::Local(2), Ref::Local(1))),
        ]),
        outputs: vec![0, 1, 2, 3],
    }
}

/// Executable evidence mapping the ten ADA readiness areas to general IR
/// programs. Some mappings intentionally share a fixture because the same
/// primitive family supports more than one future algorithm class.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaReadinessPrograms {
    pub a1_online_softmax: ResearchProgram,
    pub a2_indexed_masked_accumulation: ResearchProgram,
    pub a3_error_budget: ResearchProgram,
    pub a4_threshold_support: ResearchProgram,
    pub a5_bounds_and_reductions: ResearchProgram,
    pub a6_bounded_root_update: ResearchProgram,
    pub a7_moment_recurrence: ResearchProgram,
    pub a8_attention_recurrence: ResearchProgram,
    pub a9_distribution_statistics: ResearchProgram,
    pub a10_deterministic_oracle: ResearchProgram,
}

/// Build the A1–A10 representation fixtures with bounded dimensions/trips.
#[must_use]
pub fn ada_readiness_programs(steps: u32, vector_length: usize) -> AdaReadinessPrograms {
    AdaReadinessPrograms {
        a1_online_softmax: online_softmax_recurrence(steps),
        a2_indexed_masked_accumulation: indexed_masked_accumulation_program(vector_length),
        a3_error_budget: error_budget_program(vector_length),
        a4_threshold_support: threshold_support_program(vector_length),
        a5_bounds_and_reductions: reduction_statistics_program(vector_length),
        a6_bounded_root_update: bounded_root_recurrence(steps),
        a7_moment_recurrence: welford_recurrence(steps),
        a8_attention_recurrence: attention_recurrence(steps, vector_length),
        a9_distribution_statistics: reduction_statistics_program(vector_length),
        a10_deterministic_oracle: two_pass_softmax_building_blocks(vector_length),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{
        ExecutionPolicy, ValueTensor, VerificationLimits, execute_program, verify_program,
    };

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> ValueTensor {
        ValueTensor::new(DType::F64, shape, data).unwrap()
    }

    fn run(
        program: &ResearchProgram,
        inputs: &[ValueTensor],
        items: &[ValueTensor],
    ) -> Vec<ValueTensor> {
        execute_program(
            program,
            inputs,
            items,
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .unwrap()
        .outputs
    }

    #[test]
    fn required_reference_programs_verify_and_execute() {
        let affine = run(
            &affine_scalar_program(),
            &[
                tensor(vec![3.0], vec![]),
                tensor(vec![2.0], vec![]),
                tensor(vec![1.0], vec![]),
            ],
            &[],
        );
        assert_eq!(affine[0].data, vec![7.0]);

        let values = tensor(vec![1.0, 4.0, -2.0, 3.0], vec![4]);
        assert_eq!(
            run(
                &reduction_sum_program(4),
                std::slice::from_ref(&values),
                &[]
            )[0]
            .data,
            vec![6.0]
        );
        assert_eq!(
            run(
                &reduction_max_program(4),
                std::slice::from_ref(&values),
                &[]
            )[0]
            .data,
            vec![4.0]
        );

        let softmax = run(
            &two_pass_softmax_building_blocks(4),
            std::slice::from_ref(&values),
            &[],
        );
        assert_eq!(softmax[0].data, vec![4.0]);
        assert_eq!(softmax[1].shape, vec![4]);
        assert_eq!(softmax[2].shape, Vec::<usize>::new());

        let stream = [1.0, 2.0, 3.0]
            .into_iter()
            .map(ValueTensor::scalar_f64)
            .collect::<Vec<_>>();
        let online = run(&online_softmax_recurrence(3), &[], &stream);
        assert_eq!(online[0].data, vec![3.0]);
        let expected_l = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        assert!((online[1].data[0] - expected_l).abs() < 1.0e-15);

        let welford_items = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(ValueTensor::scalar_f64)
            .collect::<Vec<_>>();
        let welford = run(&welford_recurrence(4), &[], &welford_items);
        assert_eq!(
            welford
                .iter()
                .map(|value| value.data[0])
                .collect::<Vec<_>>(),
            vec![4.0, 2.5, 5.0]
        );

        let compensated = run(&compensated_sum_recurrence(3), &[], &stream);
        assert_eq!(compensated[0].data, vec![6.0]);
        assert_eq!(compensated[1].data, vec![0.0]);
    }

    #[test]
    fn matrix_mask_and_broadcast_reference_programs_execute() {
        let matrix = run(
            &matrix_multiplication_program(2, 2, 2),
            &[
                tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
            ],
            &[],
        );
        assert_eq!(matrix[0].data, vec![19.0, 22.0, 43.0, 50.0]);

        let masked = run(
            &masked_update_program(3),
            &[
                tensor(vec![1.0, 2.0, 3.0], vec![3]),
                tensor(vec![4.0, 5.0, 6.0], vec![3]),
                ValueTensor::new(DType::Bool, vec![3], vec![1.0, 0.0, 1.0]).unwrap(),
            ],
            &[],
        );
        assert_eq!(masked[0].data, vec![4.0, 2.0, 6.0]);

        let broadcast = run(
            &shape_broadcast_program(3),
            &[
                tensor(vec![1.0, 2.0, 3.0], vec![3]),
                tensor(vec![10.0], vec![]),
            ],
            &[],
        );
        assert_eq!(broadcast[0].data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn ada_a1_through_a10_are_verifier_backed_not_asserted() {
        let programs = ada_readiness_programs(3, 4);
        for (label, program) in [
            ("A1", programs.a1_online_softmax),
            ("A2", programs.a2_indexed_masked_accumulation),
            ("A3", programs.a3_error_budget),
            ("A4", programs.a4_threshold_support),
            ("A5", programs.a5_bounds_and_reductions),
            ("A6", programs.a6_bounded_root_update),
            ("A7", programs.a7_moment_recurrence),
            ("A8", programs.a8_attention_recurrence),
            ("A9", programs.a9_distribution_statistics),
            ("A10", programs.a10_deterministic_oracle),
        ]
        {
            assert!(
                verify_program(&program, VerificationLimits::default()).is_ok(),
                "{label} fixture failed verification"
            );
        }
    }

    #[test]
    fn bounded_root_and_attention_recurrences_execute_without_materializing_scan() {
        let root = run(
            &bounded_root_recurrence(4),
            &[ValueTensor::scalar_f64(9.0), ValueTensor::scalar_f64(3.0)],
            &[],
        );
        assert_eq!(root[0].data, vec![3.0]);

        let attention_items = vec![
            ValueTensor::scalar_f64(0.0),
            tensor(vec![1.0, 2.0], vec![2]),
            ValueTensor::scalar_f64(1.0),
            tensor(vec![3.0, 4.0], vec![2]),
        ];
        let attention = run(&attention_recurrence(2, 2), &[], &attention_items);
        assert_eq!(attention.len(), 4);
        assert!(
            attention
                .iter()
                .all(|value| value.data.iter().all(|v| v.is_finite()))
        );
    }
}
