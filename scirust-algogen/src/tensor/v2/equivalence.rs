//! Bounded behavioral comparison between two programs.
//!
//! Two programs are *indistinguishable on the compared set* when they
//! produced bit-identical outputs for every grid case where **both sides
//! executed successfully**, within a deterministic, bounded input grid built
//! from the adversarial value sets. This is finite positive evidence over the
//! *successfully executed* cases only — deliberately NOT named
//! "equivalence":
//!
//! - a case both sides reject carries **no** evidence in either direction;
//!   it is counted, never silently folded into equality;
//! - if no case produced comparable outputs at all, the result says exactly
//!   that ([`BoundedEquivalence::NoComparableEvidence`]) instead of claiming
//!   indistinguishability from zero observations;
//! - an asymmetric outcome (one side executed, the other rejected) is itself
//!   behavioral divergence;
//! - no claim is made about inputs off the grid. Structural identity remains
//!   the domain of canonical bytes.

use super::adversarial::adversarial_scalars;
use super::interpret::{ExecutionPolicy, ValueTensor, execute_program};
use super::ir::ResearchProgram;
use super::verify::VerificationLimits;

/// Hard cap on grid size; larger grids are downsampled evenly and
/// deterministically (first and last case always kept) so runs stay
/// reproducible and the cap is never exceeded.
pub const MAX_GRID_CASES: usize = 512;

/// Which side of a comparison rejected a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparedSide {
    Left,
    Right,
}

/// Outcome of one bounded equivalence check.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundedEquivalence {
    /// Bit-identical outputs on every *successfully compared* case.
    ///
    /// `compared_cases >= 1` always: zero successful comparisons are
    /// reported as [`BoundedEquivalence::NoComparableEvidence`] instead.
    /// `jointly_rejected_cases` counts grid cases both sides rejected; they
    /// contribute no evidence in either direction.
    Indistinguishable {
        compared_cases: usize,
        jointly_rejected_cases: usize,
    },
    /// Both programs executed but produced different bits. The first
    /// diverging element in deterministic order is preserved as the
    /// counterexample.
    Diverged {
        case_index: usize,
        output: usize,
        element: usize,
        left_bits: u64,
        right_bits: u64,
    },
    /// One program executed while the other rejected the same case:
    /// outcome asymmetry, i.e. behavioral divergence at that case.
    OutcomeAsymmetric {
        case_index: usize,
        rejected: ComparedSide,
    },
    /// The two executions could not be aligned (arity, output type or data
    /// length mismatch). Never silently truncated away.
    OutputContractMismatch { case_index: usize, reason: String },
    /// The programs cannot be meaningfully compared at all (signature or
    /// semantics mismatch, verification failure, unbuildable probes…).
    Incomparable { reason: String },
    /// Every generated grid case was jointly rejected by both sides: there
    /// is NO positive evidence of either equivalence or divergence.
    NoComparableEvidence {
        generated_cases: usize,
        jointly_rejected_cases: usize,
    },
}

/// Deterministically downsample `items` to at most `cap` entries, keeping
/// both endpoints and spreading the rest evenly via exact index arithmetic
/// (`i * (len - 1) / (cap - 1)`); no integer-stride drift can skip boundary
/// cases or exceed the cap.
fn downsample<T>(items: Vec<T>, cap: usize) -> Vec<T> {
    let len = items.len();
    if cap == 0
    {
        return Vec::new();
    }
    if len <= cap
    {
        return items;
    }
    let last = len - 1;
    let keep: std::collections::BTreeSet<usize> = if cap == 1
    {
        std::iter::once(0).collect()
    }
    else
    {
        (0..cap).map(|index| index * last / (cap - 1)).collect()
    };
    items
        .into_iter()
        .enumerate()
        .filter(|(index, _)| keep.contains(index))
        .map(|(_, item)| item)
        .collect()
}

/// Compare two programs over the deterministic bounded grid.
///
/// Programs must share the full signature (inputs, items, state, steps) and
/// the exact output type list; anything else is
/// [`BoundedEquivalence::Incomparable`].
pub fn bounded_equivalence(
    left: &ResearchProgram,
    right: &ResearchProgram,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> BoundedEquivalence {
    if left.inputs != right.inputs
        || left.items != right.items
        || left.state != right.state
        || left.steps != right.steps
        || left.semantics != right.semantics
    {
        return BoundedEquivalence::Incomparable {
            reason: "signatures or semantics differ".to_string(),
        };
    }

    // Output types must agree; derive them by verification.
    let (Ok(left_verified), Ok(right_verified)) = (
        super::verify_program(left, limits),
        super::verify_program(right, limits),
    )
    else
    {
        return BoundedEquivalence::Incomparable {
            reason: "one program fails verification".to_string(),
        };
    };
    if left_verified.output_types != right_verified.output_types
    {
        return BoundedEquivalence::Incomparable {
            reason: "output types differ".to_string(),
        };
    }

    // Recurrent programs are compared only within the bounded probe budget;
    // refusing is more honest than comparing an arbitrary prefix silently.
    if left.steps > super::recognize::MAX_PROBED_STEPS
    {
        return BoundedEquivalence::Incomparable {
            reason: "recurrence exceeds the bounded probe budget".to_string(),
        };
    }

    // Build the case grid: aligned per-input salts plus the pairwise scalar
    // expansion for single-input programs, capped deterministically.
    let mut grids: Vec<Vec<ValueTensor>> = Vec::new();
    for salt in 0..6
    {
        let mut case = Vec::with_capacity(left.inputs.len());
        for (index, value_type) in left.inputs.iter().enumerate()
        {
            match super::recognize::probe_tensor(value_type, index + salt)
            {
                Ok(tensor) => case.push(tensor),
                Err(error) =>
                {
                    return BoundedEquivalence::Incomparable {
                        reason: format!("input probe construction failed: {error}"),
                    };
                },
            }
        }
        grids.push(case);
    }
    if left.inputs.len() == 1
    {
        let value_type = &left.inputs[0];
        if let Some(elements) = value_type.checked_elements()
        {
            for &value in &adversarial_scalars()
            {
                let data = vec![value; elements];
                // Dtype-hostile scalar values (e.g. binary64 extremes against
                // an F32 input) simply cannot form a legal case tensor; they
                // are skipped rather than forced through with an expect.
                if let Ok(tensor) =
                    ValueTensor::new(value_type.dtype, value_type.shape.clone(), data)
                {
                    grids.push(vec![tensor]);
                }
            }
        }
    }
    let generated_cases = grids.len();
    let grids = downsample(grids, MAX_GRID_CASES);

    // Recurrent programs need an item stream per case.
    let build_items = |program: &ResearchProgram, salt: usize| -> Option<Vec<ValueTensor>> {
        let steps = program.steps.min(super::recognize::MAX_PROBED_STEPS);
        let mut items = Vec::new();
        for step in 0..steps
        {
            for (slot, value_type) in program.items.iter().enumerate()
            {
                match super::recognize::probe_tensor(
                    value_type,
                    step as usize * 7 + slot * 3 + salt,
                )
                {
                    Ok(tensor) => items.push(tensor),
                    Err(_) => return None,
                }
            }
        }
        Some(items)
    };

    let mut compared_cases = 0usize;
    let mut jointly_rejected_cases = 0usize;

    for (case_index, inputs) in grids.iter().enumerate()
    {
        let Some(items) = build_items(left, case_index)
        else
        {
            return BoundedEquivalence::Incomparable {
                reason: "item probe construction failed".to_string(),
            };
        };
        let left_run = execute_program(left, inputs, &items, policy, limits);
        let right_run = execute_program(right, inputs, &items, policy, limits);
        match (left_run, right_run)
        {
            (Ok(a), Ok(b)) =>
            {
                if a.outputs.len() != b.outputs.len()
                {
                    return BoundedEquivalence::OutputContractMismatch {
                        case_index,
                        reason: format!(
                            "output arity differs: {} vs {}",
                            a.outputs.len(),
                            b.outputs.len()
                        ),
                    };
                }
                let mut case_compared = false;
                for (output, (la, rb)) in a.outputs.iter().zip(&b.outputs).enumerate()
                {
                    if la.value_type() != rb.value_type()
                    {
                        return BoundedEquivalence::OutputContractMismatch {
                            case_index,
                            reason: format!("output {output} types differ"),
                        };
                    }
                    if la.data.len() != rb.data.len()
                    {
                        return BoundedEquivalence::OutputContractMismatch {
                            case_index,
                            reason: format!("output {output} lengths differ"),
                        };
                    }
                    for (element, (&lv, &rv)) in la.data.iter().zip(&rb.data).enumerate()
                    {
                        if lv.to_bits() != rv.to_bits()
                        {
                            return BoundedEquivalence::Diverged {
                                case_index,
                                output,
                                element,
                                left_bits: lv.to_bits(),
                                right_bits: rv.to_bits(),
                            };
                        }
                    }
                    case_compared = true;
                }
                if case_compared
                {
                    compared_cases += 1;
                }
            },
            (Ok(_), Err(_)) =>
            {
                return BoundedEquivalence::OutcomeAsymmetric {
                    case_index,
                    rejected: ComparedSide::Right,
                };
            },
            (Err(_), Ok(_)) =>
            {
                return BoundedEquivalence::OutcomeAsymmetric {
                    case_index,
                    rejected: ComparedSide::Left,
                };
            },
            (Err(_), Err(_)) =>
            {
                // Both reject the case identically: not evidence of either
                // equivalence or divergence; counted explicitly.
                jointly_rejected_cases += 1;
            },
        }
    }

    if compared_cases == 0
    {
        // Zero successful comparisons is NOT indistinguishability.
        return BoundedEquivalence::NoComparableEvidence {
            generated_cases,
            jointly_rejected_cases,
        };
    }

    BoundedEquivalence::Indistinguishable {
        compared_cases,
        jointly_rejected_cases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Bin, Op, Ref, Section};
    use crate::tensor::v2::types::{DType, ScalarValue, ValueType};

    #[test]
    fn structurally_different_but_equivalent_programs_are_indistinguishable() {
        let limits = VerificationLimits::default();
        // abs(x) vs abs(abs(x)): identical on every successfully compared
        // grid case.
        let inner = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(super::super::ir::Un::new(Ref::Input(0)))]),
            vec![0],
        );
        let nested = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Abs(super::super::ir::Un::new(Ref::Input(0))),
                Op::Abs(super::super::ir::Un::new(Ref::Local(0))),
            ]),
            vec![1],
        );
        assert_eq!(
            bounded_equivalence(&inner, &nested, ExecutionPolicy::default(), limits),
            BoundedEquivalence::Indistinguishable {
                // 6 finite salt cases + 10 finite scalar expansions; the 3
                // non-finite expansion scalars (NaN, ±∞) are jointly
                // rejected under the default policy and carry no evidence.
                compared_cases: 16,
                jointly_rejected_cases: 3,
            }
        );
    }

    #[test]
    fn diverging_programs_report_a_concrete_case() {
        let limits = VerificationLimits::default();
        let identity = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(super::super::ir::Un::new(Ref::Input(0)))]),
            vec![0],
        );
        let negate = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Neg(super::super::ir::Un::new(Ref::Input(0)))]),
            vec![0],
        );
        match bounded_equivalence(&identity, &negate, ExecutionPolicy::default(), limits)
        {
            BoundedEquivalence::Diverged { left_bits, .. } =>
            {
                assert_eq!(left_bits, 1.0f64.to_bits());
            },
            other => panic!("expected divergence, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_signatures_are_incomparable() {
        let limits = VerificationLimits::default();
        let one_input = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Const(ScalarValue::F64(1.0))]),
            vec![0],
        );
        let two_inputs = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64), ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Add(Bin::new(Ref::Input(0), Ref::Input(1)))]),
            vec![0],
        );
        assert!(matches!(
            bounded_equivalence(&one_input, &two_inputs, ExecutionPolicy::default(), limits),
            BoundedEquivalence::Incomparable { .. }
        ));
    }

    /// A program that rejects its input (non-finite intermediate under the
    /// default finite policy) versus one that executes: outcome asymmetry,
    /// never "incomparable, ignore" and never indistinguishability.
    #[test]
    fn one_side_rejecting_is_outcome_asymmetry() {
        let limits = VerificationLimits::default();
        let absorbing = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(super::super::ir::Un::new(Ref::Input(0)))]),
            vec![0],
        );
        // 1/0 = +inf is rejected by the default finite-output policy before
        // anything observable leaves the program.
        let always_rejects = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Const(ScalarValue::F64(1.0)),
                Op::Const(ScalarValue::F64(0.0)),
                Op::Div(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2],
        );
        match bounded_equivalence(
            &absorbing,
            &always_rejects,
            ExecutionPolicy::default(),
            limits,
        )
        {
            BoundedEquivalence::OutcomeAsymmetric { rejected, .. } =>
            {
                assert_eq!(rejected, ComparedSide::Right);
            },
            other => panic!("expected outcome asymmetry, got {other:?}"),
        }
    }

    /// Two programs that reject EVERY grid case must NOT be reported as
    /// indistinguishable: no observation is not equivalence.
    #[test]
    fn jointly_rejecting_programs_yield_no_comparable_evidence() {
        let reject_a = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Const(ScalarValue::F64(1.0)),
                Op::Const(ScalarValue::F64(0.0)),
                Op::Div(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2],
        );
        let reject_b = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Const(ScalarValue::F64(-1.0)),
                Op::Const(ScalarValue::F64(0.0)),
                Op::Div(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2],
        );
        let result = bounded_equivalence(
            &reject_a,
            &reject_b,
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        );
        assert_eq!(
            result,
            BoundedEquivalence::NoComparableEvidence {
                generated_cases: 19,
                jointly_rejected_cases: 19,
            },
            "both-reject-everywhere must be explicit absence of evidence"
        );
    }

    /// Mixed accounting across a full sweep: structurally different programs
    /// (`abs` versus `max(x, -x)`) agree bit-for-bit on every comparable
    /// case while the three non-finite expansion scalars are jointly
    /// rejected under the default policy — and are counted, never folded
    /// into the comparison count.
    #[test]
    fn joint_rejections_are_counted_beside_successful_comparisons() {
        let limits = VerificationLimits::default();
        let left = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(super::super::ir::Un::new(Ref::Input(0)))]),
            vec![0],
        );
        let right = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Neg(super::super::ir::Un::new(Ref::Input(0))),
                Op::Max(Bin::new(Ref::Input(0), Ref::Local(0))),
            ]),
            vec![1],
        );
        assert_eq!(
            bounded_equivalence(&left, &right, ExecutionPolicy::default(), limits),
            BoundedEquivalence::Indistinguishable {
                compared_cases: 16,
                jointly_rejected_cases: 3,
            }
        );
    }

    /// The downsampler keeps both endpoints, never exceeds the cap, and is
    /// deterministic.
    #[test]
    fn downsampling_keeps_endpoints_and_respects_the_cap() {
        let data: Vec<usize> = (0..1025).collect();
        let sampled = downsample(data, MAX_GRID_CASES);
        assert_eq!(sampled.len(), MAX_GRID_CASES);
        assert_eq!(sampled[0], 0, "first case preserved");
        assert_eq!(sampled[MAX_GRID_CASES - 1], 1024, "last case preserved");
        assert!(sampled.windows(2).all(|window| window[0] < window[1]));
        // Below the cap: unchanged.
        let small: Vec<usize> = (0..19).collect();
        assert_eq!(downsample(small, MAX_GRID_CASES).len(), 19);
    }
}
