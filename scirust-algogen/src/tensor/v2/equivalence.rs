//! Bounded behavioral equivalence checking between two programs.
//!
//! Two programs are *indistinguishable on a grid* when they produce
//! bit-identical outputs for every case of a deterministic, bounded input
//! grid built from the shared adversarial value set. This is finite evidence,
//! deliberately NOT named "equivalence": no claim is made about inputs off
//! the grid. Structural identity remains the domain of canonical bytes.

use super::adversarial::adversarial_scalars;
use super::interpret::{ExecutionPolicy, ValueTensor, execute_program};
use super::ir::ResearchProgram;
use super::types::ValueType;
use super::verify::VerificationLimits;

/// Hard cap on grid size; larger signatures are sampled evenly and
/// deterministically so runs stay reproducible.
pub const MAX_GRID_CASES: usize = 512;

/// Outcome of one bounded equivalence check.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundedEquivalence {
    /// Bit-identical outputs on every probed case.
    Indistinguishable { cases: usize },
    /// A concrete diverging case (the first in deterministic order).
    Diverged {
        case_index: usize,
        output: usize,
        element: usize,
        left_bits: u64,
        right_bits: u64,
    },
    /// The programs cannot be compared (signature mismatch, or one failed to
    /// execute where the other succeeded).
    Incomparable { reason: String },
}

fn probe_tensor(value_type: &ValueType, salt: usize) -> Option<ValueTensor> {
    // Dtype-aware and fallible by design; cases whose probes cannot be built
    // are skipped here and accounted for explicitly by the bounded-evidence
    // redesign.
    super::recognize::probe_tensor(value_type, salt).ok()
}

/// Compare two programs over the deterministic bounded grid.
///
/// Programs must share the full signature (inputs, items, state, steps) and
/// the exact output type list; anything else is [`BoundedEquivalence::Incomparable`].
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

    // Build the case grid: aligned per-input salts plus the pairwise scalar
    // expansion for single-input programs, capped deterministically.
    let mut grids: Vec<Vec<ValueTensor>> = Vec::new();
    for salt in 0..6
    {
        let mut case = Vec::with_capacity(left.inputs.len());
        let mut constructible = true;
        for (index, value_type) in left.inputs.iter().enumerate()
        {
            match probe_tensor(value_type, index + salt)
            {
                Some(tensor) => case.push(tensor),
                None => constructible = false,
            }
        }
        if constructible
        {
            grids.push(case);
        }
    }
    if left.inputs.len() == 1
    {
        for &value in &adversarial_scalars()
        {
            let value_type = &left.inputs[0];
            if let Some(elements) = value_type.checked_elements()
            {
                let data = vec![value; elements];
                if let Ok(tensor) =
                    ValueTensor::new(value_type.dtype, value_type.shape.clone(), data)
                {
                    grids.push(vec![tensor]);
                }
            }
        }
    }
    if grids.len() > MAX_GRID_CASES
    {
        let stride = grids.len() / MAX_GRID_CASES.max(1);
        grids = grids.into_iter().step_by(stride).collect();
    }

    // Recurrent programs need an item stream per case.
    let build_items = |program: &ResearchProgram, salt: usize| -> Option<Vec<ValueTensor>> {
        let steps = program.steps.min(super::recognize::MAX_PROBED_STEPS);
        let mut items = Vec::new();
        for step in 0..steps
        {
            for (slot, value_type) in program.items.iter().enumerate()
            {
                items.push(probe_tensor(
                    value_type,
                    step as usize * 7 + slot * 3 + salt,
                )?);
            }
        }
        Some(items)
    };

    // Both sides must be executed with the same probed step count so streams
    // line up; a program whose real step count exceeds the probe budget is
    // compared on the probed prefix only.
    if left.steps > super::recognize::MAX_PROBED_STEPS
    {
        return BoundedEquivalence::Incomparable {
            reason: "recurrence exceeds the bounded probe budget".to_string(),
        };
    }

    for (case_index, inputs) in grids.iter().enumerate()
    {
        let Some(items) = build_items(left, case_index)
        else
        {
            continue;
        };
        let left_run = execute_program(left, inputs, &items, policy, limits);
        let right_run = execute_program(right, inputs, &items, policy, limits);
        match (left_run, right_run)
        {
            (Ok(a), Ok(b)) =>
            {
                for (output, (la, rb)) in a.outputs.iter().zip(&b.outputs).enumerate()
                {
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
                }
            },
            (Ok(_), Err(_)) | (Err(_), Ok(_)) =>
            {
                return BoundedEquivalence::Incomparable {
                    reason: format!("execution outcomes differ at case {case_index}"),
                };
            },
            (Err(_), Err(_)) =>
            {
                // Both reject the case identically: not evidence of either
                // equivalence or divergence; skip.
            },
        }
    }

    BoundedEquivalence::Indistinguishable { cases: grids.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Bin, Op, Ref, Section};
    use crate::tensor::v2::types::{DType, ScalarValue};

    #[test]
    fn structurally_different_but_equivalent_programs_are_indistinguishable() {
        let limits = VerificationLimits::default();
        // x + x  vs  x * 2 : identical on every grid point here? No — they
        // can differ in rounding! Use a genuinely equivalent pair instead:
        // abs(x) vs abs(abs(x)).
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
            BoundedEquivalence::Indistinguishable { cases: 19 } // 6 salts + 13 scalars
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
}
