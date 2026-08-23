//! The honest arena: reference fixtures versus discovered candidates.
//!
//! Runs standardized datasets through a hand-written reference program and a
//! discovered candidate, reporting exact bit matches, error magnitudes and
//! executed-node counts. Executed nodes are a deterministic *structural work
//! proxy* — never wall time and never a hardware speedup claim; one `Exp`
//! node is not one scalar `Add` node. This is the evidence infrastructure
//! behind the claim that discovery results are compared fairly against known
//! algorithms.
//!
//! Vacuity is explicit: a dataset where nothing successfully compares can
//! never look like "zero numerical error" or "exact".

use serde::{Deserialize, Serialize};

use super::interpret::{ExecutionPolicy, execute_program};
use super::ir::ResearchProgram;
use super::search::CounterexampleSet;
use super::verify::VerificationLimits;

/// One arena matchup: a named reference, a candidate, and their dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaCase {
    pub name: String,
    pub reference: ResearchProgram,
    pub candidate: ResearchProgram,
    pub dataset: CounterexampleSet,
}

/// Aggregate result for one [`ArenaCase`].
///
/// Accounting invariants:
/// - `successful_cases + failed_cases <= total_cases`;
/// - errors (MSE / max-abs) aggregate ONLY over successfully compared
///   elements of successful cases;
/// - `mean_squared_error == 0.0` with `successful_cases == 0` means "no
///   evidence", never "no error";
/// - [`Self::exact_bit_match`] is `true` only when at least one case was
///   successfully compared AND every compared element matched bit-for-bit
///   AND no case failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaOutcome {
    pub name: String,
    /// Every successfully compared element of every non-failed case matched
    /// bit-for-bit — and there was at least one such case.
    pub exact_bit_match: bool,
    /// Output shapes/types agreed everywhere (weaker than bit equality).
    pub shapes_match: bool,
    /// Cases where either side failed to execute or could not be aligned.
    pub failed_cases: usize,
    /// Cases that executed on both sides and were fully compared.
    pub successful_cases: usize,
    pub total_cases: usize,
    /// Number of output elements actually compared across successful cases.
    pub compared_elements: u64,
    /// Mean squared error over compared elements only (`successful_cases`
    /// cases). Meaningless without checking [`Self::successful_cases`].
    pub mean_squared_error: f64,
    pub max_absolute_error: f64,
    /// Structural work proxy (deterministic), summed over successful cases.
    /// NOT runtime cost and NOT comparable across different node kinds.
    pub reference_executed_nodes: u64,
    pub candidate_executed_nodes: u64,
}

impl ArenaOutcome {
    /// Whether the candidate deserves archiving as a structural improvement:
    /// bit-exact against the reference on every case of a NON-EMPTY dataset
    /// with no failures, while evaluating strictly fewer IR nodes.
    ///
    /// This is "fewer executed nodes", not "faster": executed nodes are a
    /// deterministic structural proxy, not a cost model of hardware time.
    #[must_use]
    pub fn candidate_exact_and_fewer_executed_nodes(&self) -> bool {
        self.successful_cases > 0
            && self.successful_cases == self.total_cases
            && self.failed_cases == 0
            && self.exact_bit_match
            && self.candidate_executed_nodes < self.reference_executed_nodes
    }
}

/// Run every arena case in order with fixed policies. Deterministic:
/// identical inputs produce byte-identical outcomes.
pub fn run_arena(
    cases: &[ArenaCase],
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Vec<ArenaOutcome> {
    cases
        .iter()
        .map(|case| run_case(case, policy, limits))
        .collect()
}

fn run_case(case: &ArenaCase, policy: ExecutionPolicy, limits: VerificationLimits) -> ArenaOutcome {
    let mut outcome = ArenaOutcome {
        name: case.name.clone(),
        exact_bit_match: false,
        shapes_match: true,
        failed_cases: 0,
        successful_cases: 0,
        total_cases: case.dataset.cases.len(),
        compared_elements: 0,
        mean_squared_error: 0.0,
        max_absolute_error: 0.0,
        reference_executed_nodes: 0,
        candidate_executed_nodes: 0,
    };

    let mut all_compared_exact = true;
    let mut squared_total = 0.0f64;
    for case_data in &case.dataset.cases
    {
        let reference_run = execute_program(
            &case.reference,
            &case_data.inputs,
            &case_data.items,
            policy,
            limits,
        );
        let candidate_run = execute_program(
            &case.candidate,
            &case_data.inputs,
            &case_data.items,
            policy,
            limits,
        );
        match (reference_run, candidate_run)
        {
            (Ok(reference), Ok(candidate)) =>
            {
                if reference.outputs.len() != candidate.outputs.len()
                    || reference
                        .outputs
                        .iter()
                        .zip(&candidate.outputs)
                        .any(|(r, c)| r.value_type() != c.value_type())
                {
                    outcome.shapes_match = false;
                    outcome.failed_cases += 1;
                    all_compared_exact = false;
                    continue;
                }
                for (reference_output, candidate_output) in
                    reference.outputs.iter().zip(&candidate.outputs)
                {
                    // Lengths are guaranteed equal by the shared verified
                    // type; zip cannot silently truncate here, but the guard
                    // keeps the invariant local and loud if it ever changes.
                    debug_assert_eq!(reference_output.data.len(), candidate_output.data.len());
                    for (&r, &c) in reference_output.data.iter().zip(&candidate_output.data)
                    {
                        let difference = r - c;
                        squared_total = squared_total.mul_add(1.0, difference * difference);
                        outcome.max_absolute_error =
                            outcome.max_absolute_error.max(difference.abs());
                        all_compared_exact &= r.to_bits() == c.to_bits();
                        outcome.compared_elements += 1;
                    }
                }
                outcome.reference_executed_nodes += reference.executed_nodes;
                outcome.candidate_executed_nodes += candidate.executed_nodes;
                outcome.successful_cases += 1;
            },
            _ =>
            {
                outcome.failed_cases += 1;
                all_compared_exact = false;
            },
        }
    }
    if outcome.compared_elements > 0
    {
        outcome.mean_squared_error = squared_total / outcome.compared_elements as f64;
    }
    // Vacuity is never positive evidence: an empty or all-failed dataset
    // must not read as "bit-exact".
    outcome.exact_bit_match =
        outcome.successful_cases > 0 && outcome.failed_cases == 0 && all_compared_exact;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Op, Ref, Section};
    use crate::tensor::v2::search::CounterexampleCase;
    use crate::tensor::v2::types::{ScalarValue, ValueType};
    use crate::tensor::v2::{DType, ValueTensor};

    fn sum_dataset() -> CounterexampleSet {
        // Every case must match the programs' declared [3] signature.
        let make = |values: &[f64]| {
            assert_eq!(values.len(), 3);
            let expected = values.iter().sum::<f64>();
            CounterexampleCase {
                inputs: vec![
                    ValueTensor::new(DType::F64, vec![values.len()], values.to_vec()).unwrap(),
                ],
                items: vec![],
                expected_outputs: vec![ValueTensor::scalar_f64(expected)],
            }
        };
        CounterexampleSet::new("sum", vec![make(&[1.0, 2.0, 3.0]), make(&[-1.5, 0.5, 4.0])])
            .unwrap()
    }

    fn reduction_program(length: usize) -> ResearchProgram {
        ResearchProgram::expression(
            vec![ValueType::new(DType::F64, vec![length])],
            Section::new(vec![Op::ReduceSum(super::super::ir::Reduce {
                src: Ref::Input(0),
                axis: None,
            })]),
            vec![0],
        )
    }

    #[test]
    fn equivalent_candidates_match_references_exactly() {
        // Same reduction, different declared length is a different signature;
        // use matching lengths and an exact tie on cost.
        let case = ArenaCase {
            name: "sum_vs_sum".to_string(),
            reference: reduction_program(3),
            candidate: reduction_program(3),
            dataset: sum_dataset(),
        };
        let outcome = run_arena(
            std::slice::from_ref(&case),
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .pop()
        .unwrap();
        assert!(outcome.exact_bit_match);
        assert!(outcome.shapes_match);
        assert_eq!(outcome.failed_cases, 0);
        assert_eq!(outcome.successful_cases, 2);
        assert_eq!(outcome.compared_elements, 2);
        assert_eq!(outcome.mean_squared_error, 0.0);
        // Exact tie on nodes: not a structural improvement.
        assert!(!outcome.candidate_exact_and_fewer_executed_nodes());
    }

    #[test]
    fn wrong_candidates_are_reported_honestly() {
        // Candidate ignores its input and returns a constant.
        let constant = ResearchProgram::expression(
            vec![ValueType::new(DType::F64, vec![3])],
            Section::new(vec![Op::Const(ScalarValue::F64(6.0))]),
            vec![0],
        );
        let case = ArenaCase {
            name: "sum_vs_constant".to_string(),
            reference: reduction_program(3),
            candidate: constant,
            dataset: sum_dataset(),
        };
        let outcome = run_arena(
            std::slice::from_ref(&case),
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .pop()
        .unwrap();
        assert!(!outcome.exact_bit_match);
        assert!(outcome.max_absolute_error > 0.0 || outcome.mean_squared_error > 0.0);
        assert!(!outcome.candidate_exact_and_fewer_executed_nodes());
    }

    /// Regression: an empty dataset used to initialize `exact_bit_match =
    /// true`, making a candidate look exact with ZERO observations. Vacuous
    /// agreement is now impossible at the reporting level.
    #[test]
    fn empty_datasets_cannot_qualify_a_candidate() {
        // CounterexampleSet::new refuses empty datasets by design; the serde
        // path can still produce them, which is exactly the hostile input
        // this test pins down.
        let empty = CounterexampleSet {
            id: "empty".to_string(),
            cases: Vec::new(),
        };
        let case = ArenaCase {
            name: "vacuous".to_string(),
            reference: reduction_program(3),
            candidate: reduction_program(3),
            dataset: empty,
        };
        let outcome = run_arena(
            std::slice::from_ref(&case),
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .pop()
        .unwrap();
        assert_eq!(outcome.total_cases, 0);
        assert_eq!(outcome.successful_cases, 0);
        assert!(
            !outcome.exact_bit_match,
            "an empty dataset must not report exactness"
        );
        assert!(
            !outcome.candidate_exact_and_fewer_executed_nodes(),
            "an empty dataset must never qualify a candidate"
        );
    }

    /// A dataset where every case fails must NOT read as zero-error
    /// evidence: MSE stays 0.0 but successful_cases exposes the vacuity.
    #[test]
    fn all_failed_datasets_are_explicitly_not_zero_error() {
        // Signature mismatch: the dataset declares [3]-shaped inputs but the
        // programs expect [1]; every case fails structurally.
        let make = |values: &[f64]| CounterexampleCase {
            inputs: vec![
                ValueTensor::new(DType::F64, vec![values.len()], values.to_vec()).unwrap(),
            ],
            items: vec![],
            expected_outputs: vec![ValueTensor::scalar_f64(values.iter().sum::<f64>())],
        };
        let mismatched = CounterexampleSet::new("mismatch", vec![make(&[7.0, 8.0, 9.0])]).unwrap();
        let single_input = ResearchProgram::expression(
            vec![ValueType::new(DType::F64, vec![1])],
            Section::new(vec![Op::ReduceSum(super::super::ir::Reduce {
                src: Ref::Input(0),
                axis: None,
            })]),
            vec![0],
        );
        let case = ArenaCase {
            name: "all_fail".to_string(),
            reference: single_input.clone(),
            candidate: single_input,
            dataset: mismatched,
        };
        let outcome = run_arena(
            std::slice::from_ref(&case),
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        )
        .pop()
        .unwrap();
        assert_eq!(outcome.failed_cases, outcome.total_cases);
        assert_eq!(outcome.successful_cases, 0);
        assert_eq!(outcome.mean_squared_error, 0.0);
        assert!(
            !outcome.exact_bit_match,
            "MSE=0 over zero comparisons is absence of evidence, not zero error"
        );
    }
}
