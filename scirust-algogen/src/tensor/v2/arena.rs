//! The honest arena: reference fixtures versus discovered candidates.
//!
//! Runs standardized datasets through a hand-written reference program and a
//! discovered candidate, reporting exact bit matches, error magnitudes, and
//! executed-node counts (a deterministic structural work proxy — never wall
//! time). This is the evidence infrastructure behind the claim that discovery
//! results are compared fairly against known algorithms.

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaOutcome {
    pub name: String,
    /// Every probed element of every case matched bit-for-bit.
    pub exact_bit_match: bool,
    /// Output shapes/types agreed everywhere (weaker than bit equality).
    pub shapes_match: bool,
    /// Cases where either side failed to execute.
    pub failed_cases: usize,
    pub total_cases: usize,
    /// Mean squared error aggregated over all compared elements.
    pub mean_squared_error: f64,
    pub max_absolute_error: f64,
    /// Structural work proxy (deterministic), summed over cases.
    pub reference_executed_nodes: u64,
    pub candidate_executed_nodes: u64,
}

impl ArenaOutcome {
    /// Whether the candidate is a strict behavioral improvement worth
    /// archiving: exact on every case and structurally cheaper.
    #[must_use]
    pub fn candidate_exact_and_cheaper(&self) -> bool {
        self.exact_bit_match
            && self.failed_cases == 0
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
        exact_bit_match: true,
        shapes_match: true,
        failed_cases: 0,
        total_cases: case.dataset.cases.len(),
        mean_squared_error: 0.0,
        max_absolute_error: 0.0,
        reference_executed_nodes: 0,
        candidate_executed_nodes: 0,
    };
    if case.dataset.cases.is_empty()
    {
        return outcome;
    }

    let mut squared_total = 0.0f64;
    let mut elements = 0usize;
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
                outcome.reference_executed_nodes += reference.executed_nodes;
                outcome.candidate_executed_nodes += candidate.executed_nodes;
                if reference.outputs.len() != candidate.outputs.len()
                    || reference
                        .outputs
                        .iter()
                        .zip(&candidate.outputs)
                        .any(|(r, c)| r.value_type() != c.value_type())
                {
                    outcome.shapes_match = false;
                    outcome.exact_bit_match = false;
                    outcome.failed_cases += 1;
                    continue;
                }
                let mut case_exact = true;
                for (reference_output, candidate_output) in
                    reference.outputs.iter().zip(&candidate.outputs)
                {
                    for (&r, &c) in reference_output.data.iter().zip(&candidate_output.data)
                    {
                        let difference = r - c;
                        squared_total = squared_total.mul_add(1.0, difference * difference);
                        outcome.max_absolute_error =
                            outcome.max_absolute_error.max(difference.abs());
                        case_exact &= r.to_bits() == c.to_bits();
                        elements += 1;
                    }
                }
                outcome.exact_bit_match &= case_exact;
            },
            _ =>
            {
                outcome.failed_cases += 1;
                outcome.exact_bit_match = false;
            },
        }
    }
    if elements > 0
    {
        outcome.mean_squared_error = squared_total / elements as f64;
    }
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
        assert_eq!(outcome.mean_squared_error, 0.0);
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
    }
}
