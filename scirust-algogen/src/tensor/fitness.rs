//! Multi-objective fitness: correctness loss combined with structural cost.
//!
//! Correctness loss is the mean per-case mean-squared error against the exact
//! expected outputs. A case that fails to execute, produces the wrong shape, or
//! yields a non-finite value (which the interpreter reports as an error rather
//! than returning) contributes a fixed finite penalty, so the loss is always a
//! finite number and never `NaN`.
//!
//! Evaluation performs no wall-clock timing and uses no randomness, so a report
//! is a pure, bit-exact function of the program, the dataset and the limits.

use serde::{Deserialize, Serialize};

use super::cost::{CostReport, estimate_cost};
use super::dataset::Dataset;
use super::interpreter::execute_program;
use super::ir::TensorProgram;
use super::verify::{VerificationLimits, verify_program};

/// Penalty added for a case that cannot be evaluated correctly.
pub const CASE_FAILURE_PENALTY: f64 = 1.0e12;

/// The fitness of a single program on a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitnessReport {
    /// Mean per-case error (mean-squared error, with penalties for failures).
    pub loss: f64,

    /// Number of cases that failed to produce a correct-shaped finite output.
    pub failed_cases: usize,

    /// Deterministic structural cost.
    pub cost: CostReport,

    /// Whether the program could be statically verified for the dataset's
    /// input shapes. A `false` value means `cost` is the worst-case placeholder.
    pub evaluated: bool,
}

/// Evaluate one program against `dataset`.
pub fn evaluate_program(
    program: &TensorProgram,
    dataset: &Dataset,
    limits: VerificationLimits,
) -> FitnessReport {
    let case_count = dataset.len();

    let verified = match verify_program(program, dataset.input_shapes(), limits)
    {
        Ok(verified) => verified,
        Err(_) =>
        {
            return FitnessReport {
                loss: CASE_FAILURE_PENALTY,
                failed_cases: case_count,
                cost: CostReport::unevaluable(program.instructions.len()),
                evaluated: false,
            };
        },
    };

    let cost = estimate_cost(program, &verified);

    let mut total_loss = 0.0f64;
    let mut failed_cases = 0usize;

    for case in dataset.cases()
    {
        match execute_program(program, &case.inputs, limits)
        {
            Ok(result) if result.output.shape == case.expected.shape =>
            {
                total_loss += mean_squared_error(&result.output.data, &case.expected.data);
            },
            _ =>
            {
                total_loss += CASE_FAILURE_PENALTY;
                failed_cases += 1;
            },
        }
    }

    let loss = if case_count == 0
    {
        0.0
    }
    else
    {
        total_loss / case_count as f64
    };

    FitnessReport {
        loss,
        failed_cases,
        cost,
        evaluated: true,
    }
}

/// Evaluate a whole population sequentially, preserving input order.
pub fn evaluate_population(
    programs: &[TensorProgram],
    dataset: &Dataset,
    limits: VerificationLimits,
) -> Vec<FitnessReport> {
    programs
        .iter()
        .map(|program| evaluate_program(program, dataset, limits))
        .collect()
}

/// Evaluate a whole population in parallel with Rayon.
///
/// Each evaluation is an independent, deterministic pure function, and Rayon's
/// indexed `collect` restores input order, so the returned vector is bit-exactly
/// identical to [`evaluate_population`]. Available only with the `rayon` feature.
#[cfg(feature = "rayon")]
pub fn evaluate_population_rayon(
    programs: &[TensorProgram],
    dataset: &Dataset,
    limits: VerificationLimits,
) -> Vec<FitnessReport> {
    use rayon::prelude::*;

    programs
        .par_iter()
        .map(|program| evaluate_program(program, dataset, limits))
        .collect()
}

/// Mean-squared error between two equally sized data slices.
///
/// Both slices come from tensors of identical shape, so their lengths match.
/// Empty tensors have zero error by definition.
fn mean_squared_error(actual: &[f32], expected: &[f32]) -> f64 {
    if actual.is_empty()
    {
        return 0.0;
    }

    let sum: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| {
            let difference = a as f64 - b as f64;
            difference * difference
        })
        .sum();

    sum / actual.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorInstruction;
    use crate::tensor::dataset::TensorCase;
    use scirust_tensor_core::TensorND;

    fn tensor(data: &[f32], shape: &[usize]) -> TensorND {
        TensorND::new(data.to_vec(), shape.to_vec())
    }

    fn identity() -> TensorProgram {
        TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0)
    }

    #[test]
    fn exact_output_has_zero_loss() {
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])],
            tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        )])
        .unwrap();

        let report = evaluate_program(&identity(), &dataset, VerificationLimits::default());
        assert_eq!(report.loss, 0.0);
        assert_eq!(report.failed_cases, 0);
        assert!(report.evaluated);
    }

    #[test]
    fn manually_computed_nonzero_mse() {
        // Identity maps input to output; expected differs in the last element by
        // 1, so squared errors are [0, 0, 0, 1] and the MSE is 1/4 = 0.25.
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0, 2.0, 3.0, 4.0], &[4])],
            tensor(&[1.0, 2.0, 3.0, 5.0], &[4]),
        )])
        .unwrap();

        let report = evaluate_program(&identity(), &dataset, VerificationLimits::default());
        assert_eq!(report.loss, 0.25);
        assert_eq!(report.failed_cases, 0);
    }

    #[test]
    fn shape_mismatch_is_penalised() {
        // Identity yields shape [4] but the case expects [2, 2].
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0, 2.0, 3.0, 4.0], &[4])],
            tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        )])
        .unwrap();

        let report = evaluate_program(&identity(), &dataset, VerificationLimits::default());
        assert_eq!(report.loss, CASE_FAILURE_PENALTY);
        assert_eq!(report.failed_cases, 1);
        assert!(report.evaluated);
    }

    #[test]
    fn invalid_execution_is_penalised() {
        // A finite input and factor whose product overflows to infinity makes
        // execution fail with NonFiniteResult; the case is penalised.
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 1.0e30,
                },
            ],
            1,
        );
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0e30], &[1])],
            tensor(&[0.0], &[1]),
        )])
        .unwrap();

        let report = evaluate_program(&program, &dataset, VerificationLimits::default());
        assert_eq!(report.loss, CASE_FAILURE_PENALTY);
        assert_eq!(report.failed_cases, 1);
        assert!(report.evaluated);
    }

    #[test]
    fn repeated_sequential_evaluation_is_identical() {
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])],
            tensor(&[0.5, 1.0, 1.5, 2.0], &[2, 2]),
        )])
        .unwrap();
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 0.5,
                },
            ],
            1,
        );

        let programs = vec![program.clone(), identity()];
        let first = evaluate_population(&programs, &dataset, VerificationLimits::default());
        let second = evaluate_population(&programs, &dataset, VerificationLimits::default());
        assert_eq!(first, second);
    }

    #[test]
    fn fitness_report_survives_serde() {
        let dataset = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0], &[1])],
            tensor(&[1.0], &[1]),
        )])
        .unwrap();
        let report = evaluate_program(&identity(), &dataset, VerificationLimits::default());

        let json = serde_json::to_string(&report).unwrap();
        let decoded: FitnessReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_matches_sequential_ordered() {
        let dataset = Dataset::new(vec![
            TensorCase::new(
                vec![tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2])],
                tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            ),
            TensorCase::new(
                vec![tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2])],
                tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]),
            ),
        ])
        .unwrap();

        let programs: Vec<TensorProgram> = (0..64)
            .map(|scale| {
                TensorProgram::new(
                    vec![
                        TensorInstruction::Input { input: 0 },
                        TensorInstruction::Scale {
                            src: 0,
                            factor: scale as f32,
                        },
                    ],
                    1,
                )
            })
            .collect();

        let sequential = evaluate_population(&programs, &dataset, VerificationLimits::default());
        let parallel =
            evaluate_population_rayon(&programs, &dataset, VerificationLimits::default());
        assert_eq!(sequential, parallel);
    }
}
