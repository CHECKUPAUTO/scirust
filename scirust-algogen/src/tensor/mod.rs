//! Tensor-program generation foundations.
//!
//! This module defines a serialisable linear IR, backward liveness analysis,
//! deterministic shape inference and pre-execution resource validation.

mod active;
mod canonicalize;
mod cost;
mod crossover;
mod dataset;
mod fitness;
mod generate;
mod interpreter;
mod ir;
mod mutate;
mod population;
mod rng;
mod verify;

pub use active::analyze_active;
pub use canonicalize::prune_dead_code;
pub use cost::{CostReport, estimate_cost};
pub use crossover::{CrossoverOutcome, crossover};
pub use dataset::{Dataset, DatasetError, TensorCase};
pub use fitness::{CASE_FAILURE_PENALTY, FitnessReport, evaluate_population, evaluate_program};
pub use generate::{GenerationConfig, GenerationError, OperatorSet, generate};
pub use interpreter::{ExecutionError, ExecutionResult, execute_program};
pub use ir::{TensorInstruction, TensorProgram};
pub use mutate::{MutationKind, MutationOutcome, mutate};
pub use population::{
    EvolutionConfig, EvolutionError, EvolutionOutcome, GenerationStats, Population,
    TournamentConfig, dominates, elite, evolve, rank, tournament,
};
pub use rng::DeterministicRng;
pub use verify::{ProgramError, VerificationLimits, VerifiedProgram, verify_program};

#[cfg(feature = "rayon")]
pub use fitness::evaluate_population_rayon;

#[cfg(test)]
mod tests {
    use super::*;

    fn default_limits() -> VerificationLimits {
        VerificationLimits::default()
    }

    #[test]
    fn verifies_matmul_transpose_and_add_shapes() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Transpose2d { src: 2 },
                TensorInstruction::Input { input: 2 },
                TensorInstruction::Add { lhs: 3, rhs: 4 },
            ],
            5,
        );

        let verified = verify_program(
            &program,
            &[vec![2, 3], vec![3, 4], vec![4, 2]],
            default_limits(),
        )
        .unwrap();

        assert_eq!(
            verified.register_shapes,
            vec![
                vec![2, 3],
                vec![3, 4],
                vec![2, 4],
                vec![4, 2],
                vec![4, 2],
                vec![4, 2],
            ]
        );
        assert_eq!(verified.output_shape, vec![4, 2]);
        assert_eq!(verified.active, vec![true, true, true, true, true, true]);
        assert_eq!(verified.active_count(), 6);
    }

    #[test]
    fn liveness_excludes_dead_branches() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 1,
                    factor: 3.0,
                },
                TensorInstruction::Relu { src: 2 },
            ],
            4,
        );

        let verified =
            verify_program(&program, &[vec![2, 2], vec![2, 2]], default_limits()).unwrap();

        assert_eq!(verified.active, vec![true, false, true, false, true]);
        assert_eq!(verified.active_count(), 3);
    }

    #[test]
    fn output_does_not_have_to_be_the_final_instruction() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 10.0,
                },
            ],
            1,
        );

        let verified = verify_program(&program, &[vec![3]], default_limits()).unwrap();

        assert_eq!(verified.active, vec![true, true, false]);
        assert_eq!(verified.output_shape, vec![3]);
    }

    #[test]
    fn rejects_self_or_future_dependencies() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Add { lhs: 0, rhs: 2 },
                TensorInstruction::Input { input: 0 },
            ],
            1,
        );

        assert_eq!(
            verify_program(&program, &[vec![2, 2]], default_limits()),
            Err(ProgramError::NonCausalDependency { node: 1, source: 2 })
        );
    }

    #[test]
    fn rejects_addition_shape_mismatch() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::Add { lhs: 0, rhs: 1 },
            ],
            2,
        );

        assert!(matches!(
            verify_program(&program, &[vec![2, 3], vec![3, 2]], default_limits()),
            Err(ProgramError::AddShapeMismatch { node: 2, .. })
        ));
    }

    #[test]
    fn rejects_matmul_inner_dimension_mismatch() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
            ],
            2,
        );

        assert_eq!(
            verify_program(&program, &[vec![2, 3], vec![4, 2]], default_limits()),
            Err(ProgramError::MatMulShapeMismatch {
                node: 2,
                lhs_columns: 3,
                rhs_rows: 4,
            })
        );
    }

    #[test]
    fn rejects_invalid_output_index() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 1);

        assert_eq!(
            verify_program(&program, &[vec![1]], default_limits()),
            Err(ProgramError::OutputOutOfBounds {
                output: 1,
                instructions: 1,
            })
        );
    }

    #[test]
    fn enforces_per_tensor_element_limit() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let limits = VerificationLimits {
            max_elements_per_tensor: 8,
            ..VerificationLimits::default()
        };

        assert_eq!(
            verify_program(&program, &[vec![3, 3]], limits),
            Err(ProgramError::TensorTooLarge {
                node: 0,
                elements: 9,
                maximum: 8,
            })
        );
    }

    #[test]
    fn program_has_a_stable_serde_roundtrip() {
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

        let json = serde_json::to_string(&program).unwrap();
        let decoded: TensorProgram = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, program);
    }
}
