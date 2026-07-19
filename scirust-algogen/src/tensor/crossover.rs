//! Deterministic, structure-aware crossover of tensor programs.
//!
//! Both parents are assumed valid for the same `input_shapes`. The child is
//! built by transplanting the second parent's live computation alongside the
//! first parent's and joining them at a shape-compatible junction. The
//! construction is causal and explicit-output safe by design: the appended
//! subgraph only references registers that precede it, and the merge references
//! only earlier registers. There is no retry loop and no future-reference
//! repair.
//!
//! When the resulting child would violate the resource limits, a documented
//! fallback returns a selected parent unchanged.

use serde::{Deserialize, Serialize};

use super::canonicalize::prune_dead_code;
use super::ir::{TensorInstruction, TensorProgram};
use super::rng::DeterministicRng;
use super::verify::{VerificationLimits, verify_program};

/// The result of a crossover attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CrossoverOutcome {
    /// A valid recombined child was produced.
    Child(TensorProgram),

    /// No valid child could be formed within the limits; a parent is returned.
    ParentUnchanged(TensorProgram),
}

/// Recombine two parent programs into a valid child.
///
/// Both parents must be valid for `input_shapes`. The child embeds the live
/// computation of both parents; when their output shapes match, the child's
/// output is their element-wise sum, otherwise one parent's output is selected
/// deterministically. Returns [`CrossoverOutcome::ParentUnchanged`] if the
/// child would exceed the limits.
pub fn crossover(
    first: &TensorProgram,
    second: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    rng: &mut DeterministicRng,
) -> CrossoverOutcome {
    // Reduce each parent to its live computation before combining.
    let pruned_first = match prune_dead_code(first, input_shapes, limits)
    {
        Ok(program) => program,
        Err(_) => return CrossoverOutcome::ParentUnchanged(first.clone()),
    };
    let pruned_second = match prune_dead_code(second, input_shapes, limits)
    {
        Ok(program) => program,
        Err(_) => return CrossoverOutcome::ParentUnchanged(pruned_first),
    };

    let verified_first = match verify_program(&pruned_first, input_shapes, limits)
    {
        Ok(verified) => verified,
        Err(_) => return CrossoverOutcome::ParentUnchanged(pruned_first),
    };
    let verified_second = match verify_program(&pruned_second, input_shapes, limits)
    {
        Ok(verified) => verified,
        Err(_) => return CrossoverOutcome::ParentUnchanged(pruned_first),
    };

    let offset = pruned_first.instructions.len();
    let mut instructions = pruned_first.instructions.clone();

    // Append the second parent's instructions, shifting their internal register
    // references by `offset`. Inputs carry no source and are left untouched.
    for instruction in &pruned_second.instructions
    {
        instructions.push(shifted(instruction, offset));
    }

    let first_output = pruned_first.output;
    let second_output = offset + pruned_second.output;

    let output = if verified_first.output_shape == verified_second.output_shape
    {
        // Element-wise merge genuinely combines both parents.
        instructions.push(TensorInstruction::Add {
            lhs: first_output,
            rhs: second_output,
        });
        instructions.len() - 1
    }
    else if rng.below(2) == 0
    {
        first_output
    }
    else
    {
        second_output
    };

    let child = TensorProgram::new(instructions, output);

    match verify_program(&child, input_shapes, limits)
    {
        Ok(_) => CrossoverOutcome::Child(child),
        Err(_) => CrossoverOutcome::ParentUnchanged(pruned_first),
    }
}

/// Clone `instruction`, shifting every source register by `offset`.
fn shifted(instruction: &TensorInstruction, offset: usize) -> TensorInstruction {
    match *instruction
    {
        TensorInstruction::Input { input } => TensorInstruction::Input { input },
        TensorInstruction::Add { lhs, rhs } => TensorInstruction::Add {
            lhs: lhs + offset,
            rhs: rhs + offset,
        },
        TensorInstruction::MatMul { lhs, rhs } => TensorInstruction::MatMul {
            lhs: lhs + offset,
            rhs: rhs + offset,
        },
        TensorInstruction::Transpose2d { src } =>
        {
            TensorInstruction::Transpose2d { src: src + offset }
        },
        TensorInstruction::Relu { src } => TensorInstruction::Relu { src: src + offset },
        TensorInstruction::Scale { src, factor } => TensorInstruction::Scale {
            src: src + offset,
            factor,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DeterministicRng, GenerationConfig, OperatorSet, generate};

    fn parents() -> (TensorProgram, TensorProgram) {
        let first = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Relu { src: 2 },
            ],
            3,
        );
        let second = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Scale {
                    src: 2,
                    factor: 2.0,
                },
            ],
            3,
        );
        (first, second)
    }

    #[test]
    fn crossover_is_reproducible() {
        let (a, b) = parents();
        let input_shapes = [vec![2, 3], vec![3, 2]];

        let mut first = DeterministicRng::new(11);
        let mut second = DeterministicRng::new(11);
        let x = crossover(
            &a,
            &b,
            &input_shapes,
            VerificationLimits::default(),
            &mut first,
        );
        let y = crossover(
            &a,
            &b,
            &input_shapes,
            VerificationLimits::default(),
            &mut second,
        );

        assert_eq!(x, y);
    }

    #[test]
    fn crossover_yields_a_valid_result() {
        let (a, b) = parents();
        let input_shapes = [vec![2, 3], vec![3, 2]];

        for seed in 0..100u64
        {
            let mut rng = DeterministicRng::new(seed);
            let outcome = crossover(
                &a,
                &b,
                &input_shapes,
                VerificationLimits::default(),
                &mut rng,
            );
            let program =
                match outcome
                {
                    CrossoverOutcome::Child(program)
                    | CrossoverOutcome::ParentUnchanged(program) => program,
                };
            verify_program(&program, &input_shapes, VerificationLimits::default())
                .unwrap_or_else(|error| panic!("seed {seed} produced invalid crossover: {error}"));
        }
    }

    #[test]
    fn matching_output_shapes_merge_into_a_child() {
        // Both parents output shape [2, 2], so the child should merge them.
        let (a, b) = parents();
        let input_shapes = [vec![2, 3], vec![3, 2]];

        let mut rng = DeterministicRng::new(3);
        let outcome = crossover(
            &a,
            &b,
            &input_shapes,
            VerificationLimits::default(),
            &mut rng,
        );

        match outcome
        {
            CrossoverOutcome::Child(program) =>
            {
                assert!(matches!(
                    program.instructions[program.output],
                    TensorInstruction::Add { .. }
                ));
            },
            CrossoverOutcome::ParentUnchanged(_) => panic!("expected a merged child"),
        }
    }

    #[test]
    fn tiny_parents_do_not_panic() {
        let a = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let b = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let input_shapes = [vec![2, 2]];

        let mut rng = DeterministicRng::new(1);
        let outcome = crossover(
            &a,
            &b,
            &input_shapes,
            VerificationLimits::default(),
            &mut rng,
        );
        let program = match outcome
        {
            CrossoverOutcome::Child(program) | CrossoverOutcome::ParentUnchanged(program) =>
            {
                program
            },
        };
        verify_program(&program, &input_shapes, VerificationLimits::default()).unwrap();
    }

    #[test]
    fn falls_back_to_parent_when_limits_are_tight() {
        // A limit that cannot fit the combined child forces a parent fallback.
        let (a, b) = parents();
        let input_shapes = [vec![2, 3], vec![3, 2]];
        let limits = VerificationLimits {
            max_instructions: 4,
            ..VerificationLimits::default()
        };

        let mut rng = DeterministicRng::new(5);
        let outcome = crossover(&a, &b, &input_shapes, limits, &mut rng);
        assert!(matches!(outcome, CrossoverOutcome::ParentUnchanged(_)));
    }

    #[test]
    fn crossover_of_generated_parents_is_valid() {
        let config = GenerationConfig {
            input_shapes: vec![vec![2, 2], vec![2, 2]],
            min_instructions: 3,
            max_instructions: 6,
            operators: OperatorSet::all(),
            scale_magnitude: 2.0,
        };

        for seed in 0..80u64
        {
            let mut ra = DeterministicRng::new(seed);
            let mut rb = DeterministicRng::new(seed + 1000);
            let a = generate(&config, VerificationLimits::default(), &mut ra).unwrap();
            let b = generate(&config, VerificationLimits::default(), &mut rb).unwrap();

            let mut rc = DeterministicRng::new(seed ^ 0xAAAA);
            let outcome = crossover(
                &a,
                &b,
                &config.input_shapes,
                VerificationLimits::default(),
                &mut rc,
            );
            let program =
                match outcome
                {
                    CrossoverOutcome::Child(program)
                    | CrossoverOutcome::ParentUnchanged(program) => program,
                };
            verify_program(
                &program,
                &config.input_shapes,
                VerificationLimits::default(),
            )
            .unwrap_or_else(|error| panic!("seed {seed} invalid crossover: {error}"));
        }
    }

    #[test]
    fn outcome_survives_serde() {
        let (a, _) = parents();
        let outcome = CrossoverOutcome::Child(a);
        let json = serde_json::to_string(&outcome).unwrap();
        let decoded: CrossoverOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, outcome);
    }
}
