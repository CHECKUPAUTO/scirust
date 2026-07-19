//! Dead-code elimination for tensor programs.
//!
//! Pruning keeps exactly the transitive dependencies of the explicit output,
//! preserves their original relative order, and remaps every register index and
//! the output register. Because the interpreter already executes only the
//! active instructions in ascending index order, pruning preserves the observed
//! numerical result bit-for-bit.

use super::active::analyze_active;
use super::ir::TensorProgram;
use super::verify::{ProgramError, VerificationLimits, verify_program};

/// Remove instructions that do not contribute to the program's output.
///
/// Returns a program that has been re-verified against `input_shapes` and
/// `limits`. The operation is idempotent: pruning an already-pruned program
/// returns an equal program.
pub fn prune_dead_code(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
) -> Result<TensorProgram, ProgramError> {
    // Establish validity (and reject a malformed output index) up front.
    verify_program(program, input_shapes, limits)?;

    let active = analyze_active(program);

    // Map each retained old index to its new, compacted index.
    let mut remap = vec![usize::MAX; program.instructions.len()];
    let mut next_index = 0usize;
    for (old_index, &is_active) in active.iter().enumerate()
    {
        if is_active
        {
            remap[old_index] = next_index;
            next_index += 1;
        }
    }

    let mut instructions = Vec::with_capacity(next_index);
    for (old_index, instruction) in program.instructions.iter().enumerate()
    {
        if !active[old_index]
        {
            continue;
        }

        // Every source of an active instruction is itself active (liveness is
        // closed under dependencies), so each source has a valid remap entry.
        let mut rewired = instruction.clone();
        remap_sources(&mut rewired, &remap);
        instructions.push(rewired);
    }

    let output = remap[program.output];
    let pruned = TensorProgram::new(instructions, output);

    // The result is valid by construction; confirm it and return verified.
    verify_program(&pruned, input_shapes, limits)?;

    Ok(pruned)
}

/// Rewrite each source register of `instruction` through `remap`.
fn remap_sources(instruction: &mut super::ir::TensorInstruction, remap: &[usize]) {
    use super::ir::TensorInstruction as Instr;

    match instruction
    {
        Instr::Input { .. } =>
        {},
        Instr::Add { lhs, rhs } | Instr::MatMul { lhs, rhs } =>
        {
            *lhs = remap[*lhs];
            *rhs = remap[*rhs];
        },
        Instr::Transpose2d { src } | Instr::Relu { src } | Instr::Scale { src, .. } =>
        {
            *src = remap[*src];
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        DeterministicRng, GenerationConfig, OperatorSet, TensorInstruction, execute_program,
        generate,
    };
    use scirust_tensor_core::TensorND;

    /// A program with dead code both in the middle and at the tail; a kept
    /// instruction's source and the output both require remapping.
    fn program_with_dead_code() -> TensorProgram {
        TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 }, // 0 (kept)
                TensorInstruction::Scale {
                    // 1 (dead)
                    src: 0,
                    factor: 2.0,
                },
                TensorInstruction::Relu { src: 0 }, // 2 (kept, src stays 0)
                TensorInstruction::Scale {
                    // 3 (kept, src 2 -> 1, output)
                    src: 2,
                    factor: 3.0,
                },
                TensorInstruction::Relu { src: 1 }, // 4 (dead, tail)
            ],
            3,
        )
    }

    #[test]
    fn prune_removes_dead_and_remaps() {
        let program = program_with_dead_code();
        let pruned =
            prune_dead_code(&program, &[vec![2, 2]], VerificationLimits::default()).unwrap();

        assert_eq!(
            pruned,
            TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Relu { src: 0 },
                    TensorInstruction::Scale {
                        src: 1,
                        factor: 3.0
                    },
                ],
                2,
            )
        );
    }

    #[test]
    fn prune_is_idempotent() {
        let program = program_with_dead_code();
        let once = prune_dead_code(&program, &[vec![2, 2]], VerificationLimits::default()).unwrap();
        let twice = prune_dead_code(&once, &[vec![2, 2]], VerificationLimits::default()).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn prune_preserves_execution_bit_for_bit() {
        let program = program_with_dead_code();
        let input_shapes = [vec![2, 2]];
        let pruned =
            prune_dead_code(&program, &input_shapes, VerificationLimits::default()).unwrap();

        let inputs = vec![TensorND::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2])];

        let original = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();
        let compacted = execute_program(&pruned, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(original.output.shape, compacted.output.shape);
        for (left, right) in original.output.data.iter().zip(&compacted.output.data)
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn prune_preserves_semantics_for_generated_programs() {
        let config = GenerationConfig {
            input_shapes: vec![vec![2, 2], vec![2, 2]],
            min_instructions: 4,
            max_instructions: 10,
            operators: OperatorSet::all(),
            scale_magnitude: 2.0,
        };
        let inputs = vec![
            TensorND::new(vec![0.5, -0.5, 1.0, -1.0], vec![2, 2]),
            TensorND::new(vec![0.25, 0.75, -0.25, -0.75], vec![2, 2]),
        ];

        for seed in 0..80u64
        {
            let mut rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut rng).unwrap();

            let original = execute_program(&program, &inputs, VerificationLimits::default());
            let pruned = prune_dead_code(
                &program,
                &config.input_shapes,
                VerificationLimits::default(),
            )
            .unwrap();
            let compacted = execute_program(&pruned, &inputs, VerificationLimits::default());

            match (original, compacted)
            {
                (Ok(a), Ok(b)) =>
                {
                    assert_eq!(a.output.shape, b.output.shape, "seed {seed}");
                    for (left, right) in a.output.data.iter().zip(&b.output.data)
                    {
                        assert_eq!(left.to_bits(), right.to_bits(), "seed {seed}");
                    }
                },
                (Err(a), Err(b)) => assert_eq!(a, b, "seed {seed}"),
                (a, b) => panic!("seed {seed}: divergent execution outcomes: {a:?} vs {b:?}"),
            }
        }
    }
}
