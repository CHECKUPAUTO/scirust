//! Deterministic structural mutation of tensor programs.
//!
//! Every mutation is validity-preserving by *verification*, not by blind index
//! repair: candidate changes are enumerated, each is re-checked with
//! [`verify_program`], and only verifying candidates are eligible. The seeded
//! RNG then selects one candidate from that valid set. If no valid mutation
//! exists the program is returned unchanged with an explicit outcome, so the
//! function never loops indefinitely.

use serde::{Deserialize, Serialize};

use super::ir::{TensorInstruction, TensorProgram};
use super::rng::DeterministicRng;
use super::verify::{VerificationLimits, verify_program};

/// The category of a successful mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationKind {
    RewireSource,
    ReplaceOperator,
    PerturbScale,
    InsertInstruction,
    DeleteInstruction,
    MutateOutput,
}

/// The result of attempting a mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationOutcome {
    /// A valid mutation was applied.
    Mutated {
        kind: MutationKind,
        program: TensorProgram,
    },

    /// No valid mutation was available; the program is unchanged.
    Unchanged,
}

/// Upper bound on enumerated candidates, keeping mutation cost bounded and
/// deterministic on large programs. Candidates are enumerated in a fixed order,
/// so the cap always retains the same prefix.
const MAX_CANDIDATES: usize = 512;

/// How a selected candidate's `Scale` factor must be produced.
#[derive(Clone, Copy)]
struct ScaleAction {
    node: usize,
    /// `None` draws a fresh finite factor; `Some(old)` perturbs `old`.
    base: Option<f32>,
}

/// A verified candidate mutation.
struct Mutant {
    kind: MutationKind,
    program: TensorProgram,
    scale: Option<ScaleAction>,
}

/// Apply a single deterministic mutation to `program`.
///
/// `scale_magnitude` bounds any freshly drawn or perturbed `Scale` factor; the
/// result is always finite. Returns [`MutationOutcome::Unchanged`] when the
/// program is invalid or admits no valid mutation.
pub fn mutate(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    scale_magnitude: f32,
    rng: &mut DeterministicRng,
) -> MutationOutcome {
    let verified = match verify_program(program, input_shapes, limits)
    {
        Ok(verified) => verified,
        Err(_) => return MutationOutcome::Unchanged,
    };
    let shapes = &verified.register_shapes;

    let mut candidates: Vec<Mutant> = Vec::new();

    enumerate_rewire(program, shapes, input_shapes, limits, &mut candidates);
    enumerate_replace(program, shapes, input_shapes, limits, &mut candidates);
    enumerate_perturb_scale(program, input_shapes, limits, &mut candidates);
    enumerate_insert(program, shapes, input_shapes, limits, &mut candidates);
    enumerate_delete(program, input_shapes, limits, &mut candidates);
    enumerate_output(program, input_shapes, limits, &mut candidates);

    if candidates.is_empty()
    {
        return MutationOutcome::Unchanged;
    }

    let index = rng.below(candidates.len());
    let mut chosen = candidates.swap_remove(index);

    if let Some(action) = chosen.scale
    {
        apply_scale_action(&mut chosen.program, action, scale_magnitude, rng);
    }

    MutationOutcome::Mutated {
        kind: chosen.kind,
        program: chosen.program,
    }
}

/// Overwrite the factor of the `Scale` instruction named by `action`.
fn apply_scale_action(
    program: &mut TensorProgram,
    action: ScaleAction,
    magnitude: f32,
    rng: &mut DeterministicRng,
) {
    if let Some(&TensorInstruction::Scale { src, .. }) = program.instructions.get(action.node)
    {
        let factor = match action.base
        {
            None => rng.finite_factor(magnitude),
            Some(old) =>
            {
                let delta = rng.finite_factor(magnitude);
                let perturbed = old + delta;
                if perturbed.is_finite()
                {
                    perturbed
                }
                else
                {
                    delta
                }
            },
        };
        program.instructions[action.node] = TensorInstruction::Scale { src, factor };
    }
}

/// Push a verified candidate, respecting the enumeration cap.
fn push_if_valid(
    out: &mut Vec<Mutant>,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    kind: MutationKind,
    program: TensorProgram,
    scale: Option<ScaleAction>,
) {
    if out.len() >= MAX_CANDIDATES
    {
        return;
    }

    if verify_program(&program, input_shapes, limits).is_ok()
    {
        out.push(Mutant {
            kind,
            program,
            scale,
        });
    }
}

/// Redirect one source of an existing instruction to another earlier register.
fn enumerate_rewire(
    program: &TensorProgram,
    _shapes: &[Vec<usize>],
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    for (node, instruction) in program.instructions.iter().enumerate()
    {
        let slots = source_slots(instruction);
        for (slot, current) in slots.iter().enumerate()
        {
            for alternative in 0..node
            {
                if alternative == *current
                {
                    continue;
                }

                let mut instructions = program.instructions.clone();
                instructions[node] = with_slot(instruction, slot, alternative);
                let candidate = TensorProgram::new(instructions, program.output);
                push_if_valid(
                    out,
                    input_shapes,
                    limits,
                    MutationKind::RewireSource,
                    candidate,
                    None,
                );
            }
        }
    }
}

/// Replace an operator while keeping its operands.
fn enumerate_replace(
    program: &TensorProgram,
    shapes: &[Vec<usize>],
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    for (node, instruction) in program.instructions.iter().enumerate()
    {
        let replacements = operator_replacements(instruction, shapes);
        for replacement in replacements
        {
            let mut instructions = program.instructions.clone();
            let is_scale = matches!(replacement, TensorInstruction::Scale { .. });
            instructions[node] = replacement;
            let candidate = TensorProgram::new(instructions, program.output);
            let scale = is_scale.then_some(ScaleAction { node, base: None });
            push_if_valid(
                out,
                input_shapes,
                limits,
                MutationKind::ReplaceOperator,
                candidate,
                scale,
            );
        }
    }
}

/// Perturb the constant factor of each `Scale` instruction.
fn enumerate_perturb_scale(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    for (node, instruction) in program.instructions.iter().enumerate()
    {
        if let TensorInstruction::Scale { factor, .. } = *instruction
        {
            let candidate = program.clone();
            push_if_valid(
                out,
                input_shapes,
                limits,
                MutationKind::PerturbScale,
                candidate,
                Some(ScaleAction {
                    node,
                    base: Some(factor),
                }),
            );
        }
    }
}

/// Insert a new instruction, remapping later register indices and the output.
fn enumerate_insert(
    program: &TensorProgram,
    shapes: &[Vec<usize>],
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    let length = program.instructions.len();

    for position in 0..=length
    {
        for instruction in insertion_candidates(position, shapes, input_shapes)
        {
            let is_scale = matches!(instruction, TensorInstruction::Scale { .. });
            let candidate = insert_at(program, position, instruction);
            let scale = is_scale.then_some(ScaleAction {
                node: position,
                base: None,
            });
            push_if_valid(
                out,
                input_shapes,
                limits,
                MutationKind::InsertInstruction,
                candidate,
                scale,
            );
        }
    }
}

/// Delete an instruction that nothing else depends on.
fn enumerate_delete(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    let length = program.instructions.len();
    if length <= 1
    {
        return;
    }

    let mut referenced = vec![false; length];
    for instruction in &program.instructions
    {
        instruction.for_each_source(|source| {
            if let Some(flag) = referenced.get_mut(source)
            {
                *flag = true;
            }
        });
    }

    for (index, &is_referenced) in referenced.iter().enumerate()
    {
        if is_referenced || index == program.output
        {
            continue;
        }

        let candidate = delete_at(program, index);
        push_if_valid(
            out,
            input_shapes,
            limits,
            MutationKind::DeleteInstruction,
            candidate,
            None,
        );
    }
}

/// Move the explicit output to another register.
fn enumerate_output(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
    out: &mut Vec<Mutant>,
) {
    for output in 0..program.instructions.len()
    {
        if output == program.output
        {
            continue;
        }

        let candidate = TensorProgram::new(program.instructions.clone(), output);
        push_if_valid(
            out,
            input_shapes,
            limits,
            MutationKind::MutateOutput,
            candidate,
            None,
        );
    }
}

/// Operator replacements that keep the instruction's operands.
fn operator_replacements(
    instruction: &TensorInstruction,
    shapes: &[Vec<usize>],
) -> Vec<TensorInstruction> {
    match *instruction
    {
        TensorInstruction::Input { .. } => Vec::new(),

        TensorInstruction::Relu { src }
        | TensorInstruction::Transpose2d { src }
        | TensorInstruction::Scale { src, .. } =>
        {
            let mut options = Vec::new();
            if !matches!(instruction, TensorInstruction::Relu { .. })
            {
                options.push(TensorInstruction::Relu { src });
            }
            if !matches!(instruction, TensorInstruction::Scale { .. })
            {
                options.push(TensorInstruction::Scale { src, factor: 1.0 });
            }
            if !matches!(instruction, TensorInstruction::Transpose2d { .. })
                && shapes[src].len() == 2
            {
                options.push(TensorInstruction::Transpose2d { src });
            }
            options
        },

        TensorInstruction::Add { lhs, rhs } => vec![TensorInstruction::MatMul { lhs, rhs }],

        TensorInstruction::MatMul { lhs, rhs } => vec![TensorInstruction::Add { lhs, rhs }],
    }
}

/// New instructions that reference only registers strictly before `position`.
fn insertion_candidates(
    position: usize,
    shapes: &[Vec<usize>],
    input_shapes: &[Vec<usize>],
) -> Vec<TensorInstruction> {
    let mut candidates = Vec::new();

    for input in 0..input_shapes.len()
    {
        candidates.push(TensorInstruction::Input { input });
    }

    for (src, shape) in shapes.iter().enumerate().take(position)
    {
        candidates.push(TensorInstruction::Relu { src });
        candidates.push(TensorInstruction::Scale { src, factor: 1.0 });
        if shape.len() == 2
        {
            candidates.push(TensorInstruction::Transpose2d { src });
        }
    }

    for lhs in 0..position
    {
        for rhs in 0..position
        {
            if rhs >= lhs && shapes[lhs] == shapes[rhs]
            {
                candidates.push(TensorInstruction::Add { lhs, rhs });
            }
            if shapes[lhs].len() == 2 && shapes[rhs].len() == 2 && shapes[lhs][1] == shapes[rhs][0]
            {
                candidates.push(TensorInstruction::MatMul { lhs, rhs });
            }
        }
    }

    candidates
}

/// Current source values of an instruction, in slot order.
fn source_slots(instruction: &TensorInstruction) -> Vec<usize> {
    match *instruction
    {
        TensorInstruction::Input { .. } => Vec::new(),
        TensorInstruction::Add { lhs, rhs } | TensorInstruction::MatMul { lhs, rhs } =>
        {
            vec![lhs, rhs]
        },
        TensorInstruction::Transpose2d { src }
        | TensorInstruction::Relu { src }
        | TensorInstruction::Scale { src, .. } => vec![src],
    }
}

/// Rebuild `instruction` with source slot `slot` set to `value`.
fn with_slot(instruction: &TensorInstruction, slot: usize, value: usize) -> TensorInstruction {
    match *instruction
    {
        TensorInstruction::Input { input } => TensorInstruction::Input { input },
        TensorInstruction::Add { lhs, rhs } =>
        {
            if slot == 0
            {
                TensorInstruction::Add { lhs: value, rhs }
            }
            else
            {
                TensorInstruction::Add { lhs, rhs: value }
            }
        },
        TensorInstruction::MatMul { lhs, rhs } =>
        {
            if slot == 0
            {
                TensorInstruction::MatMul { lhs: value, rhs }
            }
            else
            {
                TensorInstruction::MatMul { lhs, rhs: value }
            }
        },
        TensorInstruction::Transpose2d { .. } => TensorInstruction::Transpose2d { src: value },
        TensorInstruction::Relu { .. } => TensorInstruction::Relu { src: value },
        TensorInstruction::Scale { factor, .. } => TensorInstruction::Scale { src: value, factor },
    }
}

/// Apply `map` to every source register of `instruction`.
fn map_sources(instruction: &mut TensorInstruction, map: impl Fn(usize) -> usize) {
    match instruction
    {
        TensorInstruction::Input { .. } =>
        {},
        TensorInstruction::Add { lhs, rhs } | TensorInstruction::MatMul { lhs, rhs } =>
        {
            *lhs = map(*lhs);
            *rhs = map(*rhs);
        },
        TensorInstruction::Transpose2d { src }
        | TensorInstruction::Relu { src }
        | TensorInstruction::Scale { src, .. } =>
        {
            *src = map(*src);
        },
    }
}

/// Insert `new_instruction` at `position`, shifting later indices up by one.
///
/// `new_instruction` must reference only registers strictly before `position`.
fn insert_at(
    program: &TensorProgram,
    position: usize,
    new_instruction: TensorInstruction,
) -> TensorProgram {
    let mut instructions = Vec::with_capacity(program.instructions.len() + 1);

    // Instructions before `position` reference only earlier registers, so their
    // sources are unaffected.
    instructions.extend_from_slice(&program.instructions[..position]);
    instructions.push(new_instruction);

    for instruction in &program.instructions[position..]
    {
        let mut shifted = instruction.clone();
        map_sources(&mut shifted, |source| {
            if source >= position
            {
                source + 1
            }
            else
            {
                source
            }
        });
        instructions.push(shifted);
    }

    let output = if program.output >= position
    {
        program.output + 1
    }
    else
    {
        program.output
    };

    TensorProgram::new(instructions, output)
}

/// Remove the instruction at `index`, shifting later indices down by one.
///
/// The caller must guarantee that no surviving instruction and not the output
/// references `index`.
fn delete_at(program: &TensorProgram, index: usize) -> TensorProgram {
    let mut instructions = Vec::with_capacity(program.instructions.len() - 1);

    for (position, instruction) in program.instructions.iter().enumerate()
    {
        if position == index
        {
            continue;
        }

        let mut shifted = instruction.clone();
        map_sources(&mut shifted, |source| {
            if source > index { source - 1 } else { source }
        });
        instructions.push(shifted);
    }

    let output = if program.output > index
    {
        program.output - 1
    }
    else
    {
        program.output
    };

    TensorProgram::new(instructions, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        DeterministicRng, GenerationConfig, OperatorSet, generate, verify_program,
    };

    fn find_first_kind(
        program: &TensorProgram,
        input_shapes: &[Vec<usize>],
        target: MutationKind,
    ) -> Option<TensorProgram> {
        for seed in 0..2_000u64
        {
            let mut rng = DeterministicRng::new(seed);
            if let MutationOutcome::Mutated { kind, program } = mutate(
                program,
                input_shapes,
                VerificationLimits::default(),
                2.0,
                &mut rng,
            )
            {
                if kind == target
                {
                    return Some(program);
                }
            }
        }
        None
    }

    #[test]
    fn insert_at_remaps_sources_and_output() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::Add { lhs: 0, rhs: 1 },
            ],
            2,
        );

        let inserted = insert_at(&program, 1, TensorInstruction::Input { input: 0 });

        assert_eq!(
            inserted,
            TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Input { input: 1 },
                    TensorInstruction::Add { lhs: 0, rhs: 2 },
                ],
                3,
            )
        );
    }

    #[test]
    fn delete_at_remaps_sources_and_output() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
                TensorInstruction::Relu { src: 0 },
            ],
            2,
        );

        let deleted = delete_at(&program, 1);

        assert_eq!(
            deleted,
            TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Relu { src: 0 },
                ],
                1,
            )
        );
    }

    #[test]
    fn mutation_is_reproducible() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Relu { src: 2 },
            ],
            3,
        );
        let input_shapes = [vec![2, 3], vec![3, 2]];

        let mut first = DeterministicRng::new(42);
        let mut second = DeterministicRng::new(42);
        let a = mutate(
            &program,
            &input_shapes,
            VerificationLimits::default(),
            2.0,
            &mut first,
        );
        let b = mutate(
            &program,
            &input_shapes,
            VerificationLimits::default(),
            2.0,
            &mut second,
        );

        assert_eq!(a, b);
    }

    #[test]
    fn every_mutation_is_valid() {
        let config = GenerationConfig {
            input_shapes: vec![vec![2, 2], vec![2, 2]],
            min_instructions: 3,
            max_instructions: 8,
            operators: OperatorSet::all(),
            scale_magnitude: 2.0,
        };

        for seed in 0..200u64
        {
            let mut gen_rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut gen_rng).unwrap();

            let mut mut_rng = DeterministicRng::new(seed ^ 0x5555);
            if let MutationOutcome::Mutated {
                program: mutated, ..
            } = mutate(
                &program,
                &config.input_shapes,
                VerificationLimits::default(),
                2.0,
                &mut mut_rng,
            )
            {
                verify_program(
                    &mutated,
                    &config.input_shapes,
                    VerificationLimits::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("seed {seed} mutated into invalid program: {error}")
                });

                for instruction in &mutated.instructions
                {
                    if let TensorInstruction::Scale { factor, .. } = *instruction
                    {
                        assert!(factor.is_finite(), "seed {seed} produced non-finite factor");
                    }
                }
            }
        }
    }

    #[test]
    fn single_instruction_program_mutates_safely() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let input_shapes = [vec![2, 2]];

        // A single-instruction program cannot change its output, but insertion
        // is still possible; either way the result must be valid and no panic
        // may occur.
        let mut rng = DeterministicRng::new(7);
        match mutate(
            &program,
            &input_shapes,
            VerificationLimits::default(),
            2.0,
            &mut rng,
        )
        {
            MutationOutcome::Mutated {
                program: mutated, ..
            } =>
            {
                verify_program(&mutated, &input_shapes, VerificationLimits::default()).unwrap();
            },
            MutationOutcome::Unchanged =>
            {},
        }
    }

    #[test]
    fn output_mutation_moves_the_output_register() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
            ],
            1,
        );
        let input_shapes = [vec![2, 2]];

        let mutated = find_first_kind(&program, &input_shapes, MutationKind::MutateOutput).unwrap();
        assert_ne!(mutated.output, program.output);
        assert_eq!(mutated.instructions, program.instructions);
        verify_program(&mutated, &input_shapes, VerificationLimits::default()).unwrap();
    }

    #[test]
    fn insertion_and_deletion_kinds_are_reachable_and_valid() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
            ],
            1,
        );
        let input_shapes = [vec![2, 2]];

        let inserted =
            find_first_kind(&program, &input_shapes, MutationKind::InsertInstruction).unwrap();
        assert_eq!(inserted.instructions.len(), program.instructions.len() + 1);
        verify_program(&inserted, &input_shapes, VerificationLimits::default()).unwrap();

        let deleted =
            find_first_kind(&program, &input_shapes, MutationKind::DeleteInstruction).unwrap();
        assert_eq!(deleted.instructions.len(), program.instructions.len() - 1);
        verify_program(&deleted, &input_shapes, VerificationLimits::default()).unwrap();
    }

    #[test]
    fn outcome_types_survive_serde() {
        let outcome = MutationOutcome::Mutated {
            kind: MutationKind::PerturbScale,
            program: TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Scale {
                        src: 0,
                        factor: 1.5,
                    },
                ],
                1,
            ),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let decoded: MutationOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, outcome);

        let unchanged = MutationOutcome::Unchanged;
        let json = serde_json::to_string(&unchanged).unwrap();
        let decoded: MutationOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, unchanged);
    }
}
