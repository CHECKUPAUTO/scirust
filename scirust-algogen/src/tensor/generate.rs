//! Deterministic, shape-aware generation of valid tensor programs.
//!
//! Programs are built so that they are accepted by [`verify_program`] *by
//! construction*: every emitted instruction reads only earlier registers, only
//! shape-compatible operations are chosen, and each register's inferred shape
//! and resource cost are tracked as the program grows. Invalid programs are
//! never produced and rejected after the fact.

use serde::{Deserialize, Serialize};

use super::ir::{TensorInstruction, TensorProgram};
use super::rng::DeterministicRng;
use super::verify::{ProgramError, VerificationLimits, verify_program};

/// The set of operators a generator may use. `Input` is always available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSet {
    pub add: bool,
    pub matmul: bool,
    pub transpose: bool,
    pub relu: bool,
    pub scale: bool,
}

impl OperatorSet {
    /// Enable every operator.
    pub fn all() -> Self {
        Self {
            add: true,
            matmul: true,
            transpose: true,
            relu: true,
            scale: true,
        }
    }
}

impl Default for OperatorSet {
    fn default() -> Self {
        Self::all()
    }
}

/// Configuration for a single generation run.
///
/// `VerificationLimits` are supplied separately to [`generate`] because they
/// are a Phase 1 type; this keeps the configuration self-describing and
/// serialisable on its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Shapes of the tensors that will be supplied to the program.
    pub input_shapes: Vec<Vec<usize>>,

    /// Minimum number of instructions (clamped to at least one).
    pub min_instructions: usize,

    /// Maximum number of instructions.
    pub max_instructions: usize,

    /// Which operators may be emitted.
    pub operators: OperatorSet,

    /// Magnitude bound for generated `Scale` factors; always finite.
    pub scale_magnitude: f32,
}

impl GenerationConfig {
    /// A permissive configuration for the given input shapes.
    pub fn new(input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            input_shapes,
            min_instructions: 1,
            max_instructions: 8,
            operators: OperatorSet::all(),
            scale_magnitude: 4.0,
        }
    }
}

/// A deterministic generation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationError {
    /// No input shapes were supplied, so no register can ever be produced.
    EmptyInputSet,

    /// Every supplied input shape violates the resource limits.
    NoUsableInputs,

    /// `min_instructions` (after clamping to at least one) exceeds the
    /// effective maximum.
    InvalidSizeBounds { minimum: usize, maximum: usize },

    /// Generation ran out of shape-compatible, budget-fitting instructions
    /// before reaching the minimum size.
    CannotReachMinimum { produced: usize, minimum: usize },

    /// The finished program unexpectedly failed verification. This indicates a
    /// generator defect and never a malformed request; it is surfaced rather
    /// than panicking.
    Verification(ProgramError),
}

/// A candidate instruction together with its inferred shape and element cost.
struct Candidate {
    instruction: TensorInstruction,
    shape: Vec<usize>,
    elements: usize,
}

/// Generate a valid tensor program from `config` using the deterministic `rng`.
pub fn generate(
    config: &GenerationConfig,
    limits: VerificationLimits,
    rng: &mut DeterministicRng,
) -> Result<TensorProgram, GenerationError> {
    if config.input_shapes.is_empty()
    {
        return Err(GenerationError::EmptyInputSet);
    }

    let effective_max = config.max_instructions.min(limits.max_instructions);
    let effective_min = config.min_instructions.max(1);

    if effective_min > effective_max
    {
        return Err(GenerationError::InvalidSizeBounds {
            minimum: effective_min,
            maximum: effective_max,
        });
    }

    // Inputs whose shape individually satisfies the per-tensor limits.
    let usable_inputs: Vec<usize> = config
        .input_shapes
        .iter()
        .enumerate()
        .filter_map(|(index, shape)| shape_elements(shape, limits).map(|_| index))
        .collect();

    if usable_inputs.is_empty()
    {
        return Err(GenerationError::NoUsableInputs);
    }

    let mut instructions: Vec<TensorInstruction> = Vec::new();
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    let mut total_elements = 0usize;

    // Seed the program with the usable inputs, respecting the instruction count
    // and the cumulative element budget.
    for &input in &usable_inputs
    {
        if instructions.len() >= effective_max
        {
            break;
        }

        let shape = &config.input_shapes[input];
        let elements = shape_elements(shape, limits).expect("usable input is within limits");

        match total_elements.checked_add(elements)
        {
            Some(next) if next <= limits.max_total_register_elements =>
            {
                total_elements = next;
                instructions.push(TensorInstruction::Input { input });
                shapes.push(shape.clone());
            },
            _ => break,
        }
    }

    // Choose a target length in `[effective_min, effective_max]`.
    let headroom = effective_max - instructions.len();
    let extra = if headroom == 0
    {
        0
    }
    else
    {
        rng.below(headroom + 1)
    };
    let target_len = (instructions.len() + extra)
        .max(effective_min)
        .min(effective_max);

    while instructions.len() < target_len
    {
        let candidates = enumerate_candidates(
            &config.input_shapes,
            &usable_inputs,
            &shapes,
            &config.operators,
            limits,
            total_elements,
        );

        if candidates.is_empty()
        {
            break;
        }

        let chosen = &candidates[rng.below(candidates.len())];
        let instruction = materialize(&chosen.instruction, config.scale_magnitude, rng);

        total_elements += chosen.elements;
        shapes.push(chosen.shape.clone());
        instructions.push(instruction);
    }

    if instructions.len() < effective_min
    {
        return Err(GenerationError::CannotReachMinimum {
            produced: instructions.len(),
            minimum: effective_min,
        });
    }

    let output = rng.below(instructions.len());
    let program = TensorProgram::new(instructions, output);

    // Defensive: the construction guarantees validity, but confirm it.
    verify_program(&program, &config.input_shapes, limits)
        .map_err(GenerationError::Verification)?;

    Ok(program)
}

/// Replace a `Scale` placeholder factor with a freshly drawn finite value.
fn materialize(
    instruction: &TensorInstruction,
    magnitude: f32,
    rng: &mut DeterministicRng,
) -> TensorInstruction {
    match *instruction
    {
        TensorInstruction::Scale { src, .. } => TensorInstruction::Scale {
            src,
            factor: rng.finite_factor(magnitude),
        },
        ref other => other.clone(),
    }
}

/// Enumerate every valid next instruction given the current registers.
///
/// The order is fixed and deterministic: input reuse, then unary operations
/// per register, then binary operations over register pairs. Each candidate is
/// already checked against the rank, per-tensor and cumulative element limits.
fn enumerate_candidates(
    input_shapes: &[Vec<usize>],
    usable_inputs: &[usize],
    shapes: &[Vec<usize>],
    operators: &OperatorSet,
    limits: VerificationLimits,
    total_elements: usize,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();

    let mut consider = |instruction: TensorInstruction, shape: Vec<usize>| {
        if let Some(elements) = shape_elements(&shape, limits)
        {
            if let Some(next) = total_elements.checked_add(elements)
            {
                if next <= limits.max_total_register_elements
                {
                    candidates.push(Candidate {
                        instruction,
                        shape,
                        elements,
                    });
                }
            }
        }
    };

    // Input reuse keeps the program extensible even when no operator applies.
    for &input in usable_inputs
    {
        consider(
            TensorInstruction::Input { input },
            input_shapes[input].clone(),
        );
    }

    // Unary operations over every register.
    for (src, shape) in shapes.iter().enumerate()
    {
        if operators.relu
        {
            consider(TensorInstruction::Relu { src }, shape.clone());
        }

        if operators.scale
        {
            consider(TensorInstruction::Scale { src, factor: 1.0 }, shape.clone());
        }

        if operators.transpose && shape.len() == 2
        {
            consider(
                TensorInstruction::Transpose2d { src },
                vec![shape[1], shape[0]],
            );
        }
    }

    // Binary operations over register pairs.
    for lhs in 0..shapes.len()
    {
        for rhs in 0..shapes.len()
        {
            if operators.add && rhs >= lhs && shapes[lhs] == shapes[rhs]
            {
                consider(TensorInstruction::Add { lhs, rhs }, shapes[lhs].clone());
            }

            if operators.matmul
                && shapes[lhs].len() == 2
                && shapes[rhs].len() == 2
                && shapes[lhs][1] == shapes[rhs][0]
            {
                consider(
                    TensorInstruction::MatMul { lhs, rhs },
                    vec![shapes[lhs][0], shapes[rhs][1]],
                );
            }
        }
    }

    candidates
}

/// Number of elements described by `shape`, or `None` if the shape violates the
/// rank limit, overflows, or exceeds the per-tensor element limit. Mirrors the
/// checks performed by [`verify_program`].
fn shape_elements(shape: &[usize], limits: VerificationLimits) -> Option<usize> {
    if shape.len() > limits.max_rank
    {
        return None;
    }

    let elements = shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))?;

    if elements > limits.max_elements_per_tensor
    {
        return None;
    }

    Some(elements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::verify_program;

    fn config() -> GenerationConfig {
        GenerationConfig {
            input_shapes: vec![vec![2, 3], vec![3, 2]],
            min_instructions: 3,
            max_instructions: 8,
            operators: OperatorSet::all(),
            scale_magnitude: 4.0,
        }
    }

    #[test]
    fn identical_seeds_produce_identical_programs() {
        let config = config();
        let mut first = DeterministicRng::new(0x1234);
        let mut second = DeterministicRng::new(0x1234);

        let a = generate(&config, VerificationLimits::default(), &mut first).unwrap();
        let b = generate(&config, VerificationLimits::default(), &mut second).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_diversify_a_batch() {
        let config = config();
        let programs: Vec<_> = (0..16u64)
            .map(|seed| {
                let mut rng = DeterministicRng::new(seed);
                generate(&config, VerificationLimits::default(), &mut rng).unwrap()
            })
            .collect();

        let distinct = programs
            .iter()
            .enumerate()
            .filter(|(index, program)| programs[..*index].iter().all(|earlier| earlier != *program))
            .count();
        assert!(
            distinct > 1,
            "expected diversity, got {distinct} distinct programs"
        );
    }

    #[test]
    fn every_generated_program_verifies() {
        let config = config();
        for seed in 0..200u64
        {
            let mut rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut rng).unwrap();
            verify_program(
                &program,
                &config.input_shapes,
                VerificationLimits::default(),
            )
            .unwrap_or_else(|error| panic!("seed {seed} produced invalid program: {error}"));
        }
    }

    #[test]
    fn generated_size_is_within_bounds() {
        let config = config();
        for seed in 0..200u64
        {
            let mut rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut rng).unwrap();
            let length = program.instructions.len();
            assert!(
                (config.min_instructions..=config.max_instructions).contains(&length),
                "seed {seed} produced length {length} outside bounds"
            );
        }
    }

    #[test]
    fn generated_scale_factors_are_always_finite() {
        let config = config();
        for seed in 0..200u64
        {
            let mut rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut rng).unwrap();
            for instruction in &program.instructions
            {
                if let TensorInstruction::Scale { factor, .. } = *instruction
                {
                    assert!(
                        factor.is_finite(),
                        "seed {seed} generated non-finite factor"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_impossible_constraints() {
        let mut rng = DeterministicRng::new(1);
        let limits = VerificationLimits::default();

        let empty = GenerationConfig {
            input_shapes: Vec::new(),
            ..config()
        };
        assert_eq!(
            generate(&empty, limits, &mut rng),
            Err(GenerationError::EmptyInputSet)
        );

        let inverted = GenerationConfig {
            min_instructions: 9,
            max_instructions: 4,
            ..config()
        };
        assert_eq!(
            generate(&inverted, limits, &mut rng),
            Err(GenerationError::InvalidSizeBounds {
                minimum: 9,
                maximum: 4,
            })
        );
    }

    #[test]
    fn honours_operator_restrictions() {
        // With only Relu enabled, no binary or transpose instruction may appear.
        let config = GenerationConfig {
            operators: OperatorSet {
                add: false,
                matmul: false,
                transpose: false,
                relu: true,
                scale: false,
            },
            ..config()
        };

        for seed in 0..64u64
        {
            let mut rng = DeterministicRng::new(seed);
            let program = generate(&config, VerificationLimits::default(), &mut rng).unwrap();
            for instruction in &program.instructions
            {
                assert!(matches!(
                    instruction,
                    TensorInstruction::Input { .. } | TensorInstruction::Relu { .. }
                ));
            }
        }
    }

    #[test]
    fn config_and_operator_set_survive_serde() {
        let config = config();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, config);

        let operators = OperatorSet::all();
        let json = serde_json::to_string(&operators).unwrap();
        let decoded: OperatorSet = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, operators);
    }
}
