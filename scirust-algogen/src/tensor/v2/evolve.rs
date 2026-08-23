//! Validity-aware deterministic mutation and crossover for V2 programs.
//!
//! These operators deliberately favor a smaller set of high-yield structural
//! edits over blind syntax splicing. Every returned child has passed the V2
//! verifier. Failed proposals are counted, never interpreted.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::tensor::DeterministicRng;

use super::canonical::canonical_bytes;
use super::ir::{Op, Ref, ResearchProgram};
use super::types::{ScalarValue, ValueType};
use super::verify::{
    ProgramError, SectionKind, VerificationLimits, VerifiedProgram, verify_program,
};

/// Mutation families enabled independently by a search curriculum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MutationKind {
    ReplaceConstant,
    ReplaceOperation,
    ReplaceOperand,
    ModifyReductionAxis,
    RebindStateUpdate,
    RebindOutput,
}

/// Bounded deterministic mutation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutationConfig {
    pub allowed: Vec<MutationKind>,
    pub constants: Vec<ScalarValue>,
    pub max_proposals: usize,
    pub verification_limits: VerificationLimits,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            allowed: vec![
                MutationKind::ReplaceConstant,
                MutationKind::ReplaceOperation,
                MutationKind::ReplaceOperand,
                MutationKind::ModifyReductionAxis,
                MutationKind::RebindStateUpdate,
                MutationKind::RebindOutput,
            ],
            constants: vec![
                ScalarValue::F32(-1.0),
                ScalarValue::F32(0.0),
                ScalarValue::F32(1.0),
                ScalarValue::F64(-1.0),
                ScalarValue::F64(0.0),
                ScalarValue::F64(1.0),
                ScalarValue::Bool(false),
                ScalarValue::Bool(true),
            ],
            max_proposals: 2_048,
            verification_limits: VerificationLimits::default(),
        }
    }
}

/// Location and class of one applied mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedMutation {
    pub kind: MutationKind,
    pub section: SectionKind,
    /// Node or binding index, according to `kind`.
    pub index: usize,
}

/// Successful mutation plus deterministic proposal diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct MutationResult {
    pub program: ResearchProgram,
    pub applied: AppliedMutation,
    pub proposed: usize,
    pub rejected: usize,
    pub duplicates: usize,
}

/// Mutation failure. The parent is never modified in place.
#[derive(Debug, Clone, PartialEq)]
pub enum MutationError {
    InvalidParent(ProgramError),
    InvalidConfiguration(&'static str),
    NoValidMutation { proposed: usize, rejected: usize },
}

impl std::fmt::Display for MutationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::InvalidParent(error) => write!(formatter, "invalid mutation parent: {error}"),
            Self::InvalidConfiguration(reason) =>
            {
                write!(formatter, "invalid mutation configuration: {reason}")
            },
            Self::NoValidMutation { proposed, rejected } => write!(
                formatter,
                "no valid mutation among {proposed} bounded proposals ({rejected} rejected)"
            ),
        }
    }
}

impl std::error::Error for MutationError {}

#[derive(Clone)]
struct MutationCandidate {
    program: ResearchProgram,
    applied: AppliedMutation,
}

/// Enumerate bounded compatible edits, verify them, and select by stable RNG.
pub fn mutate_program(
    parent: &ResearchProgram,
    config: &MutationConfig,
    seed: u64,
) -> Result<MutationResult, MutationError> {
    let verified =
        verify_program(parent, config.verification_limits).map_err(MutationError::InvalidParent)?;
    if config.max_proposals == 0
    {
        return Err(MutationError::InvalidConfiguration(
            "max_proposals must be non-zero",
        ));
    }

    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    let mut proposed = 0usize;
    let mut rejected = 0usize;
    let mut duplicates = 0usize;

    for kind in &config.allowed
    {
        if proposed >= config.max_proposals
        {
            break;
        }
        match kind
        {
            MutationKind::ReplaceConstant
            | MutationKind::ReplaceOperation
            | MutationKind::ReplaceOperand
            | MutationKind::ModifyReductionAxis =>
            {
                for section in [SectionKind::Init, SectionKind::Step, SectionKind::Finalize]
                {
                    let node_count = section_ops(parent, section).len();
                    for node in 0..node_count
                    {
                        let variants = node_variants(
                            parent,
                            &verified,
                            section,
                            node,
                            *kind,
                            &config.constants,
                        );
                        for variant in variants
                        {
                            if proposed >= config.max_proposals
                            {
                                break;
                            }
                            proposed += 1;
                            let mut child = parent.clone();
                            section_ops_mut(&mut child, section)[node] = variant;
                            consider_candidate(
                                child,
                                AppliedMutation {
                                    kind: *kind,
                                    section,
                                    index: node,
                                },
                                config.verification_limits,
                                &mut candidates,
                                &mut seen,
                                &mut rejected,
                                &mut duplicates,
                            );
                        }
                    }
                }
            },
            MutationKind::RebindStateUpdate =>
            {
                for (slot, expected) in parent.state.iter().enumerate()
                {
                    for (node, found) in verified.step_types.iter().enumerate()
                    {
                        if proposed >= config.max_proposals
                        {
                            break;
                        }
                        if found != expected || parent.next_state[slot] == node
                        {
                            continue;
                        }
                        proposed += 1;
                        let mut child = parent.clone();
                        child.next_state[slot] = node;
                        consider_candidate(
                            child,
                            AppliedMutation {
                                kind: *kind,
                                section: SectionKind::Step,
                                index: slot,
                            },
                            config.verification_limits,
                            &mut candidates,
                            &mut seen,
                            &mut rejected,
                            &mut duplicates,
                        );
                    }
                }
            },
            MutationKind::RebindOutput =>
            {
                for (output_index, &current) in parent.outputs.iter().enumerate()
                {
                    let expected = &verified.output_types[output_index];
                    for (node, found) in verified.finalize_types.iter().enumerate()
                    {
                        if proposed >= config.max_proposals
                        {
                            break;
                        }
                        if found != expected || current == node || parent.outputs.contains(&node)
                        {
                            continue;
                        }
                        proposed += 1;
                        let mut child = parent.clone();
                        child.outputs[output_index] = node;
                        consider_candidate(
                            child,
                            AppliedMutation {
                                kind: *kind,
                                section: SectionKind::Finalize,
                                index: output_index,
                            },
                            config.verification_limits,
                            &mut candidates,
                            &mut seen,
                            &mut rejected,
                            &mut duplicates,
                        );
                    }
                }
            },
        }
    }

    if candidates.is_empty()
    {
        return Err(MutationError::NoValidMutation { proposed, rejected });
    }
    let mut rng = DeterministicRng::new(seed);
    let selected = candidates.swap_remove(rng.below(candidates.len()));
    Ok(MutationResult {
        program: selected.program,
        applied: selected.applied,
        proposed,
        rejected,
        duplicates,
    })
}

#[allow(clippy::too_many_arguments)]
fn consider_candidate(
    child: ResearchProgram,
    applied: AppliedMutation,
    limits: VerificationLimits,
    candidates: &mut Vec<MutationCandidate>,
    seen: &mut BTreeSet<Vec<u8>>,
    rejected: &mut usize,
    duplicates: &mut usize,
) {
    if verify_program(&child, limits).is_err()
    {
        *rejected += 1;
        return;
    }
    let identity = canonical_bytes(&child);
    if !seen.insert(identity)
    {
        *duplicates += 1;
        return;
    }
    candidates.push(MutationCandidate {
        program: child,
        applied,
    });
}

fn node_variants(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
    section: SectionKind,
    node: usize,
    kind: MutationKind,
    constants: &[ScalarValue],
) -> Vec<Op> {
    let op = &section_ops(program, section)[node];
    match kind
    {
        MutationKind::ReplaceConstant => match op
        {
            Op::Const(current) => constants
                .iter()
                .copied()
                .filter(|candidate| candidate.dtype() == current.dtype() && candidate != current)
                .map(Op::Const)
                .collect(),
            _ => Vec::new(),
        },
        MutationKind::ReplaceOperation => replacement_operations(op),
        MutationKind::ReplaceOperand => operand_replacements(program, verified, section, node, op),
        MutationKind::ModifyReductionAxis =>
        {
            reduction_axis_replacements(program, verified, section, node, op)
        },
        MutationKind::RebindStateUpdate | MutationKind::RebindOutput => Vec::new(),
    }
}

fn replacement_operations(op: &Op) -> Vec<Op> {
    match op
    {
        Op::Add(value)
        | Op::Sub(value)
        | Op::Mul(value)
        | Op::Div(value)
        | Op::Pow(value)
        | Op::Min(value)
        | Op::Max(value) => vec![
            Op::Add(*value),
            Op::Sub(*value),
            Op::Mul(*value),
            Op::Div(*value),
            Op::Pow(*value),
            Op::Min(*value),
            Op::Max(*value),
        ]
        .into_iter()
        .filter(|candidate| candidate != op)
        .collect(),
        Op::Neg(value) | Op::Abs(value) => vec![Op::Neg(*value), Op::Abs(*value)]
            .into_iter()
            .filter(|candidate| candidate != op)
            .collect(),
        Op::Exp(value)
        | Op::Exp2(value)
        | Op::Expm1(value)
        | Op::Log(value)
        | Op::Log2(value)
        | Op::Log1p(value)
        | Op::Sqrt(value)
        | Op::Rsqrt(value)
        | Op::Sin(value)
        | Op::Cos(value)
        | Op::Tanh(value) => vec![
            Op::Exp(*value),
            Op::Exp2(*value),
            Op::Expm1(*value),
            Op::Log(*value),
            Op::Log2(*value),
            Op::Log1p(*value),
            Op::Sqrt(*value),
            Op::Rsqrt(*value),
            Op::Sin(*value),
            Op::Cos(*value),
            Op::Tanh(*value),
        ]
        .into_iter()
        .filter(|candidate| candidate != op)
        .collect(),
        Op::Eq(value)
        | Op::Ne(value)
        | Op::Lt(value)
        | Op::Le(value)
        | Op::Gt(value)
        | Op::Ge(value) => vec![
            Op::Eq(*value),
            Op::Ne(*value),
            Op::Lt(*value),
            Op::Le(*value),
            Op::Gt(*value),
            Op::Ge(*value),
        ]
        .into_iter()
        .filter(|candidate| candidate != op)
        .collect(),
        Op::And(value) | Op::Or(value) => vec![Op::And(*value), Op::Or(*value)]
            .into_iter()
            .filter(|candidate| candidate != op)
            .collect(),
        Op::ReduceSum(value)
        | Op::ReduceProd(value)
        | Op::ReduceMax(value)
        | Op::ReduceMin(value)
        | Op::ReduceMean(value) => vec![
            Op::ReduceSum(*value),
            Op::ReduceProd(*value),
            Op::ReduceMax(*value),
            Op::ReduceMin(*value),
            Op::ReduceMean(*value),
        ]
        .into_iter()
        .filter(|candidate| candidate != op)
        .collect(),
        _ => Vec::new(),
    }
}

fn operand_replacements(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
    section: SectionKind,
    node: usize,
    op: &Op,
) -> Vec<Op> {
    let available = available_refs(program, verified, section, node);
    let mut variants = Vec::new();
    let mut sources = Vec::new();
    op.for_each_ref(|reference| {
        if !sources.contains(&reference)
        {
            sources.push(reference);
        }
    });
    for source in sources
    {
        let Some(source_type) = ref_type(program, verified, section, source)
        else
        {
            continue;
        };
        for (replacement, replacement_type) in &available
        {
            if replacement == &source || replacement_type != source_type
            {
                continue;
            }
            let mut candidate = op.clone();
            candidate.map_refs(|reference| {
                if reference == source
                {
                    *replacement
                }
                else
                {
                    reference
                }
            });
            variants.push(candidate);
        }
    }
    variants
}

fn reduction_axis_replacements(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
    section: SectionKind,
    _node: usize,
    op: &Op,
) -> Vec<Op> {
    let (source, current, constructor): (Ref, Option<usize>, fn(super::ir::Reduce) -> Op) = match op
    {
        Op::ReduceSum(value) => (value.src, value.axis, Op::ReduceSum),
        Op::ReduceProd(value) => (value.src, value.axis, Op::ReduceProd),
        Op::ReduceMax(value) => (value.src, value.axis, Op::ReduceMax),
        Op::ReduceMin(value) => (value.src, value.axis, Op::ReduceMin),
        Op::ReduceMean(value) => (value.src, value.axis, Op::ReduceMean),
        _ => return Vec::new(),
    };
    let Some(value_type) = ref_type(program, verified, section, source)
    else
    {
        return Vec::new();
    };
    std::iter::once(None)
        .chain((0..value_type.shape.len()).map(Some))
        .filter(|axis| axis != &current)
        .map(|axis| constructor(super::ir::Reduce { src: source, axis }))
        .collect()
}

fn available_refs(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
    section: SectionKind,
    node: usize,
) -> Vec<(Ref, ValueType)> {
    let mut available = Vec::new();
    match section
    {
        SectionKind::Init =>
        {
            available.extend(
                program
                    .inputs
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value_type)| (Ref::Input(index), value_type)),
            );
        },
        SectionKind::Step =>
        {
            available.extend(
                program
                    .items
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value_type)| (Ref::Item(index), value_type)),
            );
            available.extend(
                program
                    .state
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value_type)| (Ref::StatePrev(index), value_type)),
            );
        },
        SectionKind::Finalize =>
        {
            available.extend(
                program
                    .inputs
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value_type)| (Ref::Input(index), value_type)),
            );
            available.extend(
                program
                    .state
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value_type)| (Ref::StateFinal(index), value_type)),
            );
        },
    }
    available.extend(
        section_types(verified, section)
            .iter()
            .take(node)
            .cloned()
            .enumerate()
            .map(|(index, value_type)| (Ref::Local(index), value_type)),
    );
    available
}

fn ref_type<'a>(
    program: &'a ResearchProgram,
    verified: &'a VerifiedProgram,
    section: SectionKind,
    reference: Ref,
) -> Option<&'a ValueType> {
    match reference
    {
        Ref::Input(index) => program.inputs.get(index),
        Ref::Item(index) => program.items.get(index),
        Ref::StatePrev(index) | Ref::StateFinal(index) => program.state.get(index),
        Ref::Local(index) => section_types(verified, section).get(index),
    }
}

fn section_types(verified: &VerifiedProgram, section: SectionKind) -> &[ValueType] {
    match section
    {
        SectionKind::Init => &verified.init_types,
        SectionKind::Step => &verified.step_types,
        SectionKind::Finalize => &verified.finalize_types,
    }
}

fn section_ops(program: &ResearchProgram, section: SectionKind) -> &[Op] {
    match section
    {
        SectionKind::Init => &program.init.ops,
        SectionKind::Step => &program.step.ops,
        SectionKind::Finalize => &program.finalize.ops,
    }
}

fn section_ops_mut(program: &mut ResearchProgram, section: SectionKind) -> &mut [Op] {
    match section
    {
        SectionKind::Init => &mut program.init.ops,
        SectionKind::Step => &mut program.step.ops,
        SectionKind::Finalize => &mut program.finalize.ops,
    }
}

/// Crossover unit selected after exact semantic-context compatibility checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossoverUnit {
    InitAndBindings,
    StepAndBindings,
    FinalizeAndOutputs,
}

/// Verified crossover result.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossoverResult {
    pub program: ResearchProgram,
    pub unit: CrossoverUnit,
    pub valid_units_considered: usize,
}

/// Why two programs cannot safely participate in section crossover.
#[derive(Debug, Clone, PartialEq)]
pub enum CrossoverError {
    InvalidLeft(ProgramError),
    InvalidRight(ProgramError),
    SemanticContextMismatch,
    OutputSignatureMismatch,
    NoValidCrossover,
}

impl std::fmt::Display for CrossoverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::InvalidLeft(error) => write!(formatter, "invalid left parent: {error}"),
            Self::InvalidRight(error) => write!(formatter, "invalid right parent: {error}"),
            Self::SemanticContextMismatch => formatter
                .write_str("parents differ in semantic regime, signature, state, or trip count"),
            Self::OutputSignatureMismatch =>
            {
                formatter.write_str("parents have different ordered output types")
            },
            Self::NoValidCrossover => formatter.write_str("no section exchange verifies"),
        }
    }
}

impl std::error::Error for CrossoverError {}

/// Exchange one whole SSA section together with its binding vector. This
/// avoids cross-scope references and partial recurrence-state splices.
pub fn crossover_programs(
    left: &ResearchProgram,
    right: &ResearchProgram,
    limits: VerificationLimits,
    seed: u64,
) -> Result<CrossoverResult, CrossoverError> {
    let left_verified = verify_program(left, limits).map_err(CrossoverError::InvalidLeft)?;
    let right_verified = verify_program(right, limits).map_err(CrossoverError::InvalidRight)?;
    if left.semantics != right.semantics
        || left.inputs != right.inputs
        || left.items != right.items
        || left.state != right.state
        || left.steps != right.steps
    {
        return Err(CrossoverError::SemanticContextMismatch);
    }
    if left_verified.output_types != right_verified.output_types
    {
        return Err(CrossoverError::OutputSignatureMismatch);
    }

    let mut candidates = Vec::new();
    for unit in [
        CrossoverUnit::InitAndBindings,
        CrossoverUnit::StepAndBindings,
        CrossoverUnit::FinalizeAndOutputs,
    ]
    {
        let mut child = left.clone();
        match unit
        {
            CrossoverUnit::InitAndBindings =>
            {
                child.init = right.init.clone();
                child.init_state.clone_from(&right.init_state);
            },
            CrossoverUnit::StepAndBindings =>
            {
                child.step = right.step.clone();
                child.next_state.clone_from(&right.next_state);
            },
            CrossoverUnit::FinalizeAndOutputs =>
            {
                child.finalize = right.finalize.clone();
                child.outputs.clone_from(&right.outputs);
            },
        }
        if &child != left && &child != right && verify_program(&child, limits).is_ok()
        {
            candidates.push((unit, child));
        }
    }
    if candidates.is_empty()
    {
        return Err(CrossoverError::NoValidCrossover);
    }
    let considered = candidates.len();
    let mut rng = DeterministicRng::new(seed);
    let (unit, program) = candidates.swap_remove(rng.below(candidates.len()));
    Ok(CrossoverResult {
        program,
        unit,
        valid_units_considered: considered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::DType;
    use crate::tensor::v2::{GenerationRequest, Grammar, GrammarProfile, generate_program};

    fn scalar() -> ValueType {
        ValueType::scalar(DType::F64)
    }

    fn generated(seed: u64) -> ResearchProgram {
        let mut request = GenerationRequest::expression(vec![scalar(), scalar()], vec![scalar()]);
        request.min_random_finalize_ops = 6;
        request.max_random_finalize_ops = 6;
        generate_program(
            &request,
            &Grammar::profile(GrammarProfile::ScalarAlgebra),
            VerificationLimits::default(),
            seed,
        )
        .unwrap()
        .program
    }

    fn generated_recurrence(seed: u64) -> ResearchProgram {
        let request = GenerationRequest {
            inputs: vec![],
            items: vec![scalar()],
            state: vec![super::super::StateSpec {
                value_type: scalar(),
                initializer: super::super::StateInitializer::Constant(ScalarValue::F64(0.0)),
            }],
            steps: 4,
            output_types: vec![scalar()],
            min_random_step_ops: 5,
            max_random_step_ops: 5,
            min_random_finalize_ops: 5,
            max_random_finalize_ops: 5,
            require_state_update: true,
        };
        generate_program(
            &request,
            &Grammar::profile(GrammarProfile::StreamingRecurrence),
            VerificationLimits::default(),
            seed,
        )
        .unwrap()
        .program
    }

    #[test]
    fn mutation_is_valid_and_reproducible() {
        let parent = generated(1);
        let config = MutationConfig::default();
        let first = mutate_program(&parent, &config, 99).unwrap();
        let second = mutate_program(&parent, &config, 99).unwrap();
        assert_eq!(first, second);
        assert_ne!(first.program, parent);
        assert!(verify_program(&first.program, config.verification_limits).is_ok());
    }

    #[test]
    fn zero_mutation_budget_is_structured_error() {
        let parent = generated(2);
        let config = MutationConfig {
            max_proposals: 0,
            ..MutationConfig::default()
        };
        assert_eq!(
            mutate_program(&parent, &config, 0),
            Err(MutationError::InvalidConfiguration(
                "max_proposals must be non-zero"
            ))
        );
    }

    #[test]
    fn crossover_checks_context_and_returns_verified_child() {
        let left = generated_recurrence(3);
        let right = generated_recurrence(4);
        let result = crossover_programs(&left, &right, VerificationLimits::default(), 5).unwrap();
        assert!(verify_program(&result.program, VerificationLimits::default()).is_ok());

        let mut incompatible = right;
        incompatible.semantics = super::super::NumericalSemantics::FiniteOnly;
        assert_eq!(
            crossover_programs(&left, &incompatible, VerificationLimits::default(), 5),
            Err(CrossoverError::SemanticContextMismatch)
        );
    }

    #[test]
    fn crossover_seed_is_reproducible() {
        let left = generated_recurrence(5);
        let right = generated_recurrence(6);
        assert_eq!(
            crossover_programs(&left, &right, VerificationLimits::default(), 7),
            crossover_programs(&left, &right, VerificationLimits::default(), 7)
        );
    }
}
