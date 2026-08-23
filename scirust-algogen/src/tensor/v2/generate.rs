//! Deterministic, typed generation grammar for V2 research programs.
//!
//! Generation is shape-directed rather than "pick an opcode and hope": every
//! proposed operation is type-inferred with the same rules as the verifier,
//! and only valid proposals enter the seeded choice set. Profiles constrain
//! operator classes and budgets; they never encode a target algorithm.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::tensor::DeterministicRng;

use super::ir::{
    AxisOp, Bin, Narrow, Op, Permute, Reduce, Ref, ResearchProgram, Section, ShapeTo, Ter, Un,
};
use super::semantics::NumericalSemantics;
use super::types::{DType, ScalarValue, ValueType, can_broadcast_to};
use super::verify::{ProgramError, SectionKind, VerificationLimits, infer_op, verify_program};

/// Coarse operator families used to bound and stage a search grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperatorClass {
    Constant,
    Arithmetic,
    Transcendental,
    Extrema,
    Comparison,
    Boolean,
    Selection,
    Reduction,
    LinearAlgebra,
    Shape,
    /// Static, bounds-checked indexing (`Narrow` in V2).
    Indexing,
}

/// Named curricula. A profile only selects a grammar; it does not contain a
/// target expression, objective, or privileged rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarProfile {
    ScalarAlgebra,
    Reduction,
    StreamingRecurrence,
    LinearAlgebra,
    MaskedTensor,
    AttentionBuildingBlocks,
    GeneralScientific,
}

/// Fully explicit search-space controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grammar {
    pub profile: GrammarProfile,
    pub semantics: NumericalSemantics,
    pub allowed_classes: Vec<OperatorClass>,
    pub allowed_dtypes: Vec<DType>,
    pub constants: Vec<ScalarValue>,
    pub max_rank: usize,
    pub max_values: usize,
    pub max_operations: usize,
    pub max_depth: usize,
    pub max_reductions: usize,
    pub max_transcendentals: usize,
    pub max_expensive_ops: usize,
    pub max_linear_algebra_ops: usize,
    pub max_shape_ops: usize,
    pub max_comparisons: usize,
    pub recurrence_allowed: bool,
    pub max_state_components: usize,
    pub max_recurrence_length: u32,
    pub indexing_allowed: bool,
    pub implicit_broadcasting_allowed: bool,
    /// Hard cap on type/shape proposals considered for one program.
    pub max_candidate_enumeration: usize,
}

impl Grammar {
    /// A deterministic default grammar for a named scientific curriculum.
    #[must_use]
    pub fn profile(profile: GrammarProfile) -> Self {
        use OperatorClass as C;

        let common = vec![
            C::Constant,
            C::Arithmetic,
            C::Extrema,
            C::Comparison,
            C::Boolean,
            C::Selection,
            C::Shape,
        ];
        let (mut allowed_classes, recurrence_allowed, indexing_allowed) = match profile
        {
            GrammarProfile::ScalarAlgebra => (
                vec![
                    C::Constant,
                    C::Arithmetic,
                    C::Extrema,
                    C::Comparison,
                    C::Boolean,
                    C::Selection,
                    C::Shape,
                ],
                false,
                false,
            ),
            GrammarProfile::Reduction =>
            {
                let mut classes = common.clone();
                classes.extend([C::Reduction, C::Transcendental]);
                (classes, false, false)
            },
            GrammarProfile::StreamingRecurrence =>
            {
                let mut classes = common.clone();
                classes.extend([C::Reduction, C::Transcendental]);
                (classes, true, false)
            },
            GrammarProfile::LinearAlgebra =>
            {
                let mut classes = common.clone();
                classes.extend([C::LinearAlgebra, C::Reduction]);
                (classes, false, false)
            },
            GrammarProfile::MaskedTensor =>
            {
                let mut classes = common.clone();
                classes.extend([C::Reduction, C::Indexing]);
                (classes, false, true)
            },
            GrammarProfile::AttentionBuildingBlocks =>
            {
                let mut classes = common.clone();
                classes.extend([
                    C::Transcendental,
                    C::Reduction,
                    C::LinearAlgebra,
                    C::Indexing,
                ]);
                (classes, true, true)
            },
            GrammarProfile::GeneralScientific => (
                vec![
                    C::Constant,
                    C::Arithmetic,
                    C::Transcendental,
                    C::Extrema,
                    C::Comparison,
                    C::Boolean,
                    C::Selection,
                    C::Reduction,
                    C::LinearAlgebra,
                    C::Shape,
                    C::Indexing,
                ],
                true,
                true,
            ),
        };
        allowed_classes.sort_unstable();
        allowed_classes.dedup();

        Self {
            profile,
            semantics: NumericalSemantics::StrictIeee,
            allowed_classes,
            allowed_dtypes: vec![DType::F32, DType::F64, DType::Bool],
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
            max_rank: 4,
            max_values: 64,
            max_operations: 64,
            max_depth: 8,
            max_reductions: 4,
            max_transcendentals: 4,
            max_expensive_ops: 8,
            max_linear_algebra_ops: 4,
            max_shape_ops: 12,
            max_comparisons: 8,
            recurrence_allowed,
            max_state_components: 4,
            max_recurrence_length: 256,
            indexing_allowed,
            implicit_broadcasting_allowed: true,
            max_candidate_enumeration: 50_000,
        }
    }

    fn allows(&self, class: OperatorClass) -> bool {
        self.allowed_classes.contains(&class)
    }

    fn allows_dtype(&self, dtype: DType) -> bool {
        self.allowed_dtypes.contains(&dtype)
    }

    fn class_limit(&self, class: OperatorClass) -> usize {
        match class
        {
            OperatorClass::Reduction => self.max_reductions,
            OperatorClass::Transcendental => self.max_transcendentals,
            OperatorClass::LinearAlgebra => self.max_linear_algebra_ops,
            OperatorClass::Shape | OperatorClass::Indexing => self.max_shape_ops,
            OperatorClass::Comparison => self.max_comparisons,
            _ => self.max_operations,
        }
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Self::profile(GrammarProfile::GeneralScientific)
    }
}

/// How one recurrence state component is initialized.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StateInitializer {
    Constant(ScalarValue),
    Input(usize),
}

/// Declared type and initializer of one state component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateSpec {
    pub value_type: ValueType,
    pub initializer: StateInitializer,
}

/// Problem signature and neutral structural constraints supplied to generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub inputs: Vec<ValueType>,
    pub items: Vec<ValueType>,
    pub state: Vec<StateSpec>,
    pub steps: u32,
    pub output_types: Vec<ValueType>,
    pub min_random_step_ops: usize,
    pub max_random_step_ops: usize,
    pub min_random_finalize_ops: usize,
    pub max_random_finalize_ops: usize,
    /// Require every next-state binding to depend on an item or a newly
    /// generated local, rather than copying `StatePrev` unchanged.
    pub require_state_update: bool,
}

impl GenerationRequest {
    /// Construct a straight-line expression-generation request.
    #[must_use]
    pub fn expression(inputs: Vec<ValueType>, output_types: Vec<ValueType>) -> Self {
        Self {
            inputs,
            items: Vec::new(),
            state: Vec::new(),
            steps: 0,
            output_types,
            min_random_step_ops: 0,
            max_random_step_ops: 0,
            min_random_finalize_ops: 1,
            max_random_finalize_ops: 8,
            require_state_update: false,
        }
    }
}

/// Deterministic rejection/accounting data for diagnosing grammar pressure.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub proposed: u64,
    pub type_or_shape_rejections: u64,
    pub depth_rejections: u64,
    pub class_budget_rejections: u64,
    pub duplicate_rejections: u64,
    pub enumeration_truncations: u64,
    pub valid_proposals: u64,
    pub emitted_ops: usize,
    pub emitted_by_class: BTreeMap<OperatorClass, usize>,
}

/// One generated and verifier-approved candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedProgram {
    pub program: ResearchProgram,
    pub stats: GenerationStats,
}

/// Structured generation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationError {
    EmptyOutputs,
    InvalidConfiguration(&'static str),
    UnsupportedDType {
        dtype: DType,
    },
    RankLimit {
        rank: usize,
        maximum: usize,
    },
    RecurrenceDisabled,
    RecurrenceLength {
        steps: u32,
        maximum: u32,
    },
    StateComponentLimit {
        components: usize,
        maximum: usize,
    },
    InvalidStateInitializer {
        slot: usize,
        reason: &'static str,
    },
    NoValidOperation {
        section: SectionKind,
    },
    NoValueOfType {
        section: SectionKind,
        value_type: ValueType,
    },
    GrammarBudgetExceeded {
        budget: &'static str,
        maximum: usize,
    },
    Verification(ProgramError),
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::EmptyOutputs => formatter.write_str("generation request has no outputs"),
            Self::InvalidConfiguration(reason) => write!(formatter, "invalid grammar: {reason}"),
            Self::UnsupportedDType { dtype } =>
            {
                write!(
                    formatter,
                    "dtype {} is disabled by the grammar",
                    dtype.name()
                )
            },
            Self::RankLimit { rank, maximum } =>
            {
                write!(formatter, "rank {rank} exceeds grammar limit {maximum}")
            },
            Self::RecurrenceDisabled =>
            {
                formatter.write_str("recurrence is disabled by the grammar")
            },
            Self::RecurrenceLength { steps, maximum } => write!(
                formatter,
                "recurrence length {steps} exceeds grammar limit {maximum}"
            ),
            Self::StateComponentLimit {
                components,
                maximum,
            } => write!(
                formatter,
                "state has {components} components, exceeding grammar limit {maximum}"
            ),
            Self::InvalidStateInitializer { slot, reason } =>
            {
                write!(formatter, "state initializer {slot} is invalid: {reason}")
            },
            Self::NoValidOperation { section } =>
            {
                write!(formatter, "no valid grammar operation exists in {section}")
            },
            Self::NoValueOfType {
                section,
                value_type,
            } => write!(
                formatter,
                "{section} cannot produce required type {}{:?}",
                value_type.dtype.name(),
                value_type.shape
            ),
            Self::GrammarBudgetExceeded { budget, maximum } =>
            {
                write!(formatter, "grammar budget {budget} exhausted at {maximum}")
            },
            Self::Verification(error) =>
            {
                write!(formatter, "generated program failed verification: {error}")
            },
        }
    }
}

impl std::error::Error for GenerationError {}

impl From<ProgramError> for GenerationError {
    fn from(error: ProgramError) -> Self {
        Self::Verification(error)
    }
}

#[derive(Debug, Clone)]
struct PoolValue {
    reference: Ref,
    value_type: ValueType,
    depth: usize,
}

#[derive(Debug, Clone)]
struct Proposal {
    op: Op,
    value_type: ValueType,
    depth: usize,
}

#[derive(Default)]
struct EmissionState {
    counts: BTreeMap<OperatorClass, usize>,
    expensive: usize,
    total: usize,
}

/// Generate one deterministic valid-by-construction candidate.
pub fn generate_program(
    request: &GenerationRequest,
    grammar: &Grammar,
    limits: VerificationLimits,
    seed: u64,
) -> Result<GeneratedProgram, GenerationError> {
    validate_request(request, grammar)?;

    let mut rng = DeterministicRng::new(seed);
    let mut stats = GenerationStats::default();
    let mut emission = EmissionState::default();
    let state_types: Vec<ValueType> = request
        .state
        .iter()
        .map(|state| state.value_type.clone())
        .collect();

    let mut init = Section::default();
    let mut init_pool = signature_pool(&request.inputs, Ref::Input);
    let mut init_state = Vec::with_capacity(request.state.len());
    for (slot, state) in request.state.iter().enumerate()
    {
        let value = emit_initializer(
            slot,
            state,
            &request.inputs,
            &mut init,
            &mut init_pool,
            grammar,
            &mut emission,
            &mut stats,
        )?;
        init_state.push(value);
    }

    let mut step = Section::default();
    let mut next_state = Vec::new();
    if !request.state.is_empty()
    {
        let mut step_pool = signature_pool(&state_types, Ref::StatePrev);
        step_pool.extend(signature_pool(&request.items, Ref::Item));
        let random_ops = random_count(
            request.min_random_step_ops,
            request.max_random_step_ops,
            &mut rng,
        );
        grow_section(
            SectionKind::Step,
            random_ops,
            &mut step,
            &mut step_pool,
            &state_types,
            grammar,
            &mut rng,
            &mut emission,
            &mut stats,
        )?;
        for state in &state_types
        {
            let binding = choose_state_binding(
                state,
                request.require_state_update,
                &mut step,
                &mut step_pool,
                grammar,
                &mut rng,
                &mut emission,
                &mut stats,
            )?;
            next_state.push(binding);
        }
    }

    let mut finalize = Section::default();
    let mut finalize_pool = signature_pool(&request.inputs, Ref::Input);
    finalize_pool.extend(signature_pool(&state_types, Ref::StateFinal));
    let random_ops = random_count(
        request.min_random_finalize_ops,
        request.max_random_finalize_ops,
        &mut rng,
    );
    grow_section(
        SectionKind::Finalize,
        random_ops,
        &mut finalize,
        &mut finalize_pool,
        &request.output_types,
        grammar,
        &mut rng,
        &mut emission,
        &mut stats,
    )?;

    let mut outputs = Vec::with_capacity(request.output_types.len());
    let mut used = BTreeSet::new();
    for value_type in &request.output_types
    {
        let output = bind_output(
            value_type,
            &mut finalize,
            &mut finalize_pool,
            &mut used,
            grammar,
            &mut rng,
            &mut emission,
            &mut stats,
        )?;
        outputs.push(output);
    }

    let program = ResearchProgram {
        semantics: grammar.semantics,
        inputs: request.inputs.clone(),
        items: request.items.clone(),
        state: state_types,
        steps: request.steps,
        init,
        init_state,
        step,
        next_state,
        finalize,
        outputs,
    };
    verify_program(&program, limits)?;
    stats.emitted_ops = emission.total;
    stats.emitted_by_class = emission.counts;
    Ok(GeneratedProgram { program, stats })
}

fn validate_request(request: &GenerationRequest, grammar: &Grammar) -> Result<(), GenerationError> {
    if request.output_types.is_empty()
    {
        return Err(GenerationError::EmptyOutputs);
    }
    if grammar.max_values == 0 || grammar.max_operations == 0
    {
        return Err(GenerationError::InvalidConfiguration(
            "max_values and max_operations must be non-zero",
        ));
    }
    if grammar.max_depth == 0
    {
        return Err(GenerationError::InvalidConfiguration(
            "max_depth must be non-zero",
        ));
    }
    if grammar.max_candidate_enumeration == 0
    {
        return Err(GenerationError::InvalidConfiguration(
            "max_candidate_enumeration must be non-zero",
        ));
    }
    if request.min_random_step_ops > request.max_random_step_ops
        || request.min_random_finalize_ops > request.max_random_finalize_ops
    {
        return Err(GenerationError::InvalidConfiguration(
            "minimum random operations exceeds maximum",
        ));
    }
    if request.state.is_empty()
    {
        if request.steps != 0 || !request.items.is_empty()
        {
            return Err(GenerationError::InvalidConfiguration(
                "items and steps require recurrence state",
            ));
        }
    }
    else
    {
        if !grammar.recurrence_allowed
        {
            return Err(GenerationError::RecurrenceDisabled);
        }
        if request.steps == 0
        {
            return Err(GenerationError::InvalidConfiguration(
                "recurrence state requires a non-zero trip count",
            ));
        }
        if request.steps > grammar.max_recurrence_length
        {
            return Err(GenerationError::RecurrenceLength {
                steps: request.steps,
                maximum: grammar.max_recurrence_length,
            });
        }
        if request.state.len() > grammar.max_state_components
        {
            return Err(GenerationError::StateComponentLimit {
                components: request.state.len(),
                maximum: grammar.max_state_components,
            });
        }
    }

    for value_type in request
        .inputs
        .iter()
        .chain(&request.items)
        .chain(request.state.iter().map(|state| &state.value_type))
        .chain(&request.output_types)
    {
        if !grammar.allows_dtype(value_type.dtype)
        {
            return Err(GenerationError::UnsupportedDType {
                dtype: value_type.dtype,
            });
        }
        if value_type.shape.len() > grammar.max_rank
        {
            return Err(GenerationError::RankLimit {
                rank: value_type.shape.len(),
                maximum: grammar.max_rank,
            });
        }
    }
    Ok(())
}

fn random_count(minimum: usize, maximum: usize, rng: &mut DeterministicRng) -> usize {
    let span = maximum.saturating_sub(minimum).saturating_add(1);
    minimum.saturating_add(rng.below(span))
}

fn signature_pool(types: &[ValueType], constructor: fn(usize) -> Ref) -> Vec<PoolValue> {
    types
        .iter()
        .enumerate()
        .map(|(index, value_type)| PoolValue {
            reference: constructor(index),
            value_type: value_type.clone(),
            depth: 0,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn emit_initializer(
    slot: usize,
    state: &StateSpec,
    inputs: &[ValueType],
    section: &mut Section,
    pool: &mut Vec<PoolValue>,
    grammar: &Grammar,
    emission: &mut EmissionState,
    stats: &mut GenerationStats,
) -> Result<usize, GenerationError> {
    match state.initializer
    {
        StateInitializer::Input(index) =>
        {
            let input = inputs
                .get(index)
                .ok_or(GenerationError::InvalidStateInitializer {
                    slot,
                    reason: "input index is out of bounds",
                })?;
            if input.dtype != state.value_type.dtype
                || can_broadcast_to(&input.shape, &state.value_type.shape).is_err()
            {
                return Err(GenerationError::InvalidStateInitializer {
                    slot,
                    reason: "input cannot broadcast to the state type",
                });
            }
            let op = if input.shape == state.value_type.shape
            {
                Op::Reshape(ShapeTo {
                    src: Ref::Input(index),
                    shape: state.value_type.shape.clone(),
                })
            }
            else
            {
                Op::BroadcastTo(ShapeTo {
                    src: Ref::Input(index),
                    shape: state.value_type.shape.clone(),
                })
            };
            emit_known(
                SectionKind::Init,
                op,
                state.value_type.clone(),
                1,
                section,
                pool,
                grammar,
                emission,
                stats,
            )
        },
        StateInitializer::Constant(constant) =>
        {
            if constant.dtype() != state.value_type.dtype || !constant.is_admissible()
            {
                return Err(GenerationError::InvalidStateInitializer {
                    slot,
                    reason: "constant dtype/admissibility does not match state",
                });
            }
            if grammar.semantics == NumericalSemantics::FiniteOnly && !constant_is_finite(constant)
            {
                return Err(GenerationError::InvalidStateInitializer {
                    slot,
                    reason: "FiniteOnly forbids an infinite initializer",
                });
            }
            let constant_id = emit_known(
                SectionKind::Init,
                Op::Const(constant),
                ValueType::scalar(constant.dtype()),
                1,
                section,
                pool,
                grammar,
                emission,
                stats,
            )?;
            if state.value_type.is_scalar()
            {
                Ok(constant_id)
            }
            else
            {
                emit_known(
                    SectionKind::Init,
                    Op::BroadcastTo(ShapeTo {
                        src: Ref::Local(constant_id),
                        shape: state.value_type.shape.clone(),
                    }),
                    state.value_type.clone(),
                    2,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                )
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn grow_section(
    kind: SectionKind,
    count: usize,
    section: &mut Section,
    pool: &mut Vec<PoolValue>,
    target_types: &[ValueType],
    grammar: &Grammar,
    rng: &mut DeterministicRng,
    emission: &mut EmissionState,
    stats: &mut GenerationStats,
) -> Result<(), GenerationError> {
    for _ in 0..count
    {
        let proposals =
            enumerate_proposals(kind, section, pool, target_types, grammar, emission, stats);
        if proposals.is_empty()
        {
            return Err(GenerationError::NoValidOperation { section: kind });
        }
        let proposal = proposals[rng.below(proposals.len())].clone();
        emit_known(
            kind,
            proposal.op,
            proposal.value_type,
            proposal.depth,
            section,
            pool,
            grammar,
            emission,
            stats,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn choose_state_binding(
    value_type: &ValueType,
    require_update: bool,
    section: &mut Section,
    pool: &mut Vec<PoolValue>,
    grammar: &Grammar,
    rng: &mut DeterministicRng,
    emission: &mut EmissionState,
    stats: &mut GenerationStats,
) -> Result<usize, GenerationError> {
    let compatible: Vec<usize> = pool
        .iter()
        .enumerate()
        .filter(|(_, value)| {
            &value.value_type == value_type
                && (!require_update || !matches!(value.reference, Ref::StatePrev(_)))
        })
        .map(|(index, _)| index)
        .collect();
    let Some(&chosen) = compatible.get(rng.below(compatible.len()))
    else
    {
        return Err(GenerationError::NoValueOfType {
            section: SectionKind::Step,
            value_type: value_type.clone(),
        });
    };
    let source = pool[chosen].clone();
    if let Ref::Local(id) = source.reference
    {
        return Ok(id);
    }
    emit_known(
        SectionKind::Step,
        Op::Reshape(ShapeTo {
            src: source.reference,
            shape: value_type.shape.clone(),
        }),
        value_type.clone(),
        source.depth.saturating_add(1),
        section,
        pool,
        grammar,
        emission,
        stats,
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_output(
    value_type: &ValueType,
    section: &mut Section,
    pool: &mut Vec<PoolValue>,
    used: &mut BTreeSet<usize>,
    grammar: &Grammar,
    rng: &mut DeterministicRng,
    emission: &mut EmissionState,
    stats: &mut GenerationStats,
) -> Result<usize, GenerationError> {
    let local_matches: Vec<usize> = pool
        .iter()
        .filter_map(|value| match value.reference
        {
            Ref::Local(id) if &value.value_type == value_type && !used.contains(&id) => Some(id),
            _ => None,
        })
        .collect();
    if let Some(&id) = local_matches.get(rng.below(local_matches.len()))
    {
        used.insert(id);
        return Ok(id);
    }

    let base_matches: Vec<PoolValue> = pool
        .iter()
        .filter(|value| {
            !matches!(value.reference, Ref::Local(_)) && &value.value_type == value_type
        })
        .cloned()
        .collect();
    if let Some(source) = base_matches.get(rng.below(base_matches.len()))
    {
        let id = emit_known(
            SectionKind::Finalize,
            Op::Reshape(ShapeTo {
                src: source.reference,
                shape: value_type.shape.clone(),
            }),
            value_type.clone(),
            source.depth.saturating_add(1),
            section,
            pool,
            grammar,
            emission,
            stats,
        )?;
        used.insert(id);
        return Ok(id);
    }

    let constant = zero(value_type.dtype);
    if !grammar.allows(OperatorClass::Constant)
    {
        return Err(GenerationError::NoValueOfType {
            section: SectionKind::Finalize,
            value_type: value_type.clone(),
        });
    }
    let scalar_id = emit_known(
        SectionKind::Finalize,
        Op::Const(constant),
        ValueType::scalar(value_type.dtype),
        1,
        section,
        pool,
        grammar,
        emission,
        stats,
    )?;
    let id = if value_type.is_scalar()
    {
        scalar_id
    }
    else
    {
        emit_known(
            SectionKind::Finalize,
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(scalar_id),
                shape: value_type.shape.clone(),
            }),
            value_type.clone(),
            2,
            section,
            pool,
            grammar,
            emission,
            stats,
        )?
    };
    used.insert(id);
    Ok(id)
}

#[allow(clippy::too_many_arguments)]
fn emit_known(
    kind: SectionKind,
    op: Op,
    value_type: ValueType,
    depth: usize,
    section: &mut Section,
    pool: &mut Vec<PoolValue>,
    grammar: &Grammar,
    emission: &mut EmissionState,
    stats: &mut GenerationStats,
) -> Result<usize, GenerationError> {
    let class = classify_op(&op);
    reserve_emission(class, depth, grammar, emission)?;
    let id = section.ops.len();
    section.ops.push(op);
    pool.push(PoolValue {
        reference: Ref::Local(id),
        value_type,
        depth,
    });
    stats.emitted_ops = emission.total;
    let _ = kind;
    Ok(id)
}

fn reserve_emission(
    class: OperatorClass,
    depth: usize,
    grammar: &Grammar,
    emission: &mut EmissionState,
) -> Result<(), GenerationError> {
    if !grammar.allows(class) || (class == OperatorClass::Indexing && !grammar.indexing_allowed)
    {
        return Err(GenerationError::GrammarBudgetExceeded {
            budget: "operator class disabled",
            maximum: 0,
        });
    }
    if depth > grammar.max_depth
    {
        return Err(GenerationError::GrammarBudgetExceeded {
            budget: "expression depth",
            maximum: grammar.max_depth,
        });
    }
    if emission.total >= grammar.max_operations || emission.total >= grammar.max_values
    {
        return Err(GenerationError::GrammarBudgetExceeded {
            budget: "operations/values",
            maximum: grammar.max_operations.min(grammar.max_values),
        });
    }
    let count = emission.counts.get(&class).copied().unwrap_or(0);
    if count >= grammar.class_limit(class)
    {
        return Err(GenerationError::GrammarBudgetExceeded {
            budget: "operator class",
            maximum: grammar.class_limit(class),
        });
    }
    if is_expensive(class) && emission.expensive >= grammar.max_expensive_ops
    {
        return Err(GenerationError::GrammarBudgetExceeded {
            budget: "expensive operations",
            maximum: grammar.max_expensive_ops,
        });
    }
    *emission.counts.entry(class).or_default() += 1;
    emission.total += 1;
    if is_expensive(class)
    {
        emission.expensive += 1;
    }
    Ok(())
}

fn enumerate_proposals(
    kind: SectionKind,
    section: &Section,
    pool: &[PoolValue],
    target_types: &[ValueType],
    grammar: &Grammar,
    emission: &EmissionState,
    stats: &mut GenerationStats,
) -> Vec<Proposal> {
    let mut proposals = Vec::new();
    let target_shapes: Vec<Vec<usize>> = target_types
        .iter()
        .map(|value| value.shape.clone())
        .chain(pool.iter().map(|value| value.value_type.shape.clone()))
        .collect();

    for &constant in &grammar.constants
    {
        offer(
            Op::Const(constant),
            OperatorClass::Constant,
            kind,
            section,
            pool,
            grammar,
            emission,
            stats,
            &mut proposals,
        );
        if enumeration_full(grammar, stats)
        {
            return truncated(proposals, stats);
        }
    }

    for value in pool
    {
        let reference = value.reference;
        for op in [
            Op::Neg(Un { src: reference }),
            Op::Abs(Un { src: reference }),
        ]
        {
            offer(
                op,
                OperatorClass::Arithmetic,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
        }
        for op in [
            Op::Exp(Un { src: reference }),
            Op::Exp2(Un { src: reference }),
            Op::Expm1(Un { src: reference }),
            Op::Log(Un { src: reference }),
            Op::Log2(Un { src: reference }),
            Op::Log1p(Un { src: reference }),
            Op::Sqrt(Un { src: reference }),
            Op::Rsqrt(Un { src: reference }),
            Op::Sin(Un { src: reference }),
            Op::Cos(Un { src: reference }),
            Op::Tanh(Un { src: reference }),
        ]
        {
            offer(
                op,
                OperatorClass::Transcendental,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
        }
        offer(
            Op::Not(Un { src: reference }),
            OperatorClass::Boolean,
            kind,
            section,
            pool,
            grammar,
            emission,
            stats,
            &mut proposals,
        );

        let axes = std::iter::once(None).chain((0..value.value_type.shape.len()).map(Some));
        for axis in axes
        {
            for op in [
                Op::ReduceSum(Reduce {
                    src: reference,
                    axis,
                }),
                Op::ReduceProd(Reduce {
                    src: reference,
                    axis,
                }),
                Op::ReduceMax(Reduce {
                    src: reference,
                    axis,
                }),
                Op::ReduceMin(Reduce {
                    src: reference,
                    axis,
                }),
                Op::ReduceMean(Reduce {
                    src: reference,
                    axis,
                }),
            ]
            {
                offer(
                    op,
                    OperatorClass::Reduction,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
        }

        for axis in 0..=value.value_type.shape.len()
        {
            offer(
                Op::Unsqueeze(AxisOp {
                    src: reference,
                    axis,
                }),
                OperatorClass::Shape,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
        }
        for (axis, &dimension) in value.value_type.shape.iter().enumerate()
        {
            if dimension == 1
            {
                offer(
                    Op::Squeeze(AxisOp {
                        src: reference,
                        axis,
                    }),
                    OperatorClass::Shape,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            if dimension > 0
            {
                for (start, len) in [(0, 1), (0, dimension), (dimension - 1, 1)]
                {
                    offer(
                        Op::Narrow(Narrow {
                            src: reference,
                            axis,
                            start,
                            len,
                        }),
                        OperatorClass::Indexing,
                        kind,
                        section,
                        pool,
                        grammar,
                        emission,
                        stats,
                        &mut proposals,
                    );
                }
            }
        }
        if value.value_type.shape.len() >= 2
        {
            let mut perm: Vec<usize> = (0..value.value_type.shape.len()).collect();
            perm.reverse();
            offer(
                Op::Transpose(Permute {
                    src: reference,
                    perm,
                }),
                OperatorClass::Shape,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
        }
        for shape in &target_shapes
        {
            offer(
                Op::Reshape(ShapeTo {
                    src: reference,
                    shape: shape.clone(),
                }),
                OperatorClass::Shape,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
            offer(
                Op::BroadcastTo(ShapeTo {
                    src: reference,
                    shape: shape.clone(),
                }),
                OperatorClass::Shape,
                kind,
                section,
                pool,
                grammar,
                emission,
                stats,
                &mut proposals,
            );
        }
        if enumeration_full(grammar, stats)
        {
            return truncated(proposals, stats);
        }
    }

    for left in pool
    {
        for right in pool
        {
            let binary = Bin::new(left.reference, right.reference);
            for op in [
                Op::Add(binary),
                Op::Sub(binary),
                Op::Mul(binary),
                Op::Div(binary),
                Op::Pow(binary),
            ]
            {
                offer(
                    op,
                    OperatorClass::Arithmetic,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            for op in [Op::Min(binary), Op::Max(binary)]
            {
                offer(
                    op,
                    OperatorClass::Extrema,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            for op in [
                Op::Eq(binary),
                Op::Ne(binary),
                Op::Lt(binary),
                Op::Le(binary),
                Op::Gt(binary),
                Op::Ge(binary),
            ]
            {
                offer(
                    op,
                    OperatorClass::Comparison,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            for op in [Op::And(binary), Op::Or(binary)]
            {
                offer(
                    op,
                    OperatorClass::Boolean,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            for op in [
                Op::Dot(binary),
                Op::MatVec(binary),
                Op::VecMat(binary),
                Op::MatMul(binary),
                Op::BatchedMatMul(binary),
                Op::Outer(binary),
            ]
            {
                offer(
                    op,
                    OperatorClass::LinearAlgebra,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            for axis in 0..left.value_type.shape.len()
            {
                offer(
                    Op::Concat {
                        lhs: left.reference,
                        rhs: right.reference,
                        axis,
                    },
                    OperatorClass::Shape,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
            }
            if enumeration_full(grammar, stats)
            {
                return truncated(proposals, stats);
            }
        }
    }

    for first in pool
    {
        for second in pool
        {
            for third in pool
            {
                let ternary = Ter::new(first.reference, second.reference, third.reference);
                offer(
                    Op::MulAdd(ternary),
                    OperatorClass::Arithmetic,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
                offer(
                    Op::Clamp(ternary),
                    OperatorClass::Extrema,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
                offer(
                    Op::Select(ternary),
                    OperatorClass::Selection,
                    kind,
                    section,
                    pool,
                    grammar,
                    emission,
                    stats,
                    &mut proposals,
                );
                if enumeration_full(grammar, stats)
                {
                    return truncated(proposals, stats);
                }
            }
        }
    }
    proposals
}

#[allow(clippy::too_many_arguments)]
fn offer(
    op: Op,
    class: OperatorClass,
    kind: SectionKind,
    section: &Section,
    pool: &[PoolValue],
    grammar: &Grammar,
    emission: &EmissionState,
    stats: &mut GenerationStats,
    proposals: &mut Vec<Proposal>,
) {
    if enumeration_full(grammar, stats)
    {
        return;
    }
    stats.proposed = stats.proposed.saturating_add(1);
    if !grammar.allows(class)
        || (class == OperatorClass::Indexing && !grammar.indexing_allowed)
        || emission.counts.get(&class).copied().unwrap_or(0) >= grammar.class_limit(class)
        || (is_expensive(class) && emission.expensive >= grammar.max_expensive_ops)
        || emission.total >= grammar.max_operations
        || emission.total >= grammar.max_values
    {
        stats.class_budget_rejections = stats.class_budget_rejections.saturating_add(1);
        return;
    }
    if grammar.semantics == NumericalSemantics::FiniteOnly
    {
        if let Op::Const(constant) = op
        {
            if !constant_is_finite(constant)
            {
                stats.type_or_shape_rejections = stats.type_or_shape_rejections.saturating_add(1);
                return;
            }
        }
    }
    if section.ops.contains(&op)
    {
        stats.duplicate_rejections = stats.duplicate_rejections.saturating_add(1);
        return;
    }

    let mut operands = Vec::new();
    let mut depths = Vec::new();
    let mut missing = false;
    op.for_each_ref(|reference| {
        if let Some(value) = pool.iter().find(|value| value.reference == reference)
        {
            operands.push(value.value_type.clone());
            depths.push(value.depth);
        }
        else
        {
            missing = true;
        }
    });
    if missing
    {
        stats.type_or_shape_rejections = stats.type_or_shape_rejections.saturating_add(1);
        return;
    }
    if !grammar.implicit_broadcasting_allowed && uses_implicit_broadcast(&op, &operands)
    {
        stats.type_or_shape_rejections = stats.type_or_shape_rejections.saturating_add(1);
        return;
    }
    let Ok(value_type) = infer_op(&op, kind, section.ops.len(), &operands)
    else
    {
        stats.type_or_shape_rejections = stats.type_or_shape_rejections.saturating_add(1);
        return;
    };
    let depth = depths.into_iter().max().unwrap_or(0).saturating_add(1);
    if depth > grammar.max_depth || value_type.shape.len() > grammar.max_rank
    {
        stats.depth_rejections = stats.depth_rejections.saturating_add(1);
        return;
    }
    if !grammar.allows_dtype(value_type.dtype)
    {
        stats.type_or_shape_rejections = stats.type_or_shape_rejections.saturating_add(1);
        return;
    }
    proposals.push(Proposal {
        op,
        value_type,
        depth,
    });
    stats.valid_proposals = stats.valid_proposals.saturating_add(1);
}

fn enumeration_full(grammar: &Grammar, stats: &GenerationStats) -> bool {
    stats.proposed >= grammar.max_candidate_enumeration as u64
}

fn truncated(proposals: Vec<Proposal>, stats: &mut GenerationStats) -> Vec<Proposal> {
    stats.enumeration_truncations = stats.enumeration_truncations.saturating_add(1);
    proposals
}

fn uses_implicit_broadcast(op: &Op, operands: &[ValueType]) -> bool {
    match op
    {
        Op::Add(_)
        | Op::Sub(_)
        | Op::Mul(_)
        | Op::Div(_)
        | Op::Pow(_)
        | Op::Min(_)
        | Op::Max(_)
        | Op::Eq(_)
        | Op::Ne(_)
        | Op::Lt(_)
        | Op::Le(_)
        | Op::Gt(_)
        | Op::Ge(_)
        | Op::And(_)
        | Op::Or(_) => operands
            .first()
            .zip(operands.get(1))
            .is_some_and(|(a, b)| a.shape != b.shape),
        Op::MulAdd(_) | Op::Clamp(_) => operands
            .windows(2)
            .any(|pair| pair[0].shape != pair[1].shape),
        Op::Select(_) => operands
            .first()
            .zip(operands.get(1))
            .is_some_and(|(mask, value)| mask.shape != value.shape),
        _ => false,
    }
}

fn is_expensive(class: OperatorClass) -> bool {
    matches!(
        class,
        OperatorClass::Transcendental | OperatorClass::Reduction | OperatorClass::LinearAlgebra
    )
}

fn constant_is_finite(constant: ScalarValue) -> bool {
    match constant
    {
        ScalarValue::F32(value) => value.is_finite(),
        ScalarValue::F64(value) => value.is_finite(),
        ScalarValue::Bool(_) => true,
    }
}

fn zero(dtype: DType) -> ScalarValue {
    match dtype
    {
        DType::F32 => ScalarValue::F32(0.0),
        DType::F64 => ScalarValue::F64(0.0),
        DType::Bool => ScalarValue::Bool(false),
    }
}

/// Classify an operation for grammar, diagnostics, mutation, and experiment
/// records. This mapping is exhaustive and deterministic.
#[must_use]
pub fn classify_op(op: &Op) -> OperatorClass {
    match op
    {
        Op::Const(_) => OperatorClass::Constant,
        Op::Add(_)
        | Op::Sub(_)
        | Op::Mul(_)
        | Op::Div(_)
        | Op::MulAdd(_)
        | Op::Pow(_)
        | Op::Neg(_)
        | Op::Abs(_) => OperatorClass::Arithmetic,
        Op::Exp(_)
        | Op::Exp2(_)
        | Op::Expm1(_)
        | Op::Log(_)
        | Op::Log2(_)
        | Op::Log1p(_)
        | Op::Sqrt(_)
        | Op::Rsqrt(_)
        | Op::Sin(_)
        | Op::Cos(_)
        | Op::Tanh(_) => OperatorClass::Transcendental,
        Op::Min(_) | Op::Max(_) | Op::Clamp(_) => OperatorClass::Extrema,
        Op::Eq(_) | Op::Ne(_) | Op::Lt(_) | Op::Le(_) | Op::Gt(_) | Op::Ge(_) =>
        {
            OperatorClass::Comparison
        },
        Op::And(_) | Op::Or(_) | Op::Not(_) => OperatorClass::Boolean,
        Op::Select(_) => OperatorClass::Selection,
        Op::ReduceSum(_)
        | Op::ReduceProd(_)
        | Op::ReduceMax(_)
        | Op::ReduceMin(_)
        | Op::ReduceMean(_) => OperatorClass::Reduction,
        Op::Dot(_)
        | Op::MatVec(_)
        | Op::VecMat(_)
        | Op::MatMul(_)
        | Op::BatchedMatMul(_)
        | Op::Outer(_) => OperatorClass::LinearAlgebra,
        Op::Reshape(_)
        | Op::Squeeze(_)
        | Op::Unsqueeze(_)
        | Op::Transpose(_)
        | Op::BroadcastTo(_)
        | Op::Concat { .. } => OperatorClass::Shape,
        Op::Narrow(_) => OperatorClass::Indexing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{canonical_bytes, verify_program};

    fn scalar() -> ValueType {
        ValueType::scalar(DType::F64)
    }

    #[test]
    fn fixed_seed_reproduces_program_and_rejection_statistics() {
        let request = GenerationRequest::expression(vec![scalar(), scalar()], vec![scalar()]);
        let grammar = Grammar::profile(GrammarProfile::ScalarAlgebra);
        let first =
            generate_program(&request, &grammar, VerificationLimits::default(), 77).unwrap();
        let second =
            generate_program(&request, &grammar, VerificationLimits::default(), 77).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            canonical_bytes(&first.program),
            canonical_bytes(&second.program)
        );
    }

    #[test]
    fn different_seeds_explore_more_than_one_structure() {
        let mut request = GenerationRequest::expression(vec![scalar(), scalar()], vec![scalar()]);
        request.min_random_finalize_ops = 4;
        request.max_random_finalize_ops = 4;
        let grammar = Grammar::profile(GrammarProfile::ScalarAlgebra);
        let identities: BTreeSet<Vec<u8>> = (0..16)
            .map(|seed| {
                canonical_bytes(
                    &generate_program(&request, &grammar, VerificationLimits::default(), seed)
                        .unwrap()
                        .program,
                )
            })
            .collect();
        assert!(identities.len() > 1);
    }

    #[test]
    fn generated_recurrence_is_valid_and_statically_bounded() {
        let request = GenerationRequest {
            inputs: vec![],
            items: vec![scalar()],
            state: vec![
                StateSpec {
                    value_type: scalar(),
                    initializer: StateInitializer::Constant(ScalarValue::F64(0.0)),
                },
                StateSpec {
                    value_type: scalar(),
                    initializer: StateInitializer::Constant(ScalarValue::F64(1.0)),
                },
            ],
            steps: 8,
            output_types: vec![scalar(), scalar()],
            min_random_step_ops: 4,
            max_random_step_ops: 4,
            min_random_finalize_ops: 4,
            max_random_finalize_ops: 4,
            require_state_update: true,
        };
        let grammar = Grammar::profile(GrammarProfile::StreamingRecurrence);
        let generated =
            generate_program(&request, &grammar, VerificationLimits::default(), 9).unwrap();
        assert_eq!(generated.program.steps, 8);
        assert_eq!(generated.program.state.len(), 2);
        assert!(verify_program(&generated.program, VerificationLimits::default()).is_ok());
    }

    #[test]
    fn profiles_never_emit_disabled_classes() {
        let mut request = GenerationRequest::expression(vec![scalar(), scalar()], vec![scalar()]);
        request.min_random_finalize_ops = 6;
        request.max_random_finalize_ops = 6;
        let grammar = Grammar::profile(GrammarProfile::ScalarAlgebra);
        for seed in 0..32
        {
            let generated =
                generate_program(&request, &grammar, VerificationLimits::default(), seed).unwrap();
            for op in &generated.program.finalize.ops
            {
                assert!(grammar.allowed_classes.contains(&classify_op(op)));
                assert!(!matches!(
                    classify_op(op),
                    OperatorClass::Reduction | OperatorClass::Transcendental
                ));
            }
        }
    }

    #[test]
    fn recurrence_and_rank_limits_fail_before_enumeration() {
        let request = GenerationRequest {
            inputs: vec![],
            items: vec![scalar()],
            state: vec![StateSpec {
                value_type: scalar(),
                initializer: StateInitializer::Constant(ScalarValue::F64(0.0)),
            }],
            steps: 2,
            output_types: vec![scalar()],
            min_random_step_ops: 1,
            max_random_step_ops: 1,
            min_random_finalize_ops: 1,
            max_random_finalize_ops: 1,
            require_state_update: false,
        };
        let grammar = Grammar::profile(GrammarProfile::ScalarAlgebra);
        assert_eq!(
            generate_program(&request, &grammar, VerificationLimits::default(), 0),
            Err(GenerationError::RecurrenceDisabled)
        );
    }

    #[test]
    fn fixed_candidate_cap_is_reported_deterministically() {
        let mut request = GenerationRequest::expression(vec![scalar(), scalar()], vec![scalar()]);
        request.min_random_finalize_ops = 1;
        request.max_random_finalize_ops = 1;
        let mut grammar = Grammar::profile(GrammarProfile::GeneralScientific);
        grammar.max_candidate_enumeration = 10;
        let generated =
            generate_program(&request, &grammar, VerificationLimits::default(), 1).unwrap();
        assert!(generated.stats.enumeration_truncations > 0);
        assert_eq!(generated.stats.proposed, 10);
    }

    #[test]
    fn classifier_covers_stable_static_indexing_separately() {
        assert_eq!(
            classify_op(&Op::Narrow(Narrow {
                src: Ref::Input(0),
                axis: 0,
                start: 0,
                len: 1,
            })),
            OperatorClass::Indexing
        );
    }

    #[test]
    fn general_profile_has_a_valid_typed_proposal_for_every_enabled_class() {
        let input_types = vec![
            ValueType::scalar(DType::F64),
            ValueType::new(DType::F64, vec![3]),
            ValueType::new(DType::F64, vec![3]),
            ValueType::new(DType::F64, vec![2, 3]),
            ValueType::new(DType::F64, vec![3, 2]),
            ValueType::new(DType::Bool, vec![3]),
        ];
        let pool = signature_pool(&input_types, Ref::Input);
        let section = Section::default();
        let mut grammar = Grammar::profile(GrammarProfile::GeneralScientific);
        grammar.max_candidate_enumeration = 200_000;
        let mut stats = GenerationStats::default();
        let proposals = enumerate_proposals(
            SectionKind::Finalize,
            &section,
            &pool,
            &input_types,
            &grammar,
            &EmissionState::default(),
            &mut stats,
        );
        let observed: BTreeSet<OperatorClass> = proposals
            .iter()
            .map(|proposal| classify_op(&proposal.op))
            .collect();
        let expected: BTreeSet<OperatorClass> = grammar.allowed_classes.iter().copied().collect();
        assert_eq!(observed, expected);
    }
}
