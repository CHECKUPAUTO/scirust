//! Deterministic counterexample evaluation, Pareto archival, and experiments.
//!
//! Correctness and structural cost remain separate objectives. Candidate
//! ranking never uses host time. Exact canonical bytes are authoritative for
//! deduplication; SHA-256 digests are compact labels only.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::canonical::{CANONICALIZATION_VERSION, canonical_bytes, program_digest};
use super::cost::{CostReport, estimate_cost};
use super::generate::{
    GeneratedProgram, GenerationError, GenerationRequest, GenerationStats, Grammar,
    generate_program,
};
use super::interpret::{ExecutionPolicy, TensorDataError, ValueTensor, execute_program};
use super::ir::{IR_VERSION, ResearchProgram};
use super::serialization::{SERIALIZATION_VERSION, serialize_program};
use super::simplify::canonicalize;
use super::types::ValueType;
use super::verify::VerificationLimits;

/// Version of the deterministic experiment/archive schema in this module.
pub const EXPERIMENT_SCHEMA_VERSION: u32 = 1;
/// Finite penalty assigned to one unusable case.
pub const CASE_FAILURE_PENALTY: f64 = 1.0e30;

/// One bounded deterministic counterexample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterexampleCase {
    pub inputs: Vec<ValueTensor>,
    /// Step-major item sequence, exactly `steps * items_per_step` tensors.
    pub items: Vec<ValueTensor>,
    pub expected_outputs: Vec<ValueTensor>,
}

/// Named deterministic fixtures. Passing them is evidence, not proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterexampleSet {
    pub id: String,
    pub cases: Vec<CounterexampleCase>,
}

/// Tensor role in a malformed counterexample diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaseTensorRole {
    Input,
    Item,
    ExpectedOutput,
}

/// Counterexample construction/compatibility failure.
#[derive(Debug, Clone, PartialEq)]
pub enum CounterexampleError {
    EmptyId,
    EmptyCases,
    InvalidTensor {
        case: usize,
        role: CaseTensorRole,
        index: usize,
        source: TensorDataError,
    },
}

impl std::fmt::Display for CounterexampleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::EmptyId => formatter.write_str("counterexample set id is empty"),
            Self::EmptyCases => formatter.write_str("counterexample set has no cases"),
            Self::InvalidTensor {
                case,
                role,
                index,
                source,
            } => write!(
                formatter,
                "counterexample {case} {role:?} tensor {index} is invalid: {source}"
            ),
        }
    }
}

impl std::error::Error for CounterexampleError {}

impl CounterexampleSet {
    /// Validate tensor layouts immediately; signature matching is checked
    /// against each candidate during evaluation.
    pub fn new(
        id: impl Into<String>,
        cases: Vec<CounterexampleCase>,
    ) -> Result<Self, CounterexampleError> {
        let id = id.into();
        if id.is_empty()
        {
            return Err(CounterexampleError::EmptyId);
        }
        if cases.is_empty()
        {
            return Err(CounterexampleError::EmptyCases);
        }
        for (case_index, case) in cases.iter().enumerate()
        {
            for (role, tensors) in [
                (CaseTensorRole::Input, case.inputs.as_slice()),
                (CaseTensorRole::Item, case.items.as_slice()),
                (
                    CaseTensorRole::ExpectedOutput,
                    case.expected_outputs.as_slice(),
                ),
            ]
            {
                for (index, tensor) in tensors.iter().enumerate()
                {
                    tensor.validate_layout().map_err(|source| {
                        CounterexampleError::InvalidTensor {
                            case: case_index,
                            role,
                            index,
                            source,
                        }
                    })?;
                }
            }
        }
        Ok(Self { id, cases })
    }

    /// Stable SHA-256 over the schema-tagged serialized fixture content.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SCIRUST-RIR2-DATASET\0");
        bytes.extend_from_slice(&EXPERIMENT_SCHEMA_VERSION.to_le_bytes());
        encode_len(&mut bytes, self.id.len());
        bytes.extend_from_slice(self.id.as_bytes());
        encode_len(&mut bytes, self.cases.len());
        for case in &self.cases
        {
            for tensors in [&case.inputs, &case.items, &case.expected_outputs]
            {
                encode_len(&mut bytes, tensors.len());
                for tensor in tensors
                {
                    bytes.push(tensor.dtype.tag());
                    encode_len(&mut bytes, tensor.shape.len());
                    for &dimension in &tensor.shape
                    {
                        encode_len(&mut bytes, dimension);
                    }
                    encode_len(&mut bytes, tensor.data.len());
                    for &value in &tensor.data
                    {
                        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
            }
        }
        sha256_hex(&bytes)
    }
}

fn encode_len(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

/// Correctness objectives, deliberately separate from structural cost.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessObjectives {
    pub mean_squared_error: f64,
    pub max_absolute_error: f64,
    pub failed_cases: usize,
    pub total_cases: usize,
    /// Bit-exact equality for every output element in every case.
    pub exact: bool,
}

/// Multiobjective fitness of one verified candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScientificFitness {
    pub correctness: CorrectnessObjectives,
    pub cost: CostReport,
}

/// Evaluate a program over a bounded set using fixed element and case order.
#[must_use]
pub fn evaluate_on_counterexamples(
    program: &ResearchProgram,
    dataset: &CounterexampleSet,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> ScientificFitness {
    let cost = estimate_cost(program, limits)
        .unwrap_or_else(|_| CostReport::unevaluable(program.node_count(), program.steps));
    let mut case_error_sum = 0.0f64;
    let mut maximum_error = 0.0f64;
    let mut failed_cases = 0usize;
    let mut exact = true;

    for case in &dataset.cases
    {
        let signature_matches = tensors_match_types(&case.inputs, &program.inputs)
            && case.items.len() == (program.steps as usize).saturating_mul(program.items.len())
            && case
                .items
                .chunks(program.items.len().max(1))
                .all(|step| program.items.is_empty() || tensors_match_types(step, &program.items));
        if !signature_matches
        {
            failed_cases += 1;
            exact = false;
            case_error_sum += CASE_FAILURE_PENALTY;
            maximum_error = maximum_error.max(CASE_FAILURE_PENALTY.sqrt());
            continue;
        }

        match execute_program(program, &case.inputs, &case.items, policy, limits)
        {
            Ok(result)
                if result.outputs.len() == case.expected_outputs.len()
                    && result
                        .outputs
                        .iter()
                        .zip(&case.expected_outputs)
                        .all(|(actual, expected)| actual.value_type() == expected.value_type()) =>
            {
                let mut squared_sum = 0.0f64;
                let mut elements = 0usize;
                for (actual, expected) in result.outputs.iter().zip(&case.expected_outputs)
                {
                    for (&actual_value, &expected_value) in actual.data.iter().zip(&expected.data)
                    {
                        let difference = actual_value - expected_value;
                        squared_sum = squared_sum.mul_add(1.0, difference * difference);
                        maximum_error = maximum_error.max(difference.abs());
                        exact &= actual_value.to_bits() == expected_value.to_bits();
                        elements += 1;
                    }
                }
                if elements != 0
                {
                    case_error_sum += squared_sum / elements as f64;
                }
            },
            Ok(_) | Err(_) =>
            {
                failed_cases += 1;
                exact = false;
                case_error_sum += CASE_FAILURE_PENALTY;
                maximum_error = maximum_error.max(CASE_FAILURE_PENALTY.sqrt());
            },
        }
    }

    ScientificFitness {
        correctness: CorrectnessObjectives {
            mean_squared_error: case_error_sum / dataset.cases.len() as f64,
            max_absolute_error: maximum_error,
            failed_cases,
            total_cases: dataset.cases.len(),
            exact,
        },
        cost,
    }
}

fn tensors_match_types(tensors: &[ValueTensor], types: &[ValueType]) -> bool {
    tensors.len() == types.len()
        && tensors
            .iter()
            .zip(types)
            .all(|(tensor, value_type)| &tensor.value_type() == value_type)
}

/// One collision-safe nondominated archive entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoEntry {
    pub program: ResearchProgram,
    pub canonical_bytes: Vec<u8>,
    pub digest: String,
    pub fitness: ScientificFitness,
    pub candidate_index: usize,
    pub candidate_seed: u64,
}

/// Result of considering a candidate for the Pareto archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveDecision {
    Duplicate,
    Dominated,
    Admitted,
    EvictedByCapacity,
}

/// Deterministic capacity-bounded nondominated archive.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoArchive {
    capacity: usize,
    entries: Vec<ParetoEntry>,
    comparisons: u64,
}

impl ParetoArchive {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::new(),
            comparisons: 0,
        }
    }

    pub fn entries(&self) -> &[ParetoEntry] {
        &self.entries
    }

    pub fn comparisons(&self) -> u64 {
        self.comparisons
    }

    /// Exact bytes, not digest equality, define duplicates.
    pub fn consider(&mut self, entry: ParetoEntry) -> ArchiveDecision {
        if self
            .entries
            .iter()
            .any(|existing| existing.canonical_bytes == entry.canonical_bytes)
        {
            return ArchiveDecision::Duplicate;
        }
        let mut dominated = false;
        for existing in &self.entries
        {
            self.comparisons = self.comparisons.saturating_add(1);
            if dominates(&existing.fitness, &entry.fitness)
            {
                dominated = true;
                break;
            }
        }
        if dominated || self.capacity == 0
        {
            return ArchiveDecision::Dominated;
        }
        self.entries.retain(|existing| {
            self.comparisons = self.comparisons.saturating_add(1);
            !dominates(&entry.fitness, &existing.fitness)
        });
        let identity = entry.canonical_bytes.clone();
        self.entries.push(entry);
        self.entries.sort_by(compare_entries);
        if self.entries.len() > self.capacity
        {
            self.entries.truncate(self.capacity);
        }
        if self
            .entries
            .iter()
            .any(|existing| existing.canonical_bytes == identity)
        {
            ArchiveDecision::Admitted
        }
        else
        {
            ArchiveDecision::EvictedByCapacity
        }
    }
}

/// Pareto dominance across correctness, work, memory, state, and depth.
#[must_use]
pub fn dominates(left: &ScientificFitness, right: &ScientificFitness) -> bool {
    let left_objectives = objectives(left);
    let right_objectives = objectives(right);
    let no_worse = left_objectives
        .iter()
        .zip(&right_objectives)
        .all(|(left, right)| left <= right);
    let strictly_better = left_objectives
        .iter()
        .zip(&right_objectives)
        .any(|(left, right)| left < right);
    no_worse && strictly_better
}

fn objectives(fitness: &ScientificFitness) -> [u64; 9] {
    let cost = &fitness.cost;
    [
        fitness.correctness.failed_cases as u64,
        ordered_nonnegative_f64(fitness.correctness.mean_squared_error),
        ordered_nonnegative_f64(fitness.correctness.max_absolute_error),
        cost.logical_flops,
        cost.peak_live_bytes,
        cost.state_bytes,
        cost.update_depth.saturating_add(cost.finalize_depth) as u64,
        cost.reduction_count,
        cost.exp_count
            .saturating_add(cost.log_count)
            .saturating_add(cost.sqrt_count)
            .saturating_add(cost.trig_count),
    ]
}

fn ordered_nonnegative_f64(value: f64) -> u64 {
    if value.is_nan() || value.is_sign_negative()
    {
        u64::MAX
    }
    else
    {
        value.to_bits()
    }
}

fn compare_entries(left: &ParetoEntry, right: &ParetoEntry) -> Ordering {
    objectives(&left.fitness)
        .cmp(&objectives(&right.fitness))
        .then_with(|| left.canonical_bytes.cmp(&right.canonical_bytes))
        .then_with(|| left.candidate_index.cmp(&right.candidate_index))
}

/// Deterministic search configuration and identity inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub source_revision: String,
    pub seed: u64,
    pub max_candidates: usize,
    pub archive_capacity: usize,
    pub stop_on_exact: bool,
    pub grammar: Grammar,
    pub request: GenerationRequest,
    pub verification_limits: VerificationLimits,
    pub execution_policy: ExecutionPolicy,
}

/// Stable rejection categories used by experiment records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionCategory {
    GenerationConfiguration,
    GenerationBudget,
    GenerationNoValidOperation,
    GenerationVerification,
    DuplicateCanonicalProgram,
}

/// One deterministic candidate record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub index: usize,
    pub seed: u64,
    pub generation_stats: Option<GenerationStats>,
    pub rejection: Option<RejectionCategory>,
    pub rejection_detail: Option<String>,
    pub program: Option<ResearchProgram>,
    pub canonical_bytes: Option<Vec<u8>>,
    pub digest: Option<String>,
    pub serialized_program: Option<String>,
    pub fitness: Option<ScientificFitness>,
    pub archive_decision: Option<ArchiveDecision>,
    /// Position in the final ordered archive, if retained.
    pub final_pareto_position: Option<usize>,
}

/// Deterministic search-engine overhead counters. These are diagnostics, never
/// fitness objectives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchDiagnostics {
    pub candidates_attempted: usize,
    pub candidates_generated: usize,
    pub generation_rejections: usize,
    pub canonical_duplicates: usize,
    pub interpreter_case_executions: usize,
    pub archive_comparisons: u64,
    pub final_archive_size: usize,
}

/// Complete reproducible run record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificExperimentArchive {
    pub schema_version: u32,
    pub ir_version: u32,
    pub canonicalization_version: u32,
    pub serialization_version: u32,
    pub config: ExperimentConfig,
    pub dataset_id: String,
    pub dataset_digest: String,
    /// Whether at least one candidate was bit-exact on every case of the
    /// (training) dataset. This is training fit, NOT a discovery or
    /// correctness claim off the dataset; CEGIS-level claims require the
    /// fresh-falsification protocol in [`super::cegis`].
    pub success: bool,
    pub records: Vec<CandidateRecord>,
    pub pareto: Vec<ParetoEntry>,
    pub diagnostics: SearchDiagnostics,
    /// Content-integrity digest, not an authenticity signature.
    pub digest: String,
}

/// Experiment setup failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ScientificExperimentError {
    EmptySourceRevision,
    ZeroCandidates,
    ZeroArchiveCapacity,
    Serialization(String),
}

impl std::fmt::Display for ScientificExperimentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::EmptySourceRevision => formatter.write_str("source revision is empty"),
            Self::ZeroCandidates => formatter.write_str("max_candidates must be non-zero"),
            Self::ZeroArchiveCapacity => formatter.write_str("archive_capacity must be non-zero"),
            Self::Serialization(error) =>
            {
                write!(formatter, "experiment serialization failed: {error}")
            },
        }
    }
}

impl std::error::Error for ScientificExperimentError {}

/// Run a deterministic generate/canonicalize/evaluate/Pareto pipeline.
pub fn run_scientific_experiment(
    config: &ExperimentConfig,
    dataset: &CounterexampleSet,
) -> Result<ScientificExperimentArchive, ScientificExperimentError> {
    if config.source_revision.is_empty()
    {
        return Err(ScientificExperimentError::EmptySourceRevision);
    }
    if config.max_candidates == 0
    {
        return Err(ScientificExperimentError::ZeroCandidates);
    }
    if config.archive_capacity == 0
    {
        return Err(ScientificExperimentError::ZeroArchiveCapacity);
    }

    let mut records = Vec::with_capacity(config.max_candidates);
    let mut archive = ParetoArchive::new(config.archive_capacity);
    let mut identities = BTreeSet::new();
    let mut diagnostics = SearchDiagnostics::default();
    let mut success = false;

    for index in 0..config.max_candidates
    {
        diagnostics.candidates_attempted += 1;
        let seed = candidate_seed(config.seed, index);
        let generated = match generate_program(
            &config.request,
            &config.grammar,
            config.verification_limits,
            seed,
        )
        {
            Ok(generated) => generated,
            Err(error) =>
            {
                diagnostics.generation_rejections += 1;
                records.push(rejected_record(index, seed, &error));
                continue;
            },
        };
        diagnostics.candidates_generated += 1;
        let GeneratedProgram {
            program,
            stats: generation_stats,
        } = generated;
        let canonicalized = canonicalize(&program, config.verification_limits)
            .map_err(|error| ScientificExperimentError::Serialization(error.to_string()))?;
        let program = canonicalized.program;
        let identity = canonical_bytes(&program);
        let digest = program_digest(&program);
        if !identities.insert(identity.clone())
        {
            diagnostics.canonical_duplicates += 1;
            records.push(CandidateRecord {
                index,
                seed,
                generation_stats: Some(generation_stats),
                rejection: Some(RejectionCategory::DuplicateCanonicalProgram),
                rejection_detail: Some("exact canonical bytes already evaluated".to_string()),
                program: Some(program),
                canonical_bytes: Some(identity),
                digest: Some(digest),
                serialized_program: None,
                fitness: None,
                archive_decision: None,
                final_pareto_position: None,
            });
            continue;
        }
        diagnostics.interpreter_case_executions = diagnostics
            .interpreter_case_executions
            .saturating_add(dataset.cases.len());
        let fitness = evaluate_on_counterexamples(
            &program,
            dataset,
            config.execution_policy,
            config.verification_limits,
        );
        let serialized_program = serialize_program(&program, config.verification_limits)
            .map_err(|error| ScientificExperimentError::Serialization(error.to_string()))?;
        let decision = archive.consider(ParetoEntry {
            program: program.clone(),
            canonical_bytes: identity.clone(),
            digest: digest.clone(),
            fitness,
            candidate_index: index,
            candidate_seed: seed,
        });
        records.push(CandidateRecord {
            index,
            seed,
            generation_stats: Some(generation_stats),
            rejection: None,
            rejection_detail: None,
            program: Some(program),
            canonical_bytes: Some(identity),
            digest: Some(digest),
            serialized_program: Some(serialized_program),
            fitness: Some(fitness),
            archive_decision: Some(decision),
            final_pareto_position: None,
        });
        if fitness.correctness.exact
        {
            success = true;
            if config.stop_on_exact
            {
                break;
            }
        }
    }

    let pareto = archive.entries().to_vec();
    for record in &mut records
    {
        let Some(identity) = &record.canonical_bytes
        else
        {
            continue;
        };
        record.final_pareto_position = pareto
            .iter()
            .position(|entry| &entry.canonical_bytes == identity);
    }
    diagnostics.archive_comparisons = archive.comparisons();
    diagnostics.final_archive_size = pareto.len();

    let mut result = ScientificExperimentArchive {
        schema_version: EXPERIMENT_SCHEMA_VERSION,
        ir_version: IR_VERSION,
        canonicalization_version: CANONICALIZATION_VERSION,
        serialization_version: SERIALIZATION_VERSION,
        config: config.clone(),
        dataset_id: dataset.id.clone(),
        dataset_digest: dataset.digest(),
        success,
        records,
        pareto,
        diagnostics,
        digest: String::new(),
    };
    result.digest = experiment_digest(&result)?;
    Ok(result)
}

/// Rerun and require an identical logical archive and content digest.
pub fn replay_scientific_experiment(
    archived: &ScientificExperimentArchive,
    dataset: &CounterexampleSet,
) -> Result<bool, ScientificExperimentError> {
    let replay = run_scientific_experiment(&archived.config, dataset)?;
    Ok(&replay == archived)
}

fn rejected_record(index: usize, seed: u64, error: &GenerationError) -> CandidateRecord {
    let category = match error
    {
        GenerationError::InvalidConfiguration(_)
        | GenerationError::EmptyOutputs
        | GenerationError::UnsupportedDType { .. }
        | GenerationError::RankLimit { .. }
        | GenerationError::RecurrenceDisabled
        | GenerationError::RecurrenceLength { .. }
        | GenerationError::StateComponentLimit { .. }
        | GenerationError::InvalidStateInitializer { .. } =>
        {
            RejectionCategory::GenerationConfiguration
        },
        GenerationError::GrammarBudgetExceeded { .. } => RejectionCategory::GenerationBudget,
        GenerationError::NoValidOperation { .. } | GenerationError::NoValueOfType { .. } =>
        {
            RejectionCategory::GenerationNoValidOperation
        },
        GenerationError::Verification(_) => RejectionCategory::GenerationVerification,
    };
    CandidateRecord {
        index,
        seed,
        generation_stats: None,
        rejection: Some(category),
        rejection_detail: Some(error.to_string()),
        program: None,
        canonical_bytes: None,
        digest: None,
        serialized_program: None,
        fitness: None,
        archive_decision: None,
        final_pareto_position: None,
    }
}

fn candidate_seed(seed: u64, index: usize) -> u64 {
    let mut value = seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn experiment_digest(
    archive: &ScientificExperimentArchive,
) -> Result<String, ScientificExperimentError> {
    let content = (
        archive.schema_version,
        archive.ir_version,
        archive.canonicalization_version,
        archive.serialization_version,
        &archive.config,
        &archive.dataset_id,
        &archive.dataset_digest,
        archive.success,
        &archive.records,
        &archive.pareto,
        archive.diagnostics,
    );
    let bytes = serde_json::to_vec(&content)
        .map_err(|error| ScientificExperimentError::Serialization(error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest
    {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{
        DType, FloatPolicy, GrammarProfile, OperatorClass, ScalarValue, StateInitializer, StateSpec,
    };

    fn scalar(value: f64) -> ValueTensor {
        ValueTensor::scalar_f64(value)
    }

    fn sum_dataset() -> CounterexampleSet {
        CounterexampleSet::new(
            "sum-recurrence-adversarial-v1",
            vec![
                CounterexampleCase {
                    inputs: vec![],
                    items: vec![scalar(1.0), scalar(2.0), scalar(3.0)],
                    expected_outputs: vec![scalar(6.0)],
                },
                CounterexampleCase {
                    inputs: vec![],
                    items: vec![scalar(0.0), scalar(-0.0), scalar(0.0)],
                    expected_outputs: vec![scalar(0.0)],
                },
                CounterexampleCase {
                    inputs: vec![],
                    items: vec![scalar(-2.0), scalar(5.0), scalar(-1.0)],
                    expected_outputs: vec![scalar(2.0)],
                },
                CounterexampleCase {
                    inputs: vec![],
                    items: vec![scalar(1.0e-12), scalar(-1.0e-12), scalar(4.0)],
                    expected_outputs: vec![scalar(4.0)],
                },
            ],
        )
        .unwrap()
    }

    fn sum_config() -> ExperimentConfig {
        let value_type = ValueType::scalar(DType::F64);
        let mut grammar = Grammar::profile(GrammarProfile::StreamingRecurrence);
        grammar.allowed_classes = vec![
            OperatorClass::Constant,
            OperatorClass::Arithmetic,
            OperatorClass::Shape,
        ];
        grammar.allowed_dtypes = vec![DType::F64];
        grammar.constants = vec![
            ScalarValue::F64(-1.0),
            ScalarValue::F64(0.0),
            ScalarValue::F64(1.0),
        ];
        grammar.max_operations = 8;
        grammar.max_values = 8;
        grammar.max_depth = 3;
        grammar.max_shape_ops = 4;
        ExperimentConfig {
            source_revision: "test-revision".to_string(),
            seed: 0xA11C_E5E5,
            max_candidates: 256,
            archive_capacity: 16,
            stop_on_exact: true,
            grammar,
            request: GenerationRequest {
                inputs: vec![],
                items: vec![value_type.clone()],
                state: vec![StateSpec {
                    value_type: value_type.clone(),
                    initializer: StateInitializer::Constant(ScalarValue::F64(0.0)),
                }],
                steps: 3,
                output_types: vec![value_type],
                min_random_step_ops: 1,
                max_random_step_ops: 1,
                min_random_finalize_ops: 0,
                max_random_finalize_ops: 0,
                require_state_update: true,
            },
            verification_limits: VerificationLimits::default(),
            execution_policy: ExecutionPolicy {
                floats: FloatPolicy::FiniteOutputs,
            },
        }
    }

    #[test]
    fn bounded_discovery_finds_sum_recurrence_without_target_ast() {
        let archive = run_scientific_experiment(&sum_config(), &sum_dataset()).unwrap();
        assert!(archive.success, "bounded negative evidence: {archive:#?}");
        let exact = archive
            .pareto
            .iter()
            .find(|entry| entry.fitness.correctness.exact)
            .unwrap();
        assert!(
            exact
                .program
                .step
                .ops
                .iter()
                .any(|op| matches!(op, super::super::Op::Add(_)))
        );
        assert!(archive.diagnostics.candidates_attempted <= 256);
    }

    #[test]
    fn complete_experiment_replays_bit_identically() {
        let config = sum_config();
        let dataset = sum_dataset();
        let first = run_scientific_experiment(&config, &dataset).unwrap();
        let second = run_scientific_experiment(&config, &dataset).unwrap();
        assert_eq!(first, second);
        assert!(replay_scientific_experiment(&first, &dataset).unwrap());
    }

    #[test]
    fn counterexample_validation_catches_public_field_bypass() {
        let malformed = ValueTensor {
            dtype: DType::Bool,
            shape: vec![],
            data: vec![2.0],
        };
        assert!(matches!(
            CounterexampleSet::new(
                "bad",
                vec![CounterexampleCase {
                    inputs: vec![malformed],
                    items: vec![],
                    expected_outputs: vec![scalar(0.0)],
                }]
            ),
            Err(CounterexampleError::InvalidTensor {
                role: CaseTensorRole::Input,
                ..
            })
        ));
    }

    #[test]
    fn digest_is_not_used_as_structural_equality() {
        let archive = run_scientific_experiment(&sum_config(), &sum_dataset()).unwrap();
        let entry = archive.pareto.first().unwrap().clone();
        let mut different = entry.clone();
        different.canonical_bytes.push(0);
        different.digest.clone_from(&entry.digest);
        let mut pareto = ParetoArchive::new(4);
        assert_eq!(pareto.consider(entry), ArchiveDecision::Admitted);
        assert_ne!(pareto.consider(different), ArchiveDecision::Duplicate);
    }

    #[test]
    fn invalid_setup_is_structured() {
        let mut config = sum_config();
        config.max_candidates = 0;
        assert_eq!(
            run_scientific_experiment(&config, &sum_dataset()),
            Err(ScientificExperimentError::ZeroCandidates)
        );
    }
}
