//! Counterexample-guided inductive synthesis (CEGIS) over the V2 search.
//!
//! The outer loop that turns one-shot search into feedback-driven discovery,
//! under an explicitly bounded protocol:
//!
//! 1. seed a dataset from deterministic probes of the *target oracle*
//!    (the target's AST is never revealed to the generator — only its
//!    input/output behavior);
//! 2. run the standard deterministic experiment on the current dataset;
//! 3. a candidate that is merely exact on the current dataset has NOT earned
//!    discovery: it must additionally survive a fresh-falsification phase —
//!    up to [`CegisConfig::fresh_probes_per_round`] new oracle probes it has
//!    never trained on. Any divergence (value mismatch, output-contract
//!    mismatch, or a rejected execution where the oracle succeeded) turns the
//!    probe into a permanent counterexample and the loop continues;
//! 4. only a candidate that survived the declared fresh-probe budget is
//!    reported as [`DiscoveryStatus::SurvivedFreshFalsification`].
//!
//! Every ingredient stays deterministic: identical target, grammar, config
//! and seed reproduce byte-identical outcomes. Success is deliberately
//! conservative terminology — *survived fresh falsification within the
//! declared budget* is finite evidence, never a proof of general correctness.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::interpret::{ExecutionPolicy, ValueTensor, execute_program};
use super::ir::ResearchProgram;
use super::recognize::{MAX_PROBED_STEPS, probe_tensor};
use super::search::{
    CounterexampleCase, CounterexampleSet, ExperimentConfig, ScientificExperimentArchive,
    ScientificExperimentError, run_scientific_experiment,
};
use super::verify::VerificationLimits;

/// First salt of the fresh-probe stream. Seed probes live in
/// `[0, seed_probes)`; every synthesis-driven probe lives at or beyond this
/// base so the two spaces cannot collide for any sane configuration.
const FRESH_SALT_BASE: usize = 1_000_000;

/// Bounded CEGIS configuration. Every experimental budget is explicit:
/// there are no hidden magic numbers in the loop.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisConfig {
    /// The per-round experiment configuration (grammar, request, budgets).
    pub base: ExperimentConfig,
    /// Maximum refinement rounds; `0` is rejected.
    pub max_rounds: usize,
    /// Number of oracle probes seeding the initial dataset; `0` is rejected.
    pub seed_probes: usize,
    /// Fresh oracle probes per round for harvesting (no exact candidate) and
    /// for the fresh-falsification phase guarding a candidate; `0` rejected.
    pub fresh_probes_per_round: usize,
    /// How many top-ranked Pareto candidates are falsified per harvest
    /// round; `0` is rejected.
    pub candidates_to_falsify: usize,
}

/// CEGIS setup or protocol failure.
#[derive(Debug, Clone, PartialEq)]
pub enum CegisError {
    InvalidExperiment(ScientificExperimentError),
    /// The target oracle itself cannot execute under the configured policy,
    /// so input/output behavior cannot even be sampled.
    TargetNotExecutable(String),
    /// A declared budget is invalid (`max_rounds`, probe counts…).
    InvalidBudget(&'static str),
}

impl std::fmt::Display for CegisError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::InvalidExperiment(error) => write!(formatter, "invalid experiment: {error}"),
            Self::TargetNotExecutable(reason) => write!(
                formatter,
                "target oracle is not executable under the configured policy: {reason}"
            ),
            Self::InvalidBudget(name) =>
            {
                write!(formatter, "invalid CEGIS budget: {name} must be non-zero")
            },
        }
    }
}

impl std::error::Error for CegisError {}

/// Why a candidate failed fresh falsification. Negative evidence is
/// first-class: the reason is recorded, never collapsed into a count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FalsificationReason {
    /// The candidate rejected a probe the oracle executed (or could not
    /// consume it at all). The detail distinguishes the two.
    RejectedOrIncompatible(String),
    /// Both executed and produced different bits.
    ValueDivergence {
        output: usize,
        element: usize,
        candidate_bits: u64,
        oracle_bits: u64,
    },
}

/// One recorded falsification event: who, when, on what evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreshFalsification {
    /// Round in which the candidate was challenged.
    pub round: usize,
    /// Canonical digest of the falsified candidate.
    pub candidate_digest: String,
    /// Deterministic salt identifying the falsifying probe.
    pub probe_salt: usize,
    /// Index of the appended counterexample in the refined dataset, if the
    /// probe was new; `None` when it duplicated known evidence.
    pub dataset_case_index: Option<usize>,
    pub reason: FalsificationReason,
}

/// Final standing of a completed CEGIS run. Wording stays conservative:
/// survival is bounded finite evidence, never discovery-as-proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscoveryStatus {
    /// No candidate was ever exact on its round's dataset within the round
    /// budget.
    NoCandidateWithinRoundBudget,
    /// At least one candidate was falsified by fresh probes and the round
    /// budget ran out before another survived. The last event is reported.
    LastCandidateFalsified {
        round: usize,
        candidate_digest: String,
    },
    /// A candidate was exact on its training dataset AND agreed with the
    /// oracle on the declared fresh-probe budget. Bounded evidence only.
    SurvivedFreshFalsification {
        compared_probes: usize,
        candidate_digest: String,
    },
}

/// One completed round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CegisRoundReport {
    pub round: usize,
    pub dataset_cases: usize,
    pub archive_size: usize,
    /// Whether an exact-on-dataset candidate emerged this round.
    pub exact_candidate_found: bool,
    /// Digest of the candidate submitted to fresh falsification, if any.
    pub challenged_candidate_digest: Option<String>,
    /// Fresh oracle comparisons performed during falsification this round.
    pub fresh_comparisons: usize,
    /// Counterexamples appended by this round (harvest or falsification).
    pub harvested_cases: usize,
    /// Probes skipped because their fingerprint was already known.
    pub duplicates_skipped: usize,
}

/// Full outcome of a bounded CEGIS run.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisOutcome {
    /// Set only under [`DiscoveryStatus::SurvivedFreshFalsification`];
    /// training-set fitness alone never populates it.
    pub discovered: Option<ResearchProgram>,
    pub status: DiscoveryStatus,
    pub rounds: Vec<CegisRoundReport>,
    /// Final refined dataset (seed probes + harvested counterexamples),
    /// including every falsifying probe.
    pub dataset: CounterexampleSet,
    pub archives: Vec<ScientificExperimentArchive>,
    /// All falsification events in order: the negative-evidence ledger.
    pub falsifications: Vec<FreshFalsification>,
}

/// Verdict of comparing one candidate against one oracle case.
#[derive(Debug, Clone, PartialEq)]
enum CandidateVerdict {
    /// Bit-identical outputs under the shared output contract.
    Agrees,
    Disagrees(FalsificationReason),
}

/// Sample one input/output pair of the target on a deterministic probe.
///
/// The target is used strictly as an executable oracle; nothing about its
/// structure reaches the caller beyond the sampled behavior.
fn oracle_case(
    target: &ResearchProgram,
    salt: usize,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Result<CounterexampleCase, String> {
    let steps = target.steps.min(MAX_PROBED_STEPS);
    let inputs: Vec<ValueTensor> = target
        .inputs
        .iter()
        .enumerate()
        .map(|(index, value_type)| {
            probe_tensor(value_type, index + salt)
                .map_err(|error| format!("input probe {index}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut items = Vec::new();
    for step in 0..steps
    {
        for (slot, value_type) in target.items.iter().enumerate()
        {
            items.push(
                probe_tensor(value_type, step as usize * 5 + slot + salt)
                    .map_err(|error| format!("item probe step {step} slot {slot}: {error}"))?,
            );
        }
    }
    let result = execute_program(target, &inputs, &items, policy, limits)
        .map_err(|error| format!("probe salt {salt}: {error}"))?;
    Ok(CounterexampleCase {
        inputs,
        items,
        expected_outputs: result.outputs,
    })
}

/// Compare a candidate's behavior against a recorded oracle case.
fn compare_candidate_to_case(
    candidate: &ResearchProgram,
    case: &CounterexampleCase,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> CandidateVerdict {
    match execute_program(candidate, &case.inputs, &case.items, policy, limits)
    {
        Ok(result) =>
        {
            if result.outputs.len() != case.expected_outputs.len()
            {
                return CandidateVerdict::Disagrees(FalsificationReason::RejectedOrIncompatible(
                    format!(
                        "output arity {} vs oracle {}",
                        result.outputs.len(),
                        case.expected_outputs.len()
                    ),
                ));
            }
            for (output, (actual, expected)) in result
                .outputs
                .iter()
                .zip(&case.expected_outputs)
                .enumerate()
            {
                if actual.value_type() != expected.value_type()
                {
                    return CandidateVerdict::Disagrees(
                        FalsificationReason::RejectedOrIncompatible(format!(
                            "output {output} type differs from the oracle"
                        )),
                    );
                }
                if actual.data.len() != expected.data.len()
                {
                    return CandidateVerdict::Disagrees(
                        FalsificationReason::RejectedOrIncompatible(format!(
                            "output {output} length differs from the oracle"
                        )),
                    );
                }
                for (element, (&a, &e)) in actual.data.iter().zip(&expected.data).enumerate()
                {
                    if a.to_bits() != e.to_bits()
                    {
                        return CandidateVerdict::Disagrees(FalsificationReason::ValueDivergence {
                            output,
                            element,
                            candidate_bits: a.to_bits(),
                            oracle_bits: e.to_bits(),
                        });
                    }
                }
            }
            CandidateVerdict::Agrees
        },
        Err(error) => CandidateVerdict::Disagrees(FalsificationReason::RejectedOrIncompatible(
            error.to_string(),
        )),
    }
}

/// Stable fingerprint of a case's observable content, used to prevent
/// duplicate counterexamples from being re-added silently.
fn case_fingerprint(case: &CounterexampleCase) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SCIRUST-RIR2-CEGIS-CASE\0");
    for tensors in [&case.inputs, &case.items, &case.expected_outputs]
    {
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        for tensor in tensors
        {
            bytes.push(tensor.dtype.tag());
            bytes.extend_from_slice(&(tensor.shape.len() as u64).to_le_bytes());
            for &dimension in &tensor.shape
            {
                bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
            }
            for &value in &tensor.data
            {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Deterministic fresh-probe salt for `(round, rank, attempt)`. Strides are
/// checked so absurd configurations produce an error instead of a collision.
fn fresh_salt(config: &CegisConfig, round: usize, rank: usize, attempt: usize) -> usize {
    let stride = config
        .candidates_to_falsify
        .checked_mul(config.fresh_probes_per_round)
        .expect("budget product checked at construction");
    FRESH_SALT_BASE
        + round
            .checked_mul(stride)
            .expect("salt space overflow checked at construction")
        + rank * config.fresh_probes_per_round
        + attempt
}

/// Run bounded counterexample-guided discovery against `target`.
///
/// The generator only ever sees input/output pairs; the target's structure
/// stays sealed inside the oracle, preserving the no-target-leakage rule:
/// [`run_scientific_experiment`] receives no reference to the target at all.
pub fn cegis_discover(
    target: &ResearchProgram,
    config: &CegisConfig,
) -> Result<CegisOutcome, CegisError> {
    validate_budgets(config)?;
    let limits = config.base.verification_limits;
    let policy = config.base.execution_policy;

    // ---- seed dataset from target probes -----------------------------------
    let mut cases = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for salt in 0..config.seed_probes
    {
        let case = oracle_case(target, salt, policy, limits)
            .map_err(|reason| CegisError::TargetNotExecutable(format!("seed probe: {reason}")))?;
        seen.insert(case_fingerprint(&case));
        cases.push(case);
    }
    let mut dataset = CounterexampleSet::new("cegis-seed", cases.clone())
        .map_err(|error| CegisError::TargetNotExecutable(error.to_string()))?;

    let mut rounds = Vec::new();
    let mut archives = Vec::new();
    let mut falsifications = Vec::new();
    let mut discovered: Option<(usize, String, ResearchProgram, usize)> = None;

    for round in 0..config.max_rounds
    {
        // Fresh per-round randomness while keeping global determinism.
        let mut round_config = config.base.clone();
        round_config.seed = config.base.seed ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let archive = run_scientific_experiment(&round_config, &dataset)
            .map_err(CegisError::InvalidExperiment)?;

        // Exact on the CURRENT dataset is necessary but not sufficient.
        let challenger = archive
            .pareto
            .iter()
            .find(|entry| entry.fitness.correctness.exact);

        let mut harvested_cases = 0usize;
        let mut fresh_comparisons = 0usize;
        let mut duplicates_skipped = 0usize;
        let mut challenged_digest = None;

        match challenger
        {
            None =>
            {
                // No exact candidate: harvest divergences from the ranked
                // front to enrich the next round's dataset.
                if round + 1 < config.max_rounds
                {
                    for (rank, entry) in archive
                        .pareto
                        .iter()
                        .take(config.candidates_to_falsify)
                        .enumerate()
                    {
                        if !candidate_consumes_signature(&entry.program, &cases[0])
                        {
                            continue;
                        }
                        for attempt in 0..config.fresh_probes_per_round
                        {
                            let salt = fresh_salt(config, round, rank, attempt);
                            let Some(case) = oracle_case(target, salt, policy, limits).ok()
                            else
                            {
                                continue;
                            };
                            fresh_comparisons += 1;
                            if let CandidateVerdict::Disagrees(_) =
                                compare_candidate_to_case(&entry.program, &case, policy, limits)
                            {
                                let fingerprint = case_fingerprint(&case);
                                if seen.insert(fingerprint)
                                {
                                    cases.push(case);
                                    harvested_cases += 1;
                                }
                                else
                                {
                                    duplicates_skipped += 1;
                                }
                                break;
                            }
                        }
                    }
                    if harvested_cases > 0
                    {
                        dataset = CounterexampleSet::new("cegis-refined", cases.clone())
                            .map_err(|error| CegisError::TargetNotExecutable(error.to_string()))?;
                    }
                }
            },
            Some(entry) =>
            {
                // Fresh-falsification phase: the candidate has never seen
                // these probes; agreement is bounded evidence only.
                challenged_digest = Some(entry.digest.clone());
                let mut falsified: Option<(usize, CounterexampleCase, FalsificationReason)> = None;
                for attempt in 0..config.fresh_probes_per_round
                {
                    let salt = fresh_salt(config, round, 0, attempt);
                    let case = oracle_case(target, salt, policy, limits).map_err(|reason| {
                        CegisError::TargetNotExecutable(format!("fresh probe: {reason}"))
                    })?;
                    fresh_comparisons += 1;
                    if let CandidateVerdict::Disagrees(reason) =
                        compare_candidate_to_case(&entry.program, &case, policy, limits)
                    {
                        falsified = Some((salt, case, reason));
                        break;
                    }
                }
                match falsified
                {
                    None =>
                    {
                        discovered = Some((
                            round,
                            entry.digest.clone(),
                            entry.program.clone(),
                            config.fresh_probes_per_round,
                        ));
                    },
                    Some((salt, case, reason)) =>
                    {
                        let fingerprint = case_fingerprint(&case);
                        let dataset_case_index = if seen.insert(fingerprint)
                        {
                            cases.push(case);
                            harvested_cases += 1;
                            Some(cases.len() - 1)
                        }
                        else
                        {
                            duplicates_skipped += 1;
                            None
                        };
                        falsifications.push(FreshFalsification {
                            round,
                            candidate_digest: entry.digest.clone(),
                            probe_salt: salt,
                            dataset_case_index,
                            reason,
                        });
                        dataset = CounterexampleSet::new("cegis-refined", cases.clone())
                            .map_err(|error| CegisError::TargetNotExecutable(error.to_string()))?;
                    },
                }
            },
        }

        rounds.push(CegisRoundReport {
            round,
            dataset_cases: dataset.cases.len(),
            archive_size: archive.pareto.len(),
            exact_candidate_found: challenger.is_some(),
            challenged_candidate_digest: challenged_digest,
            fresh_comparisons,
            harvested_cases,
            duplicates_skipped,
        });
        archives.push(archive);
        if discovered.is_some()
        {
            break;
        }
    }

    let status = match &discovered
    {
        Some((_, digest, _, compared)) => DiscoveryStatus::SurvivedFreshFalsification {
            compared_probes: *compared,
            candidate_digest: digest.clone(),
        },
        None => match falsifications.last()
        {
            Some(event) => DiscoveryStatus::LastCandidateFalsified {
                round: event.round,
                candidate_digest: event.candidate_digest.clone(),
            },
            None => DiscoveryStatus::NoCandidateWithinRoundBudget,
        },
    };

    Ok(CegisOutcome {
        discovered: discovered.map(|(_, _, program, _)| program),
        status,
        rounds,
        dataset,
        archives,
        falsifications,
    })
}

/// Whether `program`'s declared signature can consume a probe built for the
/// oracle. Incompatible candidates are skipped during harvest rather than
/// being counted as behavioral divergence.
fn candidate_consumes_signature(program: &ResearchProgram, exemplar: &CounterexampleCase) -> bool {
    program.inputs.len() == exemplar.inputs.len()
        && program
            .inputs
            .iter()
            .zip(&exemplar.inputs)
            .all(|(declared, tensor)| &tensor.value_type() == declared)
        && program.items.len() * program.steps as usize == exemplar.items.len()
}

fn validate_budgets(config: &CegisConfig) -> Result<(), CegisError> {
    if config.max_rounds == 0
    {
        return Err(CegisError::InvalidBudget("max_rounds"));
    }
    if config.seed_probes == 0
    {
        return Err(CegisError::InvalidBudget("seed_probes"));
    }
    if config.fresh_probes_per_round == 0
    {
        return Err(CegisError::InvalidBudget("fresh_probes_per_round"));
    }
    if config.candidates_to_falsify == 0
    {
        return Err(CegisError::InvalidBudget("candidates_to_falsify"));
    }
    let stride = config
        .candidates_to_falsify
        .checked_mul(config.fresh_probes_per_round)
        .ok_or(CegisError::InvalidBudget(
            "candidates_to_falsify × fresh_probes_per_round overflows",
        ))?;
    if config.max_rounds.checked_mul(stride).is_none()
    {
        return Err(CegisError::InvalidBudget(
            "max_rounds × per-round probe stride overflows",
        ));
    }
    if config.seed_probes >= FRESH_SALT_BASE
    {
        return Err(CegisError::InvalidBudget("seed_probes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Bin, Op, Ref, Section};
    use crate::tensor::v2::types::{DType, ScalarValue, ValueType};

    /// The oracle boundary: a candidate that reproduces the target only on
    /// its training case (constant output fitted to a single zero-sum
    /// observation) must be falsified by a fresh nonzero probe, and the
    /// falsification must carry structured value-level evidence.
    #[test]
    fn training_fit_alone_does_not_survive_fresh_falsification() {
        let target = super::super::reference::compensated_sum_recurrence(3);
        let policy = ExecutionPolicy::default();
        let limits = VerificationLimits::default();

        // Training case: three zero items sum to exactly 0.
        let zero_items = vec![
            ValueTensor::scalar_f64(0.0),
            ValueTensor::scalar_f64(0.0),
            ValueTensor::scalar_f64(0.0),
        ];
        let training_case = CounterexampleCase {
            inputs: vec![],
            items: zero_items,
            expected_outputs: vec![ValueTensor::scalar_f64(0.0)],
        };
        // Overfitting candidate: shares the target's recurrence signature
        // but ignores its input stream and always returns the constant 0 —
        // exactly reproducing the single all-zero training observation.
        let zero = || Section::new(vec![Op::Const(ScalarValue::F64(0.0))]);
        let two_zeros = || {
            Section::new(vec![
                Op::Const(ScalarValue::F64(0.0)),
                Op::Const(ScalarValue::F64(0.0)),
            ])
        };
        let overfit = ResearchProgram {
            items: vec![ValueType::scalar(DType::F64)],
            state: vec![ValueType::scalar(DType::F64), ValueType::scalar(DType::F64)],
            steps: 3,
            init: zero(),
            init_state: vec![0, 0],
            step: zero(),
            next_state: vec![0, 0],
            finalize: two_zeros(),
            outputs: vec![0, 1],
            ..ResearchProgram::expression(vec![], Section::default(), vec![])
        };
        assert!(
            candidate_consumes_signature(&overfit, &training_case),
            "signatures must line up for the comparison to be meaningful"
        );

        // Fresh probe: nonzero items, oracle sums to -0.5+... ≠ 0.
        let fresh = oracle_case(&target, FRESH_SALT_BASE, policy, limits)
            .expect("oracle executes on fresh probes");
        assert_ne!(
            fresh.expected_outputs[0].data[0], 0.0,
            "the fresh probe must actually exercise nonzero behavior"
        );
        let verdict = compare_candidate_to_case(&overfit, &fresh, policy, limits);
        match verdict
        {
            CandidateVerdict::Disagrees(FalsificationReason::ValueDivergence {
                candidate_bits,
                ..
            }) =>
            {
                assert_eq!(candidate_bits, 0.0f64.to_bits());
            },
            other => panic!("expected value divergence, got {other:?}"),
        }
    }

    /// Case fingerprints make duplicate counterexamples detectable: the
    /// same probe content must never be archived twice.
    #[test]
    fn identical_probes_share_a_fingerprint_and_distinct_ones_do_not() {
        let fingerprint = |sum: f64| {
            case_fingerprint(&CounterexampleCase {
                inputs: vec![],
                items: vec![
                    ValueTensor::scalar_f64(sum),
                    ValueTensor::scalar_f64(0.0),
                    ValueTensor::scalar_f64(0.0),
                ],
                expected_outputs: vec![ValueTensor::scalar_f64(sum)],
            })
        };
        assert_eq!(fingerprint(1.5), fingerprint(1.5));
        assert_ne!(fingerprint(1.5), fingerprint(-0.25));
        // Signed-zero distinction is preserved at the bit level.
        assert_ne!(fingerprint(0.0), fingerprint(-0.0));
    }

    /// Zero budgets are rejected with named fields, not silent defaults.
    #[test]
    fn zero_budgets_are_named_errors() {
        let base = valid_base_config().base;
        let probe = |mutate: &dyn Fn(&mut CegisConfig)| {
            let mut config = CegisConfig {
                base: base.clone(),
                max_rounds: 3,
                seed_probes: 4,
                fresh_probes_per_round: 2,
                candidates_to_falsify: 2,
            };
            mutate(&mut config);
            validate_budgets(&config)
        };
        assert_eq!(
            probe(&|c: &mut CegisConfig| c.max_rounds = 0),
            Err(CegisError::InvalidBudget("max_rounds"))
        );
        assert_eq!(
            probe(&|c: &mut CegisConfig| c.seed_probes = 0),
            Err(CegisError::InvalidBudget("seed_probes"))
        );
        assert_eq!(
            probe(&|c: &mut CegisConfig| c.fresh_probes_per_round = 0),
            Err(CegisError::InvalidBudget("fresh_probes_per_round"))
        );
        assert_eq!(
            probe(&|c: &mut CegisConfig| c.candidates_to_falsify = 0),
            Err(CegisError::InvalidBudget("candidates_to_falsify"))
        );
    }

    /// A target that rejects the default policy cannot even be sampled.
    #[test]
    fn unexecutable_targets_report_target_not_executable() {
        let config = valid_base_config();
        // 1/0 = +inf: rejected outright by the default finite policy.
        let rejecting_target = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![
                Op::Const(ScalarValue::F64(1.0)),
                Op::Const(ScalarValue::F64(0.0)),
                Op::Div(Bin::new(Ref::Local(0), Ref::Local(1))),
            ]),
            vec![2],
        );
        assert!(matches!(
            cegis_discover(&rejecting_target, &config),
            Err(CegisError::TargetNotExecutable(_))
        ));
    }

    fn valid_base_config() -> CegisConfig {
        CegisConfig {
            base: ExperimentConfig {
                source_revision: "cegis-test".to_string(),
                seed: 1,
                max_candidates: 4,
                archive_capacity: 4,
                stop_on_exact: true,
                grammar: crate::tensor::v2::generate::Grammar::profile(
                    crate::tensor::v2::generate::GrammarProfile::StreamingRecurrence,
                ),
                request: crate::tensor::v2::generate::GenerationRequest {
                    inputs: vec![],
                    items: vec![ValueType::scalar(DType::F64)],
                    state: vec![crate::tensor::v2::generate::StateSpec {
                        value_type: ValueType::scalar(DType::F64),
                        initializer: crate::tensor::v2::generate::StateInitializer::Constant(
                            ScalarValue::F64(0.0),
                        ),
                    }],
                    steps: 3,
                    output_types: vec![ValueType::scalar(DType::F64)],
                    min_random_step_ops: 1,
                    max_random_step_ops: 1,
                    min_random_finalize_ops: 0,
                    max_random_finalize_ops: 0,
                    require_state_update: true,
                },
                verification_limits: VerificationLimits::default(),
                execution_policy: ExecutionPolicy::default(),
            },
            max_rounds: 1,
            seed_probes: 1,
            fresh_probes_per_round: 1,
            candidates_to_falsify: 1,
        }
    }
}
