//! Counterexample-guided inductive synthesis (CEGIS) over the V2 search.
//!
//! The outer loop that turns one-shot search into feedback-driven discovery:
//!
//! 1. seed a dataset from deterministic probes of the *target oracle*
//!    (the target's AST is never revealed to the generator — only its
//!    input/output behavior);
//! 2. run the standard deterministic experiment;
//! 3. if no exact candidate emerged, harvest fresh diverging probes as new
//!    counterexamples and repeat within a bounded round budget.
//!
//! Every ingredient stays deterministic: identical target, grammar, config
//! and seed reproduce byte-identical outcomes. Discovery success remains
//! bounded negative evidence, never a proof of general correctness.

use serde::{Deserialize, Serialize};

use super::interpret::{ExecutionPolicy, ValueTensor, execute_program};
use super::ir::ResearchProgram;
use super::recognize::{MAX_PROBED_STEPS, probe_tensor};
use super::search::{
    CounterexampleCase, CounterexampleSet, ExperimentConfig, ScientificExperimentArchive,
    ScientificExperimentError, run_scientific_experiment,
};
use super::verify::VerificationLimits;

/// Bounded CEGIS configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisConfig {
    /// The per-round experiment configuration (grammar, request, budgets).
    pub base: ExperimentConfig,
    /// Maximum refinement rounds; `0` is rejected.
    pub max_rounds: usize,
}

/// CEGIS setup failure.
#[derive(Debug, Clone, PartialEq)]
pub enum CegisError {
    InvalidExperiment(ScientificExperimentError),
    /// The target oracle itself cannot execute under the configured policy,
    /// so input/output behavior cannot even be sampled.
    TargetNotExecutable(String),
    EmptyRounds,
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
            Self::EmptyRounds => formatter.write_str("max_rounds must be non-zero"),
        }
    }
}

impl std::error::Error for CegisError {}

/// One completed round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CegisRoundReport {
    pub round: usize,
    pub dataset_cases: usize,
    pub archive_size: usize,
    pub exact_candidate_found: bool,
    /// Counterexamples added by this round's harvest.
    pub harvested_cases: usize,
}

/// Full outcome of a bounded CEGIS run.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisOutcome {
    /// A candidate exact on every case of the final dataset, if one emerged.
    pub discovered: Option<ResearchProgram>,
    pub rounds: Vec<CegisRoundReport>,
    /// Final refined dataset (seed + harvested counterexamples).
    pub dataset: CounterexampleSet,
    pub archives: Vec<ScientificExperimentArchive>,
}

/// Run bounded counterexample-guided discovery against `target`.
///
/// The generator only ever sees input/output pairs; the target's structure
/// stays sealed inside the oracle, preserving the no-target-leakage rule.
pub fn cedis_discover(
    target: &ResearchProgram,
    config: &CegisConfig,
) -> Result<CegisOutcome, CegisError> {
    if config.max_rounds == 0
    {
        return Err(CegisError::EmptyRounds);
    }
    let limits = config.base.verification_limits;
    let policy = config.base.execution_policy;

    // ---- seed dataset from target probes -----------------------------------
    let mut cases = Vec::new();
    for salt in 0..4
    {
        let case = oracle_case(target, salt, policy, limits)
            .ok_or_else(|| CegisError::TargetNotExecutable(format!("probe salt {salt} failed")))?;
        cases.push(case);
    }
    let mut dataset = CounterexampleSet::new("cegis-seed", cases.clone())
        .map_err(|error| CegisError::TargetNotExecutable(error.to_string()))?;

    let mut rounds = Vec::new();
    let mut archives = Vec::new();
    let mut discovered = None;

    for round in 0..config.max_rounds
    {
        // Fresh per-round randomness while keeping global determinism.
        let mut round_config = config.base.clone();
        round_config.seed = config.base.seed ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let archive = run_scientific_experiment(&round_config, &dataset)
            .map_err(CegisError::InvalidExperiment)?;

        let exact = archive
            .pareto
            .iter()
            .find(|entry| entry.fitness.correctness.exact)
            .cloned();
        let exact_found = exact.is_some();
        if let Some(entry) = exact
        {
            discovered = Some(entry.program);
        }

        let mut harvested_cases = 0usize;
        if !exact_found && round + 1 < config.max_rounds
        {
            harvested_cases =
                harvest_counterexamples(target, &archive, &mut cases, round, policy, limits);
            if harvested_cases > 0
            {
                dataset = CounterexampleSet::new("cegis-refined", cases.clone())
                    .map_err(|error| CegisError::TargetNotExecutable(error.to_string()))?;
            }
        }

        rounds.push(CegisRoundReport {
            round,
            dataset_cases: dataset.cases.len(),
            archive_size: archive.pareto.len(),
            exact_candidate_found: exact_found,
            harvested_cases,
        });
        archives.push(archive);
        if discovered.is_some()
        {
            break;
        }
    }

    Ok(CegisOutcome {
        discovered,
        rounds,
        dataset,
        archives,
    })
}

/// Sample one input/output pair of the target on deterministic probes.
fn oracle_case(
    target: &ResearchProgram,
    salt: usize,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Option<CounterexampleCase> {
    let steps = target.steps.min(MAX_PROBED_STEPS);
    let inputs: Vec<ValueTensor> = target
        .inputs
        .iter()
        .enumerate()
        .map(|(index, value_type)| probe_tensor(value_type, index + salt))
        .collect();
    let mut items = Vec::new();
    for step in 0..steps
    {
        for (slot, value_type) in target.items.iter().enumerate()
        {
            items.push(probe_tensor(value_type, step as usize * 5 + slot + salt));
        }
    }
    let result = execute_program(target, &inputs, &items, policy, limits).ok()?;
    Some(CounterexampleCase {
        inputs,
        items,
        expected_outputs: result.outputs,
    })
}

/// Execute the best non-exact candidates on fresh probes; wherever their
/// outputs diverge from the oracle, the probe becomes a permanent
/// counterexample. Returns how many cases were added.
#[allow(clippy::too_many_arguments)]
fn harvest_counterexamples(
    target: &ResearchProgram,
    archive: &ScientificExperimentArchive,
    cases: &mut Vec<CounterexampleCase>,
    round: usize,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> usize {
    let budget = 2usize;
    let mut added = 0usize;
    // Deterministic order: archive rank, then candidate index.
    for entry in archive.pareto.iter().take(budget)
    {
        let salt = 100 + round * 10 + added;
        if let Some(case) = oracle_case(target, salt, policy, limits)
        {
            let signature_ok = entry.program.inputs.len() == case.inputs.len();
            if !signature_ok
            {
                continue;
            }
            let diverges =
                match execute_program(&entry.program, &case.inputs, &case.items, policy, limits)
                {
                    Ok(result) => result.outputs.iter().zip(&case.expected_outputs).any(
                        |(actual, expected)| {
                            actual.value_type() != expected.value_type()
                                || actual
                                    .data
                                    .iter()
                                    .zip(&expected.data)
                                    .any(|(&a, &e)| a.to_bits() != e.to_bits())
                        },
                    ),
                    Err(_) => true,
                };
            if diverges
            {
                cases.push(case);
                added += 1;
            }
        }
    }
    let _ = limits;
    added
}
