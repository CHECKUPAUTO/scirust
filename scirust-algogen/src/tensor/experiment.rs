//! Deterministic discovery-experiment runner.
//!
//! Given a validated [`TensorProblem`], the runner creates the initial
//! population, evaluates it, evolves for the generation budget, records the best
//! and Pareto-front information at every generation, and stops early only when
//! the explicit success criteria are met. It never consults wall-clock time.
//! For the same problem and seed it produces an identical [`ExperimentArchive`];
//! optional Rayon evaluation yields bit-identical reports and archives.

use serde::{Deserialize, Serialize};

use super::archive::{ExperimentArchive, HallOfFame, HallOfFameEntry};
use super::canonical::canonical_bytes;
use super::cost::CostReport;
use super::dataset::Dataset;
use super::fitness::FitnessReport;
use super::generate::GenerationError;
use super::ir::TensorProgram;
use super::population::{Population, dominates, rank};
use super::problem::{ProblemError, TensorProblem};
use super::rng::DeterministicRng;
use super::verify::VerificationLimits;

/// A deterministic experiment failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentError {
    Problem(ProblemError),
    Generation(GenerationError),
    Digest(super::digest::DigestError),
}

impl From<ProblemError> for ExperimentError {
    fn from(error: ProblemError) -> Self {
        Self::Problem(error)
    }
}

impl From<super::digest::DigestError> for ExperimentError {
    fn from(error: super::digest::DigestError) -> Self {
        Self::Digest(error)
    }
}

/// Options controlling a run (never affect deterministic content).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOptions {
    pub hall_of_fame_capacity: usize,
    /// Use Rayon evaluation when the `rayon` feature is enabled. Results are
    /// identical to sequential evaluation either way.
    pub parallel: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            hall_of_fame_capacity: 16,
            parallel: false,
        }
    }
}

/// Deterministic per-generation statistics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: usize,
    pub best_loss: f64,
    pub best_cost: CostReport,
    pub best_fingerprint: u128,
    pub valid_individuals: usize,
    pub invalid_individuals: usize,
    pub exact_solutions: usize,
    pub pareto_front_size: usize,
    /// Number of structurally distinct programs, by authoritative canonical
    /// bytes (not by a collidable fingerprint).
    pub diversity: usize,
}

/// Deterministic summary of a final population.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PopulationSummary {
    pub size: usize,
    pub valid: usize,
    pub invalid: usize,
    pub distinct: usize,
    pub best_fingerprint: u128,
}

#[cfg(feature = "rayon")]
fn evaluate(
    population: &Population,
    dataset: &Dataset,
    limits: VerificationLimits,
    parallel: bool,
) -> Vec<FitnessReport> {
    if parallel
    {
        super::fitness::evaluate_population_rayon(population.programs(), dataset, limits)
    }
    else
    {
        population.evaluate(dataset, limits)
    }
}

#[cfg(not(feature = "rayon"))]
fn evaluate(
    population: &Population,
    dataset: &Dataset,
    limits: VerificationLimits,
    _parallel: bool,
) -> Vec<FitnessReport> {
    population.evaluate(dataset, limits)
}

/// Run a reproducible discovery experiment, returning the deterministic archive.
///
/// `success` is defined as: the best individual (top-ranked across all executed
/// generations) satisfies the explicit success criteria. Early stopping happens
/// exactly when the running best first satisfies them, so `success` is always
/// consistent with `is_met(best)`.
pub fn run_experiment(
    problem: &TensorProblem,
    options: RunOptions,
) -> Result<ExperimentArchive, ExperimentError> {
    problem.validate()?;
    let dataset = problem.dataset()?;
    let limits = problem.verification_limits();
    let evolution = &problem.evolution;

    let mut rng = DeterministicRng::new(problem.seed);
    let mut hall = HallOfFame::new(options.hall_of_fame_capacity);
    let mut history = Vec::new();

    let mut population = Population::generate(
        &evolution.generation,
        evolution.population_size,
        limits,
        &mut rng,
    )
    .map_err(ExperimentError::Generation)?;
    let mut reports = evaluate(&population, &dataset, limits, options.parallel);

    let mut best = ingest_generation(
        0,
        problem.seed,
        &population,
        &reports,
        &mut hall,
        &mut history,
    );
    let mut generations_executed = 0usize;
    let mut success = problem.success.is_met(&best.fitness);

    if !success
    {
        for generation in 1..=evolution.generations
        {
            population = population.advance(&reports, evolution, limits, &mut rng);
            reports = evaluate(&population, &dataset, limits, options.parallel);
            generations_executed = generation;

            let generation_best = ingest_generation(
                generation,
                problem.seed,
                &population,
                &reports,
                &mut hall,
                &mut history,
            );
            best = better_of(best, generation_best);

            if problem.success.is_met(&best.fitness)
            {
                success = true;
                break;
            }
        }
    }

    let final_population = summarise_population(&population, &reports);

    ExperimentArchive::build(
        problem.clone(),
        problem.seed,
        success,
        generations_executed,
        history,
        final_population,
        options.hall_of_fame_capacity,
        hall.into_entries(),
        best,
    )
}

/// Record one generation and update the hall of fame; return the generation's
/// best individual (top-ranked).
fn ingest_generation(
    generation: usize,
    seed: u64,
    population: &Population,
    reports: &[FitnessReport],
    hall: &mut HallOfFame,
    history: &mut Vec<GenerationRecord>,
) -> HallOfFameEntry {
    let order = rank(population.programs(), reports);
    let best_index = order[0];
    let best = &reports[best_index];

    for (program, report) in population.programs().iter().zip(reports)
    {
        hall.consider(program, *report, generation, seed);
    }

    history.push(GenerationRecord {
        generation,
        best_loss: best.loss,
        best_cost: best.cost,
        best_fingerprint: best.fingerprint,
        valid_individuals: reports.iter().filter(|report| report.evaluated).count(),
        invalid_individuals: reports.iter().filter(|report| !report.evaluated).count(),
        exact_solutions: reports.iter().filter(|report| is_exact(report)).count(),
        pareto_front_size: pareto_front_size(reports),
        diversity: distinct_programs(population.programs()),
    });

    HallOfFameEntry {
        program: population.programs()[best_index].clone(),
        fitness: *best,
        fingerprint: best.fingerprint,
        generation,
        seed,
    }
}

fn summarise_population(population: &Population, reports: &[FitnessReport]) -> PopulationSummary {
    let best_fingerprint = if reports.is_empty()
    {
        0
    }
    else
    {
        reports[rank(population.programs(), reports)[0]].fingerprint
    };
    PopulationSummary {
        size: population.len(),
        valid: reports.iter().filter(|report| report.evaluated).count(),
        invalid: reports.iter().filter(|report| !report.evaluated).count(),
        distinct: distinct_programs(population.programs()),
        best_fingerprint,
    }
}

/// The better of two candidate best entries under the ranking order.
fn better_of(first: HallOfFameEntry, second: HallOfFameEntry) -> HallOfFameEntry {
    let programs = [first.program.clone(), second.program.clone()];
    let reports = [first.fitness, second.fitness];
    if rank(&programs, &reports)[0] == 0
    {
        first
    }
    else
    {
        second
    }
}

/// A program that solves every case exactly.
fn is_exact(report: &FitnessReport) -> bool {
    report.evaluated && report.failed_cases == 0 && report.loss == 0.0
}

/// Number of individuals in the non-dominated front.
fn pareto_front_size(reports: &[FitnessReport]) -> usize {
    reports
        .iter()
        .enumerate()
        .filter(|(index, report)| {
            !reports
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != *index && dominates(candidate, report))
        })
        .count()
}

/// Number of structurally distinct programs, by authoritative canonical bytes.
fn distinct_programs(programs: &[TensorProgram]) -> usize {
    let mut bytes: Vec<Vec<u8>> = programs.iter().map(canonical_bytes).collect();
    bytes.sort_unstable();
    bytes.dedup();
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::SuccessCriteria;
    use crate::tensor::benchmarks;

    fn tuned(
        mut problem: TensorProblem,
        seed: u64,
        generations: usize,
        success: SuccessCriteria,
    ) -> TensorProblem {
        problem.seed = seed;
        problem.evolution.generations = generations;
        problem.success = success;
        problem
    }

    #[test]
    fn same_problem_and_seed_yield_identical_archives() {
        let problem = benchmarks::relu();
        let first = run_experiment(&problem, RunOptions::default()).unwrap();
        let second = run_experiment(&problem, RunOptions::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.digest, second.digest);
    }

    #[test]
    fn different_seeds_can_diverge() {
        let base = benchmarks::matrix_multiply();
        let a = run_experiment(
            &tuned(base.clone(), 1, 15, SuccessCriteria::max_loss(-1.0)),
            RunOptions::default(),
        )
        .unwrap();
        let b = run_experiment(
            &tuned(base, 2, 15, SuccessCriteria::max_loss(-1.0)),
            RunOptions::default(),
        )
        .unwrap();
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn early_stopping_is_deterministic() {
        // A trivially satisfiable criterion is met by the initial population, so
        // the run stops at generation 0 every time.
        let problem = tuned(
            benchmarks::relu(),
            42,
            20,
            SuccessCriteria::max_loss(1.0e18),
        );
        let first = run_experiment(&problem, RunOptions::default()).unwrap();
        let second = run_experiment(&problem, RunOptions::default()).unwrap();
        assert!(first.success);
        assert_eq!(first.generations_executed, 0);
        assert_eq!(first, second);
    }

    #[test]
    fn history_length_is_exactly_generations_plus_one() {
        // Unreachable criterion: the run uses the full budget.
        let budget = 5;
        let problem = tuned(
            benchmarks::relu(),
            7,
            budget,
            SuccessCriteria::max_loss(-1.0),
        );
        let archive = run_experiment(&problem, RunOptions::default()).unwrap();
        assert!(!archive.success);
        assert_eq!(archive.generations_executed, budget);
        assert_eq!(archive.history.len(), budget + 1);
    }

    #[test]
    fn empty_and_minimal_budgets_do_not_panic() {
        let zero = tuned(benchmarks::relu(), 1, 0, SuccessCriteria::max_loss(-1.0));
        let archive = run_experiment(&zero, RunOptions::default()).unwrap();
        assert_eq!(archive.generations_executed, 0);
        assert_eq!(archive.history.len(), 1);

        let one = tuned(benchmarks::relu(), 1, 1, SuccessCriteria::max_loss(-1.0));
        let archive = run_experiment(&one, RunOptions::default()).unwrap();
        assert_eq!(archive.generations_executed, 1);
        assert_eq!(archive.history.len(), 2);
    }

    #[test]
    fn archive_carries_no_wall_clock_and_two_runs_are_equal() {
        // The archive type has no timing field; equal runs prove nothing
        // non-deterministic (such as wall-clock) leaks into its content.
        let problem = benchmarks::transpose();
        let a = run_experiment(&problem, RunOptions::default()).unwrap();
        let b = run_experiment(&problem, RunOptions::default()).unwrap();
        assert_eq!(a, b);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn sequential_and_rayon_runs_are_identical() {
        let problem = benchmarks::matrix_multiply();
        let sequential = run_experiment(
            &problem,
            RunOptions {
                hall_of_fame_capacity: 16,
                parallel: false,
            },
        )
        .unwrap();
        let parallel = run_experiment(
            &problem,
            RunOptions {
                hall_of_fame_capacity: 16,
                parallel: true,
            },
        )
        .unwrap();
        assert_eq!(sequential, parallel);
        assert_eq!(sequential.digest, parallel.digest);
    }
}
