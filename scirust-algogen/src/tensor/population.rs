//! Deterministic population manager and multi-objective selection.
//!
//! Selection is Pareto-based across the loss and every structural cost metric.
//! Ranking is a strict total order over `(program, report)` pairs: individuals
//! are sorted by non-dominated front, then lexicographically across all
//! objectives, then by the program's authoritative **canonical bytes**, and
//! finally — only for two structurally identical programs — by population index.
//! Canonical bytes are consulted before the index, so ordering never depends on
//! insertion order for distinct programs, nor on a fingerprint hash that could
//! collide. The whole engine is driven by a single explicitly seeded RNG
//! stream, so a seed reproduces every generation exactly.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::canonical::canonical_bytes;
use super::crossover::{CrossoverOutcome, crossover};
use super::dataset::Dataset;
use super::fitness::{FitnessReport, evaluate_population};
use super::generate::{GenerationConfig, GenerationError, generate};
use super::ir::TensorProgram;
use super::mutate::{MutationOutcome, mutate};
use super::rng::DeterministicRng;
use super::verify::VerificationLimits;

/// Configuration for tournament selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TournamentConfig {
    /// Number of competitors drawn per tournament (at least one).
    pub size: usize,
}

/// Configuration for a deterministic evolution run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// How initial individuals (and inserted subprograms) are generated.
    pub generation: GenerationConfig,

    /// Constant number of individuals per generation.
    pub population_size: usize,

    /// Number of offspring generations to run after the initial population.
    pub generations: usize,

    /// Number of top individuals copied unchanged into the next generation.
    pub elitism: usize,

    /// Tournament selection configuration.
    pub tournament: TournamentConfig,

    /// Magnitude bound for mutated `Scale` factors.
    pub scale_magnitude: f32,

    /// Probability in `[0, 1]` of recombining two parents rather than cloning.
    pub crossover_probability: f64,

    /// Probability in `[0, 1]` of mutating each new child.
    pub mutation_probability: f64,
}

/// Per-generation summary statistics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_loss: f64,
    pub best_failed_cases: usize,
}

/// The outcome of a deterministic evolution run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionOutcome {
    pub best_program: TensorProgram,
    pub best_fitness: FitnessReport,
    pub generations_run: usize,
    pub history: Vec<GenerationStats>,
}

/// A deterministic evolution failure.
#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionError {
    InvalidConfig(String),
    DatasetShapeMismatch,
    Generation(GenerationError),
}

impl fmt::Display for EvolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(reason) => write!(formatter, "invalid evolution config: {reason}"),
            Self::DatasetShapeMismatch =>
            {
                write!(
                    formatter,
                    "generation input shapes do not match the dataset input shapes"
                )
            },
            Self::Generation(error) => write!(formatter, "generation failed: {error:?}"),
        }
    }
}

impl Error for EvolutionError {}

/// A fixed-size collection of candidate programs.
#[derive(Debug, Clone, PartialEq)]
pub struct Population {
    individuals: Vec<TensorProgram>,
}

impl Population {
    /// Wrap an explicit list of programs.
    pub fn from_programs(individuals: Vec<TensorProgram>) -> Self {
        Self { individuals }
    }

    /// Generate `size` programs from `config` using the deterministic `rng`.
    pub fn generate(
        config: &GenerationConfig,
        size: usize,
        limits: VerificationLimits,
        rng: &mut DeterministicRng,
    ) -> Result<Self, GenerationError> {
        // `Vec::new` rather than `with_capacity(size)`: a pathological `size`
        // (e.g. `usize::MAX`) must not panic on an impossible reservation.
        let mut individuals = Vec::new();
        for _ in 0..size
        {
            individuals.push(generate(config, limits, rng)?);
        }
        Ok(Self { individuals })
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    pub fn programs(&self) -> &[TensorProgram] {
        &self.individuals
    }

    /// Evaluate every individual sequentially, preserving order.
    pub fn evaluate(&self, dataset: &Dataset, limits: VerificationLimits) -> Vec<FitnessReport> {
        evaluate_population(&self.individuals, dataset, limits)
    }

    /// Produce the next generation of the **same size**.
    ///
    /// The top `elitism` individuals are copied unchanged; the remainder are
    /// offspring from tournament-selected parents, optionally recombined and
    /// mutated. The population size is preserved exactly, so there is no drift.
    pub fn advance(
        &self,
        reports: &[FitnessReport],
        config: &EvolutionConfig,
        limits: VerificationLimits,
        rng: &mut DeterministicRng,
    ) -> Population {
        let size = self.individuals.len();
        if size == 0
        {
            return Population::from_programs(Vec::new());
        }

        let order = rank(&self.individuals, reports);
        let positions = rank_positions(&order, reports.len());
        let input_shapes = &config.generation.input_shapes;
        let elitism = config.elitism.min(size);

        let mut next = Vec::with_capacity(size);
        for &index in order.iter().take(elitism)
        {
            next.push(self.individuals[index].clone());
        }

        while next.len() < size
        {
            let parent_a = &self.individuals[tournament(&positions, config.tournament, rng)];

            let mut child = if chance(rng, config.crossover_probability)
            {
                let partner = tournament(&positions, config.tournament, rng);
                match crossover(
                    parent_a,
                    &self.individuals[partner],
                    input_shapes,
                    limits,
                    rng,
                )
                {
                    CrossoverOutcome::Child(program)
                    | CrossoverOutcome::ParentUnchanged(program) => program,
                }
            }
            else
            {
                parent_a.clone()
            };

            if chance(rng, config.mutation_probability)
            {
                if let MutationOutcome::Mutated { program, .. } =
                    mutate(&child, input_shapes, limits, config.scale_magnitude, rng)
                {
                    child = program;
                }
            }

            next.push(child);
        }

        Population::from_programs(next)
    }
}

/// Whether `first` Pareto-dominates `second`: no worse on every objective and
/// strictly better on at least one.
pub fn dominates(first: &FitnessReport, second: &FitnessReport) -> bool {
    let mut strictly_better = false;
    for ordering in objective_orderings(first, second)
    {
        match ordering
        {
            Ordering::Greater => return false,
            Ordering::Less => strictly_better = true,
            Ordering::Equal =>
            {},
        }
    }
    strictly_better
}

/// Rank `(program, report)` pairs best-first as a strict total order.
///
/// Keys, in priority:
/// 1. non-dominated Pareto front;
/// 2. the lexicographic objective order;
/// 3. the program's authoritative **canonical bytes** — an order-independent,
///    deterministic structural key that depends only on program structure,
///    never on insertion order, a collidable fingerprint hash, `HashMap`
///    iteration, memory addresses, thread scheduling or wall-clock time;
/// 4. the population index, reached only for two structurally identical programs
///    (equal canonical bytes) with otherwise equal objectives.
///
/// `programs` and `reports` must have equal length and correspond element-wise.
/// The comparator is a strict total order, so the result is independent of the
/// underlying sort's stability.
pub fn rank(programs: &[TensorProgram], reports: &[FitnessReport]) -> Vec<usize> {
    debug_assert_eq!(programs.len(), reports.len());
    let count = reports.len();
    let fronts = non_dominated_fronts(reports);
    // Precompute each program's canonical bytes once for O(n log n) comparisons.
    let bytes: Vec<Vec<u8>> = programs.iter().map(canonical_bytes).collect();

    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&left, &right| {
        fronts[left]
            .cmp(&fronts[right])
            .then_with(|| lexicographic(&reports[left], &reports[right]))
            .then_with(|| bytes[left].cmp(&bytes[right]))
            .then_with(|| left.cmp(&right))
    });
    order
}

/// Take the indices of the best `count` individuals from a ranking `order`.
pub fn elite(order: &[usize], count: usize) -> &[usize] {
    let count = count.min(order.len());
    &order[..count]
}

/// Select one individual by tournament: draw `config.size` competitors and keep
/// the one with the best (lowest) rank position. Requires a non-empty
/// population.
pub fn tournament(
    rank_positions: &[usize],
    config: TournamentConfig,
    rng: &mut DeterministicRng,
) -> usize {
    let count = rank_positions.len();
    if count == 0
    {
        // A tournament over an empty population has no valid selection; return a
        // harmless index without indexing rather than panicking. Callers must
        // not use the result for an empty population.
        return 0;
    }

    // Drawing more competitors than the population size cannot enlarge the fixed
    // candidate pool, so the effective size is capped at `count`. This keeps the
    // draw count bounded even for a pathological `config.size` such as
    // `usize::MAX`.
    let size = config.size.clamp(1, count);
    let mut best = rng.below(count);
    let mut best_position = rank_positions[best];

    for _ in 1..size
    {
        let candidate = rng.below(count);
        if rank_positions[candidate] < best_position
        {
            best = candidate;
            best_position = rank_positions[candidate];
        }
    }

    best
}

/// Run a full deterministic evolution.
pub fn evolve(
    config: &EvolutionConfig,
    dataset: &Dataset,
    limits: VerificationLimits,
    seed: u64,
) -> Result<EvolutionOutcome, EvolutionError> {
    if config.population_size == 0
    {
        return Err(EvolutionError::InvalidConfig(
            "population_size must be at least 1".to_string(),
        ));
    }
    if config.tournament.size == 0
    {
        return Err(EvolutionError::InvalidConfig(
            "tournament size must be at least 1".to_string(),
        ));
    }
    if config.elitism > config.population_size
    {
        return Err(EvolutionError::InvalidConfig(
            "elitism must not exceed population_size".to_string(),
        ));
    }
    if config.generation.input_shapes != dataset.input_shapes()
    {
        return Err(EvolutionError::DatasetShapeMismatch);
    }

    let mut rng = DeterministicRng::new(seed);
    let mut population =
        Population::generate(&config.generation, config.population_size, limits, &mut rng)
            .map_err(EvolutionError::Generation)?;
    let mut reports = population.evaluate(dataset, limits);

    let (mut best_program, mut best_fitness) = best_of(&population, &reports);
    let mut history = vec![summarise(0, &best_fitness)];

    for generation in 1..=config.generations
    {
        population = population.advance(&reports, config, limits, &mut rng);
        reports = population.evaluate(dataset, limits);

        let (candidate_program, candidate_fitness) = best_of(&population, &reports);
        if lexicographic(&candidate_fitness, &best_fitness) == Ordering::Less
        {
            best_program = candidate_program;
            best_fitness = candidate_fitness;
        }

        history.push(summarise(
            generation,
            &best_generation_fitness(&population, &reports),
        ));
    }

    Ok(EvolutionOutcome {
        best_program,
        best_fitness,
        generations_run: config.generations,
        history,
    })
}

/// The best program and fitness of a population under the ranking order.
fn best_of(population: &Population, reports: &[FitnessReport]) -> (TensorProgram, FitnessReport) {
    let order = rank(&population.individuals, reports);
    let best = order[0];
    (population.individuals[best].clone(), reports[best])
}

/// The best fitness of a generation under the ranking order.
fn best_generation_fitness(population: &Population, reports: &[FitnessReport]) -> FitnessReport {
    let order = rank(&population.individuals, reports);
    reports[order[0]]
}

fn summarise(generation: usize, fitness: &FitnessReport) -> GenerationStats {
    GenerationStats {
        generation,
        best_loss: fitness.loss,
        best_failed_cases: fitness.failed_cases,
    }
}

/// Objective-by-objective ordering of two reports; `Less` means `first` is
/// better. Every objective is minimised.
fn objective_orderings(first: &FitnessReport, second: &FitnessReport) -> [Ordering; 9] {
    [
        first.loss.total_cmp(&second.loss),
        first.failed_cases.cmp(&second.failed_cases),
        first.cost.estimated_flops.cmp(&second.cost.estimated_flops),
        first
            .cost
            .active_instructions
            .cmp(&second.cost.active_instructions),
        first
            .cost
            .peak_live_elements
            .cmp(&second.cost.peak_live_elements),
        first
            .cost
            .total_active_elements
            .cmp(&second.cost.total_active_elements),
        first
            .cost
            .generated_intermediate_bytes
            .cmp(&second.cost.generated_intermediate_bytes),
        first
            .cost
            .dead_instructions
            .cmp(&second.cost.dead_instructions),
        first.cost.bloat_ratio.total_cmp(&second.cost.bloat_ratio),
    ]
}

/// Total lexicographic order across all objectives (without the index key).
fn lexicographic(first: &FitnessReport, second: &FitnessReport) -> Ordering {
    for ordering in objective_orderings(first, second)
    {
        if ordering != Ordering::Equal
        {
            return ordering;
        }
    }
    Ordering::Equal
}

/// Assign each report its non-dominated front (0 is the best front).
fn non_dominated_fronts(reports: &[FitnessReport]) -> Vec<usize> {
    let count = reports.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut domination_count = vec![0usize; count];

    for i in 0..count
    {
        for j in 0..count
        {
            if i != j && dominates(&reports[i], &reports[j])
            {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            }
        }
    }

    let mut fronts = vec![0usize; count];
    let mut current: Vec<usize> = (0..count).filter(|&i| domination_count[i] == 0).collect();
    let mut front_index = 0usize;

    while !current.is_empty()
    {
        let mut next = Vec::new();
        for &i in &current
        {
            fronts[i] = front_index;
            for &j in &dominated_by[i]
            {
                domination_count[j] -= 1;
                if domination_count[j] == 0
                {
                    next.push(j);
                }
            }
        }
        front_index += 1;
        current = next;
    }

    fronts
}

/// Invert a ranking order into per-individual positions (0 is best).
fn rank_positions(order: &[usize], count: usize) -> Vec<usize> {
    let mut positions = vec![0usize; count];
    for (position, &index) in order.iter().enumerate()
    {
        positions[index] = position;
    }
    positions
}

/// Draw a deterministic biased coin.
fn chance(rng: &mut DeterministicRng, probability: f64) -> bool {
    const SCALE: usize = 1 << 30;
    let probability = probability.clamp(0.0, 1.0);
    let threshold = (probability * SCALE as f64) as usize;
    rng.below(SCALE) < threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cost::CostReport;
    use crate::tensor::dataset::TensorCase;
    use crate::tensor::{OperatorSet, TensorInstruction};
    use scirust_tensor_core::TensorND;

    fn report(loss: f64, flops: u64, dead: usize, fingerprint: u128) -> FitnessReport {
        FitnessReport {
            loss,
            failed_cases: 0,
            cost: CostReport {
                active_instructions: 1,
                estimated_flops: flops,
                total_active_elements: 1,
                peak_live_elements: 1,
                generated_intermediate_bytes: 0,
                dead_instructions: dead,
                bloat_ratio: dead as f64,
            },
            evaluated: true,
            fingerprint,
        }
    }

    fn dataset() -> Dataset {
        Dataset::new(vec![
            TensorCase::new(
                vec![TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
                TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ),
            TensorCase::new(
                vec![TensorND::new(vec![-1.0, 0.0, 2.0, -3.0], vec![2, 2])],
                TensorND::new(vec![0.0, 0.0, 2.0, 0.0], vec![2, 2]),
            ),
        ])
        .unwrap()
    }

    fn config() -> EvolutionConfig {
        EvolutionConfig {
            generation: GenerationConfig {
                input_shapes: vec![vec![2, 2]],
                min_instructions: 2,
                max_instructions: 6,
                operators: OperatorSet::all(),
                scale_magnitude: 2.0,
            },
            population_size: 12,
            generations: 6,
            elitism: 2,
            tournament: TournamentConfig { size: 3 },
            scale_magnitude: 2.0,
            crossover_probability: 0.7,
            mutation_probability: 0.5,
        }
    }

    #[test]
    fn dominance_is_strict() {
        let better = report(1.0, 10, 0, 1);
        let worse = report(1.0, 20, 0, 2);
        assert!(dominates(&better, &worse));
        assert!(!dominates(&worse, &better));
        // Equal objectives (fingerprint is not an objective) do not dominate.
        assert!(!dominates(&better, &report(1.0, 10, 0, 99)));
    }

    /// A distinct program whose positive `Scale` factor fixes its canonical-byte
    /// order (for positive `f32`, the bit pattern is monotonic in value).
    fn scaled(factor: f32) -> TensorProgram {
        TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale { src: 0, factor },
            ],
            1,
        )
    }

    #[test]
    fn ranking_breaks_ties_under_equal_primary_objective() {
        // Equal loss; the lower-FLOP report must rank first. Programs identical,
        // so the objective order decides.
        let programs = vec![scaled(1.0), scaled(1.0), scaled(1.0)];
        let reports = vec![
            report(1.0, 30, 0, 7),
            report(1.0, 10, 0, 8),
            report(1.0, 20, 0, 9),
        ];
        assert_eq!(rank(&programs, &reports), vec![1, 2, 0]);
    }

    /// The order of `programs` sorted by their canonical bytes.
    fn canonical_order(programs: &[TensorProgram]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..programs.len()).collect();
        order.sort_by(|&a, &b| canonical_bytes(&programs[a]).cmp(&canonical_bytes(&programs[b])));
        order
    }

    #[test]
    fn identical_objectives_rank_by_canonical_bytes_independent_of_insertion_order() {
        // All objectives equal; distinct programs are ordered by their canonical
        // bytes (an authoritative structural key), independently of insertion
        // order. The exact order is whatever the canonical bytes dictate.
        let programs = vec![scaled(3.0), scaled(1.0), scaled(2.0)];
        let reports = vec![
            report(2.0, 5, 1, 300),
            report(2.0, 5, 1, 100),
            report(2.0, 5, 1, 200),
        ];
        assert_eq!(rank(&programs, &reports), canonical_order(&programs));

        // Reordering the inputs yields the same programs in the same structural
        // order (the globally sorted canonical-byte sequence).
        let reordered = vec![scaled(2.0), scaled(3.0), scaled(1.0)];
        let reordered_reports = vec![
            report(2.0, 5, 1, 200),
            report(2.0, 5, 1, 300),
            report(2.0, 5, 1, 100),
        ];
        let order = rank(&reordered, &reordered_reports);
        let ranked_bytes: Vec<Vec<u8>> = order
            .iter()
            .map(|&i| canonical_bytes(&reordered[i]))
            .collect();
        let mut sorted_bytes: Vec<Vec<u8>> = reordered.iter().map(canonical_bytes).collect();
        sorted_bytes.sort();
        assert_eq!(ranked_bytes, sorted_bytes);
    }

    #[test]
    fn adversarial_equal_fingerprints_do_not_collapse_ranking() {
        // Distinct programs whose reports carry the SAME fingerprint field and
        // identical objectives. Ranking must still distinguish all of them by
        // canonical bytes (not the collidable fingerprint), independently of
        // input order.
        let collide = 0xDEAD_BEEF_u128;
        let programs = vec![scaled(3.0), scaled(1.0), scaled(2.0)];
        let reports = vec![
            report(2.0, 5, 1, collide),
            report(2.0, 5, 1, collide),
            report(2.0, 5, 1, collide),
        ];
        let order = rank(&programs, &reports);
        // All three distinct programs are present (none collapsed) and ordered
        // exactly by canonical bytes.
        assert_eq!(order.len(), 3);
        assert_eq!(order, canonical_order(&programs));

        // Reordering the inputs preserves the structural order.
        let reordered = vec![scaled(2.0), scaled(3.0), scaled(1.0)];
        let reordered_reports = vec![
            report(2.0, 5, 1, collide),
            report(2.0, 5, 1, collide),
            report(2.0, 5, 1, collide),
        ];
        let order = rank(&reordered, &reordered_reports);
        let ranked_bytes: Vec<Vec<u8>> = order
            .iter()
            .map(|&i| canonical_bytes(&reordered[i]))
            .collect();
        let mut sorted_bytes: Vec<Vec<u8>> = reordered.iter().map(canonical_bytes).collect();
        sorted_bytes.sort();
        assert_eq!(ranked_bytes, sorted_bytes);
    }

    #[test]
    fn structurally_identical_programs_fall_back_to_index() {
        // Same program repeated with equal objectives: equal canonical bytes, so
        // the population index breaks the tie.
        let programs = vec![scaled(1.0), scaled(1.0), scaled(1.0)];
        let reports = vec![
            report(2.0, 5, 1, 1),
            report(2.0, 5, 1, 2),
            report(2.0, 5, 1, 3),
        ];
        assert_eq!(rank(&programs, &reports), vec![0, 1, 2]);
    }

    #[test]
    fn ranking_is_repeatable() {
        let programs = vec![scaled(1.0), scaled(3.0), scaled(2.0), scaled(4.0)];
        let reports = vec![
            report(1.0, 5, 0, 11),
            report(1.0, 5, 0, 33),
            report(1.0, 5, 0, 22),
            report(0.5, 9, 2, 44),
        ];
        assert_eq!(rank(&programs, &reports), rank(&programs, &reports));

        // Index 3 has the lowest loss and ranks first; the remaining three, tied
        // on objectives, follow in canonical-byte order.
        let order = rank(&programs, &reports);
        assert_eq!(order[0], 3);
        let mut rest = [0usize, 1, 2];
        rest.sort_by(|&a, &b| canonical_bytes(&programs[a]).cmp(&canonical_bytes(&programs[b])));
        assert_eq!(&order[1..], &rest[..]);
    }

    #[test]
    fn tournament_is_reproducible() {
        let positions = vec![3, 0, 4, 1, 2];
        let cfg = TournamentConfig { size: 3 };

        let mut first = DeterministicRng::new(17);
        let mut second = DeterministicRng::new(17);
        let a: Vec<usize> = (0..20)
            .map(|_| tournament(&positions, cfg, &mut first))
            .collect();
        let b: Vec<usize> = (0..20)
            .map(|_| tournament(&positions, cfg, &mut second))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn advance_preserves_population_size() {
        let mut rng = DeterministicRng::new(1);
        let config = config();
        let population = Population::generate(
            &config.generation,
            config.population_size,
            VerificationLimits::default(),
            &mut rng,
        )
        .unwrap();

        let reports = population.evaluate(&dataset(), VerificationLimits::default());
        let next = population.advance(&reports, &config, VerificationLimits::default(), &mut rng);
        assert_eq!(next.len(), config.population_size);
    }

    #[test]
    fn empty_and_single_populations_do_not_panic() {
        // Empty population: evaluation and ranking are empty; advance stays empty.
        let empty = Population::from_programs(Vec::new());
        let reports = empty.evaluate(&dataset(), VerificationLimits::default());
        assert!(reports.is_empty());
        assert!(rank(empty.programs(), &reports).is_empty());
        let mut rng = DeterministicRng::new(0);
        let advanced = empty.advance(&reports, &config(), VerificationLimits::default(), &mut rng);
        assert_eq!(advanced.len(), 0);

        // A tournament over an empty population must not panic.
        let _ = tournament(&[], TournamentConfig { size: 4 }, &mut rng);

        // Single-individual evolution runs to completion.
        let mut single = config();
        single.population_size = 1;
        single.elitism = 1;
        let outcome = evolve(&single, &dataset(), VerificationLimits::default(), 5).unwrap();
        assert_eq!(outcome.generations_run, single.generations);
    }

    #[test]
    fn tournament_size_larger_than_population_is_bounded() {
        // An enormous tournament size must terminate and return a valid index.
        let positions = vec![2, 0, 1];
        let mut rng = DeterministicRng::new(3);
        let selected = tournament(&positions, TournamentConfig { size: usize::MAX }, &mut rng);
        assert!(selected < positions.len());
    }

    #[test]
    fn duplicate_individuals_are_ranked_deterministically() {
        // Three identical programs produce identical reports (same fingerprint);
        // ranking falls back to index and is reproducible.
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let population = Population::from_programs(vec![program.clone(), program.clone(), program]);
        let reports = population.evaluate(&dataset(), VerificationLimits::default());
        assert_eq!(reports[0], reports[1]);
        assert_eq!(rank(population.programs(), &reports), vec![0, 1, 2]);
    }

    #[test]
    fn elitism_under_fully_equal_objectives_is_deterministic() {
        // A population of identical individuals has fully equal objective vectors;
        // advancing must still be deterministic and preserve the size.
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let population = Population::from_programs(vec![program; 8]);
        let reports = population.evaluate(&dataset(), VerificationLimits::default());

        let cfg = config();
        let mut first = DeterministicRng::new(1);
        let mut second = DeterministicRng::new(1);
        let a = population.advance(&reports, &cfg, VerificationLimits::default(), &mut first);
        let b = population.advance(&reports, &cfg, VerificationLimits::default(), &mut second);
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn zero_generation_budget_runs_only_the_initial_population() {
        let mut cfg = config();
        cfg.generations = 0;
        let outcome = evolve(&cfg, &dataset(), VerificationLimits::default(), 4).unwrap();
        assert_eq!(outcome.generations_run, 0);
        assert_eq!(outcome.history.len(), 1);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn ranking_matches_between_sequential_and_rayon_evaluation() {
        use crate::tensor::evaluate_population_rayon;

        let mut rng = DeterministicRng::new(123);
        let cfg = config();
        let population = Population::generate(
            &cfg.generation,
            cfg.population_size,
            VerificationLimits::default(),
            &mut rng,
        )
        .unwrap();
        let dataset = dataset();

        let sequential = population.evaluate(&dataset, VerificationLimits::default());
        let parallel = evaluate_population_rayon(
            population.programs(),
            &dataset,
            VerificationLimits::default(),
        );
        assert_eq!(sequential, parallel);
        assert_eq!(
            rank(population.programs(), &sequential),
            rank(population.programs(), &parallel)
        );
    }

    #[test]
    fn evolution_is_reproducible() {
        let config = config();
        let dataset = dataset();
        let first = evolve(&config, &dataset, VerificationLimits::default(), 2024).unwrap();
        let second = evolve(&config, &dataset, VerificationLimits::default(), 2024).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn elitism_never_worsens_the_best_loss() {
        let config = config();
        let outcome = evolve(&config, &dataset(), VerificationLimits::default(), 7).unwrap();

        for window in outcome.history.windows(2)
        {
            assert!(
                window[1].best_loss <= window[0].best_loss,
                "best loss increased from {} to {}",
                window[0].best_loss,
                window[1].best_loss
            );
        }
    }

    #[test]
    fn rejects_invalid_configuration() {
        let mut bad = config();
        bad.population_size = 0;
        assert!(matches!(
            evolve(&bad, &dataset(), VerificationLimits::default(), 0),
            Err(EvolutionError::InvalidConfig(_))
        ));

        let mut mismatched = config();
        mismatched.generation.input_shapes = vec![vec![3, 3]];
        assert_eq!(
            evolve(&mismatched, &dataset(), VerificationLimits::default(), 0),
            Err(EvolutionError::DatasetShapeMismatch)
        );
    }

    #[test]
    fn best_program_is_valid_and_executable() {
        let config = config();
        let dataset = dataset();
        let outcome = evolve(&config, &dataset, VerificationLimits::default(), 99).unwrap();

        // The reported best must execute on the dataset without panicking.
        for case in dataset.cases()
        {
            let _ = crate::tensor::execute_program(
                &outcome.best_program,
                &case.inputs,
                VerificationLimits::default(),
            );
        }
        assert!(outcome.best_fitness.evaluated);
    }

    #[test]
    fn public_types_survive_serde() {
        let config = config();
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<EvolutionConfig>(&json).unwrap(),
            config
        );

        let outcome = evolve(&config, &dataset(), VerificationLimits::default(), 1).unwrap();
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(
            serde_json::from_str::<EvolutionOutcome>(&json).unwrap(),
            outcome
        );

        let tournament = TournamentConfig { size: 4 };
        let json = serde_json::to_string(&tournament).unwrap();
        assert_eq!(
            serde_json::from_str::<TournamentConfig>(&json).unwrap(),
            tournament
        );
    }
}
