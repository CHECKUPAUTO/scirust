//! scirust-evo — Evolutionary Algorithms
//!
//! All optimizers are **deterministic / reproducible**: each holds a seeded
//! `StdRng` (interior-mutable via `RefCell`) instead of `thread_rng()`, so a
//! given seed reproduces the exact same trajectory across runs and threads —
//! consistent with SciRust's bit-exact reproducibility discipline.

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::cell::RefCell;

/// Default seed used by `Default`/`new` constructors. Override by constructing
/// with an explicit seed (`*_seeded`) for independent reproducible streams.
const EVO_DEFAULT_SEED: u64 = 0x5C12_3E70;

// ============================================================
// GA
// ============================================================

#[derive(Debug, Clone)]
pub struct Individual {
    pub genome: Vec<f64>,
    pub fitness: f64,
}

impl Individual {
    pub fn new(genome: Vec<f64>) -> Self {
        Self {
            genome,
            fitness: f64::NEG_INFINITY,
        }
    }
}

/// Ordering for a **descending** fitness sort that never panics. A NaN fitness
/// (e.g. from a fitness function that overflowed or divided 0/0) is treated as
/// the *least* fit — sorted to the bottom, matching the `NEG_INFINITY` default —
/// so a broken individual cannot dominate elitism/recombination. Replaces the
/// `partial_cmp().unwrap()` that panicked on NaN across the optimizers.
#[inline]
fn cmp_fitness_desc(a: f64, b: f64) -> std::cmp::Ordering {
    let a = if a.is_nan() { f64::NEG_INFINITY } else { a };
    let b = if b.is_nan() { f64::NEG_INFINITY } else { b };
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

pub struct GeneticAlgorithm {
    pub pop_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism: usize,
    pub bounds: (f64, f64),
    rng: RefCell<StdRng>,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self::seeded(EVO_DEFAULT_SEED)
    }
}

impl GeneticAlgorithm {
    /// Construct with an explicit seed for a reproducible random stream.
    pub fn seeded(seed: u64) -> Self {
        Self {
            pop_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism: 2,
            bounds: (-5.0, 5.0),
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn evolve<F>(&self, population: &mut Vec<Individual>, fitness_fn: F)
    where
        F: Fn(&[Individual]) -> Vec<f64>,
    {
        // Degenerate input: with an empty population there is nothing to select
        // parents from, so evolution is a no-op rather than a panic.
        if population.is_empty()
        {
            return;
        }
        let fitnesses = fitness_fn(population);
        for (i, fit) in fitnesses.iter().enumerate()
        {
            population[i].fitness = *fit;
        }
        // Descending by fitness, NaN-safe (see `cmp_fitness_desc`).
        population.sort_by(|a, b| cmp_fitness_desc(a.fitness, b.fitness));
        // Carry over the elite, clamping to the population size so a small
        // population (e.g. smaller than `elitism`) cannot slice out of bounds.
        let elite = self.elitism.min(population.len());
        let mut new_pop = population[..elite].to_vec();
        // Parents are drawn from the top half, but never from an empty range:
        // with 0 or 1 survivors the pool is the single best individual.
        let pool = (population.len() / 2).max(1);
        let mut rng = self.rng.borrow_mut();
        while new_pop.len() < self.pop_size
        {
            let p1 = &population[rng.gen_range(0..pool)];
            let p2 = &population[rng.gen_range(0..pool)];
            let mut child = if rng.gen::<f64>() < self.crossover_rate && !p1.genome.is_empty()
            {
                let point = rng.gen_range(0..p1.genome.len());
                let mut g = p1.genome[..point].to_vec();
                g.extend_from_slice(&p2.genome[point..]);
                Individual::new(g)
            }
            else
            {
                p1.clone()
            };
            if rng.gen::<f64>() < self.mutation_rate
            {
                let normal = Normal::new(0.0, 0.5).unwrap();
                for gene in &mut child.genome
                {
                    *gene += normal.sample(&mut *rng);
                    *gene = gene.clamp(self.bounds.0, self.bounds.1);
                }
            }
            new_pop.push(child);
        }
        population.clear();
        population.extend(new_pop);
    }
    pub fn init_pop(&self, dims: usize) -> Vec<Individual> {
        let mut rng = self.rng.borrow_mut();
        let u = Uniform::new_inclusive(self.bounds.0, self.bounds.1);
        (0..self.pop_size)
            .map(|_| Individual::new((0..dims).map(|_| u.sample(&mut *rng)).collect()))
            .collect()
    }
}

// ============================================================
// CMA-ES (simplifié)
// ============================================================

pub struct CmaEs {
    pub dims: usize,
    pub lambda: usize,
    pub mu: usize,
    pub sigma: f64,
    pub bounds: (f64, f64),
    rng: RefCell<StdRng>,
}

impl CmaEs {
    pub fn new(dims: usize) -> Self {
        Self::seeded(dims, EVO_DEFAULT_SEED.wrapping_add(1))
    }
    /// Construct with an explicit seed for a reproducible random stream.
    pub fn seeded(dims: usize, seed: u64) -> Self {
        let lambda = 4 + (3.0 * (dims as f64).ln()).floor() as usize;
        Self {
            dims,
            lambda,
            mu: lambda / 2,
            sigma: 0.5,
            bounds: (-5.0, 5.0),
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
    pub fn step<F>(&mut self, theta: &mut [f64], fitness_fn: F) -> Vec<Individual>
    where
        F: Fn(&[f64]) -> f64,
    {
        let (dims, sigma, bounds, lambda, mu) =
            (self.dims, self.sigma, self.bounds, self.lambda, self.mu);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = self.rng.borrow_mut();
        let mut offspring: Vec<Individual> = (0..lambda)
            .map(|_| {
                let mut g = theta.to_vec();
                for val in g.iter_mut().take(dims)
                {
                    *val += normal.sample(&mut *rng) * sigma;
                    *val = val.clamp(bounds.0, bounds.1);
                }
                Individual::new(g)
            })
            .collect();
        let fitnesses = offspring
            .iter()
            .map(|ind| fitness_fn(&ind.genome))
            .collect::<Vec<_>>();
        for (i, fit) in fitnesses.iter().enumerate()
        {
            offspring[i].fitness = *fit;
        }
        offspring.sort_by(|a, b| cmp_fitness_desc(a.fitness, b.fitness));
        let w: Vec<f64> = (0..mu).map(|i| (mu as f64 - i as f64).max(0.0)).collect();
        let sw: f64 = w.iter().sum();
        for (i, val) in theta.iter_mut().enumerate().take(dims)
        {
            *val = (0..mu).map(|j| offspring[j].genome[i] * w[j] / sw).sum();
        }
        offspring
    }
}

// ============================================================
// OpenES
// ============================================================

pub struct OpenEs {
    pub dims: usize,
    pub pop_size: usize,
    pub sigma: f64,
    pub alpha: f64,
    pub bounds: (f64, f64),
    rng: RefCell<StdRng>,
}

impl OpenEs {
    pub fn new(dims: usize) -> Self {
        Self::seeded(dims, EVO_DEFAULT_SEED.wrapping_add(2))
    }
    /// Construct with an explicit seed for a reproducible random stream.
    pub fn seeded(dims: usize, seed: u64) -> Self {
        Self {
            dims,
            pop_size: 4 + (3 * dims / 2),
            sigma: 0.1,
            alpha: 0.05,
            bounds: (-5.0, 5.0),
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
    pub fn step<F>(&self, theta: &mut [f64], fitness_fn: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = self.rng.borrow_mut();
        let mut noise = Vec::with_capacity(self.pop_size);
        let mut rewards = Vec::with_capacity(self.pop_size);
        for _ in 0..self.pop_size
        {
            let eps: Vec<f64> = (0..self.dims).map(|_| normal.sample(&mut *rng)).collect();
            let perturbed: Vec<f64> = theta
                .iter()
                .zip(&eps)
                .map(|(t, e)| (t + self.sigma * e).clamp(self.bounds.0, self.bounds.1))
                .collect();
            let r = fitness_fn(&perturbed);
            noise.push(eps);
            rewards.push(r);
        }
        let mr = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let sr = (rewards.iter().map(|r| (r - mr).powi(2)).sum::<f64>() / rewards.len() as f64)
            .sqrt()
            .max(1e-8);
        let mut grad = vec![0.0; self.dims];
        for i in 0..self.pop_size
        {
            let adv = (rewards[i] - mr) / sr;
            for j in 0..self.dims
            {
                grad[j] += adv * noise[i][j];
            }
        }
        let f = self.alpha / (self.pop_size as f64 * self.sigma);
        for j in 0..self.dims
        {
            theta[j] += f * grad[j];
            theta[j] = theta[j].clamp(self.bounds.0, self.bounds.1);
        }
        // Max reward, NaN-safe: `unwrap_or(Equal)` avoids a panic if a reward is
        // NaN; `unwrap_or(0.0)` covers an (unexpected) empty reward set.
        rewards
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
}

// ============================================================
// NSGA-II
// ============================================================

#[derive(Debug, Clone)]
pub struct MoIndividual {
    pub genome: Vec<f64>,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl MoIndividual {
    pub fn new(genome: Vec<f64>) -> Self {
        Self {
            genome,
            objectives: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
        }
    }
}

pub struct Nsga2 {
    pub pop_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub bounds: (f64, f64),
    rng: RefCell<StdRng>,
}

impl Default for Nsga2 {
    fn default() -> Self {
        Self::seeded(EVO_DEFAULT_SEED.wrapping_add(3))
    }
}

fn dominates(a: &MoIndividual, b: &MoIndividual) -> bool {
    let mut at_least_one = false;
    for (ai, bi) in a.objectives.iter().zip(&b.objectives)
    {
        if ai > bi
        {
            return false;
        }
        if ai < bi
        {
            at_least_one = true;
        }
    }
    at_least_one
}

impl Nsga2 {
    /// Construct with an explicit seed for a reproducible random stream.
    pub fn seeded(seed: u64) -> Self {
        Self {
            pop_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            bounds: (-5.0, 5.0),
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn evolve<F>(&self, population: &mut Vec<MoIndividual>, objectives_fn: F)
    where
        F: Fn(&[MoIndividual]) -> Vec<Vec<f64>>,
    {
        let n = self.pop_size;
        // 1. Evaluate + rank the current parents so crowded-tournament selection
        //    sees valid ranks and (per-front) crowding distances.
        let objs = objectives_fn(population);
        for (i, obj) in objs.iter().enumerate()
        {
            population[i].objectives = obj.clone();
        }
        self.non_dominated_sort(population);
        self.assign_crowding_per_front(population);

        // 2. Produce an offspring population Q of size `n` from the parents P
        //    via crowded tournament selection + single-point crossover + mutation.
        let mut offspring = Vec::with_capacity(n);
        {
            let mut rng = self.rng.borrow_mut();
            while offspring.len() < n
            {
                let p1 = Self::tournament_select(population, &mut rng);
                let p2 = Self::tournament_select(population, &mut rng);
                let (mut c1, mut c2) = if rng.gen::<f64>() < self.crossover_rate
                {
                    let point = rng.gen_range(0..p1.genome.len());
                    let mut g1 = p1.genome[..point].to_vec();
                    g1.extend_from_slice(&p2.genome[point..]);
                    let mut g2 = p2.genome[..point].to_vec();
                    g2.extend_from_slice(&p1.genome[point..]);
                    (MoIndividual::new(g1), MoIndividual::new(g2))
                }
                else
                {
                    (p1.clone(), p2.clone())
                };
                Self::mutate(&mut c1, &mut rng, self.mutation_rate, self.bounds);
                Self::mutate(&mut c2, &mut rng, self.mutation_rate, self.bounds);
                offspring.push(c1);
                if offspring.len() < n
                {
                    offspring.push(c2);
                }
            }
        }
        // 3. Evaluate the offspring objectives.
        let objs = objectives_fn(&offspring);
        for (i, obj) in objs.iter().enumerate()
        {
            offspring[i].objectives = obj.clone();
        }

        // 4. Elitist (μ+λ) survival: combine R = P ∪ Q (size 2n) and rank the
        //    union. This is what makes NSGA-II elitist — the best non-dominated
        //    solutions found so far are never lost between generations.
        let mut combined = std::mem::take(population);
        combined.append(&mut offspring);
        self.non_dominated_sort(&mut combined);

        // 5. Fill the next generation one whole front at a time. The front that
        //    does not fit entirely is truncated by *descending* crowding
        //    distance (computed within that front) to preserve diversity.
        let max_rank = combined.iter().map(|ind| ind.rank).max().unwrap_or(0);
        let mut survivors: Vec<MoIndividual> = Vec::with_capacity(n);
        let mut rank = 1;
        while rank <= max_rank && survivors.len() < n
        {
            let mut front: Vec<MoIndividual> = combined
                .iter()
                .filter(|ind| ind.rank == rank)
                .cloned()
                .collect();
            rank += 1;
            if front.is_empty()
            {
                continue;
            }
            self.assign_crowding_per_front(&mut front);
            if survivors.len() + front.len() <= n
            {
                survivors.append(&mut front);
            }
            else
            {
                front.sort_by(|a, b| {
                    b.crowding_distance
                        .partial_cmp(&a.crowding_distance)
                        .unwrap()
                });
                let need = n - survivors.len();
                survivors.extend(front.into_iter().take(need));
            }
        }
        // `survivors` carry ranks/distances from the combined pool, so callers
        // can read the current Pareto front directly via `rank == 1`.
        *population = survivors;
    }
    fn non_dominated_sort(&self, pop: &mut [MoIndividual]) {
        let n = pop.len();
        let mut dc = vec![0; n];
        let mut dom = vec![Vec::new(); n];
        for i in 0..n
        {
            for j in 0..n
            {
                if i == j
                {
                    continue;
                }
                if dominates(&pop[i], &pop[j])
                {
                    dom[i].push(j);
                }
                else if dominates(&pop[j], &pop[i])
                {
                    dc[i] += 1;
                }
            }
        }
        let mut cf: Vec<usize> = (0..n).filter(|i| dc[*i] == 0).collect();
        for &i in &cf
        {
            pop[i].rank = 1;
        }
        let mut rank = 1;
        while !cf.is_empty()
        {
            let mut nf = Vec::new();
            for &i in &cf
            {
                for &j in &dom[i]
                {
                    dc[j] -= 1;
                    if dc[j] == 0
                    {
                        pop[j].rank = rank + 1;
                        nf.push(j);
                    }
                }
            }
            rank += 1;
            cf = nf;
        }
    }
    /// Assign NSGA-II crowding distance to every member of a single Pareto
    /// front. `pop` must be non-empty and contain individuals of one rank whose
    /// `objectives` all have the same length; boundary points in each objective
    /// receive infinite distance, interior points the sum of normalized
    /// neighbour gaps.
    fn assign_crowding_per_front(&self, pop: &mut [MoIndividual]) {
        let no = pop[0].objectives.len();
        let n = pop.len();
        for ind in pop.iter_mut()
        {
            ind.crowding_distance = 0.0;
        }
        for m in 0..no
        {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|a, b| {
                pop[*a].objectives[m]
                    .partial_cmp(&pop[*b].objectives[m])
                    .unwrap()
            });
            pop[idx[0]].crowding_distance = f64::INFINITY;
            pop[idx[n - 1]].crowding_distance = f64::INFINITY;
            let fmin = pop[idx[0]].objectives[m];
            let fmax = pop[idx[n - 1]].objectives[m];
            let range = (fmax - fmin).max(1e-10);
            for i in 1..(n - 1)
            {
                pop[idx[i]].crowding_distance +=
                    (pop[idx[i + 1]].objectives[m] - pop[idx[i - 1]].objectives[m]) / range;
            }
        }
    }
    fn tournament_select(pop: &[MoIndividual], rng: &mut StdRng) -> MoIndividual {
        let i1 = rng.gen_range(0..pop.len());
        let i2 = rng.gen_range(0..pop.len());
        if pop[i1].rank < pop[i2].rank
        {
            pop[i1].clone()
        }
        else if pop[i1].rank > pop[i2].rank
        {
            pop[i2].clone()
        }
        else if pop[i1].crowding_distance > pop[i2].crowding_distance
        {
            pop[i1].clone()
        }
        else
        {
            pop[i2].clone()
        }
    }
    fn mutate(ind: &mut MoIndividual, rng: &mut StdRng, mutation_rate: f64, bounds: (f64, f64)) {
        let normal = Normal::new(0.0, 0.5).unwrap();
        for gene in &mut ind.genome
        {
            if rng.gen::<f64>() < mutation_rate
            {
                *gene += normal.sample(rng);
                *gene = gene.clamp(bounds.0, bounds.1);
            }
        }
    }
    pub fn init_pop(&self, dims: usize) -> Vec<MoIndividual> {
        let mut rng = self.rng.borrow_mut();
        let u = Uniform::new_inclusive(self.bounds.0, self.bounds.1);
        (0..self.pop_size)
            .map(|_| MoIndividual::new((0..dims).map(|_| u.sample(&mut *rng)).collect()))
            .collect()
    }
}

// ============================================================
// Benchmarks
// ============================================================

pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s: f64 = x
        .iter()
        .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
        .sum();
    10.0 * n + s
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ZDT1 objectives: f1 = x0, g = 1 + 9*mean(x_{1..}), f2 = g*(1 - sqrt(f1/g)).
    // The true Pareto front is g == 1, i.e. f2 = 1 - sqrt(f1) for f1 in [0, 1].
    fn zdt1(inds: &[MoIndividual]) -> Vec<Vec<f64>> {
        inds.iter()
            .map(|ind| {
                let x = &ind.genome;
                let f1 = x[0].clamp(0.0, 1.0);
                let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() as f64 - 1.0);
                let h = 1.0 - (f1 / g).sqrt();
                vec![f1, g * h]
            })
            .collect()
    }

    // Re-derived dominance oracle (minimization), independent of the crate's
    // private `dominates`, used to certify the rank-1 set is a true Pareto set.
    fn oracle_dominates(a: &[f64], b: &[f64]) -> bool {
        let mut strictly_better = false;
        for (ai, bi) in a.iter().zip(b)
        {
            if ai > bi
            {
                return false;
            }
            if ai < bi
            {
                strictly_better = true;
            }
        }
        strictly_better
    }

    // ----- benchmark functions: exact closed-form oracles -----

    #[test]
    fn sphere_known_values() {
        assert_eq!(sphere(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(sphere(&[1.0, 2.0, 2.0]), 9.0);
        assert_eq!(sphere(&[-3.0]), 9.0);
        // Sum of squares is strictly positive away from the origin.
        assert!(sphere(&[0.5, -0.5]) > 0.0);
    }

    #[test]
    fn rastrigin_known_values() {
        // Global optimum at the origin is exactly 0.
        assert_eq!(rastrigin(&[0.0; 5]), 0.0);
        // At integer coordinates cos(2*pi*k) == 1, so rastrigin == sum of squares.
        assert!((rastrigin(&[1.0]) - 1.0).abs() < 1e-9);
        assert!((rastrigin(&[2.0, -3.0]) - 13.0).abs() < 1e-9);
        // Hand value: 0.25 - 10*cos(pi) + 10 = 20.25.
        assert!((rastrigin(&[0.5]) - 20.25).abs() < 1e-9);
        // Away from the optimum the value is strictly positive.
        assert!(rastrigin(&[0.3, -0.7]) > 0.0);
    }

    // ----- Individual / MoIndividual constructors -----

    #[test]
    fn individual_new_initializes_fields() {
        let ind = Individual::new(vec![1.0, -2.0, 3.0]);
        assert_eq!(ind.genome, vec![1.0, -2.0, 3.0]);
        // Unevaluated individuals must start at -inf so any real fitness wins.
        assert_eq!(ind.fitness, f64::NEG_INFINITY);
    }

    #[test]
    fn mo_individual_new_initializes_fields() {
        let ind = MoIndividual::new(vec![0.5, 0.5]);
        assert_eq!(ind.genome, vec![0.5, 0.5]);
        assert!(ind.objectives.is_empty());
        assert_eq!(ind.rank, 0);
        assert_eq!(ind.crowding_distance, 0.0);
    }

    // ----- GA: structure of init_pop -----

    #[test]
    fn ga_init_pop_shape_and_bounds() {
        let ga = GeneticAlgorithm::seeded(1);
        let pop = ga.init_pop(7);
        assert_eq!(pop.len(), ga.pop_size);
        for ind in &pop
        {
            assert_eq!(ind.genome.len(), 7);
            for &g in &ind.genome
            {
                assert!(
                    g >= ga.bounds.0 && g <= ga.bounds.1,
                    "init gene {g} out of bounds {:?}",
                    ga.bounds
                );
            }
            assert_eq!(ind.fitness, f64::NEG_INFINITY);
        }
    }

    // ----- GA: elitism preserves the best, best fitness is monotone -----

    #[test]
    fn ga_elitism_preserves_best_and_is_monotone() {
        // Maximize sum of genome. The known optimum is dims * upper_bound.
        let dims = 8usize;
        let mut ga = GeneticAlgorithm::seeded(2024);
        ga.pop_size = 60;
        ga.elitism = 2;
        let mut pop = ga.init_pop(dims);
        let fit = |inds: &[Individual]| -> Vec<f64> {
            inds.iter().map(|i| i.genome.iter().sum::<f64>()).collect()
        };

        let mut prev_best = f64::NEG_INFINITY;
        for _ in 0..60
        {
            // Determine the incumbent best parent *before* this generation, by
            // evaluating the parents exactly as `evolve` does internally.
            let scores = fit(&pop);
            let (bi, &incumbent_fit) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let incumbent_genome = pop[bi].genome.clone();

            ga.evolve(&mut pop, fit);

            // Elitism guarantee (within this generation): the best parent is
            // copied verbatim into the offspring, so it must still be present.
            assert!(
                pop.iter().any(|i| i.genome == incumbent_genome),
                "elitism dropped the best parent (genome {incumbent_genome:?})"
            );
            // The best fitness of the new generation is at least the incumbent's
            // and never decreases across generations.
            let out_best = pop
                .iter()
                .filter(|i| i.fitness.is_finite())
                .map(|i| i.fitness)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                out_best >= incumbent_fit - 1e-12,
                "best fitness dropped below incumbent {incumbent_fit} -> {out_best}"
            );
            assert!(
                out_best >= prev_best - 1e-12,
                "best fitness regressed across generations {prev_best} -> {out_best}"
            );
            prev_best = out_best;
        }
        // Converges to the known optimum (8 * 5.0 == 40.0).
        let optimum = dims as f64 * ga.bounds.1;
        assert!(
            prev_best > optimum - 1e-6,
            "GA did not reach the known optimum {optimum}, got {prev_best}"
        );
    }

    #[test]
    fn ga_minimizes_sphere_near_zero() {
        let mut ga = GeneticAlgorithm::seeded(11);
        ga.pop_size = 50;
        let mut pop = ga.init_pop(6);
        let start = pop
            .iter()
            .map(|i| sphere(&i.genome))
            .fold(f64::INFINITY, f64::min);
        for _ in 0..120
        {
            ga.evolve(&mut pop, |inds| {
                inds.iter().map(|i| -sphere(&i.genome)).collect()
            });
        }
        let best = pop
            .iter()
            .map(|i| sphere(&i.genome))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best < start,
            "GA did not improve over the initial population"
        );
        assert!(best < 0.1, "GA min sphere = {best}, expected < 0.1");
    }

    #[test]
    fn ga_crossover_of_identical_parents_yields_that_parent() {
        // With an all-identical population and no mutation, crossover of two
        // identical parents reproduces the parent exactly, so the whole
        // generation stays equal to the seed genome.
        let mut ga = GeneticAlgorithm::seeded(3);
        ga.pop_size = 16;
        ga.mutation_rate = 0.0;
        ga.elitism = 1;
        ga.crossover_rate = 1.0;
        let g = vec![1.5, -2.5, 0.75, 4.0];
        let mut pop: Vec<Individual> = (0..ga.pop_size)
            .map(|_| Individual::new(g.clone()))
            .collect();
        ga.evolve(&mut pop, |inds| {
            inds.iter().map(|i| i.genome.iter().sum::<f64>()).collect()
        });
        assert_eq!(pop.len(), ga.pop_size);
        for ind in &pop
        {
            assert_eq!(ind.genome, g, "identical-parent crossover changed a genome");
        }
    }

    #[test]
    fn ga_mutation_respects_bounds() {
        // Force mutation on every individual every generation and assert genes
        // stay clamped inside the configured bounds.
        let mut ga = GeneticAlgorithm::seeded(4);
        ga.pop_size = 40;
        ga.mutation_rate = 1.0;
        ga.bounds = (-1.0, 1.0);
        let mut pop = ga.init_pop(5);
        for _ in 0..30
        {
            ga.evolve(&mut pop, |inds| {
                inds.iter().map(|i| -sphere(&i.genome)).collect()
            });
            for ind in &pop
            {
                for &gene in &ind.genome
                {
                    assert!(
                        (-1.0..=1.0).contains(&gene),
                        "gene {gene} escaped bounds after mutation"
                    );
                }
            }
        }
    }

    #[test]
    fn ga_seeded_is_reproducible() {
        let run = || {
            let mut ga = GeneticAlgorithm::seeded(777);
            ga.pop_size = 30;
            let mut pop = ga.init_pop(5);
            for _ in 0..25
            {
                ga.evolve(&mut pop, |inds| {
                    inds.iter().map(|i| -sphere(&i.genome)).collect()
                });
            }
            pop.iter()
                .flat_map(|i| i.genome.iter().map(|v| v.to_bits()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(run(), run(), "seeded GA must be bit-reproducible");
    }

    #[test]
    fn ga_evolve_empty_population_is_noop() {
        // A degenerate empty population must not panic on the elitism slice or
        // the parent-selection range; evolution is simply a no-op.
        let ga = GeneticAlgorithm::seeded(1);
        let mut pop: Vec<Individual> = Vec::new();
        ga.evolve(&mut pop, |inds| {
            inds.iter().map(|i| i.genome.iter().sum::<f64>()).collect()
        });
        assert!(pop.is_empty());
    }

    #[test]
    fn ga_evolve_tiny_population_does_not_panic() {
        // With a single survivor the "top half" pool is empty (len/2 == 0);
        // previously this produced an empty gen_range and panicked. The pool
        // must fall back to the single best individual and fill to pop_size.
        let mut ga = GeneticAlgorithm::seeded(2);
        ga.pop_size = 4;
        ga.elitism = 1;
        ga.mutation_rate = 0.0;
        let mut pop = vec![Individual::new(vec![1.0, 2.0, 3.0])];
        ga.evolve(&mut pop, |inds| {
            inds.iter().map(|i| i.genome.iter().sum::<f64>()).collect()
        });
        assert_eq!(pop.len(), ga.pop_size);
    }

    #[test]
    fn ga_evolve_empty_genomes_does_not_panic() {
        // Zero-length genomes make the crossover point range (0..genome.len())
        // empty; crossover must be skipped instead of panicking.
        let mut ga = GeneticAlgorithm::seeded(3);
        ga.pop_size = 6;
        ga.elitism = 1;
        ga.crossover_rate = 1.0;
        ga.mutation_rate = 1.0;
        let mut pop: Vec<Individual> = (0..ga.pop_size)
            .map(|_| Individual::new(Vec::new()))
            .collect();
        ga.evolve(&mut pop, |inds| vec![0.0; inds.len()]);
        assert_eq!(pop.len(), ga.pop_size);
        for ind in &pop
        {
            assert!(ind.genome.is_empty());
        }
    }

    #[test]
    fn ga_default_matches_explicit_default_seed() {
        let ga = GeneticAlgorithm::default();
        assert_eq!(ga.pop_size, 100);
        assert_eq!(ga.elitism, 2);
        assert_eq!(ga.bounds, (-5.0, 5.0));
        // Default must reuse the documented default seed: identical first draws.
        let a = GeneticAlgorithm::default().init_pop(4);
        let b = GeneticAlgorithm::seeded(EVO_DEFAULT_SEED).init_pop(4);
        let bits = |p: &[Individual]| {
            p.iter()
                .flat_map(|i| i.genome.iter().map(|v| v.to_bits()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(bits(&a), bits(&b));
    }

    // ----- CMA-ES -----

    #[test]
    fn cmaes_lambda_mu_derivation() {
        // lambda = 4 + floor(3 * ln(dims)); mu = lambda / 2.
        for dims in [2usize, 4, 8, 16, 32]
        {
            let cma = CmaEs::seeded(dims, 0);
            let expected_lambda = 4 + (3.0 * (dims as f64).ln()).floor() as usize;
            assert_eq!(cma.lambda, expected_lambda, "lambda for dims={dims}");
            assert_eq!(cma.mu, expected_lambda / 2, "mu for dims={dims}");
        }
    }

    #[test]
    fn cmaes_step_returns_lambda_offspring() {
        let mut cma = CmaEs::seeded(6, 5);
        let lambda = cma.lambda;
        let mut theta = vec![1.0; 6];
        let off = cma.step(&mut theta, |x| -sphere(x));
        assert_eq!(off.len(), lambda);
        for ind in &off
        {
            assert_eq!(ind.genome.len(), 6);
            assert!(ind.fitness.is_finite());
        }
    }

    #[test]
    fn cmaes_recombination_is_weighted_mean_of_best_mu() {
        // Oracle: after a step, theta must equal the rank-weighted mean of the
        // best-mu offspring (weights mu, mu-1, ..., 1). Recompute independently.
        let mut cma = CmaEs::seeded(4, 77);
        let mu = cma.mu;
        let mut theta = vec![2.0; 4];
        let mut off = cma.step(&mut theta, |x| -sphere(x));
        off.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let w: Vec<f64> = (0..mu).map(|i| (mu as f64 - i as f64).max(0.0)).collect();
        let sw: f64 = w.iter().sum();
        for (d, &actual) in theta.iter().enumerate()
        {
            let expected: f64 = (0..mu).map(|j| off[j].genome[d] * w[j] / sw).sum();
            assert!(
                (actual - expected).abs() < 1e-12,
                "dim {d}: theta {actual} != weighted mean {expected}"
            );
        }
    }

    #[test]
    fn cmaes_minimizes_sphere_near_zero() {
        // This simplified ES has a fixed step size (no sigma adaptation), so in
        // 8-D it converges to a small neighbourhood of the optimum rather than
        // machine zero. Require a large relative improvement plus a loose, seed
        // -robust absolute bound (the plateau sits well under 1.0).
        let mut cma = CmaEs::seeded(8, 99);
        let mut theta = vec![3.0; 8];
        let start = sphere(&theta); // = 8 * 9 = 72
        for _ in 0..80
        {
            cma.step(&mut theta, |x| -sphere(x));
        }
        let end = sphere(&theta);
        assert!(
            end < start * 0.05,
            "CMA-ES gave <20x improvement: {start} -> {end}"
        );
        assert!(end < 1.5, "CMA-ES sphere = {end}, expected well under 1.5");
    }

    #[test]
    fn cmaes_seeded_is_reproducible() {
        let run = || {
            let mut cma = CmaEs::seeded(6, 314);
            let mut theta = vec![2.0; 6];
            for _ in 0..30
            {
                cma.step(&mut theta, |x| -sphere(x));
            }
            theta.iter().map(|v| v.to_bits()).collect::<Vec<u64>>()
        };
        assert_eq!(run(), run(), "seeded CMA-ES must be bit-reproducible");
    }

    // ----- OpenES -----

    #[test]
    fn openes_gradient_points_uphill_on_linear_reward() {
        // For a linear reward f(x)=sum(x), the ES natural-gradient estimate is
        // +1 per component, so one step from the origin must increase sum(theta).
        // (Sign oracle: a flipped update would *decrease* it.)
        let oes = OpenEs::seeded(8, 1234);
        let mut theta = vec![0.0; 8];
        oes.step(&mut theta, |x| x.iter().sum::<f64>());
        let moved: f64 = theta.iter().sum();
        assert!(
            moved > 0.0,
            "OpenES moved downhill on a linear reward: {moved}"
        );
    }

    #[test]
    fn openes_step_returns_batch_maximum() {
        // Oracle: capture every reward the optimizer actually evaluates this
        // step and assert the returned value is exactly their maximum.
        let oes = OpenEs::seeded(5, 11);
        let seen = std::cell::RefCell::new(Vec::new());
        let mut theta = vec![1.0; 5];
        let returned = oes.step(&mut theta, |x| {
            let r = -sphere(x);
            seen.borrow_mut().push(r);
            r
        });
        let seen = seen.into_inner();
        assert_eq!(
            seen.len(),
            oes.pop_size,
            "step must evaluate pop_size samples"
        );
        let batch_max = seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            returned.to_bits(),
            batch_max.to_bits(),
            "step must return the batch maximum {batch_max}, returned {returned}"
        );
    }

    #[test]
    fn openes_minimizes_sphere_near_zero() {
        let oes = OpenEs::seeded(8, 99);
        let mut theta = vec![3.0; 8];
        let start = sphere(&theta);
        for _ in 0..250
        {
            oes.step(&mut theta, |x| -sphere(x));
        }
        let end = sphere(&theta);
        assert!(end < start, "OpenES did not improve: {start} -> {end}");
        assert!(end < 0.5, "OpenES sphere = {end}, expected < 0.5");
    }

    #[test]
    fn openes_respects_bounds() {
        let mut oes = OpenEs::seeded(4, 7);
        oes.bounds = (-2.0, 2.0);
        let mut theta = vec![0.0; 4];
        for _ in 0..50
        {
            // Reward that pulls strongly toward +infinity; clamping must hold.
            oes.step(&mut theta, |x| x.iter().sum::<f64>());
            for &t in &theta
            {
                assert!((-2.0..=2.0).contains(&t), "theta {t} escaped bounds");
            }
        }
    }

    // ----- NSGA-II -----

    #[test]
    fn nsga2_init_pop_shape_and_bounds() {
        let mut nsga = Nsga2::seeded(1);
        nsga.bounds = (0.0, 1.0);
        let pop = nsga.init_pop(9);
        assert_eq!(pop.len(), nsga.pop_size);
        for ind in &pop
        {
            assert_eq!(ind.genome.len(), 9);
            for &g in &ind.genome
            {
                assert!((0.0..=1.0).contains(&g), "init gene {g} out of bounds");
            }
        }
    }

    #[test]
    fn nsga2_rank1_set_is_a_true_pareto_set() {
        // After evolution, no population member may dominate any rank-1 member.
        let mut nsga = Nsga2::seeded(5);
        nsga.pop_size = 50;
        nsga.bounds = (0.0, 1.0);
        let mut pop = nsga.init_pop(8);
        for _ in 0..40
        {
            nsga.evolve(&mut pop, zdt1);
        }
        let front: Vec<&MoIndividual> = pop.iter().filter(|i| i.rank == 1).collect();
        assert!(!front.is_empty(), "empty Pareto front");
        for f in &front
        {
            for other in &pop
            {
                assert!(
                    !oracle_dominates(&other.objectives, &f.objectives),
                    "a rank-1 member is dominated: {:?} dominated by {:?}",
                    f.objectives,
                    other.objectives
                );
            }
        }
    }

    #[test]
    fn nsga2_elitism_keeps_per_objective_minimum_monotone() {
        // Elitist (mu+lambda) survival => the best value reached for each
        // objective can never get worse from one generation to the next.
        // This property is what the non-elitist (full-replacement) variant
        // violates.
        let mut nsga = Nsga2::seeded(9);
        nsga.pop_size = 40;
        nsga.bounds = (0.0, 1.0);
        let mut pop = nsga.init_pop(8);
        nsga.evolve(&mut pop, zdt1);
        let mut best0 = pop
            .iter()
            .map(|i| i.objectives[0])
            .fold(f64::INFINITY, f64::min);
        let mut best1 = pop
            .iter()
            .map(|i| i.objectives[1])
            .fold(f64::INFINITY, f64::min);
        for _ in 0..60
        {
            nsga.evolve(&mut pop, zdt1);
            let m0 = pop
                .iter()
                .map(|i| i.objectives[0])
                .fold(f64::INFINITY, f64::min);
            let m1 = pop
                .iter()
                .map(|i| i.objectives[1])
                .fold(f64::INFINITY, f64::min);
            assert!(
                m0 <= best0 + 1e-12,
                "objective 0 minimum regressed {best0} -> {m0}"
            );
            assert!(
                m1 <= best1 + 1e-12,
                "objective 1 minimum regressed {best1} -> {m1}"
            );
            best0 = best0.min(m0);
            best1 = best1.min(m1);
        }
    }

    #[test]
    fn nsga2_converges_to_known_zdt1_front() {
        // The true ZDT1 front is f2 = 1 - sqrt(f1). Measure the mean distance of
        // the rank-1 set to that curve and require convergence to a small value.
        let mut nsga = Nsga2::seeded(5);
        nsga.pop_size = 60;
        nsga.bounds = (0.0, 1.0);
        let mut pop = nsga.init_pop(10);
        let gd = |pop: &[MoIndividual]| -> f64 {
            let front: Vec<&MoIndividual> = pop.iter().filter(|i| i.rank == 1).collect();
            let n = front.len().max(1) as f64;
            front
                .iter()
                .map(|i| (i.objectives[1] - (1.0 - i.objectives[0].sqrt())).abs())
                .sum::<f64>()
                / n
        };
        nsga.evolve(&mut pop, zdt1);
        let start = gd(&pop);
        for _ in 0..80
        {
            nsga.evolve(&mut pop, zdt1);
        }
        let end = gd(&pop);
        assert!(
            end < start,
            "NSGA-II did not improve toward the front: {start} -> {end}"
        );
        assert!(
            end < 1e-2,
            "mean distance to ZDT1 front = {end}, expected < 1e-2"
        );
    }

    #[test]
    fn nsga2_seeded_is_reproducible() {
        let run = || {
            let mut nsga = Nsga2::seeded(123);
            nsga.pop_size = 40;
            nsga.bounds = (0.0, 1.0);
            let mut pop = nsga.init_pop(6);
            for _ in 0..30
            {
                nsga.evolve(&mut pop, zdt1);
            }
            pop.iter()
                .flat_map(|i| i.genome.iter().map(|v| v.to_bits()))
                .collect::<Vec<u64>>()
        };
        assert_eq!(run(), run(), "seeded NSGA-II must be bit-reproducible");
    }

    // Regression: a fitness function that yields NaN must not panic the sort and
    // must not let the broken individual dominate (NaN is treated as least fit).
    #[test]
    fn evolve_tolerates_nan_fitness_without_panicking() {
        let ga = GeneticAlgorithm::seeded(7);
        let mut pop: Vec<Individual> = (0..8).map(|i| Individual::new(vec![i as f64])).collect();
        // Individual 3 gets a NaN fitness; the rest score by their gene value.
        ga.evolve(&mut pop, |p| {
            p.iter()
                .map(|ind| {
                    if ind.genome[0] == 3.0
                    {
                        f64::NAN
                    }
                    else
                    {
                        ind.genome[0]
                    }
                })
                .collect()
        });
        // Must have produced a full new population without panicking, and the
        // best-carried elite must be a finite (non-NaN) individual.
        assert_eq!(pop.len(), ga.pop_size);
        assert!(
            pop[0].fitness.is_finite() || pop[0].fitness == f64::NEG_INFINITY,
            "elite fitness must not be NaN"
        );
    }
}
