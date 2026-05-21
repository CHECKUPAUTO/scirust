//! scirust-evo — Evolutionary Algorithms

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

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
        Self { genome, fitness: f64::NEG_INFINITY }
    }
}

pub struct GeneticAlgorithm {
    pub pop_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism: usize,
    pub bounds: (f64, f64),
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self { pop_size: 100, mutation_rate: 0.1, crossover_rate: 0.8, elitism: 2, bounds: (-5.0, 5.0) }
    }
}

impl GeneticAlgorithm {
    pub fn evolve<F>(&self, population: &mut Vec<Individual>, fitness_fn: F
    ) where F: Fn(&[Individual]) -> Vec<f64> {
        let fitnesses = fitness_fn(population);
        for (i, fit) in fitnesses.iter().enumerate() { population[i].fitness = *fit; }
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let mut new_pop = population[..self.elitism].to_vec();
        let mut rng = thread_rng();
        while new_pop.len() < self.pop_size {
            let p1 = &population[rng.gen_range(0..population.len()/2)];
            let p2 = &population[rng.gen_range(0..population.len()/2)];
            let mut child = if rng.gen::<f64>() < self.crossover_rate {
                let point = rng.gen_range(0..p1.genome.len());
                let mut g = p1.genome[..point].to_vec();
                g.extend_from_slice(&p2.genome[point..]); Individual::new(g)
            } else { p1.clone() };
            if rng.gen::<f64>() < self.mutation_rate {
                let normal = Normal::new(0.0, 0.5).unwrap();
                for gene in &mut child.genome {
                    *gene += normal.sample(&mut rng);
                    *gene = gene.clamp(self.bounds.0, self.bounds.1);
                }
            }
            new_pop.push(child);
        }
        population.clear(); population.extend(new_pop);
    }
    pub fn init_pop(&self, dims: usize) -> Vec<Individual> {
        let mut rng = thread_rng();
        let u = Uniform::new_inclusive(self.bounds.0, self.bounds.1);
        (0..self.pop_size).map(|_| Individual::new((0..dims).map(|_| u.sample(&mut rng)).collect())).collect()
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
}

impl CmaEs {
    pub fn new(dims: usize) -> Self {
        let lambda = 4 + (3.0 * (dims as f64).ln()).floor() as usize;
        Self { dims, lambda, mu: lambda/2, sigma: 0.5, bounds: (-5.0, 5.0) }
    }
    pub fn step<F>(&mut self, theta: &mut Vec<f64>, fitness_fn: F
    ) -> Vec<Individual> where F: Fn(&[f64]) -> f64 {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut offspring: Vec<Individual> = (0..self.lambda).map(|_| {
            let mut g = theta.clone();
            for i in 0..self.dims { g[i] += normal.sample(&mut rng) * self.sigma; g[i] = g[i].clamp(self.bounds.0, self.bounds.1); }
            Individual::new(g)
        }).collect();
        let fitnesses = offspring.iter().map(|ind| fitness_fn(&ind.genome)).collect::<Vec<_>>();
        for (i, fit) in fitnesses.iter().enumerate() { offspring[i].fitness = *fit; }
        offspring.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let w: Vec<f64> = (0..self.mu).map(|i| (self.mu as f64 - i as f64).max(0.0)).collect();
        let sw: f64 = w.iter().sum();
        for i in 0..self.dims { theta[i] = (0..self.mu).map(|j| offspring[j].genome[i] * w[j] / sw).sum(); }
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
}

impl OpenEs {
    pub fn new(dims: usize) -> Self {
        Self { dims, pop_size: 4 + (3 * dims / 2), sigma: 0.1, alpha: 0.05, bounds: (-5.0, 5.0) }
    }
    pub fn step<F>(&self, theta: &mut Vec<f64>, fitness_fn: F
    ) -> f64 where F: Fn(&[f64]) -> f64 {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut noise = Vec::with_capacity(self.pop_size);
        let mut rewards = Vec::with_capacity(self.pop_size);
        for _ in 0..self.pop_size {
            let eps: Vec<f64> = (0..self.dims).map(|_| normal.sample(&mut rng)).collect();
            let perturbed: Vec<f64> = theta.iter().zip(&eps).map(|(t, e)| (t + self.sigma * e).clamp(self.bounds.0, self.bounds.1)).collect();
            let r = fitness_fn(&perturbed);
            noise.push(eps); rewards.push(r);
        }
        let mr = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let sr = (rewards.iter().map(|r| (r - mr).powi(2)).sum::<f64>() / rewards.len() as f64).sqrt().max(1e-8);
        let mut grad = vec![0.0; self.dims];
        for i in 0..self.pop_size {
            let adv = (rewards[i] - mr) / sr;
            for j in 0..self.dims { grad[j] += adv * noise[i][j]; }
        }
        let f = self.alpha / (self.pop_size as f64 * self.sigma);
        for j in 0..self.dims { theta[j] += f * grad[j]; theta[j] = theta[j].clamp(self.bounds.0, self.bounds.1); }
        *rewards.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
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
    pub fn new(genome: Vec<f64>) -> Self { Self { genome, objectives: Vec::new(), rank: 0, crowding_distance: 0.0 } }
}

pub struct Nsga2 {
    pub pop_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub bounds: (f64, f64),
}

impl Default for Nsga2 {
    fn default() -> Self { Self { pop_size: 100, mutation_rate: 0.1, crossover_rate: 0.9, bounds: (-5.0, 5.0) } }
}

fn dominates(a: &MoIndividual, b: &MoIndividual) -> bool {
    let mut at_least_one = false;
    for (ai, bi) in a.objectives.iter().zip(&b.objectives) {
        if ai > bi { return false; }
        if ai < bi { at_least_one = true; }
    }
    at_least_one
}

impl Nsga2 {
    pub fn evolve<F>(&self, population: &mut Vec<MoIndividual>, objectives_fn: F
    ) where F: Fn(&[MoIndividual]) -> Vec<Vec<f64>> {
        let objs = objectives_fn(population);
        for (i, obj) in objs.iter().enumerate() { population[i].objectives = obj.clone(); }
        self.non_dominated_sort(population);
        self.crowding_distance(population);
        let mut new_pop = Vec::with_capacity(self.pop_size);
        let mut rng = thread_rng();
        while new_pop.len() < self.pop_size {
            let p1 = self.tournament_select(population, &mut rng);
            let p2 = self.tournament_select(population, &mut rng);
            let (mut c1, mut c2) = if rng.gen::<f64>() < self.crossover_rate {
                let point = rng.gen_range(0..p1.genome.len());
                let mut g1 = p1.genome[..point].to_vec(); g1.extend_from_slice(&p2.genome[point..]);
                let mut g2 = p2.genome[..point].to_vec(); g2.extend_from_slice(&p1.genome[point..]);
                (MoIndividual::new(g1), MoIndividual::new(g2))
            } else { (p1.clone(), p2.clone()) };
            self.mutate(&mut c1, &mut rng); self.mutate(&mut c2, &mut rng);
            new_pop.push(c1); if new_pop.len() < self.pop_size { new_pop.push(c2); }
        }
        *population = new_pop;
    }
    fn non_dominated_sort(&self, pop: &mut [MoIndividual]) {
        let n = pop.len();
        let mut dc = vec![0; n]; let mut dom = vec![Vec::new(); n];
        for i in 0..n { for j in 0..n { if i == j { continue; }
            if dominates(&pop[i], &pop[j]) { dom[i].push(j); } else if dominates(&pop[j], &pop[i]) { dc[i] += 1; }
        }}
        let mut cf: Vec<usize> = (0..n).filter(|i| dc[*i] == 0).collect();
        for &i in &cf { pop[i].rank = 1; }
        let mut rank = 1;
        while !cf.is_empty() {
            let mut nf = Vec::new();
            for &i in &cf { for &j in &dom[i] { dc[j] -= 1; if dc[j] == 0 { pop[j].rank = rank + 1; nf.push(j); } } }
            rank += 1; cf = nf;
        }
    }
    fn crowding_distance(&self, pop: &mut [MoIndividual]) {
        let no = pop[0].objectives.len(); let n = pop.len();
        for ind in pop.iter_mut() { ind.crowding_distance = 0.0; }
        for m in 0..no {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|a, b| pop[*a].objectives[m].partial_cmp(&pop[*b].objectives[m]).unwrap());
            pop[idx[0]].crowding_distance = f64::INFINITY;
            pop[idx[n-1]].crowding_distance = f64::INFINITY;
            let fmin = pop[idx[0]].objectives[m]; let fmax = pop[idx[n-1]].objectives[m];
            let range = (fmax - fmin).max(1e-10);
            for i in 1..(n-1) { pop[idx[i]].crowding_distance += (pop[idx[i+1]].objectives[m] - pop[idx[i-1]].objectives[m]) / range; }
        }
    }
    fn tournament_select(&self, pop: &[MoIndividual], rng: &mut ThreadRng
    ) -> MoIndividual {
        let i1 = rng.gen_range(0..pop.len()); let i2 = rng.gen_range(0..pop.len());
        if pop[i1].rank < pop[i2].rank { pop[i1].clone() }
        else if pop[i1].rank > pop[i2].rank { pop[i2].clone() }
        else if pop[i1].crowding_distance > pop[i2].crowding_distance { pop[i1].clone() }
        else { pop[i2].clone() }
    }
    fn mutate(&self, ind: &mut MoIndividual, rng: &mut ThreadRng) {
        let normal = Normal::new(0.0, 0.5).unwrap();
        for gene in &mut ind.genome { if rng.gen::<f64>() < self.mutation_rate { *gene += normal.sample(rng); *gene = gene.clamp(self.bounds.0, self.bounds.1); } }
    }
    pub fn init_pop(&self, dims: usize) -> Vec<MoIndividual> {
        let mut rng = thread_rng(); let u = Uniform::new_inclusive(self.bounds.0, self.bounds.1);
        (0..self.pop_size).map(|_| MoIndividual::new((0..dims).map(|_| u.sample(&mut rng)).collect())).collect()
    }
}

// ============================================================
// Benchmarks
// ============================================================

pub fn sphere(x: &[f64]) -> f64 { x.iter().map(|xi| xi * xi).sum() }
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s: f64 = x.iter().map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum();
    10.0 * n + s
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ga_sphere() {
        let mut ga = GeneticAlgorithm::default();
        ga.pop_size = 50;
        let mut pop = ga.init_pop(10);
        for _ in 0..100 { ga.evolve(&mut pop, |inds| inds.iter().map(|ind| -sphere(&ind.genome)).collect()); }
        pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        assert!(pop[0].fitness > -1.0, "GA converge");
    }

    #[test]
    fn cmaes_sphere() {
        let mut cma = CmaEs::new(10);
        let mut theta = vec![2.0; 10];
        for _ in 0..50 { cma.step(&mut theta, |x| -sphere(x)); }
        assert!(-sphere(&theta) > -2.0, "CMA-ES converge");
    }

    #[test]
    fn openes_sphere() {
        let mut openes = OpenEs::new(10);
        let mut theta = vec![2.0; 10];
        let mut best = f64::NEG_INFINITY;
        for _ in 0..100 { let r = openes.step(&mut theta, |x| -sphere(x)); if r > best { best = r; } }
        assert!(best > -5.0, "OpenES converge");
    }

    #[test]
    fn nsga2_zdt1() {
        let mut nsga = Nsga2::default();
        nsga.pop_size = 50;
        let mut pop = nsga.init_pop(10);
        for _ in 0..50 {
            nsga.evolve(&mut pop, |inds| inds.iter().map(|ind| {
                let x = &ind.genome;
                let f1 = x[0].clamp(0.0, 1.0); // clamp pour éviter division par zéro
                let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() as f64 - 1.0);
                let h = if g > 0.0 { 1.0 - (f1 / g).sqrt() } else { 1.0 };
                vec![f1, g * h]
            }).collect());
        }
        let front: Vec<&MoIndividual> = pop.iter().filter(|ind| ind.rank == 1).collect();
        assert!(!front.is_empty(), "Pareto front trouvé");
    }

    #[test]
    fn bench_rastrigin() {
        let x = vec![0.0; 10];
        assert_eq!(rastrigin(&x), 0.0);
    }
}
