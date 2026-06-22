//! # Population-Based Training (PBT)
//!
//! PBT (Jaderberg et al., 2017) trains a *population* of models that improve
//! themselves in two ways at once:
//!
//! - **exploit** — under-performing members copy the weights *and*
//!   hyper-parameters of a better member, and
//! - **explore** — they then perturb those hyper-parameters.
//!
//! So the population continually rediscovers a good hyper-parameter schedule
//! *during* training instead of fixing it up front — a self-improvement loop
//! over the optimiser's own settings. The reported best-so-far is monotone.
//!
//! Implement [`PbtTask`] and call [`Pbt::run`].

use crate::{Fitness, Guard, Report, StopReason, rng_from_seed};
use rand::Rng;
use rand::rngs::StdRng;

/// A task trainable with population-based training.
pub trait PbtTask {
    /// The mutable hyper-parameters PBT searches over (e.g. a learning rate).
    type Hyper: Clone;

    /// Initialise one member's parameters and hyper-parameters.
    fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, Self::Hyper);

    /// Run one training step: mutate `params` in place using `hyper` and return
    /// the member's new score (higher is better).
    fn step(&self, params: &mut Vec<f64>, hyper: &Self::Hyper, rng: &mut StdRng) -> Fitness;

    /// Perturb hyper-parameters when a member exploits a better one.
    fn perturb(&self, hyper: &Self::Hyper, rng: &mut StdRng) -> Self::Hyper;
}

/// One individual in the population.
#[derive(Debug, Clone)]
struct Member<H> {
    params: Vec<f64>,
    hyper: H,
    score: Fitness,
}

/// Driver for population-based training.
#[derive(Debug, Clone)]
pub struct Pbt {
    seed: u64,
    pop_size: usize,
    steps_per_gen: usize,
    /// Fraction of the worst members replaced each generation (0..0.5].
    exploit_frac: f64,
}

impl Pbt {
    /// New driver with defaults (population 16, 1 step/gen, exploit bottom 25%).
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            pop_size: 16,
            steps_per_gen: 1,
            exploit_frac: 0.25,
        }
    }

    /// Population size.
    pub fn pop_size(mut self, n: usize) -> Self {
        self.pop_size = n.max(2);
        self
    }

    /// Training steps run per member per generation.
    pub fn steps_per_gen(mut self, n: usize) -> Self {
        self.steps_per_gen = n.max(1);
        self
    }

    /// Fraction of the worst members replaced each generation.
    pub fn exploit_frac(mut self, f: f64) -> Self {
        self.exploit_frac = f.clamp(0.0, 0.5);
        self
    }

    /// Run PBT until the guard stops it. Returns the best member's parameters,
    /// its hyper-parameters, and an auditable [`Report`].
    pub fn run<T: PbtTask>(&self, task: &T, guard: &Guard) -> (Vec<f64>, T::Hyper, Report) {
        let mut rng = rng_from_seed(self.seed);

        // Initialise the population.
        let mut pop: Vec<Member<T::Hyper>> = (0..self.pop_size)
            .map(|_| {
                let (params, hyper) = task.init_member(&mut rng);
                Member {
                    params,
                    hyper,
                    score: f64::NEG_INFINITY,
                }
            })
            .collect();

        let mut best_params = pop[0].params.clone();
        let mut best_hyper = pop[0].hyper.clone();
        let mut best_so_far = f64::NEG_INFINITY;

        let mut history = Vec::with_capacity(guard.max_iters);
        let mut accepted = 0usize;
        let mut since_improve = 0usize;
        let mut iterations = 0usize;
        let mut stop_reason = StopReason::MaxIterations;

        let n_replace = ((self.pop_size as f64) * self.exploit_frac).floor() as usize;

        for gen in 0..guard.max_iters
        {
            iterations = gen + 1;

            // 1. Train every member for a few steps.
            for m in pop.iter_mut()
            {
                for _ in 0..self.steps_per_gen
                {
                    m.score = task.step(&mut m.params, &m.hyper, &mut rng);
                }
            }

            // 2. Rank by score (best first).
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&a, &b| pop[b].score.partial_cmp(&pop[a].score).unwrap());

            // 3. Exploit + explore: the worst `n_replace` copy a top member and
            //    perturb its hyper-parameters.
            if n_replace > 0
            {
                let top = &order[..n_replace.max(1)];
                let bottom: Vec<usize> = order[pop.len() - n_replace..].to_vec();
                for &loser in &bottom
                {
                    let winner = top[rng.gen_range(0..top.len())];
                    let (w_params, w_hyper, w_score) = (
                        pop[winner].params.clone(),
                        pop[winner].hyper.clone(),
                        pop[winner].score,
                    );
                    pop[loser].params = w_params;
                    pop[loser].hyper = task.perturb(&w_hyper, &mut rng);
                    pop[loser].score = w_score;
                }
            }

            // 4. Track the (monotone) best member.
            let gen_best = order[0];
            if pop[gen_best].score > best_so_far + guard.min_delta
            {
                best_so_far = pop[gen_best].score;
                best_params = pop[gen_best].params.clone();
                best_hyper = pop[gen_best].hyper.clone();
                accepted += 1;
                since_improve = 0;
            }
            else
            {
                since_improve += 1;
            }
            history.push(best_so_far);

            if let Some(t) = guard.target
            {
                if best_so_far >= t
                {
                    stop_reason = StopReason::TargetReached;
                    break;
                }
            }
            if guard.patience > 0 && since_improve >= guard.patience
            {
                stop_reason = StopReason::Converged;
                break;
            }
        }

        (
            best_params,
            best_hyper,
            Report {
                iterations,
                accepted,
                best_fitness: best_so_far,
                history,
                stop_reason,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimise a 1-D quadratic by gradient descent, where the *learning rate*
    /// is the hyper-parameter PBT must tune. Too-large rates diverge, too-small
    /// crawl; PBT should converge on a good rate and drive the score toward 0.
    struct LrSearch;
    impl PbtTask for LrSearch {
        type Hyper = f64; // learning rate

        fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, f64) {
            // Start at x = 10, with a randomly (often badly) chosen LR.
            let lr = rng.gen_range(0.001..1.5);
            (vec![10.0], lr)
        }
        fn step(&self, params: &mut Vec<f64>, hyper: &f64, _rng: &mut StdRng) -> Fitness {
            // f(x) = x^2, f'(x) = 2x. GD update with possible divergence.
            let x = params[0];
            let new_x = x - hyper * 2.0 * x;
            params[0] = new_x;
            -new_x * new_x // fitness = -loss
        }
        fn perturb(&self, hyper: &f64, rng: &mut StdRng) -> f64 {
            let factor = if rng.gen_bool(0.5) { 0.8 } else { 1.25 };
            (hyper * factor).clamp(1e-4, 1.5)
        }
    }

    #[test]
    fn pbt_tunes_learning_rate() {
        let (params, _lr, report) = Pbt::new(2024)
            .pop_size(24)
            .steps_per_gen(1)
            .run(&LrSearch, &Guard::new().max_iters(100).target(-1e-9));
        assert!(report.is_monotone(), "best-so-far must not decrease");
        assert!(
            params[0].abs() < 1e-3,
            "PBT should drive x to ~0, got {}",
            params[0]
        );
    }

    #[test]
    fn pbt_population_must_be_at_least_two() {
        let p = Pbt::new(1).pop_size(0);
        assert_eq!(p.pop_size, 2);
    }
}
