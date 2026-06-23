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

use crate::{Fitness, Guard, LoopState, Report, rng_from_seed};
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
        let mut ctrl = LoopState::new(guard, f64::NEG_INFINITY);

        let n_replace = ((self.pop_size as f64) * self.exploit_frac).floor() as usize;

        while ctrl.next_iter()
        {
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
            if ctrl.offer(pop[gen_best].score)
            {
                best_params = pop[gen_best].params.clone();
                best_hyper = pop[gen_best].hyper.clone();
            }
            if ctrl.done()
            {
                break;
            }
        }

        (best_params, best_hyper, ctrl.into_report())
    }

    /// Like [`run`](Self::run) but trains the population **in parallel** (rayon):
    /// each member owns a deterministically-seeded RNG and is trained on its own
    /// thread, which is the natural parallelism in PBT (members are independent
    /// between the exploit/explore syncs). Worth it when [`PbtTask::step`] is
    /// expensive. Requires the `parallel` feature.
    ///
    /// Per-member RNGs are seeded from the driver seed, so the run is
    /// **reproducible** (same seed ⇒ same result, regardless of thread count).
    /// It is *not* bit-identical to the sequential [`run`](Self::run): PBT draws
    /// its random numbers in a different order when members advance
    /// concurrently. Selection/perturbation still happens sequentially under a
    /// single coordination RNG, so it stays deterministic.
    #[cfg(feature = "parallel")]
    pub fn run_parallel<T>(&self, task: &T, guard: &Guard) -> (Vec<f64>, T::Hyper, Report)
    where
        T: PbtTask + Sync,
        T::Hyper: Send,
    {
        use rayon::prelude::*;

        // One deterministic RNG per member (independent streams) + one shared
        // coordination RNG for the sequential exploit/explore step.
        const SPREAD: u64 = 0x9E37_79B9_7F4A_7C15; // golden-ratio odd constant
        let mut rngs: Vec<StdRng> = (0..self.pop_size)
            .map(|i| rng_from_seed(self.seed ^ SPREAD.wrapping_mul(i as u64 + 1)))
            .collect();
        let mut coord = rng_from_seed(self.seed);

        let mut pop: Vec<Member<T::Hyper>> = rngs
            .iter_mut()
            .map(|r| {
                let (params, hyper) = task.init_member(r);
                Member {
                    params,
                    hyper,
                    score: f64::NEG_INFINITY,
                }
            })
            .collect();

        let mut best_params = pop[0].params.clone();
        let mut best_hyper = pop[0].hyper.clone();
        let mut ctrl = LoopState::new(guard, f64::NEG_INFINITY);

        let n_replace = ((self.pop_size as f64) * self.exploit_frac).floor() as usize;

        while ctrl.next_iter()
        {
            // 1. Train every member in parallel, each with its own RNG.
            pop.par_iter_mut()
                .zip(rngs.par_iter_mut())
                .for_each(|(m, r)| {
                    for _ in 0..self.steps_per_gen
                    {
                        m.score = task.step(&mut m.params, &m.hyper, r);
                    }
                });

            // 2. Rank by score (best first).
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&a, &b| pop[b].score.partial_cmp(&pop[a].score).unwrap());

            // 3. Exploit + explore under the coordination RNG (sequential).
            if n_replace > 0
            {
                let top = &order[..n_replace.max(1)];
                let bottom: Vec<usize> = order[pop.len() - n_replace..].to_vec();
                for &loser in &bottom
                {
                    let winner = top[coord.gen_range(0..top.len())];
                    let (w_params, w_hyper, w_score) = (
                        pop[winner].params.clone(),
                        pop[winner].hyper.clone(),
                        pop[winner].score,
                    );
                    pop[loser].params = w_params;
                    pop[loser].hyper = task.perturb(&w_hyper, &mut coord);
                    pop[loser].score = w_score;
                }
            }

            // 4. Track the (monotone) best member.
            let gen_best = order[0];
            if ctrl.offer(pop[gen_best].score)
            {
                best_params = pop[gen_best].params.clone();
                best_hyper = pop[gen_best].hyper.clone();
            }
            if ctrl.done()
            {
                break;
            }
        }

        (best_params, best_hyper, ctrl.into_report())
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

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_pbt_is_reproducible_and_tunes_lr() {
        let mk = || {
            Pbt::new(2024)
                .pop_size(24)
                .steps_per_gen(1)
                .run_parallel(&LrSearch, &Guard::new().max_iters(100).target(-1e-9))
        };
        let (pa, _la, ra) = mk();
        let (pb, _lb, rb) = mk();
        // Same seed ⇒ identical result, independent of thread scheduling.
        assert_eq!(pa, pb, "parallel PBT must be reproducible");
        assert_eq!(ra.history, rb.history, "convergence curve must be stable");
        assert!(ra.is_monotone(), "best-so-far must not decrease");
        assert!(
            pa[0].abs() < 1e-3,
            "PBT should drive x to ~0, got {}",
            pa[0]
        );
    }

    #[test]
    fn pbt_population_must_be_at_least_two() {
        let p = Pbt::new(1).pop_size(0);
        assert_eq!(p.pop_size, 2);
    }
}
