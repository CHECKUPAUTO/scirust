//! Bridge from this crate's [`Environment`] to `scirust_learning::rl::Env`
//! (enabled by the `rl` feature).
//!
//! Wrapping any [`Environment`] that also implements [`FiniteActionSpace`] in
//! [`RlEnv`] lets the reinforcement-learning agents in `scirust-learning`
//! (tabular Q-learning, PPO, deep) drive it directly — the two traits already
//! agree on the `reset` / `step → (state, reward, done)` shape, and the
//! adapter supplies the `available_actions` the RL trait additionally needs.
//!
//! ```
//! # // (requires the `rl` feature)
//! use scirust_sim::envs::GridWorld;
//! use scirust_sim::rl_bridge::RlEnv;
//! use scirust_learning::rl::Env;
//!
//! let mut env = RlEnv::new(GridWorld::new(4, 4, (0, 0), (3, 3)).unwrap());
//! let start = env.reset();
//! assert_eq!(env.available_actions(&start).len(), 4);
//! ```

use crate::env::{Environment, FiniteActionSpace};
use scirust_learning::rl::Env;

/// Adapter that presents a [`FiniteActionSpace`] environment as a
/// `scirust_learning::rl::Env`.
///
/// The observation becomes the RL `State` and the action the RL `Action`; the
/// reward and `done` flag pass through unchanged. `available_actions` reports
/// the wrapped environment's full (state-independent) action set.
#[derive(Debug, Clone)]
pub struct RlEnv<E>(pub E);

impl<E> RlEnv<E> {
    /// Wrap an environment.
    pub fn new(env: E) -> Self {
        RlEnv(env)
    }

    /// Borrow the wrapped environment.
    pub fn inner(&self) -> &E {
        &self.0
    }

    /// Mutably borrow the wrapped environment.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.0
    }

    /// Unwrap, returning the environment.
    pub fn into_inner(self) -> E {
        self.0
    }
}

impl<E> Env for RlEnv<E>
where
    E: Environment + FiniteActionSpace,
{
    type State = E::Observation;
    type Action = E::Action;

    fn reset(&mut self) -> Self::State {
        self.0.reset()
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool) {
        let outcome = self.0.step(action);
        (outcome.observation, outcome.reward, outcome.done)
    }

    fn available_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        self.0.actions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::{CartPole, GridWorld, Move, Push};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use scirust_learning::rl::tabular::TabularAgent;

    #[test]
    fn adapter_exposes_the_finite_action_set() {
        let mut grid = RlEnv::new(GridWorld::new(5, 5, (0, 0), (4, 4)).unwrap());
        let s0 = grid.reset();
        assert_eq!(s0, (0, 0));
        assert_eq!(
            grid.available_actions(&s0),
            vec![Move::Up, Move::Down, Move::Left, Move::Right]
        );
        let (s1, reward, done) = grid.step(&Move::Right);
        assert_eq!(s1, (1, 0));
        assert!((reward + 1.0).abs() < 1e-12 && !done);

        let mut pole = RlEnv::new(CartPole::new(1));
        let obs = pole.reset();
        assert_eq!(pole.available_actions(&obs), vec![Push::Left, Push::Right]);
        assert!(pole.available_actions(&obs).len() == 2);
    }

    /// A hand-coded optimal policy driven purely through the RL `Env` trait
    /// reaches the goal in the Manhattan distance — this checks the adapter
    /// plumbing independently of any learning dynamics.
    #[test]
    fn optimal_policy_through_the_rl_trait_takes_the_manhattan_distance() {
        let mut env = RlEnv::new(GridWorld::new(6, 5, (0, 0), (4, 3)).unwrap());
        let goal = (4usize, 3usize);
        let mut state = env.reset();
        let mut steps = 0;
        loop
        {
            let action = if state.0 < goal.0
            {
                Move::Right
            }
            else if state.1 < goal.1
            {
                Move::Up
            }
            else
            {
                unreachable!("already at the goal column and row")
            };
            let (next, _reward, done) = env.step(&action);
            steps += 1;
            state = next;
            if done
            {
                break;
            }
            assert!(steps < 50, "policy failed to reach the goal");
        }
        assert_eq!(steps, 4 + 3); // Manhattan distance from (0,0) to (4,3)
    }

    /// Tabular Q-learning, run against `RlEnv<GridWorld>` through the RL trait,
    /// converges to the shortest path: the greedy policy reaches the goal in
    /// exactly the Manhattan distance. This is the end-to-end proof that a
    /// `scirust-learning` agent trains on a `scirust-sim` environment.
    #[test]
    fn tabular_q_learning_finds_the_shortest_path() {
        let mut env = RlEnv::new(GridWorld::new(5, 5, (0, 0), (4, 4)).unwrap());
        let manhattan = 4 + 4;
        let mut agent: TabularAgent<(usize, usize), Move> = TabularAgent::new(0.5, 0.95, 0.2);
        let mut rng = StdRng::seed_from_u64(42);

        // Train.
        for _ in 0..5_000
        {
            let mut state = env.reset();
            for _ in 0..200
            {
                let actions = env.available_actions(&state);
                let action = agent
                    .act_epsilon_greedy(&state, &actions, &mut rng)
                    .unwrap();
                let (next, reward, done) = env.step(&action);
                let next_actions = env.available_actions(&next);
                agent.update_q(&state, &action, reward, &next, &next_actions, done);
                state = next;
                if done
                {
                    break;
                }
            }
        }

        // Evaluate the greedy policy (no exploration).
        agent.epsilon = 0.0;
        let mut state = env.reset();
        let mut steps = 0;
        let mut reached = false;
        for _ in 0..100
        {
            let actions = env.available_actions(&state);
            let action = agent
                .act_epsilon_greedy(&state, &actions, &mut rng)
                .unwrap();
            let (next, _reward, done) = env.step(&action);
            steps += 1;
            state = next;
            if done
            {
                reached = true;
                break;
            }
        }
        assert!(reached, "greedy policy never reached the goal");
        assert_eq!(
            steps, manhattan,
            "greedy path is not the shortest ({steps} steps)"
        );
    }
}
