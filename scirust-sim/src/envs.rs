//! Ready-made control environments implementing the gym-style
//! [`Environment`] trait: the classic cart-pole balancing task and a
//! deterministic grid world.

use crate::engine::SimError;
use crate::env::{Environment, Step};
use crate::rng::SplitMix64;

/// Which way the cart is pushed at a [`CartPole`] step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Push {
    /// Apply the force toward negative `x`.
    Left,
    /// Apply the force toward positive `x`.
    Right,
}

/// The classic cart-pole balancing task (Barto, Sutton & Anderson 1983),
/// with the physics constants of the reference implementation: a 1 kg cart
/// on a frictionless track, a 0.1 kg pole of half-length 0.5 m, a ±10 N
/// bang-bang force, integrated at 0.02 s per step.
///
/// Observation `[x, ẋ, θ, θ̇]`; reward 1 per step; the episode ends when
/// `|x| > 2.4` m or `|θ| > 12°`. Reset draws each state component uniformly
/// from `[-0.05, 0.05)` using the seed supplied at construction, so episodes
/// replay bit-identically for equal seeds.
#[derive(Debug, Clone)]
pub struct CartPole {
    state: [f64; 4],
    rng: SplitMix64,
    done: bool,
}

impl CartPole {
    const FORCE_MAG: f64 = 10.0;
    const GRAVITY: f64 = 9.8;
    const HALF_LENGTH: f64 = 0.5;
    const MASS_CART: f64 = 1.0;
    const MASS_POLE: f64 = 0.1;
    const TAU: f64 = 0.02;
    /// 12 degrees, the classic failure angle.
    const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
    const X_THRESHOLD: f64 = 2.4;

    /// Create the environment with the seed that drives every reset.
    pub fn new(seed: u64) -> Self {
        CartPole {
            state: [0.0; 4],
            rng: SplitMix64::new(seed),
            done: true,
        }
    }

    /// The current observation `[x, ẋ, θ, θ̇]`.
    pub fn observation(&self) -> [f64; 4] {
        self.state
    }
}

impl Environment for CartPole {
    type Action = Push;
    type Observation = [f64; 4];

    fn reset(&mut self) -> [f64; 4] {
        for component in &mut self.state
        {
            *component = -0.05 + 0.1 * self.rng.next_f64();
        }
        self.done = false;
        self.state
    }

    fn step(&mut self, action: &Push) -> Step<[f64; 4]> {
        if self.done
        {
            // Stepping a finished episode is a no-op, per the trait contract.
            return Step {
                observation: self.state,
                reward: 0.0,
                done: true,
            };
        }
        let force = match action
        {
            Push::Left => -Self::FORCE_MAG,
            Push::Right => Self::FORCE_MAG,
        };
        let [x, x_dot, theta, theta_dot] = self.state;
        let total_mass = Self::MASS_CART + Self::MASS_POLE;
        let polemass_length = Self::MASS_POLE * Self::HALF_LENGTH;
        let (sin_theta, cos_theta) = theta.sin_cos();

        let temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass;
        let theta_acc = (Self::GRAVITY * sin_theta - cos_theta * temp)
            / (Self::HALF_LENGTH
                * (4.0 / 3.0 - Self::MASS_POLE * cos_theta * cos_theta / total_mass));
        let x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass;

        // Explicit Euler in the reference implementation's update order.
        self.state = [
            x + Self::TAU * x_dot,
            x_dot + Self::TAU * x_acc,
            theta + Self::TAU * theta_dot,
            theta_dot + Self::TAU * theta_acc,
        ];
        self.done =
            self.state[0].abs() > Self::X_THRESHOLD || self.state[2].abs() > Self::THETA_THRESHOLD;
        Step {
            observation: self.state,
            reward: 1.0,
            done: self.done,
        }
    }
}

/// A movement direction in a [`GridWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Increase `y` by one.
    Up,
    /// Decrease `y` by one.
    Down,
    /// Decrease `x` by one.
    Left,
    /// Increase `x` by one.
    Right,
}

/// A deterministic grid world: cells `(x, y)` with `x < width`, `y < height`,
/// a start, a goal and optional walls. Each step costs a reward of `-1`;
/// moving into a wall or off the grid leaves the agent in place; the episode
/// ends on reaching the goal, so an episode's total reward is minus the
/// number of steps taken — maximized by the shortest path.
///
/// The canonical fully-observable shortest-path task for tabular RL agents
/// and planning tests; the dynamics contain no randomness at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridWorld {
    width: usize,
    height: usize,
    start: (usize, usize),
    goal: (usize, usize),
    walls: Vec<(usize, usize)>,
    position: (usize, usize),
    done: bool,
}

impl GridWorld {
    /// Create a grid of the given size with a start and a goal cell, both in
    /// bounds and distinct.
    pub fn new(
        width: usize,
        height: usize,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Result<Self, SimError> {
        if width == 0 || height == 0
        {
            return Err(SimError::BadInput(
                "grid must have positive size".to_string(),
            ));
        }
        for (name, cell) in [("start", start), ("goal", goal)]
        {
            if cell.0 >= width || cell.1 >= height
            {
                return Err(SimError::BadInput(format!(
                    "{name} {cell:?} is outside the {width}x{height} grid"
                )));
            }
        }
        if start == goal
        {
            return Err(SimError::BadInput("start and goal must differ".to_string()));
        }
        Ok(GridWorld {
            width,
            height,
            start,
            goal,
            walls: Vec::new(),
            position: start,
            done: false,
        })
    }

    /// Add a wall cell; it must be in bounds and on neither the start nor the
    /// goal.
    pub fn add_wall(&mut self, cell: (usize, usize)) -> Result<(), SimError> {
        if cell.0 >= self.width || cell.1 >= self.height
        {
            return Err(SimError::BadInput(format!(
                "wall {cell:?} is outside the {}x{} grid",
                self.width, self.height
            )));
        }
        if cell == self.start || cell == self.goal
        {
            return Err(SimError::BadInput(
                "a wall cannot cover the start or the goal".to_string(),
            ));
        }
        if !self.walls.contains(&cell)
        {
            self.walls.push(cell);
        }
        Ok(())
    }

    /// The goal cell.
    pub fn goal(&self) -> (usize, usize) {
        self.goal
    }
}

impl Environment for GridWorld {
    type Action = Move;
    type Observation = (usize, usize);

    fn reset(&mut self) -> (usize, usize) {
        self.position = self.start;
        self.done = false;
        self.position
    }

    fn step(&mut self, action: &Move) -> Step<(usize, usize)> {
        if self.done
        {
            return Step {
                observation: self.position,
                reward: 0.0,
                done: true,
            };
        }
        let (x, y) = self.position;
        let target = match action
        {
            Move::Up if y + 1 < self.height => (x, y + 1),
            Move::Down if y > 0 => (x, y - 1),
            Move::Left if x > 0 => (x - 1, y),
            Move::Right if x + 1 < self.width => (x + 1, y),
            _ => (x, y),
        };
        if !self.walls.contains(&target)
        {
            self.position = target;
        }
        self.done = self.position == self.goal;
        Step {
            observation: self.position,
            reward: -1.0,
            done: self.done,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::run_episode;

    /// The classic hand policy: push toward the side the pole is falling to.
    fn lean_policy(obs: &[f64; 4]) -> Push {
        if obs[2] + obs[3] > 0.0
        {
            Push::Right
        }
        else
        {
            Push::Left
        }
    }

    #[test]
    // Ignored under Miri: Miri deliberately perturbs the last bits of the
    // transcendental float intrinsics (exp/ln/sin/cos) to model their
    // platform freedom, so bit-identity across runs is not expected under
    // the interpreter. Native Build & Test jobs enforce it.
    #[cfg_attr(miri, ignore)]
    fn cartpole_episodes_replay_bit_identically_for_equal_seeds() {
        let run = |seed: u64| {
            let mut env = CartPole::new(seed);
            let first = env.reset();
            let mut rewards = 0.0;
            let mut states = vec![first];
            loop
            {
                let outcome = env.step(&lean_policy(states.last().unwrap()));
                rewards += outcome.reward;
                states.push(outcome.observation);
                if outcome.done || states.len() > 500
                {
                    return (rewards, states);
                }
            }
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11).1[0], run(12).1[0]);
    }

    #[test]
    fn reset_draws_all_components_within_the_documented_range() {
        let mut env = CartPole::new(3);
        for _ in 0..10
        {
            let obs = env.reset();
            assert!(obs.iter().all(|c| (-0.05..0.05).contains(c)), "{obs:?}");
        }
    }

    #[test]
    fn balancing_beats_pushing_blindly() {
        // A constant push topples the pole almost immediately; the classic
        // lean-direction policy keeps it up for much longer.
        let mut env = CartPole::new(7);
        let (_, blind_steps) = run_episode(&mut env, |_| Push::Right, 500);
        let (balanced_reward, balanced_steps) = run_episode(&mut env, lean_policy, 500);
        assert!(
            blind_steps < 100,
            "blind policy survived {blind_steps} steps"
        );
        assert!(
            balanced_steps >= 3 * blind_steps,
            "lean policy: {balanced_steps} vs blind: {blind_steps}"
        );
        assert!((balanced_reward - balanced_steps as f64).abs() < 1e-12);
    }

    #[test]
    fn stepping_a_finished_cartpole_episode_stays_done_without_reward() {
        let mut env = CartPole::new(5);
        env.reset();
        // Push one way until failure.
        while !env.step(&Push::Right).done
        {}
        let after = env.step(&Push::Left);
        assert!(after.done);
        assert!((after.reward).abs() < 1e-15);
    }

    #[test]
    fn greedy_gridworld_policy_takes_exactly_the_manhattan_distance() {
        let mut world = GridWorld::new(6, 5, (0, 0), (4, 3)).unwrap();
        let goal = world.goal();
        let policy = |pos: &(usize, usize)| {
            if pos.0 < goal.0
            {
                Move::Right
            }
            else if pos.0 > goal.0
            {
                Move::Left
            }
            else if pos.1 < goal.1
            {
                Move::Up
            }
            else
            {
                Move::Down
            }
        };
        let (reward, steps) = run_episode(&mut world, policy, 100);
        assert_eq!(steps, 7); // Manhattan distance |4-0| + |3-0|
        assert!((reward - (-7.0)).abs() < 1e-12);
    }

    #[test]
    fn walls_and_edges_block_movement() {
        let mut world = GridWorld::new(3, 3, (0, 0), (2, 2)).unwrap();
        world.add_wall((1, 0)).unwrap();
        let start = world.reset();
        // Blocked by the wall to the right: the agent stays put.
        let outcome = world.step(&Move::Right);
        assert_eq!(outcome.observation, start);
        // Blocked by the grid edge below and to the left.
        assert_eq!(world.step(&Move::Down).observation, start);
        assert_eq!(world.step(&Move::Left).observation, start);
        // Moving up is free.
        assert_eq!(world.step(&Move::Up).observation, (0, 1));
    }

    #[test]
    fn gridworld_construction_is_validated() {
        assert!(GridWorld::new(0, 3, (0, 0), (1, 1)).is_err());
        assert!(GridWorld::new(3, 3, (3, 0), (1, 1)).is_err());
        assert!(GridWorld::new(3, 3, (0, 0), (0, 3)).is_err());
        assert!(GridWorld::new(3, 3, (1, 1), (1, 1)).is_err());
        let mut world = GridWorld::new(3, 3, (0, 0), (2, 2)).unwrap();
        assert!(world.add_wall((5, 5)).is_err());
        assert!(world.add_wall((0, 0)).is_err());
        assert!(world.add_wall((2, 2)).is_err());
        assert!(world.add_wall((1, 1)).is_ok());
    }
}
