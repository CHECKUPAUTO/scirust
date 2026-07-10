//! Agent-in-the-loop simulation: the gym-style [`Environment`] trait and the
//! [`run_episode`] driver.
//!
//! An [`Environment`] is a simulation an external decision-maker interacts
//! with step by step — a reinforcement-learning agent, a hand-written control
//! policy, or a test harness. The `reset` / `step(action)` shape mirrors
//! `scirust_learning::rl::Env`, whose `step` also returns
//! `(state, reward, done)`, so agents written against either trait can be
//! bridged with a few lines.

/// The result of one environment step.
#[derive(Debug, Clone, PartialEq)]
pub struct Step<O> {
    /// The observation after the action was applied.
    pub observation: O,
    /// The scalar reward earned by the step.
    pub reward: f64,
    /// `true` when the episode ended with this step (goal reached, failure
    /// condition hit, or time limit exhausted); the environment must be
    /// `reset` before stepping again.
    pub done: bool,
}

/// A simulation an agent interacts with through discrete decision steps.
///
/// Implementations must be deterministic: any randomness (e.g. in the reset
/// state) comes from an explicit seed supplied at construction, so a seeded
/// environment plus a deterministic policy replays the exact same episode.
pub trait Environment {
    /// What the agent observes after each step.
    type Observation: Clone;
    /// What the agent does at each step.
    type Action: Clone;

    /// Put the environment back into a start state and return the first
    /// observation.
    fn reset(&mut self) -> Self::Observation;

    /// Apply `action`, advance the simulation and report what happened.
    /// Stepping a finished episode (after `done`) is a contract violation;
    /// implementations keep returning `done = true` rather than panicking.
    fn step(&mut self, action: &Self::Action) -> Step<Self::Observation>;
}

/// Run one episode: `reset`, then repeatedly ask `policy` for an action and
/// `step` until the episode ends or `max_steps` is reached.
///
/// Returns the total reward and the number of steps taken. A `max_steps` of
/// zero returns `(0.0, 0)` without stepping.
pub fn run_episode<E, P>(env: &mut E, mut policy: P, max_steps: usize) -> (f64, usize)
where
    E: Environment,
    P: FnMut(&E::Observation) -> E::Action,
{
    let mut observation = env.reset();
    let mut total_reward = 0.0;
    for step_index in 0..max_steps
    {
        let action = policy(&observation);
        let outcome = env.step(&action);
        total_reward += outcome.reward;
        if outcome.done
        {
            return (total_reward, step_index + 1);
        }
        observation = outcome.observation;
    }
    (total_reward, max_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts down from 3; action `true` decrements, reward 1 per step.
    struct Countdown {
        remaining: u32,
    }

    impl Environment for Countdown {
        type Action = bool;
        type Observation = u32;

        fn reset(&mut self) -> u32 {
            self.remaining = 3;
            self.remaining
        }

        fn step(&mut self, action: &bool) -> Step<u32> {
            if *action && self.remaining > 0
            {
                self.remaining -= 1;
            }
            Step {
                observation: self.remaining,
                reward: 1.0,
                done: self.remaining == 0,
            }
        }
    }

    #[test]
    fn run_episode_reports_reward_and_length() {
        let mut env = Countdown { remaining: 0 };
        let (reward, steps) = run_episode(&mut env, |_| true, 100);
        assert_eq!(steps, 3);
        assert!((reward - 3.0).abs() < 1e-12);
    }

    #[test]
    fn run_episode_honours_the_step_budget() {
        let mut env = Countdown { remaining: 0 };
        // A policy that never acts: the episode never ends on its own.
        let (reward, steps) = run_episode(&mut env, |_| false, 10);
        assert_eq!(steps, 10);
        assert!((reward - 10.0).abs() < 1e-12);
        let (reward, steps) = run_episode(&mut env, |_| true, 0);
        assert_eq!((reward, steps), (0.0, 0));
    }
}
