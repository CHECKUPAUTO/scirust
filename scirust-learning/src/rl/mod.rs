//! Reinforcement Learning (RL) module.

/// Trait defining a RL environment.
pub trait Env {
    type State: Clone;
    type Action: Clone;

    /// Reset the environment to an initial state.
    fn reset(&mut self) -> Self::State;

    /// Take an action and return (next_state, reward, done).
    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool);

    /// Return available actions for the current state.
    fn available_actions(&self, state: &Self::State) -> Vec<Self::Action>;
}

/// Trait defining a RL agent.
pub trait Agent<E: Env> {
    /// Choose an action for the given state.
    fn act(&self, state: &E::State) -> E::Action;

    /// Update the agent's internal state/knowledge based on a transition.
    fn update(
        &mut self,
        state: &E::State,
        action: &E::Action,
        reward: f64,
        next_state: &E::State,
        done: bool,
    );
}

pub mod deep;
pub mod ppo;
pub mod tabular;
