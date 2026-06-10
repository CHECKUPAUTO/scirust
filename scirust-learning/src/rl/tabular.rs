use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

pub struct TabularAgent<S, A> {
    pub q_table: HashMap<(S, A), f64>,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

impl<S, A> TabularAgent<S, A>
where
    S: Clone + Hash + Eq,
    A: Clone + Hash + Eq,
{
    pub fn new(alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            alpha,
            gamma,
            epsilon,
        }
    }

    pub fn get_q(&self, state: &S, action: &A) -> f64 {
        *self
            .q_table
            .get(&(state.clone(), action.clone()))
            .unwrap_or(&0.0)
    }

    pub fn act_epsilon_greedy<R: Rng + ?Sized>(&self, state: &S, actions: &[A], rng: &mut R) -> A {
        if rng.gen_bool(self.epsilon)
        {
            let idx = rng.gen_range(0..actions.len());
            actions[idx].clone()
        }
        else
        {
            actions
                .iter()
                .max_by(|a1, a2| {
                    let q1 = self.get_q(state, a1);
                    let q2 = self.get_q(state, a2);
                    q1.partial_cmp(&q2).unwrap()
                })
                .unwrap()
                .clone()
        }
    }

    pub fn update_q(
        &mut self,
        state: &S,
        action: &A,
        reward: f64,
        next_state: &S,
        next_actions: &[A],
        done: bool,
    ) {
        let max_next_q = if done || next_actions.is_empty()
        {
            0.0
        }
        else
        {
            next_actions
                .iter()
                .map(|a| self.get_q(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let current_q = self.get_q(state, action);
        let new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q);
        self.q_table.insert((state.clone(), action.clone()), new_q);
    }
}
