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

    pub fn act_epsilon_greedy<R: Rng + ?Sized>(
        &self,
        state: &S,
        actions: &[A],
        rng: &mut R,
    ) -> Option<A> {
        if actions.is_empty()
        {
            return None;
        }
        if rng.gen_bool(self.epsilon)
        {
            let idx = rng.gen_range(0..actions.len());
            Some(actions[idx].clone())
        }
        else
        {
            actions
                .iter()
                .max_by(|a1, a2| {
                    let q1 = self.get_q(state, a1);
                    let q2 = self.get_q(state, a2);
                    q1.total_cmp(&q2)
                })
                .cloned()
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn act_epsilon_greedy_empty_actions_returns_none() {
        // Before the fix this panicked (gen_range(0..0) or max_by().unwrap()
        // on an empty iterator). Now it must return None for both branches.
        let mut rng = StdRng::seed_from_u64(7);
        let no_actions: [&str; 0] = [];

        // epsilon = 1.0 forces the exploration branch.
        let explore: TabularAgent<&str, &str> = TabularAgent::new(0.1, 0.9, 1.0);
        assert_eq!(
            explore.act_epsilon_greedy(&"s", &no_actions, &mut rng),
            None
        );

        // epsilon = 0.0 forces the exploitation (max_by) branch.
        let exploit: TabularAgent<&str, &str> = TabularAgent::new(0.1, 0.9, 0.0);
        assert_eq!(
            exploit.act_epsilon_greedy(&"s", &no_actions, &mut rng),
            None
        );
    }

    #[test]
    fn act_epsilon_greedy_handles_nan_q_values() {
        // Before the fix, partial_cmp(&q2).unwrap() panicked when a stored
        // Q-value was NaN. total_cmp is total over all f64, so this must not
        // panic and must still return one of the provided actions.
        let mut rng = StdRng::seed_from_u64(1);
        let mut agent: TabularAgent<&str, &str> = TabularAgent::new(0.1, 0.9, 0.0);
        agent.q_table.insert(("s", "a"), f64::NAN);
        agent.q_table.insert(("s", "b"), 1.0);

        let actions = ["a", "b"];
        let chosen = agent
            .act_epsilon_greedy(&"s", &actions, &mut rng)
            .expect("non-empty action set yields Some");
        assert!(actions.contains(&chosen));
    }

    #[test]
    fn act_epsilon_greedy_exploits_best_action() {
        // Determinism sanity check for the greedy branch (epsilon = 0.0).
        let mut rng = StdRng::seed_from_u64(3);
        let mut agent: TabularAgent<&str, &str> = TabularAgent::new(0.1, 0.9, 0.0);
        agent.q_table.insert(("s", "a"), 0.5);
        agent.q_table.insert(("s", "b"), 2.0);
        agent.q_table.insert(("s", "c"), -1.0);

        let actions = ["a", "b", "c"];
        assert_eq!(
            agent.act_epsilon_greedy(&"s", &actions, &mut rng),
            Some("b")
        );
    }
}
