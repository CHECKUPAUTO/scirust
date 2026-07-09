use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::Module;
use scirust_core::nn::rng::PcgEngine;

/// Deep Q-Network (DQN) Agent.
pub struct DQNAgent<M: Module> {
    pub model: M,
    pub target_model: M,
    pub optimizer: Adam,
    pub gamma: f64,
    pub epsilon: f64,
    pub memory: Vec<Transition>,
    pub batch_size: usize,
    pub rng: PcgEngine,
}

#[derive(Clone)]
pub struct Transition {
    pub state: Tensor,
    pub action: usize,
    pub reward: f64,
    pub next_state: Tensor,
    pub done: bool,
}

impl<M: Module> DQNAgent<M> {
    pub fn new(
        model: M,
        target_model: M,
        optimizer: Adam,
        gamma: f64,
        epsilon: f64,
        batch_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            model,
            target_model,
            optimizer,
            gamma,
            epsilon,
            memory: Vec::new(),
            batch_size,
            rng: PcgEngine::new(seed),
        }
    }

    pub fn act(&mut self, state: &Tensor, action_dim: usize) -> usize {
        if self.rng.float() < self.epsilon as f32
        {
            (self.rng.next_u32() as usize) % action_dim
        }
        else
        {
            let tape = Tape::new();
            let s_var = tape.input(state.clone());
            let q_values = self.model.forward(&tape, s_var);
            let vals = tape.value(q_values.idx());
            // Argmax over Q-values, robust to NaN/inf: a diverging network can
            // emit non-finite Q-values, and `partial_cmp(NaN).unwrap()` would
            // panic (crashing action selection). Filter to finite values first —
            // then `partial_cmp` is total and never `None`; if every value is
            // non-finite, fall back to action 0.
            vals.data
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    pub fn remember(&mut self, transition: Transition) {
        self.memory.push(transition);
        if self.memory.len() > 10000
        {
            self.memory.remove(0);
        }
    }

    pub fn train_step(&mut self) {
        if self.memory.len() < self.batch_size
        {
            return;
        }

        // Fisher-Yates shuffle for sampling (deterministic)
        let mut indices: Vec<usize> = (0..self.memory.len()).collect();
        for i in (1..indices.len()).rev()
        {
            let j = (self.rng.next_u32() as usize) % (i + 1);
            indices.swap(i, j);
        }
        let batch_indices = &indices[0..self.batch_size];

        let tape = Tape::new();
        let mut total_loss = tape.input(Tensor::from_vec(vec![0.0], 1, 1));

        for &idx in batch_indices
        {
            let t = &self.memory[idx];
            let s_var = tape.input(t.state.clone());
            let q_values = self.model.forward(&tape, s_var);

            let target = if t.done
            {
                t.reward as f32
            }
            else
            {
                let ns_var = tape.input(t.next_state.clone());
                let next_q_values = self.target_model.forward(&tape, ns_var);
                let next_vals = tape.value(next_q_values.idx());
                let max_next_q = next_vals
                    .data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (t.reward as f32) + (self.gamma as f32) * max_next_q
            };

            let q_a = q_values.slice_cols(t.action, 1);
            let target_var = tape.input(Tensor::from_vec(vec![target], 1, 1));
            let diff = q_a.sub(target_var);
            let loss = diff.hadamard(diff);
            total_loss = total_loss.add(loss);
        }

        total_loss.backward();
        self.optimizer.step(&self.model.parameter_indices(), &tape);
        self.model.sync(&tape);
    }
}
