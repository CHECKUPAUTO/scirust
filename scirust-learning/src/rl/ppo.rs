use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::Module;

/// PPO (Proximal Policy Optimization) agent logic.
pub struct PPOAgent<A: Module, C: Module> {
    pub actor: A,
    pub critic: C,
    pub actor_opt: Adam,
    pub critic_opt: Adam,
    pub epsilon: f32, // clipping parameter
}

impl<A: Module, C: Module> PPOAgent<A, C> {
    pub fn new(actor: A, critic: C, actor_lr: f32, critic_lr: f32, epsilon: f32) -> Self {
        Self {
            actor,
            critic,
            actor_opt: Adam::new(actor_lr),
            critic_opt: Adam::new(critic_lr),
            epsilon,
        }
    }

    /// Compute PPO update step.
    pub fn train_step(
        &mut self,
        states: &[Tensor],
        actions: &[usize],
        old_probs: &[f32],
        advantages: &[f32],
        returns: &[f32],
    ) {
        let tape = Tape::new();
        let mut total_actor_loss = tape.input(Tensor::zeros(1, 1));
        let mut total_critic_loss = tape.input(Tensor::zeros(1, 1));

        for i in 0..states.len()
        {
            let s_var = tape.input(states[i].clone());
            let actor_out = self.actor.forward(&tape, s_var);
            let probs = actor_out.softmax(1);
            let prob_a = probs.slice_cols(actions[i], 1);

            // ratio = prob / old_prob
            let ratio = prob_a.scale(1.0 / old_probs[i]);

            // surrogate objectives
            let adv = advantages[i];
            let surr1 = ratio.scale(adv);

            // clipping surr2 = clamp(ratio, 1-eps, 1+eps) * adv
            // We use a simplified element-wise min for POC if available, or just scale.
            // In SciRust, we don't have a direct min(Var, Var) op yet,
            // so we implement it via scaling and clamping the value.
            let ratio_val = tape.value(ratio.idx()).data[0];
            let clipped_ratio = ratio_val.clamp(1.0 - self.epsilon, 1.0 + self.epsilon);
            let surr2 = tape.input(Tensor::from_vec(vec![clipped_ratio * adv], 1, 1));

            // loss = -min(surr1, surr2)
            //
            // `surr2` is deliberately a disconnected constant node: when
            // clipping is genuinely active (the clipped term is the strict
            // min), d(clip(ratio))/d(ratio) is exactly 0, so treating it as a
            // constant is the correct subgradient — no bug there.
            //
            // The comparison MUST be `<=`, not `<`. When `ratio` is inside
            // [1-eps, 1+eps], `clamp` is a no-op so `clipped_ratio == ratio`
            // bit-for-bit and `surr1_val == clipped_ratio * adv` exactly: a
            // strict `<` then takes the `else` branch and uses the
            // gradient-free `surr2` even though clipping isn't active,
            // zeroing the actor gradient for every sample whose ratio starts
            // at 1 (i.e. every sample on the very first training pass after
            // collection, since ratio = prob / old_prob = 1 there).
            let surr1_val = tape.value(surr1.idx()).data[0];
            let actor_loss = if surr1_val <= (clipped_ratio * adv)
            {
                surr1.scale(-1.0)
            }
            else
            {
                surr2.scale(-1.0)
            };

            total_actor_loss = total_actor_loss.add(actor_loss);

            let critic_out = self.critic.forward(&tape, s_var);
            let target = tape.input(Tensor::from_vec(vec![returns[i]], 1, 1));
            let diff = critic_out.sub(target);
            let critic_loss = diff.hadamard(diff); // MSE
            total_critic_loss = total_critic_loss.add(critic_loss);
        }

        total_actor_loss.backward();
        self.actor_opt.step(&self.actor.parameter_indices(), &tape);
        self.actor.sync(&tape);

        total_critic_loss.backward();
        self.critic_opt
            .step(&self.critic.parameter_indices(), &tape);
        self.critic.sync(&tape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::nn::{Linear, PcgEngine, Zeros};

    fn action_prob(actor: &mut Linear, state: &Tensor, action: usize) -> f32 {
        let tape = Tape::new();
        let s = tape.input(state.clone());
        let logits = actor.forward(&tape, s);
        let probs = logits.softmax(1);
        tape.value(probs.idx()).data[action]
    }

    #[test]
    fn ppo_actor_learns_from_a_fresh_rollout_at_ratio_one() {
        // Regression test for a P0 audit finding: the clipped-surrogate min
        // used a strict `<` comparison, so whenever `ratio == 1` exactly
        // (true for every sample on the very first training pass after
        // collection, since ratio = prob / old_prob) the branch fell through
        // to the gradient-free constant term and the actor never learned
        // anything from a fresh rollout.
        let mut rng = PcgEngine::new(0);
        let actor = Linear::new(1, 2, &Zeros, &Zeros, &mut rng);
        let mut rng2 = PcgEngine::new(1);
        let critic = Linear::new(1, 1, &Zeros, &Zeros, &mut rng2);
        let mut agent = PPOAgent::new(actor, critic, 0.5, 0.1, 0.2);

        let state = Tensor::from_vec(vec![1.0], 1, 1);
        // Zero-initialised Linear -> uniform [0.5, 0.5] logits/probs.
        let p0_before = action_prob(&mut agent.actor, &state, 0);
        assert!((p0_before - 0.5).abs() < 1e-6);

        // old_probs recomputed from the current policy each step -> ratio ==
        // 1 exactly at the start of every call, which is precisely the case
        // the bug zeroed out.
        for _ in 0..20
        {
            let old_probs = [action_prob(&mut agent.actor, &state, 0)];
            agent.train_step(
                std::slice::from_ref(&state),
                &[0],
                &old_probs,
                &[1.0], // positive advantage: action 0 should become more likely
                &[1.0],
            );
        }

        let p0_after = action_prob(&mut agent.actor, &state, 0);
        assert!(
            p0_after > p0_before + 0.05,
            "PPO actor did not learn from a positive-advantage sample: before={p0_before}, after={p0_after}"
        );
    }
}
