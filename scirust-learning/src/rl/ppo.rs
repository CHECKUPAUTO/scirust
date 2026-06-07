use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::nn::Module;
use scirust_core::autodiff::optim::{Optimizer, Adam};

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

        for i in 0..states.len() {
            let s_var = tape.input(states[i].clone());
            let actor_out = self.actor.forward(&tape, s_var.clone());
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
            // We take the smaller of the two surrogates to be conservative
            let surr1_val = tape.value(surr1.idx()).data[0];
            let actor_loss = if surr1_val < (clipped_ratio * adv) {
                surr1.scale(-1.0)
            } else {
                surr2.scale(-1.0)
            };

            total_actor_loss = total_actor_loss.add(actor_loss);

            let critic_out = self.critic.forward(&tape, s_var);
            let target = tape.input(Tensor::from_vec(vec![returns[i]], 1, 1));
            let diff = critic_out.sub(target);
            let critic_loss = diff.hadamard(diff.clone()); // MSE
            total_critic_loss = total_critic_loss.add(critic_loss);
        }

        total_actor_loss.backward();
        self.actor_opt.step(&self.actor.parameter_indices(), &tape);
        self.actor.sync(&tape);

        total_critic_loss.backward();
        self.critic_opt.step(&self.critic.parameter_indices(), &tape);
        self.critic.sync(&tape);
    }
}
