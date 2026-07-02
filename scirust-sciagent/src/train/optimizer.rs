use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::scheduler::LrSchedule;
use scirust_core::nn::nd_optim::newton_schulz_orthogonalize;
use std::collections::HashMap;

/// Muon optimizer (Jordan et al. 2024) adapted for the Tape AD system.
///
/// Momentum, then Newton-Schulz orthogonalisation for 2-D weight matrices,
/// plain momentum SGD for 1-D parameters (biases, norms).
pub struct MuonOptimizer {
    lr: f32,
    momentum: f32,
    ns_steps: usize,
    weight_decay: f32,
    mu: HashMap<usize, Tensor>,
}

impl MuonOptimizer {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.95,
            ns_steps: 5,
            weight_decay: 0.0,
            mu: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn with_ns_steps(mut self, steps: usize) -> Self {
        self.ns_steps = steps;
        self
    }
}

impl Optimizer for MuonOptimizer {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        for &idx in params
        {
            let mut value = tape.value(idx);
            let grad = tape.grad(idx);
            assert_eq!(
                value.shape(),
                grad.shape(),
                "Muon::step: shape mismatch param/grad (idx={})",
                idx
            );

            let rows = value.rows;
            let cols = value.cols;
            let is_matrix = rows >= 2 && cols >= 2;

            let mk = self
                .mu
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(rows, cols));

            // momentum: m = β·m + (1−β)·g
            for i in 0..value.data.len()
            {
                mk.data[i] = self.momentum * mk.data[i] + (1.0 - self.momentum) * grad.data[i];
            }

            if is_matrix
            {
                // orthogonalise the momentum, then apply
                let ortho = newton_schulz_orthogonalize(&mk.data, rows, cols, self.ns_steps);
                let scale = (rows as f32 / cols as f32).max(1.0).sqrt();
                for (v, &o) in value.data.iter_mut().zip(&ortho)
                {
                    let decay = self.weight_decay * *v;
                    *v -= self.lr * (scale * o + decay);
                }
            }
            else
            {
                // 1-D params: plain momentum SGD
                for i in 0..value.data.len()
                {
                    let decay = self.weight_decay * value.data[i];
                    value.data[i] -= self.lr * (mk.data[i] + decay);
                }
            }

            tape.set_value(idx, value);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Generic training optimizer — choose Adam or Muon via the enum.
pub enum TrainOptimizer {
    Adam(Adam),
    Muon(MuonOptimizer),
}

impl TrainOptimizer {
    pub fn new_adam(lr: f32) -> Self {
        Self::Adam(Adam::new(lr))
    }

    pub fn new_muon(lr: f32) -> Self {
        Self::Muon(MuonOptimizer::new(lr))
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        match &mut self
        {
            Self::Adam(a) => a.weight_decay = wd,
            Self::Muon(m) => m.weight_decay = wd,
        }
        self
    }

    pub fn step(&mut self, params: &[usize], tape: &Tape) {
        match self
        {
            Self::Adam(a) => a.step(params, tape),
            Self::Muon(m) => m.step(params, tape),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        match self
        {
            Self::Adam(a) => a.set_lr(lr),
            Self::Muon(m) => m.set_lr(lr),
        }
    }

    pub fn lr(&self) -> f32 {
        match self
        {
            Self::Adam(a) => a.lr(),
            Self::Muon(m) => m.lr(),
        }
    }

    pub fn apply_schedule(&mut self, scheduler: &impl LrSchedule, step: usize) {
        let lr = scheduler.lr_at(step);
        self.set_lr(lr);
    }

    pub fn clip_grad_norm(&self, tape: &Tape, max_norm: f32) {
        tape.clip_grad_norm(max_norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_core::autodiff::reverse::{Tape, Tensor};

    #[test]
    fn test_optimizer_basic() {
        let mut opt = TrainOptimizer::new_adam(0.1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0f32], 1, 1));
        let loss = x.hadamard(x).sum();
        loss.backward();
        opt.step(&[x.idx()], &tape);
        let new_x = tape.value(x.idx()).data[0];
        assert!(new_x < 10.0, "Optimizer should decrease param");
    }

    #[test]
    fn test_muon_basic() {
        let mut opt = TrainOptimizer::new_muon(0.1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0f32], 1, 1));
        let loss = x.hadamard(x).sum();
        loss.backward();
        opt.step(&[x.idx()], &tape);
        let new_x = tape.value(x.idx()).data[0];
        assert!(new_x < 10.0, "Muon should decrease param");
    }

    #[test]
    fn test_muon_matrix() {
        let mut opt = TrainOptimizer::new_muon(0.1);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let orig_sum: f32 = tape.value(x.idx()).data.iter().sum();
        let loss = x.hadamard(x).sum();
        loss.backward();
        opt.step(&[x.idx()], &tape);
        let new_sum: f32 = tape.value(x.idx()).data.iter().sum();
        assert!(new_sum < orig_sum, "Muon should decrease matrix params");
    }

    #[test]
    fn test_muon_deterministic() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let loss = a.hadamard(a).sum();
        loss.backward();

        let tape2 = Tape::new();
        let a2 = tape2.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let loss2 = a2.hadamard(a2).sum();
        loss2.backward();

        let mut opt1 = TrainOptimizer::new_muon(0.1);
        let mut opt2 = TrainOptimizer::new_muon(0.1);
        opt1.step(&[a.idx()], &tape);
        opt2.step(&[a2.idx()], &tape2);
        assert_eq!(
            tape.value(a.idx()).data,
            tape2.value(a2.idx()).data,
            "Muon should be deterministic"
        );
    }

    #[test]
    fn test_apply_schedule() {
        struct FakeSched;
        impl LrSchedule for FakeSched {
            fn lr_at(&self, _step: usize) -> f32 {
                0.5
            }
        }
        let mut opt = TrainOptimizer::new_adam(0.001);
        opt.apply_schedule(&FakeSched, 0);
        assert!((opt.lr() - 0.5).abs() < 1e-7);
    }
}
