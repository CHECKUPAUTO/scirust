// scirust-core/src/autodiff/optim.rs
//
// Optimiseurs pour entraînement de modèles construits sur la Tape AD.
//
// Convention :
//   - Les paramètres sont des Var créés via tape.input(...).
//   - Après backward(), tape.grad(idx) contient le gradient.
//   - L'optimiseur lit le gradient et met à jour la valeur stockée
//     dans le tape via Tape::set_value(idx, ...).
//
// Pour itérer correctement (multi-époque), recréer un Tape neuf à
// chaque pas et ré-injecter les paramètres mis à jour. La Tape
// accumule en effet les noeuds — exemple complet dans v3_features_demo.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor};

// ------------------------------------------------------------------ //
//  Trait Optimizer                                                    //
// ------------------------------------------------------------------ //

pub trait Optimizer {
    /// Applique un pas de descente sur les paramètres listés.
    /// Lit les gradients via `tape.grad(idx)`, écrit les valeurs via
    /// `tape.set_value(idx, ...)`.
    fn step(&mut self, params: &[usize], tape: &Tape);
}

// ------------------------------------------------------------------ //
//  Helper requis sur Tape (à ajouter dans reverse.rs si absent)       //
// ------------------------------------------------------------------ //
//
//  impl Tape {
//      pub fn set_value(&self, idx: usize, value: Tensor) {
//          self.values.borrow_mut()[idx] = value;
//      }
//  }
//
//  Le patch v3 l'ajoute automatiquement.

// ------------------------------------------------------------------ //
//  SGD avec momentum optionnel                                        //
// ------------------------------------------------------------------ //

pub struct Sgd {
    pub lr:        f32,
    pub momentum:  f32,
    pub weight_decay: f32,
    velocities:    HashMap<usize, Tensor>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self { lr, momentum: 0.0, weight_decay: 0.0, velocities: HashMap::new() }
    }

    pub fn with_momentum(mut self, m: f32) -> Self { self.momentum = m; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.weight_decay = wd; self }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        for &idx in params {
            let mut value = tape.value(idx);
            let grad     = tape.grad(idx);
            assert_eq!(value.shape(), grad.shape(),
                       "Sgd::step : shape mismatch param/grad");

            // Récupère ou initialise la vélocité
            let n = value.data.len();
            let v = self.velocities.entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));

            for i in 0..n {
                let mut g = grad.data[i];
                // Weight decay (L2 regularization)
                if self.weight_decay > 0.0 {
                    g += self.weight_decay * value.data[i];
                }
                // Momentum : v = momentum * v + g
                v.data[i] = self.momentum * v.data[i] + g;
                // Update : θ -= lr * v
                value.data[i] -= self.lr * v.data[i];
            }
            tape.set_value(idx, value);
        }
    }
}

// ------------------------------------------------------------------ //
//  Adam — Adaptive Moment Estimation (Kingma & Ba, 2014)              //
// ------------------------------------------------------------------ //

pub struct Adam {
    pub lr:      f32,
    pub beta1:   f32,
    pub beta2:   f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    t: usize,                       // step counter (pour bias correction)
    m: HashMap<usize, Tensor>,      // 1er moment (moyenne mobile gradient)
    v: HashMap<usize, Tensor>,      // 2e moment (moyenne mobile gradient²)
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0,
            t: 0, m: HashMap::new(), v: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1; self.beta2 = b2; self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd; self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);  // bias correction 1
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);  // bias correction 2

        for &idx in params {
            let mut value = tape.value(idx);
            let grad     = tape.grad(idx);
            assert_eq!(value.shape(), grad.shape());

            let m = self.m.entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));
            let v = self.v.entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));

            for i in 0..value.data.len() {
                let mut g = grad.data[i];
                if self.weight_decay > 0.0 {
                    g += self.weight_decay * value.data[i];
                }

                // m = β1 m + (1−β1) g
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
                // v = β2 v + (1−β2) g²
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                // bias correction
                let m_hat = m.data[i] / bc1;
                let v_hat = v.data[i] / bc2;

                // θ -= lr * m̂ / (√v̂ + ε)
                value.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            tape.set_value(idx, value);
        }
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_step_decreases_param() {
        // f(x) = x²/2, gradient = x. SGD avec lr=0.1 : x ← x - 0.1*x = 0.9 x
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0], 1, 1));

        // Calcule un faux gradient (sans passer par backward) : on triche
        // en simulant un gradient = current value
        // Pour le test on va faire backward sur f = sum(x*x)*0.5
        let half = tape.input(Tensor::from_vec(vec![0.5], 1, 1));
        let sq = x.hadamard(x);
        let loss = sq.hadamard(half).sum();
        loss.backward();

        let mut opt = Sgd::new(0.1);
        opt.step(&[x.idx()], &tape);

        let new_x = tape.value(x.idx()).data[0];
        // x ← 10 - 0.1 * 10 = 9.0
        assert!((new_x - 9.0).abs() < 1e-5, "got {new_x}");
    }

    #[test]
    fn adam_step_runs() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0, -3.0], 1, 2));
        let loss = x.hadamard(x).sum();
        loss.backward();

        let mut opt = Adam::new(0.1);
        opt.step(&[x.idx()], &tape);

        let new_x = tape.value(x.idx());
        // Le gradient est 2x = (10, -6). Adam premier pas avec bias-corrected
        // moments rapproche x de 0 par lr*sign(grad) ≈ ±0.1
        // x[0] : 5.0 - lr*~1 ≈ 4.9
        // x[1] : -3.0 - lr*~(-1) ≈ -2.9
        assert!(new_x.data[0] < 5.0, "x[0] = {} should decrease", new_x.data[0]);
        assert!(new_x.data[1] > -3.0, "x[1] = {} should increase", new_x.data[1]);
    }
}
