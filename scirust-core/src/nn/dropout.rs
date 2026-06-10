// scirust-core/src/nn/dropout.rs
//
// Dropout — régularisation par masquage aléatoire.
//
// Inverted dropout : en mode train, les éléments conservés sont
// rescalés par 1/(1-p) pour que l'espérance de la sortie reste
// identique à l'input. En mode eval, c'est l'identité.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

pub struct Dropout {
    pub p: f32,
    pub training: bool,
    rng: PcgEngine,
}

impl Dropout {
    pub fn new(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout::new: p doit être dans [0, 1), got {}",
            p
        );
        Self {
            p,
            training: true,
            rng: PcgEngine::new(seed),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn generate_mask(&mut self, rows: usize, cols: usize) -> Tensor {
        let n = rows * cols;
        let scale = 1.0 / (1.0 - self.p);
        let mut data = vec![0.0f32; n];
        for item in data.iter_mut()
        {
            let r: f32 = self.rng.float();
            if r >= self.p
            {
                *item = scale;
            }
        }
        Tensor::from_vec(data, rows, cols)
    }
}

impl Module for Dropout {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        if !self.training
        {
            return input;
        }
        let (rows, cols) = input.shape();
        let mask = self.generate_mask(rows, cols);
        let mask_var = tape.input(mask);
        input.try_hadamard(mask_var).unwrap()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }
    fn sync(&mut self, _tape: &Tape) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn eval_mode_identity() {
        let mut dropout = Dropout::new(0.5, 42);
        dropout.set_training(false);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let y = dropout.forward(&tape, x);

        let v = tape.value(y.idx());
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn train_mode_zero_ratio() {
        let mut dropout = Dropout::new(0.5, 42);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 1000], 10, 100));
        let y = dropout.forward(&tape, x);

        let v = tape.value(y.idx());
        let n_zeros = v.data.iter().filter(|&&x| x == 0.0).count();
        let ratio = n_zeros as f32 / v.data.len() as f32;

        // Tolérance statistique large : attendu ~50%, accepte 40-60%
        assert!(
            ratio > 0.40 && ratio < 0.60,
            "Dropout p=0.5 : zero ratio = {:.2}%, expected ~50%",
            ratio * 100.0
        );
    }

    #[test]
    fn train_mode_scale() {
        let mut dropout = Dropout::new(0.5, 42);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![2.0; 100], 1, 100));
        let y = dropout.forward(&tape, x);

        let v = tape.value(y.idx());
        // Chaque élément est soit 0 soit 2.0 * scale = 4.0
        for &val in &v.data
        {
            assert!(
                val == 0.0 || (val - 4.0).abs() < 1e-5,
                "Dropout output should be 0 or 4.0, got {}",
                val
            );
        }
    }

    #[test]
    fn gradient_flows_through_nonzero() {
        let mut dropout = Dropout::new(0.5, 42);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![3.0; 10], 1, 10));
        let x_idx = x.idx();

        let y = dropout.forward(&tape, x);
        let loss = y.sum();
        loss.backward();

        let g = tape.grad(x_idx);
        let v = tape.value(y.idx());

        // Le gradient doit être non-nul UNIQUEMENT où le mask est non-nul
        // grad = upstream(1) * mask, donc soit 0 soit scale
        let scale = 1.0 / (1.0 - 0.5);
        for i in 0..10
        {
            if v.data[i] != 0.0
            {
                assert!(
                    (g.data[i] - scale).abs() < 1e-5,
                    "grad[{}] should be scale={}, got {}",
                    i,
                    scale,
                    g.data[i]
                );
            }
            else
            {
                assert_eq!(g.data[i], 0.0, "grad[{}] should be 0 where mask is 0", i);
            }
        }
    }

    #[test]
    fn reproducible_with_same_seed() {
        let mut d1 = Dropout::new(0.5, 123);
        let mut d2 = Dropout::new(0.5, 123);

        let tape1 = Tape::new();
        let x1 = tape1.input(Tensor::from_vec(vec![1.0; 20], 1, 20));
        let y1 = d1.forward(&tape1, x1);
        let v1 = tape1.value(y1.idx());

        let tape2 = Tape::new();
        let x2 = tape2.input(Tensor::from_vec(vec![1.0; 20], 1, 20));
        let y2 = d2.forward(&tape2, x2);
        let v2 = tape2.value(y2.idx());

        assert_eq!(v1.data, v2.data, "Same seed should produce identical masks");
    }

    #[test]
    fn dropout_p_zero_is_identity() {
        let mut dropout = Dropout::new(0.0, 42);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let y = dropout.forward(&tape, x);
        let v = tape.value(y.idx());
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dropout_p_high_mostly_zeros() {
        let mut dropout = Dropout::new(0.99, 42);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 1000], 10, 100));
        let y = dropout.forward(&tape, x);
        let v = tape.value(y.idx());
        let n_zeros = v.data.iter().filter(|&&x| x == 0.0).count();
        // With p=0.99, expect ~99% zeros, accept >95%
        assert!(
            n_zeros > 950,
            "Dropout p=0.99: expected mostly zeros, got {} zeros out of 1000",
            n_zeros
        );
    }
}
