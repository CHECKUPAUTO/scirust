// scirust-core/src/nn/loss/mse.rs
//
// Mean Squared Error.
//
//   MSE(pred, target) = (1/N) · Σᵢ (pred_i - target_i)²
//
// où N = pred.rows × pred.cols (nombre total d'éléments).
//
// Gradient (par rapport à pred) :
//   ∂MSE/∂pred_i = (2/N) · (pred_i - target_i)
//
// Mais comme on s'appuie sur l'autograd qui pousse les bonnes Op
// (Sub, Mul, Sum, Scale), on n'a rien à coder à la main pour le backward —
// la chain rule fait le travail.

use crate::autodiff::reverse::{Tape, Var};
use crate::nn::loss::Loss;

pub struct MseLoss;

impl MseLoss {
    pub fn new() -> Self {
        MseLoss
    }
}

impl Default for MseLoss {
    fn default() -> Self {
        MseLoss
    }
}

impl Loss for MseLoss {
    fn forward<'t>(&self, _tape: &'t Tape, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (rows, cols) = pred.shape();
        let n = (rows * cols) as f32;

        let diff = pred.try_sub(target).unwrap(); // pred - target
        let sq = diff.try_hadamard(diff).unwrap(); // (pred - target)²
        let s = sq.sum(); // somme
        s.scale(1.0 / n) // / N
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn mse_zero_when_pred_equals_target() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let loss = MseLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0].abs() < 1e-6,
            "MSE should be 0 when pred == target, got {}",
            v.data[0]
        );
    }

    #[test]
    fn mse_value_correct() {
        // pred = [1, 2], target = [3, 5]
        // diff = [-2, -3], sq = [4, 9], sum = 13, /N = 13/2 = 6.5
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let target = tape.input(Tensor::from_vec(vec![3.0, 5.0], 1, 2));
        let loss = MseLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(
            (v.data[0] - 6.5).abs() < 1e-5,
            "MSE = {} expected 6.5",
            v.data[0]
        );
    }

    #[test]
    fn mse_gradient_correct() {
        // ∂MSE/∂pred_i = (2/N) · (pred_i - target_i)
        // pred = [1, 2], target = [3, 5], N = 2
        // grad_pred = [(2/2)·(1-3), (2/2)·(2-5)] = [-2, -3]
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let target = tape.input(Tensor::from_vec(vec![3.0, 5.0], 1, 2));
        let loss = MseLoss::new().forward(&tape, pred, target);
        tape.backward(loss.idx());

        let g = tape.grad(pred.idx());
        assert!(
            (g.data[0] - (-2.0)).abs() < 1e-5,
            "grad[0] = {} expected -2",
            g.data[0]
        );
        assert!(
            (g.data[1] - (-3.0)).abs() < 1e-5,
            "grad[1] = {} expected -3",
            g.data[1]
        );
    }

    #[test]
    fn mse_works_with_2d_shapes() {
        // pred (2, 3), target (2, 3)
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0; 6], 2, 3));
        let loss = MseLoss::new().forward(&tape, pred, target);
        // sum(1+4+9+16+25+36)/6 = 91/6 ≈ 15.166...
        let v = tape.value(loss.idx());
        assert!((v.data[0] - 91.0 / 6.0).abs() < 1e-4);
    }
}
