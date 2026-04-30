// scirust-core/src/nn/activation.rs
//
// Wrappers Module pour les activations sans paramètres.
//
// Pourquoi en faire des Module ? Pour pouvoir les empiler dans Sequential
// au même titre que Linear. Sans paramètres : parameter_indices() retourne
// un Vec vide, sync() est un no-op.

use crate::autodiff::reverse::{Tape, Var};
use crate::nn::module::Module;

// ---------- ReLU ---------- //

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { ReLU }
}

impl Default for ReLU {
    fn default() -> Self { ReLU }
}

impl Module for ReLU {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        input.relu()
    }

    fn parameter_indices(&self) -> Vec<usize> { Vec::new() }
    fn sync(&mut self, _tape: &Tape) {}
}

// ---------- Sigmoid ---------- //

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self { Sigmoid }
}

impl Default for Sigmoid {
    fn default() -> Self { Sigmoid }
}

impl Module for Sigmoid {
    fn forward<'t>(&mut self, _tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        input.sigmoid()
    }

    fn parameter_indices(&self) -> Vec<usize> { Vec::new() }
    fn sync(&mut self, _tape: &Tape) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn relu_forward_clamps_negatives() {
        let mut act = ReLU::new();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], 1, 4));
        let y = act.forward(&tape, x);
        assert_eq!(tape.value(y.idx()).data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn sigmoid_forward_in_zero_one() {
        let mut act = Sigmoid::new();
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![-100.0, 0.0, 100.0], 1, 3));
        let y = act.forward(&tape, x);
        let v = tape.value(y.idx());
        assert!(v.data[0] >= 0.0 && v.data[0] < 0.01);
        assert!((v.data[1] - 0.5).abs() < 1e-6);
        assert!(v.data[2] > 0.99 && v.data[2] <= 1.0);
    }

    #[test]
    fn activations_have_no_parameters() {
        let relu = ReLU::new();
        let sig = Sigmoid::new();
        assert!(relu.parameter_indices().is_empty());
        assert!(sig.parameter_indices().is_empty());
    }
}
