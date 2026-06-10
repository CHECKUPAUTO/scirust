// scirust-core/src/nn/loss/nll.rs
//
// Negative Log-Likelihood Loss.
//
// Attend des *log-probabilities* (sortie de LogSoftmax) en entrée,
// pas des logits bruts.  C'est la seconde moitié de la cross-entropy :
//
//   NLLLoss(log_probs, target) = - mean( Σᵢ targetᵢ · log_probsᵢ )
//
// Contrairement à CrossEntropyLoss, NLLLoss ne fait PAS de softmax
// interne — l'appelant doit explicitement passer par LogSoftmax
// d'abord si les sorties du modèle sont des logits.
//
// Utilisation typique :
//   let log_probs = model.forward(&tape, x).log_softmax(1);
//   let loss = NLLLoss::new().forward(&tape, log_probs, target);

use crate::autodiff::reverse::{Tape, Var};
use crate::nn::loss::Loss;

pub struct NllLoss;

impl NllLoss {
    pub fn new() -> Self {
        NllLoss
    }
}

impl Default for NllLoss {
    fn default() -> Self {
        NllLoss
    }
}

impl Loss for NllLoss {
    /// pred  : (batch, n_classes) — log-probabilities (≤ 0, sortie de LogSoftmax)
    /// target: (batch, n_classes) — encodage one-hot
    /// renvoie : Var scalaire = mean(-sum(target ⊙ pred))
    fn forward<'t>(&self, _tape: &'t Tape, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        assert_eq!(
            target.shape(),
            pred.shape(),
            "NLLLoss: target shape {:?} != pred shape {:?}",
            target.shape(),
            pred.shape()
        );

        let (batch, _n_classes) = pred.shape();

        // -sum(target ⊙ pred) / batch
        let prod = target.try_hadamard(pred).unwrap();
        prod.sum().scale(-1.0 / batch as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn nll_zero_when_target_matches_log_softmax() {
        // log_probs = log(softmax([10, 0, 0])) ≈ [0, -10, -10]
        // target one-hot classe 0 → loss ≈ 0
        let tape = Tape::new();
        let log_probs = tape.input(Tensor::from_vec(vec![0.0, -10.0, -10.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0], 1, 3));
        let loss = NllLoss::new().forward(&tape, log_probs, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0].abs() < 0.01,
            "NLL should be near 0, got {}",
            v.data[0]
        );
    }

    #[test]
    fn nll_high_when_confident_wrong() {
        // log_probs = log(softmax([10, 0, 0])) ≈ [0, -10, -10]
        // target one-hot classe 1 → loss ≈ 10
        let tape = Tape::new();
        let log_probs = tape.input(Tensor::from_vec(vec![0.0, -10.0, -10.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0, 1.0, 0.0], 1, 3));
        let loss = NllLoss::new().forward(&tape, log_probs, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0] > 9.0 && v.data[0] < 11.0,
            "NLL should be ~10, got {}",
            v.data[0]
        );
    }

    #[test]
    fn nll_gradient_flows_to_pred() {
        let tape = Tape::new();
        let log_probs = tape.input(Tensor::from_vec(vec![-1.0, -2.0, -3.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0, 1.0, 0.0], 1, 3));
        let loss = NllLoss::new().forward(&tape, log_probs, target);
        tape.backward(loss.idx());

        let g = tape.grad(log_probs.idx());
        let max_abs: f32 = g.data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "Gradient on log_probs is zero");

        // Propriété : ∂NLL/∂log_probs = -target / batch
        // batch = 1, target = [0, 1, 0] → grad = [0, -1, 0]
        assert!(
            (g.data[0]).abs() < 1e-5,
            "grad[0] = {} expected 0",
            g.data[0]
        );
        assert!(
            (g.data[1] - (-1.0)).abs() < 1e-5,
            "grad[1] = {} expected -1",
            g.data[1]
        );
        assert!(
            (g.data[2]).abs() < 1e-5,
            "grad[2] = {} expected 0",
            g.data[2]
        );
    }

    #[test]
    fn nll_multi_batch_works() {
        // batch=2, n_classes=3
        // Sample 0: target classe 0, log_probs = [0, -5, -5]
        // Sample 1: target classe 1, log_probs = [-5, 0, -5]
        // NLL = -(0 + 0) / 2 = 0
        let tape = Tape::new();
        let log_probs = tape.input(Tensor::from_vec(
            vec![0.0, -5.0, -5.0, -5.0, 0.0, -5.0],
            2,
            3,
        ));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3));
        let loss = NllLoss::new().forward(&tape, log_probs, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0].abs() < 0.01,
            "Multi-batch NLL should be near 0, got {}",
            v.data[0]
        );
    }
}
