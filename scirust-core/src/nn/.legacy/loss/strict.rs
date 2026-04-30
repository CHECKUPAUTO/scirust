// scirust-core/src/nn/loss/strict.rs
//
// Losses strictes unifiées — remplace les 3 versions précédentes :
//
//   v5     : BCE strict, CE naïf (incorrect pour batch > 1)
//   v6     : softmax row-wise correct via SumAxis + Reciprocal
//   v6.1   : log_softmax stable via max-trick (notre version finale)
//
// Toutes les fonctions ici sont les versions FINALES correctes :
//   - softmax           : row-wise stable
//   - log_softmax       : max-trick stable, gère les logits jusqu'à 1e3+
//   - CrossEntropyLoss  : utilise log_softmax stable
//   - BceLoss           : BCE strict avec log
//   - MaeLoss           : smooth L1 (sqrt(diff² + eps))

use crate::autodiff::reverse::{Tensor, Var};
use crate::nn::loss::Loss;

// ================================================================== //
//  Softmax — version row-wise stable                                  //
// ================================================================== //

/// softmax stable via max-trick.
/// Input  : (N, C)
/// Output : (N, C) avec chaque ligne sommant à 1.
pub fn softmax<'t>(logits: Var<'t>) -> Var<'t> {
    let max_per_row = logits.clone().max_axis(1);          // (N, 1)
    let neg_max = max_per_row.neg();
    let shifted = logits.add_broadcast(neg_max);           // (N, C), max=0
    let exp_s = shifted.exp();
    let sum_s = exp_s.clone().sum_axis(1);                 // (N, 1)
    let inv = sum_s.reciprocal();
    exp_s.mul_broadcast(inv)
}

/// log_softmax stable via max-trick.
/// log_softmax(x)[i,j] = (x[i,j] - max_i) - log(Σ exp(x[i,k] - max_i))
pub fn log_softmax<'t>(logits: Var<'t>) -> Var<'t> {
    let max_per_row = logits.clone().max_axis(1);
    let shifted = logits.add_broadcast(max_per_row.neg());
    let exp_shifted = shifted.clone().exp();
    let log_sum = exp_shifted.sum_axis(1).log();
    shifted.add_broadcast(log_sum.neg())
}

// ================================================================== //
//  CrossEntropy — version stable (renommée canoniquement)             //
// ================================================================== //

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward<'t>(&self, logits: Var<'t>, target_one_hot: Var<'t>) -> Var<'t> {
        let (rows, _) = logits.shape();
        let n = rows as f32;
        let lsm = log_softmax(logits);
        let prod = target_one_hot.hadamard(lsm);
        prod.sum().scale(-1.0 / n)
    }
}

// Alias de compatibilité avec le code v6.1
pub type CrossEntropyLossStable = CrossEntropyLoss;

// ================================================================== //
//  BCE — Binary Cross-Entropy strict                                  //
// ================================================================== //

pub struct BceLoss;

impl Loss for BceLoss {
    fn forward<'t>(&self, p: Var<'t>, y: Var<'t>) -> Var<'t> {
        let (rows, cols) = p.shape();
        let n = (rows * cols) as f32;
        let log_p = p.clone().log();
        let tape = p.tape();
        let ones = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let one_minus_p = ones.sub(p.clone());
        let log_omp = one_minus_p.log();
        let term1 = y.clone().hadamard(log_p);
        let ones2 = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let one_minus_y = ones2.sub(y);
        let term2 = one_minus_y.hadamard(log_omp);
        term1.add(term2).sum().scale(-1.0 / n)
    }
}

// ================================================================== //
//  MAE — Mean Absolute Error (smooth L1)                              //
// ================================================================== //

pub struct MaeLoss { pub epsilon: f32 }
impl Default for MaeLoss { fn default() -> Self { Self { epsilon: 1e-6 } } }

impl Loss for MaeLoss {
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (rows, cols) = pred.shape();
        let n = (rows * cols) as f32;
        let diff = pred.sub(target);
        let sq = diff.clone().hadamard(diff);
        let tape = sq.tape();
        let eps_t = tape.input(
            Tensor::from_vec(vec![self.epsilon; rows * cols], rows, cols));
        sq.add(eps_t).sqrt().sum().scale(1.0 / n)
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sum_to_one() {
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 0.0, 0.0, 0.0, 0.0,
                 5.0, -1.0, 2.0, 3.0], 3, 4));
        let p = softmax(logits);
        let pt = tape.value(p.idx());
        for i in 0..3 {
            let s: f32 = pt.data[i*4..(i+1)*4].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sum = {s}");
        }
    }

    #[test]
    fn softmax_handles_large_logits() {
        // Sans max-trick, exp(1000) = inf → NaN partout
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![1000.0, 0.0, -500.0], 1, 3));
        let p = softmax(logits);
        let pt = tape.value(p.idx());
        assert!(pt.data.iter().all(|x| x.is_finite()));
        // La première classe domine
        assert!(pt.data[0] > 0.999);
    }

    #[test]
    fn log_softmax_stable_on_extremes() {
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![1000.0, 0.0, -500.0], 1, 3));
        let lsm = log_softmax(logits);
        let v = tape.value(lsm.idx());
        // Le premier devrait être ~0, les autres très négatifs mais finis
        assert!(v.data[0].abs() < 1e-3);
        assert!(v.data[1].is_finite());
        assert!(v.data[2].is_finite());
    }

    #[test]
    fn cross_entropy_correct_on_batch() {
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(
            vec![10.0, 0.0, 0.0,    // pred classe 0
                 10.0, 0.0, 0.0],   // pred classe 0
            2, 3));
        let target = tape.input(Tensor::from_vec(
            vec![1.0, 0.0, 0.0,    // truth classe 0 ✓
                 0.0, 1.0, 0.0],   // truth classe 1 ✗
            2, 3));
        let loss = CrossEntropyLoss.forward(logits, target);
        let val = tape.value(loss.idx()).data[0];
        // Loss moyenne ≈ (0 + 10) / 2 = 5
        assert!(val > 3.0 && val < 7.0, "got {val}");
    }

    #[test]
    fn alias_compatibility() {
        // CrossEntropyLossStable est un alias vers CrossEntropyLoss
        let _: CrossEntropyLossStable = CrossEntropyLoss;
    }
}
