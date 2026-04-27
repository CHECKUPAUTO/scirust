// scirust-core/src/nn/loss/strict_v6_1.rs
//
// Mise à jour des fonctions strictes en exploitant Op::MaxAxis.
//
// La nouveauté centrale : log_softmax STABLE.
//
//   log_softmax(x) = x - logsumexp(x)
//                  = (x - max(x)) - log(sum(exp(x - max(x))))
//
// La soustraction du max ligne-par-ligne empêche l'overflow de exp(x)
// quand les logits sont grands (cas typique des dernières couches d'un
// CNN sans BatchNorm en sortie).

use crate::autodiff::reverse::{Var};
use crate::nn::loss::Loss;

/// log_softmax stable via max-trick.
/// logits : (N, C) → log_softmax : (N, C)
pub fn log_softmax_stable<'t>(logits: Var<'t>) -> Var<'t> {
    // 1. max par ligne : (N, 1)
    let max_per_row = logits.clone().max_axis(1);
    // 2. soustrait pour stabiliser : (N, C)
    let neg_max = max_per_row.neg();
    let shifted = logits.add_broadcast(neg_max);   // shifted_max = 0 par ligne
    // 3. log-sum-exp sur shifted
    let exp_shifted = shifted.clone().exp();
    let sum_per_row = exp_shifted.sum_axis(1);     // (N, 1)
    let log_sum     = sum_per_row.log();           // (N, 1)
    // 4. shifted - log_sum (broadcast)
    shifted.add_broadcast(log_sum.neg())
}

/// softmax stable.
pub fn softmax_stable<'t>(logits: Var<'t>) -> Var<'t> {
    let max_per_row = logits.clone().max_axis(1);
    let neg_max = max_per_row.neg();
    let shifted = logits.add_broadcast(neg_max);
    let exp_s = shifted.exp();
    let sum_s = exp_s.clone().sum_axis(1);
    let inv = sum_s.reciprocal();
    exp_s.mul_broadcast(inv)
}

/// CrossEntropy stable utilisant log_softmax_stable.
/// target_one_hot : (N, C)
pub struct CrossEntropyLossStable;

impl Loss for CrossEntropyLossStable {
    fn forward<'t>(&self, logits: Var<'t>, target_one_hot: Var<'t>) -> Var<'t> {
        let (rows, _) = logits.shape();
        let n = rows as f32;
        let lsm = log_softmax_stable(logits);
        let prod = target_one_hot.hadamard(lsm);
        prod.sum().scale(-1.0 / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};

    #[test]
    fn log_softmax_stable_handles_large_logits() {
        // Sans max-trick : exp(1000) → +inf, NaN partout.
        // Avec : on soustrait max, logits relatifs deviennent {0, -1000, -1500}
        // → exp = {1, ~0, ~0}, somme = 1, log = 0
        // → log_softmax = {0, -1000, -1500}
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![1000.0, 0.0, -500.0], 1, 3));
        let lsm = log_softmax_stable(logits);
        let v = tape.value(lsm.idx());
        // Le premier doit être très proche de 0
        assert!(v.data[0].abs() < 1e-3, "got {}", v.data[0]);
        // Les autres doivent être finis
        assert!(v.data[1].is_finite());
        assert!(v.data[2].is_finite());
    }

    #[test]
    fn softmax_stable_sums_to_one_per_row() {
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(
            vec![100.0, 200.0, 300.0,
                 -50.0, 0.0, 50.0], 2, 3));
        let p = softmax_stable(logits);
        let pt = tape.value(p.idx());
        for i in 0..2 {
            let s: f32 = pt.data[i*3..(i+1)*3].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
        }
        // Pour la première ligne, le 3e (300) domine massivement
        assert!(pt.data[2] > 0.999);
    }

    #[test]
    fn cross_entropy_stable_matches_naive_on_normal_inputs() {
        // Pour des logits modérés, stable et naïf donnent le même résultat
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0, 0.0, 1.0], 1, 3));
        let loss = CrossEntropyLossStable.forward(logits, target);
        let val = tape.value(loss.idx()).data[0];
        // Vérification analytique : softmax(1,2,3) ≈ (0.09, 0.245, 0.665)
        // log(0.665) ≈ -0.408
        // CE = -log(0.665) ≈ 0.408
        assert!((val - 0.408).abs() < 0.01, "got {val}");
    }
}
