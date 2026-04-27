// scirust-core/src/nn/loss/strict_v6.rs
//
// Refactor de strict.rs en utilisant les nouvelles ops Op::SumAxis,
// Op::Reciprocal et Op::Reshape (v6).
//
// CHANGEMENTS PAR RAPPORT À v5 :
//   - softmax devient correctement row-wise pour batch > 1
//   - log_softmax stable (avec sub max-trick à venir en v6.1)
//   - CrossEntropyLoss utilise sum_axis au lieu d'un faux broadcast scalaire
//   - Plus de unsafe tape_of(...) — on prend &Tape en paramètre quand nécessaire
//
// Le module remplace l'implémentation v5 de strict.rs après application
// du patch sur reverse.rs (qui ajoute SumAxis/Reshape/Reciprocal).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::loss::Loss;

// ================================================================== //
//  Softmax row-wise correct                                           //
// ================================================================== //
//
//   softmax(x)[i,j] = exp(x[i,j]) / Σ_k exp(x[i,k])
//
// Avec sum_axis(axis=1, keep_dims=true) on obtient un (N,1) qu'on peut
// broadcast diviser dans (N,C). C'est désormais correct pour tout batch.

pub fn softmax<'t>(logits: Var<'t>) -> Var<'t> {
    let exp_l = logits.exp();
    // (N, C) → (N, 1)
    let sum_per_row = exp_l.clone().sum_axis(1);
    // 1 / sum, puis hadamard via broadcast
    let inv_sum = sum_per_row.reciprocal();
    // exp_l : (N, C), inv_sum : (N, 1) → broadcast lors du hadamard via
    // mul_broadcast (nécessite que mul_broadcast existe ; sinon on
    // élargit inv_sum manuellement).
    exp_l.mul_broadcast(inv_sum)
}

/// log_softmax stable.
///   log_softmax(x)[i,j] = x[i,j] - log(Σ_k exp(x[i,k]))
///
/// La forme stable soustrait le max ligne-par-ligne avant l'exp pour
/// éviter les overflows. En v6 on n'a pas encore Op::MaxAxis, donc cette
/// version fait l'opération naïve. Pour des logits bornés (post-couche
/// linéaire bien initialisée), ça suffit. v6.1 ajoutera MaxAxis.
pub fn log_softmax<'t>(logits: Var<'t>) -> Var<'t> {
    let exp_l = logits.clone().exp();
    let sum_per_row = exp_l.sum_axis(1);    // (N, 1)
    let log_sum = sum_per_row.log();        // (N, 1)
    // logits - log_sum broadcasté
    let neg_log = log_sum.neg();
    logits.add_broadcast(neg_log)
}

// ================================================================== //
//  CrossEntropy correcte sur batch                                    //
// ================================================================== //
//
//   CE = -(1/N) Σ_i Σ_c y[i,c] · log_softmax(logits)[i,c]
//
// La sommation par lignes est correcte grâce à log_softmax row-wise.

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

// ================================================================== //
//  BCE stricte (inchangée vs v5, juste réécrite proprement)           //
// ================================================================== //

pub struct BceLoss;

impl Loss for BceLoss {
    fn forward<'t>(&self, p: Var<'t>, y: Var<'t>) -> Var<'t> {
        let (rows, cols) = p.shape();
        let n = (rows * cols) as f32;
        // log(p) et log(1-p)
        let log_p = p.clone().log();
        // 1 - p via sub d'un tensor de uns. On a besoin d'un Tape ;
        // on le récupère depuis p via la méthode publique ajoutée par
        // le patch v5 : Var::tape().
        let tape = p.tape();
        let ones = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let one_minus_p = ones.sub(p.clone());
        let log_omp = one_minus_p.log();
        let term1 = y.clone().hadamard(log_p);
        // 1 - y
        let ones2 = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let one_minus_y = ones2.sub(y);
        let term2 = one_minus_y.hadamard(log_omp);
        term1.add(term2).sum().scale(-1.0 / n)
    }
}

// ================================================================== //
//  MAE soft (inchangée vs v5)                                         //
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
        let eps_t = tape.input(Tensor::from_vec(vec![self.epsilon; rows * cols], rows, cols));
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
    fn softmax_sums_to_one_per_row() {
        // Pour un batch de 3 lignes × 4 classes, chaque ligne doit sommer à 1
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 0.0, 0.0, 0.0, 0.0,
                 5.0, -1.0, 2.0, 3.0], 3, 4));
        let p = softmax(logits);
        let pt = tape.value(p.idx());
        for i in 0..3 {
            let row_sum: f32 = pt.data[i*4..(i+1)*4].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5,
                    "row {i} sums to {row_sum}");
        }
    }

    #[test]
    fn softmax_uniform_input_uniform_output() {
        // logits égaux → probas égales = 1/C
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![3.0; 5], 1, 5));
        let p = softmax(logits);
        let pt = tape.value(p.idx());
        for x in &pt.data {
            assert!((x - 0.2).abs() < 1e-5);
        }
    }

    #[test]
    fn cross_entropy_correct_batch() {
        // Batch de 2 : la première ligne est confiante-correcte (loss bas),
        // la deuxième est confiante-fausse (loss élevé)
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
        assert!(val > 3.0 && val < 7.0, "loss = {val}");
    }

    #[test]
    fn softmax_grad_propagates() {
        // y = sum(softmax(x)). dy/dx[i] = softmax(x)[i] * (1 - 1) = 0 ?
        // En fait : sum(softmax) est constante = 1, donc dy/dx = 0 partout.
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let y = softmax(x).sum();
        y.backward();
        let g = tape.grad(x.idx());
        // Tous les gradients devraient être ~0 (au bruit numérique près)
        assert!(g.data.iter().all(|&v| v.abs() < 1e-4),
                "gradients should be ~0, got {:?}", g.data);
    }
}
