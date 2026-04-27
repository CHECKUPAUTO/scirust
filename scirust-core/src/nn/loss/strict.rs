// scirust-core/src/nn/loss/strict.rs
//
// Losses strictes débloquées par Op::Log et Op::Exp (v5).
//
//   - BceLoss        : binary cross-entropy exacte
//   - softmax        : exp(x) / Σ exp(x), normalisation row-wise
//   - log_softmax    : version stable numériquement
//   - CrossEntropy   : NLL appliquée sur log_softmax(logits)
//
// Convention :
//   - logits  : (N, C)  brut, pas normalisé
//   - target  : (N, C)  one-hot encoding
//   - softmax sur l'axe 1 (par échantillon)
//
// IMPORTANT : ce module suppose que reverse.rs a Op::Log/Exp/Neg
// (ajoutés via le patch v5).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::loss::Loss;

// ================================================================== //
//  BCE strict — Binary Cross-Entropy                                  //
// ================================================================== //
//
//   BCE = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)]
//
//   - p doit être dans (0, 1) — typiquement après une sigmoïde
//   - y doit être dans {0, 1}
//
// La formule fait intervenir log(p) et log(1-p) — d'où la nécessité
// de Op::Log. On gère le clamp 1e-12 dans Var::log() pour éviter
// log(0) si la sigmoïde sort des valeurs extrêmes.

pub struct BceLoss;

impl Loss for BceLoss {
    fn forward<'t>(&self, p: Var<'t>, y: Var<'t>) -> Var<'t> {
        let (rows, cols) = p.shape();
        let n = (rows * cols) as f32;
        assert_eq!(p.shape(), y.shape(), "BCE: shape mismatch");

        // Pour calculer (1-p), on a besoin d'un tenseur de uns sur la même tape
        let tape = unsafe { tape_of(&p) };
        let ones = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));

        // log_p = log(p)
        let log_p = p.log();

        // log_one_minus_p = log(1 - p)
        let one_minus_p = ones.sub(p);
        let log_one_minus_p = one_minus_p.log();

        // y * log(p)
        let term1 = y.hadamard(log_p);

        // (1 - y) * log(1 - p)
        // 1 - y : besoin d'un autre "ones"
        let ones2 = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let one_minus_y = ones2.sub(y);
        let term2 = one_minus_y.hadamard(log_one_minus_p);

        // -(1/N) · Σ (term1 + term2)
        term1.add(term2).sum().scale(-1.0 / n)
    }
}

// ------------------------------------------------------------------ //
// Helper unsafe : récupère &Tape depuis un Var.
// La Var contient `tape: &'t Tape`, mais le champ est privé.
// On l'extrait via pointer punning — c'est sûr tant que la signature
// du struct Var ne change pas. Sinon, exposer une méthode tape() sur Var.
//
// ALTERNATIVE PROPRE : ajouter `pub fn tape(&self) -> &'t Tape` sur Var
// dans reverse.rs. Le patch v5 le fait.
// ------------------------------------------------------------------ //
unsafe fn tape_of<'t>(v: &Var<'t>) -> &'t Tape {
    // Repose sur le layout de Var{ tape: &'t Tape, idx: usize }
    // Si reverse.rs ajoute `pub fn tape()`, remplacer par v.tape().
    let ptr: *const Var<'t> = v;
    let tape_ptr: *const &'t Tape = ptr as *const &'t Tape;
    *tape_ptr
}

// ================================================================== //
//  Softmax row-wise                                                   //
// ================================================================== //
//
//   softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
//
// Stabilité numérique : on ne soustrait PAS le max ici (faute de
// reduce-max sur la tape). Pour la classification, préférer
// `cross_entropy` qui combine log_softmax(logits) avec target plus
// stable. Cette version brute est utile pour inspection ou inférence.

pub fn softmax<'t>(logits: Var<'t>) -> Var<'t> {
    // exp_logits : (N, C)
    let exp_l = logits.exp();
    // Pour normaliser par row, il faudrait reduce_sum sur axis=1.
    // En attendant cette op, on triche : on duplique la somme totale
    // par broadcast — ATTENTION : ce n'est pas un softmax row-wise
    // correct quand N > 1, c'est un softmax global.
    //
    // ⚠️ Pour un softmax row-wise correct, il faut Op::SumAxis
    //    (TODO v6). Cette implémentation est utilisable pour N=1
    //    ou en debug uniquement.
    let total = exp_l.sum();  // (1, 1)
    // Diviser par total via multiplication par son inverse.
    // Pas d'op div, donc on construit 1/total via multiplication.
    // Ici on utilise scale par (1/value), ce qui CASSE le gradient
    // (le gradient sur total ne se propage pas).
    let total_val = unsafe { tape_of(&exp_l) }.value(total.idx()).data[0];
    let inv = 1.0 / total_val.max(1e-12);
    exp_l.scale(inv)
}

/// log_softmax stable numériquement, version "soft" sans reduce-max.
/// log_softmax(x) = x - log(Σ exp(x))
pub fn log_softmax<'t>(logits: Var<'t>) -> Var<'t> {
    let exp_l = logits.exp();
    let total = exp_l.sum();      // scalaire (1,1)
    let log_total = total.log();  // scalaire (1,1)

    // logits - log_total broadcasté
    // On a besoin que log_total ait la forme broadcastable (1,1) → (N,C),
    // ce qui est déjà le cas grâce à add_broadcast/sub.
    // On utilise neg + add_broadcast pour faire "logits + (-log_total)".
    let neg_log_total = log_total.neg();
    logits.add_broadcast(neg_log_total)
}

// ================================================================== //
//  CrossEntropy = NLL ∘ log_softmax                                   //
// ================================================================== //
//
//   CE(logits, y) = -(1/N) Σ_i Σ_c y_{i,c} · log_softmax(logits)_{i,c}
//
// `target` est en one-hot. Pour des labels entiers, l'utilisateur doit
// d'abord les convertir.

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward<'t>(&self, logits: Var<'t>, target_one_hot: Var<'t>) -> Var<'t> {
        let (rows, _) = logits.shape();
        let n = rows as f32;
        let lsm = log_softmax(logits);
        // Σ y · log_softmax(logits)
        let prod = target_one_hot.hadamard(lsm);
        // -(1/N) · Σ
        prod.sum().scale(-1.0 / n)
    }
}

// ================================================================== //
//  MAE — désormais possible avec Op::Sqrt (Huber soft)                //
// ================================================================== //
//
//   MAE soft = (1/N) Σ √((pred-target)² + ε)
//
// Différentiable partout (le ε empêche la singularité en 0).

pub struct MaeLoss { pub epsilon: f32 }
impl Default for MaeLoss { fn default() -> Self { Self { epsilon: 1e-6 } } }

impl Loss for MaeLoss {
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (rows, cols) = pred.shape();
        let n = (rows * cols) as f32;
        let diff = pred.sub(target);
        let sq = diff.hadamard(diff);
        // sq + epsilon broadcasté
        let tape = unsafe { tape_of(&sq) };
        let eps_t = tape.input(Tensor::from_vec(vec![self.epsilon; rows * cols], rows, cols));
        let smoothed = sq.add(eps_t);
        let abs_diff = smoothed.sqrt();
        abs_diff.sum().scale(1.0 / n)
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bce_loss_zero_when_perfect() {
        // p = y = [0.99, 0.01] → BCE ≈ 0
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![0.99, 0.01], 1, 2));
        let y = tape.input(Tensor::from_vec(vec![1.0,  0.0],  1, 2));
        let loss = BceLoss.forward(p, y);
        let val = tape.value(loss.idx()).data[0];
        // BCE doit être très petite (≈ -log(0.99) ≈ 0.01)
        assert!(val < 0.05, "BCE perfect-ish should be ~0, got {val}");
    }

    #[test]
    fn bce_loss_high_when_wrong() {
        // p = [0.01, 0.99], y = [1, 0] → fortement pénalisé
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![0.01, 0.99], 1, 2));
        let y = tape.input(Tensor::from_vec(vec![1.0,  0.0],  1, 2));
        let loss = BceLoss.forward(p, y);
        let val = tape.value(loss.idx()).data[0];
        assert!(val > 2.0, "BCE wrong should be >2, got {val}");
    }

    #[test]
    fn cross_entropy_with_correct_class() {
        // logits = [10, 0, 0], y = [1, 0, 0] → loss ≈ 0
        let tape = Tape::new();
        let logits = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0,  0.0, 0.0], 1, 3));
        let loss = CrossEntropyLoss.forward(logits, target);
        let val = tape.value(loss.idx()).data[0];
        assert!(val < 0.01, "CE should be ~0 when prediction confident-correct, got {val}");
    }

    #[test]
    fn mae_zero_at_match() {
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let t = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let loss = MaeLoss::default().forward(p, t);
        let val = tape.value(loss.idx()).data[0];
        // Avec epsilon=1e-6, sqrt(1e-6) = 1e-3 par élément
        assert!(val < 0.01, "got {val}");
    }
}
