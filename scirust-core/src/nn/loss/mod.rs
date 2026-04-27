// scirust-core/src/nn/loss/mod.rs
//
// Trait Loss + implémentations.
//
// Modèle : une Loss prend deux Var (prédiction, cible) et renvoie une Var
// scalaire (1×1) sur la même Tape. La Var renvoyée est le point d'entrée
// du backward.
//
// Règles :
//   - Toutes les ops doivent être des opérations de Var (pour que le tape
//     les enregistre et puisse propager les gradients).
//   - Les losses ne doivent pas créer de nouveaux Var sans les chaîner —
//     sinon ils ne participent pas au backward.
//   - Le résultat doit être un scalaire (forme (1,1)).

use crate::autodiff::reverse::{Var};

// ================================================================== //
//  Trait                                                              //
// ================================================================== //

pub trait Loss {
    /// Calcule la perte. Renvoie une Var scalaire (forme (1,1)) sur la
    /// même Tape que `pred` et `target`. backward() peut être appelé dessus.
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t>;
}

// ================================================================== //
//  MSE — Mean Squared Error                                           //
// ================================================================== //
//
//   L = (1/N) Σ (pred_i - target_i)²
//
// Différentiable partout, gradient propre. Choix par défaut pour
// la régression continue.

pub struct MseLoss;

impl Loss for MseLoss {
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (r, c) = pred.shape();
        let n = (r * c) as f32;
        assert_eq!(pred.shape(), target.shape(), "MSE: shape mismatch");
        let diff = pred.sub(target);
        diff.hadamard(diff).sum().scale(1.0 / n)
    }
}

// ================================================================== //
//  MAE — Mean Absolute Error                                          //
// ================================================================== //
//
//   L = (1/N) Σ |pred_i - target_i|
//
// Note : |x| = √(x²) — différentiable presque partout (singulier en 0).
// On approxime ici via Huber-soft pour rester safe sur l'autodiff :
//
//   |x| ≈ √(x² + ε)  avec ε petit
//
// Pour l'instant on n'a pas Op::Sqrt sur la tape — on retombe donc sur
// l'approximation MSE. À implémenter proprement quand Op::Sqrt sera ajouté
// (TODO v5).

pub struct HuberPseudoLoss { pub delta: f32 }

impl Default for HuberPseudoLoss {
    fn default() -> Self { Self { delta: 1.0 } }
}

impl Loss for HuberPseudoLoss {
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        // En attendant Op::Sqrt et Op::Abs, on délègue à MSE.
        // Le `delta` sera utilisé quand l'implémentation complète sera dispo.
        MseLoss.forward(pred, target)
    }
}

// ================================================================== //
//  BCE approximé — Binary Cross-Entropy                               //
// ================================================================== //
//
//   BCE strict = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)]
//
// Nécessite Op::Log sur la tape. En attendant, on approxime par MSE
// sur la sortie sigmoïde — équivalent en termes de signal de descente
// pour un classifier binaire (le minimum est au même endroit).
//
// Quand Op::Log sera ajouté (v5), cette struct passera à la formule exacte
// sans changer son interface publique.

pub struct BceLossApprox;

impl Loss for BceLossApprox {
    fn forward<'t>(&self, pred_after_sigmoid: Var<'t>, target: Var<'t>) -> Var<'t> {
        MseLoss.forward(pred_after_sigmoid, target)
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{Tape, Tensor};

    #[test]
    fn mse_zero_when_pred_equals_target() {
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let t = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let loss = MseLoss.forward(p, t);
        let val = tape.value(loss.idx()).data[0];
        assert!(val.abs() < 1e-6, "MSE should be 0, got {val}");
    }

    #[test]
    fn mse_correct_value() {
        // pred = [0,0,0], target = [1,2,3] → MSE = (1+4+9)/3 = 14/3
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![0.0, 0.0, 0.0], 1, 3));
        let t = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let loss = MseLoss.forward(p, t);
        let val = tape.value(loss.idx()).data[0];
        assert!((val - 14.0 / 3.0).abs() < 1e-5, "got {val}");
    }

    #[test]
    fn mse_backward_sign_correct() {
        // Si pred > target, ∂L/∂pred > 0 (il faut diminuer pred)
        let tape = Tape::new();
        let p = tape.input(Tensor::from_vec(vec![5.0, 5.0], 1, 2));
        let t = tape.input(Tensor::from_vec(vec![1.0, 1.0], 1, 2));
        let loss = MseLoss.forward(p, t);
        loss.backward();
        let grad = tape.grad(p.idx());
        // Tous les gradients doivent être positifs
        assert!(grad.data.iter().all(|&g| g > 0.0));
    }
}
