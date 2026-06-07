// scirust-core/src/nn/loss/mod.rs
//
// Fonctions de perte (loss).
//
// Une loss prend une prédiction (Var) et une cible (Var ou Tensor encodé
// comme Var input) et retourne un scalaire (Var de shape (1, 1)) sur lequel
// on appelle backward().
//
// Toutes les loss sont sans paramètres entraînables — elles ne sont pas
// des Module. Elles exposent simplement une méthode forward.

pub mod cross_entropy;
pub mod mse;
pub mod nll;
pub mod pinn;

pub use cross_entropy::CrossEntropyLoss;
pub use mse::MseLoss;
pub use nll::NllLoss;
pub use pinn::PinnLossEvaluator;

use crate::autodiff::reverse::{Tape, Var};

/// Trait commun pour toutes les fonctions de perte.
///
/// Convention :
///   - prediction est ce que produit le modèle (logits ou activations)
///   - target est la vraie réponse (one-hot pour classif, valeurs pour régression)
///   - retour : Var scalaire (shape (1, 1)) sur lequel appeler backward()
pub trait Loss {
    fn forward<'t>(&self, tape: &'t Tape, pred: Var<'t>, target: Var<'t>) -> Var<'t>;
}
