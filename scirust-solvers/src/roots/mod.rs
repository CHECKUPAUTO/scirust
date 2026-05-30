//! Recherche de racines de fonctions scalaires `f: R → R`.
//!
//! - `bisection` : sûre, lente, demande un intervalle `[a, b]` avec changement de signe.
//! - `secant`    : rapide (super-linéaire) mais peut diverger, demande deux estimations.
//! - `newton`    : très rapide (quadratique) ; calcule `f'` automatiquement via autodiff.
//! - `brent`     : robustesse de la bissection + vitesse de la sécante. **Choix par défaut**.

pub mod bisection;
pub mod brent;
pub mod newton;
pub mod secant;

pub use bisection::bisection;
pub use brent::brent;
pub use newton::{newton, newton_with_derivative};
pub use secant::secant;
