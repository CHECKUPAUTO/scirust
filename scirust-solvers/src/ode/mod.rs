//! Solveurs d'équations différentielles ordinaires (EDO).
//!
//! Forme : `dy/dt = f(t, y)` avec y ∈ R^n.
//!
//! - `rk4`     : Runge-Kutta classique d'ordre 4, pas fixe. Simple et robuste.
//! - `dopri5`  : Dormand-Prince 5(4), pas adaptatif avec estimation d'erreur
//!              embarquée. Choix par défaut pour les problèmes non-raides.

pub mod dopri5;
pub mod rk4;

pub use dopri5::{OdeOutput, dopri5};
pub use rk4::rk4_fixed;
