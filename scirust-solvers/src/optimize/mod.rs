//! Optimisation sans contrainte : `min f(x)` pour `f: R^n → R`.
//!
//! - `gradient_descent` : descente de gradient avec recherche linéaire (backtracking)
//! - `bfgs`             : quasi-Newton BFGS, le standard pour problèmes lisses
//! - `nelder_mead`      : sans dérivée, robuste mais lent

pub mod bfgs;
pub mod gradient;
pub mod nelder_mead;
pub mod spg;

pub use bfgs::bfgs;
pub use gradient::gradient_descent;
pub use nelder_mead::nelder_mead;
pub use spg::spg;
