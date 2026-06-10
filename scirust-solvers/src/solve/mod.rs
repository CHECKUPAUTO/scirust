//! API unifiée de résolution d'équations.
//!
//! L'idée : tu décris ton équation, on choisit le bon solveur.
//!
//! - Équation symbolique linéaire/quadratique → racines exactes via
//!   `scirust-symbolic`
//! - Équation symbolique polynomiale degré ≥ 3 → coefficients extraits →
//!   `polynomial::roots`
//! - Équation symbolique générale ou closure Rust → Brent ou Newton

pub mod symbolic_bridge;
pub mod unified;

pub use symbolic_bridge::{expr_to_closure, extract_polynomial_coeffs};
pub use unified::{SolveResult, solve, solve_in_interval};
