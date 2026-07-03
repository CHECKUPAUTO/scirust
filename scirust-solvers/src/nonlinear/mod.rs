//! Systèmes non-linéaires F: R^n → R^n. On cherche `x` tel que `F(x) = 0`.
//!
//! - `newton_system` : Newton-Raphson avec jacobienne calculée par autodiff
//!   (passe N appels au callback en mode forward Dual)
//! - `broyden`       : quasi-Newton qui met à jour la jacobienne sans la
//!   recalculer — utile quand l'évaluation de F est coûteuse

pub mod anderson;
pub mod broyden;
pub mod newton;

pub use anderson::anderson_accelerate;
pub use broyden::broyden;
pub use newton::{newton_system, newton_system_jac};
