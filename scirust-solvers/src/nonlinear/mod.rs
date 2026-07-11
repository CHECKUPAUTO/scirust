//! Systèmes non-linéaires F: R^n → R^n. On cherche `x` tel que `F(x) = 0`.
//!
//! - `newton_system` : Newton-Raphson avec jacobienne calculée par autodiff
//!   (passe N appels au callback en mode forward Dual)
//! - `broyden`       : quasi-Newton qui met à jour la jacobienne sans la
//!   recalculer — utile quand l'évaluation de F est coûteuse
//! - `levenberg_marquardt` : moindres carrés non linéaires pour un résidu
//!   R^n → R^m (m pas forcément égal à n) — généralise le cas au système
//!   carré exact que résout `newton_system`

pub mod anderson;
pub mod broyden;
pub mod levenberg_marquardt;
pub mod newton;

pub use anderson::anderson_accelerate;
pub use broyden::broyden;
pub use levenberg_marquardt::levenberg_marquardt;
pub use newton::{newton_system, newton_system_jac};
