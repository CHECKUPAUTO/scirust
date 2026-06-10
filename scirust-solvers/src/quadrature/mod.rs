//! Intégration numérique — quadrature de fonctions `f: R → R` ou `R → R^n`.
//!
//! - `simpson_adaptive` : Simpson avec subdivision adaptative récursive ;
//!   bon défaut universel, gère les changements de courbure.
//! - `gauss_legendre`   : ordre fixe (5, 10, 20) ; précision spectaculaire sur
//!   les fonctions lisses analytiques.
//! - `romberg`          : extrapolation de Richardson sur la règle du trapèze ;
//!   converge rapidement sur fonctions très lisses.

pub mod gauss;
pub mod romberg;
pub mod simpson;

pub use gauss::{GaussOrder, gauss_legendre};
pub use romberg::romberg;
pub use simpson::simpson_adaptive;
