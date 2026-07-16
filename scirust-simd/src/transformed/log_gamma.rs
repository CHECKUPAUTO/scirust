// scirust-simd/src/transformed/log_gamma.rs
//
// # Transformation « log-Gamma » φ(x) = ln Γ(x + 1)
//
// ## Définition & domaine
//
// `φ(x) = ln Γ(x+1)`, domaine latent `x > −1`. L'image est `[φ_min, ∞)` avec
// `φ_min = ln Γ(x*+1) ≈ −0.1215` atteint au point d'inflexion `x* ≈ 0.4616`.
// Sur les entiers, `φ(n) = ln(n!)`.
//
// ## Inversibilité — ATTENTION
//
// Comme [`super::reciprocal_gamma`], `φ` **n'est pas** globalement injective :
// `Γ(x+1)` étant minimale en `x*`, `ln Γ(x+1)` y est **minimale**, décroissante
// sur `(−1, x*]` puis croissante sur `[x*, ∞)`. Décodage **faillible**
// (`y < φ_min` ⇒ [`InverseError::OutOfRange`]) et **paramétré par branche**
// ([`GammaBranch`]).
//
// ## Propriétés numériques
//
// * Dérivée `φ'(x) = ψ(x+1)` (digamma) — plus stable que `ReciprocalGamma`
//   (pas de division par `Γ`).
// * `φ(x) → +∞` quand `x → −1⁺` et quand `x → +∞`.
// * Coût : un `ln Γ` (Lanczos) par encodage ; ~100 bissections par décodage.

use super::branch::{GAMMA_TURN_X, GammaBranch, invert_unimodal};
use super::special::{digamma, ln_gamma};
use super::transform::{DomainError, InverseError, ScalarTransform};

/// Transformation `φ(x) = ln Γ(x+1)` (domaine `x > −1`, **non** injective).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogGamma;

impl LogGamma {
    /// Borne inférieure de l'image, `φ_min = φ(x*) = ln Γ(x*+1) ≈ −0.1215`.
    #[must_use]
    pub fn min_value() -> f64 {
        ln_gamma(GAMMA_TURN_X + 1.0)
    }
}

impl ScalarTransform<f64> for LogGamma {
    type Branch = GammaBranch;
    const NAME: &'static str = "LogGamma";

    #[inline]
    fn is_globally_invertible() -> bool {
        false
    }

    fn encode(x: f64) -> Result<f64, DomainError> {
        if !x.is_finite()
        {
            return Err(DomainError::NotFinite { value: x });
        }
        if x <= -1.0
        {
            return Err(DomainError::BelowDomain {
                value: x,
                lower_bound: -1.0,
            });
        }
        Ok(ln_gamma(x + 1.0))
    }

    fn derivative(x: f64) -> Result<f64, DomainError> {
        if !x.is_finite()
        {
            return Err(DomainError::NotFinite { value: x });
        }
        if x <= -1.0
        {
            return Err(DomainError::BelowDomain {
                value: x,
                lower_bound: -1.0,
            });
        }
        // d/dx ln Γ(x+1) = ψ(x+1).
        Ok(digamma(x + 1.0))
    }

    fn decode(y: f64, branch: GammaBranch) -> Result<f64, InverseError> {
        if !y.is_finite()
        {
            return Err(InverseError::NotFinite { value: y });
        }
        if y < Self::min_value()
        {
            return Err(InverseError::OutOfRange { value: y });
        }
        invert_unimodal(|x| ln_gamma(x + 1.0), GAMMA_TURN_X, -1.0, y, branch)
            .ok_or(InverseError::NoSolutionInBranch)
    }
}
