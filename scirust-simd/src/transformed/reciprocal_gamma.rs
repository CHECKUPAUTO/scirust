// scirust-simd/src/transformed/reciprocal_gamma.rs
//
// # Transformation « Gamma réciproque » φ(x) = 1 / Γ(x + 1)
//
// ## Définition & domaine
//
// `φ(x) = 1/Γ(x+1)`, domaine latent `x > −1` (⇔ `z = x+1 > 0`, où `Γ` est
// finie et strictement positive). L'image est `(0, φ_max]` avec
// `φ_max = 1/Γ(x*+1) ≈ 1.1292` atteint au point d'inflexion `x* ≈ 0.4616`.
//
// ## Inversibilité — ATTENTION
//
// `φ` **n'est pas** globalement injective. `Γ(x+1)` est minimale en `x*`, donc
// `φ` y est **maximale** : elle croît sur `(−1, x*]` puis décroît sur `[x*, ∞)`.
// Un même encodé `y ∈ (0, φ_max)` possède donc **deux** antécédents. Le décodage
// est par conséquent :
// * **faillible** (`y ∉ (0, φ_max]` ⇒ [`InverseError::OutOfRange`]) ;
// * **paramétré par branche** ([`GammaBranch`]) — l'ambiguïté n'est jamais
//   masquée par un inverse arbitraire.
//
// ## Propriétés numériques
//
// * `φ` est `C^∞` sur `(−1, ∞)`, dérivée `φ'(x) = −ψ(x+1)/Γ(x+1)`.
// * `φ(x) → 0⁺` quand `x → −1⁺` (car `Γ(0⁺) → +∞`) et quand `x → +∞`.
// * Coût : un `Γ` (Lanczos) par encodage ; ~100 bissections + un `Γ` chacune par
//   décodage. Aucun `unsafe`, aucune allocation.

use super::branch::{GAMMA_TURN_X, GammaBranch, invert_unimodal};
use super::special::{digamma, gamma};
use super::transform::{DomainError, InverseError, ScalarTransform};

/// Transformation `φ(x) = 1/Γ(x+1)` (domaine `x > −1`, **non** injective).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReciprocalGamma;

impl ReciprocalGamma {
    /// Borne supérieure de l'image, `φ_max = φ(x*) = 1/Γ(x*+1) ≈ 1.1292`.
    #[must_use]
    pub fn max_value() -> f64 {
        1.0 / gamma(GAMMA_TURN_X + 1.0)
    }
}

impl ScalarTransform<f64> for ReciprocalGamma {
    type Branch = GammaBranch;
    const NAME: &'static str = "ReciprocalGamma";

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
        Ok(1.0 / gamma(x + 1.0))
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
        // d/dx 1/Γ(x+1) = −Γ'(x+1)/Γ(x+1)² = −ψ(x+1)/Γ(x+1).
        let z = x + 1.0;
        Ok(-digamma(z) / gamma(z))
    }

    fn decode(y: f64, branch: GammaBranch) -> Result<f64, InverseError> {
        if !y.is_finite()
        {
            return Err(InverseError::NotFinite { value: y });
        }
        if y <= 0.0 || y > Self::max_value()
        {
            return Err(InverseError::OutOfRange { value: y });
        }
        invert_unimodal(|x| 1.0 / gamma(x + 1.0), GAMMA_TURN_X, -1.0, y, branch)
            .ok_or(InverseError::NoSolutionInBranch)
    }
}
