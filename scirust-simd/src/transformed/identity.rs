// scirust-simd/src/transformed/identity.rs
//
// # Transformation identité φ(x) = x
//
// Le cas de contrôle : encode et décode sont l'identité, la dérivée vaut `1`, la
// transformation est globalement inversible. `Quaternion<TransformedScalar<T,
// Identity>>` reproduit **exactement** `Quaternion<T>` (défaut de transformation
// nul, Modèle A ≡ Modèle B), ce qui sert de test de non-régression du cadre.

use crate::fixed::NumericScalar;

use super::transform::{DomainError, InverseError, ScalarTransform};

/// Transformation identité `φ(x) = x` (domaine `ℝ`, globalement inversible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Identity;

impl<T: NumericScalar> ScalarTransform<T> for Identity {
    type Branch = ();
    const NAME: &'static str = "Identity";

    #[inline]
    fn is_globally_invertible() -> bool {
        true
    }

    #[inline]
    fn encode(x: T) -> Result<T, DomainError> {
        Ok(x)
    }

    #[inline]
    fn derivative(_x: T) -> Result<T, DomainError> {
        Ok(T::one())
    }

    #[inline]
    fn decode(y: T, _branch: ()) -> Result<T, InverseError> {
        Ok(y)
    }
}
