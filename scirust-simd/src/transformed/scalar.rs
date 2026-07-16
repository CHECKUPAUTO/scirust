// scirust-simd/src/transformed/scalar.rs
//
// # `TransformedScalar<T, F>` — un scalaire à représentation transformée
//
// Enveloppe un scalaire dont la valeur **latente** est faisant autorité ; la
// valeur **encodée** `φ(latent)` est calculée à la demande. Le paramètre `F`
// (marqueur de type, jamais stocké) fixe la transformation.
//
// L'arithmétique ([`NumericScalar`]) opère sur la valeur **latente** : ainsi
// `Hypercomplex<TransformedScalar<T, F>, N>` calcule l'algèbre en coordonnées
// latentes (base du « Modèle A », cf. [`super::hypercomplex`]). L'encodage est
// une opération explicite ([`TransformedScalar::encoded`]), jamais implicite.
//
// API (distinction nette latent ↔ encodé) :
// * [`from_latent`](TransformedScalar::from_latent) — construit depuis le latent.
// * [`latent`](TransformedScalar::latent) — lit le latent (autoritatif).
// * [`encoded`](TransformedScalar::encoded) — calcule `φ(latent)` (faillible).
// * [`try_from_encoded`](TransformedScalar::try_from_encoded) — décode `φ⁻¹(y)`
//   sur une branche (faillible, ambiguïté explicite).

use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use crate::fixed::NumericScalar;

use super::transform::{DomainError, InverseError, ScalarTransform};

/// Scalaire à représentation transformée : latent autoritatif, encodé calculé.
pub struct TransformedScalar<T, F> {
    latent: T,
    _marker: PhantomData<F>,
}

// Impls manuelles : `F` n'est qu'un marqueur, on n'exige donc jamais de borne
// dessus (contrairement à ce que produiraient les `derive`).
impl<T: Clone, F> Clone for TransformedScalar<T, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            latent: self.latent.clone(),
            _marker: PhantomData,
        }
    }
}
impl<T: Copy, F> Copy for TransformedScalar<T, F> {}
impl<T: PartialEq, F> PartialEq for TransformedScalar<T, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.latent == other.latent
    }
}
impl<T: PartialOrd, F> PartialOrd for TransformedScalar<T, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.latent.partial_cmp(&other.latent)
    }
}
impl<T: core::fmt::Debug, F> core::fmt::Debug for TransformedScalar<T, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TransformedScalar")
            .field("latent", &self.latent)
            .finish()
    }
}

impl<T, F> TransformedScalar<T, F> {
    /// Construit depuis la valeur latente (autoritative).
    #[inline]
    pub const fn from_latent(latent: T) -> Self {
        Self {
            latent,
            _marker: PhantomData,
        }
    }
}

impl<T: Copy, F> TransformedScalar<T, F> {
    /// Lit la valeur latente (autoritative).
    #[inline]
    pub fn latent(self) -> T {
        self.latent
    }
}

impl<T: Copy, F: ScalarTransform<T>> TransformedScalar<T, F> {
    /// Calcule la valeur encodée `φ(latent)` (faillible : domaine).
    #[inline]
    pub fn encoded(self) -> Result<T, DomainError> {
        F::encode(self.latent)
    }

    /// `φ'(latent)` (faillible : domaine).
    #[inline]
    pub fn derivative(self) -> Result<T, DomainError> {
        F::derivative(self.latent)
    }

    /// Décode `φ⁻¹(encoded)` sur `branch` et construit le scalaire (faillible).
    #[inline]
    pub fn try_from_encoded(encoded: T, branch: F::Branch) -> Result<Self, InverseError> {
        F::decode(encoded, branch).map(Self::from_latent)
    }

    /// Décode sur la branche principale (`Branch::default()`).
    #[inline]
    pub fn try_from_encoded_principal(encoded: T) -> Result<Self, InverseError> {
        F::decode_principal(encoded).map(Self::from_latent)
    }
}

// ------------------------------------------------------------------ //
//  Arithmétique : opère sur le LATENT (Modèle A dans l'algèbre)        //
// ------------------------------------------------------------------ //

impl<T: Add<Output = T>, F> Add for TransformedScalar<T, F> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self::from_latent(self.latent + r.latent)
    }
}
impl<T: Sub<Output = T>, F> Sub for TransformedScalar<T, F> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self::from_latent(self.latent - r.latent)
    }
}
impl<T: Mul<Output = T>, F> Mul for TransformedScalar<T, F> {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self::from_latent(self.latent * r.latent)
    }
}
impl<T: Neg<Output = T>, F> Neg for TransformedScalar<T, F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_latent(-self.latent)
    }
}

impl<T: NumericScalar, F> NumericScalar for TransformedScalar<T, F> {
    #[inline]
    fn zero() -> Self {
        Self::from_latent(T::zero())
    }
    #[inline]
    fn one() -> Self {
        Self::from_latent(T::one())
    }
    #[inline]
    fn abs(self) -> Self {
        Self::from_latent(self.latent.abs())
    }
    #[inline]
    fn from_i32(value: i32) -> Self {
        Self::from_latent(T::from_i32(value))
    }
}
