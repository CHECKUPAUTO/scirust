// scirust-simd/src/fixed/traits.rs
//
// # `NumericScalar` — abstraction de scalaire algébrique
//
// Trait fournissant le minimum arithmétique (anneau commutatif + `abs` +
// ordre) permettant d'écrire des algorithmes génériques une seule fois et de
// les instancier indifféremment sur `f32`, `f64` **ou** sur les types virgule
// fixe. C'est la brique prévue pour que de futures structures écrivent
// naturellement :
//
// ```ignore
// struct Quaternion<T: NumericScalar> { w: T, x: T, y: T, z: T }
// // fonctionne pour Quaternion<f32> ET Quaternion<FixedI32<16>>
// ```
//
// Le trait ne présuppose **aucune** propriété que la virgule fixe ne possède
// pas (pas d'infini, pas de NaN) : il reste dans l'anneau ordonné.

use core::ops::{Add, Mul, Neg, Sub};

use super::repr::FixedStorage;
use super::types::Fixed;

/// Scalaire algébrique ordonné : `+ − ×`, négation, `zero`/`one`, `abs`.
///
/// Implémenté pour `f32`, `f64` et tout [`Fixed<I, FRAC>`]. Les opérateurs
/// utilisent la politique par défaut du type (enveloppe pour la virgule fixe).
pub trait NumericScalar:
    Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Élément neutre additif.
    fn zero() -> Self;
    /// Élément neutre multiplicatif.
    fn one() -> Self;
    /// Valeur absolue.
    fn abs(self) -> Self;
    /// Petit entier littéral → scalaire (pour les coefficients de formules).
    fn from_i32(value: i32) -> Self;
}

impl<I: FixedStorage, const FRAC: u32> NumericScalar for Fixed<I, FRAC> {
    #[inline(always)]
    fn zero() -> Self {
        Fixed::zero()
    }
    #[inline(always)]
    fn one() -> Self {
        Fixed::one()
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Fixed::abs(self)
    }
    #[inline(always)]
    fn from_i32(value: i32) -> Self {
        // Passe par f64 puis saturation : couvre i32 et i64 sans surcharge
        // spécifique au stockage, et reste exact pour les petits entiers.
        Self::from_int_saturating(I::from_f64_saturating(value as f64))
    }
}

macro_rules! impl_numeric_scalar_float {
    ($ty:ty) => {
        impl NumericScalar for $ty {
            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn one() -> Self {
                1.0
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            #[inline(always)]
            fn from_i32(value: i32) -> Self {
                value as $ty
            }
        }
    };
}

impl_numeric_scalar_float!(f32);
impl_numeric_scalar_float!(f64);
