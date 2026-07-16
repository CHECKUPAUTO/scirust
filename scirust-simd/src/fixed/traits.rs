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

// ------------------------------------------------------------------ //
//  RealScalar — extension « corps réel » (racine + transcendantes)    //
// ------------------------------------------------------------------ //

/// Scalaire réel : [`NumericScalar`] + racine, inverse et transcendantes.
///
/// Séparé de [`NumericScalar`] parce que ces opérations ne sont **pas exactes**
/// (arrondi borné) — un algorithme purement algébrique (produit hypercomplexe,
/// GEMM entier) reste sur `NumericScalar` ; seuls ceux qui ont besoin de
/// `sin`/`sqrt`/… (slerp, normalisation, activations) demandent `RealScalar`.
///
/// Implémenté pour `f32`, `f64` (délégation à `std`) et `FixedI32<FRAC>`
/// (virgule fixe déterministe, cf. [`super::transcendental`] et [`super::math`]).
/// Les versions virgule fixe ont des bornes d'erreur ULP prouvées par test
/// exhaustif.
pub trait RealScalar: NumericScalar {
    /// Racine carrée (`≤ 0` ↦ `0` en virgule fixe).
    fn sqrt(self) -> Self;
    /// Inverse `1/x` (`x = 0` ↦ saturation en virgule fixe).
    fn recip(self) -> Self;
    /// Exponentielle `eˣ`.
    fn exp(self) -> Self;
    /// `2ˣ`.
    fn exp2(self) -> Self;
    /// Logarithme naturel (`≤ 0` ↦ `min` en virgule fixe).
    fn ln(self) -> Self;
    /// Logarithme base 2.
    fn log2(self) -> Self;
    /// Sinus.
    fn sin(self) -> Self;
    /// Cosinus.
    fn cos(self) -> Self;
    /// Tangente hyperbolique.
    fn tanh(self) -> Self;
    /// Sigmoïde logistique `1/(1+e^{-x})`.
    fn sigmoid(self) -> Self;
}

macro_rules! impl_real_scalar_float {
    ($ty:ty) => {
        impl RealScalar for $ty {
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            #[inline(always)]
            fn recip(self) -> Self {
                <$ty>::recip(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            #[inline(always)]
            fn exp2(self) -> Self {
                <$ty>::exp2(self)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <$ty>::ln(self)
            }
            #[inline(always)]
            fn log2(self) -> Self {
                <$ty>::log2(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                <$ty>::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                <$ty>::cos(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                <$ty>::tanh(self)
            }
            #[inline(always)]
            fn sigmoid(self) -> Self {
                1.0 / (1.0 + (-self).exp())
            }
        }
    };
}

impl_real_scalar_float!(f32);
impl_real_scalar_float!(f64);

impl<const FRAC: u32> RealScalar for Fixed<i32, FRAC> {
    #[inline(always)]
    fn sqrt(self) -> Self {
        super::math::sqrt(self)
    }
    #[inline(always)]
    fn recip(self) -> Self {
        // Saturation sur x=0 / débordement (pas d'infini en virgule fixe).
        super::math::reciprocal(self).unwrap_or_else(|| {
            if self.is_negative()
            {
                Fixed::min_value()
            }
            else
            {
                Fixed::max_value()
            }
        })
    }
    #[inline(always)]
    fn exp(self) -> Self {
        super::transcendental::exp(self)
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        super::transcendental::exp2(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        super::transcendental::ln(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        super::transcendental::log2(self)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        super::transcendental::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        super::transcendental::cos(self)
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        super::transcendental::tanh(self)
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        super::transcendental::sigmoid(self)
    }
}
