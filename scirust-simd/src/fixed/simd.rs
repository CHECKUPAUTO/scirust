// scirust-simd/src/fixed/simd.rs
//
// # Vecteurs virgule fixe SIMD
//
// [`FixedI32x8`] (8 lanes `FixedI32<FRAC>`, 256 bits) et [`FixedI64x4`]
// (4 lanes `FixedI64<FRAC>`, 256 bits) sur `std::simd`. Le stockage
// `#[repr(transparent)]` de [`Fixed`] garantit qu'un `[FixedI32<F>; 8]` a le
// layout d'un `[i32; 8]` : la conversion tableau ↔ vecteur passe par les bruts,
// sans `unsafe` ni copie cachée.
//
// Les opérateurs `+ − * −x` sont implémentés (enveloppants, comme le scalaire).
//
// ## Multiplication
//
// * `FixedI32x8` : **entièrement vectorisée**. Chaque moitié `i32x4` est élargie
//   en `i64x4` (produit exact), décalée de `FRAC` avec arrondi **vers zéro**
//   (pour coïncider bit-à-bit avec l'opérateur scalaire), puis rétrécie en
//   `i32x4`. Tout reste en registres.
// * `FixedI64x4` : `std::simd` n'a pas de vecteur `i128`, donc la multiplication
//   est **scalarisée** lane-à-lane via [`FixedI64`](super::FixedI64) (add/sub/
//   neg/min/max/abs/comparaison/sélection restent vectorisés). Documenté.

use core::ops::{Add, Mul, Neg, Sub};
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::SimdInt;
use std::simd::{Mask, Select, i32x4, i32x8, i64x4, simd_swizzle};

use super::types::Fixed;

/// Applique un décalage arrondi **vers zéro** de `frac` bits à un `i64x4`,
/// lane à lane.
///
/// Vers-zéro = plancher pour les positifs, plafond pour les négatifs. On ajoute
/// un biais `(2^frac − 1)` aux lanes négatives (détectées par le masque de signe
/// `v >> 63`) avant le décalage arithmétique — identique à la sémantique scalaire.
#[inline(always)]
fn shift_toward_zero_i64x4(v: i64x4, frac: u32) -> i64x4 {
    if frac == 0
    {
        return v;
    }
    let sign = v >> i64x4::splat(63); // −1 (tous bits) si négatif, 0 sinon
    let bias = sign & i64x4::splat((1i64 << frac) - 1);
    (v + bias) >> i64x4::splat(frac as i64)
}

// ================================================================== //
//  FixedI32x8                                                         //
// ================================================================== //

/// Vecteur de 8 nombres `FixedI32<FRAC>` (256 bits).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedI32x8<const FRAC: u32>(pub i32x8);

impl<const FRAC: u32> FixedI32x8<FRAC> {
    /// Depuis 8 lanes brutes.
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: i32x8) -> Self {
        Self(raw)
    }

    /// Depuis un tableau de scalaires virgule fixe.
    #[inline]
    #[must_use]
    pub fn from_array(values: [Fixed<i32, FRAC>; 8]) -> Self {
        Self(i32x8::from_array(values.map(|v| v.0)))
    }

    /// Vers un tableau de scalaires virgule fixe.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [Fixed<i32, FRAC>; 8] {
        self.0.to_array().map(Fixed::from_raw)
    }

    /// Toutes lanes = `value`.
    #[inline(always)]
    #[must_use]
    pub fn splat(value: Fixed<i32, FRAC>) -> Self {
        Self(i32x8::splat(value.0))
    }

    /// Vecteur nul.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self(i32x8::splat(0))
    }

    /// FMA lane à lane : `self·b + c` (produit virgule fixe arrondi vers zéro
    /// puis addition exacte enveloppante).
    #[inline]
    #[must_use]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    /// Minimum lane à lane.
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self(self.0.simd_min(rhs.0))
    }
    /// Maximum lane à lane.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self(self.0.simd_max(rhs.0))
    }
    /// Restreint à `[lo, hi]` lane à lane.
    #[inline(always)]
    #[must_use]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Valeur absolue **saturante** (`MIN` ↦ `MAX`), pour coïncider avec le
    /// scalaire.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        let a = self.0.abs(); // enveloppant : MIN ↦ MIN
        let is_min = self.0.simd_eq(i32x8::splat(i32::MIN));
        Self(is_min.select(i32x8::splat(i32::MAX), a))
    }

    /// Masque `self == rhs` lane à lane.
    #[inline(always)]
    #[must_use]
    pub fn simd_eq(self, rhs: Self) -> Mask<i32, 8> {
        self.0.simd_eq(rhs.0)
    }
    /// Masque `self < rhs`.
    #[inline(always)]
    #[must_use]
    pub fn simd_lt(self, rhs: Self) -> Mask<i32, 8> {
        self.0.simd_lt(rhs.0)
    }
    /// Masque `self <= rhs`.
    #[inline(always)]
    #[must_use]
    pub fn simd_le(self, rhs: Self) -> Mask<i32, 8> {
        self.0.simd_le(rhs.0)
    }

    /// Sélection lane à lane : `mask ? a : b` (blend).
    #[inline(always)]
    #[must_use]
    pub fn select(mask: Mask<i32, 8>, a: Self, b: Self) -> Self {
        Self(mask.select(a.0, b.0))
    }

    /// Somme horizontale **exacte** des 8 lanes (accumulée en `i64` pour éviter
    /// tout débordement intermédiaire), rendue en scalaire virgule fixe
    /// enveloppant. Ordre de réduction fixe → déterministe.
    #[inline]
    #[must_use]
    pub fn reduce_sum(self) -> Fixed<i32, FRAC> {
        let mut acc: i64 = 0;
        for lane in self.0.to_array()
        {
            acc += lane as i64;
        }
        Fixed::from_raw(acc as i32)
    }
}

impl<const FRAC: u32> Add for FixedI32x8<FRAC> {
    type Output = Self;
    /// Addition enveloppante lane à lane.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<const FRAC: u32> Sub for FixedI32x8<FRAC> {
    type Output = Self;
    /// Soustraction enveloppante lane à lane.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<const FRAC: u32> Neg for FixedI32x8<FRAC> {
    type Output = Self;
    /// Négation enveloppante lane à lane.
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i32x8::splat(0) - self.0)
    }
}
impl<const FRAC: u32> Mul for FixedI32x8<FRAC> {
    type Output = Self;
    /// Multiplication virgule fixe vectorisée (enveloppante, arrondi vers zéro).
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Moitiés basses/hautes en i32x4 (extraction de moitié de registre).
        let a_lo: i32x4 = simd_swizzle!(self.0, [0, 1, 2, 3]);
        let a_hi: i32x4 = simd_swizzle!(self.0, [4, 5, 6, 7]);
        let b_lo: i32x4 = simd_swizzle!(rhs.0, [0, 1, 2, 3]);
        let b_hi: i32x4 = simd_swizzle!(rhs.0, [4, 5, 6, 7]);
        // Élargissement exact i32→i64, produit, décalage arrondi vers zéro.
        let p_lo = shift_toward_zero_i64x4(a_lo.cast::<i64>() * b_lo.cast::<i64>(), FRAC);
        let p_hi = shift_toward_zero_i64x4(a_hi.cast::<i64>() * b_hi.cast::<i64>(), FRAC);
        // Rétrécissement (troncature) puis recombinaison.
        let r_lo: i32x4 = p_lo.cast::<i32>();
        let r_hi: i32x4 = p_hi.cast::<i32>();
        Self(simd_swizzle!(r_lo, r_hi, [0, 1, 2, 3, 4, 5, 6, 7]))
    }
}

// ================================================================== //
//  FixedI64x4                                                         //
// ================================================================== //

/// Vecteur de 4 nombres `FixedI64<FRAC>` (256 bits).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedI64x4<const FRAC: u32>(pub i64x4);

impl<const FRAC: u32> FixedI64x4<FRAC> {
    /// Depuis 4 lanes brutes.
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: i64x4) -> Self {
        Self(raw)
    }

    /// Depuis un tableau de scalaires virgule fixe.
    #[inline]
    #[must_use]
    pub fn from_array(values: [Fixed<i64, FRAC>; 4]) -> Self {
        Self(i64x4::from_array(values.map(|v| v.0)))
    }
    /// Vers un tableau de scalaires.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [Fixed<i64, FRAC>; 4] {
        self.0.to_array().map(Fixed::from_raw)
    }
    /// Toutes lanes = `value`.
    #[inline(always)]
    #[must_use]
    pub fn splat(value: Fixed<i64, FRAC>) -> Self {
        Self(i64x4::splat(value.0))
    }
    /// Vecteur nul.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self(i64x4::splat(0))
    }

    /// FMA lane à lane : `self·b + c`.
    #[inline]
    #[must_use]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    /// Minimum lane à lane (vectorisé).
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self(self.0.simd_min(rhs.0))
    }
    /// Maximum lane à lane (vectorisé).
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self(self.0.simd_max(rhs.0))
    }
    /// Restreint à `[lo, hi]`.
    #[inline(always)]
    #[must_use]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Valeur absolue saturante (`MIN` ↦ `MAX`).
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        let a = self.0.abs();
        let is_min = self.0.simd_eq(i64x4::splat(i64::MIN));
        Self(is_min.select(i64x4::splat(i64::MAX), a))
    }

    /// Masque `self == rhs`.
    #[inline(always)]
    #[must_use]
    pub fn simd_eq(self, rhs: Self) -> Mask<i64, 4> {
        self.0.simd_eq(rhs.0)
    }
    /// Masque `self < rhs`.
    #[inline(always)]
    #[must_use]
    pub fn simd_lt(self, rhs: Self) -> Mask<i64, 4> {
        self.0.simd_lt(rhs.0)
    }
    /// Sélection lane à lane : `mask ? a : b`.
    #[inline(always)]
    #[must_use]
    pub fn select(mask: Mask<i64, 4>, a: Self, b: Self) -> Self {
        Self(mask.select(a.0, b.0))
    }

    /// Somme horizontale **exacte** des 4 lanes (accumulée en `i128`), ordre
    /// fixe → déterministe.
    #[inline]
    #[must_use]
    pub fn reduce_sum(self) -> Fixed<i64, FRAC> {
        let mut acc: i128 = 0;
        for lane in self.0.to_array()
        {
            acc += lane as i128;
        }
        Fixed::from_raw(acc as i64)
    }
}

impl<const FRAC: u32> Add for FixedI64x4<FRAC> {
    type Output = Self;
    /// Addition enveloppante lane à lane (vectorisée).
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<const FRAC: u32> Sub for FixedI64x4<FRAC> {
    type Output = Self;
    /// Soustraction enveloppante lane à lane (vectorisée).
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<const FRAC: u32> Neg for FixedI64x4<FRAC> {
    type Output = Self;
    /// Négation enveloppante lane à lane (vectorisée).
    #[inline(always)]
    fn neg(self) -> Self {
        Self(i64x4::splat(0) - self.0)
    }
}
impl<const FRAC: u32> Mul for FixedI64x4<FRAC> {
    type Output = Self;
    /// Multiplication virgule fixe **scalarisée** (pas de `i128` SIMD) :
    /// lane à lane via [`FixedI64`](super::FixedI64). Enveloppante, arrondi
    /// vers zéro — identique au scalaire.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        let mut out = [Fixed::<i64, FRAC>::from_raw(0); 4];
        for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b.iter()))
        {
            *o = x.wrapping_mul(y);
        }
        Self::from_array(out)
    }
}
