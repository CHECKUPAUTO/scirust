// scirust-simd/src/reductions/simd_scalar.rs
//
// # `SimdScalar` — abstraction de vecteur de lanes
//
// Trait interne unifiant les types `std::simd` de largeurs variées
// (`f32x4/8/16`, `f64x2/4/8`, …) derrière une même interface : splat, zéro,
// FMA, réductions horizontales, accès lane. Il évite que chaque module de la
// crate (activations, normes, attention, quantification, hypercomplexe…)
// réimplémente ses propres opérations par largeur.
//
// L'implémentation est **générique** (`impl<T, const N> for Simd<T, N>`) : une
// seule définition couvre toutes les largeurs supportées, sans duplication.
//
// Ce trait sert de socle aux réductions de [`super`]. Il n'expose que ce dont
// les kernels ont besoin ; il n'est pas destiné à devenir une algèbre SIMD
// complète.

use core::ops::{Add, Mul, Sub};
use std::simd::num::SimdFloat;
use std::simd::{Simd, SimdElement, StdFloat};

/// Vecteur SIMD de `LANES` scalaires `Scalar`, avec les opérations élémentaires
/// nécessaires aux réductions génériques.
///
/// Implémenté en bloc pour tout `Simd<T, N>` flottant supporté — donc
/// `f32x4`, `f32x8`, `f32x16`, `f64x2`, `f64x4`, `f64x8`, etc.
pub trait SimdScalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    /// Type scalaire d'une lane (`f32`, `f64`, …).
    type Scalar: Copy;
    /// Nombre de lanes du vecteur.
    const LANES: usize;

    /// Vecteur dont toutes les lanes valent `value`.
    fn splat(value: Self::Scalar) -> Self;
    /// Vecteur nul (toutes lanes à 0).
    fn zero() -> Self;
    /// Charge `LANES` scalaires contigus depuis un slice (panique si trop court).
    fn from_slice(slice: &[Self::Scalar]) -> Self;
    /// Valeur de la lane `i` (0 ≤ i < LANES).
    fn lane(self, i: usize) -> Self::Scalar;

    /// FMA vectorielle fusionnée : `self * a + b` (un seul arrondi par lane).
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Valeur absolue lane-à-lane.
    fn abs(self) -> Self;
    /// Maximum lane-à-lane (sémantique IEEE `maxNum` : ignore les NaN isolés).
    fn simd_max(self, other: Self) -> Self;
    /// Minimum lane-à-lane (sémantique IEEE `minNum`).
    fn simd_min(self, other: Self) -> Self;

    /// Somme horizontale des lanes. ⚠️ Ordre de réduction **non spécifié**
    /// (dépend du matériel) : rapide mais non reproductible bit à bit. Pour une
    /// somme déterministe, réduire les lanes via [`SimdScalar::lane`] dans un
    /// ordre d'indice fixe.
    fn reduce_sum(self) -> Self::Scalar;
    /// Maximum horizontal des lanes.
    fn reduce_max(self) -> Self::Scalar;
    /// Minimum horizontal des lanes.
    fn reduce_min(self) -> Self::Scalar;
}

// Depuis le retrait de `LaneCount`/`SupportedLaneCount` de `std::simd`, la
// largeur `N` n'est plus contrainte par un trait : la limite (≤ 64 lanes) est
// imposée à la monomorphisation. Le seul témoin dont on a besoin est que
// `Simd<T, N>` soit un flottant (`SimdFloat`) disposant de FMA (`StdFloat`).
impl<T, const N: usize> SimdScalar for Simd<T, N>
where
    T: SimdElement,
    Self: SimdFloat<Scalar = T>
        + StdFloat
        + Default
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>,
{
    type Scalar = T;
    const LANES: usize = N;

    #[inline(always)]
    fn splat(value: T) -> Self {
        Simd::splat(value)
    }

    #[inline(always)]
    fn zero() -> Self {
        // `Default` d'un `Simd` flottant = toutes lanes à 0.0.
        Self::default()
    }

    #[inline(always)]
    fn from_slice(slice: &[T]) -> Self {
        Simd::from_slice(slice)
    }

    #[inline(always)]
    fn lane(self, i: usize) -> T {
        // `as_array` est un accès registre→pile trivial ; utilisé seulement
        // hors boucle chaude (réduction finale déterministe / Kahan).
        self.as_array()[i]
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        StdFloat::mul_add(self, a, b)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        SimdFloat::abs(self)
    }

    #[inline(always)]
    fn simd_max(self, other: Self) -> Self {
        SimdFloat::simd_max(self, other)
    }

    #[inline(always)]
    fn simd_min(self, other: Self) -> Self {
        SimdFloat::simd_min(self, other)
    }

    #[inline(always)]
    fn reduce_sum(self) -> T {
        SimdFloat::reduce_sum(self)
    }

    #[inline(always)]
    fn reduce_max(self) -> T {
        SimdFloat::reduce_max(self)
    }

    #[inline(always)]
    fn reduce_min(self) -> T {
        SimdFloat::reduce_min(self)
    }
}
