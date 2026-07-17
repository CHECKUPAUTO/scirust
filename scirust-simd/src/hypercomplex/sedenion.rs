// scirust-simd/src/hypercomplex/sedenion.rs
//
// Sédénions 𝕊 sur un registre 512 bits (`f32x16`).
//
// Un sédénion s = Σ sᵢ·eᵢ (i = 0..15) occupe les 16 lanes d'un `f32x16` :
//
//   lane :   0 ..  7    8 .. 15
//   s    = [ octonion a | octonion b ]
//
// La vue Cayley-Dickson s = (a, b) coïncide avec les deux moitiés
// 256 bits du registre. Sur AVX-512 le sédénion tient dans **un** registre
// ZMM et la séparation est une extraction de moitié (`vextractf32x8`) ;
// sur AVX2 LLVM alloue une paire de YMM et sur NEON quatre registres Q —
// dans tous les cas le déroulage est entièrement décidé à la compilation
// (`-C target-cpu=native`), sans boucle ni indirection à l'exécution.
//
// ## Note de résidence registre (AArch64)
//
// Le produit sédénion développe 16 produits de Hamilton : c'est le noyau le
// plus lourd de la pile. La forme d'**accumulation séquentielle** de `Mul`
// (deux accumulateurs au lieu de quatre produits simultanés) le rend
// register-résident (zéro spill de boucle chaude) sur les cœurs out-of-order
// visés — Neoverse N1/V1 (Graviton 2/3), Apple Silicon — et sur x86_64
// AVX-512. Sur `generic`/petit cœur in-order (Cortex-A72) et AVX2 (16
// registres), quelques spills subsistent faute de registres. Le tout est
// mesuré par `scripts/asm_spill_check.sh` — ne PAS affirmer « tout registre »
// sans relancer cette preuve.
//
// 𝕊 n'est ni associatif, ni alternatif, et possède des diviseurs de zéro
// (voir les tests) : c'est le prix de la 4ᵉ itération de Cayley-Dickson.

use core::ops::{Add, Mul, Neg, Sub};
use std::simd::{f32x16, num::SimdFloat, simd_swizzle};

use super::octonion::OctonionSimd;

/// Masque de conjugaison sédénionique : s̄ = s₀ − Σ sᵢ·eᵢ (i ≥ 1).
const CONJ_SIGNS: f32x16 = f32x16::from_array([
    1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
]);

/// Sédénion SIMD : wrapper transparent d'un `f32x16` (512 bits).
///
/// `#[repr(C, align(64))]` : alignement sur une ligne de cache complète,
/// requis pour les chargements `vmovaps zmm` alignés en AVX-512 et optimal
/// pour le préchargement sur ARM64.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SedenionSimd(pub f32x16);

impl SedenionSimd {
    /// Sédénion nul.
    pub const ZERO: Self = Self(f32x16::from_array([0.0; 16]));
    /// Unité réelle e₀ = 1.
    pub const ONE: Self = Self(f32x16::from_array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]));

    /// Construit un sédénion depuis ses 16 coordonnées `[e0, .., e15]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(coeffs: [f32; 16]) -> Self {
        Self(f32x16::from_array(coeffs))
    }

    /// Retourne les 16 coordonnées `[e0, .., e15]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 16] {
        self.0.to_array()
    }

    /// Élément de base eᵢ (i ∈ 0..16).
    #[inline(always)]
    #[must_use]
    pub fn unit(i: usize) -> Self {
        assert!(i < 16, "SedenionSimd::unit: index de base hors [0, 16)");
        let mut coeffs = [0.0f32; 16];
        coeffs[i] = 1.0;
        Self(f32x16::from_array(coeffs))
    }

    /// Sépare le sédénion s = (a, b) en ses deux octonions de
    /// Cayley-Dickson **sans copie mémoire**.
    ///
    /// Mêmes garanties qu'[`OctonionSimd::split`] : les swizzles constants
    /// [0..8) / [8..16) sont des extractions de moitié de registre
    /// (`vextractf32x8` sur AVX-512, renommage de registres sur AVX2/NEON
    /// où les moitiés vivent déjà dans des registres distincts).
    #[inline(always)]
    #[must_use]
    pub fn split(self) -> (OctonionSimd, OctonionSimd) {
        let a = simd_swizzle!(self.0, [0, 1, 2, 3, 4, 5, 6, 7]);
        let b = simd_swizzle!(self.0, [8, 9, 10, 11, 12, 13, 14, 15]);
        (OctonionSimd(a), OctonionSimd(b))
    }

    /// Recompose un sédénion depuis ses deux octonions (a, b).
    /// Swizzle de concaténation : indices 0..7 → `a`, 8..15 → `b`
    /// (`vinsertf32x8` sur AVX-512, no-op de renommage ailleurs).
    #[inline(always)]
    #[must_use]
    pub fn join(a: OctonionSimd, b: OctonionSimd) -> Self {
        Self(simd_swizzle!(
            a.0,
            b.0,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        ))
    }

    /// Conjugaison sédénionique s̄ (négation des 15 lanes imaginaires),
    /// un unique XOR de bits de signe sur le registre 512 bits.
    #[inline(always)]
    #[must_use]
    pub fn conj(self) -> Self {
        Self(self.0 * CONJ_SIGNS)
    }

    /// Norme au carré ‖s‖² = Σ sᵢ².
    ///
    /// ⚠️ Contrairement aux octonions, ‖x·y‖² ≠ ‖x‖²·‖y‖² en général :
    /// 𝕊 n'est pas une algèbre de composition (diviseurs de zéro).
    #[inline(always)]
    #[must_use]
    pub fn norm_sqr(self) -> f32 {
        (self.0 * self.0).reduce_sum()
    }

    /// Norme ‖s‖ = √(Σ sᵢ²).
    #[inline(always)]
    #[must_use]
    pub fn norm(self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Multiplication par un scalaire réel.
    #[inline(always)]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self(self.0 * f32x16::splat(s))
    }

    /// Sédénion unitaire de même direction, `s / ‖s‖`.
    ///
    /// Indéfini pour `s = 0` (produit `NaN`/`inf`, comme la division réelle
    /// par zéro).
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        self.scale(1.0 / self.norm())
    }

    /// Inverse `s⁻¹ = s̄ / ‖s‖²`, tel que `s⁻¹·s = s·s⁻¹ = 1`.
    ///
    /// L'identité `s̄·s = s·s̄ = ‖s‖²·1` tient à **tout** niveau de la
    /// construction de Cayley-Dickson — y compris 𝕊 — donc tout sédénion
    /// non nul est inversible des deux côtés, malgré l'existence de
    /// diviseurs de zéro (voir le test `sedenion_zero_divisors`).
    ///
    /// ⚠️ Ceci ne contredit pas les diviseurs de zéro : l'argument classique
    /// « `s` inversible et `s·t = 0` avec `t ≠ 0` sont incompatibles » repose
    /// sur l'associativité (`s⁻¹·(s·t) = (s⁻¹·s)·t = t`), qui **échoue** sur
    /// 𝕊. `s` peut donc être parfaitement inversible tout en admettant, pour
    /// un `t` non nul indépendant, `s·t = 0`. Indéfini pour `s = 0`.
    #[inline(always)]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conj().scale(1.0 / self.norm_sqr())
    }
}

impl Add for SedenionSimd {
    type Output = Self;
    /// Une seule addition vectorielle 512 bits (`vaddps zmm`) ou deux
    /// additions 256 bits déroulées par LLVM selon la cible.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for SedenionSimd {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for SedenionSimd {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul for SedenionSimd {
    type Output = Self;

    /// Produit de sédénions par la même récursion de Cayley-Dickson,
    /// un étage au-dessus des octonions :
    ///
    /// ```text
    ///   (a, b) * (c, d) = (a·c − d̄·b,  d·a + b·c̄),   a, b, c, d ∈ 𝕆
    /// ```
    ///
    /// **Accumulation séquentielle** : plutôt que de matérialiser les quatre
    /// produits d'octonions simultanément (`a·c`, `d̄·b`, `d·a`, `b·c̄`), on
    /// n'en garde que DEUX accumulateurs (`lo`, `hi`) et on consomme chaque
    /// entrée au plus tôt — `a` meurt après `d·a`, `d` après `d̄·b`. Cela
    /// réduit la pression registre sur NEON (voir `scripts/asm_spill_check.sh`),
    /// sans changer le résultat flottant (même ordre d'opérations par
    /// composante) ni pénaliser x86_64.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.split();
        let (c, d) = rhs.split();

        // Deux accumulateurs d'octonions ; ordre choisi pour libérer les
        // entrées tôt (produits 𝕆 non commutatifs — ordre des facteurs strict).
        let lo = a * c; //                       LO ← a·c
        let hi = d * a; //                       HI ← d·a     (a désormais mort)
        let lo = lo - d.conj() * b; //           LO ← LO − d̄·b (d désormais mort)
        let hi = hi + b * c.conj(); //           HI ← HI + b·c̄ (b, c morts)

        Self::join(lo, hi)
    }
}
