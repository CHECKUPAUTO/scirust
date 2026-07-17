// scirust-simd/src/hypercomplex/octonion.rs
//
// Octonions 𝕆 sur un registre 256 bits (`f32x8`).
//
// Un octonion o = Σ oᵢ·eᵢ (i = 0..7, e₀ = 1) occupe les 8 lanes d'un
// `f32x8` dans l'ordre naturel :
//
//   lane :   0    1    2    3    4    5    6    7
//   o    = [ e0,  e1,  e2,  e3 | e4,  e5,  e6,  e7 ]
//            └── quaternion a ──┘└── quaternion b ──┘
//
// La vue Cayley-Dickson o = (a, b) correspond exactement aux deux moitiés
// 128 bits du registre : sur x86_64/AVX2 ce sont les lanes basses/hautes
// d'un YMM, sur ARM64/NEON les deux registres Q jumelés que LLVM alloue
// pour un vecteur 256 bits. La « séparation » est donc gratuite.

use core::ops::{Add, Mul, Neg, Sub};
use std::simd::{f32x4, f32x8, num::SimdFloat, simd_swizzle};

use super::quat::{quat_conj, quat_mul};

/// Masque de conjugaison octonionique : ō = o₀ − Σ oᵢ·eᵢ (i ≥ 1).
const CONJ_SIGNS: f32x8 = f32x8::from_array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);

/// Octonion SIMD : wrapper transparent d'un `f32x8` (256 bits).
///
/// `#[repr(C, align(32))]` garantit qu'un tableau d'`OctonionSimd` est
/// chargeable par `vmovaps` alignés (32 octets) et que le type traverse
/// une frontière FFI avec un layout défini.
///
/// Type valeur pur : `Copy`, aucune allocation, toutes les opérations
/// restent en registres.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct OctonionSimd(pub f32x8);

impl OctonionSimd {
    /// Octonion nul.
    pub const ZERO: Self = Self(f32x8::from_array([0.0; 8]));
    /// Unité réelle e₀ = 1.
    pub const ONE: Self = Self(f32x8::from_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));

    /// Construit un octonion depuis ses 8 coordonnées `[e0, .., e7]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(coeffs: [f32; 8]) -> Self {
        Self(f32x8::from_array(coeffs))
    }

    /// Retourne les 8 coordonnées `[e0, .., e7]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 8] {
        self.0.to_array()
    }

    /// Élément de base eᵢ (i ∈ 0..8).
    #[inline(always)]
    #[must_use]
    pub fn unit(i: usize) -> Self {
        assert!(i < 8, "OctonionSimd::unit: index de base hors [0, 8)");
        let mut coeffs = [0.0f32; 8];
        coeffs[i] = 1.0;
        Self(f32x8::from_array(coeffs))
    }

    /// Sépare l'octonion o = (a, b) en ses deux quaternions de
    /// Cayley-Dickson **sans copie mémoire**.
    ///
    /// `simd_swizzle!` avec indices constants [0..4) et [4..8) est reconnu
    /// par LLVM comme une extraction de moitié de registre :
    /// `vextractf128` (ou un simple renommage de la moitié basse) sur
    /// AVX2, et une réutilisation directe des deux registres Q sur NEON
    /// où le vecteur 256 bits vit déjà en paire {q_lo, q_hi}. Coût : 0 ou
    /// 1 µop, jamais de passage par la pile.
    #[inline(always)]
    #[must_use]
    pub fn split(self) -> (f32x4, f32x4) {
        let a = simd_swizzle!(self.0, [0, 1, 2, 3]); // moitié basse  → quaternion a
        let b = simd_swizzle!(self.0, [4, 5, 6, 7]); // moitié haute → quaternion b
        (a, b)
    }

    /// Recompose un octonion depuis ses deux quaternions (a, b).
    ///
    /// Le swizzle à deux entrées concatène les registres : indices 0..3
    /// adressent `a`, indices 4..7 adressent `b`. LLVM émet un
    /// `vinsertf128` (x86) ou laisse simplement la paire NEON en place.
    #[inline(always)]
    #[must_use]
    pub fn join(a: f32x4, b: f32x4) -> Self {
        Self(simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]))
    }

    /// Conjugaison octonionique ō (négation des 7 lanes imaginaires).
    /// Compilée en un unique XOR de bits de signe.
    #[inline(always)]
    #[must_use]
    pub fn conj(self) -> Self {
        Self(self.0 * CONJ_SIGNS)
    }

    /// Norme au carré ‖o‖² = Σ oᵢ² (réduction horizontale du registre).
    #[inline(always)]
    #[must_use]
    pub fn norm_sqr(self) -> f32 {
        (self.0 * self.0).reduce_sum()
    }

    /// Norme ‖o‖ = √(Σ oᵢ²).
    #[inline(always)]
    #[must_use]
    pub fn norm(self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Multiplication par un scalaire réel.
    #[inline(always)]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self(self.0 * f32x8::splat(s))
    }

    /// Octonion unitaire de même direction, `o / ‖o‖`.
    ///
    /// Indéfini pour `o = 0` (produit `NaN`/`inf`, comme la division réelle
    /// par zéro).
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        self.scale(1.0 / self.norm())
    }

    /// Inverse `o⁻¹ = ō / ‖o‖²`, tel que `o⁻¹·o = o·o⁻¹ = 1`.
    ///
    /// 𝕆 est une algèbre de division normée (alternative) : `ō·o = o·ō =
    /// ‖o‖²·1` exactement, donc tout élément non nul est inversible des
    /// deux côtés. Indéfini pour `o = 0`.
    #[inline(always)]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conj().scale(1.0 / self.norm_sqr())
    }
}

impl Add for OctonionSimd {
    type Output = Self;
    /// Addition composante à composante : une seule instruction `vaddps`.
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for OctonionSimd {
    type Output = Self;
    /// Soustraction composante à composante : une seule instruction `vsubps`.
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for OctonionSimd {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul for OctonionSimd {
    type Output = Self;

    /// Produit d'octonions par la formule de Cayley-Dickson :
    ///
    /// ```text
    ///   (a, b) * (c, d) = (a·c − d̄·b,  d·a + b·c̄)
    /// ```
    ///
    /// avec a, b, c, d ∈ ℍ et `·` le produit de Hamilton vectorisé
    /// ([`quat_mul`]). ⚠️ ℍ n'est pas commutatif : l'ordre des opérandes
    /// de chaque `quat_mul` reproduit strictement la formule.
    ///
    /// Bilan après inlining : 4 produits de Hamilton (4 × [4 shuffles +
    /// 4 broadcasts + 1 mul + 3 FMA]) + 2 conjugaisons (XOR de signes) +
    /// 1 add + 1 sub + les extractions/insertions de moitiés — soit une
    /// trentaine de µops vectoriels, sans aucun accès mémoire.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = self.split();
        let (c, d) = rhs.split();

        // Partie basse : a·c − d̄·b
        let lo = quat_mul(a, c) - quat_mul(quat_conj(d), b);
        // Partie haute : d·a + b·c̄
        let hi = quat_mul(d, a) + quat_mul(b, quat_conj(c));

        Self::join(lo, hi)
    }
}
