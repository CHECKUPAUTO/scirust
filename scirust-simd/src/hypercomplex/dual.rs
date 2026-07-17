// scirust-simd/src/hypercomplex/dual.rs
//
// Différenciation automatique forward-mode pour 𝕆 et 𝕊.
//
// On étend l'algèbre A (octonions ou sédénions) par un infinitésimal
// nilpotent ε avec ε² = 0 : les éléments de A[ε] s'écrivent v + ε·d où
// `v` est la valeur et `d` la dérivée directionnelle transportée.
//
// ## ε et la non-associativité
//
// ε est adjoint comme élément **central** : il commute avec tout A et
// l'extension est A ⊗ ℝ[ε]/(ε²) — le produit tensoriel d'algèbres, dans
// lequel les scalaires ℝ[ε] commutent et s'associent avec tout. La
// non-associativité de 𝕆/𝕊 vit entièrement dans les coefficients :
//
//   (v₁ + ε·d₁)(v₂ + ε·d₂) = v₁v₂ + ε·(d₁v₂ + v₁d₂) + ε²·(d₁d₂)
//                          = v₁v₂ + ε·(d₁v₂ + v₁d₂)
//
// C'est exactement la règle de Leibniz **non commutative** : l'ordre des
// facteurs dans d₁v₂ et v₁d₂ est significatif et doit être préservé
// (contrairement au cas réel où l'on écrirait v·d' + v'·d sans y penser).
//
// Chaque opération duale reste 100 % registre : un dual = 2 registres
// SIMD, un produit dual = 3 produits hypercomplexes + 1 addition.

use core::ops::{Add, Mul, Neg, Sub};
use std::simd::num::SimdFloat;

use super::octonion::OctonionSimd;
use super::sedenion::SedenionSimd;

/// Génère un type dual forward-mode au-dessus d'une algèbre hypercomplexe
/// SIMD. Les deux instanciations (𝕆, 𝕊) partagent strictement la même
/// règle de Leibniz non commutative — la macro évite toute divergence.
macro_rules! impl_dual {
    ($(#[$doc:meta])* $dual:ident, $base:ty) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Default)]
        pub struct $dual {
            /// Partie valeur (coefficient de 1).
            pub val: $base,
            /// Partie dérivée (coefficient de ε).
            pub eps: $base,
        }

        impl $dual {
            /// Construit v + ε·d.
            #[inline(always)]
            #[must_use]
            pub const fn new(val: $base, eps: $base) -> Self {
                Self { val, eps }
            }

            /// Lifte une constante : dérivée nulle (∂c/∂t = 0).
            #[inline(always)]
            #[must_use]
            pub const fn constant(val: $base) -> Self {
                Self { val, eps: <$base>::ZERO }
            }

            /// Point d'entrée de la différenciation : la variable
            /// x(t) = val + t·dir, vue en t = 0 avec ∂x/∂t = dir.
            /// Pour une dérivée « canonique », prendre dir = 1 (e₀).
            #[inline(always)]
            #[must_use]
            pub const fn variable(val: $base, dir: $base) -> Self {
                Self { val, eps: dir }
            }

            /// Conjugaison duale : la conjugaison est ℝ-linéaire, donc
            /// elle commute avec ∂ — on conjugue les deux parties.
            #[inline(always)]
            #[must_use]
            pub fn conj(self) -> Self {
                Self { val: self.val.conj(), eps: self.eps.conj() }
            }

            /// Norme au carré duale : f = ‖v‖², f' = 2⟨v, d⟩.
            /// (‖v‖² = Σ vᵢ² ⇒ ∂‖v‖² = 2 Σ vᵢ·dᵢ, produit scalaire réel.)
            #[inline(always)]
            #[must_use]
            pub fn norm_sqr(self) -> (f32, f32) {
                let value = self.val.norm_sqr();
                // ⟨v, d⟩ via le même chemin registre que norm_sqr :
                // multiplication lane à lane puis réduction horizontale.
                let dot = (self.val.0 * self.eps.0)
                    .reduce_sum();
                (value, 2.0 * dot)
            }

            /// Norme duale : f = ‖v‖, f' = ⟨v, d⟩/‖v‖ (règle de dérivation en
            /// chaîne de `√` appliquée à [`Self::norm_sqr`] : `(√u)' = u'/(2√u)`).
            /// Indéfini pour `v = 0` (comme `1/0`).
            #[inline(always)]
            #[must_use]
            pub fn norm(self) -> (f32, f32) {
                let (n2, dn2) = self.norm_sqr();
                let n = n2.sqrt();
                (n, dn2 / (2.0 * n))
            }

            /// Normalisation duale : f = v/‖v‖, dérivée par la règle du
            /// quotient (`v` vectoriel, `‖v‖` scalaire réel).
            /// Indéfini pour `v = 0`.
            #[inline(always)]
            #[must_use]
            pub fn normalize(self) -> Self {
                let (n, dn) = self.norm();
                let inv_n = 1.0 / n;
                let dinv_n = -dn / (n * n); // (1/f)' = −f'/f²
                Self {
                    val: self.val.scale(inv_n),
                    eps: self.eps.scale(inv_n) + self.val.scale(dinv_n),
                }
            }

            /// Inverse dual : f(t) = v(t)⁻¹ = v̄(t)/‖v(t)‖², dérivée par la
            /// règle du produit (`v̄(t)` vectoriel, `1/‖v(t)‖²` scalaire réel).
            /// Indéfini pour `v = 0`.
            #[inline(always)]
            #[must_use]
            pub fn inverse(self) -> Self {
                let (n2, dn2) = self.norm_sqr();
                let r = 1.0 / n2;
                let dr = -dn2 / (n2 * n2); // (1/f)' = −f'/f²
                let c = self.conj();
                Self {
                    val: c.val.scale(r),
                    eps: c.eps.scale(r) + c.val.scale(dr),
                }
            }
        }

        impl Add for $dual {
            type Output = Self;
            /// (v₁ + ε·d₁) + (v₂ + ε·d₂) = (v₁+v₂) + ε·(d₁+d₂).
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self { val: self.val + rhs.val, eps: self.eps + rhs.eps }
            }
        }

        impl Sub for $dual {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self { val: self.val - rhs.val, eps: self.eps - rhs.eps }
            }
        }

        impl Neg for $dual {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self { val: -self.val, eps: -self.eps }
            }
        }

        impl Mul for $dual {
            type Output = Self;
            /// Règle de Leibniz non commutative (ε central, ε² = 0) :
            ///
            /// ```text
            ///   (v₁ + ε·d₁)(v₂ + ε·d₂) = v₁v₂ + ε·(d₁v₂ + v₁d₂)
            /// ```
            ///
            /// L'ordre des opérandes dans `d₁·v₂` et `v₁·d₂` est
            /// STRICTEMENT celui de la formule : 𝕆 et 𝕊 ne sont pas
            /// commutatifs, intervertir fausserait la dérivée.
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self {
                    val: self.val * rhs.val,
                    eps: self.eps * rhs.val + self.val * rhs.eps,
                }
            }
        }
    };
}

impl_dual!(
    /// Octonion dual v + ε·d pour l'AD forward-mode sur 𝕆.
    ///
    /// 2 × 256 bits = deux registres YMM (ou deux paires NEON) ; le
    /// produit dual coûte 3 produits d'octonions + 1 addition, toujours
    /// sans allocation.
    DualOctonion,
    OctonionSimd
);

impl_dual!(
    /// Sédénion dual v + ε·d pour l'AD forward-mode sur 𝕊.
    ///
    /// 2 × 512 bits = deux registres ZMM sur AVX-512. Attention : les
    /// diviseurs de zéro de 𝕊 se propagent aux duaux (l'AD n'y change
    /// rien — c'est une propriété de l'algèbre sous-jacente).
    DualSedenion,
    SedenionSimd
);
