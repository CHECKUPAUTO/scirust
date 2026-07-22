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
//
// ## exp/ln/powf duaux
//
// Au-delà de `conj`/`norm`/`normalize`/`inverse`, [`DualOctonion`]/
// [`DualSedenion`] exposent `exp`/`ln`/`powf`, pendants duaux des mêmes
// fonctions sur [`OctonionSimd`]/[`SedenionSimd`] : `f'` est la dérivée de
// Fréchet de `f` en `v`, dans la direction `d`, obtenue en différentiant
// terme à terme le développement `exp(w·e₀+p) = eʷ·(cosθ·e₀ + sinc(θ)·p)`.
// Un fait clé simplifie les points quasi réels (`θ = ‖p‖ → 0`) : `v` y est
// au premier ordre un élément **central** (`w·e₀`), donc la dérivée de
// Fréchet d'une fonction analytique s'y réduit exactement à `f'(w)·d` —
// sans distinguer partie réelle/imaginaire de la direction `d`.

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

            /// Mise à l'échelle réelle duale : `t` est une constante (non
            /// dérivée), donc `(t·f)' = t·f'` — linéarité en `f`.
            #[inline(always)]
            #[must_use]
            pub fn scale(self, t: f32) -> Self {
                Self { val: self.val.scale(t), eps: self.eps.scale(t) }
            }

            /// Exponentielle duale : `f = exp(v)`, `f' = D(exp)ᵥ[d]`
            /// (dérivée de Fréchet), obtenue en différentiant chaque terme
            /// de `exp(w·e₀ + p) = eʷ·(cos θ·e₀ + sinc(θ)·p)` (`θ = ‖p‖`,
            /// `sinc(θ) = sin(θ)/θ`) — cf. [`OctonionSimd::exp`]/
            /// [`SedenionSimd::exp`], dont on réutilise le même seuil.
            ///
            /// Pour `θ < tiny`, `v` est au premier ordre un élément
            /// **central** (`w·e₀`, qui commute avec tout) : la dérivée de
            /// Fréchet d'une fonction analytique en un point central se
            /// réduit exactement à `f'(w)·d` — ici `eʷ·d`, sans distinguer
            /// partie réelle/imaginaire de `d` (cf. calcul fonctionnel sur
            /// élément central).
            #[inline]
            #[must_use]
            pub fn exp(self) -> Self {
                let w = self.val.to_array()[0];
                let pure = self.val - <$base>::ONE.scale(w);
                let dw = self.eps.to_array()[0];
                let dp = self.eps - <$base>::ONE.scale(dw);

                let theta = pure.norm();
                let exp_w = w.exp();
                let tiny = 1e-4; // cf. OctonionSimd::exp / SedenionSimd::exp.

                if theta < tiny
                {
                    Self { val: <$base>::ONE.scale(exp_w), eps: self.eps.scale(exp_w) }
                }
                else
                {
                    let (sin_t, cos_t) = (theta.sin(), theta.cos());
                    let sinc = sin_t / theta;
                    // ⟨pure, dp⟩ / θ : même règle que la dérivée de
                    // Self::norm (θ est le rayon euclidien réel de la
                    // partie pure).
                    let dtheta = (pure.0 * dp.0).reduce_sum() / theta;
                    let dsinc = (theta * cos_t - sin_t) / (theta * theta);

                    let c = exp_w * cos_t;
                    let s = exp_w * sinc;
                    let dc = exp_w * (dw * cos_t - sin_t * dtheta);
                    let ds = exp_w * (dw * sinc + dsinc * dtheta);

                    Self {
                        val: <$base>::ONE.scale(c) + pure.scale(s),
                        eps: <$base>::ONE.scale(dc) + pure.scale(ds) + dp.scale(s),
                    }
                }
            }

            /// Logarithme dual, réciproque de [`Self::exp`] restreinte à sa
            /// branche principale — même seuil/convention que
            /// [`OctonionSimd::ln`]/[`SedenionSimd::ln`]. Indéfini pour
            /// `v = 0` (comme [`Self::inverse`]).
            ///
            /// Sur la coupure de branche (`v` quasi réel négatif), la
            /// **valeur** retient déjà une direction imaginaire arbitraire
            /// (`e₁`, indépendante de `d`) : sa dérivée y est donc, comme la
            /// valeur, non rigoureusement définie — seule la partie réelle
            /// (`ln|w|`, dérivée `1/w`) est propagée, la correction `π·e₁`
            /// étant traitée comme constante (dérivée nulle).
            #[inline]
            #[must_use]
            pub fn ln(self) -> Self {
                let w = self.val.to_array()[0];
                let pure = self.val - <$base>::ONE.scale(w);
                let dw = self.eps.to_array()[0];
                let dp = self.eps - <$base>::ONE.scale(dw);

                let theta = pure.norm();
                let (o_norm, do_norm) = self.norm();
                let ln_norm = o_norm.ln();
                let dln_norm = do_norm / o_norm;
                let tiny = 1e-4; // cf. OctonionSimd::ln / SedenionSimd::ln.

                if theta < tiny
                {
                    if w >= 0.0
                    {
                        // `v` central au premier ordre : même argument que
                        // Self::exp, avec ln'(w) = 1/w.
                        Self { val: <$base>::ONE.scale(ln_norm), eps: self.eps.scale(1.0 / w) }
                    }
                    else
                    {
                        Self {
                            val: <$base>::ONE.scale(ln_norm)
                                + <$base>::unit(1).scale(core::f32::consts::PI),
                            eps: <$base>::ONE.scale(dw / w),
                        }
                    }
                }
                else
                {
                    let dtheta = (pure.0 * dp.0).reduce_sum() / theta;
                    let ratio = (w / o_norm).clamp(-1.0, 1.0);
                    let theta_acos = ratio.acos();
                    // ‖v‖² = w² + θ² exactement (décomposition réel/pur
                    // orthogonale) ⇒ √(1 − ratio²) = θ/‖v‖ : évite un second
                    // appel trigonométrique pour dérouler la dérivée d'acos.
                    let dphi = (w * do_norm - dw * o_norm) / (o_norm * theta);
                    let k = theta_acos / theta;
                    let dk = (dphi * theta - theta_acos * dtheta) / (theta * theta);

                    Self {
                        val: <$base>::ONE.scale(ln_norm) + pure.scale(k),
                        eps: <$base>::ONE.scale(dln_norm) + pure.scale(dk) + dp.scale(k),
                    }
                }
            }

            /// Puissance réelle duale `f(t) = v(t)ᵗ = exp(t·ln(v(t)))`
            /// (`t` constant, non dérivé) : simple composition de
            /// [`Self::ln`], [`Self::scale`] et [`Self::exp`], la règle de
            /// dérivation en chaîne se propageant automatiquement à travers
            /// les trois. Indéfini pour `v = 0` (via [`Self::ln`]).
            #[inline]
            #[must_use]
            pub fn powf(self, t: f32) -> Self {
                self.ln().scale(t).exp()
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
