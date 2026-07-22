// scirust-simd/src/geometry/curves.rs
//
// # Courbes d'interpolation — Bézier cubique & Catmull–Rom
//
// Splines paramétriques **génériques sur le scalaire et sur la dimension**
// (`[T; D]` : 2D, 3D, ou plus) : les mêmes formules servent `f32`, `f64` et la
// virgule fixe déterministe. Utile pour lisser une trajectoire de robot, une
// caméra ou tout chemin plan/spatial de façon reproductible bit-à-bit.
//
// ## Bézier cubique
//
// [`bezier_cubic`] évalue `B(u) = (1−u)³·p₀ + 3(1−u)²u·p₁ + 3(1−u)u²·p₂ +
// u³·p₃`, `u ∈ [0, 1]` : la courbe part de `p₀` (`u=0`), arrive à `p₃` (`u=1`)
// et est **tangente** aux segments `p₀p₁` et `p₂p₃` aux extrémités.
// [`bezier_cubic_tangent`] en donne la dérivée `B′(u)` (le vecteur vitesse,
// utile pour orienter un mobile le long de la courbe). Purement algébriques
// (aucune transcendante) — disponibles dès [`NumericScalar`].
//
// ## Catmull–Rom
//
// [`catmull_rom`] évalue une spline de Catmull–Rom sur le segment central
// `[p₁, p₂]` à partir de quatre points de contrôle `p₀, p₁, p₂, p₃` : elle
// **passe par** `p₁` (`u=0`) et `p₂` (`u=1`), avec des tangentes déduites des
// voisins (`(p₂−p₀)/2` en `p₁`, `(p₃−p₁)/2` en `p₂`), d'où une courbe `C¹`
// **interpolante** (contrairement à Bézier/B-spline, qui n'atteignent pas
// leurs points de contrôle intermédiaires). Le facteur `1/2` demande
// [`RealScalar`] (`recip`).

use crate::fixed::{NumericScalar, RealScalar};

/// Interpolation linéaire par composante `(1−u)·a + u·b` d'un point `[T; D]`.
#[inline]
fn lerp<T: NumericScalar, const D: usize>(a: [T; D], b: [T; D], u: T) -> [T; D] {
    let mut out = [T::zero(); D];
    let one_minus = T::one() - u;
    for k in 0..D
    {
        out[k] = one_minus * a[k] + u * b[k];
    }
    out
}

/// Bézier cubique `B(u)` aux points de contrôle `p0..p3`, `u ∈ [0, 1]`
/// (cf. en-tête de module). Évalué par une chaîne de [`lerp`] (l'algorithme de
/// De Casteljau), numériquement stable et sans puissances explicites.
#[must_use]
pub fn bezier_cubic<T: NumericScalar, const D: usize>(
    p0: [T; D],
    p1: [T; D],
    p2: [T; D],
    p3: [T; D],
    u: T,
) -> [T; D] {
    // De Casteljau : trois niveaux d'interpolation linéaire.
    let a = lerp(p0, p1, u);
    let b = lerp(p1, p2, u);
    let c = lerp(p2, p3, u);
    let d = lerp(a, b, u);
    let e = lerp(b, c, u);
    lerp(d, e, u)
}

/// Dérivée `B′(u) = 3(1−u)²(p₁−p₀) + 6(1−u)u(p₂−p₁) + 3u²(p₃−p₂)` de la Bézier
/// cubique — le vecteur tangent (vitesse) à la courbe (cf. en-tête de module).
#[must_use]
pub fn bezier_cubic_tangent<T: NumericScalar, const D: usize>(
    p0: [T; D],
    p1: [T; D],
    p2: [T; D],
    p3: [T; D],
    u: T,
) -> [T; D] {
    let one_minus = T::one() - u;
    let three = T::from_i32(3);
    let six = T::from_i32(6);
    let w0 = three * one_minus * one_minus;
    let w1 = six * one_minus * u;
    let w2 = three * u * u;
    let mut out = [T::zero(); D];
    for k in 0..D
    {
        out[k] = w0 * (p1[k] - p0[k]) + w1 * (p2[k] - p1[k]) + w2 * (p3[k] - p2[k]);
    }
    out
}

/// Spline de Catmull–Rom sur le segment central `[p1, p2]`, `u ∈ [0, 1]`
/// (cf. en-tête de module) : passe exactement par `p1` (`u=0`) et `p2` (`u=1`).
///
/// Base : `C(u) = ½·[ 2p₁ + (p₂−p₀)u + (2p₀−5p₁+4p₂−p₃)u² +
/// (−p₀+3p₁−3p₂+p₃)u³ ]` (variante uniforme, tension standard).
#[must_use]
pub fn catmull_rom<T: RealScalar, const D: usize>(
    p0: [T; D],
    p1: [T; D],
    p2: [T; D],
    p3: [T; D],
    u: T,
) -> [T; D] {
    let half = T::from_i32(2).recip();
    let (two, three, four, five) = (
        T::from_i32(2),
        T::from_i32(3),
        T::from_i32(4),
        T::from_i32(5),
    );
    let u2 = u * u;
    let u3 = u2 * u;
    let mut out = [T::zero(); D];
    for k in 0..D
    {
        let c0 = two * p1[k];
        let c1 = (p2[k] - p0[k]) * u;
        let c2 = (two * p0[k] - five * p1[k] + four * p2[k] - p3[k]) * u2;
        let c3 = (-p0[k] + three * p1[k] - three * p2[k] + p3[k]) * u3;
        out[k] = half * (c0 + c1 + c2 + c3);
    }
    out
}
