// scirust-simd/src/fixed/activation.rs
//
// # Fonctions d'activation déterministes (inférence quantifiée)
//
// Activations élément-par-élément pour vecteurs virgule fixe (ou flottants),
// compagnes naturelles du GEMM déterministe [`super::linalg`] : appliquées à la
// sortie d'une couche linéaire, elles complètent une inférence **quantifiée
// reproductible bit-à-bit**.
//
// Deux familles selon les primitives requises :
//
// * **Exactes** (`relu`, `relu6`, `clamp`, `hardtanh`, `leaky_relu`) : bornées
//   aux opérations d'anneau ordonné [`NumericScalar`] — donc `f32`, `f64` **et**
//   tout [`Fixed`] (y compris le stockage `i16` audio). Aucune approximation :
//   ce sont des min/max et des combinaisons affines exactes en virgule fixe.
// * **Douces « hard »** (`hardsigmoid`, `hardswish`) : nécessitent la division
//   par 6, donc l'inverse [`RealScalar::recip`] — restreintes à `f32`, `f64` et
//   `FixedI32<FRAC>`. Ce sont les variantes linéaires par morceaux de la
//   sigmoïde et de la swish (MobileNetV3), sans transcendante, donc rapides et
//   déterministes.
//
// Toutes sont **sans branche imprévisible côté données scientifiques** : le
// résultat ne dépend que de la valeur d'entrée, jamais de l'ordre ou du
// parallélisme.

use super::traits::{NumericScalar, RealScalar};

/// Maximum de deux scalaires (retourne `a` en cas d'égalité).
#[inline(always)]
fn max2<T: NumericScalar>(a: T, b: T) -> T {
    if a >= b { a } else { b }
}

/// Minimum de deux scalaires (retourne `a` en cas d'égalité).
#[inline(always)]
fn min2<T: NumericScalar>(a: T, b: T) -> T {
    if a <= b { a } else { b }
}

/// `clamp(x, lo, hi)` : restreint `x` à `[lo, hi]`. Suppose `lo <= hi`.
#[inline]
#[must_use]
pub fn clamp<T: NumericScalar>(x: T, lo: T, hi: T) -> T {
    min2(max2(x, lo), hi)
}

/// `ReLU(x) = max(x, 0)` — exact.
#[inline]
#[must_use]
pub fn relu<T: NumericScalar>(x: T) -> T {
    max2(x, T::zero())
}

/// `ReLU6(x) = clamp(x, 0, 6)` — exact (borne haute des réseaux quantifiés).
#[inline]
#[must_use]
pub fn relu6<T: NumericScalar>(x: T) -> T {
    clamp(x, T::zero(), T::from_i32(6))
}

/// `Hardtanh(x) = clamp(x, lo, hi)` — exact. Défaut usuel : `lo = -1`, `hi = 1`.
#[inline]
#[must_use]
pub fn hardtanh<T: NumericScalar>(x: T, lo: T, hi: T) -> T {
    clamp(x, lo, hi)
}

/// `LeakyReLU(x) = x` si `x ≥ 0`, sinon `slope·x` — exact.
#[inline]
#[must_use]
pub fn leaky_relu<T: NumericScalar>(x: T, slope: T) -> T {
    if x >= T::zero() { x } else { slope * x }
}

/// `HardSigmoid(x) = clamp(x/6 + 1/2, 0, 1)`.
///
/// Variante linéaire par morceaux de la sigmoïde (nulle sous `−3`, unité
/// au-dessus de `3`, affine entre). En virgule fixe, `1/6` et `1/2` sont
/// approchés par [`RealScalar::recip`] (erreur bornée à la résolution).
#[inline]
#[must_use]
pub fn hardsigmoid<T: RealScalar>(x: T) -> T {
    let sixth = T::from_i32(6).recip();
    let half = T::from_i32(2).recip();
    clamp(x * sixth + half, T::zero(), T::one())
}

/// `HardSwish(x) = x · HardSigmoid(x)`.
///
/// Variante linéaire par morceaux de la swish (MobileNetV3) : nulle sous `−3`,
/// identité au-dessus de `3`, `x·(x+3)/6` entre.
#[inline]
#[must_use]
pub fn hardswish<T: RealScalar>(x: T) -> T {
    x * hardsigmoid(x)
}

/// Applique une activation **en place** à tout un slice (sortie de couche).
///
/// `f` est appliquée élément par élément ; l'ordre n'affecte pas le résultat
/// (activation ponctuelle). Ergonomique sur la sortie de [`super::linalg`] :
/// `apply_inplace(&mut y, relu)`.
#[inline]
pub fn apply_inplace<T: Copy>(data: &mut [T], f: impl Fn(T) -> T) {
    for v in data.iter_mut()
    {
        *v = f(*v);
    }
}
