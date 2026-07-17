// scirust-simd/src/fixed/reductions.rs
//
// # Réductions virgule fixe
//
// `sum`, `dot`, `l1_norm`, `l2_norm`, `linf_norm`, `cosine_similarity`, `min`,
// `max`, `argmin`, `argmax`, génériques sur le scalaire virgule fixe via
// [`FixedReducible`].
//
// ## Reproductibilité (point clé)
//
// **L'addition virgule fixe est exacte et associative** (c'est de l'addition
// entière) : `a + b + c` donne le même bit quel que soit l'ordre, le nombre de
// lanes SIMD, l'architecture ou le nombre de threads. Il n'existe donc **qu'une
// seule** somme — déjà déterministe. Les variantes « déterministe » et « Kahan »
// des réductions flottantes n'ont **pas de sens ici** : aucune erreur d'arrondi
// à compenser dans une somme. C'est un avantage structurel de la virgule fixe,
// pas une lacune. (`dot` arrondit en revanche **chaque produit** avant sommation
// — cet arrondi est documenté et lui aussi déterministe.)
//
// * `sum` / `l1_norm` : accumulation entière exacte jusqu'au rétrécissement
//   final (enveloppant). Bit-à-bit reproductible.
// * `dot` : chaque produit arrondi vers zéro (comme `*`), puis sommé exactement.
// * `l2_norm` : `sqrt(dot(a, a))` (voir [`super::math`]).
// * `linf_norm` : `maxᵢ |aᵢ|`, parcours scalaire (ordre entier total).
// * `cosine_similarity` : `dot(a,b) / (‖a‖·‖b‖)` ; `0` si une norme est nulle.
// * `min`/`max`/`argmin`/`argmax` : parcours scalaire (ordre entier total) ;
//   `argmin`/`argmax` renvoient le **premier** indice extrémal.
//
// La voie SIMD accélère `sum`/`dot`/`l1_norm` du chemin `i32` (accumulation
// élargie `i64` en lanes) ; le chemin `i64` charge scalaire mais accumule en
// `i128` pour rester exact (pas de vecteur `i128`).

use std::simd::i32x8;
use std::simd::num::SimdInt;

use super::math::sqrt;
use super::simd::FixedI32x8;
use super::types::Fixed;

/// Scalaire virgule fixe réductible : primitives de réduction exactes/SIMD.
///
/// Implémenté pour `FixedI32<FRAC>` (SIMD `i32`) et `FixedI64<FRAC>`.
pub trait FixedReducible: Copy + Ord {
    /// Zéro.
    const ZERO: Self;
    /// Addition enveloppante.
    fn wrapping_add(self, other: Self) -> Self;
    /// Multiplication virgule fixe (arrondi vers zéro, enveloppante).
    fn wrapping_mul(self, other: Self) -> Self;
    /// Valeur absolue saturante.
    fn abs(self) -> Self;
    /// Racine carrée virgule fixe (voir [`super::math::sqrt`]).
    fn sqrt(self) -> Self;
    /// Division vérifiée (`None` si diviseur nul ou débordement).
    fn checked_div(self, other: Self) -> Option<Self>;

    /// Somme exacte (déterministe) de tout le slice.
    fn reduce_sum(data: &[Self]) -> Self;
    /// Somme des valeurs absolues (norme L1).
    fn reduce_l1(data: &[Self]) -> Self;
    /// Produit scalaire Σ (aᵢ·bᵢ), chaque produit arrondi vers zéro.
    /// Suppose `a.len() == b.len()`.
    fn reduce_dot(a: &[Self], b: &[Self]) -> Self;
}

impl<const FRAC: u32> FixedReducible for Fixed<i32, FRAC> {
    const ZERO: Self = Fixed::from_raw(0);

    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        Fixed::wrapping_add(self, other)
    }
    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        Fixed::wrapping_mul(self, other)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Fixed::abs(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt(self)
    }
    #[inline(always)]
    fn checked_div(self, other: Self) -> Option<Self> {
        Fixed::checked_div(self, other)
    }

    #[inline]
    fn reduce_sum(data: &[Self]) -> Self {
        // Accumulation par lane en i64 (exacte) puis combinaison en i128.
        let mut acc = std::simd::i64x8::splat(0);
        let mut chunks = data.chunks_exact(8);
        for chunk in chunks.by_ref()
        {
            acc += load_i32x8(chunk).cast::<i64>();
        }
        let mut total: i128 = 0;
        for lane in acc.to_array()
        {
            total += lane as i128;
        }
        for &v in chunks.remainder()
        {
            total += v.0 as i128;
        }
        Fixed::from_raw(total as i32)
    }

    #[inline]
    fn reduce_l1(data: &[Self]) -> Self {
        let mut acc = std::simd::i64x8::splat(0);
        let mut chunks = data.chunks_exact(8);
        for chunk in chunks.by_ref()
        {
            let v = FixedI32x8::<FRAC>::from_raw(load_i32x8(chunk));
            acc += v.abs().0.cast::<i64>();
        }
        let mut total: i128 = 0;
        for lane in acc.to_array()
        {
            total += lane as i128;
        }
        for &v in chunks.remainder()
        {
            total += v.abs().0 as i128;
        }
        Fixed::from_raw(total as i32)
    }

    #[inline]
    fn reduce_dot(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = std::simd::i64x8::splat(0);
        let mut ca = a.chunks_exact(8);
        let mut cb = b.chunks_exact(8);
        for (ka, kb) in ca.by_ref().zip(cb.by_ref())
        {
            let va = FixedI32x8::<FRAC>::from_raw(load_i32x8(ka));
            let vb = FixedI32x8::<FRAC>::from_raw(load_i32x8(kb));
            acc += (va * vb).0.cast::<i64>();
        }
        let mut total: i128 = 0;
        for lane in acc.to_array()
        {
            total += lane as i128;
        }
        for (&x, &y) in ca.remainder().iter().zip(cb.remainder())
        {
            total += x.wrapping_mul(y).0 as i128;
        }
        Fixed::from_raw(total as i32)
    }
}

impl<const FRAC: u32> FixedReducible for Fixed<i64, FRAC> {
    const ZERO: Self = Fixed::from_raw(0);

    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        Fixed::wrapping_add(self, other)
    }
    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        Fixed::wrapping_mul(self, other)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Fixed::abs(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt(self)
    }
    #[inline(always)]
    fn checked_div(self, other: Self) -> Option<Self> {
        Fixed::checked_div(self, other)
    }

    #[inline]
    fn reduce_sum(data: &[Self]) -> Self {
        // Pas de vecteur i128 : accumulation exacte en i128 scalaire.
        let mut total: i128 = 0;
        for &v in data
        {
            total += v.0 as i128;
        }
        Fixed::from_raw(total as i64)
    }

    #[inline]
    fn reduce_l1(data: &[Self]) -> Self {
        let mut total: i128 = 0;
        for &v in data
        {
            total += v.abs().0 as i128;
        }
        Fixed::from_raw(total as i64)
    }

    #[inline]
    fn reduce_dot(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        let mut total: i128 = 0;
        for (&x, &y) in a.iter().zip(b.iter())
        {
            total += x.wrapping_mul(y).0 as i128;
        }
        Fixed::from_raw(total as i64)
    }
}

/// Charge 8 scalaires `FixedI32<FRAC>` contigus en un `i32x8` de bruts.
#[inline(always)]
fn load_i32x8<const FRAC: u32>(chunk: &[Fixed<i32, FRAC>]) -> i32x8 {
    let mut raw = [0i32; 8];
    for (slot, x) in raw.iter_mut().zip(chunk)
    {
        *slot = x.0;
    }
    i32x8::from_array(raw)
}

// ------------------------------------------------------------------ //
//  API publique générique                                             //
// ------------------------------------------------------------------ //

/// Somme exacte et **déterministe** (bit-à-bit reproductible) du slice.
/// Enveloppe au rétrécissement final si le total dépasse la plage.
#[inline]
#[must_use]
pub fn sum<T: FixedReducible>(data: &[T]) -> T {
    T::reduce_sum(data)
}

/// Norme L1 `Σ |aᵢ|` (exacte, déterministe).
#[inline]
#[must_use]
pub fn l1_norm<T: FixedReducible>(data: &[T]) -> T {
    T::reduce_l1(data)
}

/// Produit scalaire `Σ aᵢ·bᵢ` (chaque produit arrondi vers zéro puis sommé
/// exactement). Panique si `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn dot<T: FixedReducible>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "dot: longueurs différentes");
    T::reduce_dot(a, b)
}

/// Norme L2 au carré `Σ aᵢ²` (= `dot(a, a)`).
#[inline]
#[must_use]
pub fn l2_norm_sqr<T: FixedReducible>(a: &[T]) -> T {
    T::reduce_dot(a, a)
}

/// Norme L2 (euclidienne) `√Σ aᵢ²`.
#[inline]
#[must_use]
pub fn l2_norm<T: FixedReducible>(a: &[T]) -> T {
    T::reduce_dot(a, a).sqrt()
}

/// Norme L∞ (maximum absolu) `maxᵢ |aᵢ|`, ou `T::ZERO` si `data` est vide
/// (convention : norme du vecteur nul). Exacte, déterministe.
#[inline]
#[must_use]
pub fn linf_norm<T: FixedReducible>(data: &[T]) -> T {
    data.iter()
        .copied()
        .map(FixedReducible::abs)
        .reduce(|a, b| if a >= b { a } else { b })
        .unwrap_or(T::ZERO)
}

/// Similarité cosinus `⟨a,b⟩ / (‖a‖·‖b‖)`. Renvoie `0` si une norme est nulle.
/// Panique si `a.len() != b.len()`.
#[inline]
#[must_use]
pub fn cosine_similarity<T: FixedReducible>(a: &[T], b: &[T]) -> T {
    let dot_ab = dot(a, b);
    let denom = l2_norm(a).wrapping_mul(l2_norm(b));
    if denom == T::ZERO
    {
        T::ZERO
    }
    else
    {
        dot_ab.checked_div(denom).unwrap_or(T::ZERO)
    }
}

/// Minimum du slice, ou `None` si vide. Déterministe.
#[inline]
#[must_use]
pub fn min<T: FixedReducible>(data: &[T]) -> Option<T> {
    data.iter()
        .copied()
        .reduce(|a, b| if a <= b { a } else { b })
}

/// Maximum du slice, ou `None` si vide. Déterministe.
#[inline]
#[must_use]
pub fn max<T: FixedReducible>(data: &[T]) -> Option<T> {
    data.iter()
        .copied()
        .reduce(|a, b| if a >= b { a } else { b })
}

/// Indice du **premier** minimum, ou `None` si vide.
#[inline]
#[must_use]
pub fn argmin<T: FixedReducible>(data: &[T]) -> Option<usize> {
    let mut best: Option<(usize, T)> = None;
    for (i, &v) in data.iter().enumerate()
    {
        if best.is_none_or(|(_, b)| v < b)
        {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

/// Indice du **premier** maximum, ou `None` si vide.
#[inline]
#[must_use]
pub fn argmax<T: FixedReducible>(data: &[T]) -> Option<usize> {
    let mut best: Option<(usize, T)> = None;
    for (i, &v) in data.iter().enumerate()
    {
        if best.is_none_or(|(_, b)| v > b)
        {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}
