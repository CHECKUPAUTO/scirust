// scirust-simd/src/fixed/repr.rs
//
// # Plomberie entière générique : `FixedStorage` et `WideInt`
//
// Ces deux traits internes abstraient l'entier de stockage (`i32`, `i64`, …) et
// son type « élargi » (`i64`, `i128`) utilisé comme accumulateur exact pour la
// multiplication et la division en virgule fixe. Grâce à eux, tout le sous-
// système virgule fixe est écrit **une seule fois**, générique sur le stockage
// (`Fixed<I, FRAC>`), sans macro de duplication ni `unsafe`.
//
// Un produit de deux `i32` tient exactement dans un `i64` ; un produit de deux
// `i64` tient exactement dans un `i128`. C'est l'invariant clé qui rend la
// multiplication virgule fixe exacte avant arrondi.

use core::fmt::{Debug, Display};
use core::hash::Hash;

/// Entier élargi servant d'accumulateur exact (produit/dividende décalé).
///
/// Implémenté pour `i32` (élargi de `i16`), `i64` (élargi de `i32`) et `i128`
/// (élargi de `i64`). `Display` est requis pour le formatage décimal exact de
/// [`super::Fixed`].
pub trait WideInt: Copy + Ord + Debug + Display {
    /// 0 élargi.
    const ZERO: Self;
    /// 1 élargi.
    const ONE: Self;

    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    /// Décalage à gauche (les bits sortants sont perdus — sûr ici car les
    /// valeurs restent dans la plage exacte de l'accumulateur).
    fn shl(self, n: u32) -> Self;
    /// Décalage arithmétique à droite (arrondi vers −∞).
    fn shr(self, n: u32) -> Self;
    /// Division tronquée vers zéro (sémantique `/` de Rust).
    fn div_trunc(self, other: Self) -> Self;
    /// Bit de poids faible (parité) — utile à l'arrondi au pair le plus proche.
    fn is_odd(self) -> bool;
}

macro_rules! impl_wide_int {
    ($ty:ty) => {
        impl WideInt for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;

            #[inline(always)]
            fn wrapping_add(self, other: Self) -> Self {
                <$ty>::wrapping_add(self, other)
            }
            #[inline(always)]
            fn wrapping_sub(self, other: Self) -> Self {
                <$ty>::wrapping_sub(self, other)
            }
            #[inline(always)]
            fn wrapping_mul(self, other: Self) -> Self {
                <$ty>::wrapping_mul(self, other)
            }
            #[inline(always)]
            fn shl(self, n: u32) -> Self {
                self << n
            }
            #[inline(always)]
            fn shr(self, n: u32) -> Self {
                // `>>` sur entier signé = décalage arithmétique (arrondi −∞).
                self >> n
            }
            #[inline(always)]
            fn div_trunc(self, other: Self) -> Self {
                self / other
            }
            #[inline(always)]
            fn is_odd(self) -> bool {
                self & 1 != 0
            }
        }
    };
}

impl_wide_int!(i32);
impl_wide_int!(i64);
impl_wide_int!(i128);

/// Entier de stockage d'un nombre en virgule fixe.
///
/// Fournit l'élargissement exact ([`WideInt`]), l'arithmétique enveloppante /
/// vérifiée / saturante, le rétrécissement depuis l'accumulateur élargi selon
/// une politique d'overflow, et les conversions flottantes.
///
/// Implémenté pour `i16` (élargi `i32`), `i32` (élargi `i64`) et `i64` (élargi
/// `i128`).
pub trait FixedStorage: Copy + Ord + Eq + Hash + Debug {
    /// Accumulateur exact pour produit/division.
    type Wide: WideInt;

    /// Zéro.
    const ZERO: Self;
    /// Un (au sens entier brut, PAS la valeur 1.0 en virgule fixe).
    const ONE: Self;
    /// Borne inférieure entière.
    const MIN: Self;
    /// Borne supérieure entière.
    const MAX: Self;
    /// Nombre de bits de la représentation (32, 64).
    const BITS: u32;

    /// Élargit exactement vers l'accumulateur.
    fn to_wide(self) -> Self::Wide;

    /// Rétrécit l'accumulateur en tronquant les bits de poids fort (enveloppe).
    fn from_wide_wrapping(wide: Self::Wide) -> Self;
    /// Rétrécit si et seulement si la valeur tient dans `[MIN, MAX]`.
    fn from_wide_checked(wide: Self::Wide) -> Option<Self>;
    /// Rétrécit en saturant à `[MIN, MAX]`.
    fn from_wide_saturating(wide: Self::Wide) -> Self;

    fn wrapping_neg(self) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_shl(self, n: u32) -> Self;
    fn checked_neg(self) -> Option<Self>;
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn saturating_neg(self) -> Self;
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_sub(self, other: Self) -> Self;
    /// `|self|` saturé (`MIN` ↦ `MAX`, car `−MIN` déborde).
    fn saturating_abs(self) -> Self;
    /// `|self|` vérifié (`None` pour `MIN`).
    fn checked_abs(self) -> Option<Self>;

    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;
    /// Convertit un `f64` en entier de stockage, **saturant** (`as` de Rust :
    /// hors plage → borne, NaN → 0). L'appelant garantit déjà la plage.
    fn from_f64_saturating(value: f64) -> Self;
    /// Idem depuis un `f32`.
    fn from_f32_saturating(value: f32) -> Self;
}

macro_rules! impl_fixed_storage {
    ($int:ty, $wide:ty) => {
        impl FixedStorage for $int {
            type Wide = $wide;

            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN: Self = <$int>::MIN;
            const MAX: Self = <$int>::MAX;
            const BITS: u32 = <$int>::BITS;

            #[inline(always)]
            fn to_wide(self) -> $wide {
                self as $wide
            }
            #[inline(always)]
            fn from_wide_wrapping(wide: $wide) -> Self {
                wide as $int
            }
            #[inline(always)]
            fn from_wide_checked(wide: $wide) -> Option<Self> {
                if wide >= <$int>::MIN as $wide && wide <= <$int>::MAX as $wide
                {
                    Some(wide as $int)
                }
                else
                {
                    None
                }
            }
            #[inline(always)]
            fn from_wide_saturating(wide: $wide) -> Self {
                if wide < <$int>::MIN as $wide
                {
                    <$int>::MIN
                }
                else if wide > <$int>::MAX as $wide
                {
                    <$int>::MAX
                }
                else
                {
                    wide as $int
                }
            }
            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                <$int>::wrapping_neg(self)
            }
            #[inline(always)]
            fn wrapping_add(self, other: Self) -> Self {
                <$int>::wrapping_add(self, other)
            }
            #[inline(always)]
            fn wrapping_sub(self, other: Self) -> Self {
                <$int>::wrapping_sub(self, other)
            }
            #[inline(always)]
            fn wrapping_shl(self, n: u32) -> Self {
                <$int>::wrapping_shl(self, n)
            }
            #[inline(always)]
            fn checked_neg(self) -> Option<Self> {
                <$int>::checked_neg(self)
            }
            #[inline(always)]
            fn checked_add(self, other: Self) -> Option<Self> {
                <$int>::checked_add(self, other)
            }
            #[inline(always)]
            fn checked_sub(self, other: Self) -> Option<Self> {
                <$int>::checked_sub(self, other)
            }
            #[inline(always)]
            fn saturating_neg(self) -> Self {
                <$int>::saturating_neg(self)
            }
            #[inline(always)]
            fn saturating_add(self, other: Self) -> Self {
                <$int>::saturating_add(self, other)
            }
            #[inline(always)]
            fn saturating_sub(self, other: Self) -> Self {
                <$int>::saturating_sub(self, other)
            }
            #[inline(always)]
            fn saturating_abs(self) -> Self {
                <$int>::saturating_abs(self)
            }
            #[inline(always)]
            fn checked_abs(self) -> Option<Self> {
                <$int>::checked_abs(self)
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn to_f32(self) -> f32 {
                self as f32
            }
            #[inline(always)]
            fn from_f64_saturating(value: f64) -> Self {
                value as $int
            }
            #[inline(always)]
            fn from_f32_saturating(value: f32) -> Self {
                value as $int
            }
        }
    };
}

impl_fixed_storage!(i16, i32);
impl_fixed_storage!(i32, i64);
impl_fixed_storage!(i64, i128);
