// scirust-simd/src/fixed/convert.rs
//
// # Conversions
//
// | Conversion | Sémantique | Pertes / échec |
// |---|---|---|
// | `From<i32>` → `FixedI32<F>` | valeur entière → fixe | **saturante** hors plage (documenté) |
// | `From<i64>` → `FixedI64<F>` | idem | saturante |
// | `TryFrom<f32/f64>` → `Fixed` | `round_ties_even(v·2^F)` | `Err` si NaN/∞/hors plage ; arrondi sinon |
// | `From<Fixed>` → `f32`/`f64` | `raw / 2^F` | perte au-delà de 24 (f32) / 53 (f64) bits |
//
// `From<i32>`/`From<i64>` saturent plutôt que d'envelopper : corrompre
// silencieusement une donnée scientifique par enveloppe serait pire qu'une
// saturation visible. Les variantes `from_int_wrapping` / `from_int_checked`
// existent pour les autres besoins.

use super::repr::FixedStorage;
use super::rounding::RoundingMode;
use super::types::Fixed;

/// Erreur de conversion flottant → virgule fixe : la valeur est NaN, infinie,
/// ou hors de la plage représentable après arrondi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TryFromFloatError;

impl core::fmt::Display for TryFromFloatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "valeur flottante non représentable en virgule fixe (NaN, ∞ ou hors plage)"
        )
    }
}

impl core::error::Error for TryFromFloatError {}

// -------- entier → fixe (saturant) -------- //

impl<const FRAC: u32> From<i32> for Fixed<i32, FRAC> {
    /// Valeur entière → Q_FRAC, **saturante** si hors plage.
    #[inline]
    fn from(value: i32) -> Self {
        Self::from_int_saturating(value)
    }
}

impl<const FRAC: u32> From<i64> for Fixed<i64, FRAC> {
    /// Valeur entière → Q_FRAC, **saturante** si hors plage.
    #[inline]
    fn from(value: i64) -> Self {
        Self::from_int_saturating(value)
    }
}

// -------- flottant → fixe (faillible, arrondi au pair) -------- //

impl<I: FixedStorage, const FRAC: u32> TryFrom<f64> for Fixed<I, FRAC> {
    type Error = TryFromFloatError;
    /// `round_ties_even(value · 2^FRAC)`. `Err` si NaN/∞/hors plage.
    #[inline]
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::from_f64(value, RoundingMode::NearestEven).ok_or(TryFromFloatError)
    }
}

impl<I: FixedStorage, const FRAC: u32> TryFrom<f32> for Fixed<I, FRAC> {
    type Error = TryFromFloatError;
    /// `value as f64` (sans perte) puis [`TryFrom<f64>`](Self).
    #[inline]
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::from_f64(value as f64, RoundingMode::NearestEven).ok_or(TryFromFloatError)
    }
}

// -------- fixe → flottant (Into via From) -------- //

impl<I: FixedStorage, const FRAC: u32> From<Fixed<I, FRAC>> for f64 {
    /// `raw / 2^FRAC`. Sans perte tant que la magnitude tient sur 53 bits.
    #[inline]
    fn from(value: Fixed<I, FRAC>) -> f64 {
        value.to_f64()
    }
}

impl<I: FixedStorage, const FRAC: u32> From<Fixed<I, FRAC>> for f32 {
    /// `raw / 2^FRAC`. Perte au-delà de 24 bits significatifs.
    #[inline]
    fn from(value: Fixed<I, FRAC>) -> f32 {
        value.to_f32()
    }
}
