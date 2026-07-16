// scirust-simd/src/fixed/rescale.rs
//
// # Changement de résolution virgule fixe (requantification)
//
// [`rescale`] convertit `Fixed<I, FROM>` en `Fixed<I, TO>` (même stockage `I`,
// fraction différente) en préservant la **valeur réelle représentée**. C'est
// la primitive de requantification entre étages d'un pipeline quantifié —
// par exemple ajuster la résolution d'une activation avant de l'injecter dans
// [`super::layer::Linear`] suivant, sans jamais passer par un flottant.
//
// ## Méthode
//
// `raw_to = raw_from · 2^(TO−FROM)`, calculé dans l'accumulateur élargi
// ([`super::repr::WideInt`]) déjà utilisé pour la multiplication/division :
//
// * `TO ≥ FROM` (résolution **plus fine**) : décalage à gauche exact, aucune
//   perte — les bits ajoutés sont des zéros.
// * `TO < FROM` (résolution **plus grossière**) : décalage à droite **arrondi**
//   ([`super::rounding::round_shift`], même politique explicite que la
//   multiplication).
//
// Le rétrécissement final vers `I` applique la politique d'overflow explicite
// ([`super::overflow::OverflowMode`]) : augmenter `FRAC` réduit le nombre de
// bits entiers disponibles, donc une valeur de grande magnitude peut ne plus
// tenir dans le nouveau format — enveloppée, vérifiée ou saturée selon le
// choix de l'appelant, jamais tronquée silencieusement sans politique.

use super::overflow::{OverflowMode, narrow};
use super::repr::{FixedStorage, WideInt};
use super::rounding::{RoundingMode, round_shift};
use super::types::Fixed;

/// Convertit `Fixed<I, FROM>` en `Fixed<I, TO>`, en préservant la valeur
/// réelle. `None` uniquement en overflow [`OverflowMode::Checked`] débordant.
#[inline]
#[must_use]
pub fn rescale<I: FixedStorage, const FROM: u32, const TO: u32>(
    x: Fixed<I, FROM>,
    rounding: RoundingMode,
    overflow: OverflowMode,
) -> Option<Fixed<I, TO>> {
    let wide = x.to_raw().to_wide();
    let shifted = if TO >= FROM
    {
        wide.shl(TO - FROM)
    }
    else
    {
        round_shift(wide, FROM - TO, rounding)
    };
    narrow::<I>(shifted, overflow).map(Fixed::from_raw)
}

/// [`rescale`] enveloppant (troncature vers zéro à la réduction de résolution).
#[inline]
#[must_use]
pub fn rescale_wrapping<I: FixedStorage, const FROM: u32, const TO: u32>(
    x: Fixed<I, FROM>,
) -> Fixed<I, TO> {
    rescale(x, RoundingMode::TowardZero, OverflowMode::Wrap).expect("Wrap ne renvoie jamais None")
}

/// [`rescale`] saturant (troncature vers zéro à la réduction de résolution).
#[inline]
#[must_use]
pub fn rescale_saturating<I: FixedStorage, const FROM: u32, const TO: u32>(
    x: Fixed<I, FROM>,
) -> Fixed<I, TO> {
    rescale(x, RoundingMode::TowardZero, OverflowMode::Saturate)
        .expect("Saturate ne renvoie jamais None")
}
