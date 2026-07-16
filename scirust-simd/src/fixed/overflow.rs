// scirust-simd/src/fixed/overflow.rs
//
// # Politiques d'overflow
//
// Une opération virgule fixe peut produire un résultat hors de la plage du
// type de stockage. Trois politiques explicites sont offertes, exposées à la
// fois comme méthodes dédiées (`wrapping_*`, `checked_*`, `saturating_*`) et
// comme mode runtime [`OverflowMode`] pour les API paramétrées.
//
// ⚠️ **Défaut des opérateurs** (`+`, `-`, `*`, `/`, `-x`) : **enveloppe**
// ([`OverflowMode::Wrap`]). Ce choix diffère volontairement de l'arithmétique
// entière de Rust (qui **panique en debug** et enveloppe en release) : un
// panic dépendant du profil de compilation violerait le déterminisme
// bit-à-bit exigé par SciRust. L'enveloppe est déterministe quel que soit le
// profil ; les variantes vérifiée/saturante restent accessibles explicitement.

use super::repr::FixedStorage;

/// Politique appliquée quand un résultat déborde le type de stockage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum OverflowMode {
    /// Tronque modulo 2^bits (déterministe, sans branche). Défaut des opérateurs.
    #[default]
    Wrap,
    /// Renvoie `None` en cas de débordement (voir [`super::Fixed`] `checked_*`).
    Checked,
    /// Sature à `[MIN, MAX]`.
    Saturate,
}

/// Rétrécit un accumulateur élargi selon `mode`.
///
/// `None` uniquement en mode [`OverflowMode::Checked`] débordant ; les modes
/// `Wrap`/`Saturate` renvoient toujours `Some`.
#[inline(always)]
pub(crate) fn narrow<I: FixedStorage>(wide: I::Wide, mode: OverflowMode) -> Option<I> {
    match mode
    {
        OverflowMode::Wrap => Some(I::from_wide_wrapping(wide)),
        OverflowMode::Checked => I::from_wide_checked(wide),
        OverflowMode::Saturate => Some(I::from_wide_saturating(wide)),
    }
}
