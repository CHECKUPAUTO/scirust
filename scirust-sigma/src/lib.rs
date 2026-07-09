//! `scirust-sigma` — bornes structurelles σ (« couvercle de zéro ») par régime
//! numérique déterministe.
//!
//! ## Principe
//!
//! Chaque représentation numérique discrète possède un **σ** = le plus petit
//! élément strictement positif représentable *dans sa voie*. C'est le
//! « couvercle de zéro » du régime : en dessous de σ il ne reste que le zéro
//! (ou des valeurs non portables).
//!
//! Toute garde contre zéro — un dénominateur minoré, un `.max(ε)`, un seuil
//! `if x.abs() < ε` — placée **sous** le σ de sa voie n'est pas une garde :
//!
//! - sur la voie f32 *sanitized* (voie 3, GPU), tout `|x| < f32::MIN_POSITIVE`
//!   est écrasé à `0.0` par `sanitize_f32` — la garde est alors morte ;
//! - sur matériel GPU, les sous-normaux dépendent du réglage FTZ/DAZ du driver
//!   — une garde qui repose dessus n'est pas portable.
//!
//! L'invariant central est [`is_valid_guard_f32`] : une garde n'est licite que
//! si elle est **≥ σ** de son régime.
//!
//! ## Table des σ
//!
//! | Régime | σ | Valeur |
//! |---|---|---|
//! | Entier exact | `1` | [`SIGMA_INTEGER_EXACT`] |
//! | Fixe Q15.16 | `2⁻¹⁶` | [`SIGMA_Q15_16`] |
//! | Fixe Q31.32 | `2⁻³²` | [`SIGMA_Q31_32`] |
//! | f32 *sanitized* (voie 3, GPU) | plus petit **normal** | [`SIGMA_SANITIZED_F32`] |
//! | f32 brut (CPU, sous-normaux vivants) | plus petit **sous-normal** | [`SIGMA_RAW_F32`] |
//! | f64 brut (CPU, sous-normaux vivants) | plus petit **sous-normal** | [`SIGMA_RAW_F64`] |
//!
//! ## Contrat consommé
//!
//! Le seuil de `sanitize_f32` dans `scirust-gpu/src/deterministic.rs`
//! (= `f32::MIN_POSITIVE`) EST le σ de la voie sanitized. L'alignement des deux
//! est vérifié par `tests/sanitize_alignment.rs` : ce test casse si l'un des
//! deux seuils bouge sans l'autre.
//!
//! ## Zéro dépendance
//!
//! Cette bibliothèque n'utilise que `std`. Le binaire `epsilon-audit` (audit
//! lexical des ~2 600 littéraux epsilon du workspace) vit dans la même crate
//! mais n'ajoute aucune dépendance à la bibliothèque.

#![forbid(unsafe_code)]

/// Régime numérique déterministe. À chaque voie correspond un σ distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SigmaRegime {
    /// Voie 1 : arithmétique entière pure (crypto Zq). σ = 1.
    IntegerExact,
    /// Voie 2 : virgule fixe Q15.16. σ = 2⁻¹⁶.
    FixedQ15_16,
    /// Voie 2 : virgule fixe Q31.32. σ = 2⁻³².
    FixedQ31_32,
    /// Voie 3 : f32 + Kahan + `sanitize_f32`. σ = plus petit f32 **normal**.
    SanitizedF32,
    /// f32 brut CPU (sous-normaux vivants). σ = plus petit f32 **sous-normal**.
    RawF32,
    /// f64 brut CPU (sous-normaux vivants). σ = plus petit f64 **sous-normal**.
    RawF64,
}

/// σ de l'arithmétique entière exacte (voie 1) : `1`.
pub const SIGMA_INTEGER_EXACT: i64 = 1;

/// σ de la voie 3 (f32 *sanitized*) : plus petit f32 **normal**.
/// Bit-à-bit `0x0080_0000` ≈ `1.1754944e-38`.
pub const SIGMA_SANITIZED_F32: f32 = f32::MIN_POSITIVE;

/// σ du f32 brut : plus petit f32 **sous-normal**.
/// Bit-à-bit `0x0000_0001` ≈ `1.4e-45`.
pub const SIGMA_RAW_F32: f32 = f32::from_bits(1);

/// σ du f64 brut : plus petit f64 **sous-normal**.
/// Bit-à-bit `0x0000_0000_0000_0001` ≈ `4.9e-324`.
pub const SIGMA_RAW_F64: f64 = f64::from_bits(1);

/// σ de la virgule fixe Q15.16 : `2⁻¹⁶ = 1 / 65536`.
pub const SIGMA_Q15_16: f64 = 1.0 / 65536.0;

/// σ de la virgule fixe Q31.32 : `2⁻³² = 1 / 4294967296`.
pub const SIGMA_Q31_32: f64 = 1.0 / 4294967296.0;

/// σ du régime exprimé en `f32`, ou `None` si le type ne s'applique pas.
///
/// Seul [`SigmaRegime::RawF64`] renvoie `None` : son σ (plus petit sous-normal
/// f64) s'annule (underflow) une fois converti en f32, donc aucune valeur f32
/// ne le représente. Pour ce régime, utiliser [`sigma_f64`].
pub fn sigma_f32(regime: SigmaRegime) -> Option<f32> {
    match regime
    {
        SigmaRegime::IntegerExact => Some(SIGMA_INTEGER_EXACT as f32),
        SigmaRegime::FixedQ15_16 => Some(SIGMA_Q15_16 as f32),
        SigmaRegime::FixedQ31_32 => Some(SIGMA_Q31_32 as f32),
        SigmaRegime::SanitizedF32 => Some(SIGMA_SANITIZED_F32),
        SigmaRegime::RawF32 => Some(SIGMA_RAW_F32),
        SigmaRegime::RawF64 => None,
    }
}

/// σ du régime exprimé en `f64`. Tous les régimes ont un σ représentable en
/// f64 (le type est toujours `Option` par symétrie avec [`sigma_f32`], mais ne
/// renvoie jamais `None` ici).
pub fn sigma_f64(regime: SigmaRegime) -> Option<f64> {
    match regime
    {
        SigmaRegime::IntegerExact => Some(SIGMA_INTEGER_EXACT as f64),
        SigmaRegime::FixedQ15_16 => Some(SIGMA_Q15_16),
        SigmaRegime::FixedQ31_32 => Some(SIGMA_Q31_32),
        SigmaRegime::SanitizedF32 => Some(SIGMA_SANITIZED_F32 as f64),
        SigmaRegime::RawF32 => Some(SIGMA_RAW_F32 as f64),
        SigmaRegime::RawF64 => Some(SIGMA_RAW_F64),
    }
}

/// Minore `x` par le σ du régime : `x.max(σ)`. Garde de dénominateur canonique
/// — le résultat est toujours ≥ σ, donc jamais écrasé sur la voie sanitized.
///
/// Comportements de bord (choisis, testés, sans surprise silencieuse) :
/// - `x` négatif ou `0.0` → renvoie σ (la garde remonte au plancher) ;
/// - `x` NaN → renvoie σ (`f32::max` renvoie l'argument non-NaN) ;
/// - régime sans σ f32 ([`SigmaRegime::RawF64`]) → renvoie `x` inchangé (aucun
///   plancher f32 n'existe pour ce régime ; utiliser [`guard_denominator_f64`]).
pub fn guard_denominator_f32(x: f32, regime: SigmaRegime) -> f32 {
    match sigma_f32(regime)
    {
        Some(sigma) => x.max(sigma),
        None => x,
    }
}

/// Équivalent f64 de [`guard_denominator_f32`]. Tous les régimes possèdent un σ
/// f64, donc le plancher est toujours appliqué.
pub fn guard_denominator_f64(x: f64, regime: SigmaRegime) -> f64 {
    match sigma_f64(regime)
    {
        Some(sigma) => x.max(sigma),
        None => x,
    }
}

/// Invariant central : une garde f32 est licite ssi elle est ≥ σ du régime.
///
/// - Une garde `< σ` (ou `0.0`, ou négative) est **morte** sur la voie
///   sanitized : `sanitize_f32` l'écraserait. → `false`.
/// - Une garde NaN n'est jamais licite. → `false`.
/// - Régime sans σ f32 ([`SigmaRegime::RawF64`]) : l'API f32 ne s'applique pas.
///   → `false` (utiliser [`is_valid_guard_f64`]).
pub fn is_valid_guard_f32(guard: f32, regime: SigmaRegime) -> bool {
    match sigma_f32(regime)
    {
        Some(sigma) => guard >= sigma,
        None => false,
    }
}

/// Équivalent f64 de [`is_valid_guard_f32`].
///
/// Une garde NaN ou `< σ` renvoie `false`. Tous les régimes ont un σ f64, donc
/// il n'y a pas de cas « type inapplicable ».
pub fn is_valid_guard_f64(guard: f64, regime: SigmaRegime) -> bool {
    match sigma_f64(regime)
    {
        Some(sigma) => guard >= sigma,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Valeurs bit-à-bit exactes des σ ---

    #[test]
    fn sigma_bit_patterns_are_exact() {
        assert_eq!(SIGMA_SANITIZED_F32.to_bits(), 0x0080_0000);
        assert_eq!(SIGMA_RAW_F32.to_bits(), 0x0000_0001);
        assert_eq!(SIGMA_RAW_F64.to_bits(), 0x0000_0000_0000_0001);
        // 2⁻¹⁶ : exposant biaisé 1023-16 = 1007 = 0x3EF, mantisse nulle.
        assert_eq!(SIGMA_Q15_16.to_bits(), 0x3EF0_0000_0000_0000);
        // 2⁻³² : exposant biaisé 1023-32 = 991 = 0x3DF, mantisse nulle.
        assert_eq!(SIGMA_Q31_32.to_bits(), 0x3DF0_0000_0000_0000);
    }

    #[test]
    fn sigma_raw_f32_is_below_sanitized() {
        // Monotonie vérifiée à la COMPILATION : le plancher sous-normal est
        // strictement sous le plancher normal. Pour deux f32 positifs, l'ordre
        // des valeurs = l'ordre des bits. C'est POURQUOI une garde valide sur la
        // voie brute peut être morte sur la voie sanitized.
        const { assert!(SIGMA_RAW_F32.to_bits() < SIGMA_SANITIZED_F32.to_bits()) };
        const { assert!(SIGMA_RAW_F32.to_bits() > 0) };
    }

    // --- sigma_f32 / sigma_f64 : régimes applicables ou non ---

    #[test]
    fn sigma_f32_none_only_for_raw_f64() {
        assert!(sigma_f32(SigmaRegime::RawF64).is_none());
        for regime in [
            SigmaRegime::IntegerExact,
            SigmaRegime::FixedQ15_16,
            SigmaRegime::FixedQ31_32,
            SigmaRegime::SanitizedF32,
            SigmaRegime::RawF32,
        ]
        {
            assert!(
                sigma_f32(regime).is_some(),
                "{regime:?} devrait avoir un σ f32"
            );
        }
    }

    #[test]
    fn sigma_f64_is_always_some() {
        for regime in [
            SigmaRegime::IntegerExact,
            SigmaRegime::FixedQ15_16,
            SigmaRegime::FixedQ31_32,
            SigmaRegime::SanitizedF32,
            SigmaRegime::RawF32,
            SigmaRegime::RawF64,
        ]
        {
            assert!(
                sigma_f64(regime).is_some(),
                "{regime:?} devrait avoir un σ f64"
            );
        }
    }

    // --- Invariant central is_valid_guard_f32 ---

    #[test]
    fn valid_guard_above_sigma_is_accepted() {
        // 1e-30 est très en dessous de 1.0 mais TRÈS au-dessus de σ (1.17e-38).
        assert!(is_valid_guard_f32(1e-30, SigmaRegime::SanitizedF32));
        assert!(is_valid_guard_f32(
            SIGMA_SANITIZED_F32,
            SigmaRegime::SanitizedF32
        ));
    }

    #[test]
    fn dead_guard_below_sigma_is_rejected() {
        // 1e-40 est un sous-normal f32 : sous σ → garde morte.
        assert!(!is_valid_guard_f32(1e-40, SigmaRegime::SanitizedF32));
        // Le plus grand sous-normal (σ_sanitized / 2) reste sous σ.
        assert!(!is_valid_guard_f32(
            SIGMA_SANITIZED_F32 / 2.0,
            SigmaRegime::SanitizedF32
        ));
    }

    #[test]
    fn guard_edge_values_are_rejected() {
        assert!(!is_valid_guard_f32(0.0, SigmaRegime::SanitizedF32));
        assert!(!is_valid_guard_f32(-1.0, SigmaRegime::SanitizedF32));
        assert!(!is_valid_guard_f32(f32::NAN, SigmaRegime::SanitizedF32));
    }

    #[test]
    fn raw_f64_regime_has_no_f32_guard() {
        // Sans σ f32, aucune garde f32 n'est déclarée licite (utiliser la f64).
        assert!(!is_valid_guard_f32(1e-10, SigmaRegime::RawF64));
        assert!(is_valid_guard_f64(1e-10, SigmaRegime::RawF64));
    }

    // --- guard_denominator : comportements de bord définis ---

    #[test]
    fn guard_denominator_lifts_zero_negative_and_nan_to_sigma() {
        let sigma = SIGMA_SANITIZED_F32;
        assert_eq!(guard_denominator_f32(0.0, SigmaRegime::SanitizedF32), sigma);
        assert_eq!(
            guard_denominator_f32(-5.0, SigmaRegime::SanitizedF32),
            sigma
        );
        // f32::max(NaN, σ) == σ : le NaN est remonté au plancher, pas propagé.
        assert_eq!(
            guard_denominator_f32(f32::NAN, SigmaRegime::SanitizedF32),
            sigma
        );
    }

    #[test]
    fn guard_denominator_passes_values_above_sigma_through() {
        assert_eq!(guard_denominator_f32(1.0, SigmaRegime::SanitizedF32), 1.0);
        assert_eq!(guard_denominator_f64(2.5, SigmaRegime::FixedQ15_16), 2.5);
    }

    #[test]
    fn guard_denominator_raw_f64_regime_is_identity_in_f32() {
        // Pas de σ f32 pour RawF64 → la valeur ressort inchangée.
        assert_eq!(guard_denominator_f32(0.0, SigmaRegime::RawF64), 0.0);
        assert_eq!(guard_denominator_f32(-3.0, SigmaRegime::RawF64), -3.0);
        // La variante f64, elle, applique bien le plancher.
        assert_eq!(
            guard_denominator_f64(0.0, SigmaRegime::RawF64),
            SIGMA_RAW_F64
        );
    }

    #[test]
    fn integer_and_fixed_regimes_have_expected_sigma() {
        assert_eq!(sigma_f64(SigmaRegime::IntegerExact), Some(1.0));
        assert_eq!(sigma_f64(SigmaRegime::FixedQ15_16), Some(1.0 / 65536.0));
        assert_eq!(
            sigma_f64(SigmaRegime::FixedQ31_32),
            Some(1.0 / 4294967296.0)
        );
    }
}
