//! Métrologie — **planéité** et **rectitude** estimées par la méthode
//! min-max (enveloppe) : l'erreur de forme est approchée par l'**étendue** des
//! relevés palpés sur un plan (ou une ligne) de référence.
//!
//! ```text
//! erreur de planéité   E_plan  = max(readings) − min(readings)   (étendue des écarts)
//! erreur de rectitude  E_rect  = max(readings) − min(readings)   (même règle, ligne 1D)
//! conformité           conforme ⇔ (max − min) ≤ tolerance
//! ```
//!
//! `readings` écarts signés de chaque point palpé par rapport au plan (ou à la
//! ligne) de référence (m) ; `E_plan`, `E_rect` erreurs de forme estimées (m) ;
//! `tolerance` largeur de la zone de tolérance de forme spécifiée au dessin (m).
//!
//! **Convention** : SI cohérent ; il suffit que tous les relevés et la tolérance
//! partagent la même unité pour une étendue et une comparaison correctes.
//! **Limite honnête** : l'étendue `max − min` est une **approximation par
//! excès** de la vraie zone minimale (l'enveloppe par moindres carrés ou par
//! zone minimale donne une valeur **≤** à celle-ci). Les relevés sont supposés
//! déjà exprimés par rapport à un **plan (ou une ligne) de référence** ; ce
//! module ne réajuste pas le plan des moindres carrés. Aucune tolérance de forme
//! n'est imposée : elle est **fournie par l'appelant** (jamais de « défaut »
//! inventé).

/// Étendue `max(readings) − min(readings)` (m) d'un jeu de relevés, socle commun
/// aux erreurs de planéité et de rectitude par la méthode min-max.
///
/// Panique si `readings` est vide ou contient une valeur non finie.
fn range_of(readings: &[f64]) -> f64 {
    assert!(
        !readings.is_empty(),
        "il faut au moins un relevé pour estimer l'erreur de forme"
    );
    assert!(
        readings.iter().all(|r| r.is_finite()),
        "tous les relevés doivent être finis"
    );
    let max = readings.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = readings.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

/// Erreur de planéité `E_plan = max(readings) − min(readings)` (m), approximation
/// par l'étendue de la zone de tolérance enveloppant les points palpés.
///
/// Approximation par excès : la vraie zone minimale est `≤` à cette valeur.
///
/// Panique si `readings` est vide ou contient une valeur non finie.
pub fn flatness_error(readings: &[f64]) -> f64 {
    range_of(readings)
}

/// Erreur de rectitude `E_rect = max(readings) − min(readings)` (m) le long d'une
/// ligne, obtenue par la même méthode min-max que la planéité.
///
/// Approximation par excès : la vraie zone minimale est `≤` à cette valeur.
///
/// Panique si `readings` est vide ou contient une valeur non finie.
pub fn straightness_error(readings: &[f64]) -> f64 {
    range_of(readings)
}

/// Conformité de planéité : renvoie `true` ssi l'étendue des relevés tient dans
/// la zone `tolerance`, soit `flatness_error(readings) <= tolerance`.
///
/// Panique si `readings` est vide, contient une valeur non finie, ou si
/// `tolerance < 0`.
pub fn flatness_is_within(readings: &[f64], tolerance: f64) -> bool {
    assert!(
        tolerance >= 0.0,
        "la tolérance de planéité ne peut être négative"
    );
    flatness_error(readings) <= tolerance
}

/// Conformité de rectitude : renvoie `true` ssi l'étendue des relevés tient dans
/// la zone `tolerance`, soit `straightness_error(readings) <= tolerance`.
///
/// Panique si `readings` est vide, contient une valeur non finie, ou si
/// `tolerance < 0`.
pub fn straightness_is_within(readings: &[f64], tolerance: f64) -> bool {
    assert!(
        tolerance >= 0.0,
        "la tolérance de rectitude ne peut être négative"
    );
    straightness_error(readings) <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flatness_equals_straightness_on_same_readings() {
        // Les deux estimateurs partagent la règle min-max : mêmes relevés,
        // même étendue.
        let readings = [0.0, 1.2e-5, -3.0e-6, 8.0e-6, -1.0e-5];
        assert_relative_eq!(
            flatness_error(&readings),
            straightness_error(&readings),
            max_relative = 1e-12
        );
    }

    #[test]
    fn flatness_is_max_minus_min() {
        // Cas chiffré : relevés d'un comparateur (µm) sur une glissière rectifiée.
        // max = +12 µm, min = −5 µm → étendue = 17 µm.
        let readings = [0.0, 12e-6, -5e-6, 7e-6, 3e-6];
        assert_relative_eq!(flatness_error(&readings), 17e-6, max_relative = 1e-9);
    }

    #[test]
    fn perfect_plane_has_zero_error() {
        // Cas limite : tous les points au même niveau → erreur nulle.
        let readings = [4.0e-6, 4.0e-6, 4.0e-6];
        assert_relative_eq!(flatness_error(&readings), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn error_is_translation_invariant() {
        // L'étendue ne dépend que des écarts relatifs : décaler tous les relevés
        // d'un même offset (rezérotage du plan) ne change pas l'erreur.
        let readings = [0.0, 9.0e-6, -4.0e-6, 2.0e-6];
        let offset = 3.3e-5;
        let shifted: Vec<f64> = readings.iter().map(|r| r + offset).collect();
        assert_relative_eq!(
            flatness_error(&readings),
            flatness_error(&shifted),
            max_relative = 1e-12
        );
    }

    #[test]
    fn error_scales_linearly() {
        // Homogénéité de degré 1 : multiplier les écarts par k multiplie
        // l'étendue par k.
        let readings = [0.0, 6.0e-6, -2.0e-6, 5.0e-6];
        let k = 2.5_f64;
        let scaled: Vec<f64> = readings.iter().map(|r| r * k).collect();
        assert_relative_eq!(
            straightness_error(&scaled),
            k * straightness_error(&readings),
            max_relative = 1e-12
        );
    }

    #[test]
    fn within_matches_error_boundary() {
        // Cohérence stricte entre le booléen et l'erreur estimée : conforme à
        // l'égalité, refusé juste en dessous.
        let readings = [0.0, 12e-6, -5e-6, 7e-6];
        let e = flatness_error(&readings); // 17 µm
        assert!(flatness_is_within(&readings, e));
        assert!(flatness_is_within(&readings, e + 1e-9));
        assert!(!flatness_is_within(&readings, e - 1e-9));
        // La rectitude suit exactement la même frontière.
        assert!(straightness_is_within(&readings, e));
        assert!(!straightness_is_within(&readings, e - 1e-9));
    }

    #[test]
    #[should_panic(expected = "au moins un relevé")]
    fn empty_readings_panics() {
        flatness_error(&[]);
    }
}
