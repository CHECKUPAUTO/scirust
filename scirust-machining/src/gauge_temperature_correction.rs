//! Correction thermique des mesures dimensionnelles rapportées à la
//! température de référence de 20 °C (ISO 1).
//!
//! ```text
//! longueur corrigée      Lc = L·(1 + αp·(20 − Tp) − αs·(20 − Ts))
//! erreur différentielle   e  = L·(αp − αs)·(T − 20)
//! ramenée à la référence  L20 = L·(1 + α·(20 − T))
//! décalage / référence    Δ  = L·α·(T − 20)
//! ```
//!
//! `L` longueur mesurée (m), `Lc` longueur corrigée (m), `L20` longueur à
//! 20 °C (m), `αp`/`αs` coefficients de dilatation linéaire de la pièce et de
//! l'étalon/instrument (1/K), `Tp`/`Ts`/`T` températures (°C), `e`/`Δ`
//! écarts de longueur (m). La référence est fixée à 20 °C (ISO 1).
//!
//! **Limite honnête** : correction **linéaire** au premier ordre autour de
//! 20 °C, suppose l'**équilibre thermique** (pièce et instrument stabilisés).
//! Les coefficients de dilatation `α` sont **fournis par l'appelant** ; aucune
//! valeur matériau n'est supposée « par défaut ». Complète [`crate::thermal`].

/// Température de référence dimensionnelle normalisée (°C, ISO 1).
pub const GAUGETEMP_REFERENCE_TEMPERATURE_C: f64 = 20.0;

/// Longueur corrigée `Lc = L·(1 + αp·(20 − Tp) − αs·(20 − Ts))` (m).
///
/// Ramène une mesure `measured_length` (pièce à `part_temp_c`, lue sur un
/// instrument à `scale_temp_c`) à la référence 20 °C, en tenant compte des
/// dilatations respectives de la pièce (`part_expansion`) et de l'échelle
/// (`scale_expansion`).
///
/// Panique si `measured_length <= 0` ou si une température est <= −273,15 °C.
pub fn gaugetemp_corrected_length(
    measured_length: f64,
    part_expansion: f64,
    scale_expansion: f64,
    part_temp_c: f64,
    scale_temp_c: f64,
) -> f64 {
    assert!(
        measured_length > 0.0,
        "la longueur mesurée doit être strictement positive"
    );
    assert!(
        part_temp_c > -273.15 && scale_temp_c > -273.15,
        "les températures doivent être supérieures au zéro absolu"
    );
    let r = GAUGETEMP_REFERENCE_TEMPERATURE_C;
    measured_length
        * (1.0 + part_expansion * (r - part_temp_c) - scale_expansion * (r - scale_temp_c))
}

/// Erreur de dilatation différentielle `e = L·(αp − αs)·(T − 20)` (m).
///
/// Écart de mesure dû à la différence de dilatation entre la pièce
/// (`part_expansion`) et l'étalon (`scale_expansion`) à la température
/// commune `temperature_c` ; nul à 20 °C ou lorsque `αp = αs`.
///
/// Panique si `length <= 0` ou si `temperature_c <= −273,15 °C`.
pub fn gaugetemp_differential_expansion_error(
    length: f64,
    part_expansion: f64,
    scale_expansion: f64,
    temperature_c: f64,
) -> f64 {
    assert!(length > 0.0, "la longueur doit être strictement positive");
    assert!(
        temperature_c > -273.15,
        "la température doit être supérieure au zéro absolu"
    );
    length
        * (part_expansion - scale_expansion)
        * (temperature_c - GAUGETEMP_REFERENCE_TEMPERATURE_C)
}

/// Longueur d'une pièce ramenée à 20 °C `L20 = L·(1 + α·(20 − T))` (m).
///
/// Corrige une mesure `length` d'une pièce de coefficient `alpha_per_k` prise
/// à `temperature_c` pour la rapporter à la référence 20 °C (matériau unique).
///
/// Panique si `length <= 0` ou si `temperature_c <= −273,15 °C`.
pub fn dimcorr_to_reference_length(length: f64, alpha_per_k: f64, temperature_c: f64) -> f64 {
    assert!(length > 0.0, "la longueur doit être strictement positive");
    assert!(
        temperature_c > -273.15,
        "la température doit être supérieure au zéro absolu"
    );
    length * (1.0 + alpha_per_k * (GAUGETEMP_REFERENCE_TEMPERATURE_C - temperature_c))
}

/// Décalage de longueur par rapport à la référence `Δ = L·α·(T − 20)` (m).
///
/// Allongement (positif) ou raccourcissement (négatif) d'une pièce de
/// longueur `length` et de coefficient `alpha_per_k` par rapport à sa cote à
/// 20 °C, à la température `temperature_c`.
///
/// Panique si `length <= 0` ou si `temperature_c <= −273,15 °C`.
pub fn dimcorr_thermal_offset(length: f64, alpha_per_k: f64, temperature_c: f64) -> f64 {
    assert!(length > 0.0, "la longueur doit être strictement positive");
    assert!(
        temperature_c > -273.15,
        "la température doit être supérieure au zéro absolu"
    );
    length * alpha_per_k * (temperature_c - GAUGETEMP_REFERENCE_TEMPERATURE_C)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn no_correction_at_reference_temperature() {
        // Pièce et instrument à 20 °C : la longueur corrigée égale la mesure.
        let l = 0.1;
        assert_relative_eq!(
            gaugetemp_corrected_length(l, 23e-6, 11.5e-6, 20.0, 20.0),
            l,
            epsilon = 1e-15
        );
    }

    #[test]
    fn identical_coefficients_and_temperatures_cancel() {
        // αp = αs et Tp = Ts : les corrections se compensent exactement.
        let l = 0.25;
        assert_relative_eq!(
            gaugetemp_corrected_length(l, 12e-6, 12e-6, 45.0, 45.0),
            l,
            epsilon = 1e-15
        );
    }

    #[test]
    fn corrected_length_matches_differential_error() {
        // Lorsque Tp = Ts = T : Lc = L − e (identité algébrique).
        let l = 0.1;
        let (ap, as_, t) = (23e-6, 11.5e-6, 30.0);
        let corrected = gaugetemp_corrected_length(l, ap, as_, t, t);
        let err = gaugetemp_differential_expansion_error(l, ap, as_, t);
        assert_relative_eq!(corrected, l - err, epsilon = 1e-15);
    }

    #[test]
    fn differential_error_realistic_case() {
        // Pièce aluminium (23e-6/K) mesurée sur étalon acier (11,5e-6/K),
        // L = 0,1 m à 30 °C : e = 0,1·(23e-6 − 11,5e-6)·(30 − 20)
        //                       = 0,1·11,5e-6·10 = 1,15e-5 m = 11,5 µm.
        assert_relative_eq!(
            gaugetemp_differential_expansion_error(0.1, 23e-6, 11.5e-6, 30.0),
            1.15e-5,
            epsilon = 1e-18
        );
    }

    #[test]
    fn differential_error_proportional_to_length() {
        // e est linéaire en L : doubler la longueur double l'erreur.
        let e1 = gaugetemp_differential_expansion_error(0.05, 23e-6, 11.5e-6, 35.0);
        let e2 = gaugetemp_differential_expansion_error(0.10, 23e-6, 11.5e-6, 35.0);
        assert_relative_eq!(e2, 2.0 * e1, epsilon = 1e-18);
    }

    #[test]
    fn reference_length_and_offset_are_consistent() {
        // L20 = L + Δ à signe près : L·(1 + α·(20 − T)) = L − L·α·(T − 20).
        let (l, alpha, t) = (0.2, 11.5e-6, 40.0);
        let l20 = dimcorr_to_reference_length(l, alpha, t);
        let offset = dimcorr_thermal_offset(l, alpha, t);
        assert_relative_eq!(l20, l - offset, epsilon = 1e-15);
        // À 40 °C l'acier s'est dilaté : la cote à 20 °C est plus petite.
        assert!(l20 < l);
    }

    #[test]
    #[should_panic(expected = "la longueur mesurée doit être strictement positive")]
    fn corrected_length_rejects_nonpositive_length() {
        let _ = gaugetemp_corrected_length(0.0, 12e-6, 12e-6, 25.0, 25.0);
    }
}
