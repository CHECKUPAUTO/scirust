//! Encrassement d'échangeur de chaleur — effet de la **résistance
//! d'encrassement** `Rf` sur le coefficient global d'échange et le
//! surdimensionnement de surface nécessaire pour le compenser.
//!
//! ```text
//! coefficient encrassé   1/Uf = 1/Uc + Rf   ⇒  Uf = 1/(1/Uc + Rf)
//! résistance mesurée     Rf   = 1/Uf − 1/Uc
//! facteur de propreté    CF   = Uf/Uc
//! surdimensionnement      OS   = 1 + Uc·Rf   (= Uc/Uf = 1/CF)
//! ```
//!
//! `Uc` coefficient global **propre** (W/(m²·K)), `Uf` coefficient global
//! **encrassé** (W/(m²·K)), `Rf` résistance d'encrassement rapportée à la
//! surface (m²·K/W), `CF` facteur de propreté (sans dimension, dans `]0, 1]`),
//! `OS` facteur de surdimensionnement de surface (sans dimension, `≥ 1`).
//!
//! **Convention** : SI cohérent. **Limite honnête** : la résistance
//! d'encrassement `Rf` est **fournie** par l'appelant (tables TEMA ou mesure ;
//! elle dépend du fluide, de la vitesse et de la température) de même que le
//! coefficient propre `Uc` ; régime **permanent** et encrassement **uniforme**
//! sur la surface. Complète [`crate::heat_exchanger`].

/// Coefficient global **encrassé** `Uf = 1/(1/Uc + Rf)` (W/(m²·K)).
///
/// Panique si `clean_coefficient <= 0` ou `fouling_resistance < 0`.
pub fn fouling_fouled_overall_coefficient(clean_coefficient: f64, fouling_resistance: f64) -> f64 {
    assert!(
        clean_coefficient > 0.0,
        "le coefficient propre doit être strictement positif"
    );
    assert!(
        fouling_resistance >= 0.0,
        "la résistance d'encrassement doit être positive ou nulle"
    );
    1.0 / (1.0 / clean_coefficient + fouling_resistance)
}

/// Résistance d'encrassement déduite des coefficients propre et encrassé
/// `Rf = 1/Uf − 1/Uc` (m²·K/W).
///
/// Panique si un coefficient est `<= 0` ou si `fouled_coefficient` dépasse
/// `clean_coefficient` (résistance négative, physiquement impossible).
pub fn fouling_resistance_from_coefficients(
    clean_coefficient: f64,
    fouled_coefficient: f64,
) -> f64 {
    assert!(
        clean_coefficient > 0.0 && fouled_coefficient > 0.0,
        "les coefficients doivent être strictement positifs"
    );
    assert!(
        fouled_coefficient <= clean_coefficient,
        "le coefficient encrassé ne peut pas dépasser le coefficient propre"
    );
    1.0 / fouled_coefficient - 1.0 / clean_coefficient
}

/// Facteur de propreté `CF = Uf/Uc` (sans dimension, dans `]0, 1]`).
///
/// Panique si un coefficient est `<= 0` ou si `fouled_coefficient` dépasse
/// `clean_coefficient`.
pub fn fouling_cleanliness_factor(fouled_coefficient: f64, clean_coefficient: f64) -> f64 {
    assert!(
        clean_coefficient > 0.0 && fouled_coefficient > 0.0,
        "les coefficients doivent être strictement positifs"
    );
    assert!(
        fouled_coefficient <= clean_coefficient,
        "le coefficient encrassé ne peut pas dépasser le coefficient propre"
    );
    fouled_coefficient / clean_coefficient
}

/// Facteur de surdimensionnement de surface `OS = 1 + Uc·Rf` (sans dimension).
///
/// Surface supplémentaire à prévoir pour conserver le même flux malgré
/// l'encrassement (`OS = Uc/Uf = 1/CF`).
///
/// Panique si `clean_coefficient <= 0` ou `fouling_resistance < 0`.
pub fn fouling_area_oversize_factor(clean_coefficient: f64, fouling_resistance: f64) -> f64 {
    assert!(
        clean_coefficient > 0.0,
        "le coefficient propre doit être strictement positif"
    );
    assert!(
        fouling_resistance >= 0.0,
        "la résistance d'encrassement doit être positive ou nulle"
    );
    1.0 + clean_coefficient * fouling_resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reciprocite_coefficient_resistance() {
        // Rf → Uf → Rf doit boucler exactement.
        let (uc, rf) = (2000.0_f64, 0.0002_f64);
        let uf = fouling_fouled_overall_coefficient(uc, rf);
        let rf_retrouve = fouling_resistance_from_coefficients(uc, uf);
        assert_relative_eq!(rf_retrouve, rf, epsilon = 1e-12);
    }

    #[test]
    fn surdimensionnement_inverse_de_la_proprete() {
        // OS = 1/CF et OS = Uc/Uf.
        let (uc, rf) = (1500.0_f64, 0.00035_f64);
        let uf = fouling_fouled_overall_coefficient(uc, rf);
        let cf = fouling_cleanliness_factor(uf, uc);
        let os = fouling_area_oversize_factor(uc, rf);
        assert_relative_eq!(os, 1.0 / cf, epsilon = 1e-12);
        assert_relative_eq!(os, uc / uf, epsilon = 1e-12);
    }

    #[test]
    fn cas_propre_sans_encrassement() {
        // Rf = 0 : Uf = Uc, CF = 1, OS = 1.
        let uc = 3200.0_f64;
        let uf = fouling_fouled_overall_coefficient(uc, 0.0);
        assert_relative_eq!(uf, uc, epsilon = 1e-9);
        assert_relative_eq!(fouling_cleanliness_factor(uf, uc), 1.0, epsilon = 1e-12);
        assert_relative_eq!(fouling_area_oversize_factor(uc, 0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cas_chiffre_eau_de_refroidissement() {
        // Uc = 2000 W/(m²·K), Rf = 0.0002 m²·K/W (ordre de grandeur TEMA).
        // 1/Uf = 0.0005 + 0.0002 = 0.0007  ⇒  Uf = 1428.571428… W/(m²·K).
        let (uc, rf) = (2000.0_f64, 0.0002_f64);
        let uf = fouling_fouled_overall_coefficient(uc, rf);
        assert_relative_eq!(uf, 1.0 / 0.0007, epsilon = 1e-9);
        // CF = 1428.5714… / 2000 = 0.7142857…
        assert_relative_eq!(
            fouling_cleanliness_factor(uf, uc),
            0.714_285_714_285_714,
            epsilon = 1e-12
        );
        // OS = 1 + 2000·0.0002 = 1.4.
        assert_relative_eq!(fouling_area_oversize_factor(uc, rf), 1.4, epsilon = 1e-12);
    }

    #[test]
    fn resistance_croissante_diminue_le_coefficient() {
        // Plus Rf est grand, plus Uf est petit (monotonie).
        let uc = 2500.0_f64;
        let uf_faible = fouling_fouled_overall_coefficient(uc, 0.0001);
        let uf_fort = fouling_fouled_overall_coefficient(uc, 0.0005);
        assert!(uf_fort < uf_faible);
    }

    #[test]
    #[should_panic(expected = "coefficient propre doit être strictement positif")]
    fn coefficient_propre_negatif_panique() {
        let _ = fouling_fouled_overall_coefficient(-100.0, 0.0002);
    }
}
