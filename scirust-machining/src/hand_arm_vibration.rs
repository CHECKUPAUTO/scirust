//! Exposition aux vibrations **main-bras** (HAV) : valeur journalière normalisée
//! `A(8)`, dose partielle et durée avant d'atteindre une valeur limite.
//!
//! ```text
//! exposition journalière   A(8) = a·√(T/8)
//! dose partielle           P    = a²·T
//! A(8) combiné             A(8) = √( Σ Pᵢ / 8 )
//! durée avant limite       T_L  = 8·(a_L/a)²
//! ```
//!
//! `a` valeur d'accélération vibratoire pondérée en fréquence (somme vectorielle
//! ISO 5349, m/s²), `T` durée d'exposition effective (h), `A(8)` valeur
//! d'exposition journalière normalisée sur 8 h (m/s²), `P` dose partielle
//! (m²·s⁻⁴·h) permettant de cumuler plusieurs outils, `a_L` valeur d'action ou
//! limite réglementaire (m/s²), `T_L` durée d'exposition à `a` avant d'atteindre
//! `a_L` (h).
//!
//! **Convention** : accélération en m/s², durées en heures (le `8` du dénominateur
//! est la période de référence de 8 h). **Limite honnête** : la magnitude
//! vibratoire pondérée `a` est **fournie** par l'appelant (mesure ISO 5349 sur
//! l'outil réel) ; la normalisation est purement énergétique sur 8 h et ne
//! modélise ni la transmissibilité des gants ni la posture. Les valeurs d'action
//! et limite usuelles (2,5 et 5 m/s²) sont **fournies** par la réglementation
//! applicable — jamais présumées ici.

/// Valeur d'exposition journalière normalisée `A(8) = a·√(T/8)` (m/s²).
///
/// Panique si `vibration_magnitude < 0` ou `exposure_duration_hours < 0`.
pub fn hav_daily_exposure_a8(vibration_magnitude: f64, exposure_duration_hours: f64) -> f64 {
    assert!(
        vibration_magnitude >= 0.0 && exposure_duration_hours >= 0.0,
        "a ≥ 0 et T ≥ 0 requis"
    );
    vibration_magnitude * (exposure_duration_hours / 8.0_f64).sqrt()
}

/// Dose partielle `P = a²·T` (m²·s⁻⁴·h), sommable entre outils.
///
/// Panique si `vibration_magnitude < 0` ou `exposure_duration_hours < 0`.
pub fn hav_partial_exposure(vibration_magnitude: f64, exposure_duration_hours: f64) -> f64 {
    assert!(
        vibration_magnitude >= 0.0 && exposure_duration_hours >= 0.0,
        "a ≥ 0 et T ≥ 0 requis"
    );
    vibration_magnitude * vibration_magnitude * exposure_duration_hours
}

/// `A(8)` combiné de plusieurs outils `A(8) = √( Σ Pᵢ / 8 )` (m/s²),
/// où chaque `Pᵢ` provient de [`hav_partial_exposure`].
///
/// Panique si `partial_exposures` est vide ou contient une dose `< 0`.
pub fn hav_combined_a8(partial_exposures: &[f64]) -> f64 {
    assert!(
        !partial_exposures.is_empty(),
        "au moins une dose partielle requise"
    );
    assert!(
        partial_exposures.iter().all(|&p| p >= 0.0),
        "chaque dose partielle P ≥ 0 requise"
    );
    let sum: f64 = partial_exposures.iter().sum();
    (sum / 8.0_f64).sqrt()
}

/// Durée d'exposition à `a` avant d'atteindre la valeur limite fournie
/// `T_L = 8·(a_L/a)²` (h).
///
/// Panique si `vibration_magnitude <= 0` ou `limit_value < 0`.
pub fn hav_time_to_limit(vibration_magnitude: f64, limit_value: f64) -> f64 {
    assert!(
        vibration_magnitude > 0.0 && limit_value >= 0.0,
        "a > 0 et a_L ≥ 0 requis"
    );
    let ratio = limit_value / vibration_magnitude;
    8.0 * ratio * ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn a8_over_full_reference_period_equals_magnitude() {
        // T = 8 h ⇒ A(8) = a·√1 = a (cas limite de la période de référence).
        assert_relative_eq!(hav_daily_exposure_a8(4.2, 8.0), 4.2, max_relative = 1e-12);
    }

    #[test]
    fn a8_realistic_case() {
        // a = 5 m/s² pendant 2 h : A(8) = 5·√(2/8) = 5·0,5 = 2,5 m/s².
        assert_relative_eq!(hav_daily_exposure_a8(5.0, 2.0), 2.5, max_relative = 1e-12);
    }

    #[test]
    fn a8_proportional_to_magnitude() {
        // Pour une durée fixée, A(8) ∝ a : doubler a double A(8).
        let single = hav_daily_exposure_a8(3.0, 4.0);
        let doubled = hav_daily_exposure_a8(6.0, 4.0);
        assert_relative_eq!(doubled / single, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn combined_of_single_tool_equals_direct_a8() {
        // Un seul outil : √(P/8) avec P = a²·T redonne a·√(T/8).
        let (a, t) = (5.0, 2.0);
        let partial = hav_partial_exposure(a, t);
        assert_relative_eq!(
            hav_combined_a8(&[partial]),
            hav_daily_exposure_a8(a, t),
            max_relative = 1e-12
        );
    }

    #[test]
    fn combined_of_two_tools_matches_energy_sum() {
        // Deux outils : A(8) = √((a1²·T1 + a2²·T2)/8).
        let p1 = hav_partial_exposure(5.0, 2.0); // 50
        let p2 = hav_partial_exposure(3.0, 1.0); // 9
        let expected = ((50.0_f64 + 9.0) / 8.0).sqrt();
        assert_relative_eq!(hav_combined_a8(&[p1, p2]), expected, max_relative = 1e-12);
    }

    #[test]
    fn time_to_limit_inverts_daily_exposure() {
        // À T = T_L, l'exposition journalière vaut exactement la limite (réciprocité).
        let (a, limit) = (5.0, 2.5);
        let t_l = hav_time_to_limit(a, limit); // 8·(0,5)² = 2 h
        assert_relative_eq!(t_l, 2.0, max_relative = 1e-12);
        assert_relative_eq!(hav_daily_exposure_a8(a, t_l), limit, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "a > 0")]
    fn zero_magnitude_time_to_limit_panics() {
        hav_time_to_limit(0.0, 2.5);
    }
}
