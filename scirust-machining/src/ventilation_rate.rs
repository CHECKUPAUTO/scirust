//! **Ventilation d'atelier** — taux de renouvellement d'air (ACH), débit
//! nécessaire pour atteindre un taux visé, et débit de dilution d'un polluant.
//!
//! ```text
//! renouvellements horaires   ACH = Q / V
//! débit pour un ACH visé      Q   = V · ACH
//! débit de dilution           Q   = safety · G / C
//! ```
//!
//! `Q` débit volumique d'air (m³·h⁻¹), `V` volume du local (m³),
//! `ACH` renouvellements d'air par heure (h⁻¹), `G` débit de génération du
//! polluant (m³·h⁻¹ de vapeur, ou toute unité de « quantité par heure »),
//! `C` concentration cible admissible (fraction volumique, m³·m⁻³, homogène à
//! `G`/`Q`), `safety` facteur de sécurité sans dimension (≥ 1).
//!
//! **Convention** : unités cohérentes ; débits et volumes en système horaire
//! (m³·h⁻¹, m³). `G` et `C` doivent être exprimés dans la même base pour que
//! `G/C` ait la dimension d'un débit.
//!
//! **Limite honnête** : hypothèse de **mélange parfait** (concentration
//! uniforme et instantanée dans tout le local) et de **régime permanent**. Le
//! débit de génération `G`, la concentration cible `C`, le facteur de sécurité
//! `safety`, les débits et les volumes sont **fournis par l'appelant** ; aucune
//! valeur limite d'exposition, aucun taux réglementaire ni aucun débit « par
//! défaut » n'est inventé ici.

/// Taux de renouvellement d'air `ACH = Q / V` (renouvellements par heure).
///
/// `volumetric_flow_m3h` et `room_volume_m3` en unités horaires cohérentes
/// (m³·h⁻¹ et m³) ; le résultat est en h⁻¹.
///
/// Panique si `room_volume_m3 <= 0` ou `volumetric_flow_m3h < 0`.
pub fn ach_air_changes_per_hour(volumetric_flow_m3h: f64, room_volume_m3: f64) -> f64 {
    assert!(
        room_volume_m3 > 0.0,
        "le volume du local doit être strictement positif"
    );
    assert!(
        volumetric_flow_m3h >= 0.0,
        "le débit volumique doit être positif ou nul"
    );
    volumetric_flow_m3h / room_volume_m3
}

/// Débit d'air nécessaire pour atteindre un taux de renouvellement visé
/// `Q = V · ACH`.
///
/// `room_volume_m3` en m³, `target_ach` en h⁻¹ ; le résultat est en m³·h⁻¹.
/// Réciproque de [`ach_air_changes_per_hour`].
///
/// Panique si `room_volume_m3 <= 0` ou `target_ach < 0`.
pub fn ach_required_flow(room_volume_m3: f64, target_ach: f64) -> f64 {
    assert!(
        room_volume_m3 > 0.0,
        "le volume du local doit être strictement positif"
    );
    assert!(
        target_ach >= 0.0,
        "le taux de renouvellement visé doit être positif ou nul"
    );
    room_volume_m3 * target_ach
}

/// Débit d'air de **dilution** d'un polluant `Q = safety · G / C`.
///
/// `generation_rate` débit de génération du polluant (m³·h⁻¹ de vapeur, ou
/// toute quantité par heure), `target_concentration` concentration cible
/// admissible (même base que `G`/`Q`, p. ex. fraction volumique), `safety_factor`
/// facteur de sécurité sans dimension (≥ 1). Résultat en m³·h⁻¹ d'air neuf.
///
/// Panique si `generation_rate < 0`, `target_concentration <= 0` ou
/// `safety_factor < 1`.
pub fn vent_dilution_flow_for_contaminant(
    generation_rate: f64,
    target_concentration: f64,
    safety_factor: f64,
) -> f64 {
    assert!(
        generation_rate >= 0.0,
        "le débit de génération du polluant doit être positif ou nul"
    );
    assert!(
        target_concentration > 0.0,
        "la concentration cible doit être strictement positive"
    );
    assert!(
        safety_factor >= 1.0,
        "le facteur de sécurité doit être supérieur ou égal à 1"
    );
    safety_factor * generation_rate / target_concentration
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ach_and_required_flow_are_reciprocal() {
        // V·ACH doit redonner Q, et Q/V doit redonner ACH.
        let v = 300.0_f64;
        let q = 1800.0_f64;
        let ach = ach_air_changes_per_hour(q, v);
        assert_relative_eq!(ach_required_flow(v, ach), q, max_relative = 1e-12);
        assert_relative_eq!(ach_air_changes_per_hour(ach_required_flow(v, 6.0), v), 6.0);
    }

    #[test]
    fn ach_realistic_case() {
        // Local de 300 m³ ventilé à 1800 m³/h → 6 renouvellements/h.
        assert_relative_eq!(ach_air_changes_per_hour(1800.0, 300.0), 6.0);
        // Réciproquement, viser 6 ACH dans 300 m³ demande 1800 m³/h.
        assert_relative_eq!(ach_required_flow(300.0, 6.0), 1800.0);
    }

    #[test]
    fn required_flow_proportional_to_volume_and_ach() {
        // Q ∝ V (à ACH fixé) et Q ∝ ACH (à V fixé).
        let base = ach_required_flow(300.0, 6.0);
        assert_relative_eq!(ach_required_flow(600.0, 6.0), 2.0 * base);
        assert_relative_eq!(ach_required_flow(300.0, 12.0), 2.0 * base);
    }

    #[test]
    fn dilution_flow_realistic_case() {
        // G = 0,05 m³/h de vapeur, C = 1e-4 (100 ppm), safety = 3
        // → Q = 3·0,05/1e-4 = 1500 m³/h d'air neuf.
        assert_relative_eq!(
            vent_dilution_flow_for_contaminant(0.05, 1e-4, 3.0),
            1500.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn dilution_flow_scales_with_safety_and_generation() {
        // Q ∝ safety, ∝ G, et ∝ 1/C.
        let q = vent_dilution_flow_for_contaminant(0.05, 1e-4, 2.0);
        assert_relative_eq!(vent_dilution_flow_for_contaminant(0.05, 1e-4, 4.0), 2.0 * q);
        assert_relative_eq!(vent_dilution_flow_for_contaminant(0.10, 1e-4, 2.0), 2.0 * q);
        assert_relative_eq!(
            vent_dilution_flow_for_contaminant(0.05, 0.5e-4, 2.0),
            2.0 * q
        );
    }

    #[test]
    fn zero_generation_needs_no_dilution() {
        // Aucun polluant généré → débit de dilution nul.
        assert_relative_eq!(vent_dilution_flow_for_contaminant(0.0, 1e-4, 3.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "concentration cible")]
    fn zero_target_concentration_panics() {
        vent_dilution_flow_for_contaminant(0.05, 0.0, 3.0);
    }
}
