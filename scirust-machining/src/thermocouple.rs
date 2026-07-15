//! Thermocouple (effet Seebeck) — force électromotrice, mesure de température et
//! compensation de soudure froide dans l'approximation linéaire du coefficient de Seebeck.
//!
//! ```text
//! f.é.m. linéaire        E   = S · (T_hot − T_cold)
//! température (réciproque) T_hot = T_ref + E / S
//! loi des temp. interm.  E_tot = E_meas + E_ref
//! sensibilité            s   = S
//! ```
//!
//! `S` coefficient de Seebeck du couple de matériaux (V/K), `T_hot`, `T_cold`,
//! `T_ref` températures de jonction (K, absolues), `E`, `E_meas`, `E_ref`,
//! `E_tot` forces électromotrices (V), `s` sensibilité (V/K). La compensation de
//! soudure froide applique la **loi des températures intermédiaires** : on ajoute
//! à la f.é.m. mesurée la f.é.m. de la soudure froide ramenée à la référence.
//!
//! **Convention** : SI cohérent (températures en kelvin, tensions en volt).
//! **Limite honnête** : approximation **linéaire** — le coefficient de Seebeck `S`
//! est supposé **constant** sur la plage et **fourni par l'appelant** selon le
//! couple de matériaux (type K, J, T…), aucune valeur « par défaut » n'est
//! inventée. La réponse réelle est **polynomiale** ; ce module ne remplace pas les
//! tables normalisées **ITS-90** ni les polynômes de référence des thermocouples.

/// Force électromotrice approximée linéaire `E = S · (T_hot − T_cold)` (V).
///
/// Panique si `seebeck_coefficient <= 0`, `hot_junction_temp <= 0` ou
/// `cold_junction_temp <= 0`.
pub fn thermocouple_emf_linear(
    seebeck_coefficient: f64,
    hot_junction_temp: f64,
    cold_junction_temp: f64,
) -> f64 {
    assert!(
        seebeck_coefficient > 0.0,
        "le coefficient de Seebeck doit être strictement positif"
    );
    assert!(
        hot_junction_temp > 0.0,
        "la température de jonction chaude doit être strictement positive (K)"
    );
    assert!(
        cold_junction_temp > 0.0,
        "la température de jonction froide doit être strictement positive (K)"
    );
    seebeck_coefficient * (hot_junction_temp - cold_junction_temp)
}

/// Température de jonction chaude déduite de la f.é.m. `T_hot = T_ref + E / S`
/// (K) — réciproque de [`thermocouple_emf_linear`], avec compensation de soudure
/// froide par la température de référence.
///
/// Panique si `seebeck_coefficient <= 0` ou `reference_temperature <= 0`.
pub fn thermocouple_temperature_from_emf(
    emf: f64,
    seebeck_coefficient: f64,
    reference_temperature: f64,
) -> f64 {
    assert!(
        seebeck_coefficient > 0.0,
        "le coefficient de Seebeck doit être strictement positif"
    );
    assert!(
        reference_temperature > 0.0,
        "la température de référence doit être strictement positive (K)"
    );
    reference_temperature + emf / seebeck_coefficient
}

/// Compensation de soudure froide par la loi des températures intermédiaires
/// `E_tot = E_meas + E_ref` (V) : ajout de la f.é.m. de la soudure froide
/// ramenée à la référence.
///
/// Panique si `measured_emf` ou `reference_junction_emf` n'est pas fini.
pub fn thermocouple_cold_junction_correction(
    measured_emf: f64,
    reference_junction_emf: f64,
) -> f64 {
    assert!(
        measured_emf.is_finite(),
        "la f.é.m. mesurée doit être finie"
    );
    assert!(
        reference_junction_emf.is_finite(),
        "la f.é.m. de la soudure froide doit être finie"
    );
    measured_emf + reference_junction_emf
}

/// Sensibilité du thermocouple `s = S` (V/K) : par définition, dans
/// l'approximation linéaire, la sensibilité égale le coefficient de Seebeck.
///
/// Panique si `seebeck_coefficient <= 0`.
pub fn thermocouple_sensitivity(seebeck_coefficient: f64) -> f64 {
    assert!(
        seebeck_coefficient > 0.0,
        "le coefficient de Seebeck doit être strictement positif"
    );
    seebeck_coefficient
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reciprocity_emf_temperature() {
        // Aller-retour T_hot → E → T_hot : identité exacte.
        let s = 41e-6;
        let t_hot = 373.15;
        let t_cold = 293.15;
        let emf = thermocouple_emf_linear(s, t_hot, t_cold);
        let back = thermocouple_temperature_from_emf(emf, s, t_cold);
        assert_relative_eq!(back, t_hot, epsilon = 1e-9);
    }

    #[test]
    fn emf_reference_case() {
        // Type K ≈ 41 µV/K, T_hot=373,15 K, T_cold=293,15 K (ΔT=80 K)
        // → E = 41e-6 · 80 = 3,28 mV.
        let emf = thermocouple_emf_linear(41e-6, 373.15, 293.15);
        assert_relative_eq!(emf, 3.28e-3, epsilon = 1e-12);
    }

    #[test]
    fn emf_vanishes_when_junctions_equal() {
        // Jonctions à la même température : f.é.m. nulle.
        let emf = thermocouple_emf_linear(41e-6, 300.0, 300.0);
        assert_relative_eq!(emf, 0.0, epsilon = 1e-18);
    }

    #[test]
    fn emf_proportional_to_seebeck() {
        // À ΔT fixé, doubler S double la f.é.m.
        let e1 = thermocouple_emf_linear(20e-6, 400.0, 300.0);
        let e2 = thermocouple_emf_linear(40e-6, 400.0, 300.0);
        assert_relative_eq!(e2 / e1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn cold_junction_correction_adds_reference_emf() {
        // E_meas=3,28 mV, E_ref=1,20 mV → E_tot=4,48 mV, et la sensibilité
        // rend bien le coefficient de Seebeck fourni.
        let e_tot = thermocouple_cold_junction_correction(3.28e-3, 1.20e-3);
        assert_relative_eq!(e_tot, 4.48e-3, epsilon = 1e-12);
        assert_relative_eq!(thermocouple_sensitivity(41e-6), 41e-6, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "coefficient de Seebeck")]
    fn zero_seebeck_panics() {
        thermocouple_emf_linear(0.0, 373.15, 293.15);
    }
}
