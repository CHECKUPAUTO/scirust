//! **Apport de chaleur en soudage à l'arc** — puissance d'arc, apport linéique
//! et énergie déposée par unité de longueur du cordon.
//!
//! ```text
//! puissance d'arc      P = U·I
//! apport de chaleur    H = η·U·I / v = η·P / v          (J/m)
//! énergie linéique     E = H                            (identité, J/m)
//! ```
//!
//! `η` rendement d'arc du procédé (sans dimension), `U` tension d'arc (V), `I`
//! intensité de soudage (A), `P` puissance électrique d'arc (W), `v` vitesse
//! d'avance de la torche (m/s), `H` apport de chaleur linéique (J/m), `E`
//! énergie déposée par unité de longueur (J/m).
//!
//! **Convention** : SI. **Limite honnête** : modèle en **régime quasi-stationnaire**,
//! où la source d'arc avance à vitesse constante et l'apport `H = η·U·I/v` est
//! l'énergie **nette** déposée par mètre de cordon (avant conduction dans la pièce).
//! Le rendement d'arc `η` **dépend du procédé** (arc submergé, MIG/MAG, TIG, à
//! électrode enrobée…) et est **fourni par l'appelant** ; aucune valeur « par
//! défaut » n'est inventée. Le partage entre pièce et environnement, la géométrie
//! du bain et les pertes par rayonnement/spatter ne sont pas modélisés. Voir
//! [`crate::welds`] (dimensionnement des cordons) et [`crate::thermal`].

/// Puissance électrique d'arc `P = U·I` (W).
///
/// Panique si `voltage < 0` ou `current < 0`.
pub fn weld_arc_power(voltage: f64, current: f64) -> f64 {
    assert!(
        voltage >= 0.0 && current >= 0.0,
        "tension U ≥ 0 et intensité I ≥ 0 requises"
    );
    voltage * current
}

/// Apport de chaleur linéique `H = η·U·I / v` (J/m).
///
/// Panique si `arc_efficiency` ∉ [0, 1], si `voltage < 0`, si `current < 0`
/// ou si `travel_speed <= 0`.
pub fn weld_heat_input(arc_efficiency: f64, voltage: f64, current: f64, travel_speed: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&arc_efficiency),
        "le rendement d'arc η doit être dans [0, 1]"
    );
    assert!(
        travel_speed > 0.0,
        "la vitesse d'avance v doit être strictement positive"
    );
    arc_efficiency * weld_arc_power(voltage, current) / travel_speed
}

/// Énergie déposée par unité de longueur `E = H` (J/m).
///
/// Identité de convention : l'apport de chaleur linéique **est** l'énergie
/// déposée par mètre de cordon.
///
/// Panique si `heat_input < 0`.
pub fn weld_energy_per_length(heat_input: f64) -> f64 {
    assert!(heat_input >= 0.0, "l'apport de chaleur H ≥ 0 requis");
    heat_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn arc_power_is_voltage_times_current() {
        // U=25 V, I=200 A → P = 5000 W.
        assert_relative_eq!(weld_arc_power(25.0, 200.0), 5000.0, epsilon = 1e-9);
    }

    #[test]
    fn heat_input_matches_efficiency_times_power_over_speed() {
        // Identité H = η·P/v, cohérence avec weld_arc_power.
        let eta = 0.8_f64;
        let (u, i, v) = (25.0_f64, 200.0_f64, 5e-3_f64);
        let h = weld_heat_input(eta, u, i, v);
        assert_relative_eq!(h, eta * weld_arc_power(u, i) / v, epsilon = 1e-6);
    }

    #[test]
    fn realistic_case_joules_per_metre() {
        // MIG/MAG : η=0,8 ; U=25 V ; I=200 A ; v=5 mm/s=5e-3 m/s.
        // H = 0,8·25·200 / 5e-3 = 4000 / 5e-3 = 8e5 J/m (= 800 kJ/m).
        let h = weld_heat_input(0.8, 25.0, 200.0, 5e-3);
        assert_relative_eq!(h, 8e5, epsilon = 1e-3);
    }

    #[test]
    fn heat_input_inversely_proportional_to_speed() {
        // Doubler la vitesse d'avance divise l'apport linéique par deux.
        let slow = weld_heat_input(0.7, 24.0, 180.0, 4e-3);
        let fast = weld_heat_input(0.7, 24.0, 180.0, 8e-3);
        assert_relative_eq!(fast, slow / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn energy_per_length_is_identity() {
        // E = H : l'énergie linéique reproduit l'apport de chaleur.
        let h = weld_heat_input(0.6, 22.0, 150.0, 6e-3);
        assert_relative_eq!(weld_energy_per_length(h), h, epsilon = 1e-12);
    }

    #[test]
    fn full_efficiency_recovers_arc_energy_per_length() {
        // À η=1, l'apport net égale la puissance d'arc divisée par la vitesse.
        let h = weld_heat_input(1.0, 30.0, 250.0, 5e-3);
        assert_relative_eq!(h, weld_arc_power(30.0, 250.0) / 5e-3, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "vitesse d'avance v doit être strictement positive")]
    fn zero_travel_speed_panics() {
        weld_heat_input(0.8, 25.0, 200.0, 0.0);
    }
}
