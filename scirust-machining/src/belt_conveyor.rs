//! **Bande transporteuse** — tension effective d'entraînement, débit massique
//! transporté, puissance au tambour moteur et puissance d'élévation.
//!
//! ```text
//! tension effective    T_e = F_f                       (somme des résistances)
//! débit massique       q_m = v·λ                        (kg/s)
//! puissance moteur      P_d = T_e·v                      (W)
//! puissance élévation   P_l = q_m·g·H                    (W)
//! ```
//!
//! `T_e` tension effective d'entraînement (N), `F_f` effort de frottement /
//! résistance totale du convoyeur (N), `v` vitesse de la bande (m/s), `λ` charge
//! linéique transportée (masse par mètre de bande, kg/m), `q_m` débit massique
//! (kg/s), `g` accélération de la pesanteur (m/s²), `H` hauteur d'élévation (m),
//! `P_d` puissance au tambour moteur (W), `P_l` puissance d'élévation (W).
//!
//! **Convention** : SI ; unités cohérentes exigées. **Limite honnête** : régime
//! **établi** (vitesse constante, pas de phase transitoire de démarrage), bande
//! supposée pleine et uniformément chargée. L'effort de frottement / résistance
//! totale `F_f` (résistances principale, secondaire, de pente…) est **fourni par
//! l'appelant** : ce module ne modélise pas les coefficients de résistance ni la
//! géométrie du parcours. La valeur de `g` est également fournie par l'appelant
//! (aucune constante « par défaut » n'est imposée). Distinct de la mécanique de
//! courroie de transmission de [`crate::belts`].

/// Tension effective d'entraînement `T_e = F_f` (report de l'effort résistant
/// total sur le brin d'entraînement, en régime établi).
///
/// Panique si `total_friction_force < 0`.
pub fn conveyor_effective_tension(total_friction_force: f64) -> f64 {
    assert!(
        total_friction_force >= 0.0,
        "l'effort de frottement total F_f ≥ 0 est requis"
    );
    total_friction_force
}

/// Débit massique transporté `q_m = v·λ` (kg/s).
///
/// Panique si `belt_speed < 0` ou `load_per_length < 0`.
pub fn conveyor_mass_flow(belt_speed: f64, load_per_length: f64) -> f64 {
    assert!(
        belt_speed >= 0.0 && load_per_length >= 0.0,
        "v ≥ 0 et λ ≥ 0 requis"
    );
    belt_speed * load_per_length
}

/// Puissance au tambour moteur `P_d = T_e·v` (W).
///
/// Panique si `effective_tension < 0` ou `belt_speed < 0`.
pub fn conveyor_drive_power(effective_tension: f64, belt_speed: f64) -> f64 {
    assert!(
        effective_tension >= 0.0 && belt_speed >= 0.0,
        "T_e ≥ 0 et v ≥ 0 requis"
    );
    effective_tension * belt_speed
}

/// Puissance d'élévation `P_l = q_m·g·H` (W), part de puissance dédiée à monter
/// la charge de la hauteur `H`.
///
/// Panique si un paramètre est `< 0`.
pub fn conveyor_lift_power(mass_flow: f64, height: f64, gravity: f64) -> f64 {
    assert!(
        mass_flow >= 0.0 && height >= 0.0 && gravity >= 0.0,
        "q_m ≥ 0, H ≥ 0 et g ≥ 0 requis"
    );
    mass_flow * gravity * height
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn effective_tension_reports_friction() {
        // En régime établi, la tension effective vaut exactement l'effort résistant.
        assert_relative_eq!(conveyor_effective_tension(3200.0), 3200.0, epsilon = 1e-9);
    }

    #[test]
    fn mass_flow_is_linear_in_speed_and_load() {
        // q_m = v·λ : doubler la vitesse double le débit à charge linéique fixe.
        let q1 = conveyor_mass_flow(1.5, 40.0);
        let q2 = conveyor_mass_flow(3.0, 40.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-9);
        // Cas chiffré : v=2 m/s, λ=50 kg/m → q_m = 100 kg/s.
        assert_relative_eq!(conveyor_mass_flow(2.0, 50.0), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn drive_power_matches_tension_times_speed() {
        // P_d = T_e·v : cohérence avec la tension effective déduite du frottement.
        let te = conveyor_effective_tension(3200.0);
        assert_relative_eq!(conveyor_drive_power(te, 2.5), 8000.0, epsilon = 1e-9);
    }

    #[test]
    fn zero_speed_gives_zero_transport() {
        // À l'arrêt : aucun débit et aucune puissance d'entraînement.
        assert_relative_eq!(conveyor_mass_flow(0.0, 50.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(conveyor_drive_power(3200.0, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn lift_power_equals_potential_energy_rate() {
        // P_l = q_m·g·H doit égaler le taux d'énergie potentielle gagnée.
        // q_m=100 kg/s, H=10 m, g=9,81 → P_l = 100·9,81·10 = 9810 W.
        let qm = conveyor_mass_flow(2.0, 50.0);
        assert_relative_eq!(conveyor_lift_power(qm, 10.0, 9.81), 9810.0, epsilon = 1e-9);
        // Élévation nulle : aucune puissance d'élévation requise.
        assert_relative_eq!(conveyor_lift_power(qm, 0.0, 9.81), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "v ≥ 0 et λ ≥ 0 requis")]
    fn negative_speed_panics() {
        conveyor_mass_flow(-1.0, 50.0);
    }
}
