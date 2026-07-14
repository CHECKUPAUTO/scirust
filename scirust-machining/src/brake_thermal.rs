//! **Échauffement de frein** — énergie cinétique dissipée à l'arrêt, élévation de
//! température du disque et puissance moyenne de freinage.
//!
//! ```text
//! énergie dissipée      E = ½·m·v²           (énergie cinétique de translation)
//! élévation de temp.    ΔT = E/(m_disc·c)     (échauffement adiabatique du disque)
//! puissance moyenne     P = E/t               (t durée de l'arrêt)
//! ```
//!
//! `m` masse du véhicule en translation (kg), `v` vitesse initiale (m/s), `E`
//! énergie dissipée (J), `m_disc` masse du disque de frein (kg), `c` chaleur
//! massique du matériau du disque (J·kg⁻¹·K⁻¹), `ΔT` élévation de température (K),
//! `t` durée de l'arrêt (s), `P` puissance moyenne dissipée (W).
//!
//! **Convention** : SI. **Limite honnête** : on suppose que **toute** l'énergie
//! cinétique de translation est convertie en chaleur **dans le disque**, en
//! régime **adiabatique** (aucune évacuation par convection/rayonnement/conduction
//! pendant le freinage, ni partage avec les plaquettes ou le moyeu), et une
//! chaleur massique `c` **constante** sur la plage de température. La chaleur
//! massique `c` et les masses sont **fournies par l'appelant** ; aucune valeur
//! matériau « par défaut » n'est inventée. Voir [`crate::frictional_heating`]
//! (contact glissant) et [`crate::brakes`] (couple de freinage).

/// Énergie cinétique de translation dissipée à l'arrêt `E = ½·m·v²`.
///
/// Panique si `mass < 0` ou `velocity < 0`.
pub fn brake_dissipated_energy(mass: f64, velocity: f64) -> f64 {
    assert!(mass >= 0.0 && velocity >= 0.0, "m ≥ 0 et v ≥ 0 requis");
    0.5 * mass * velocity * velocity
}

/// Élévation de température adiabatique du disque `ΔT = E/(m_disc·c)`.
///
/// Panique si `energy < 0`, `disc_mass <= 0` ou `specific_heat <= 0`.
pub fn brake_temperature_rise(energy: f64, disc_mass: f64, specific_heat: f64) -> f64 {
    assert!(energy >= 0.0, "l'énergie dissipée doit être positive");
    assert!(
        disc_mass > 0.0 && specific_heat > 0.0,
        "m_disc > 0 et c > 0 requis (dénominateur non nul)"
    );
    energy / (disc_mass * specific_heat)
}

/// Puissance moyenne dissipée pendant l'arrêt `P = E/t`.
///
/// Panique si `energy < 0` ou `stop_time <= 0`.
pub fn brake_power(energy: f64, stop_time: f64) -> f64 {
    assert!(energy >= 0.0, "l'énergie dissipée doit être positive");
    assert!(stop_time > 0.0, "t > 0 requis (dénominateur non nul)");
    energy / stop_time
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn energy_scales_with_velocity_squared() {
        // Doubler v quadruple l'énergie dissipée (E ∝ v²).
        let e1 = brake_dissipated_energy(1200.0, 10.0);
        let e2 = brake_dissipated_energy(1200.0, 20.0);
        assert_relative_eq!(e2, 4.0 * e1, epsilon = 1e-9);
    }

    #[test]
    fn energy_realistic_case() {
        // Véhicule de 1200 kg à 20 m/s (72 km/h) → E = ½·1200·400 = 240 000 J.
        assert_relative_eq!(
            brake_dissipated_energy(1200.0, 20.0),
            240_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn temperature_rise_inverse_of_thermal_mass() {
        // ΔT ∝ 1/(m_disc·c) : doubler la masse du disque halve l'échauffement.
        let dt1 = brake_temperature_rise(240_000.0, 5.0, 460.0);
        let dt2 = brake_temperature_rise(240_000.0, 10.0, 460.0);
        assert_relative_eq!(dt2, 0.5 * dt1, epsilon = 1e-9);
    }

    #[test]
    fn temperature_rise_realistic_case() {
        // 240 kJ dans un disque acier de 5 kg (c=460 J/kg/K) → ΔT ≈ 104,3 K.
        let dt = brake_temperature_rise(240_000.0, 5.0, 460.0);
        assert_relative_eq!(dt, 240_000.0 / (5.0 * 460.0), epsilon = 1e-9);
        assert_relative_eq!(dt, 104.347_826_086_956_5, epsilon = 1e-6);
    }

    #[test]
    fn power_is_energy_over_time() {
        // 240 kJ dissipés en 4 s → P = 60 kW ; identité E = P·t.
        let p = brake_power(240_000.0, 4.0);
        assert_relative_eq!(p, 60_000.0, epsilon = 1e-9);
        assert_relative_eq!(p * 4.0, 240_000.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "t > 0 requis")]
    fn zero_stop_time_panics() {
        brake_power(240_000.0, 0.0);
    }
}
