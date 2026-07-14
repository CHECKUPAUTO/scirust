//! **Échauffement par frottement** — effort, puissance dissipée, densité de flux
//! et élévation de température d'un contact glissant.
//!
//! ```text
//! effort de frottement F_f = μ·F_N
//! puissance dissipée   P = F_f·v = μ·F_N·v
//! densité de flux      q'' = P/A
//! élévation de temp.   ΔT = P·R_th                    (R_th résistance thermique)
//! ```
//!
//! `μ` coefficient de frottement, `F_N` charge normale (N), `F_f` effort de
//! frottement (N), `v` vitesse de glissement (m/s), `P` puissance dissipée (W),
//! `A` aire de contact (m²), `q''` densité de flux thermique (W/m²), `R_th`
//! résistance thermique du chemin d'évacuation (K/W), `ΔT` élévation de
//! température (K).
//!
//! **Convention** : SI. **Limite honnête** : toute la puissance de frottement est
//! supposée **convertie en chaleur** ; l'élévation `ΔT = P·R_th` est un modèle de
//! résistance **globale** en **régime établi** (le partage du flux entre les deux
//! corps et la température **flash** locale ne sont pas traités). `μ` et `R_th`
//! sont fournis par l'appelant. Voir [`crate::friction`] (statique/glissement) et
//! [`crate::thermal_network`] (résistances).

/// Effort de frottement `F_f = μ·F_N`.
///
/// Panique si `friction_coefficient < 0` ou `normal_load < 0`.
pub fn friction_force(friction_coefficient: f64, normal_load: f64) -> f64 {
    assert!(
        friction_coefficient >= 0.0 && normal_load >= 0.0,
        "μ ≥ 0 et F_N ≥ 0 requis"
    );
    friction_coefficient * normal_load
}

/// Puissance dissipée `P = μ·F_N·v`.
///
/// Panique si une grandeur `< 0`.
pub fn friction_power(friction_coefficient: f64, normal_load: f64, sliding_speed: f64) -> f64 {
    assert!(
        sliding_speed >= 0.0,
        "la vitesse de glissement doit être positive"
    );
    friction_force(friction_coefficient, normal_load) * sliding_speed
}

/// Densité de flux thermique `q'' = P/A`.
///
/// Panique si `power < 0` ou `area <= 0`.
pub fn heat_flux(power: f64, area: f64) -> f64 {
    assert!(power >= 0.0 && area > 0.0, "P ≥ 0 et A > 0 requis");
    power / area
}

/// Élévation de température en régime établi `ΔT = P·R_th`.
///
/// Panique si `power < 0` ou `thermal_resistance < 0`.
pub fn temperature_rise(power: f64, thermal_resistance: f64) -> f64 {
    assert!(
        power >= 0.0 && thermal_resistance >= 0.0,
        "P ≥ 0 et R_th ≥ 0 requis"
    );
    power * thermal_resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn friction_force_from_coefficient() {
        // μ=0,15, F_N=2000 N → F_f = 300 N.
        assert_relative_eq!(friction_force(0.15, 2000.0), 300.0, epsilon = 1e-9);
    }

    #[test]
    fn power_is_force_times_speed() {
        // F_f=300 N à 2 m/s → 600 W.
        let p = friction_power(0.15, 2000.0, 2.0);
        assert_relative_eq!(p, 300.0 * 2.0, epsilon = 1e-9);
        assert_relative_eq!(p, friction_force(0.15, 2000.0) * 2.0, epsilon = 1e-9);
    }

    #[test]
    fn zero_speed_dissipates_nothing() {
        // À l'arrêt (v=0), pas de puissance dissipée malgré la charge.
        assert_relative_eq!(friction_power(0.15, 2000.0, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn flux_and_temperature_rise() {
        // 600 W sur 1e-3 m² → q'' = 6e5 W/m² ; avec R_th=0,05 K/W → ΔT=30 K.
        assert_relative_eq!(heat_flux(600.0, 1e-3), 6e5, epsilon = 1e-6);
        assert_relative_eq!(temperature_rise(600.0, 0.05), 30.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_area_flux_panics() {
        heat_flux(600.0, 0.0);
    }
}
