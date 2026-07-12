//! Rayonnement thermique — loi de **Stefan-Boltzmann**, émittance d'un corps
//! gris, échange net avec l'environnement et coefficient de rayonnement linéarisé.
//!
//! ```text
//! corps noir        M = σ·T⁴
//! corps gris        M = ε·σ·T⁴
//! échange net       Q = ε·σ·A·(Ts⁴ − Tenv⁴)      (petit corps dans une enceinte)
//! coeff. linéarisé  hr = ε·σ·(Ts + Tenv)·(Ts² + Tenv²)
//! ```
//!
//! `σ` constante de Stefan-Boltzmann (5,6704·10⁻⁸ W/(m²·K⁴)), `ε` émissivité
//! (`0`–`1`), `A` aire (m²), `T` températures **absolues** (K), `hr` coefficient
//! de rayonnement équivalent (W/(m²·K)) permettant de traiter le rayonnement
//! comme une convection.
//!
//! **Convention** : températures en **kelvin**. **Limite honnête** : corps gris
//! diffus ; l'échange net donné vaut pour un **petit corps** entouré d'une grande
//! enceinte (facteur de forme unité). Les échanges entre surfaces finies (facteurs
//! de forme, enceintes à N surfaces) sont à composer par l'appelant. `ε` est fourni.

/// Constante de Stefan-Boltzmann `σ` (W/(m²·K⁴)).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Émittance d'un **corps noir** `M = σ·T⁴` (W/m²).
///
/// Panique si `temperature_k < 0`.
pub fn blackbody_emissive_power(temperature_k: f64) -> f64 {
    assert!(
        temperature_k >= 0.0,
        "la température absolue doit être positive"
    );
    STEFAN_BOLTZMANN * temperature_k.powi(4)
}

/// Émittance d'un **corps gris** `M = ε·σ·T⁴` (W/m²).
///
/// Panique si `emissivity` hors `[0, 1]`.
pub fn gray_body_emissive_power(emissivity: f64, temperature_k: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&emissivity),
        "l'émissivité doit être dans [0, 1]"
    );
    emissivity * blackbody_emissive_power(temperature_k)
}

/// Échange net d'un petit corps gris avec son environnement
/// `Q = ε·σ·A·(Ts⁴ − Tenv⁴)` (W).
///
/// Panique si `emissivity` hors `[0, 1]`.
pub fn net_radiation_to_surroundings(
    emissivity: f64,
    area: f64,
    surface_temp_k: f64,
    surroundings_temp_k: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&emissivity),
        "l'émissivité doit être dans [0, 1]"
    );
    emissivity * STEFAN_BOLTZMANN * area * (surface_temp_k.powi(4) - surroundings_temp_k.powi(4))
}

/// Coefficient de rayonnement linéarisé `hr = ε·σ·(Ts + Tenv)·(Ts² + Tenv²)`
/// (W/(m²·K)).
///
/// Panique si `emissivity` hors `[0, 1]`.
pub fn radiation_coefficient(
    emissivity: f64,
    surface_temp_k: f64,
    surroundings_temp_k: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&emissivity),
        "l'émissivité doit être dans [0, 1]"
    );
    emissivity
        * STEFAN_BOLTZMANN
        * (surface_temp_k + surroundings_temp_k)
        * (surface_temp_k * surface_temp_k + surroundings_temp_k * surroundings_temp_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn blackbody_at_1000k() {
        // M = σ·1000⁴ = 5,6704e-8·1e12 = 56 704 W/m².
        assert_relative_eq!(
            blackbody_emissive_power(1000.0),
            STEFAN_BOLTZMANN * 1e12,
            epsilon = 1e-6
        );
    }

    #[test]
    fn gray_body_scales_by_emissivity() {
        // ε=0,8 → 80 % de l'émittance du corps noir.
        assert_relative_eq!(
            gray_body_emissive_power(0.8, 500.0),
            0.8 * blackbody_emissive_power(500.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn net_exchange_vanishes_at_thermal_equilibrium() {
        // Ts = Tenv → échange net nul.
        assert_relative_eq!(
            net_radiation_to_surroundings(0.9, 2.0, 350.0, 350.0),
            0.0,
            epsilon = 1e-9
        );
        // Ts > Tenv → flux sortant positif.
        assert!(net_radiation_to_surroundings(0.9, 2.0, 400.0, 300.0) > 0.0);
    }

    #[test]
    fn linearised_coefficient_matches_net_flux() {
        // Q doit valoir hr·A·(Ts − Tenv) (linéarisation exacte du rayonnement).
        let (eps, a, ts, tenv) = (0.85, 1.5, 420.0, 300.0);
        let q = net_radiation_to_surroundings(eps, a, ts, tenv);
        let hr = radiation_coefficient(eps, ts, tenv);
        assert_relative_eq!(q, hr * a * (ts - tenv), max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "émissivité")]
    fn emissivity_above_one_panics() {
        gray_body_emissive_power(1.5, 500.0);
    }
}
