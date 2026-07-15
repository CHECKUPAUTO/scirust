//! Générateur thermoélectrique (effet Seebeck) — récupération d'énergie à partir
//! d'un gradient de température : facteur de mérite, tension à vide, puissance
//! maximale sur charge adaptée et rendement de conversion.
//!
//! ```text
//! facteur de mérite   ZT     = S² · σ · T / κ
//! tension à vide      V_oc   = S · ΔT
//! puissance maximale  P_max  = (S · ΔT)² / (4 · R_int)
//! rendement           η      = (T_h − T_c)/T_h · (√(1+ZT) − 1) / (√(1+ZT) + T_c/T_h)
//! ```
//!
//! `S` coefficient de Seebeck du matériau (V/K), `σ` conductivité électrique
//! (S/m), `κ` conductivité thermique (W/(m·K)), `T`, `T_h`, `T_c` températures
//! absolues (K), `ΔT` différence de température (K), `R_int` résistance interne du
//! module (Ω), `ZT` facteur de mérite adimensionnel, `V_oc` tension à vide (V),
//! `P_max` puissance électrique maximale (W), `η` rendement adimensionnel. Le
//! rendement est le produit du rendement de Carnot `(T_h − T_c)/T_h` par un facteur
//! matériau croissant avec `ZT`.
//!
//! **Convention** : SI cohérent (kelvin, volt, ohm, watt) ; températures
//! **absolues** en kelvin.
//! **Limite honnête** : le coefficient de Seebeck `S`, les conductivités `σ` et
//! `κ`, le `ZT` moyen et la résistance interne `R_int` sont des **propriétés du
//! matériau et du module FOURNIES par l'appelant** ; aucune valeur « par défaut »
//! n'est inventée. Les propriétés sont supposées **constantes** sur la plage de
//! température ; la puissance maximale suppose une **charge adaptée**
//! (résistance externe = résistance interne). Ce modèle idéalisé ne remplace pas
//! une simulation à propriétés dépendant de la température ni les effets de
//! contact et de pertes parasites.

/// Facteur de mérite adimensionnel `ZT = S² · σ · T / κ`.
///
/// Panique si `electrical_conductivity <= 0`, `thermal_conductivity <= 0` ou
/// `temperature <= 0`.
pub fn teg_figure_of_merit_zt(
    seebeck_coefficient: f64,
    electrical_conductivity: f64,
    thermal_conductivity: f64,
    temperature: f64,
) -> f64 {
    assert!(
        electrical_conductivity > 0.0,
        "la conductivité électrique doit être strictement positive (S/m)"
    );
    assert!(
        thermal_conductivity > 0.0,
        "la conductivité thermique doit être strictement positive (W/(m·K))"
    );
    assert!(
        temperature > 0.0,
        "la température absolue doit être strictement positive (K)"
    );
    seebeck_coefficient * seebeck_coefficient * electrical_conductivity * temperature
        / thermal_conductivity
}

/// Tension à vide `V_oc = S · ΔT` (V).
///
/// Panique si `temperature_difference < 0` (le côté chaud doit être au moins
/// aussi chaud que le côté froid).
pub fn teg_open_circuit_voltage(seebeck_coefficient: f64, temperature_difference: f64) -> f64 {
    assert!(
        temperature_difference >= 0.0,
        "la différence de température doit être positive ou nulle (K)"
    );
    seebeck_coefficient * temperature_difference
}

/// Puissance électrique maximale sur charge adaptée
/// `P_max = (S · ΔT)² / (4 · R_int)` (W).
///
/// Panique si `internal_resistance <= 0` ou `temperature_difference < 0`.
pub fn teg_max_power(
    seebeck_coefficient: f64,
    temperature_difference: f64,
    internal_resistance: f64,
) -> f64 {
    assert!(
        internal_resistance > 0.0,
        "la résistance interne doit être strictement positive (Ω)"
    );
    assert!(
        temperature_difference >= 0.0,
        "la différence de température doit être positive ou nulle (K)"
    );
    (seebeck_coefficient * temperature_difference).powi(2) / (4.0 * internal_resistance)
}

/// Rendement de conversion
/// `η = (T_h − T_c)/T_h · (√(1+ZT) − 1) / (√(1+ZT) + T_c/T_h)` (adimensionnel) :
/// rendement de Carnot multiplié par le facteur matériau croissant avec `ZT`.
///
/// Panique si `hot_temperature <= 0`, `cold_temperature <= 0`,
/// `cold_temperature > hot_temperature` ou `zt_average < 0`.
pub fn teg_efficiency(hot_temperature: f64, cold_temperature: f64, zt_average: f64) -> f64 {
    assert!(
        hot_temperature > 0.0,
        "la température chaude doit être strictement positive (K)"
    );
    assert!(
        cold_temperature > 0.0,
        "la température froide doit être strictement positive (K)"
    );
    assert!(
        cold_temperature <= hot_temperature,
        "la température froide ne peut pas dépasser la température chaude (K)"
    );
    assert!(
        zt_average >= 0.0,
        "le facteur de mérite moyen ZT doit être positif ou nul"
    );
    let carnot = (hot_temperature - cold_temperature) / hot_temperature;
    let root = (1.0 + zt_average).sqrt();
    carnot * (root - 1.0) / (root + cold_temperature / hot_temperature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zt_reference_case() {
        // S=200 µV/K, σ=1e5 S/m, κ=1,5 W/(m·K), T=300 K :
        // ZT = (2e-4)² · 1e5 · 300 / 1,5 = 4e-8 · 1e5 · 300 / 1,5
        //    = 4e-3 · 300 / 1,5 = 1,2 / 1,5 = 0,8.
        let zt = teg_figure_of_merit_zt(200e-6, 1e5, 1.5, 300.0);
        assert_relative_eq!(zt, 0.8, epsilon = 1e-12);
    }

    #[test]
    fn zt_proportional_to_temperature() {
        // À propriétés fixées, ZT est linéaire en température absolue.
        let z1 = teg_figure_of_merit_zt(200e-6, 1e5, 1.5, 300.0);
        let z2 = teg_figure_of_merit_zt(200e-6, 1e5, 1.5, 600.0);
        assert_relative_eq!(z2 / z1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn open_circuit_voltage_reference() {
        // V_oc = S · ΔT = 2e-4 · 100 = 0,02 V ; nulle si ΔT = 0.
        let v = teg_open_circuit_voltage(200e-6, 100.0);
        assert_relative_eq!(v, 0.02, epsilon = 1e-12);
        assert_relative_eq!(teg_open_circuit_voltage(200e-6, 0.0), 0.0, epsilon = 1e-18);
    }

    #[test]
    fn max_power_matches_voltage_squared_over_resistance() {
        // P_max = (S·ΔT)² / (4·R) = V_oc² / (4·R).
        // V_oc = 0,02 V, R = 0,5 Ω → 4e-4 / 2 = 2e-4 W.
        let p = teg_max_power(200e-6, 100.0, 0.5);
        assert_relative_eq!(p, 2e-4, epsilon = 1e-15);
        let v = teg_open_circuit_voltage(200e-6, 100.0);
        assert_relative_eq!(p, v * v / (4.0 * 0.5), epsilon = 1e-18);
    }

    #[test]
    fn efficiency_reference_case() {
        // T_h=600 K, T_c=300 K, ZT=1 :
        // Carnot = 300/600 = 0,5 ; √2 = 1,414213562…
        // η = 0,5 · (√2 − 1)/(√2 + 0,5) = 0,5 · 0,414213562/1,914213562
        //   ≈ 0,5 · 0,216385 ≈ 0,108193.
        let eta = teg_efficiency(600.0, 300.0, 1.0);
        assert_relative_eq!(eta, 0.108193, epsilon = 1e-3);
    }

    #[test]
    fn efficiency_approaches_carnot_for_large_zt() {
        // Quand ZT → ∞, le facteur matériau → 1 et η → rendement de Carnot.
        let carnot = (600.0 - 300.0) / 600.0;
        let eta = teg_efficiency(600.0, 300.0, 1e10);
        assert_relative_eq!(eta, carnot, epsilon = 1e-3);
    }

    #[test]
    fn efficiency_vanishes_without_gradient() {
        // Sans gradient (T_c = T_h), rendement de Carnot nul donc η = 0.
        let eta = teg_efficiency(400.0, 400.0, 1.0);
        assert_relative_eq!(eta, 0.0, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "résistance interne")]
    fn zero_resistance_panics() {
        teg_max_power(200e-6, 100.0, 0.0);
    }
}
