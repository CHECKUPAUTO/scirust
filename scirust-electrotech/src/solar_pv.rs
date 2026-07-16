//! **Cellule et panneau photovoltaïque** — facteur de forme d'une cellule,
//! rendement de conversion, correction en température de la puissance nominale,
//! puissance et tension d'un champ (association série/parallèle de modules) aux
//! conditions standard de mesure (STC), en régime permanent.
//!
//! ```text
//! facteur de forme      FF = P_max / (V_oc · I_sc)
//! rendement             η  = P_max / (E · A)
//! puissance du champ    P_champ = P_module · N_s · N_p
//! correction en temp.   P(T) = P_nom · (1 + γ·(T_cell − T_ref))
//! tension du champ      V_champ = V_module · N_s
//! ```
//!
//! `FF` facteur de forme (sans dimension, entre 0 et 1), `P_max` puissance au
//! point de puissance maximale (W), `V_oc` tension en circuit ouvert (V),
//! `I_sc` courant de court-circuit (A), `η` rendement de conversion (sans
//! dimension), `E` éclairement / irradiance reçue (W/m²), `A` surface de la
//! cellule ou du module (m²), `P_champ` puissance du champ (W), `P_module`
//! puissance d'un module (W), `N_s` nombre de modules en série, `N_p` nombre de
//! branches en parallèle, `P(T)` puissance corrigée en température (W), `P_nom`
//! puissance nominale STC (W), `γ` coefficient de température de la puissance
//! (1/°C, typiquement négatif, p. ex. −0,004 soit −0,4 %/°C), `T_cell`
//! température de la cellule (°C), `T_ref` température de référence STC (°C,
//! usuellement 25 °C), `V_champ` tension du champ (V), `V_module` tension d'un
//! module (V).
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances en W,
//! irradiance en W/m², surface en m², températures en °C, coefficient de
//! température en 1/°C. **Limite honnête** : les caractéristiques `V_oc`,
//! `I_sc`, `P_max`, `P_nom` et le coefficient `γ` sont **fournis par la fiche
//! technique aux conditions STC** (1000 W/m², 25 °C, AM 1,5) ; l'irradiance `E`
//! et la surface `A` sont **fournies** par l'appelant ; la correction en
//! température est **linéaire** (coefficient `γ` fourni). Ce module ne
//! reconstitue **ni la courbe I-V complète** (équation de la diode), **ni les
//! pertes d'ombrage, de câblage, de désadaptation (mismatch) ou d'onduleur** ;
//! il est **distinct de scirust-bms** (état de charge / santé d'un accumulateur).

/// Facteur de forme d'une cellule `FF = P_max / (V_oc · I_sc)` (sans
/// dimension), rapport entre la puissance maximale et le produit `V_oc·I_sc`.
///
/// Panique si `max_power < 0`, si `open_circuit_voltage <= 0` ou si
/// `short_circuit_current <= 0`.
pub fn pv_fill_factor(
    max_power: f64,
    open_circuit_voltage: f64,
    short_circuit_current: f64,
) -> f64 {
    assert!(
        max_power >= 0.0,
        "la puissance maximale P_max doit être ≥ 0"
    );
    assert!(
        open_circuit_voltage > 0.0,
        "la tension de circuit ouvert V_oc doit être > 0"
    );
    assert!(
        short_circuit_current > 0.0,
        "le courant de court-circuit I_sc doit être > 0"
    );
    max_power / (open_circuit_voltage * short_circuit_current)
}

/// Rendement de conversion `η = P_max / (E · A)` (sans dimension), rapport de la
/// puissance électrique maximale à la puissance lumineuse reçue.
///
/// Panique si `max_power < 0`, si `irradiance <= 0` ou si `cell_area <= 0`.
pub fn pv_efficiency(max_power: f64, irradiance: f64, cell_area: f64) -> f64 {
    assert!(
        max_power >= 0.0,
        "la puissance maximale P_max doit être ≥ 0"
    );
    assert!(irradiance > 0.0, "l'irradiance E doit être > 0");
    assert!(cell_area > 0.0, "la surface A doit être > 0");
    max_power / (irradiance * cell_area)
}

/// Puissance d'un champ `P_champ = P_module · N_s · N_p` (W), association de
/// `N_s` modules en série par `N_p` branches en parallèle.
///
/// Panique si `module_power < 0`, si `series_count <= 0` ou si
/// `parallel_count <= 0`.
pub fn pv_array_power(module_power: f64, series_count: f64, parallel_count: f64) -> f64 {
    assert!(
        module_power >= 0.0,
        "la puissance d'un module P_module doit être ≥ 0"
    );
    assert!(
        series_count > 0.0,
        "le nombre de modules en série N_s doit être > 0"
    );
    assert!(
        parallel_count > 0.0,
        "le nombre de branches en parallèle N_p doit être > 0"
    );
    module_power * series_count * parallel_count
}

/// Puissance corrigée en température `P(T) = P_nom · (1 + γ·(T_cell − T_ref))`
/// (W), le coefficient `γ` étant typiquement négatif.
///
/// Panique si `nominal_power < 0`.
pub fn pv_temperature_corrected_power(
    nominal_power: f64,
    power_temperature_coefficient: f64,
    cell_temperature: f64,
    reference_temperature: f64,
) -> f64 {
    assert!(
        nominal_power >= 0.0,
        "la puissance nominale P_nom doit être ≥ 0"
    );
    nominal_power
        * (1.0 + power_temperature_coefficient * (cell_temperature - reference_temperature))
}

/// Tension d'un champ `V_champ = V_module · N_s` (V), `N_s` modules en série.
///
/// Panique si `module_voltage < 0` ou si `series_count <= 0`.
pub fn pv_array_voltage(module_voltage: f64, series_count: f64) -> f64 {
    assert!(
        module_voltage >= 0.0,
        "la tension d'un module V_module doit être ≥ 0"
    );
    assert!(
        series_count > 0.0,
        "le nombre de modules en série N_s doit être > 0"
    );
    module_voltage * series_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fill_factor_realistic_case() {
        // Cellule : P_max = 3,2 W, V_oc = 0,6 V, I_sc = 8 A.
        //   FF = 3,2 / (0,6·8) = 3,2 / 4,8 = 0,666666...
        // Recalcul : 0,6·8 = 4,8 ; 3,2/4,8 = 2/3 = 0,6666666666...
        let ff = pv_fill_factor(3.2, 0.6, 8.0);
        assert_relative_eq!(ff, 2.0 / 3.0, epsilon = 1e-9);
        // Un FF est sans dimension et reste inférieur à 1.
        assert!(ff < 1.0);
    }

    #[test]
    fn efficiency_at_stc_is_ratio_of_powers() {
        // Module : P_max = 300 W, E = 1000 W/m², A = 1,6 m².
        //   η = 300 / (1000·1,6) = 300 / 1600 = 0,1875
        // Recalcul : 1000·1,6 = 1600 ; 300/1600 = 0,1875 (soit 18,75 %).
        let eta = pv_efficiency(300.0, 1000.0, 1.6);
        assert_relative_eq!(eta, 0.1875, epsilon = 1e-9);
    }

    #[test]
    fn array_power_scales_linearly_with_module_counts() {
        // P_champ ∝ N_s et ∝ N_p : doubler l'une double la puissance.
        let base = pv_array_power(400.0, 10.0, 4.0);
        assert_relative_eq!(
            pv_array_power(400.0, 20.0, 4.0) / base,
            2.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pv_array_power(400.0, 10.0, 8.0) / base,
            2.0,
            epsilon = 1e-12
        );
        // Cas chiffré : 400·10·4 = 16000 W.
        assert_relative_eq!(base, 16_000.0, epsilon = 1e-9);
    }

    #[test]
    fn temperature_correction_reduces_power_when_hot() {
        // P_nom = 300 W, γ = −0,004 /°C, T_cell = 45 °C, T_ref = 25 °C.
        //   P = 300·(1 + (−0,004)·(45 − 25)) = 300·(1 − 0,08) = 300·0,92 = 276 W
        // Recalcul : (45−25) = 20 ; −0,004·20 = −0,08 ; 1−0,08 = 0,92 ;
        //            300·0,92 = 276 W.
        let p = pv_temperature_corrected_power(300.0, -0.004, 45.0, 25.0);
        assert_relative_eq!(p, 276.0, epsilon = 1e-9);
        // À la température de référence, la puissance reste nominale.
        assert_relative_eq!(
            pv_temperature_corrected_power(300.0, -0.004, 25.0, 25.0),
            300.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn array_voltage_multiplies_module_voltage_by_series_count() {
        // V_module = 40 V, N_s = 12 : V_champ = 40·12 = 480 V.
        let v = pv_array_voltage(40.0, 12.0);
        assert_relative_eq!(v, 480.0, epsilon = 1e-9);
        // Un seul module en série redonne la tension du module.
        assert_relative_eq!(pv_array_voltage(40.0, 1.0), 40.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "l'irradiance E doit être > 0")]
    fn efficiency_rejects_zero_irradiance() {
        let _ = pv_efficiency(300.0, 0.0, 1.6);
    }
}
