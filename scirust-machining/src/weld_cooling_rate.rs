//! **Refroidissement en soudage** — solution de Rosenthal pour une plaque
//! épaisse (source ponctuelle mobile, conduction tridimensionnelle) : temps de
//! refroidissement `t8/5` entre deux températures et vitesse moyenne associée.
//!
//! ```text
//! temps de refroidissement   t8/5 = Q/(2·π·k)·(1/(T_low−T0) − 1/(T_high−T0))   (s)
//! vitesse de refroidissement CR   = (T_high − T_low)/t                          (K/s)
//! ```
//!
//! `Q` apport de chaleur linéique **net** (J·m⁻¹, rendement d'arc déjà inclus),
//! `k` conductivité thermique du métal (W·m⁻¹·K⁻¹), `T0` température de
//! préchauffage / température initiale de la plaque (K ou °C), `T_high` et
//! `T_low` les deux températures bornant l'intervalle (K ou °C, typiquement 800
//! et 500 pour le `t8/5`), `t` temps de refroidissement (s), `CR` vitesse
//! moyenne de refroidissement (K·s⁻¹).
//!
//! **Convention** : SI ; toutes les températures dans la **même** échelle (les
//! écarts `T − T0` sont ce qui compte, donc K ou °C indifféremment tant que
//! l'échelle est homogène). **Limite honnête** : c'est la **solution analytique
//! de Rosenthal** pour une **plaque épaisse** (régime 3D, source ponctuelle
//! mobile en quasi-régime, propriétés thermiques constantes, pas de changement
//! de phase ni de pertes de surface) ; elle n'est **pas** valable pour une tôle
//! mince (régime 2D). L'apport linéique **net** `Q` (rendement d'arc compris),
//! la conductivité `k` et la température de préchauffage `T0` sont **fournis par
//! l'appelant** ; aucune valeur matériau ni rendement « par défaut » n'est
//! inventé. Voir [`crate::weld_heat_input`] (calcul de `Q`) et
//! [`crate::weld_preheat`] (choix de `T0`).

use core::f64::consts::PI;

/// Temps de refroidissement `t8/5 = Q/(2·π·k)·(1/(T_low−T0) − 1/(T_high−T0))`
/// (s), solution de Rosenthal pour une plaque épaisse (conduction 3D).
///
/// `heat_input` est l'apport linéique net `Q` (J·m⁻¹, rendement d'arc inclus),
/// `preheat_temp` la température initiale `T0`, `thermal_conductivity` la
/// conductivité `k` (W·m⁻¹·K⁻¹), `temp_high` et `temp_low` les bornes chaude et
/// froide de l'intervalle (même échelle que `T0`).
///
/// Panique si `heat_input <= 0`, si `thermal_conductivity <= 0`, si
/// `temp_high <= temp_low`, ou si `temp_low <= preheat_temp` (l'écart au
/// préchauffage doit rester strictement positif).
pub fn weldcool_cooling_time_thick_plate(
    heat_input: f64,
    preheat_temp: f64,
    thermal_conductivity: f64,
    temp_high: f64,
    temp_low: f64,
) -> f64 {
    assert!(
        heat_input > 0.0,
        "l'apport linéique net Q doit être strictement positif"
    );
    assert!(
        thermal_conductivity > 0.0,
        "la conductivité thermique k doit être strictement positive"
    );
    assert!(
        temp_high > temp_low,
        "T_high doit être strictement supérieure à T_low"
    );
    assert!(
        temp_low > preheat_temp,
        "T_low doit rester strictement au-dessus du préchauffage T0"
    );
    let dt_low = temp_low - preheat_temp;
    let dt_high = temp_high - preheat_temp;
    heat_input / (2.0 * PI * thermal_conductivity) * (1.0 / dt_low - 1.0 / dt_high)
}

/// Vitesse moyenne de refroidissement `CR = (T_high − T_low)/t` (K·s⁻¹) sur
/// l'intervalle de température considéré.
///
/// `temp_high` et `temp_low` sont les bornes chaude et froide (même échelle),
/// `cooling_time` le temps de refroidissement `t` (s).
///
/// Panique si `temp_high <= temp_low` ou si `cooling_time <= 0`.
pub fn weldcool_cooling_rate(temp_high: f64, temp_low: f64, cooling_time: f64) -> f64 {
    assert!(
        temp_high > temp_low,
        "T_high doit être strictement supérieure à T_low"
    );
    assert!(
        cooling_time > 0.0,
        "le temps de refroidissement t doit être strictement positif"
    );
    (temp_high - temp_low) / cooling_time
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cooling_time_proportional_to_heat_input() {
        // t ∝ Q : doubler l'apport linéique double le temps de refroidissement.
        let t1 = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 25.0, 800.0, 500.0);
        let t2 = weldcool_cooling_time_thick_plate(2.0e6, 20.0, 25.0, 800.0, 500.0);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn cooling_time_inverse_with_conductivity() {
        // t ∝ 1/k : doubler la conductivité halve le temps de refroidissement.
        let t1 = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 25.0, 800.0, 500.0);
        let t2 = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 50.0, 800.0, 500.0);
        assert_relative_eq!(t2, 0.5 * t1, epsilon = 1e-9);
    }

    #[test]
    fn preheat_increases_cooling_time() {
        // Préchauffer réduit les écarts T−T0, donc augmente t8/5 (monotonie).
        let cold = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 25.0, 800.0, 500.0);
        let warm = weldcool_cooling_time_thick_plate(1.0e6, 150.0, 25.0, 800.0, 500.0);
        assert!(warm > cold);
    }

    #[test]
    fn realistic_thick_plate_t85() {
        // Q = 1,0 MJ/m (1 kJ/mm), k = 25 W/m/K, T0 = 20 °C, 800→500 °C.
        // t8/5 = 1e6/(2π·25)·(1/480 − 1/780)
        //      = 6366,19772368·(300/374400) ≈ 5,10111997089408 s.
        let t = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 25.0, 800.0, 500.0);
        assert_relative_eq!(t, 5.101_119_970_894_08, epsilon = 1e-9);
    }

    #[test]
    fn cooling_rate_reciprocal_of_time() {
        // Identité CR·t = T_high − T_low : la vitesse déduite du t8/5 restitue
        // exactement l'écart de température sur l'intervalle.
        let t = weldcool_cooling_time_thick_plate(1.0e6, 20.0, 25.0, 800.0, 500.0);
        let cr = weldcool_cooling_rate(800.0, 500.0, t);
        assert_relative_eq!(cr * t, 300.0, epsilon = 1e-9);
        // Valeur chiffrée : 300 / 5,10111997089408 ≈ 58,8106144752 K/s.
        assert_relative_eq!(cr, 58.810_614_475_200_94, epsilon = 1e-9);
    }

    #[test]
    fn cooling_rate_inverse_with_time() {
        // CR ∝ 1/t : doubler la durée halve la vitesse moyenne.
        let cr1 = weldcool_cooling_rate(800.0, 500.0, 5.0);
        let cr2 = weldcool_cooling_rate(800.0, 500.0, 10.0);
        assert_relative_eq!(cr2, 0.5 * cr1, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "au-dessus du préchauffage T0")]
    fn preheat_above_low_temp_panics() {
        // T0 = 500 = T_low : écart nul, division par zéro évitée par assertion.
        weldcool_cooling_time_thick_plate(1.0e6, 500.0, 25.0, 800.0, 500.0);
    }
}
