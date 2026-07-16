//! Psychrométrie de l'**air humide** — humidité absolue (rapport de mélange),
//! humidité relative, pression partielle de vapeur, enthalpie spécifique,
//! pression de vapeur au point de rosée et volume spécifique de l'air humide.
//!
//! ```text
//! humidité absolue       W   = 0.622·pv/(P − pv)                    [kg/kg air sec]
//! humidité relative      phi = pv/psat                              [-]
//! pression partielle     pv  = W·P/(0.622 + W)                      [Pa]
//! enthalpie (réf 0 °C)   h   = 1.006·t + W·(2501 + 1.86·t)          [kJ/kg air sec]
//! pression au pt de rosée pd  = W·P/(0.622 + W)                     [Pa]
//! volume spécifique      v   = R_a·T·(1 + 1.608·W)/P                [m³/kg air sec]
//! ```
//!
//! `pv` pression partielle de vapeur d'eau [Pa], `P` pression totale [Pa],
//! `psat` pression de vapeur **saturante** à la température considérée [Pa],
//! `W` humidité absolue (rapport de mélange, kg d'eau par kg d'air **sec**,
//! sans dimension), `phi` humidité relative [sans dimension, ∈ 0..1], `t`
//! température [°C], `h` enthalpie spécifique de l'air humide rapportée à
//! l'air sec [kJ · kg air sec⁻¹], `pd` pression de vapeur au **point de rosée**
//! [Pa], `T` température absolue [K], `R_a` constante spécifique de l'**air
//! sec** [J · kg⁻¹ · K⁻¹], `v` volume spécifique rapporté à l'air sec
//! [m³ · kg air sec⁻¹]. Le facteur 0.622 = M_eau/M_air ≈ 18.015/28.97 et
//! 1.608 = 1/0.622.
//!
//! **Limite honnête** : la pression de vapeur **saturante** `psat` (via une
//! table psychrométrique ou une corrélation d'Antoine/Magnus **externe**) et la
//! constante spécifique de l'air sec `R_a` sont **FOURNIES** par l'appelant —
//! jamais supposées « par défaut ». Le mélange air sec + vapeur d'eau est
//! assimilé à des **gaz parfaits** ; la corrélation d'enthalpie est la forme
//! usuelle rapportée à l'**air sec** avec **référence 0 °C** (chaleur latente de
//! vaporisation 2501 kJ/kg, capacités 1.006 et 1.86 kJ·kg⁻¹·K⁻¹). Aucune
//! propriété physique ni aucun coefficient de transfert n'est **jamais** inventé
//! ici. Ce module **complète** `drying` (courbe de séchage) sans le dupliquer.

/// Humidité absolue (rapport de mélange) `W = 0.622·pv/(P − pv)`
/// [kg d'eau · kg air sec⁻¹, sans dimension], masse de vapeur d'eau par unité de
/// masse d'air **sec**.
///
/// `partial_vapor_pressure` (pv) pression partielle de vapeur d'eau [Pa] et
/// `total_pressure` (P) pression totale [Pa].
///
/// Panique si `partial_vapor_pressure < 0`, `total_pressure <= 0` ou si
/// `total_pressure <= partial_vapor_pressure` (dénominateur `P − pv` non
/// strictement positif).
pub fn psy_humidity_ratio(partial_vapor_pressure: f64, total_pressure: f64) -> f64 {
    assert!(
        partial_vapor_pressure >= 0.0,
        "pv ≥ 0 requis (pression partielle de vapeur)"
    );
    assert!(total_pressure > 0.0, "P > 0 requis (pression totale)");
    assert!(
        total_pressure > partial_vapor_pressure,
        "P > pv requis (dénominateur P − pv > 0)"
    );
    0.622 * partial_vapor_pressure / (total_pressure - partial_vapor_pressure)
}

/// Humidité relative `phi = pv/psat` [sans dimension, ∈ 0..1], rapport de la
/// pression partielle de vapeur à la pression saturante à la même température.
///
/// `partial_vapor_pressure` (pv) pression partielle de vapeur [Pa] et
/// `saturation_pressure` (psat) pression de vapeur saturante [Pa], **fournie**
/// par l'appelant (table ou corrélation externe).
///
/// Panique si `partial_vapor_pressure < 0` ou `saturation_pressure <= 0`.
pub fn psy_relative_humidity(partial_vapor_pressure: f64, saturation_pressure: f64) -> f64 {
    assert!(
        partial_vapor_pressure >= 0.0,
        "pv ≥ 0 requis (pression partielle de vapeur)"
    );
    assert!(
        saturation_pressure > 0.0,
        "psat > 0 requis (pression de vapeur saturante)"
    );
    partial_vapor_pressure / saturation_pressure
}

/// Pression partielle de vapeur `pv = W·P/(0.622 + W)` [Pa] reconstruite à
/// partir de l'humidité absolue. C'est l'opération réciproque de
/// [`psy_humidity_ratio`].
///
/// `humidity_ratio` (W) humidité absolue [kg eau · kg air sec⁻¹] et
/// `total_pressure` (P) pression totale [Pa].
///
/// Panique si `humidity_ratio < 0` ou `total_pressure <= 0`.
pub fn psy_vapor_pressure_from_humidity(humidity_ratio: f64, total_pressure: f64) -> f64 {
    assert!(humidity_ratio >= 0.0, "W ≥ 0 requis (humidité absolue)");
    assert!(total_pressure > 0.0, "P > 0 requis (pression totale)");
    humidity_ratio * total_pressure / (0.622 + humidity_ratio)
}

/// Enthalpie spécifique de l'air humide (référence 0 °C)
/// `h = 1.006·t + W·(2501 + 1.86·t)` [kJ · kg air sec⁻¹], somme de la chaleur
/// sensible de l'air sec et de l'enthalpie de la vapeur (latente + sensible).
///
/// `temperature_celsius` (t) température [°C] et `humidity_ratio` (W) humidité
/// absolue [kg eau · kg air sec⁻¹].
///
/// Panique si `humidity_ratio < 0`.
pub fn psy_enthalpy(temperature_celsius: f64, humidity_ratio: f64) -> f64 {
    assert!(humidity_ratio >= 0.0, "W ≥ 0 requis (humidité absolue)");
    1.006 * temperature_celsius + humidity_ratio * (2501.0 + 1.86 * temperature_celsius)
}

/// Pression de vapeur au **point de rosée** `pd = W·P/(0.622 + W)` [Pa]. Le
/// point de rosée est la température de saturation à la pression partielle de
/// vapeur ; sa pression saturante vaut donc exactement la pression partielle de
/// vapeur `pv` du mélange (même expression que
/// [`psy_vapor_pressure_from_humidity`]).
///
/// `humidity_ratio` (W) humidité absolue [kg eau · kg air sec⁻¹] et
/// `total_pressure` (P) pression totale [Pa].
///
/// Panique si `humidity_ratio < 0` ou `total_pressure <= 0`.
pub fn psy_dew_point_saturation_pressure(humidity_ratio: f64, total_pressure: f64) -> f64 {
    assert!(humidity_ratio >= 0.0, "W ≥ 0 requis (humidité absolue)");
    assert!(total_pressure > 0.0, "P > 0 requis (pression totale)");
    humidity_ratio * total_pressure / (0.622 + humidity_ratio)
}

/// Volume spécifique de l'air humide (rapporté à l'air sec)
/// `v = R_a·T·(1 + 1.608·W)/P` [m³ · kg air sec⁻¹], obtenu par la loi des gaz
/// parfaits appliquée au mélange air sec + vapeur.
///
/// `temperature_kelvin` (T) température absolue [K], `humidity_ratio` (W)
/// humidité absolue [kg eau · kg air sec⁻¹], `total_pressure` (P) pression
/// totale [Pa] et `dry_air_gas_constant` (R_a) constante spécifique de l'air sec
/// [J · kg⁻¹ · K⁻¹], **fournie** par l'appelant.
///
/// Panique si `temperature_kelvin <= 0`, `humidity_ratio < 0`,
/// `total_pressure <= 0` ou `dry_air_gas_constant <= 0`.
pub fn psy_specific_volume(
    temperature_kelvin: f64,
    humidity_ratio: f64,
    total_pressure: f64,
    dry_air_gas_constant: f64,
) -> f64 {
    assert!(
        temperature_kelvin > 0.0,
        "T > 0 requis (température absolue en K)"
    );
    assert!(humidity_ratio >= 0.0, "W ≥ 0 requis (humidité absolue)");
    assert!(total_pressure > 0.0, "P > 0 requis (pression totale)");
    assert!(
        dry_air_gas_constant > 0.0,
        "R_a > 0 requis (constante spécifique de l'air sec)"
    );
    dry_air_gas_constant * temperature_kelvin * (1.0 + 1.608 * humidity_ratio) / total_pressure
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn humidity_ratio_and_vapor_pressure_are_reciprocal() {
        // pv = 2000 Pa, P = 101325 Pa.
        // W = 0.622·2000/(101325 − 2000) = 1244/99325 = 0.012524540649383338.
        let w = psy_humidity_ratio(2000.0_f64, 101325.0_f64);
        assert_relative_eq!(w, 0.012524540649383338, max_relative = 1e-3);
        // Réciprocité : reconstruire pv depuis W redonne 2000 Pa.
        assert_relative_eq!(
            psy_vapor_pressure_from_humidity(w, 101325.0_f64),
            2000.0,
            max_relative = 1e-9
        );
        // Le point de rosée a la même pression saturante que pv.
        assert_relative_eq!(
            psy_dew_point_saturation_pressure(w, 101325.0_f64),
            psy_vapor_pressure_from_humidity(w, 101325.0_f64),
            max_relative = 1e-12
        );
    }

    #[test]
    fn relative_humidity_bounds() {
        // À saturation (pv = psat) l'humidité relative vaut 1.
        assert_relative_eq!(
            psy_relative_humidity(3169.0_f64, 3169.0_f64),
            1.0,
            max_relative = 1e-12
        );
        // Air parfaitement sec (pv = 0) ⇒ phi = 0.
        assert_relative_eq!(
            psy_relative_humidity(0.0_f64, 3169.0_f64),
            0.0,
            epsilon = 1e-12
        );
        // pv = 2000 Pa, psat = 3169 Pa ⇒ phi = 2000/3169 = 0.6311139160618492.
        assert_relative_eq!(
            psy_relative_humidity(2000.0_f64, 3169.0_f64),
            0.6311139160618492,
            max_relative = 1e-3
        );
    }

    #[test]
    fn enthalpy_computed_case_and_dry_air_limit() {
        // t = 25 °C, W = 0.01 :
        // h = 1.006·25 + 0.01·(2501 + 1.86·25)
        //   = 25.15 + 0.01·2547.5 = 25.15 + 25.475 = 50.625 kJ/kg air sec.
        assert_relative_eq!(
            psy_enthalpy(25.0_f64, 0.01_f64),
            50.625,
            max_relative = 1e-3
        );
        // Air sec (W = 0) à 0 °C : enthalpie de référence nulle.
        assert_relative_eq!(psy_enthalpy(0.0_f64, 0.0_f64), 0.0, epsilon = 1e-12);
        // Air sec (W = 0) : h = 1.006·t (chaleur sensible seule).
        assert_relative_eq!(
            psy_enthalpy(30.0_f64, 0.0_f64),
            1.006 * 30.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn vapor_pressure_proportional_to_total_pressure() {
        // pv = W·P/(0.622 + W) ∝ P à W fixé : doubler P double pv.
        let base = psy_vapor_pressure_from_humidity(0.01_f64, 100000.0_f64);
        let doubled = psy_vapor_pressure_from_humidity(0.01_f64, 200000.0_f64);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn specific_volume_computed_case_and_dry_air() {
        // T = 300 K, W = 0.01, P = 101325 Pa, R_a = 287 J·kg⁻¹·K⁻¹ :
        // v = 287·300·(1 + 1.608·0.01)/101325
        //   = 86100·1.01608/101325 = 87484.488/101325 = 0.8634047668393784.
        assert_relative_eq!(
            psy_specific_volume(300.0_f64, 0.01_f64, 101325.0_f64, 287.0_f64),
            0.8634047668393784,
            max_relative = 1e-3
        );
        // Air sec (W = 0) : on retrouve v = R_a·T/P (loi des gaz parfaits).
        assert_relative_eq!(
            psy_specific_volume(300.0_f64, 0.0_f64, 101325.0_f64, 287.0_f64),
            287.0 * 300.0 / 101325.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "P > pv requis")]
    fn humidity_ratio_panics_when_total_below_partial() {
        // P ≤ pv ⇒ dénominateur P − pv ≤ 0 : entrée invalide.
        let _ = psy_humidity_ratio(101325.0_f64, 2000.0_f64);
    }
}
