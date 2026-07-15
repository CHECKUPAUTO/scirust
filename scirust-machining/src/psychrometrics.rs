//! **Psychrométrie de l'air humide** — teneur en eau, humidité relative et
//! enthalpie spécifique d'un mélange air sec + vapeur d'eau assimilé à des gaz
//! parfaits.
//!
//! ```text
//! teneur en eau        w   = 0,622·pv / (p − pv)
//! pression de vapeur   pv  = w·p / (0,622 + w)        (réciproque)
//! humidité relative    φ   = pv / psat
//! enthalpie spécifique h   = 1,006·T + w·(2501 + 1,86·T)
//! ```
//!
//! `pv` pression partielle de vapeur d'eau (Pa), `p` pression totale **absolue**
//! (Pa), `psat` pression de vapeur **saturante** à la température considérée (Pa),
//! `w` teneur en eau / humidité absolue (kg d'eau par kg d'air **sec**), `φ`
//! humidité relative (sans dimension, 0 à 1), `T` température sèche (°C), `h`
//! enthalpie spécifique du mélange rapportée au kg d'air **sec** (kJ·kg⁻¹).
//!
//! **Convention** : rapport des masses molaires eau/air sec `≈ 0,622`, chaleur
//! massique de l'air sec `1,006 kJ·kg⁻¹·K⁻¹`, chaleur latente de vaporisation à
//! 0 °C `2501 kJ·kg⁻¹` et chaleur massique de la vapeur `1,86 kJ·kg⁻¹·K⁻¹` ;
//! température en **degrés Celsius**, pressions en Pa, unités SI par ailleurs.
//!
//! **Limite honnête** : air sec et vapeur d'eau supposés **gaz parfaits** à la
//! **pression atmosphérique standard**, coefficients thermodynamiques figés aux
//! valeurs usuelles ci-dessus (pas de dépendance en pression totale). Les
//! pressions partielle et saturante — ainsi que la pression totale et la
//! température de procédé — sont des **données fournies par l'appelant** (tables
//! ou corrélation de saturation externe) : aucune valeur « par défaut » n'est
//! inventée ici. Complète [`crate::air_receiver`] et [`crate::air_flow`].

/// Rapport des masses molaires eau/air sec (18,015 / 28,966 ≈ 0,622).
const WATER_AIR_MASS_RATIO: f64 = 0.622;

/// Chaleur massique de l'air sec (kJ·kg⁻¹·K⁻¹).
const DRY_AIR_SPECIFIC_HEAT: f64 = 1.006;

/// Chaleur latente de vaporisation de l'eau à 0 °C (kJ·kg⁻¹).
const WATER_LATENT_HEAT_0C: f64 = 2501.0;

/// Chaleur massique de la vapeur d'eau (kJ·kg⁻¹·K⁻¹).
const WATER_VAPOR_SPECIFIC_HEAT: f64 = 1.86;

/// Teneur en eau (humidité absolue) `w = 0,622·pv / (p − pv)`.
///
/// `partial_vapor_pressure` pression partielle de vapeur `pv` (Pa),
/// `total_pressure` pression totale absolue `p` (Pa) ; renvoie la teneur en eau
/// en kg d'eau par kg d'air **sec**. Réciproque de
/// [`psychro_partial_pressure_from_ratio`].
///
/// Panique si `partial_vapor_pressure < 0`, si `total_pressure <= 0` ou si
/// `partial_vapor_pressure >= total_pressure`.
pub fn psychro_humidity_ratio(partial_vapor_pressure: f64, total_pressure: f64) -> f64 {
    assert!(
        partial_vapor_pressure >= 0.0,
        "pression partielle de vapeur négative interdite"
    );
    assert!(
        total_pressure > 0.0,
        "pression totale strictement positive requise"
    );
    assert!(
        partial_vapor_pressure < total_pressure,
        "pression partielle strictement inférieure à la pression totale requise"
    );
    WATER_AIR_MASS_RATIO * partial_vapor_pressure / (total_pressure - partial_vapor_pressure)
}

/// Pression partielle de vapeur déduite de la teneur en eau
/// `pv = w·p / (0,622 + w)`.
///
/// `humidity_ratio` teneur en eau `w` (kg d'eau par kg d'air sec),
/// `total_pressure` pression totale absolue `p` (Pa) ; renvoie la pression
/// partielle de vapeur en Pa. Réciproque de [`psychro_humidity_ratio`].
///
/// Panique si `humidity_ratio < 0` ou si `total_pressure <= 0`.
pub fn psychro_partial_pressure_from_ratio(humidity_ratio: f64, total_pressure: f64) -> f64 {
    assert!(humidity_ratio >= 0.0, "teneur en eau négative interdite");
    assert!(
        total_pressure > 0.0,
        "pression totale strictement positive requise"
    );
    humidity_ratio * total_pressure / (WATER_AIR_MASS_RATIO + humidity_ratio)
}

/// Humidité relative `φ = pv / psat`.
///
/// `partial_vapor_pressure` pression partielle de vapeur `pv` (Pa),
/// `saturation_pressure` pression de vapeur saturante `psat` (Pa) à la
/// température considérée ; renvoie l'humidité relative sans dimension
/// (`1,0` = air saturé). Réciproque de
/// [`psychro_partial_pressure_from_humidity`].
///
/// Panique si `partial_vapor_pressure < 0` ou si `saturation_pressure <= 0`.
pub fn psychro_relative_humidity(partial_vapor_pressure: f64, saturation_pressure: f64) -> f64 {
    assert!(
        partial_vapor_pressure >= 0.0,
        "pression partielle de vapeur négative interdite"
    );
    assert!(
        saturation_pressure > 0.0,
        "pression de vapeur saturante strictement positive requise"
    );
    partial_vapor_pressure / saturation_pressure
}

/// Pression partielle de vapeur déduite de l'humidité relative `pv = φ·psat`.
///
/// `relative_humidity` humidité relative `φ` (sans dimension, 0 à 1),
/// `saturation_pressure` pression de vapeur saturante `psat` (Pa) ; renvoie la
/// pression partielle de vapeur en Pa. Réciproque de
/// [`psychro_relative_humidity`].
///
/// Panique si `relative_humidity` hors de `[0, 1]` ou si
/// `saturation_pressure <= 0`.
pub fn psychro_partial_pressure_from_humidity(
    relative_humidity: f64,
    saturation_pressure: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&relative_humidity),
        "humidité relative attendue dans l'intervalle [0, 1]"
    );
    assert!(
        saturation_pressure > 0.0,
        "pression de vapeur saturante strictement positive requise"
    );
    relative_humidity * saturation_pressure
}

/// Enthalpie spécifique de l'air humide (par kg d'air sec)
/// `h = 1,006·T + w·(2501 + 1,86·T)`.
///
/// `dry_bulb_temp` température sèche `T` (°C), `humidity_ratio` teneur en eau `w`
/// (kg d'eau par kg d'air sec) ; renvoie l'enthalpie en kJ par kg d'air **sec**.
/// La référence est l'air sec et l'eau liquide à 0 °C (`h = 0` pour `T = 0` et
/// `w = 0`).
///
/// Panique si `humidity_ratio < 0`.
pub fn psychro_specific_enthalpy_moist_air(dry_bulb_temp: f64, humidity_ratio: f64) -> f64 {
    assert!(humidity_ratio >= 0.0, "teneur en eau négative interdite");
    DRY_AIR_SPECIFIC_HEAT * dry_bulb_temp
        + humidity_ratio * (WATER_LATENT_HEAT_0C + WATER_VAPOR_SPECIFIC_HEAT * dry_bulb_temp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn humidity_ratio_and_partial_pressure_are_reciprocal() {
        // Un aller-retour w → pv → w doit restituer la pression partielle.
        let (pv, p) = (2000.0_f64, 101_325.0);
        let w = psychro_humidity_ratio(pv, p);
        let pv_back = psychro_partial_pressure_from_ratio(w, p);
        assert_relative_eq!(pv_back, pv, epsilon = 1e-9);
    }

    #[test]
    fn relative_humidity_and_partial_pressure_are_reciprocal() {
        // φ = pv/psat puis pv = φ·psat doit boucler exactement.
        let (pv, psat) = (1500.0_f64, 3169.0);
        let phi = psychro_relative_humidity(pv, psat);
        let pv_back = psychro_partial_pressure_from_humidity(phi, psat);
        assert_relative_eq!(pv_back, pv, epsilon = 1e-9);
    }

    #[test]
    fn saturated_air_gives_unit_relative_humidity() {
        // À saturation, pv = psat donc φ = 1.
        let phi = psychro_relative_humidity(3169.0, 3169.0);
        assert_relative_eq!(phi, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dry_air_enthalpy_matches_hand_calc() {
        // Air sec (w = 0) à 25 °C : h = 1,006·25 = 25,15 kJ/kg.
        let h = psychro_specific_enthalpy_moist_air(25.0, 0.0);
        assert_relative_eq!(h, 25.15, epsilon = 1e-12);
    }

    #[test]
    fn moist_air_enthalpy_matches_hand_calc() {
        // T = 25 °C, w = 0,010 kg/kg :
        // h = 1,006·25 + 0,010·(2501 + 1,86·25)
        //   = 25,15 + 0,010·2547,5 = 25,15 + 25,475 = 50,625 kJ/kg.
        let h = psychro_specific_enthalpy_moist_air(25.0, 0.010);
        assert_relative_eq!(h, 50.625, epsilon = 1e-9);
    }

    #[test]
    fn realistic_humidity_ratio() {
        // pv = 2000 Pa, p = 101 325 Pa :
        // w = 0,622·2000 / (101325 − 2000) = 1244 / 99325 ≈ 0,0125245 kg/kg.
        let w = psychro_humidity_ratio(2000.0, 101_325.0);
        assert_relative_eq!(w, 1244.0 / 99_325.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pression partielle strictement inférieure")]
    fn humidity_ratio_rejects_pv_above_total() {
        let _ = psychro_humidity_ratio(120_000.0, 101_325.0);
    }
}
