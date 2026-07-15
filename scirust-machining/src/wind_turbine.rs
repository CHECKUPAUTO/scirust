//! **Éolienne** — extraction de la puissance du vent et **limite de Betz**.
//!
//! ```text
//! aire balayée      A = π·R²
//! puissance dispo   P_dispo = ½·ρ·A·v³
//! puissance récup.  P_récup = P_dispo·Cp
//! limite de Betz    Cp_max = 16 / 27 ≈ 0,593
//! rapidité          λ = ω·R / v
//! ```
//!
//! `ρ` masse volumique de l'air (kg·m⁻³), `A` aire balayée par le rotor (m²),
//! `R` rayon du rotor (m), `v` vitesse du vent (m·s⁻¹), `P_dispo` puissance
//! cinétique disponible du vent (W), `Cp` coefficient de puissance (rendement
//! aérodynamique réel, sans dimension), `P_récup` puissance récupérée à l'arbre
//! (W), `ω` vitesse angulaire du rotor (rad·s⁻¹), `λ` paramètre de rapidité
//! (sans dimension).
//!
//! **Convention** : unités SI ; la puissance croît en `v³`.
//!
//! **Limite honnête** : le coefficient de puissance `Cp` est une **donnée de
//! procédé fournie par l'appelant** (rendement aérodynamique réel, borné par la
//! limite de Betz `16/27`), de même que la masse volumique de l'air. Aucune
//! valeur « par défaut » n'est inventée ici. On ne modélise **ni** la courbe
//! `Cp(λ)`, **ni** le décrochage aérodynamique, **ni** les pertes mécaniques /
//! électriques de la chaîne de conversion. Complète [`crate::air_flow`].

use core::f64::consts::PI;

/// Aire balayée par le rotor `A = π·R²`.
///
/// `rotor_radius` rayon du rotor (m) ; renvoie l'aire balayée en m².
///
/// Panique si `rotor_radius <= 0`.
pub fn windturb_swept_area(rotor_radius: f64) -> f64 {
    assert!(
        rotor_radius > 0.0,
        "rayon du rotor strictement positif requis"
    );
    PI * rotor_radius * rotor_radius
}

/// Puissance cinétique disponible du vent traversant le disque rotor
/// `P_dispo = ½·ρ·A·v³`.
///
/// `air_density` masse volumique de l'air (kg·m⁻³), `swept_area` aire balayée
/// (m², p. ex. issue de [`windturb_swept_area`]), `wind_speed` vitesse du vent
/// (m·s⁻¹) ; renvoie la puissance disponible en W.
///
/// Panique si un paramètre est `<= 0`.
pub fn windturb_available_power(air_density: f64, swept_area: f64, wind_speed: f64) -> f64 {
    assert!(
        air_density > 0.0 && swept_area > 0.0 && wind_speed > 0.0,
        "masse volumique, aire balayée et vitesse du vent strictement positives requises"
    );
    0.5 * air_density * swept_area * wind_speed.powi(3)
}

/// Puissance récupérée à l'arbre `P_récup = P_dispo·Cp` (coefficient de
/// puissance fourni).
///
/// `available_power` puissance disponible du vent (W, p. ex. issue de
/// [`windturb_available_power`]), `power_coefficient` coefficient de puissance
/// `Cp` (sans dimension, physiquement borné par [`windturb_betz_limit`]) ;
/// renvoie la puissance récupérée en W.
///
/// Panique si `available_power < 0` ou si `power_coefficient` n'est pas dans
/// `[0, 1]`.
pub fn windturb_extracted_power(available_power: f64, power_coefficient: f64) -> f64 {
    assert!(
        available_power >= 0.0,
        "puissance disponible positive ou nulle requise"
    );
    assert!(
        (0.0..=1.0).contains(&power_coefficient),
        "coefficient de puissance dans l'intervalle [0, 1] requis"
    );
    available_power * power_coefficient
}

/// Limite de Betz : coefficient de puissance maximal théorique `Cp_max = 16/27`.
///
/// Renvoie la constante `16/27 ≈ 0,593` (sans dimension) ; ne panique jamais.
pub fn windturb_betz_limit() -> f64 {
    16.0 / 27.0
}

/// Paramètre de rapidité (tip speed ratio) `λ = ω·R / v`.
///
/// `rotor_angular_speed` vitesse angulaire du rotor (rad·s⁻¹), `rotor_radius`
/// rayon du rotor (m), `wind_speed` vitesse du vent (m·s⁻¹) ; renvoie `λ` (sans
/// dimension).
///
/// Panique si `rotor_angular_speed < 0`, ou si `rotor_radius <= 0`, ou si
/// `wind_speed <= 0`.
pub fn windturb_tip_speed_ratio(
    rotor_angular_speed: f64,
    rotor_radius: f64,
    wind_speed: f64,
) -> f64 {
    assert!(
        rotor_angular_speed >= 0.0,
        "vitesse angulaire du rotor positive ou nulle requise"
    );
    assert!(
        rotor_radius > 0.0 && wind_speed > 0.0,
        "rayon du rotor et vitesse du vent strictement positifs requis"
    );
    rotor_angular_speed * rotor_radius / wind_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn swept_area_scales_with_radius_squared() {
        // A ∝ R² : doubler le rayon quadruple l'aire balayée.
        let base = windturb_swept_area(20.0);
        let doubled = windturb_swept_area(40.0);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-9);
        // Cas chiffré : A(1) = π.
        assert_relative_eq!(windturb_swept_area(1.0), PI, epsilon = 1e-12);
    }

    #[test]
    fn available_power_scales_with_cube_of_wind_speed() {
        // P ∝ v³ : doubler la vitesse du vent multiplie la puissance par 8.
        let base = windturb_available_power(1.225, 100.0, 6.0);
        let doubled = windturb_available_power(1.225, 100.0, 12.0);
        assert_relative_eq!(doubled, 8.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn available_power_realistic_hand_calc() {
        // ρ = 1,225 kg/m³ ; A = 100 m² ; v = 12 m/s :
        // P = 0,5·1,225·100·12³ = 0,6125·100·1728 = 61,25·1728 = 105 840 W.
        let p = windturb_available_power(1.225, 100.0, 12.0);
        assert_relative_eq!(p, 105_840.0, epsilon = 1e-6);
    }

    #[test]
    fn extracted_power_bounded_by_available() {
        // P_récup = P_dispo·Cp, linéaire en Cp et jamais supérieur à P_dispo.
        let dispo = windturb_available_power(1.225, 100.0, 12.0);
        let recup = windturb_extracted_power(dispo, 0.45);
        assert_relative_eq!(recup, 0.45 * 105_840.0, epsilon = 1e-6);
        // Cp = 1 (borne haute admise) restitue la puissance disponible.
        assert_relative_eq!(windturb_extracted_power(dispo, 1.0), dispo, epsilon = 1e-9);
    }

    #[test]
    fn betz_limit_value_and_ordering() {
        // Cp_max = 16/27 ≈ 0,592592…
        assert_relative_eq!(windturb_betz_limit(), 16.0 / 27.0, epsilon = 1e-12);
        assert_relative_eq!(windturb_betz_limit(), 0.592_592_592_592, epsilon = 1e-9);
        // La limite de Betz est bien un coefficient de puissance admissible (< 1).
        assert!(windturb_betz_limit() < 1.0);
    }

    #[test]
    fn tip_speed_ratio_identity_and_case() {
        // λ = ω·R / v. Cas chiffré : ω = 2 rad/s, R = 40 m, v = 10 m/s → λ = 8.
        assert_relative_eq!(
            windturb_tip_speed_ratio(2.0, 40.0, 10.0),
            8.0,
            epsilon = 1e-12
        );
        // Rotor immobile → λ = 0.
        assert_relative_eq!(
            windturb_tip_speed_ratio(0.0, 40.0, 10.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "intervalle [0, 1]")]
    fn extracted_power_rejects_coefficient_above_one() {
        let _ = windturb_extracted_power(1000.0, 1.5);
    }
}
