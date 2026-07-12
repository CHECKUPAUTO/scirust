//! Vérins **pneumatiques** — effort théorique, effort utile (rendement) et
//! **consommation d'air** libre par course.
//!
//! ```text
//! effort sortie   F+ = p·(π/4)·D²
//! effort rentrée  F− = p·(π/4)·(D² − d²)      (côté tige, section annulaire)
//! effort utile    F_u = η·F                    (η rendement, pertes de frottement)
//! air libre/course V_lib = (π/4)·D²·L·(p_eff + p_atm)/p_atm
//! ```
//!
//! `p` pression **relative** (effective, Pa), `D` alésage, `d` tige (m), `L`
//! course (m), `η` rendement (≈ 0,85–0,95), `V_lib` volume d'air **détendu à
//! l'atmosphère** (m³) — la donnée qui dimensionne le compresseur. `p_atm`
//! pression atmosphérique (≈ 101 325 Pa).
//!
//! **Convention** : unités SI cohérentes ; pressions en Pa. **Limite honnête** :
//! effort **statique** (vérin en équilibre) ; l'air est traité comme
//! incompressible pour l'effort mais le calcul de consommation ramène le volume
//! balayé aux conditions atmosphériques (rapport de compression `p_abs/p_atm`).
//! Pour les vérins hydrauliques, voir [`crate::hydraulic_cylinders`].

use core::f64::consts::PI;

/// Effort théorique en poussée (sortie de tige) `F = p·(π/4)·D²`.
///
/// Panique si `bore_diameter <= 0`.
pub fn extend_force(gauge_pressure: f64, bore_diameter: f64) -> f64 {
    assert!(
        bore_diameter > 0.0,
        "l'alésage doit être strictement positif"
    );
    gauge_pressure * PI / 4.0 * bore_diameter * bore_diameter
}

/// Effort théorique en traction (rentrée) `F = p·(π/4)·(D² − d²)`.
///
/// Panique si `bore_diameter <= 0` ou `rod_diameter >= bore_diameter`.
pub fn retract_force(gauge_pressure: f64, bore_diameter: f64, rod_diameter: f64) -> f64 {
    assert!(
        bore_diameter > 0.0 && rod_diameter >= 0.0 && rod_diameter < bore_diameter,
        "0 ≤ d < D requis"
    );
    gauge_pressure * PI / 4.0 * (bore_diameter * bore_diameter - rod_diameter * rod_diameter)
}

/// Effort **utile** après rendement `F_u = η·F`.
///
/// Panique si `efficiency` sort de `]0, 1]`.
pub fn useful_force(theoretical_force: f64, efficiency: f64) -> f64 {
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    theoretical_force * efficiency
}

/// Consommation d'air **libre** (détendu à l'atmosphère) pour une course simple
/// `V_lib = (π/4)·D²·L·(p_eff + p_atm)/p_atm`.
///
/// Panique si un paramètre géométrique `<= 0` ou `atmospheric_pressure <= 0`.
pub fn free_air_per_stroke(
    bore_diameter: f64,
    stroke: f64,
    gauge_pressure: f64,
    atmospheric_pressure: f64,
) -> f64 {
    assert!(
        bore_diameter > 0.0 && stroke > 0.0 && atmospheric_pressure > 0.0 && gauge_pressure >= 0.0,
        "géométrie et pression atmosphérique strictement positives requises"
    );
    let swept = PI / 4.0 * bore_diameter * bore_diameter * stroke;
    swept * (gauge_pressure + atmospheric_pressure) / atmospheric_pressure
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn extend_force_from_pressure_and_area() {
        // Ø50 mm à 6 bar (600 kPa) → F = 600e3·π/4·0,05² ≈ 1178 N.
        let f = extend_force(600e3, 0.050);
        assert_relative_eq!(f, 600e3 * PI / 4.0 * 0.05 * 0.05, epsilon = 1e-6);
        assert!(f > 1170.0 && f < 1185.0);
    }

    #[test]
    fn retract_force_smaller_than_extend() {
        // La section annulaire (côté tige) est plus petite → effort de rentrée moindre.
        let fe = extend_force(600e3, 0.050);
        let fr = retract_force(600e3, 0.050, 0.020);
        assert!(fr < fe);
        // Rapport = (D²−d²)/D² = 1 − (d/D)².
        assert_relative_eq!(fr / fe, 1.0 - (0.020_f64 / 0.050).powi(2), epsilon = 1e-9);
    }

    #[test]
    fn efficiency_reduces_force() {
        assert_relative_eq!(useful_force(1000.0, 0.9), 900.0, epsilon = 1e-9);
    }

    #[test]
    fn free_air_scales_with_absolute_pressure_ratio() {
        // À 6 bar relatif (≈ 7 bar absolu), 1 course consomme ≈ 7× le volume balayé.
        let p_atm = 101_325.0;
        let v = free_air_per_stroke(0.050, 0.100, 600e3, p_atm);
        let swept = PI / 4.0 * 0.05 * 0.05 * 0.100;
        assert_relative_eq!(v, swept * (600e3 + p_atm) / p_atm, epsilon = 1e-12);
        assert!(v > swept * 6.9 && v < swept * 7.0);
    }

    #[test]
    #[should_panic(expected = "0 ≤ d < D")]
    fn rod_larger_than_bore_panics() {
        retract_force(600e3, 0.050, 0.060);
    }
}
