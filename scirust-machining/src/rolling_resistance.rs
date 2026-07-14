//! **Résistance au roulement** — coefficient, effort résistant, coefficient issu
//! du bras de résistance et puissance dissipée.
//!
//! ```text
//! effort résistant   F_r = C_rr·F_N
//! coeff. (bras)      C_rr = b/R                       (b bras de résistance, R rayon)
//! puissance          P = F_r·v
//! résistance en pente F = C_rr·F_N·cos θ + F_N·sin θ   (roulement + gravité)
//! ```
//!
//! `C_rr` coefficient de résistance au roulement (sans dimension), `F_N` charge
//! normale (N), `F_r` effort résistant (N), `b` bras de résistance (décalage de la
//! réaction normale vers l'avant, m), `R` rayon de roulement (m), `v` vitesse
//! (m/s), `θ` angle de la pente (rad).
//!
//! **Convention** : SI ; angle en radians. **Limite honnête** : modèle à
//! coefficient `C_rr` **constant** (indépendant de la vitesse et de la charge, ce
//! qui est une approximation) ; `C_rr` (ou le bras `b`) provient d'essais fournis
//! par l'appelant. Distinct du frottement sec de glissement de
//! [`crate::friction`].

/// Effort résistant au roulement `F_r = C_rr·F_N`.
///
/// Panique si un paramètre `< 0`.
pub fn rolling_resistance_force(coefficient: f64, normal_load: f64) -> f64 {
    assert!(
        coefficient >= 0.0 && normal_load >= 0.0,
        "C_rr ≥ 0 et F_N ≥ 0 requis"
    );
    coefficient * normal_load
}

/// Coefficient de résistance issu du bras de résistance `C_rr = b/R`.
///
/// Panique si `resistance_arm < 0` ou `rolling_radius <= 0`.
pub fn coefficient_from_arm(resistance_arm: f64, rolling_radius: f64) -> f64 {
    assert!(
        resistance_arm >= 0.0 && rolling_radius > 0.0,
        "b ≥ 0 et R > 0 requis"
    );
    resistance_arm / rolling_radius
}

/// Puissance dissipée par la résistance au roulement `P = F_r·v`.
///
/// Panique si une grandeur `< 0`.
pub fn rolling_power(resistance_force: f64, velocity: f64) -> f64 {
    assert!(
        resistance_force >= 0.0 && velocity >= 0.0,
        "F_r ≥ 0 et v ≥ 0 requis"
    );
    resistance_force * velocity
}

/// Effort total pour avancer sur une pente
/// `F = C_rr·F_N·cos θ + F_N·sin θ` (roulement + composante de gravité).
///
/// Panique si `coefficient < 0` ou `normal_load < 0`.
pub fn resistance_on_grade(coefficient: f64, normal_load: f64, grade_angle_rad: f64) -> f64 {
    assert!(
        coefficient >= 0.0 && normal_load >= 0.0,
        "C_rr ≥ 0 et F_N ≥ 0 requis"
    );
    coefficient * normal_load * grade_angle_rad.cos() + normal_load * grade_angle_rad.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn resistance_force_from_coefficient() {
        // C_rr=0,015, F_N=10 000 N → F_r = 150 N.
        assert_relative_eq!(
            rolling_resistance_force(0.015, 10_000.0),
            150.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn coefficient_from_resistance_arm() {
        // b=0,3 mm, R=0,3 m → C_rr = 0,001.
        assert_relative_eq!(coefficient_from_arm(0.3e-3, 0.3), 0.001, epsilon = 1e-12);
    }

    #[test]
    fn power_grows_with_speed() {
        assert!(rolling_power(150.0, 25.0) > rolling_power(150.0, 10.0));
        assert_relative_eq!(rolling_power(150.0, 20.0), 3000.0, epsilon = 1e-9);
    }

    #[test]
    fn flat_grade_reduces_to_rolling_only() {
        // À θ=0 : F = C_rr·F_N (pas de composante de gravité).
        assert_relative_eq!(
            resistance_on_grade(0.015, 10_000.0, 0.0),
            rolling_resistance_force(0.015, 10_000.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn grade_adds_gravity_component() {
        // Sur pente, la composante F_N·sin θ domine le terme de roulement.
        let f = resistance_on_grade(0.015, 10_000.0, 0.1);
        assert!(f > resistance_on_grade(0.015, 10_000.0, 0.0));
        assert_relative_eq!(
            f,
            0.015 * 10_000.0 * (0.1f64).cos() + 10_000.0 * (0.1f64).sin(),
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "R > 0")]
    fn zero_radius_panics() {
        coefficient_from_arm(0.3e-3, 0.0);
    }
}
