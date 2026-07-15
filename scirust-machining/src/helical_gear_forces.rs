//! Efforts sur la denture d'un **engrenage hélicoïdal** — décomposition de
//! l'effort au diamètre primitif en composantes tangentielle, radiale et axiale.
//!
//! ```text
//! effort tangentiel   Ft = P / v                       (P puissance, v vitesse primitive)
//! effort radial       Fr = Ft·tan(αn) / cos(β)
//! effort axial        Fa = Ft·tan(β)
//! effort normal       Fn = Ft / (cos(αn)·cos(β))
//! identité            Fn² = Ft² + Fr² + Fa²
//! ```
//!
//! `P` puissance transmise (W), `v` vitesse au cercle primitif (m/s),
//! `Ft`/`Fr`/`Fa`/`Fn` efforts tangentiel/radial/axial/normal (N), `αn` angle de
//! pression **normal** (rad), `β` angle d'hélice (rad).
//!
//! **Convention** : unités SI (W, m/s, N, rad). **Limite honnête** : denture
//! hélicoïdale **standard**, effort supposé appliqué au **diamètre primitif**,
//! frottement **négligé**. L'angle de pression normal `αn`, l'angle d'hélice `β`,
//! la puissance et la vitesse sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » de matériau, de procédé, de géométrie ou de coût n'est inventée
//! ici. Complète [`crate::gears`].

/// Effort **tangentiel** (utile) au diamètre primitif `Ft = P/v` (N).
///
/// `power` en W, `pitch_line_velocity` en m/s.
///
/// Panique si `pitch_line_velocity <= 0`.
pub fn helical_tangential_force(power: f64, pitch_line_velocity: f64) -> f64 {
    assert!(
        pitch_line_velocity > 0.0,
        "la vitesse primitive doit être strictement positive"
    );
    power / pitch_line_velocity
}

/// Effort **radial** (séparateur) `Fr = Ft·tan(αn)/cos(β)` (N).
///
/// `tangential` en N, `normal_pressure_angle_rad` et `helix_angle_rad` en rad.
///
/// Panique si `normal_pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `helix_angle_rad` n'est pas dans `[0, π/2[`.
pub fn helical_radial_force(
    tangential: f64,
    normal_pressure_angle_rad: f64,
    helix_angle_rad: f64,
) -> f64 {
    assert_normal_pressure_angle(normal_pressure_angle_rad);
    assert_helix_angle(helix_angle_rad);
    tangential * normal_pressure_angle_rad.tan() / helix_angle_rad.cos()
}

/// Effort **axial** (poussée) `Fa = Ft·tan(β)` (N).
///
/// `tangential` en N, `helix_angle_rad` en rad.
///
/// Panique si `helix_angle_rad` n'est pas dans `[0, π/2[`.
pub fn helical_axial_force(tangential: f64, helix_angle_rad: f64) -> f64 {
    assert_helix_angle(helix_angle_rad);
    tangential * helix_angle_rad.tan()
}

/// Effort **normal** total à la denture `Fn = Ft/(cos(αn)·cos(β))` (N).
///
/// C'est le module de l'effort résultant : `Fn² = Ft² + Fr² + Fa²`.
///
/// Panique si `normal_pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `helix_angle_rad` n'est pas dans `[0, π/2[`.
pub fn helical_normal_force(
    tangential: f64,
    normal_pressure_angle_rad: f64,
    helix_angle_rad: f64,
) -> f64 {
    assert_normal_pressure_angle(normal_pressure_angle_rad);
    assert_helix_angle(helix_angle_rad);
    tangential / (normal_pressure_angle_rad.cos() * helix_angle_rad.cos())
}

/// Module de l'effort **résultant** `F = √(Ft² + Fr² + Fa²)` (N).
///
/// Doit coïncider avec [`helical_normal_force`] par construction.
///
/// Panique si l'un des efforts n'est pas fini.
pub fn helical_resultant_force(tangential: f64, radial: f64, axial: f64) -> f64 {
    assert!(
        tangential.is_finite() && radial.is_finite() && axial.is_finite(),
        "les efforts doivent être finis"
    );
    (tangential * tangential + radial * radial + axial * axial).sqrt()
}

#[inline]
fn assert_normal_pressure_angle(normal_pressure_angle_rad: f64) {
    assert!(
        (0.0..core::f64::consts::FRAC_PI_2).contains(&normal_pressure_angle_rad),
        "l'angle de pression normal doit être dans [0, π/2["
    );
}

#[inline]
fn assert_helix_angle(helix_angle_rad: f64) {
    assert!(
        (0.0..core::f64::consts::FRAC_PI_2).contains(&helix_angle_rad),
        "l'angle d'hélice doit être dans [0, π/2["
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangential_force_is_power_over_velocity() {
        // P = 1000 W à v = 2 m/s → Ft = 500 N.
        assert_relative_eq!(
            helical_tangential_force(1000.0, 2.0),
            500.0,
            epsilon = 1e-12
        );
        // Proportionnalité en puissance : doubler P double Ft.
        let f1 = helical_tangential_force(1000.0, 2.0);
        let f2 = helical_tangential_force(2000.0, 2.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-12);
    }

    #[test]
    fn components_satisfy_pythagorean_identity() {
        // Fn² = Ft² + Fr² + Fa² par construction trigonométrique.
        let ft = 500.0;
        let alpha_n = 20.0_f64.to_radians();
        let beta = 25.0_f64.to_radians();
        let fr = helical_radial_force(ft, alpha_n, beta);
        let fa = helical_axial_force(ft, beta);
        let fn_ = helical_normal_force(ft, alpha_n, beta);
        assert_relative_eq!(fr * fr + fa * fa + ft * ft, fn_ * fn_, epsilon = 1e-9);
        // La résultante explicite doit égaler l'effort normal.
        assert_relative_eq!(helical_resultant_force(ft, fr, fa), fn_, epsilon = 1e-9);
    }

    #[test]
    fn spur_gear_limit_when_helix_angle_is_zero() {
        // β = 0 : denture droite → pas d'effort axial, Fr = Ft·tan(αn).
        let ft = 800.0;
        let alpha_n = 20.0_f64.to_radians();
        assert_relative_eq!(helical_axial_force(ft, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            helical_radial_force(ft, alpha_n, 0.0),
            ft * alpha_n.tan(),
            epsilon = 1e-12
        );
        // À β = 0, Fn se réduit au cas denture droite Ft/cos(αn).
        assert_relative_eq!(
            helical_normal_force(ft, alpha_n, 0.0),
            ft / alpha_n.cos(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn axial_force_grows_with_helix_angle() {
        // Fa = Ft·tan(β) est croissant en β sur [0, π/2[.
        let ft = 600.0;
        let low = helical_axial_force(ft, 10.0_f64.to_radians());
        let high = helical_axial_force(ft, 30.0_f64.to_radians());
        assert!(high > low);
        // Proportionnalité en effort tangentiel.
        assert_relative_eq!(
            helical_axial_force(2.0 * ft, 30.0_f64.to_radians()),
            2.0 * high,
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_gearbox_case() {
        // Cas chiffré : P = 15 kW, v = 5 m/s, αn = 20°, β = 30°.
        let ft = helical_tangential_force(15_000.0, 5.0);
        assert_relative_eq!(ft, 3000.0, epsilon = 1e-9);
        let alpha_n = 20.0_f64.to_radians();
        let beta = 30.0_f64.to_radians();
        // Fa = Ft·tan(30°) = 3000·0.577350... ≈ 1732.05 N.
        assert_relative_eq!(
            helical_axial_force(ft, beta),
            3000.0 * beta.tan(),
            epsilon = 1e-9
        );
        // Fr = Ft·tan(20°)/cos(30°).
        assert_relative_eq!(
            helical_radial_force(ft, alpha_n, beta),
            3000.0 * alpha_n.tan() / beta.cos(),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "vitesse primitive doit être strictement positive")]
    fn zero_velocity_panics() {
        helical_tangential_force(1000.0, 0.0);
    }
}
