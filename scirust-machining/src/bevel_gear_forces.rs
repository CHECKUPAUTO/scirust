//! Efforts sur un **engrenage conique droit** — décomposition de l'effort au
//! rayon moyen en composantes tangentielle, radiale et axiale.
//!
//! ```text
//! effort tangentiel   Ft = P / v            (P puissance, v vitesse primitive)
//! effort radial       Fr = Ft·tan(α)·cos(γ)
//! effort axial        Fa = Ft·tan(α)·sin(γ)
//! module (radial+axial) Fs = Ft·tan(α)   →   Fr² + Fa² = Fs²
//! ```
//!
//! `P` puissance transmise (W), `v` vitesse au cercle primitif moyen (m/s),
//! `Ft`/`Fr`/`Fa` efforts tangentiel/radial/axial (N), `α` angle de pression
//! (rad), `γ` angle de cône primitif (rad, voir [`bevel_worm_gears`]).
//!
//! **Convention** : unités SI (W, m/s, N, rad). **Limite honnête** : denture
//! conique **droite**, effort supposé appliqué au **rayon moyen**, frottement
//! **négligé**. L'angle de pression `α`, l'angle de cône `γ`, la puissance et la
//! vitesse sont **fournis par l'appelant** ; aucune valeur « par défaut » de
//! matériau, de procédé ou de géométrie n'est inventée ici. Complète
//! [`bevel_worm_gears`] qui fournit les angles de cône.

/// Effort **tangentiel** (utile) au rayon moyen `Ft = P/v` (N).
///
/// `power` en W, `pitch_line_velocity` en m/s.
///
/// Panique si `pitch_line_velocity <= 0`.
pub fn bevel_tangential_force(power: f64, pitch_line_velocity: f64) -> f64 {
    assert!(
        pitch_line_velocity > 0.0,
        "la vitesse primitive doit être strictement positive"
    );
    power / pitch_line_velocity
}

/// Effort **radial** (séparateur) `Fr = Ft·tan(α)·cos(γ)` (N).
///
/// `tangential` en N, `pressure_angle_rad` et `pitch_cone_angle_rad` en rad.
///
/// Panique si `pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `pitch_cone_angle_rad` n'est pas dans `[0, π/2]`.
pub fn bevel_radial_force(
    tangential: f64,
    pressure_angle_rad: f64,
    pitch_cone_angle_rad: f64,
) -> f64 {
    assert_pressure_angle(pressure_angle_rad);
    assert_cone_angle(pitch_cone_angle_rad);
    tangential * pressure_angle_rad.tan() * pitch_cone_angle_rad.cos()
}

/// Effort **axial** (poussée) `Fa = Ft·tan(α)·sin(γ)` (N).
///
/// `tangential` en N, `pressure_angle_rad` et `pitch_cone_angle_rad` en rad.
///
/// Panique si `pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `pitch_cone_angle_rad` n'est pas dans `[0, π/2]`.
pub fn bevel_axial_force(
    tangential: f64,
    pressure_angle_rad: f64,
    pitch_cone_angle_rad: f64,
) -> f64 {
    assert_pressure_angle(pressure_angle_rad);
    assert_cone_angle(pitch_cone_angle_rad);
    tangential * pressure_angle_rad.tan() * pitch_cone_angle_rad.sin()
}

/// Effort **séparateur** (module radial+axial) `Fs = Ft·tan(α)` (N).
///
/// Vérifie l'identité `Fr² + Fa² = Fs²` puisque `cos²γ + sin²γ = 1`.
///
/// Panique si `pressure_angle_rad` n'est pas dans `[0, π/2[`.
pub fn bevel_separating_force(tangential: f64, pressure_angle_rad: f64) -> f64 {
    assert_pressure_angle(pressure_angle_rad);
    tangential * pressure_angle_rad.tan()
}

/// Module de l'effort **résultant** total `F = √(Ft² + Fr² + Fa²)` (N).
///
/// Panique si l'un des efforts n'est pas fini.
pub fn bevel_resultant_force(tangential: f64, radial: f64, axial: f64) -> f64 {
    assert!(
        tangential.is_finite() && radial.is_finite() && axial.is_finite(),
        "les efforts doivent être finis"
    );
    (tangential * tangential + radial * radial + axial * axial).sqrt()
}

#[inline]
fn assert_pressure_angle(pressure_angle_rad: f64) {
    assert!(
        (0.0..core::f64::consts::FRAC_PI_2).contains(&pressure_angle_rad),
        "l'angle de pression doit être dans [0, π/2["
    );
}

#[inline]
fn assert_cone_angle(pitch_cone_angle_rad: f64) {
    assert!(
        (0.0..=core::f64::consts::FRAC_PI_2).contains(&pitch_cone_angle_rad),
        "l'angle de cône primitif doit être dans [0, π/2]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangential_force_is_power_over_velocity() {
        // P = 1000 W à v = 2 m/s → Ft = 500 N.
        assert_relative_eq!(bevel_tangential_force(1000.0, 2.0), 500.0, epsilon = 1e-12);
        // Proportionnalité en puissance : doubler P double Ft.
        let f1 = bevel_tangential_force(1000.0, 2.0);
        let f2 = bevel_tangential_force(2000.0, 2.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-12);
    }

    #[test]
    fn radial_and_axial_split_the_separating_force() {
        // Fr = Fs·cos γ, Fa = Fs·sin γ → Fr² + Fa² = Fs².
        let ft = 500.0;
        let alpha = 20.0_f64.to_radians();
        let gamma = 30.0_f64.to_radians();
        let fr = bevel_radial_force(ft, alpha, gamma);
        let fa = bevel_axial_force(ft, alpha, gamma);
        let fs = bevel_separating_force(ft, alpha);
        assert_relative_eq!(fr * fr + fa * fa, fs * fs, epsilon = 1e-9);
    }

    #[test]
    fn radial_axial_swap_when_cone_is_complementary() {
        // À γ = π/2 - γ', cos et sin s'échangent : Fr(γ) = Fa(π/2 - γ).
        let ft = 500.0;
        let alpha = 20.0_f64.to_radians();
        let gamma = 30.0_f64.to_radians();
        let comp = core::f64::consts::FRAC_PI_2 - gamma;
        assert_relative_eq!(
            bevel_radial_force(ft, alpha, gamma),
            bevel_axial_force(ft, alpha, comp),
            epsilon = 1e-9
        );
    }

    #[test]
    fn axial_vanishes_for_flat_cone() {
        // γ = 0 (roue plane dégénérée) → toute la composante séparatrice est radiale.
        let ft = 500.0;
        let alpha = 20.0_f64.to_radians();
        assert_relative_eq!(bevel_axial_force(ft, alpha, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            bevel_radial_force(ft, alpha, 0.0),
            bevel_separating_force(ft, alpha),
            epsilon = 1e-12
        );
    }

    #[test]
    fn resultant_matches_direct_computation() {
        // Cas chiffré : P = 3000 W, v = 3 m/s → Ft = 1000 N, α = 20°, γ = 25°.
        let ft = bevel_tangential_force(3000.0, 3.0);
        assert_relative_eq!(ft, 1000.0, epsilon = 1e-12);
        let alpha = 20.0_f64.to_radians();
        let gamma = 25.0_f64.to_radians();
        let fr = bevel_radial_force(ft, alpha, gamma);
        let fa = bevel_axial_force(ft, alpha, gamma);
        let r = bevel_resultant_force(ft, fr, fa);
        // Résultante = √(Ft² + Fs²) car Fr² + Fa² = Fs².
        let fs = bevel_separating_force(ft, alpha);
        assert_relative_eq!(r, (ft * ft + fs * fs).sqrt(), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "vitesse primitive doit être strictement positive")]
    fn zero_velocity_panics() {
        bevel_tangential_force(1000.0, 0.0);
    }
}
