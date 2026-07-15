//! Efforts sur une **roue et vis sans fin** (worm gear) — décomposition de la
//! charge normale au diamètre primitif en composantes tangentielle, axiale et
//! séparatrice, avec les identités croisées vis ↔ roue.
//!
//! ```text
//! effort tangentiel vis   Ft_vis = P / v_vis            (P puissance, v_vis vitesse primitive de la vis)
//! effort axial vis        Fa_vis = Ft_vis / tan(λ)      (= effort tangentiel de la roue)
//! effort séparateur       Fr     = Ft_vis·tan(αn) / sin(λ)
//! charge normale          Fn     = Ft_vis / (cos(αn)·sin(λ))
//! identité (orthogonale)  Fn² = Ft_vis² + Fa_vis² + Fr²
//! croisement vis ↔ roue   Ft_vis = Fa_roue ,  Fa_vis = Ft_roue ,  Fr_vis = Fr_roue
//! ```
//!
//! `P` puissance transmise (W), `v_vis` vitesse au cercle primitif de la vis
//! (m/s), `Ft_vis`/`Fa_vis`/`Fr`/`Fn` efforts tangentiel/axial/séparateur/normal
//! (N), `αn` angle de pression **normal** (rad), `λ` angle d'hélice (avance) de la
//! vis (rad).
//!
//! **Convention** : unités SI (W, m/s, N, rad), efforts appliqués au **diamètre
//! primitif**. **Limite honnête** : contact **idéal** (denture parfaite), le
//! **frottement est traité séparément** — voir [`crate::bevel_worm_gears`] pour le
//! rendement `η = tan λ/tan(λ+φ)` et l'auto-blocage. L'angle de pression normal
//! `αn`, l'angle d'hélice `λ`, la puissance et la vitesse sont **fournis par
//! l'appelant** ; aucune valeur « par défaut » de matériau, de procédé, de
//! géométrie ou de coût n'est inventée ici. Complète [`crate::gears`].

/// Effort **tangentiel** (moteur) sur la vis au diamètre primitif
/// `Ft_vis = P/v_vis` (N).
///
/// `power` en W, `worm_pitch_velocity` (vitesse primitive de la vis) en m/s.
///
/// Panique si `worm_pitch_velocity <= 0`.
pub fn worm_tangential_force(power: f64, worm_pitch_velocity: f64) -> f64 {
    assert!(
        worm_pitch_velocity > 0.0,
        "la vitesse primitive de la vis doit être strictement positive"
    );
    power / worm_pitch_velocity
}

/// Effort **axial** sur la vis `Fa_vis = Ft_vis/tan(λ)` (N).
///
/// Par réciprocité de la roue et vis, cet effort **égale l'effort tangentiel de
/// la roue** (`Fa_vis = Ft_roue`).
///
/// `worm_tangential` en N, `lead_angle_rad` (angle d'hélice `λ`) en rad.
///
/// Panique si `lead_angle_rad` n'est pas dans `]0, π/2[`.
pub fn worm_axial_force(worm_tangential: f64, lead_angle_rad: f64) -> f64 {
    assert_lead_angle(lead_angle_rad);
    worm_tangential / lead_angle_rad.tan()
}

/// Effort **séparateur** (radial) roue-vis `Fr = Ft_vis·tan(αn)/sin(λ)` (N).
///
/// Il est **commun à la vis et à la roue** (`Fr_vis = Fr_roue`).
///
/// `worm_tangential` en N, `normal_pressure_angle_rad` (`αn`) et `lead_angle_rad`
/// (`λ`) en rad.
///
/// Panique si `normal_pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `lead_angle_rad` n'est pas dans `]0, π/2[`.
pub fn worm_separating_force(
    worm_tangential: f64,
    normal_pressure_angle_rad: f64,
    lead_angle_rad: f64,
) -> f64 {
    assert_normal_pressure_angle(normal_pressure_angle_rad);
    assert_lead_angle(lead_angle_rad);
    worm_tangential * normal_pressure_angle_rad.tan() / lead_angle_rad.sin()
}

/// Charge **normale** totale à la denture `Fn = Ft_vis/(cos(αn)·sin(λ))` (N).
///
/// C'est le module de la résultante des trois composantes orthogonales :
/// `Fn² = Ft_vis² + Fa_vis² + Fr²`.
///
/// `worm_tangential` en N, `normal_pressure_angle_rad` (`αn`) et `lead_angle_rad`
/// (`λ`) en rad.
///
/// Panique si `normal_pressure_angle_rad` n'est pas dans `[0, π/2[` ou si
/// `lead_angle_rad` n'est pas dans `]0, π/2[`.
pub fn worm_normal_force(
    worm_tangential: f64,
    normal_pressure_angle_rad: f64,
    lead_angle_rad: f64,
) -> f64 {
    assert_normal_pressure_angle(normal_pressure_angle_rad);
    assert_lead_angle(lead_angle_rad);
    worm_tangential / (normal_pressure_angle_rad.cos() * lead_angle_rad.sin())
}

#[inline]
fn assert_normal_pressure_angle(normal_pressure_angle_rad: f64) {
    assert!(
        (0.0..core::f64::consts::FRAC_PI_2).contains(&normal_pressure_angle_rad),
        "l'angle de pression normal doit être dans [0, π/2["
    );
}

#[inline]
fn assert_lead_angle(lead_angle_rad: f64) {
    assert!(
        lead_angle_rad > 0.0 && lead_angle_rad < core::f64::consts::FRAC_PI_2,
        "l'angle d'hélice de la vis doit être dans ]0, π/2["
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangential_force_is_power_over_velocity() {
        // P = 3000 W à v_vis = 3 m/s → Ft_vis = 1000 N.
        assert_relative_eq!(worm_tangential_force(3000.0, 3.0), 1000.0, epsilon = 1e-12);
        // Proportionnalité en puissance : doubler P double Ft_vis.
        let f1 = worm_tangential_force(3000.0, 3.0);
        let f2 = worm_tangential_force(6000.0, 3.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-12);
    }

    #[test]
    fn components_satisfy_pythagorean_identity() {
        // Fn² = Ft_vis² + Fa_vis² + Fr² par construction trigonométrique.
        let ft = 1000.0;
        let alpha_n = 20.0_f64.to_radians();
        let lambda = 10.0_f64.to_radians();
        let fa = worm_axial_force(ft, lambda);
        let fr = worm_separating_force(ft, alpha_n, lambda);
        let fn_ = worm_normal_force(ft, alpha_n, lambda);
        assert_relative_eq!(ft * ft + fa * fa + fr * fr, fn_ * fn_, epsilon = 1e-6);
    }

    #[test]
    fn cross_relation_worm_axial_equals_gear_tangential() {
        // Réciprocité : l'effort axial de la vis = effort tangentiel de la roue.
        // Frottement idéal → Fa_vis = Ft_vis / tan(λ).
        let ft_worm = 1000.0;
        let lambda = 10.0_f64.to_radians();
        let ft_gear = worm_axial_force(ft_worm, lambda);
        assert_relative_eq!(ft_gear, ft_worm / lambda.tan(), epsilon = 1e-12);
        // Forte démultiplication d'effort quand λ est petit (Ft_gear ≫ Ft_vis).
        assert!(ft_gear > 5.0 * ft_worm);
    }

    #[test]
    fn separating_force_vanishes_at_zero_pressure_angle_and_is_proportional() {
        // αn = 0 : denture sans composante séparatrice → Fr = 0.
        let lambda = 12.0_f64.to_radians();
        assert_relative_eq!(
            worm_separating_force(1000.0, 0.0, lambda),
            0.0,
            epsilon = 1e-12
        );
        // Proportionnalité en effort tangentiel de la vis.
        let alpha_n = 20.0_f64.to_radians();
        let fr1 = worm_separating_force(1000.0, alpha_n, lambda);
        let fr2 = worm_separating_force(2000.0, alpha_n, lambda);
        assert_relative_eq!(fr2, 2.0 * fr1, epsilon = 1e-12);
    }

    #[test]
    fn realistic_worm_drive_case() {
        // Cas chiffré : P = 3 kW, v_vis = 3 m/s, αn = 20°, λ = 10°.
        let ft = worm_tangential_force(3000.0, 3.0);
        assert_relative_eq!(ft, 1000.0, epsilon = 1e-9);
        let alpha_n = 20.0_f64.to_radians();
        let lambda = 10.0_f64.to_radians();
        // Fa_vis = 1000/tan(10°) ≈ 5671 N (effort tangentiel de la roue).
        assert_relative_eq!(
            worm_axial_force(ft, lambda),
            1000.0 / lambda.tan(),
            epsilon = 1e-9
        );
        // Fr = 1000·tan(20°)/sin(10°) ≈ 2096 N.
        assert_relative_eq!(
            worm_separating_force(ft, alpha_n, lambda),
            1000.0 * alpha_n.tan() / lambda.sin(),
            epsilon = 1e-9
        );
        // Fn = 1000/(cos(20°)·sin(10°)) ≈ 6128 N.
        assert_relative_eq!(
            worm_normal_force(ft, alpha_n, lambda),
            1000.0 / (alpha_n.cos() * lambda.sin()),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "l'angle d'hélice de la vis doit être dans ]0, π/2[")]
    fn zero_lead_angle_panics() {
        worm_axial_force(1000.0, 0.0);
    }
}
