//! Moment moteur sur **vilebrequin** (mécanisme bielle-manivelle) : conversion
//! de l'effort de piston en couple à partir de l'angle de manivelle.
//!
//! ```text
//! rapport d'obliquité   λ = r/L
//! moment sur vilebrequin  M = F·r·(sinθ + (λ/2)·sin2θ)   (obliquité au 1er ordre)
//! ```
//!
//! `F` effort de piston (N, projeté sur l'axe du cylindre), `r` rayon de manivelle
//! (m), `L` longueur de bielle (m), `θ` angle de manivelle mesuré depuis le point
//! mort haut (rad), `λ` rapport d'obliquité (sans dimension). Le moment `M` est en
//! N·m. Unités SI cohérentes.
//!
//! **Limite honnête** : cinématique bielle-manivelle avec obliquité **approchée**
//! (développement au 1er ordre en `λ`, valable pour `L ≳ 3r`). L'effort de piston
//! `F` est **fourni** par l'appelant (pression des gaz, inertie, etc.) : aucune
//! constante physique, matériau ou procédé n'est inventée ici. Les forces
//! d'inertie des masses en mouvement et le frottement sont **négligés** ; à
//! l'appelant de les ajouter à l'effort `F` ou au moment résultant.

/// Rapport d'obliquité `λ = r/L` (sans dimension).
///
/// Panique si `connecting_rod_length <= 0` ou `crank_radius < 0`.
pub fn crank_obliquity_ratio(crank_radius: f64, connecting_rod_length: f64) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    assert!(
        connecting_rod_length > 0.0,
        "la longueur de bielle doit être strictement positive"
    );
    crank_radius / connecting_rod_length
}

/// Moment sur le vilebrequin `M = F·r·(sinθ + (λ/2)·sin2θ)` produit par
/// l'effort de piston `F` (approximation de l'obliquité au 1er ordre).
///
/// Panique si `crank_radius < 0` ou si `obliquity_ratio` n'est pas dans `[0, 1[`.
pub fn crank_piston_force_to_torque(
    piston_force: f64,
    crank_radius: f64,
    crank_angle_rad: f64,
    obliquity_ratio: f64,
) -> f64 {
    assert!(
        crank_radius >= 0.0,
        "le rayon de manivelle doit être positif ou nul"
    );
    assert!(
        (0.0..1.0).contains(&obliquity_ratio),
        "le rapport d'obliquité doit être dans [0, 1["
    );
    piston_force
        * crank_radius
        * (crank_angle_rad.sin() + 0.5 * obliquity_ratio * (2.0 * crank_angle_rad).sin())
}

/// Moment de rotation dû à l'effort des gaz `M = F_gaz·r·(sinθ + (λ/2)·sin2θ)`
/// (même cinématique que [`crank_piston_force_to_torque`], effort fourni).
///
/// Panique si `crank_radius < 0` ou si `obliquity_ratio` n'est pas dans `[0, 1[`.
pub fn crank_turning_moment_gas(
    gas_force: f64,
    crank_radius: f64,
    crank_angle_rad: f64,
    obliquity_ratio: f64,
) -> f64 {
    crank_piston_force_to_torque(gas_force, crank_radius, crank_angle_rad, obliquity_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn obliquity_ratio_is_r_over_l() {
        // λ = r/L : 0.05/0.20 = 0.25.
        assert_relative_eq!(crank_obliquity_ratio(0.05, 0.20), 0.25, epsilon = 1e-12);
    }

    #[test]
    fn torque_vanishes_at_dead_centers() {
        // θ=0 et θ=π : sinθ = 0 et sin2θ = 0 → M = 0 aux deux points morts.
        assert_relative_eq!(
            crank_piston_force_to_torque(10_000.0, 0.05, 0.0, 0.25),
            0.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            crank_piston_force_to_torque(10_000.0, 0.05, PI, 0.25),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn torque_equals_force_times_radius_at_quarter_turn() {
        // À θ=π/2 : sin2θ = sinπ = 0, le terme d'obliquité s'annule quelle que
        // soit λ → M = F·r exactement.
        let m = crank_piston_force_to_torque(8_000.0, 0.04, FRAC_PI_2, 0.30);
        assert_relative_eq!(m, 8_000.0 * 0.04, epsilon = 1e-9);
    }

    #[test]
    fn torque_is_linear_in_force() {
        // M ∝ F : doubler l'effort double le moment (mêmes r, θ, λ).
        let m1 = crank_piston_force_to_torque(5_000.0, 0.06, 1.1, 0.20);
        let m2 = crank_piston_force_to_torque(10_000.0, 0.06, 1.1, 0.20);
        assert_relative_eq!(m2, 2.0 * m1, epsilon = 1e-9);
    }

    #[test]
    fn realistic_torque_at_thirty_degrees() {
        // F=8000 N, r=0.04 m, λ=0.25, θ=π/6.
        // sinθ = 1/2, sin2θ = sin(π/3) = √3/2.
        // M = 8000·0.04·(0.5 + 0.125·√3/2) = 320·0.608253… = 194.6410… N·m.
        let expected = 320.0 * (0.5 + 0.125 * (3.0_f64.sqrt() / 2.0));
        assert_relative_eq!(
            crank_piston_force_to_torque(8_000.0, 0.04, PI / 6.0, 0.25),
            expected,
            epsilon = 1e-9
        );
        // Valeur numérique de contrôle.
        assert_relative_eq!(expected, 194.641_016_151_377_5, epsilon = 1e-9);
    }

    #[test]
    fn gas_moment_matches_piston_force_formula() {
        // crank_turning_moment_gas est la même cinématique : mêmes entrées → même
        // résultat que crank_piston_force_to_torque.
        let f = crank_piston_force_to_torque(12_000.0, 0.045, 0.7, 0.28);
        let g = crank_turning_moment_gas(12_000.0, 0.045, 0.7, 0.28);
        assert_relative_eq!(f, g, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "rapport d'obliquité")]
    fn obliquity_out_of_range_panics() {
        // λ = 1.5 hors de [0, 1[ : bielle plus courte que la manivelle.
        crank_piston_force_to_torque(10_000.0, 0.05, FRAC_PI_2, 1.5);
    }
}
