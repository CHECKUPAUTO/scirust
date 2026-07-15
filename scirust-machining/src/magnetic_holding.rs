//! Force de maintien magnétique par **traction de Maxwell** (attraction d'un
//! entrefer entre un aimant/électro-aimant et une pièce ferromagnétique).
//!
//! ```text
//! traction d'un pôle   F   = B²·A/(2·µ0)        (N)
//! maintien deux pôles  F2  = 2·B²·A/(2·µ0)      = B²·A/µ0   (N)
//! induction requise    B   = sqrt(2·µ0·F/A)     (T, réciproque)
//! ```
//!
//! `B` induction magnétique dans l'entrefer (T = Wb·m⁻²), `A` aire d'un pôle en
//! contact (m²), `µ0` perméabilité du vide (H·m⁻¹ = T·m·A⁻¹), `F` force
//! d'attraction d'un pôle (N), `F2` force de maintien d'un aimant à **deux**
//! pôles refermant le circuit sur la pièce (N).
//!
//! **Convention** : unités SI strictes.
//! **Limite honnête** : modèle d'entrefer **nul idéal**, matériau **non saturé**,
//! flux **uniforme** dans l'entrefer ; l'induction `B` est **fournie par
//! l'appelant** — aucune valeur « par défaut » n'est inventée. Les fuites de flux
//! et la réluctance du circuit fer sont **négligées** ; la force réelle d'un
//! dispositif est donc inférieure à ces bornes idéales.

/// Perméabilité magnétique du vide `µ0` (H·m⁻¹), valeur physique de référence.
pub const MU0_VACUUM: f64 = 1.256_637_061_4e-6;

/// Traction de Maxwell d'un **seul** pôle `F = B²·A/(2·µ0)` (N).
///
/// Panique si `flux_density < 0`, `pole_area <= 0` ou `permeability_vacuum <= 0`.
pub fn magnetic_maxwell_pull(flux_density: f64, pole_area: f64, permeability_vacuum: f64) -> f64 {
    assert!(flux_density >= 0.0, "B ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    flux_density * flux_density * pole_area / (2.0 * permeability_vacuum)
}

/// Force de maintien d'un aimant à **deux pôles** `F2 = B²·A/µ0` (N),
/// soit le double de la traction d'un pôle unique.
///
/// Panique si `flux_density < 0`, `pole_area <= 0` ou `permeability_vacuum <= 0`.
pub fn magnetic_holding_force_two_poles(
    flux_density: f64,
    pole_area: f64,
    permeability_vacuum: f64,
) -> f64 {
    assert!(flux_density >= 0.0, "B ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    flux_density * flux_density * pole_area / permeability_vacuum
}

/// Induction requise pour une traction d'un pôle donnée
/// `B = sqrt(2·µ0·F/A)` (T) — réciproque de [`magnetic_maxwell_pull`].
///
/// Panique si `force < 0`, `pole_area <= 0` ou `permeability_vacuum <= 0`.
pub fn magnetic_flux_density_for_force(
    force: f64,
    pole_area: f64,
    permeability_vacuum: f64,
) -> f64 {
    assert!(force >= 0.0, "F ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    (2.0_f64 * permeability_vacuum * force / pole_area).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn maxwell_pull_realistic_case() {
        // Cas chiffré : B = 1 T, A = 0.01 m² (100 cm²).
        // F = 1²·0.01/(2·µ0) = 0.01/2.5132741228e-6 = 3978.8735774 N.
        let f = magnetic_maxwell_pull(1.0, 0.01, MU0_VACUUM);
        assert_relative_eq!(f, 3978.8735774, max_relative = 1e-9);
    }

    #[test]
    fn two_poles_is_double_single_pole() {
        // F2 = 2·F : identité entre les deux modèles.
        let f1 = magnetic_maxwell_pull(1.2, 0.008, MU0_VACUUM);
        let f2 = magnetic_holding_force_two_poles(1.2, 0.008, MU0_VACUUM);
        assert_relative_eq!(f2, 2.0 * f1, max_relative = 1e-12);
    }

    #[test]
    fn flux_density_is_reciprocal_of_pull() {
        // sqrt(2·µ0·F/A) rend B : réciprocité exacte.
        let b = 0.9;
        let f = magnetic_maxwell_pull(b, 0.02, MU0_VACUUM);
        let b_back = magnetic_flux_density_for_force(f, 0.02, MU0_VACUUM);
        assert_relative_eq!(b_back, b, max_relative = 1e-12);
    }

    #[test]
    fn pull_proportional_to_area() {
        // F ∝ A : doubler l'aire du pôle double la force.
        let f1 = magnetic_maxwell_pull(0.8, 0.005, MU0_VACUUM);
        let f2 = magnetic_maxwell_pull(0.8, 0.010, MU0_VACUUM);
        assert_relative_eq!(f2 / f1, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn pull_quadratic_in_flux_density() {
        // F ∝ B² : doubler B quadruple la force.
        let f1 = magnetic_maxwell_pull(0.5, 0.01, MU0_VACUUM);
        let f2 = magnetic_maxwell_pull(1.0, 0.01, MU0_VACUUM);
        assert_relative_eq!(f2 / f1, 4.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_area_panics() {
        magnetic_maxwell_pull(1.0, 0.0, MU0_VACUUM);
    }
}
