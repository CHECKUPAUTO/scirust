//! Ressort **à force constante** (ruban d'acier à ressort enroulé, *constant-force
//! spring*) — effort de déroulement quasi constant, contrainte de flexion du ruban
//! et rayon naturel d'enroulement.
//!
//! ```text
//! effort         F = E·b·t³ / (C_f · R_n²)             (C_f = 26.4)
//! contrainte     σ = E·t / (2·R_n)
//! rayon naturel  R_n = √( E·b·t³ / (C_f · F) )         (réciproque de F)
//! ```
//!
//! `E` module de Young (Pa), `b` largeur du ruban (m), `t` épaisseur du ruban (m),
//! `R_n` rayon naturel d'enroulement du ruban (m), `C_f = 26.4` constante empirique
//! usuelle (sans dimension), `F` effort de déroulement quasi constant (N), `σ`
//! contrainte de flexion dans le ruban (Pa).
//!
//! **Convention** : SI cohérent (m, N, Pa). **Limite honnête** : ruban d'acier à
//! ressort enroulé, effort **quasi constant** une fois passé le premier tour de
//! déroulement (régime établi), formule empirique classique avec `C_f = 26.4`. Le
//! module de Young, les dimensions et le rayon naturel sont **fournis par
//! l'appelant** : ce module ne compose que ces primitives et n'invente aucune
//! constante de matériau ni de procédé. La **fatigue** du ruban (endurance en
//! cyclage) n'est **pas** traitée ici : voir la crate `scirust-fatigue`.

/// Constante empirique `C_f` de l'effort d'un ressort à force constante
/// (sans dimension).
pub const CONSTANT_FORCE_SPRING_FORCE_CONSTANT: f64 = 26.4;

/// Effort de déroulement quasi constant d'un ressort à force constante :
/// `F = E·b·t³ / (C_f · R_n²)` (N).
///
/// Panique si `youngs_modulus < 0`, si `width < 0`, si `thickness < 0`
/// ou si `natural_radius <= 0`.
pub fn constant_force_spring_force(
    youngs_modulus: f64,
    width: f64,
    thickness: f64,
    natural_radius: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(width >= 0.0, "width doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(natural_radius > 0.0, "natural_radius doit être > 0");
    youngs_modulus * width * thickness.powi(3)
        / (CONSTANT_FORCE_SPRING_FORCE_CONSTANT * natural_radius.powi(2))
}

/// Contrainte de flexion dans le ruban d'un ressort à force constante :
/// `σ = E·t / (2·R_n)` (Pa).
///
/// Panique si `youngs_modulus < 0`, si `thickness < 0`
/// ou si `natural_radius <= 0`.
pub fn constant_force_spring_stress(
    youngs_modulus: f64,
    thickness: f64,
    natural_radius: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(natural_radius > 0.0, "natural_radius doit être > 0");
    youngs_modulus * thickness / (2.0 * natural_radius)
}

/// Rayon naturel d'enroulement d'un ruban donnant un effort visé :
/// `R_n = √( E·b·t³ / (C_f · F) )` (m).
///
/// Réciproque de [`constant_force_spring_force`] par rapport à `natural_radius`.
///
/// Panique si `youngs_modulus < 0`, si `width < 0`, si `thickness < 0`
/// ou si `force <= 0`.
pub fn constant_force_spring_natural_radius(
    youngs_modulus: f64,
    width: f64,
    thickness: f64,
    force: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(width >= 0.0, "width doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(force > 0.0, "force doit être > 0");
    (youngs_modulus * width * thickness.powi(3) / (CONSTANT_FORCE_SPRING_FORCE_CONSTANT * force))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn force_matches_hand_computed_case() {
        // Cas chiffré réaliste (acier à ressort) calculé à la main :
        // E = 200e9, b = 0.02, t = 3e-4, R_n = 0.01.
        // num = 200e9·0.02·(3e-4)³ = 4e9·2.7e-11 = 0.108
        // den = 26.4·(0.01)² = 26.4·1e-4 = 2.64e-3
        // F = 0.108 / 2.64e-3 = 40.909090… N
        let f = constant_force_spring_force(200.0e9, 0.02, 3.0e-4, 0.01);
        assert_relative_eq!(f, 0.108 / 2.64e-3, max_relative = 1e-12);
    }

    #[test]
    fn force_and_natural_radius_are_reciprocal() {
        // R_n → F → R_n doit redonner le rayon de départ.
        let (e, b, t) = (210.0e9, 0.015, 2.5e-4_f64);
        let radius = 0.008_f64;
        let force = constant_force_spring_force(e, b, t, radius);
        assert_relative_eq!(
            constant_force_spring_natural_radius(e, b, t, force),
            radius,
            max_relative = 1e-12
        );
    }

    #[test]
    fn force_scales_with_thickness_cubed() {
        // F ∝ t³ : doubler l'épaisseur multiplie l'effort par 8.
        let base = constant_force_spring_force(200.0e9, 0.02, 3.0e-4, 0.01);
        let doubled = constant_force_spring_force(200.0e9, 0.02, 6.0e-4, 0.01);
        assert_relative_eq!(doubled, 8.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn force_is_inversely_proportional_to_radius_squared() {
        // F ∝ 1/R_n² : doubler le rayon divise l'effort par 4.
        let base = constant_force_spring_force(200.0e9, 0.02, 3.0e-4, 0.01);
        let bigger = constant_force_spring_force(200.0e9, 0.02, 3.0e-4, 0.02);
        assert_relative_eq!(bigger, base / 4.0, max_relative = 1e-12);
    }

    #[test]
    fn stress_matches_hand_computed_case_and_is_inverse_in_radius() {
        // σ = E·t / (2·R_n) avec E = 200e9, t = 3e-4, R_n = 0.02.
        // σ = 6e7 / 0.04 = 1.5e9 Pa
        let sigma = constant_force_spring_stress(200.0e9, 3.0e-4, 0.02);
        assert_relative_eq!(sigma, 1.5e9, max_relative = 1e-12);
        // σ ∝ 1/R_n : moitié du rayon double la contrainte.
        let sigma_half = constant_force_spring_stress(200.0e9, 3.0e-4, 0.01);
        assert_relative_eq!(sigma_half, 2.0 * sigma, max_relative = 1e-12);
    }

    #[test]
    fn stress_is_linear_in_thickness() {
        // σ ∝ t à rayon fixé : doubler l'épaisseur double la contrainte.
        let s1 = constant_force_spring_stress(200.0e9, 3.0e-4, 0.01);
        let s2 = constant_force_spring_stress(200.0e9, 6.0e-4, 0.01);
        assert_relative_eq!(s2, 2.0 * s1, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "natural_radius doit être > 0")]
    fn zero_natural_radius_panics() {
        constant_force_spring_force(200.0e9, 0.02, 3.0e-4, 0.0);
    }
}
