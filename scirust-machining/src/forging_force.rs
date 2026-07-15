//! **Effort de forgeage** — refoulement à matrice ouverte, matriçage à matrice
//! fermée et contrainte d'écoulement par la loi de Hollomon.
//!
//! ```text
//! contrainte d'écoulement (Hollomon)  σf = K · ε^n
//! refoulement matrice ouverte          F  = σf · A · (1 + μ·D / (3·h))
//! matriçage matrice fermée             F  = Kf · σf · Ap
//! ```
//!
//! `σf` contrainte d'écoulement (Pa), `K` coefficient de résistance (Pa), `ε`
//! déformation vraie (sans dimension), `n` exposant d'écrouissage (sans dimension),
//! `A` aire de contact instantanée du cylindre refoulé (m²), `μ` facteur de
//! frottement (sans dimension), `D` diamètre de contact (m), `h` hauteur courante
//! (m), `Kf` facteur de forme de la pièce matricée (sans dimension), `Ap` aire
//! projetée de la pièce (m²), `F` effort de forgeage (N).
//!
//! **Convention** : SI strict (Pa, m, m², N). **Limite honnête** : déformation
//! plastique idéale ; la contrainte d'écoulement (loi de Hollomon, coefficient `K`
//! et exposant `n` **fournis par l'appelant**) ainsi que le facteur de frottement
//! `μ` et le facteur de forme `Kf` sont **fournis par l'appelant** — aucune valeur
//! « par défaut » matériau/procédé n'est inventée. La température et la vitesse de
//! déformation ne sont pas modélisées : la contrainte d'écoulement passée doit déjà
//! correspondre aux conditions réelles (à chaud ou à froid). Distinct de
//! [`crate::blanking_force`] (découpage) et de [`crate::broaching`] (usinage).

/// Contrainte d'écoulement selon la loi de Hollomon `σf = K · ε^n`.
///
/// `strength_coefficient` (K) et le résultat en Pa ; `true_strain` (ε) et
/// `strain_hardening_exponent` (n) sans dimension.
///
/// Panique si `strength_coefficient <= 0`, si `true_strain < 0` ou si
/// `strain_hardening_exponent < 0`.
pub fn forging_flow_stress(
    strength_coefficient: f64,
    true_strain: f64,
    strain_hardening_exponent: f64,
) -> f64 {
    assert!(
        strength_coefficient > 0.0,
        "coefficient de résistance K > 0 requis"
    );
    assert!(true_strain >= 0.0, "déformation vraie ε ≥ 0 requise");
    assert!(
        strain_hardening_exponent >= 0.0,
        "exposant d'écrouissage n ≥ 0 requis"
    );
    strength_coefficient * true_strain.powf(strain_hardening_exponent)
}

/// Effort de refoulement à matrice ouverte d'un cylindre
/// `F = σf · A · (1 + μ·D / (3·h))`, où le terme `μ·D/(3·h)` est le facteur de
/// forme dû au frottement radial.
///
/// `flow_stress` (σf) en Pa, `contact_area` (A) en m², `friction_factor` (μ) sans
/// dimension, `diameter` (D) et `height` (h) en m ; résultat en N.
///
/// Panique si `flow_stress <= 0`, si `contact_area <= 0`, si `friction_factor < 0`,
/// si `diameter <= 0` ou si `height <= 0`.
pub fn forging_open_die_force(
    flow_stress: f64,
    contact_area: f64,
    friction_factor: f64,
    diameter: f64,
    height: f64,
) -> f64 {
    assert!(flow_stress > 0.0, "contrainte d'écoulement σf > 0 requise");
    assert!(contact_area > 0.0, "aire de contact A > 0 requise");
    assert!(friction_factor >= 0.0, "facteur de frottement μ ≥ 0 requis");
    assert!(diameter > 0.0, "diamètre D > 0 requis");
    assert!(height > 0.0, "hauteur h > 0 requise");
    flow_stress * contact_area * (1.0 + friction_factor * diameter / (3.0 * height))
}

/// Effort de matriçage à matrice fermée `F = Kf · σf · Ap`, avec facteur de forme
/// `Kf` **fourni par l'appelant** (typiquement grand pour les pièces à nervures ou
/// à toiles minces).
///
/// `flow_stress` (σf) en Pa, `projected_area` (Ap) en m², `shape_factor` (Kf) sans
/// dimension ; résultat en N.
///
/// Panique si `flow_stress <= 0`, si `projected_area <= 0` ou si `shape_factor <= 0`.
pub fn forging_closed_die_force(flow_stress: f64, projected_area: f64, shape_factor: f64) -> f64 {
    assert!(flow_stress > 0.0, "contrainte d'écoulement σf > 0 requise");
    assert!(projected_area > 0.0, "aire projetée Ap > 0 requise");
    assert!(shape_factor > 0.0, "facteur de forme Kf > 0 requis");
    shape_factor * flow_stress * projected_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn flow_stress_hollomon_worked_case() {
        // K = 600 MPa, ε = 0,5, n = 0,2 : σf = 600e6 · 0,5^0,2.
        // 0,5^0,2 = exp(0,2·ln 0,5) = exp(-0,138629436) = 0,870550563.
        // σf = 600e6 · 0,870550563 = 5,22330338e8 Pa.
        let sigma = forging_flow_stress(600.0e6, 0.5, 0.2);
        assert_relative_eq!(sigma, 5.223303379_f64 * 1.0e8, epsilon = 1.0e2);
    }

    #[test]
    fn flow_stress_at_unit_strain_equals_k() {
        // ε = 1 ⇒ ε^n = 1 quel que soit n : σf = K.
        assert_relative_eq!(
            forging_flow_stress(450.0e6, 1.0, 0.15),
            450.0e6,
            epsilon = 1e-3
        );
    }

    #[test]
    fn open_die_no_friction_reduces_to_sigma_area() {
        // μ = 0 ⇒ facteur de forme = 1 ⇒ F = σf · A.
        let sigma = 200.0e6;
        let area = 0.01;
        assert_relative_eq!(
            forging_open_die_force(sigma, area, 0.0, 0.1, 0.05),
            sigma * area,
            epsilon = 1e-6
        );
    }

    #[test]
    fn open_die_worked_case() {
        // σf = 200 MPa, A = 0,01 m², μ = 0,3, D = 0,1 m, h = 0,05 m.
        // facteur = 1 + 0,3·0,1/(3·0,05) = 1 + 0,03/0,15 = 1,2.
        // F = 200e6 · 0,01 · 1,2 = 2,4e6 N.
        assert_relative_eq!(
            forging_open_die_force(200.0e6, 0.01, 0.3, 0.1, 0.05),
            2.4e6,
            epsilon = 1e-3
        );
    }

    #[test]
    fn closed_die_linear_in_shape_factor() {
        // F = Kf · σf · Ap : doubler Kf double l'effort ; cas chiffré Kf = 8.
        let f1 = forging_closed_die_force(200.0e6, 0.005, 8.0);
        let f2 = forging_closed_die_force(200.0e6, 0.005, 16.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-3);
        // 8 · 200e6 · 0,005 = 8e6 N.
        assert_relative_eq!(f1, 8.0e6, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "facteur de forme Kf > 0 requis")]
    fn closed_die_rejects_zero_shape_factor() {
        forging_closed_die_force(200.0e6, 0.005, 0.0);
    }
}
