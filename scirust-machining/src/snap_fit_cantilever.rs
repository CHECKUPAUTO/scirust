//! Encliquetage (**snap-fit**) à poutre **cantilever** — déformation maximale,
//! effort de déflexion du crochet et effort d'emmanchement.
//!
//! ```text
//! déformation max   ε   = 1.5 · y · t / L²
//! effort déflexion  F   = E · w · t³ · y / (4 · L³)
//! effort montage    Fm  = F · (μ + tan α) / (1 − μ · tan α)
//! ```
//!
//! `y` flèche imposée à l'extrémité du crochet (m), `t` épaisseur de la poutre à
//! l'encastrement (m), `L` longueur libre de la poutre (m), `E` module de Young
//! du matériau (Pa), `w` largeur de la poutre (m), `ε` déformation en fibre
//! extrême (sans dimension), `F` effort perpendiculaire de déflexion (N), `μ`
//! coefficient de frottement crochet/contrepièce (sans dimension), `α` angle de
//! la face d'insertion (rad), `Fm` effort axial d'emmanchement (N).
//!
//! **Convention** : SI cohérent (longueurs en m, `E` en Pa, efforts en N, angle
//! en radians). **Limite honnête** : poutre **encastrée-libre** de section
//! rectangulaire constante, théorie d'Euler-Bernoulli en **petites déformations**
//! (crochet droit, pas d'effilement) ; le module de Young `E` et la déformation
//! admissible `ε` du matériau sont **fournis par l'appelant** — aucune valeur
//! matériau n'est inventée ici. Le dénominateur `1 − μ·tan α` doit rester
//! strictement positif (condition de non auto-blocage à l'emmanchement).

/// Déformation maximale en fibre extrême `ε = 1.5·y·t/L²` (sans dimension).
///
/// Panique si `length <= 0`, `thickness < 0` ou `deflection < 0`.
pub fn snap_max_strain(deflection: f64, thickness: f64, length: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur libre L doit être strictement positive"
    );
    assert!(thickness >= 0.0, "l'épaisseur t ne peut être négative");
    assert!(deflection >= 0.0, "la flèche y ne peut être négative");
    1.5 * deflection * thickness / (length * length)
}

/// Effort perpendiculaire de déflexion du crochet
/// `F = E·w·t³·y/(4·L³)` (N).
///
/// Panique si `length <= 0` ou si un paramètre géométrique/matériau est négatif.
pub fn snap_deflection_force(
    youngs_modulus: f64,
    width: f64,
    thickness: f64,
    length: f64,
    deflection: f64,
) -> f64 {
    assert!(
        length > 0.0,
        "la longueur libre L doit être strictement positive"
    );
    assert!(
        youngs_modulus >= 0.0,
        "le module de Young E ne peut être négatif"
    );
    assert!(width >= 0.0, "la largeur w ne peut être négative");
    assert!(thickness >= 0.0, "l'épaisseur t ne peut être négative");
    assert!(deflection >= 0.0, "la flèche y ne peut être négative");
    youngs_modulus * width * thickness.powi(3) * deflection / (4.0 * length.powi(3))
}

/// Effort axial d'emmanchement `Fm = F·(μ + tan α)/(1 − μ·tan α)` (N).
///
/// Panique si `friction_coefficient < 0`, si `deflection_force < 0`, ou si le
/// dénominateur `1 − μ·tan α <= 0` (auto-blocage à l'insertion).
pub fn snap_mating_force(
    deflection_force: f64,
    friction_coefficient: f64,
    insertion_angle_rad: f64,
) -> f64 {
    assert!(
        deflection_force >= 0.0,
        "l'effort de déflexion F ne peut être négatif"
    );
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement μ ne peut être négatif"
    );
    let tan_a = insertion_angle_rad.tan();
    let den = 1.0 - friction_coefficient * tan_a;
    assert!(
        den > 0.0,
        "1 − μ·tan α doit être strictement positif (pas d'auto-blocage)"
    );
    deflection_force * (friction_coefficient + tan_a) / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn strain_scales_linearly_with_deflection() {
        // ε est linéaire en la flèche y à géométrie fixée.
        let e1 = snap_max_strain(0.001, 0.002, 0.020);
        let e2 = snap_max_strain(0.002, 0.002, 0.020);
        assert_relative_eq!(e2, 2.0 * e1, max_relative = 1e-12);
    }

    #[test]
    fn strain_realistic_value() {
        // Crochet PP : y=2 mm, t=2 mm, L=20 mm → ε = 1.5·0.002·0.002/0.020² = 1.5 %.
        let eps = snap_max_strain(0.002, 0.002, 0.020);
        assert_relative_eq!(eps, 0.015, max_relative = 1e-12);
    }

    #[test]
    fn strain_halves_when_length_grows_root_two() {
        // ε ∝ 1/L² : allonger L d'un facteur √2 divise ε par 2.
        let short = snap_max_strain(0.001, 0.002, 0.020);
        let long = snap_max_strain(0.001, 0.002, 0.020 * (2.0_f64).sqrt());
        assert_relative_eq!(long, short / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn deflection_force_matches_beam_strain_relation() {
        // Identité de poutre : F = E·w·t²·ε/(6·L), avec ε = snap_max_strain(y,t,L).
        let (e, w, t, l, y) = (1.3e9, 0.010, 0.002, 0.020, 0.002);
        let f = snap_deflection_force(e, w, t, l, y);
        let eps = snap_max_strain(y, t, l);
        let f_from_strain = e * w * t * t * eps / (6.0 * l);
        assert_relative_eq!(f, f_from_strain, max_relative = 1e-12);
        assert!(f > 0.0);
    }

    #[test]
    fn mating_force_zero_angle_is_pure_friction() {
        // À α = 0 : Fm = F·μ (la face frontale ne fournit aucune composante de coin).
        let f = 50.0;
        let fm = snap_mating_force(f, 0.3, 0.0);
        assert_relative_eq!(fm, f * 0.3, max_relative = 1e-12);
    }

    #[test]
    fn mating_force_frictionless_is_tan_alpha() {
        // Sans frottement (μ=0) : Fm = F·tan α ; à α = π/4, Fm = F.
        let f = 50.0;
        let fm = snap_mating_force(f, 0.0, PI / 4.0);
        assert_relative_eq!(fm, f, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pas d'auto-blocage")]
    fn self_locking_mating_panics() {
        // μ·tan α > 1 (μ=2, α=π/4) → dénominateur négatif : auto-blocage refusé.
        snap_mating_force(50.0, 2.0, PI / 4.0);
    }
}
