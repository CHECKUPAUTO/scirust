//! Module de section en flexion (`Z = I/c`) : modules de résistance de sections
//! usuelles et contrainte de flexion associée.
//!
//! ```text
//! rectangle plein         Z = b·h² / 6
//! cercle plein            Z = π·d³ / 32
//! couronne circulaire     Z = π·(D⁴ − d⁴) / (32·D)
//! contrainte de flexion   σ = M / Z
//! ```
//!
//! `Z` module de résistance en flexion (m³), `b` largeur de la section (m), `h`
//! hauteur de la section dans le plan de flexion (m), `d` diamètre (cercle plein)
//! ou diamètre intérieur (couronne) (m), `D` diamètre extérieur (couronne) (m),
//! `M` moment fléchissant (N·m), `σ` contrainte de flexion à la fibre extrême
//! (Pa), `I` moment quadratique de la section (m⁴), `c` distance de l'axe neutre
//! à la fibre extrême (m).
//!
//! **Convention** : SI cohérent, `Z = I/c` avec `c` la distance à la fibre
//! extrême. **Limite honnête** : flexion pure ÉLASTIQUE, sections symétriques
//! par rapport à l'axe neutre, matériau homogène ; on néglige l'effort tranchant,
//! le voilement et toute plastification. Les constantes matériaux (contrainte
//! admissible, coefficient de sécurité) et les charges (moment fléchissant) sont
//! FOURNIES par l'appelant et ne sont jamais supposées ici.

use core::f64::consts::PI;

/// Module de résistance en flexion d'une section rectangulaire pleine
/// `Z = b·h² / 6` (m³), flexion autour de l'axe parallèle à la largeur `b`.
///
/// `width` largeur `b` en m, `height` hauteur `h` (dans le plan de flexion) en m.
///
/// Panique si `width <= 0` ou `height <= 0`.
pub fn section_modulus_rectangle(width: f64, height: f64) -> f64 {
    assert!(width > 0.0, "la largeur doit être strictement positive");
    assert!(height > 0.0, "la hauteur doit être strictement positive");
    width * height * height / 6.0
}

/// Module de résistance en flexion d'une section circulaire pleine
/// `Z = π·d³ / 32` (m³).
///
/// `diameter` diamètre `d` en m.
///
/// Panique si `diameter <= 0`.
pub fn section_modulus_solid_circle(diameter: f64) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    PI * diameter * diameter * diameter / 32.0
}

/// Module de résistance en flexion d'une couronne circulaire (tube)
/// `Z = π·(D⁴ − d⁴) / (32·D)` (m³), rapporté à la fibre extrême `c = D/2`.
///
/// `outer_diameter` diamètre extérieur `D` en m, `inner_diameter` diamètre
/// intérieur `d` en m.
///
/// Panique si `outer_diameter <= 0`, `inner_diameter < 0` ou
/// `inner_diameter >= outer_diameter`.
pub fn section_modulus_hollow_circle(outer_diameter: f64, inner_diameter: f64) -> f64 {
    assert!(
        outer_diameter > 0.0,
        "le diamètre extérieur doit être strictement positif"
    );
    assert!(
        inner_diameter >= 0.0,
        "le diamètre intérieur doit être positif"
    );
    assert!(
        inner_diameter < outer_diameter,
        "le diamètre intérieur doit être inférieur au diamètre extérieur"
    );
    let d_out4 = outer_diameter.powi(4);
    let d_in4 = inner_diameter.powi(4);
    PI * (d_out4 - d_in4) / (32.0 * outer_diameter)
}

/// Contrainte de flexion à la fibre extrême `σ = M / Z` (Pa).
///
/// `bending_moment` moment fléchissant `M` en N·m, `section_modulus` module de
/// résistance `Z` en m³.
///
/// Panique si `section_modulus <= 0`.
pub fn section_bending_stress(bending_moment: f64, section_modulus: f64) -> f64 {
    assert!(
        section_modulus > 0.0,
        "le module de résistance doit être strictement positif"
    );
    bending_moment / section_modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hollow_circle_reduces_to_solid_when_inner_is_zero() {
        // Couronne d'intérieur nul = cercle plein : Z = π·D³/32.
        let d = 0.05_f64;
        assert_relative_eq!(
            section_modulus_hollow_circle(d, 0.0),
            section_modulus_solid_circle(d),
            epsilon = 1e-18
        );
    }

    #[test]
    fn rectangle_scales_with_height_squared() {
        // Z ∝ h² à largeur constante : doubler h quadruple Z.
        let b = 0.03_f64;
        let z1 = section_modulus_rectangle(b, 0.1);
        let z2 = section_modulus_rectangle(b, 0.2);
        assert_relative_eq!(z2 / z1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn solid_circle_scales_with_diameter_cubed() {
        // Z ∝ d³ : doubler le diamètre multiplie Z par 8.
        let z1 = section_modulus_solid_circle(0.02);
        let z2 = section_modulus_solid_circle(0.04);
        assert_relative_eq!(z2 / z1, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn bending_stress_is_inverse_of_modulus() {
        // σ·Z = M : la contrainte multipliée par le module redonne le moment.
        let (m, z) = (1500.0_f64, 8.0e-5_f64);
        let sigma = section_bending_stress(m, z);
        assert_relative_eq!(sigma * z, m, epsilon = 1e-9);
    }

    #[test]
    fn realistic_rectangular_beam_case() {
        // Section 50×100 mm, M = 1000 N·m :
        // Z = 0,05 · 0,1² / 6 = 8,333…e-5 m³
        // σ = 1000 / (0,0005/6) = 1000·6/0,0005 = 12 MPa.
        let z = section_modulus_rectangle(0.05, 0.1);
        assert_relative_eq!(z, 0.0005_f64 / 6.0, epsilon = 1e-15);
        let sigma = section_bending_stress(1000.0, z);
        assert_relative_eq!(sigma, 12.0e6, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "le diamètre intérieur doit être inférieur au diamètre extérieur")]
    fn hollow_circle_rejects_inner_ge_outer() {
        section_modulus_hollow_circle(0.05, 0.05);
    }
}
