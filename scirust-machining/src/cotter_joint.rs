//! Assemblage par clavette transversale (**cotter joint**) — dimensionnement
//! statique en cisaillement, matage et traction de la tige.
//!
//! ```text
//! cisaillement (double) clavette   tau   = P / (2·b·t)
//! matage clavette / tige           sigma_c = P / (t·d)
//! traction de la tige              sigma_t = P / (π/4·d²)
//! ```
//!
//! `P` charge axiale statique transmise par l'assemblage (N), `b` largeur de la
//! clavette (m), `t` épaisseur de la clavette (m), `d` diamètre de la tige (m),
//! `tau` contrainte de cisaillement dans la clavette en double cisaillement (Pa),
//! `sigma_c` contrainte de matage (compression) à l'interface clavette/tige (Pa),
//! `sigma_t` contrainte de traction dans la section pleine de la tige (Pa).
//!
//! **Convention** : SI cohérent — charges en N, dimensions en m, contraintes en Pa.
//!
//! **Limite honnête** : modèle statique élémentaire supposant un **chargement
//! axial pur** et une **répartition uniforme** des contraintes sur chaque section
//! (deux plans de cisaillement égaux pour la clavette, pression de matage
//! constante sur `t·d`, traction constante sur la section pleine). La réalité
//! comporte des concentrations de contrainte (angles de la mortaise, congés), une
//! traction affaiblie par la section réduite au droit de la clavette, du frottement
//! et d'éventuelles charges de fatigue — non couverts ici. Les contraintes
//! **admissibles** du matériau et les coefficients de sécurité dépendent du
//! matériau, du procédé et des conditions d'emploi : ils sont **fournis par
//! l'appelant** — aucune valeur « par défaut » n'est inventée dans ce module.

use core::f64::consts::PI;

/// Contrainte de cisaillement dans la clavette en **double cisaillement**
/// `tau = P / (2·b·t)`.
///
/// `load` = `P` (N), `cotter_width` = `b` (m), `cotter_thickness` = `t` (m) ;
/// renvoie une contrainte (Pa).
///
/// Panique si `load < 0`, `cotter_width <= 0` ou `cotter_thickness <= 0`.
pub fn cotter_shear_stress(load: f64, cotter_width: f64, cotter_thickness: f64) -> f64 {
    assert!(
        load >= 0.0 && cotter_width > 0.0 && cotter_thickness > 0.0,
        "P ≥ 0, b > 0 et t > 0 requis"
    );
    load / (2.0 * cotter_width * cotter_thickness)
}

/// Contrainte de matage (compression) à l'interface clavette/tige
/// `sigma_c = P / (t·d)`.
///
/// `load` = `P` (N), `cotter_thickness` = `t` (m), `rod_diameter` = `d` (m) ;
/// renvoie une contrainte (Pa).
///
/// Panique si `load < 0`, `cotter_thickness <= 0` ou `rod_diameter <= 0`.
pub fn cotter_crushing_stress(load: f64, cotter_thickness: f64, rod_diameter: f64) -> f64 {
    assert!(
        load >= 0.0 && cotter_thickness > 0.0 && rod_diameter > 0.0,
        "P ≥ 0, t > 0 et d > 0 requis"
    );
    load / (cotter_thickness * rod_diameter)
}

/// Contrainte de traction dans la section pleine de la tige
/// `sigma_t = P / (π/4·d²)`.
///
/// `load` = `P` (N), `rod_diameter` = `d` (m) ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0` ou `rod_diameter <= 0`.
pub fn rod_tensile_stress(load: f64, rod_diameter: f64) -> f64 {
    assert!(load >= 0.0 && rod_diameter > 0.0, "P ≥ 0 et d > 0 requis");
    load / (PI / 4.0 * rod_diameter.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shear_stress_inverse_of_double_area() {
        // tau·(2·b·t) redonne la charge P : réciprocité contrainte ↔ charge.
        let (p, b, t) = (50e3_f64, 0.030, 0.012);
        let tau = cotter_shear_stress(p, b, t);
        assert_relative_eq!(tau * (2.0 * b * t), p, epsilon = 1e-6);
    }

    #[test]
    fn double_shear_is_half_of_single_area_stress() {
        // Le double cisaillement partage la charge sur deux plans : la contrainte
        // vaut la moitié de P/(b·t).
        let (p, b, t) = (40e3_f64, 0.025, 0.010);
        let tau = cotter_shear_stress(p, b, t);
        assert_relative_eq!(tau, p / (b * t) / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn crushing_stress_inversely_proportional_to_thickness() {
        // Doubler l'épaisseur de la clavette halve la contrainte de matage.
        let single = cotter_crushing_stress(30e3, 0.010, 0.040);
        let double = cotter_crushing_stress(30e3, 0.020, 0.040);
        assert_relative_eq!(double, single / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn tensile_stress_scales_as_inverse_square_of_diameter() {
        // sigma_t ∝ 1/d² : doubler le diamètre divise la contrainte par 4.
        let base = rod_tensile_stress(20e3, 0.020);
        let big = rod_tensile_stress(20e3, 0.040);
        assert_relative_eq!(big, base / 4.0, epsilon = 1e-6);
    }

    #[test]
    fn realistic_cotter_joint() {
        // Tige d = 40 mm, clavette b = 30 mm × t = 12 mm, charge P = 50 kN.
        // Traction : aire = π/4·(0,04)² = 1,2566e-3 m² → sigma_t ≈ 39,79 MPa.
        assert_relative_eq!(
            rod_tensile_stress(50e3, 0.040),
            50e3 / (PI / 4.0 * 0.040_f64.powi(2)),
            epsilon = 1.0
        );
        assert_relative_eq!(rod_tensile_stress(50e3, 0.040), 39.789e6, epsilon = 1e4);
        // Matage : sigma_c = 50e3 / (0,012·0,040) = 104,17 MPa.
        assert_relative_eq!(
            cotter_crushing_stress(50e3, 0.012, 0.040),
            104.1667e6,
            epsilon = 1e3
        );
        // Cisaillement double : tau = 50e3 / (2·0,030·0,012) = 69,44 MPa.
        assert_relative_eq!(
            cotter_shear_stress(50e3, 0.030, 0.012),
            69.444e6,
            epsilon = 1e3
        );
    }

    #[test]
    fn zero_load_gives_zero_stress() {
        // Cas limite : charge nulle → contraintes nulles sur toutes les sections.
        assert_relative_eq!(cotter_shear_stress(0.0, 0.030, 0.012), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            cotter_crushing_stress(0.0, 0.012, 0.040),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(rod_tensile_stress(0.0, 0.040), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "d > 0")]
    fn zero_rod_diameter_panics() {
        rod_tensile_stress(1000.0, 0.0);
    }
}
