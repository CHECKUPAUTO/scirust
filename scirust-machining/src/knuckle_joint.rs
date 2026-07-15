//! Assemblage à chape et œil (**knuckle joint**) — dimensionnement statique en
//! cisaillement de l'axe, matage de l'œil et de la chape, et traction de la tige.
//!
//! ```text
//! cisaillement (double) de l'axe   tau     = P / (2·(π/4)·d²)
//! matage œil (single eye)          sigma_e = P / (d·t)
//! matage chape (double fork)       sigma_f = P / (2·d·t1)
//! traction de la tige              sigma_t = P / ((π/4)·D²)
//! ```
//!
//! `P` charge axiale statique transmise par l'assemblage (N), `d` diamètre de
//! l'axe (m), `t` épaisseur de l'œil central de la chape (m), `t1` épaisseur d'une
//! branche de la fourche (m), `D` diamètre de la tige pleine (m), `tau` contrainte
//! de cisaillement dans l'axe en double cisaillement (Pa), `sigma_e` contrainte de
//! matage (compression) à l'interface axe/œil central (Pa), `sigma_f` contrainte de
//! matage à l'interface axe/fourche répartie sur les deux branches (Pa),
//! `sigma_t` contrainte de traction dans la section pleine de la tige (Pa).
//!
//! **Convention** : SI cohérent — charges en N, dimensions en m, contraintes en Pa.
//!
//! **Limite honnête** : modèle statique élémentaire supposant un **chargement
//! axial pur**, des **axes ajustés sans jeu** (pas de flexion de l'axe assimilé à
//! une poutre) et une **répartition uniforme** des contraintes sur chaque section
//! (deux plans de cisaillement égaux pour l'axe, pression de matage constante,
//! traction constante sur la section pleine). La réalité comporte de la flexion de
//! l'axe si le jeu est présent, des concentrations de contrainte (congés, bords
//! des œillets), une traction affaiblie par la section réduite au droit de l'axe,
//! du frottement et d'éventuelles charges de fatigue — non couverts ici. Les
//! contraintes **admissibles** du matériau et les coefficients de sécurité
//! dépendent du matériau, du procédé et des conditions d'emploi : ils sont
//! **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée dans
//! ce module.

use core::f64::consts::PI;

/// Contrainte de cisaillement dans l'axe en **double cisaillement**
/// `tau = P / (2·(π/4)·d²)`.
///
/// `load` = `P` (N), `pin_diameter` = `d` (m) ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0` ou `pin_diameter <= 0`.
pub fn knuckle_pin_shear_stress(load: f64, pin_diameter: f64) -> f64 {
    assert!(load >= 0.0 && pin_diameter > 0.0, "P ≥ 0 et d > 0 requis");
    load / (2.0 * (PI / 4.0) * pin_diameter.powi(2))
}

/// Contrainte de matage (compression) à l'interface axe/œil central de la chape
/// `sigma_e = P / (d·t)`.
///
/// `load` = `P` (N), `pin_diameter` = `d` (m), `eye_thickness` = `t` (m) ;
/// renvoie une contrainte (Pa).
///
/// Panique si `load < 0`, `pin_diameter <= 0` ou `eye_thickness <= 0`.
pub fn knuckle_eye_crushing_stress(load: f64, pin_diameter: f64, eye_thickness: f64) -> f64 {
    assert!(
        load >= 0.0 && pin_diameter > 0.0 && eye_thickness > 0.0,
        "P ≥ 0, d > 0 et t > 0 requis"
    );
    load / (pin_diameter * eye_thickness)
}

/// Contrainte de matage (compression) à l'interface axe/fourche, répartie sur les
/// **deux branches** de la fourche `sigma_f = P / (2·d·t1)`.
///
/// `load` = `P` (N), `pin_diameter` = `d` (m), `fork_thickness` = `t1` (m) épaisseur
/// d'**une** branche ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0`, `pin_diameter <= 0` ou `fork_thickness <= 0`.
pub fn knuckle_fork_crushing_stress(load: f64, pin_diameter: f64, fork_thickness: f64) -> f64 {
    assert!(
        load >= 0.0 && pin_diameter > 0.0 && fork_thickness > 0.0,
        "P ≥ 0, d > 0 et t1 > 0 requis"
    );
    load / (2.0 * pin_diameter * fork_thickness)
}

/// Contrainte de traction dans la section pleine de la tige
/// `sigma_t = P / ((π/4)·D²)`.
///
/// `load` = `P` (N), `rod_diameter` = `D` (m) ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0` ou `rod_diameter <= 0`.
pub fn knuckle_rod_tensile_stress(load: f64, rod_diameter: f64) -> f64 {
    assert!(load >= 0.0 && rod_diameter > 0.0, "P ≥ 0 et D > 0 requis");
    load / ((PI / 4.0) * rod_diameter.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pin_shear_inverse_of_double_area() {
        // tau·(2·(π/4)·d²) redonne la charge P : réciprocité contrainte ↔ charge.
        let (p, d) = (60e3_f64, 0.030);
        let tau = knuckle_pin_shear_stress(p, d);
        assert_relative_eq!(tau * (2.0 * (PI / 4.0) * d.powi(2)), p, epsilon = 1e-6);
    }

    #[test]
    fn pin_double_shear_is_half_of_single_section_stress() {
        // Le double cisaillement partage la charge sur deux sections : la contrainte
        // vaut la moitié de P/((π/4)·d²).
        let (p, d) = (50e3_f64, 0.025);
        let tau = knuckle_pin_shear_stress(p, d);
        assert_relative_eq!(tau, p / ((PI / 4.0) * d.powi(2)) / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn fork_crushing_is_half_of_eye_crushing_at_equal_thickness() {
        // À épaisseur égale (t = t1), la fourche (deux branches) supporte deux fois
        // plus de surface de matage que l'œil : sigma_f = sigma_e / 2.
        let (p, d, t) = (40e3_f64, 0.028, 0.015);
        let eye = knuckle_eye_crushing_stress(p, d, t);
        let fork = knuckle_fork_crushing_stress(p, d, t);
        assert_relative_eq!(fork, eye / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn rod_tensile_scales_as_inverse_square_of_diameter() {
        // sigma_t ∝ 1/D² : doubler le diamètre divise la contrainte par 4.
        let base = knuckle_rod_tensile_stress(30e3, 0.020);
        let big = knuckle_rod_tensile_stress(30e3, 0.040);
        assert_relative_eq!(big, base / 4.0, epsilon = 1e-6);
    }

    #[test]
    fn realistic_knuckle_joint() {
        // Axe d = 30 mm, œil t = 20 mm, branche t1 = 15 mm, tige D = 40 mm,
        // charge P = 60 kN.
        // Cisaillement double : tau = 60e3 / (2·(π/4)·0,03²) = 42,44 MPa.
        assert_relative_eq!(
            knuckle_pin_shear_stress(60e3, 0.030),
            42.441e6,
            epsilon = 1e3
        );
        // Matage œil : sigma_e = 60e3 / (0,03·0,02) = 100 MPa.
        assert_relative_eq!(
            knuckle_eye_crushing_stress(60e3, 0.030, 0.020),
            100.0e6,
            epsilon = 1.0
        );
        // Matage fourche : sigma_f = 60e3 / (2·0,03·0,015) = 66,67 MPa.
        assert_relative_eq!(
            knuckle_fork_crushing_stress(60e3, 0.030, 0.015),
            66.6667e6,
            epsilon = 1e3
        );
        // Traction tige : sigma_t = 60e3 / ((π/4)·0,04²) = 47,75 MPa.
        assert_relative_eq!(
            knuckle_rod_tensile_stress(60e3, 0.040),
            47.746e6,
            epsilon = 1e3
        );
    }

    #[test]
    fn zero_load_gives_zero_stress() {
        // Cas limite : charge nulle → contraintes nulles sur toutes les sections.
        assert_relative_eq!(knuckle_pin_shear_stress(0.0, 0.030), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            knuckle_eye_crushing_stress(0.0, 0.030, 0.020),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            knuckle_fork_crushing_stress(0.0, 0.030, 0.015),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(knuckle_rod_tensile_stress(0.0, 0.040), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "d > 0")]
    fn zero_pin_diameter_panics() {
        knuckle_pin_shear_stress(1000.0, 0.0);
    }
}
