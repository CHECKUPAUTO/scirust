//! Mécanique du contact de **Hertz** — pression et dimensions de contact entre
//! solides élastiques, en contact linéaire (deux cylindres parallèles) ou
//! ponctuel (deux sphères). Base de dimensionnement des engrenages (pression au
//! flanc), roulements, cames et contacts roue-rail.
//!
//! Deux propriétés effectives résument le couple de solides :
//!
//! ```text
//! module effectif :  1/E* = (1−ν₁²)/E₁ + (1−ν₂²)/E₂
//! rayon effectif  :  1/R  = 1/R₁ + 1/R₂        (R = ∞ pour un plan)
//! ```
//!
//! **Contact linéaire** (effort par unité de longueur `w = F/L`) :
//!
//! ```text
//! demi-largeur   b = √(4·w·R / (π·E*))
//! pression max   p₀ = √(w·E* / (π·R)) = 2w/(π·b)
//! ```
//!
//! **Contact ponctuel** (effort `F`, contact circulaire) :
//!
//! ```text
//! rayon          a = ∛(3·F·R / (4·E*))
//! pression max   p₀ = 3F/(2π·a²)
//! ```
//!
//! **Limite honnête** : théorie de Hertz — solides parfaitement élastiques,
//! isotropes, en petites déformations, sans frottement ni lubrification, surfaces
//! lisses et contact non conforme. Elle donne la pression de contact statique ;
//! elle ne couvre ni la plastification (au-delà de la limite d'élasticité), ni la
//! fatigue de contact (durée de vie), ni les effets élastohydrodynamiques (EHD)
//! d'un contact lubrifié. Unités : `E` et pressions en MPa, longueurs en mm,
//! efforts en N, `w` en N/mm.

use core::f64::consts::PI;

/// Rayon effectif `R` (mm) de deux surfaces de rayons `r1`, `r2` (mm) :
/// `1/R = 1/r1 + 1/r2`. Passer `f64::INFINITY` pour une surface plane.
///
/// Panique si les deux rayons sont infinis (contact plan/plan indéfini).
pub fn effective_radius(r1_mm: f64, r2_mm: f64) -> f64 {
    let inv = 1.0 / r1_mm + 1.0 / r2_mm;
    assert!(inv > 0.0, "au moins une surface doit être courbe");
    1.0 / inv
}

/// Module d'élasticité effectif `E*` (MPa) du couple de matériaux :
/// `1/E* = (1−ν₁²)/E₁ + (1−ν₂²)/E₂`.
///
/// `e1`, `e2` sont les modules de Young (MPa) et `nu1`, `nu2` les coefficients
/// de Poisson. Panique si un module est non strictement positif.
pub fn effective_modulus(e1_mpa: f64, nu1: f64, e2_mpa: f64, nu2: f64) -> f64 {
    assert!(
        e1_mpa > 0.0 && e2_mpa > 0.0,
        "les modules de Young doivent être strictement positifs"
    );
    1.0 / ((1.0 - nu1 * nu1) / e1_mpa + (1.0 - nu2 * nu2) / e2_mpa)
}

/// Demi-largeur `b` (mm) du contact linéaire pour un effort linéique
/// `load_per_length` (N/mm), un rayon effectif `r_eff` (mm) et un module
/// effectif `e_star` (MPa) : `b = √(4·w·R / (π·E*))`.
///
/// Panique si une entrée est non strictement positive.
pub fn line_contact_half_width(load_per_length_n_mm: f64, r_eff_mm: f64, e_star_mpa: f64) -> f64 {
    assert!(
        load_per_length_n_mm > 0.0 && r_eff_mm > 0.0 && e_star_mpa > 0.0,
        "effort linéique, rayon effectif et module effectif doivent être positifs"
    );
    (4.0 * load_per_length_n_mm * r_eff_mm / (PI * e_star_mpa)).sqrt()
}

/// Pression de contact maximale `p₀` (MPa) d'un contact linéaire :
/// `p₀ = √(w·E* / (π·R))`.
///
/// Panique si une entrée est non strictement positive.
pub fn line_contact_max_pressure(load_per_length_n_mm: f64, r_eff_mm: f64, e_star_mpa: f64) -> f64 {
    assert!(
        load_per_length_n_mm > 0.0 && r_eff_mm > 0.0 && e_star_mpa > 0.0,
        "effort linéique, rayon effectif et module effectif doivent être positifs"
    );
    (load_per_length_n_mm * e_star_mpa / (PI * r_eff_mm)).sqrt()
}

/// Rayon `a` (mm) du contact ponctuel circulaire pour un effort `force` (N),
/// un rayon effectif `r_eff` (mm) et un module effectif `e_star` (MPa) :
/// `a = ∛(3·F·R / (4·E*))`.
///
/// Panique si une entrée est non strictement positive.
pub fn point_contact_radius(force_n: f64, r_eff_mm: f64, e_star_mpa: f64) -> f64 {
    assert!(
        force_n > 0.0 && r_eff_mm > 0.0 && e_star_mpa > 0.0,
        "effort, rayon effectif et module effectif doivent être positifs"
    );
    (3.0 * force_n * r_eff_mm / (4.0 * e_star_mpa)).cbrt()
}

/// Pression de contact maximale `p₀` (MPa) d'un contact ponctuel :
/// `p₀ = 3F / (2π·a²)`, `a` étant le rayon donné par [`point_contact_radius`].
///
/// Panique si une entrée est non strictement positive.
pub fn point_contact_max_pressure(force_n: f64, r_eff_mm: f64, e_star_mpa: f64) -> f64 {
    let a = point_contact_radius(force_n, r_eff_mm, e_star_mpa);
    3.0 * force_n / (2.0 * PI * a * a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Acier : E = 210 000 MPa, ν = 0,3 → E* = 210000/1,82 ≈ 115384,6 MPa.
    fn steel_pair() -> f64 {
        effective_modulus(210_000.0, 0.3, 210_000.0, 0.3)
    }

    #[test]
    fn effective_modulus_of_steel_pair() {
        assert_relative_eq!(steel_pair(), 210_000.0 / 1.82, epsilon = 1e-6);
    }

    #[test]
    fn effective_radius_of_equal_cylinders_is_half() {
        // 1/R = 1/10 + 1/10 = 1/5 → R = 5 mm.
        assert_relative_eq!(effective_radius(10.0, 10.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn effective_radius_against_a_plane_is_the_curved_radius() {
        // Plan ⇒ r2 = ∞ : R = r1.
        assert_relative_eq!(effective_radius(8.0, f64::INFINITY), 8.0, epsilon = 1e-12);
    }

    #[test]
    fn line_contact_two_pressure_formulas_agree() {
        // p₀ direct = 2w/(π·b) reconstruit à partir de la demi-largeur.
        let e = steel_pair();
        let (w, r) = (100.0, 5.0);
        let b = line_contact_half_width(w, r, e);
        let p_direct = line_contact_max_pressure(w, r, e);
        let p_from_b = 2.0 * w / (PI * b);
        assert_relative_eq!(p_direct, p_from_b, epsilon = 1e-9);
        // valeur numérique attendue ≈ 857 MPa.
        assert_relative_eq!(p_direct, 857.0, epsilon = 1.0);
    }

    #[test]
    fn point_contact_two_pressure_formulas_agree() {
        // p₀ = 3F/(2π·a²) avec a = point_contact_radius : cohérence interne.
        let e = steel_pair();
        let (f, r) = (500.0, 5.0);
        let a = point_contact_radius(f, r, e);
        let p_direct = point_contact_max_pressure(f, r, e);
        let p_from_a = 3.0 * f / (2.0 * PI * a * a);
        assert_relative_eq!(p_direct, p_from_a, epsilon = 1e-9);
    }

    #[test]
    fn heavier_load_raises_contact_pressure() {
        let e = steel_pair();
        let light = line_contact_max_pressure(50.0, 5.0, e);
        let heavy = line_contact_max_pressure(200.0, 5.0, e);
        assert!(heavy > light);
    }

    #[test]
    #[should_panic(expected = "au moins une surface")]
    fn two_planes_have_no_defined_contact() {
        effective_radius(f64::INFINITY, f64::INFINITY);
    }
}
