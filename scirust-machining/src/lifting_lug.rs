//! Oreille de levage / chape (statique) — contraintes de dimensionnement d'un
//! trou d'articulation chargé par un axe (matage, section nette, cisaillement
//! de l'axe en double cisaillement).
//!
//! ```text
//! matage sur l'axe            σ_br = F/(d·t)
//! contrainte section nette    σ_net = F/((w − d)·t)
//! cisaillement double de l'axe τ = F/(2·A)      avec A = π·d²/4
//! ```
//!
//! `F` charge appliquée à l'oreille (N), `d` diamètre de l'axe / du trou (m),
//! `t` épaisseur de la tôle de l'oreille (m), `w` largeur de la tôle au droit
//! du trou (m), `A` aire de section de l'axe (m²), `σ_br` pression de matage
//! (Pa), `σ_net` contrainte de traction dans la section nette (Pa), `τ`
//! contrainte de cisaillement dans l'axe (Pa).
//!
//! **Convention** : SI cohérent (N, m, Pa). **Limite honnête** : chargement
//! **statique** aligné avec la traction de l'oreille ; contraintes nominales
//! moyennées sur la section, **sans** facteur de concentration de contrainte au
//! bord du trou, **sans** vérification de fatigue ni de rupture du bord (arrachement
//! de « joue »). Le double cisaillement suppose une chape à deux flasques
//! reprenant l'axe sur deux plans. Les contraintes admissibles, coefficients de
//! sécurité et propriétés matériaux sont **fournis par l'appelant** (aucune
//! valeur « par défaut » n'est inventée ici).

use core::f64::consts::PI;

/// Pression de **matage** sur l'axe `σ_br = F/(d·t)` (Pa).
///
/// Contrainte de contact nominale entre l'axe et le bord du trou, moyennée sur
/// la surface projetée `d·t`.
///
/// Panique si `load < 0`, `pin_diameter <= 0` ou `thickness <= 0`.
pub fn lug_pin_bearing_stress(load: f64, pin_diameter: f64, thickness: f64) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive");
    assert!(
        pin_diameter > 0.0,
        "le diamètre de l'axe doit être strictement positif"
    );
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    load / (pin_diameter * thickness)
}

/// Contrainte de traction dans la **section nette** `σ_net = F/((w − d)·t)` (Pa).
///
/// Contrainte moyenne dans la matière restante de part et d'autre du trou, sur
/// la section nette de largeur `w − d` et d'épaisseur `t`.
///
/// Panique si `load < 0`, `thickness <= 0` ou si `plate_width <= hole_diameter`
/// (section nette nulle ou négative).
pub fn lug_net_section_stress(
    load: f64,
    plate_width: f64,
    hole_diameter: f64,
    thickness: f64,
) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive");
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(
        plate_width > hole_diameter,
        "la largeur de tôle doit être supérieure au diamètre du trou"
    );
    load / ((plate_width - hole_diameter) * thickness)
}

/// Aire de section de l'axe `A = π·d²/4` (m²).
///
/// Panique si `pin_diameter <= 0`.
pub fn lug_pin_area(pin_diameter: f64) -> f64 {
    assert!(
        pin_diameter > 0.0,
        "le diamètre de l'axe doit être strictement positif"
    );
    PI * pin_diameter * pin_diameter / 4.0
}

/// Contrainte de **cisaillement double** de l'axe `τ = F/(2·A)` (Pa).
///
/// L'axe d'une chape à deux flasques est cisaillé sur deux plans ; la
/// contrainte se répartit donc sur `2·A`.
///
/// Panique si `load < 0` ou `pin_area <= 0`.
pub fn lug_double_shear_stress(load: f64, pin_area: f64) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive");
    assert!(
        pin_area > 0.0,
        "l'aire de l'axe doit être strictement positive"
    );
    load / (2.0 * pin_area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bearing_stress_scales_inversely_with_area() {
        // σ_br = F/(d·t) : doubler t divise la pression de matage par 2.
        let f = 50_000.0;
        let d = 0.020;
        let t = 0.010;
        let s1 = lug_pin_bearing_stress(f, d, t);
        let s2 = lug_pin_bearing_stress(f, d, 2.0 * t);
        assert_relative_eq!(s1, 2.0 * s2, epsilon = 1e-9);
        // Cohérence dimensionnelle : σ_br·(d·t) = F.
        assert_relative_eq!(s1 * (d * t), f, epsilon = 1e-6);
    }

    #[test]
    fn net_section_uses_remaining_width() {
        // w=60 mm, d=20 mm, t=10 mm, F=40 000 N → σ_net = F/((w−d)·t).
        // En N/mm² : 40000/((60−20)·10) = 100 MPa.
        let sigma = lug_net_section_stress(40_000.0, 60.0, 20.0, 10.0);
        assert_relative_eq!(sigma, 100.0, epsilon = 1e-9);
    }

    #[test]
    fn net_section_diverges_as_width_approaches_diameter() {
        // Quand w → d⁺, la section nette → 0 donc σ_net → ∞ (croissance stricte).
        let f = 10_000.0;
        let d = 20.0;
        let t = 5.0;
        let s_wide = lug_net_section_stress(f, 40.0, d, t);
        let s_narrow = lug_net_section_stress(f, 22.0, d, t);
        assert!(s_narrow > s_wide);
    }

    #[test]
    fn double_shear_is_half_of_single_shear() {
        // Identité : sur deux plans, τ vaut la moitié de F/A (simple cisaillement).
        let f = 80_000.0;
        let d = 0.025;
        let a = lug_pin_area(d);
        assert_relative_eq!(a, PI * d * d / 4.0, epsilon = 1e-12);
        let tau_double = lug_double_shear_stress(f, a);
        let tau_single = f / a;
        assert_relative_eq!(tau_double, tau_single / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn double_shear_known_value() {
        // d=20 mm → A = π·100 = 314.159 mm² ; F=25 000 N.
        // τ = 25000/(2·314.159) ≈ 39.789 MPa (unités N, mm, MPa).
        let a = lug_pin_area(20.0);
        let tau = lug_double_shear_stress(25_000.0, a);
        assert_relative_eq!(tau, 25_000.0 / (2.0 * PI * 100.0), epsilon = 1e-9);
        assert_relative_eq!(tau, 39.788_735_77, epsilon = 1e-6);
    }

    #[test]
    fn stresses_are_proportional_to_load() {
        // Toutes les contraintes sont linéaires en F (élasticité, statique).
        let d = 0.018;
        let t = 0.008;
        let a = lug_pin_area(d);
        let br = lug_pin_bearing_stress(2.0 * 1_000.0, d, t);
        let br0 = lug_pin_bearing_stress(1_000.0, d, t);
        let sh = lug_double_shear_stress(2.0 * 1_000.0, a);
        let sh0 = lug_double_shear_stress(1_000.0, a);
        assert_relative_eq!(br, 2.0 * br0, epsilon = 1e-9);
        assert_relative_eq!(sh, 2.0 * sh0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "largeur de tôle doit être supérieure")]
    fn net_section_zero_width_panics() {
        // w = d : section nette nulle → interdit.
        lug_net_section_stress(1_000.0, 20.0, 20.0, 10.0);
    }
}
