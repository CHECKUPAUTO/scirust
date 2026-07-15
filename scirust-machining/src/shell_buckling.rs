//! Flambage d'une coque cylindrique mince — contrainte critique en compression
//! axiale (théorie classique), pression externe critique (tube long) et charge
//! axiale critique associée.
//!
//! ```text
//! contrainte axiale critique  σcr = E·t / (r·√(3·(1−ν²)))
//! pression externe critique   pcr = E/(4·(1−ν²)) · (t/r)³     (tube long)
//! charge axiale critique      Ncr = σ·2·π·r·t
//! ```
//!
//! `E` module de Young (Pa), `t` épaisseur de paroi (m), `r` rayon moyen (m),
//! `ν` coefficient de Poisson (sans dimension), `σ` contrainte de compression
//! (Pa), `p` pression (Pa), `N` charge axiale (N).
//!
//! **Convention** : SI cohérent, coque supposée mince (`t ≪ r`). **Limite
//! honnête** : théorie **classique** de la coque parfaite (élastique linéaire).
//! Le flambage réel est très sensible aux imperfections géométriques : le
//! facteur d'abattement (« knock-down factor ») est **fourni par l'appelant** ;
//! aucune valeur de `E`, `ν`, `t/r` ni knock-down n'est inventée ici.

use core::f64::consts::PI;

/// Contrainte axiale critique de flambage d'une coque cylindrique mince
/// `σcr = E·t / (r·√(3·(1−ν²)))` (Pa), compression axiale, théorie classique.
///
/// Panique si `youngs_modulus <= 0`, `thickness <= 0`, `radius <= 0`, ou si
/// `poisson_ratio` n'est pas dans `[0, 0,5[`.
pub fn shell_axial_critical_stress(
    youngs_modulus_pa: f64,
    thickness_m: f64,
    radius_m: f64,
    poisson_ratio: f64,
) -> f64 {
    assert!(
        youngs_modulus_pa > 0.0,
        "le module de Young doit être strictement positif"
    );
    assert!(
        thickness_m > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(radius_m > 0.0, "le rayon doit être strictement positif");
    assert!(
        (0.0..0.5).contains(&poisson_ratio),
        "le coefficient de Poisson doit être dans [0, 0,5["
    );
    youngs_modulus_pa * thickness_m
        / (radius_m * (3.0 * (1.0 - poisson_ratio * poisson_ratio)).sqrt())
}

/// Pression externe critique d'un tube cylindrique **long**
/// `pcr = E/(4·(1−ν²)) · (t/r)³` (Pa) — forme de von Mises/Donnell dans la
/// limite `L ≫ r` (l'influence de la longueur devient négligeable).
///
/// Panique si `youngs_modulus <= 0`, `thickness <= 0`, `radius <= 0`,
/// `length <= 0`, ou si `poisson_ratio` n'est pas dans `[0, 0,5[`.
pub fn shell_external_pressure_critical(
    youngs_modulus_pa: f64,
    thickness_m: f64,
    radius_m: f64,
    length_m: f64,
    poisson_ratio: f64,
) -> f64 {
    assert!(
        youngs_modulus_pa > 0.0,
        "le module de Young doit être strictement positif"
    );
    assert!(
        thickness_m > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(radius_m > 0.0, "le rayon doit être strictement positif");
    assert!(length_m > 0.0, "la longueur doit être strictement positive");
    assert!(
        (0.0..0.5).contains(&poisson_ratio),
        "le coefficient de Poisson doit être dans [0, 0,5["
    );
    let ratio = thickness_m / radius_m;
    youngs_modulus_pa / (4.0 * (1.0 - poisson_ratio * poisson_ratio)) * ratio.powi(3)
}

/// Charge axiale critique `Ncr = σ·2·π·r·t` (N) : contrainte de compression
/// multipliée par l'aire de la section annulaire mince `2·π·r·t`.
///
/// Panique si `critical_stress < 0`, `radius <= 0`, ou `thickness <= 0`.
pub fn shell_critical_axial_load(critical_stress_pa: f64, radius_m: f64, thickness_m: f64) -> f64 {
    assert!(
        critical_stress_pa >= 0.0,
        "la contrainte critique doit être positive ou nulle"
    );
    assert!(radius_m > 0.0, "le rayon doit être strictement positif");
    assert!(
        thickness_m > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    critical_stress_pa * 2.0 * PI * radius_m * thickness_m
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_critical_stress_matches_closed_form() {
        // E=210 GPa, t=2 mm, r=0,5 m, ν=0,3.
        // √(3·(1−0,09)) = √2,73 = 1,6522711...
        // σcr = 210e9·0,002 / (0,5·1,6522711) = 4,20e8 / 0,82613556 = 5,0839e8 Pa.
        let sigma = shell_axial_critical_stress(210e9, 0.002, 0.5, 0.3);
        let denom = 0.5_f64 * (3.0_f64 * (1.0 - 0.3_f64 * 0.3)).sqrt();
        assert_relative_eq!(sigma, 210e9 * 0.002 / denom, epsilon = 1e-6);
        assert_relative_eq!(sigma, 5.0839e8, epsilon = 5e4);
    }

    #[test]
    fn axial_stress_is_linear_in_thickness() {
        // À r, E, ν fixés, σcr ∝ t : doubler t double la contrainte.
        let s1 = shell_axial_critical_stress(210e9, 0.002, 0.5, 0.3);
        let s2 = shell_axial_critical_stress(210e9, 0.004, 0.5, 0.3);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-6);
    }

    #[test]
    fn critical_load_is_stress_times_annular_area() {
        // Réciprocité : Ncr/(2πrt) doit redonner la contrainte de départ.
        let (r, t) = (0.5_f64, 0.002_f64);
        let sigma = shell_axial_critical_stress(210e9, t, r, 0.3);
        let n = shell_critical_axial_load(sigma, r, t);
        assert_relative_eq!(n / (2.0 * PI * r * t), sigma, epsilon = 1e-3);
    }

    #[test]
    fn external_pressure_scales_as_cube_of_thickness_ratio() {
        // pcr ∝ (t/r)³ : tripler t/r multiplie la pression par 27.
        let p1 = shell_external_pressure_critical(210e9, 0.002, 0.5, 5.0, 0.3);
        let p3 = shell_external_pressure_critical(210e9, 0.006, 0.5, 5.0, 0.3);
        assert_relative_eq!(p3, 27.0 * p1, epsilon = 1e-6);
    }

    #[test]
    fn external_pressure_matches_closed_form() {
        // E=210 GPa, t=2 mm, r=0,5 m, ν=0,3 → t/r=0,004.
        // pcr = 210e9/(4·0,91)·(0,004)³ = 5,7692e10·6,4e-8 = 3692,3 Pa.
        let p = shell_external_pressure_critical(210e9, 0.002, 0.5, 5.0, 0.3);
        let expected = 210e9 / (4.0 * (1.0 - 0.3_f64 * 0.3)) * (0.004_f64).powi(3);
        assert_relative_eq!(p, expected, epsilon = 1e-9);
        assert_relative_eq!(p, 3692.3, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "coefficient de Poisson")]
    fn out_of_range_poisson_panics() {
        shell_axial_critical_stress(210e9, 0.002, 0.5, 0.6);
    }
}
