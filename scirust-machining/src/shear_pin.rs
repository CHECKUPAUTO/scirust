//! Goupille de cisaillement / limiteur de couple — dimensionnement à la rupture
//! par cisaillement direct de la section pleine.
//!
//! ```text
//! aire cisaillée    A = n·π/4·d²
//! effort de rupture F = τ_u·A = τ_u·n·π/4·d²
//! couple de rupture C = F·r = τ_u·(n·π/4·d²)·r
//! diamètre visé     d = sqrt( 4·C / (τ_u·n·π·r) )
//! ```
//!
//! `d` diamètre de la goupille (m), `n` nombre de plans de cisaillement (1 ou 2),
//! `A` aire cisaillée totale (m²), `τ_u` résistance au cisaillement ultime (Pa),
//! `F` effort de cisaillement à la rupture (N), `r` rayon d'implantation de la
//! goupille sur son cercle (m), `C` couple de déclenchement / rupture (N·m). Sur
//! `n` plans, la même goupille présente `n` sections cisaillées en parallèle.
//!
//! **Limite honnête** : rupture par **cisaillement direct** de la section pleine ;
//! la résistance au cisaillement ultime `τ_u` (usuellement ~0,6·Rm mais **fournie
//! par l'appelant**), le nombre de plans et le rayon d'implantation `r` sont
//! **fournis** ; la flexion de la goupille et la concentration de contrainte sont
//! **ignorées**. Aucune constante matériau ou procédé n'est inventée ici.

use core::f64::consts::PI;

/// Aire cisaillée totale d'une goupille `A = n·π/4·d²` (m²).
///
/// La goupille présente `n` sections circulaires pleines cisaillées en parallèle
/// (1 plan pour un montage en console, 2 plans pour une chape).
///
/// Panique si `pin_diameter <= 0` ou `shear_planes <= 0`.
pub fn shearpin_shear_area(pin_diameter: f64, shear_planes: f64) -> f64 {
    assert!(
        pin_diameter > 0.0,
        "le diamètre de la goupille doit être strictement positif"
    );
    assert!(
        shear_planes > 0.0,
        "le nombre de plans de cisaillement doit être strictement positif"
    );
    shear_planes * PI * pin_diameter * pin_diameter / 4.0
}

/// Effort de cisaillement à la rupture `F = τ_u·n·π/4·d²` (N).
///
/// Produit de la résistance au cisaillement ultime par l'aire cisaillée totale.
///
/// Panique si `ultimate_shear_strength < 0`, `pin_diameter <= 0` ou
/// `shear_planes <= 0`.
pub fn shearpin_breaking_force(
    ultimate_shear_strength: f64,
    pin_diameter: f64,
    shear_planes: f64,
) -> f64 {
    assert!(
        ultimate_shear_strength >= 0.0,
        "la résistance au cisaillement ultime doit être positive ou nulle"
    );
    assert!(
        pin_diameter > 0.0,
        "le diamètre de la goupille doit être strictement positif"
    );
    assert!(
        shear_planes > 0.0,
        "le nombre de plans de cisaillement doit être strictement positif"
    );
    ultimate_shear_strength * shear_planes * PI * pin_diameter * pin_diameter / 4.0
}

/// Couple de rupture / de déclenchement `C = τ_u·(n·π/4·d²)·r` (N·m).
///
/// L'effort de cisaillement à la rupture multiplié par le rayon d'implantation
/// de la goupille donne le couple maximal transmissible avant rupture.
///
/// Panique si `ultimate_shear_strength < 0`, `pin_diameter <= 0`,
/// `shear_planes <= 0` ou `pin_circle_radius <= 0`.
pub fn shearpin_breaking_torque(
    ultimate_shear_strength: f64,
    pin_diameter: f64,
    shear_planes: f64,
    pin_circle_radius: f64,
) -> f64 {
    assert!(
        pin_circle_radius > 0.0,
        "le rayon d'implantation doit être strictement positif"
    );
    shearpin_breaking_force(ultimate_shear_strength, pin_diameter, shear_planes) * pin_circle_radius
}

/// Diamètre de goupille pour un couple de déclenchement visé
/// `d = sqrt( 4·C / (τ_u·n·π·r) )` (m).
///
/// Inverse de [`shearpin_breaking_torque`] : donne le diamètre qui rompt
/// exactement au couple cible.
///
/// Panique si `target_torque < 0`, `ultimate_shear_strength <= 0`,
/// `shear_planes <= 0` ou `pin_circle_radius <= 0`.
pub fn shearpin_diameter_for_torque(
    target_torque: f64,
    ultimate_shear_strength: f64,
    shear_planes: f64,
    pin_circle_radius: f64,
) -> f64 {
    assert!(
        target_torque >= 0.0,
        "le couple visé doit être positif ou nul"
    );
    assert!(
        ultimate_shear_strength > 0.0,
        "la résistance au cisaillement ultime doit être strictement positive"
    );
    assert!(
        shear_planes > 0.0,
        "le nombre de plans de cisaillement doit être strictement positif"
    );
    assert!(
        pin_circle_radius > 0.0,
        "le rayon d'implantation doit être strictement positif"
    );
    (4.0 * target_torque / (ultimate_shear_strength * shear_planes * PI * pin_circle_radius)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shear_area_double_is_twice_single() {
        // Identité : 2 plans → deux fois l'aire d'un seul plan.
        let single = shearpin_shear_area(0.005, 1.0);
        let double = shearpin_shear_area(0.005, 2.0);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-15);
    }

    #[test]
    fn shear_area_numeric_case() {
        // d = 5 mm, 1 plan : A = π/4·(0,005)² = π·6,25e-6 = 1,963495e-5 m².
        let area = shearpin_shear_area(0.005, 1.0);
        assert_relative_eq!(area, PI * 6.25e-6, epsilon = 1e-15);
        assert_relative_eq!(area, 1.963_495e-5, max_relative = 1e-3);
    }

    #[test]
    fn breaking_force_matches_strength_times_area() {
        // Identité de définition : F = τ_u·A.
        let tau_u = 360.0e6_f64;
        let force = shearpin_breaking_force(tau_u, 0.005, 1.0);
        let area = shearpin_shear_area(0.005, 1.0);
        assert_relative_eq!(force, tau_u * area, epsilon = 1e-6);
        // Cas chiffré : 360 MPa × 1,963495e-5 m² ≈ 7068,58 N.
        assert_relative_eq!(force, 7068.58, max_relative = 1e-3);
    }

    #[test]
    fn breaking_torque_is_force_times_radius() {
        // Identité : C = F·r.
        let tau_u = 360.0e6_f64;
        let force = shearpin_breaking_force(tau_u, 0.005, 1.0);
        let torque = shearpin_breaking_torque(tau_u, 0.005, 1.0, 0.020);
        assert_relative_eq!(torque, force * 0.020, epsilon = 1e-9);
        // ≈ 7068,58 N × 0,020 m ≈ 141,37 N·m.
        assert_relative_eq!(torque, 141.37, max_relative = 1e-3);
    }

    #[test]
    fn diameter_for_torque_inverts_breaking_torque() {
        // Réciprocité : le diamètre calculé pour le couple de rupture redonne
        // le diamètre d'origine.
        let tau_u = 360.0e6_f64;
        let (d, n, r) = (0.005_f64, 1.0_f64, 0.020_f64);
        let torque = shearpin_breaking_torque(tau_u, d, n, r);
        let d_back = shearpin_diameter_for_torque(torque, tau_u, n, r);
        assert_relative_eq!(d_back, d, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "diamètre de la goupille")]
    fn zero_diameter_area_panics() {
        shearpin_shear_area(0.0, 1.0);
    }
}
