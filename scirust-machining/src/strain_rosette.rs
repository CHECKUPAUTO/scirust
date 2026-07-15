//! Rosette de déformation **rectangulaire** à trois jauges (0°/45°/90°) :
//! déformations principales, orientation et déformation de cisaillement maximale
//! en surface.
//!
//! ```text
//! déformation moyenne   ε_moy = (ε0 + ε90)/2
//! rayon (cercle de Mohr) R = √( ((ε0−ε90)/2)² + ((2·ε45−ε0−ε90)/2)² )
//! principales           ε1 = ε_moy + R ,  ε2 = ε_moy − R      (ε1 ≥ ε2)
//! orientation           θp = ½·atan2(2·ε45−ε0−ε90 , ε0−ε90)
//! cisaillement maximal   γmax = ε1 − ε2 = 2·R
//! ```
//!
//! `ε0, ε45, ε90` déformations mesurées par les trois jauges (sans dimension,
//! m/m ou µε de façon cohérente), `θp` en radians, `γmax` déformation de
//! cisaillement d'ingénierie (sans dimension). Traction positive.
//!
//! **Limite honnête** : rosette **RECTANGULAIRE** (jauges à 0°/45°/90°), état de
//! **déformation plane** en surface libre, jauges FOURNIES par l'appelant.
//! Aucune constante matériau/procédé n'est inventée ici : la conversion des
//! déformations en contraintes (module d'Young `E`, coefficient de Poisson `ν`)
//! est à la charge de l'appelant via [`crate::hookes_law_3d`].

/// Vérifie que toutes les déformations d'une jauge sont finies.
///
/// Panique si l'une des déformations n'est pas finie (NaN ou infinie).
fn assert_finite_strains(strain_0: f64, strain_45: f64, strain_90: f64) {
    assert!(
        strain_0.is_finite() && strain_45.is_finite() && strain_90.is_finite(),
        "les déformations des trois jauges doivent être finies"
    );
}

/// Déformation moyenne `ε_moy = (ε0 + ε90)/2` de la rosette rectangulaire.
///
/// Panique si l'une des déformations n'est pas finie.
pub fn rosette_mean_strain(strain_0: f64, strain_45: f64, strain_90: f64) -> f64 {
    assert_finite_strains(strain_0, strain_45, strain_90);
    (strain_0 + strain_90) / 2.0
}

/// Rayon du cercle de Mohr des déformations
/// `R = √( ((ε0−ε90)/2)² + ((2·ε45−ε0−ε90)/2)² )` pour une rosette rectangulaire.
///
/// Panique si l'une des déformations n'est pas finie.
pub fn rosette_mohr_radius(strain_0: f64, strain_45: f64, strain_90: f64) -> f64 {
    assert_finite_strains(strain_0, strain_45, strain_90);
    let a = (strain_0 - strain_90) / 2.0;
    let b = (2.0 * strain_45 - strain_0 - strain_90) / 2.0;
    (a * a + b * b).sqrt()
}

/// Déformations principales `(ε1, ε2)` avec `ε1 ≥ ε2`, d'une rosette
/// rectangulaire 0°/45°/90° : `ε1 = ε_moy + R`, `ε2 = ε_moy − R`.
///
/// Panique si l'une des déformations n'est pas finie.
pub fn rosette_principal_strains_rectangular(
    strain_0: f64,
    strain_45: f64,
    strain_90: f64,
) -> (f64, f64) {
    let mean = rosette_mean_strain(strain_0, strain_45, strain_90);
    let r = rosette_mohr_radius(strain_0, strain_45, strain_90);
    (mean + r, mean - r)
}

/// Orientation de la direction principale (rad)
/// `θp = ½·atan2(2·ε45−ε0−ε90 , ε0−ε90)` pour une rosette rectangulaire.
///
/// Panique si l'une des déformations n'est pas finie.
pub fn rosette_principal_angle_rectangular(strain_0: f64, strain_45: f64, strain_90: f64) -> f64 {
    assert_finite_strains(strain_0, strain_45, strain_90);
    0.5 * (2.0 * strain_45 - strain_0 - strain_90).atan2(strain_0 - strain_90)
}

/// Déformation de cisaillement (d'ingénierie) maximale dans le plan
/// `γmax = ε1 − ε2 = 2·R` pour une rosette rectangulaire.
///
/// Panique si l'une des déformations n'est pas finie.
pub fn rosette_max_shear_strain(strain_0: f64, strain_45: f64, strain_90: f64) -> f64 {
    2.0 * rosette_mohr_radius(strain_0, strain_45, strain_90)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_8, SQRT_2};

    #[test]
    fn principal_sum_equals_gauge_sum() {
        // Identité : ε1 + ε2 = 2·ε_moy = ε0 + ε90 (invariant de trace 2D).
        let (e0, e45, e90) = (600e-6, 250e-6, -150e-6);
        let (eps1, eps2) = rosette_principal_strains_rectangular(e0, e45, e90);
        assert_relative_eq!(eps1 + eps2, e0 + e90, epsilon = 1e-15);
    }

    #[test]
    fn max_shear_equals_difference_of_principals() {
        // Identité : γmax = ε1 − ε2 = 2·R.
        let (e0, e45, e90) = (300e-6, 500e-6, 100e-6);
        let (eps1, eps2) = rosette_principal_strains_rectangular(e0, e45, e90);
        assert_relative_eq!(
            rosette_max_shear_strain(e0, e45, e90),
            eps1 - eps2,
            epsilon = 1e-15
        );
    }

    #[test]
    fn hydrostatic_state_has_no_shear() {
        // Déformation isotrope ε0=ε45=ε90=c : R=0, ε1=ε2=c, θp=0, γmax=0.
        let c = 420e-6;
        let (eps1, eps2) = rosette_principal_strains_rectangular(c, c, c);
        assert_relative_eq!(eps1, c, epsilon = 1e-15);
        assert_relative_eq!(eps2, c, epsilon = 1e-15);
        assert_relative_eq!(rosette_max_shear_strain(c, c, c), 0.0, epsilon = 1e-15);
        assert_relative_eq!(
            rosette_principal_angle_rectangular(c, c, c),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn gauges_aligned_with_principal_axes_give_zero_angle() {
        // Si 2·ε45 = ε0 + ε90, le terme de cisaillement s'annule : θp = 0 et les
        // jauges 0°/90° sont déjà principales (ε1=ε0, ε2=ε90 quand ε0>ε90).
        let (e0, e90) = (400e-6, -200e-6);
        let e45 = (e0 + e90) / 2.0;
        assert_relative_eq!(
            rosette_principal_angle_rectangular(e0, e45, e90),
            0.0,
            epsilon = 1e-15
        );
        let (eps1, eps2) = rosette_principal_strains_rectangular(e0, e45, e90);
        assert_relative_eq!(eps1, e0, epsilon = 1e-15);
        assert_relative_eq!(eps2, e90, epsilon = 1e-15);
    }

    #[test]
    fn realistic_case_computed_by_hand() {
        // Cas chiffré : ε0=200µε, ε45=200µε, ε90=0.
        // ε_moy = 100µε ; (ε0−ε90)/2 = 100µε ; (2ε45−ε0−ε90)/2 = 100µε ;
        // R = 100µε·√2 ; ε1 = 100µε·(1+√2) ; ε2 = 100µε·(1−√2) ;
        // γmax = 2R = 200µε·√2 ; θp = ½·atan2(200,200) = ½·(π/4) = π/8.
        let (e0, e45, e90) = (200e-6, 200e-6, 0.0);
        let (eps1, eps2) = rosette_principal_strains_rectangular(e0, e45, e90);
        assert_relative_eq!(eps1, 100e-6 * (1.0 + SQRT_2), epsilon = 1e-15);
        assert_relative_eq!(eps2, 100e-6 * (1.0 - SQRT_2), epsilon = 1e-15);
        assert_relative_eq!(
            rosette_max_shear_strain(e0, e45, e90),
            200e-6 * SQRT_2,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            rosette_principal_angle_rectangular(e0, e45, e90),
            FRAC_PI_8,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "doivent être finies")]
    fn non_finite_strain_panics() {
        rosette_principal_strains_rectangular(200e-6, f64::NAN, 0.0);
    }
}
