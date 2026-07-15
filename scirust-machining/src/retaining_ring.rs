//! Anneau élastique (**circlip**) — capacité de charge axiale d'un assemblage
//! par gorge, en régime statique.
//!
//! ```text
//! matage admissible de la gorge   F_g   = π·Dg·hg·sigma_y / S
//! cisaillement de l'anneau        F_s   = π·Dr·tr·tau
//! ```
//!
//! `Dg` diamètre de fond de gorge (m), `hg` profondeur radiale de la gorge (m),
//! `sigma_y` limite d'élasticité du matériau de la gorge (Pa), `S` coefficient de
//! sécurité (sans dimension), `F_g` effort axial admissible avant matage
//! (écrasement plastique) de la lèvre de gorge (N) ; `Dr` diamètre de l'anneau au
//! plan de cisaillement (m), `tr` épaisseur axiale de l'anneau (m), `tau` résistance
//! au cisaillement du matériau de l'anneau (Pa), `F_s` effort axial provoquant le
//! cisaillement de l'anneau (N).
//!
//! **Convention** : SI cohérent — dimensions en m, contraintes en Pa, efforts en N,
//! coefficient de sécurité sans dimension.
//!
//! **Limite honnête** : deux modes de ruine statiques élémentaires sont couverts,
//! le **matage (écrasement) de la gorge** — surface cylindrique développée
//! `π·Dg·hg` supposée uniformément chargée jusqu'à `sigma_y` — et le
//! **cisaillement de l'anneau** — surface cylindrique `π·Dr·tr` supposée
//! uniformément cisaillée. La répartition réelle n'est pas uniforme (ouverture du
//! circlip, portée partielle, congé de gorge), et le voilement de l'anneau, le
//! délogement radial (« dishing »), la fatigue et les chocs ne sont **pas**
//! modélisés. Les **limites matière** (`sigma_y`, `tau`) et le coefficient de
//! sécurité `S` dépendent du matériau, du procédé et des conditions d'emploi :
//! ils sont **fournis par l'appelant** — aucune valeur « par défaut » n'est
//! inventée dans ce module.

use core::f64::consts::PI;

/// Effort axial admissible avant matage de la gorge
/// `F_g = π·Dg·hg·sigma_y / S`.
///
/// `groove_diameter` = `Dg` (m), `groove_depth` = `hg` (m),
/// `groove_yield_strength` = `sigma_y` (Pa), `safety_factor` = `S` (sans
/// dimension) ; renvoie un effort (N).
///
/// Panique si `groove_diameter <= 0`, `groove_depth <= 0`,
/// `groove_yield_strength < 0` ou `safety_factor <= 0`.
pub fn ring_thrust_capacity(
    groove_diameter: f64,
    groove_depth: f64,
    groove_yield_strength: f64,
    safety_factor: f64,
) -> f64 {
    assert!(
        groove_diameter > 0.0
            && groove_depth > 0.0
            && groove_yield_strength >= 0.0
            && safety_factor > 0.0,
        "Dg > 0, hg > 0, sigma_y ≥ 0 et S > 0 requis"
    );
    PI * groove_diameter * groove_depth * groove_yield_strength / safety_factor
}

/// Effort axial provoquant le cisaillement de l'anneau
/// `F_s = π·Dr·tr·tau`.
///
/// `ring_diameter` = `Dr` (m), `ring_thickness` = `tr` (m),
/// `shear_strength` = `tau` (Pa) ; renvoie un effort (N).
///
/// Panique si `ring_diameter <= 0`, `ring_thickness <= 0` ou `shear_strength < 0`.
pub fn ring_shear_capacity(ring_diameter: f64, ring_thickness: f64, shear_strength: f64) -> f64 {
    assert!(
        ring_diameter > 0.0 && ring_thickness > 0.0 && shear_strength >= 0.0,
        "Dr > 0, tr > 0 et tau ≥ 0 requis"
    );
    PI * ring_diameter * ring_thickness * shear_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn thrust_capacity_reciprocity_with_area() {
        // F_g·S/(π·Dg·hg) redonne sigma_y : réciprocité effort ↔ contrainte.
        let (dg, hg, sy, s) = (0.050_f64, 0.0015, 250e6, 2.0);
        let fg = ring_thrust_capacity(dg, hg, sy, s);
        assert_relative_eq!(fg * s / (PI * dg * hg), sy, epsilon = 1.0);
    }

    #[test]
    fn thrust_capacity_inverse_proportional_to_safety_factor() {
        // Doubler le coefficient de sécurité halve l'effort admissible.
        let base = ring_thrust_capacity(0.050, 0.0015, 250e6, 2.0);
        let doubled = ring_thrust_capacity(0.050, 0.0015, 250e6, 4.0);
        assert_relative_eq!(doubled, base / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn shear_capacity_proportional_to_thickness() {
        // F_s ∝ tr : doubler l'épaisseur double l'effort de cisaillement.
        let base = ring_shear_capacity(0.050, 0.0020, 300e6);
        let thick = ring_shear_capacity(0.050, 0.0040, 300e6);
        assert_relative_eq!(thick, 2.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn realistic_circlip() {
        // Gorge Dg = 50 mm, hg = 1,5 mm, sigma_y = 250 MPa, S = 2.
        // Surface développée = π·0,050·0,0015 = 2,3562e-4 m² ; F_g = surf·250e6/2.
        let fg = ring_thrust_capacity(0.050, 0.0015, 250e6, 2.0);
        assert_relative_eq!(fg, PI * 0.050 * 0.0015 * 250e6 / 2.0, epsilon = 1e-6);
        assert_relative_eq!(fg, 29452.4_f64, epsilon = 1.0);
        // Anneau Dr = 50 mm, tr = 2 mm, tau = 300 MPa.
        // F_s = π·0,050·0,002·300e6 = 94247,8 N.
        let fs = ring_shear_capacity(0.050, 0.0020, 300e6);
        assert_relative_eq!(fs, 94247.78_f64, epsilon = 1.0);
    }

    #[test]
    fn zero_strength_gives_zero_capacity() {
        // Cas limite : résistance nulle → capacité nulle pour les deux modes.
        assert_relative_eq!(
            ring_thrust_capacity(0.050, 0.0015, 0.0, 2.0),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            ring_shear_capacity(0.050, 0.0020, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "S > 0")]
    fn zero_safety_factor_panics() {
        ring_thrust_capacity(0.050, 0.0015, 250e6, 0.0);
    }
}
