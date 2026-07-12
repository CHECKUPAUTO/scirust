//! Constantes élastiques d'un matériau **isotrope** — relations entre module de
//! Young `E`, module de cisaillement `G`, module de compressibilité `K`,
//! coefficient de Poisson `ν` et premier coefficient de **Lamé** `λ`.
//!
//! ```text
//! cisaillement      G = E / (2·(1 + ν))
//! compressibilité   K = E / (3·(1 − 2ν))
//! Young (depuis G)  E = 2·G·(1 + ν)
//! Poisson           ν = E/(2G) − 1
//! Lamé              λ = E·ν / ((1 + ν)·(1 − 2ν))
//! ```
//!
//! `E, G, K, λ` en Pa, `ν` sans dimension (`−1 < ν < 0,5` pour un solide stable ;
//! `ν = 0,5` incompressible). Deux constantes indépendantes suffisent à décrire
//! un matériau isotrope ; les autres s'en déduisent.
//!
//! **Convention** : SI cohérent. **Limite honnête** : élasticité **linéaire
//! isotrope** ; ne s'applique ni aux matériaux anisotropes (composites — voir
//! [`crate::composites`]), ni au domaine plastique.

/// Module de cisaillement `G = E/(2·(1 + ν))` (Pa).
///
/// Panique si `ν <= −1`.
pub fn shear_modulus_from_e_nu(youngs_modulus: f64, poisson: f64) -> f64 {
    assert!(poisson > -1.0, "ν doit vérifier ν > −1");
    youngs_modulus / (2.0 * (1.0 + poisson))
}

/// Module de compressibilité `K = E/(3·(1 − 2ν))` (Pa).
///
/// Panique si `ν >= 0,5` (matériau incompressible ou instable).
pub fn bulk_modulus_from_e_nu(youngs_modulus: f64, poisson: f64) -> f64 {
    assert!(poisson < 0.5, "ν doit rester strictement inférieur à 0,5");
    youngs_modulus / (3.0 * (1.0 - 2.0 * poisson))
}

/// Module de Young déduit de `G` et `ν` : `E = 2·G·(1 + ν)` (Pa).
pub fn youngs_modulus_from_g_nu(shear_modulus: f64, poisson: f64) -> f64 {
    2.0 * shear_modulus * (1.0 + poisson)
}

/// Coefficient de Poisson `ν = E/(2G) − 1`.
///
/// Panique si `shear_modulus <= 0`.
pub fn poisson_from_e_g(youngs_modulus: f64, shear_modulus: f64) -> f64 {
    assert!(shear_modulus > 0.0, "G doit être strictement positif");
    youngs_modulus / (2.0 * shear_modulus) - 1.0
}

/// Premier coefficient de Lamé `λ = E·ν/((1 + ν)·(1 − 2ν))` (Pa).
///
/// Panique si `ν <= −1` ou `ν >= 0,5`.
pub fn lame_first_parameter(youngs_modulus: f64, poisson: f64) -> f64 {
    assert!(poisson > -1.0 && poisson < 0.5, "−1 < ν < 0,5 requis");
    youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn steel_shear_modulus() {
        // E=210 GPa, ν=0,3 → G = 210/(2,6) ≈ 80,77 GPa.
        let g = shear_modulus_from_e_nu(210e9, 0.3);
        assert_relative_eq!(g, 210e9 / 2.6, epsilon = 1.0);
        assert!(g > 80e9 && g < 81e9);
    }

    #[test]
    fn round_trip_e_g_nu() {
        // Partant de (E, ν) → G, on retrouve E et ν.
        let (e, nu) = (210e9, 0.3);
        let g = shear_modulus_from_e_nu(e, nu);
        assert_relative_eq!(youngs_modulus_from_g_nu(g, nu), e, max_relative = 1e-12);
        assert_relative_eq!(poisson_from_e_g(e, g), nu, epsilon = 1e-12);
    }

    #[test]
    fn bulk_modulus_diverges_near_incompressible() {
        // K croît quand ν → 0,5.
        let k1 = bulk_modulus_from_e_nu(210e9, 0.3);
        let k2 = bulk_modulus_from_e_nu(210e9, 0.45);
        assert!(k2 > k1);
    }

    #[test]
    fn lame_parameter_positive_for_metals() {
        assert!(lame_first_parameter(210e9, 0.3) > 0.0);
    }

    #[test]
    #[should_panic(expected = "inférieur à 0,5")]
    fn incompressible_bulk_panics() {
        bulk_modulus_from_e_nu(210e9, 0.5);
    }
}
