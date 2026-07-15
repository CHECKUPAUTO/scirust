//! Relations entre constantes élastiques d'un matériau **linéaire isotrope**
//! (module de Young `E`, cisaillement `G`, compressibilité `K`, Poisson `ν`,
//! premier coefficient de Lamé `λ`).
//!
//! ```text
//! cisaillement       G = E / (2·(1 + ν))
//! compressibilité    K = E / (3·(1 − 2ν))
//! Young (depuis G)   E = 2·G·(1 + ν)
//! Poisson            ν = E/(2G) − 1
//! Lamé (λ)           λ = E·ν / ((1 + ν)·(1 − 2ν))
//! ```
//!
//! Légende (SI cohérent) :
//! - `E, G, K, λ` : modules élastiques, en pascals (Pa) ;
//! - `ν` : coefficient de Poisson, sans dimension.
//!
//! Domaine de validité physique : `−1 < ν < 0,5` (solide stable ; `ν = 0,5`
//! correspond au cas incompressible, exclu ici car `K → ∞`).
//!
//! **Limite honnête** : ces relations ne valent que pour un matériau
//! **élastique linéaire isotrope**, où deux constantes indépendantes suffisent
//! à décrire tout le comportement. Les modules et le coefficient de Poisson
//! sont **fournis par l'appelant** (mesurés ou tirés d'une fiche matériau) ;
//! aucune valeur matériau, physique ou procédé n'est inventée ici. Ne
//! s'applique ni aux matériaux anisotropes, ni au domaine plastique.

/// Module de cisaillement `G = E/(2·(1 + ν))` (Pa).
///
/// Panique si `poisson_ratio <= −1`.
pub fn elastic_shear_modulus_from_e_nu(youngs_modulus: f64, poisson_ratio: f64) -> f64 {
    assert!(
        poisson_ratio > -1.0,
        "le coefficient de Poisson doit vérifier ν > −1"
    );
    youngs_modulus / (2.0 * (1.0 + poisson_ratio))
}

/// Module de compressibilité `K = E/(3·(1 − 2ν))` (Pa).
///
/// Panique si `poisson_ratio >= 0,5` (matériau incompressible ou instable).
pub fn elastic_bulk_modulus_from_e_nu(youngs_modulus: f64, poisson_ratio: f64) -> f64 {
    assert!(
        poisson_ratio < 0.5,
        "le coefficient de Poisson doit rester strictement inférieur à 0,5"
    );
    youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
}

/// Module de Young déduit de `G` et `ν` : `E = 2·G·(1 + ν)` (Pa).
///
/// Panique si `poisson_ratio <= −1`.
pub fn elastic_youngs_from_g_nu(shear_modulus: f64, poisson_ratio: f64) -> f64 {
    assert!(
        poisson_ratio > -1.0,
        "le coefficient de Poisson doit vérifier ν > −1"
    );
    2.0 * shear_modulus * (1.0 + poisson_ratio)
}

/// Coefficient de Poisson `ν = E/(2G) − 1` (sans dimension).
///
/// Panique si `shear_modulus <= 0`.
pub fn elastic_poisson_from_e_g(youngs_modulus: f64, shear_modulus: f64) -> f64 {
    assert!(
        shear_modulus > 0.0,
        "le module de cisaillement doit être strictement positif"
    );
    youngs_modulus / (2.0 * shear_modulus) - 1.0
}

/// Premier coefficient de Lamé `λ = E·ν/((1 + ν)·(1 − 2ν))` (Pa).
///
/// Panique si `poisson_ratio <= −1` ou `poisson_ratio >= 0,5`.
pub fn elastic_lame_lambda(youngs_modulus: f64, poisson_ratio: f64) -> f64 {
    assert!(
        poisson_ratio > -1.0 && poisson_ratio < 0.5,
        "le coefficient de Poisson doit vérifier −1 < ν < 0,5"
    );
    youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shear_modulus_matches_formula() {
        // E = 200 GPa, ν = 0,3 → G = 200/2,6 ≈ 76,923 GPa.
        let g = elastic_shear_modulus_from_e_nu(200e9, 0.3);
        assert_relative_eq!(g, 200e9 / 2.6, max_relative = 1e-12);
        assert!(g > 76e9 && g < 77e9);
    }

    #[test]
    fn round_trip_e_g_nu() {
        // (E, ν) → G, puis on retrouve E et ν par les relations réciproques.
        let (e, nu) = (200e9, 0.3);
        let g = elastic_shear_modulus_from_e_nu(e, nu);
        assert_relative_eq!(elastic_youngs_from_g_nu(g, nu), e, max_relative = 1e-12);
        assert_relative_eq!(elastic_poisson_from_e_g(e, g), nu, epsilon = 1e-12);
    }

    #[test]
    fn bulk_equals_lambda_plus_two_thirds_g() {
        // Identité isotrope exacte : K = λ + (2/3)·G.
        let (e, nu) = (200e9, 0.3);
        let k = elastic_bulk_modulus_from_e_nu(e, nu);
        let lambda = elastic_lame_lambda(e, nu);
        let g = elastic_shear_modulus_from_e_nu(e, nu);
        assert_relative_eq!(k, lambda + 2.0 / 3.0 * g, max_relative = 1e-12);
    }

    #[test]
    fn bulk_modulus_grows_toward_incompressible() {
        // K croît quand ν → 0,5 (dénominateur 1 − 2ν → 0⁺).
        let k1 = elastic_bulk_modulus_from_e_nu(200e9, 0.3);
        let k2 = elastic_bulk_modulus_from_e_nu(200e9, 0.45);
        assert!(k2 > k1);
    }

    #[test]
    fn lame_lambda_realistic_metal_value() {
        // E = 200 GPa, ν = 0,3 → λ = 60e9/0,52 ≈ 115,385 GPa.
        let lambda = elastic_lame_lambda(200e9, 0.3);
        assert_relative_eq!(lambda, 60e9 / 0.52, max_relative = 1e-12);
        assert!(lambda > 0.0);
    }

    #[test]
    #[should_panic(expected = "inférieur à 0,5")]
    fn bulk_modulus_incompressible_panics() {
        elastic_bulk_modulus_from_e_nu(200e9, 0.5);
    }
}
