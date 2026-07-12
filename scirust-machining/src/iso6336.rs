//! Résistance des engrenages au flanc (pitting) — pression de contact selon
//! **ISO 6336-2**, en complément du modèle de flexion de Lewis de [`crate::gears`].
//!
//! La contrainte de contact nominale (au point primitif) s'écrit :
//!
//! ```text
//! σH0 = ZE·ZH·Zε·Zβ · √( Ft/(d1·b) · (u+1)/u )
//! ```
//!
//! et la contrainte de contact de service applique les facteurs de charge :
//!
//! ```text
//! σH = σH0 · √( KA·KV·KHβ·KHα )
//! ```
//!
//! avec :
//! - `ZE` facteur d'élasticité (√MPa), dérivé du couple de matériaux ;
//! - `ZH` facteur de zone (géométrie de la denture) ;
//! - `Zε` facteur de conduite, `Zβ` facteur d'angle d'hélice ;
//! - `Ft` effort tangentiel (N), `d1` diamètre primitif du pignon (mm),
//!   `b` largeur de denture (mm), `u = z2/z1` rapport ;
//! - `KA` facteur d'application, `KV` facteur dynamique, `KHβ`/`KHα` facteurs de
//!   répartition longitudinale/transversale.
//!
//! Le facteur d'élasticité relie ce module à [`crate::hertz`] :
//! `ZE = √(E\* / π)`, où `E\*` est le module effectif de Hertz. Pour un couple
//! acier/acier (E = 206 000 MPa, ν = 0,3), `ZE ≈ 189,8 √MPa`.
//!
//! **Limite honnête** : ISO 6336 est une norme volumineuse ; ce module en
//! implémente le **cœur** (contrainte de contact et coefficient de sécurité au
//! pitting `SH = σHP/σH`). Les nombreux facteurs (`ZH`, `Zε`, `Zβ`, `KA`, `KV`,
//! `KHβ`, `KHα`) et la contrainte admissible `σHP` (facteurs de durée, de
//! lubrifiant, de rugosité, de dureté…) sont des données que l'appelant calcule
//! d'après la norme ou son bureau d'études — la crate assemble la contrainte à
//! partir d'eux.

use crate::hertz::effective_modulus;
use core::f64::consts::PI;

/// Facteur de zone standard `ZH ≈ 2,495` d'une denture droite normalisée à
/// angle de pression 20° (sans déport) — valeur de référence.
pub const ZH_STANDARD_SPUR_20: f64 = 2.495;

/// Facteur d'élasticité `ZE = √(E\* / π)` (√MPa) du couple de matériaux, où
/// `E\*` est le module effectif de Hertz. Modules de Young en MPa, coefficients
/// de Poisson sans dimension.
pub fn elasticity_factor_ze(e1_mpa: f64, nu1: f64, e2_mpa: f64, nu2: f64) -> f64 {
    (effective_modulus(e1_mpa, nu1, e2_mpa, nu2) / PI).sqrt()
}

/// Contrainte de contact **nominale** `σH0` (MPa) :
/// `σH0 = ZE·ZH·Zε·Zβ·√(Ft/(d1·b)·(u+1)/u)`.
///
/// `ft` (N), `d1`, `b` (mm), `u = z2/z1`, facteurs `ze` (√MPa), `zh`, `z_eps`,
/// `z_beta` (sans dimension). Panique si `d1`, `b` ou `u` ≤ 0.
#[allow(clippy::too_many_arguments)]
pub fn nominal_contact_stress(
    ft_n: f64,
    d1_mm: f64,
    b_mm: f64,
    u: f64,
    ze: f64,
    zh: f64,
    z_eps: f64,
    z_beta: f64,
) -> f64 {
    assert!(
        d1_mm > 0.0 && b_mm > 0.0 && u > 0.0,
        "diamètre, largeur et rapport doivent être strictement positifs"
    );
    let load = ft_n / (d1_mm * b_mm) * (u + 1.0) / u;
    ze * zh * z_eps * z_beta * load.sqrt()
}

/// Contrainte de contact **de service** `σH = σH0·√(KA·KV·KHβ·KHα)` (MPa).
///
/// Panique si un facteur de charge est ≤ 0.
pub fn contact_stress(sigma_h0_mpa: f64, ka: f64, kv: f64, k_hbeta: f64, k_halpha: f64) -> f64 {
    assert!(
        ka > 0.0 && kv > 0.0 && k_hbeta > 0.0 && k_halpha > 0.0,
        "les facteurs de charge doivent être strictement positifs"
    );
    sigma_h0_mpa * (ka * kv * k_hbeta * k_halpha).sqrt()
}

/// Coefficient de sécurité au pitting `SH = σHP / σH` : contrainte admissible
/// de flanc `sigma_hp` (MPa) sur contrainte de service `sigma_h` (MPa).
///
/// Panique si `sigma_h <= 0`.
pub fn safety_factor_pitting(sigma_hp_mpa: f64, sigma_h_mpa: f64) -> f64 {
    assert!(
        sigma_h_mpa > 0.0,
        "la contrainte de service doit être strictement positive"
    );
    sigma_hp_mpa / sigma_h_mpa
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn steel_elasticity_factor_is_about_190() {
        // Acier/acier, E=206000 MPa, ν=0,3 → ZE ≈ 189,8 √MPa.
        assert_relative_eq!(
            elasticity_factor_ze(206_000.0, 0.3, 206_000.0, 0.3),
            189.8,
            epsilon = 0.2
        );
    }

    #[test]
    fn nominal_contact_stress_matches_the_formula() {
        // Ft=10000, d1=100, b=40, u=3, ZE=189,8, ZH=2,495, Zε=0,9, Zβ=1.
        let s = nominal_contact_stress(10_000.0, 100.0, 40.0, 3.0, 189.8, 2.495, 0.9, 1.0);
        let load: f64 = 10_000.0 / (100.0 * 40.0) * (4.0 / 3.0);
        let expected = 189.8 * 2.495 * 0.9 * 1.0 * load.sqrt();
        assert_relative_eq!(s, expected, epsilon = 1e-9);
        // ordre de grandeur attendu ≈ 778 MPa.
        assert_relative_eq!(s, 778.0, epsilon = 2.0);
    }

    #[test]
    fn load_factors_raise_the_service_stress() {
        let s0 = 700.0;
        // tous facteurs = 1 → σH = σH0.
        assert_relative_eq!(contact_stress(s0, 1.0, 1.0, 1.0, 1.0), s0, epsilon = 1e-9);
        // facteurs > 1 → σH > σH0.
        assert!(contact_stress(s0, 1.25, 1.1, 1.2, 1.0) > s0);
    }

    #[test]
    fn safety_factor_is_allowable_over_service() {
        // σHP=1200, σH=800 → SH = 1,5.
        assert_relative_eq!(safety_factor_pitting(1200.0, 800.0), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn higher_ratio_lowers_the_contact_stress_factor() {
        // (u+1)/u décroît quand u croît → σH0 plus faible à effort égal.
        let low_u = nominal_contact_stress(10_000.0, 100.0, 40.0, 2.0, 189.8, 2.495, 1.0, 1.0);
        let high_u = nominal_contact_stress(10_000.0, 100.0, 40.0, 6.0, 189.8, 2.495, 1.0, 1.0);
        assert!(high_u < low_u);
    }

    #[test]
    #[should_panic(expected = "strictement positifs")]
    fn zero_face_width_panics() {
        nominal_contact_stress(10_000.0, 100.0, 0.0, 3.0, 189.8, 2.495, 0.9, 1.0);
    }
}
