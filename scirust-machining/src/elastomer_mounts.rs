//! Plots (supports) **élastomères** — raideurs en compression et en cisaillement
//! d'un bloc de caoutchouc, via le **facteur de forme**.
//!
//! ```text
//! facteur de forme   S = A_chargée / A_libre (de gonflement)
//! module apparent    Ea = E0·(1 + 2·S²)
//! raideur compression kc = Ea·A/t
//! raideur cisaillement kτ = G·A/t
//! flèche             x = F/k
//! ```
//!
//! `A` aire chargée (m²), `t` épaisseur du bloc (m), `S` facteur de forme
//! (rapport surface chargée / surface libre de gonflement), `E0` module de Young
//! du caoutchouc (Pa), `Ea` module apparent en compression (rigidifié par le
//! confinement), `G` module de cisaillement (Pa). Un bloc large et mince (S
//! grand) est bien plus raide en compression qu'en cisaillement.
//!
//! **Convention** : SI cohérent. **Limite honnête** : élasticité **linéaire**
//! (petites déformations) d'un bloc parfaitement collé ; la relation
//! `Ea = E0(1+2S²)` est une **corrélation** (caoutchouc peu compressible) dont le
//! coefficient dépend du compound — l'appelant l'ajuste. Pas d'hyperélasticité,
//! de fluage ni d'échauffement.

/// Facteur de forme `S = A_chargée / A_libre`.
///
/// Panique si `bulge_area <= 0`.
pub fn shape_factor(loaded_area: f64, bulge_area: f64) -> f64 {
    assert!(
        bulge_area > 0.0,
        "l'aire libre de gonflement doit être strictement positive"
    );
    loaded_area / bulge_area
}

/// Module apparent en compression `Ea = E0·(1 + 2·S²)` (Pa).
pub fn apparent_compression_modulus(youngs_modulus: f64, shape_factor: f64) -> f64 {
    youngs_modulus * (1.0 + 2.0 * shape_factor * shape_factor)
}

/// Raideur en compression `kc = Ea·A/t` (N/m).
///
/// Panique si `thickness <= 0`.
pub fn compression_stiffness(apparent_modulus: f64, area: f64, thickness: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    apparent_modulus * area / thickness
}

/// Raideur en cisaillement `kτ = G·A/t` (N/m).
///
/// Panique si `thickness <= 0`.
pub fn shear_stiffness(shear_modulus: f64, area: f64, thickness: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    shear_modulus * area / thickness
}

/// Flèche sous effort `x = F/k` (m).
///
/// Panique si `stiffness <= 0`.
pub fn deflection(force: f64, stiffness: f64) -> f64 {
    assert!(stiffness > 0.0, "la raideur doit être strictement positive");
    force / stiffness
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shape_factor_of_a_disc() {
        // Disque Ø50 mm, t=10 mm : A_chargée=π·0,025², A_libre=π·0,05·0,01 → S=D/(4t)=1,25.
        use core::f64::consts::PI;
        let loaded = PI * 0.025 * 0.025;
        let bulge = PI * 0.050 * 0.010;
        assert_relative_eq!(shape_factor(loaded, bulge), 1.25, epsilon = 1e-9);
    }

    #[test]
    fn confinement_stiffens_compression() {
        // Ea = E0(1+2S²) : à S=1,25 → facteur 1+3,125 = 4,125.
        let ea = apparent_compression_modulus(4e6, 1.25);
        assert_relative_eq!(ea, 4e6 * 4.125, epsilon = 1e-3);
        assert!(ea > 4e6);
    }

    #[test]
    fn compression_much_stiffer_than_shear() {
        // Même bloc : kc (via Ea) ≫ kτ (via G, faible pour le caoutchouc).
        let (area, t) = (2e-3, 0.010);
        let ea = apparent_compression_modulus(4e6, 1.25);
        let kc = compression_stiffness(ea, area, t);
        let ks = shear_stiffness(1.3e6, area, t);
        assert!(kc > ks);
    }

    #[test]
    fn deflection_is_force_over_stiffness() {
        // k=1e6 N/m, F=500 N → x = 0,5 mm.
        assert_relative_eq!(deflection(500.0, 1e6), 5e-4, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "aire libre")]
    fn zero_bulge_area_panics() {
        shape_factor(1e-3, 0.0);
    }
}
