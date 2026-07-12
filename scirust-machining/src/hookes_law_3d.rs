//! Loi de **Hooke généralisée** (3D, isotrope) — déformations à partir d'un état
//! de contrainte triaxial, cisaillement, déformation volumique et pression
//! hydrostatique.
//!
//! ```text
//! déformation normale  εx = (1/E)·[σx − ν·(σy + σz)]   (et permutations)
//! cisaillement         γ = τ/G
//! déformation volumique εv = (1 − 2ν)/E·(σx + σy + σz)
//! pression hydrostatique p = (σx + σy + σz)/3
//! ```
//!
//! `E` module de Young (Pa), `ν` coefficient de Poisson, `G` module de
//! cisaillement (Pa), `σ` contraintes (Pa, traction positive), `ε`/`γ`
//! déformations (sans dimension). L'effet de Poisson couple les trois directions.
//!
//! **Convention** : SI cohérent, traction positive. **Limite honnête** :
//! élasticité **linéaire isotrope** en petites déformations ; les constantes
//! `E`, `ν`, `G` (liées par [`crate::elasticity_relations`]) sont fournies par
//! l'appelant. Pas d'anisotropie ni de plasticité.

/// Déformation normale dans une direction `εx = (1/E)·[σ_axial − ν·(σ_t1 + σ_t2)]`.
///
/// Panique si `youngs_modulus <= 0`.
pub fn axial_strain(
    youngs_modulus: f64,
    poisson: f64,
    sigma_axial: f64,
    sigma_transverse1: f64,
    sigma_transverse2: f64,
) -> f64 {
    assert!(youngs_modulus > 0.0, "E doit être strictement positif");
    (sigma_axial - poisson * (sigma_transverse1 + sigma_transverse2)) / youngs_modulus
}

/// Déformation angulaire de cisaillement `γ = τ/G`.
///
/// Panique si `shear_modulus <= 0`.
pub fn shear_strain(tau: f64, shear_modulus: f64) -> f64 {
    assert!(shear_modulus > 0.0, "G doit être strictement positif");
    tau / shear_modulus
}

/// Déformation volumique `εv = (1 − 2ν)/E·(σx + σy + σz)`.
///
/// Panique si `youngs_modulus <= 0`.
pub fn volumetric_strain(youngs_modulus: f64, poisson: f64, sx: f64, sy: f64, sz: f64) -> f64 {
    assert!(youngs_modulus > 0.0, "E doit être strictement positif");
    (1.0 - 2.0 * poisson) / youngs_modulus * (sx + sy + sz)
}

/// Contrainte (pression) hydrostatique `p = (σx + σy + σz)/3` (Pa).
pub fn hydrostatic_stress(sx: f64, sy: f64, sz: f64) -> f64 {
    (sx + sy + sz) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn uniaxial_reduces_to_sigma_over_e() {
        // σx seul → εx = σx/E ; les directions transverses se contractent de −ν·σx/E.
        let (e, nu, s) = (210e9, 0.3, 100e6);
        assert_relative_eq!(
            axial_strain(e, nu, s, 0.0, 0.0),
            s / e,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            axial_strain(e, nu, 0.0, s, 0.0),
            -nu * s / e,
            max_relative = 1e-12
        );
    }

    #[test]
    fn shear_strain_definition() {
        // τ=50 MPa, G=80 GPa → γ = 6,25e-4.
        assert_relative_eq!(shear_strain(50e6, 80e9), 50e6 / 80e9, epsilon = 1e-15);
    }

    #[test]
    fn hydrostatic_state_and_volumetric_strain() {
        // État hydrostatique σx=σy=σz=σ → p=σ, εv = 3σ(1−2ν)/E.
        let (e, nu, s) = (210e9, 0.3, 60e6);
        assert_relative_eq!(hydrostatic_stress(s, s, s), s, epsilon = 1e-6);
        assert_relative_eq!(
            volumetric_strain(e, nu, s, s, s),
            3.0 * s * (1.0 - 2.0 * nu) / e,
            max_relative = 1e-12
        );
    }

    #[test]
    fn incompressible_has_no_volume_change() {
        // ν=0,5 → εv = 0 quel que soit l'état de contrainte.
        assert_relative_eq!(
            volumetric_strain(210e9, 0.5, 100e6, 50e6, -30e6),
            0.0,
            epsilon = 1e-20
        );
    }

    #[test]
    #[should_panic(expected = "E doit être")]
    fn zero_modulus_panics() {
        axial_strain(0.0, 0.3, 100e6, 0.0, 0.0);
    }
}
