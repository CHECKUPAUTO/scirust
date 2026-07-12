//! RDM — **sollicitations composées** : superposition traction/flexion,
//! contrainte équivalente de flexion+torsion et moments idéaux de dimensionnement
//! d'arbre.
//!
//! ```text
//! traction + flexion   σ = F/A ± M·c/I
//! flexion + torsion (surface d'un arbre)
//!   von Mises          σ_eq = √(σ_f² + 3·τ²)
//!   moment idéal (Guest) M_eq = ½·(M + √(M² + T²))
//!   couple idéal (Rankine) T_eq = √(M² + T²)
//! ```
//!
//! `F` effort normal (N), `A` aire (m²), `M` moment fléchissant (N·m), `c`
//! distance à la fibre extrême, `I` moment quadratique, `σ_f` contrainte de
//! flexion, `τ` cisaillement de torsion, `T` couple (N·m). Les moments idéaux
//! ramènent un état combiné à une flexion (ou torsion) équivalente.
//!
//! **Convention** : contraintes cohérentes, traction positive. **Limite
//! honnête** : superposition **élastique linéaire** (petites déformations) ; le
//! signe `±` de la flexion est à choisir selon la fibre. Les contraintes
//! élémentaires proviennent de [`crate::shafts`]/[`crate::beams`].

/// Contrainte combinée traction + flexion `σ = F/A + M·c/I`.
///
/// Panique si `area <= 0` ou `i <= 0`.
pub fn combined_axial_bending(force: f64, area: f64, moment: f64, dist_c: f64, i: f64) -> f64 {
    assert!(area > 0.0 && i > 0.0, "A > 0 et I > 0 requis");
    force / area + moment * dist_c / i
}

/// Contrainte équivalente de **von Mises** flexion + torsion `σ_eq = √(σ_f² + 3τ²)`.
pub fn von_mises_bending_torsion(bending_stress: f64, shear_stress: f64) -> f64 {
    (bending_stress * bending_stress + 3.0 * shear_stress * shear_stress).sqrt()
}

/// Moment idéal de flexion (théorie du cisaillement max / Guest)
/// `M_eq = ½·(M + √(M² + T²))`.
pub fn equivalent_bending_moment(moment: f64, torque: f64) -> f64 {
    0.5 * (moment + (moment * moment + torque * torque).sqrt())
}

/// Couple idéal (théorie de la contrainte normale max / Rankine)
/// `T_eq = √(M² + T²)`.
pub fn equivalent_twisting_moment(moment: f64, torque: f64) -> f64 {
    (moment * moment + torque * torque).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_and_bending_superpose() {
        // F/A = 100e6/... choisir : F=10000, A=1e-4 → 100 MPa ; M·c/I ajouté.
        let s = combined_axial_bending(10_000.0, 1e-4, 1000.0, 0.05, 1e-6);
        assert_relative_eq!(s, 10_000.0 / 1e-4 + 1000.0 * 0.05 / 1e-6, epsilon = 1.0);
    }

    #[test]
    fn pure_bending_von_mises_equals_bending() {
        // τ=0 → σ_eq = σ_f.
        assert_relative_eq!(von_mises_bending_torsion(120e6, 0.0), 120e6, epsilon = 1e-3);
        // τ pur : σ_eq = √3·τ.
        assert_relative_eq!(
            von_mises_bending_torsion(0.0, 50e6),
            (3.0f64).sqrt() * 50e6,
            epsilon = 1e-3
        );
    }

    #[test]
    fn equivalent_moments_from_m_and_t() {
        // M=3, T=4 → √(M²+T²)=5 ; M_eq = ½(3+5)=4.
        assert_relative_eq!(equivalent_twisting_moment(3.0, 4.0), 5.0, epsilon = 1e-12);
        assert_relative_eq!(equivalent_bending_moment(3.0, 4.0), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_bending_equivalents_reduce() {
        // T=0 → T_eq = M et M_eq = M.
        assert_relative_eq!(equivalent_twisting_moment(7.0, 0.0), 7.0, epsilon = 1e-12);
        assert_relative_eq!(equivalent_bending_moment(7.0, 0.0), 7.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_area_panics() {
        combined_axial_bending(10_000.0, 0.0, 1000.0, 0.05, 1e-6);
    }
}
