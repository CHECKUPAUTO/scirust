//! Arbres de transmission — torsion et flexion des sections circulaires
//! (pleines ou creuses) : modules de section, contraintes de cisaillement et
//! de flexion, contrainte équivalente de von Mises sous charge combinée, et
//! angle de torsion.
//!
//! Pour une section circulaire pleine de diamètre `d` (mm) :
//!
//! ```text
//! module de torsion   Wt = π·d³/16          contrainte τ = T / Wt
//! module de flexion   W  = π·d³/32          contrainte σ = M / W
//! inertie polaire     J  = π·d⁴/32          torsion  θ = T·L / (G·J)
//! ```
//!
//! Sous **flexion + torsion** combinées, la contrainte équivalente de von Mises
//! sur la fibre extérieure vaut :
//!
//! ```text
//! σ_eq = √(σ² + 3·τ²) = (32/(π·d³)) · √(M² + ¾·T²)
//! ```
//!
//! **Convention d'unités** : couple `T` et moment `M` en **N·m**, diamètres et
//! longueurs en **mm**, module de cisaillement `G` en **MPa**, contraintes en
//! **MPa**, angle de torsion en **degrés**. Les conversions N·m → N·mm sont
//! internes.
//!
//! **Limite honnête** : résistance des matériaux élastique linéaire, section
//! circulaire, contraintes nominales sans concentration. Les coefficients de
//! concentration de contrainte (épaulements, rainures, `Kt`/`Kf`), le flambage,
//! les vitesses critiques (flexion/torsion) et la fatigue multiaxiale sont hors
//! périmètre — à traiter avec les données de géométrie locale et de matériau.

use core::f64::consts::PI;

/// Module de résistance à la torsion `Wt = π·d³/16` (mm³) d'une section pleine.
pub fn polar_section_modulus_solid(diameter_mm: f64) -> f64 {
    PI * diameter_mm.powi(3) / 16.0
}

/// Module de résistance à la flexion `W = π·d³/32` (mm³) d'une section pleine.
pub fn section_modulus_solid(diameter_mm: f64) -> f64 {
    PI * diameter_mm.powi(3) / 32.0
}

/// Module de torsion d'une section creuse `Wt = π·(D⁴−d⁴)/(16·D)` (mm³),
/// diamètres extérieur `outer` et intérieur `inner` (mm).
///
/// Panique si `outer <= inner` ou `outer <= 0`.
pub fn polar_section_modulus_hollow(outer_mm: f64, inner_mm: f64) -> f64 {
    assert!(
        outer_mm > inner_mm && outer_mm > 0.0,
        "le diamètre extérieur doit dépasser l'intérieur et être positif"
    );
    PI * (outer_mm.powi(4) - inner_mm.powi(4)) / (16.0 * outer_mm)
}

/// Module de flexion d'une section creuse `W = π·(D⁴−d⁴)/(32·D)` (mm³).
///
/// Panique si `outer <= inner` ou `outer <= 0`.
pub fn section_modulus_hollow(outer_mm: f64, inner_mm: f64) -> f64 {
    assert!(
        outer_mm > inner_mm && outer_mm > 0.0,
        "le diamètre extérieur doit dépasser l'intérieur et être positif"
    );
    PI * (outer_mm.powi(4) - inner_mm.powi(4)) / (32.0 * outer_mm)
}

/// Contrainte de cisaillement de torsion `τ = T / Wt` (MPa), couple `torque`
/// (N·m) et module de torsion `wt` (mm³).
///
/// Panique si `wt <= 0`.
pub fn torsional_shear_stress(torque_nm: f64, wt_mm3: f64) -> f64 {
    assert!(
        wt_mm3 > 0.0,
        "le module de torsion doit être strictement positif"
    );
    torque_nm * 1000.0 / wt_mm3
}

/// Contrainte de flexion `σ = M / W` (MPa), moment `moment` (N·m) et module de
/// flexion `w` (mm³).
///
/// Panique si `w <= 0`.
pub fn bending_stress(moment_nm: f64, w_mm3: f64) -> f64 {
    assert!(
        w_mm3 > 0.0,
        "le module de flexion doit être strictement positif"
    );
    moment_nm * 1000.0 / w_mm3
}

/// Contrainte équivalente de von Mises `σ_eq = √(σ² + 3·τ²)` (MPa) sur la fibre
/// extérieure d'un arbre plein sous flexion `moment` (N·m) et torsion `torque`
/// (N·m) : `σ_eq = (32000/(π·d³))·√(M² + ¾·T²)`.
///
/// Panique si `diameter <= 0`.
pub fn von_mises_solid(moment_nm: f64, torque_nm: f64, diameter_mm: f64) -> f64 {
    assert!(
        diameter_mm > 0.0,
        "le diamètre doit être strictement positif"
    );
    let coeff = 32.0 * 1000.0 / (PI * diameter_mm.powi(3));
    coeff * (moment_nm * moment_nm + 0.75 * torque_nm * torque_nm).sqrt()
}

/// Angle de torsion `θ = T·L / (G·J)` (degrés) d'un arbre plein, couple
/// `torque` (N·m), longueur `length` (mm), diamètre `diameter` (mm) et module
/// de cisaillement `g` (MPa) ; `J = π·d⁴/32`.
///
/// Panique si `diameter <= 0` ou `g <= 0`.
pub fn angle_of_twist_deg(torque_nm: f64, length_mm: f64, diameter_mm: f64, g_mpa: f64) -> f64 {
    assert!(
        diameter_mm > 0.0 && g_mpa > 0.0,
        "diamètre et module de cisaillement doivent être strictement positifs"
    );
    let j = PI * diameter_mm.powi(4) / 32.0;
    let theta_rad = torque_nm * 1000.0 * length_mm / (g_mpa * j);
    theta_rad.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn solid_section_moduli_follow_the_formulas() {
        // d=40 : Wt = π·64000/16 = 4000π ≈ 12566,4 ; W = 2000π ≈ 6283,2.
        assert_relative_eq!(
            polar_section_modulus_solid(40.0),
            4000.0 * PI,
            epsilon = 1e-6
        );
        assert_relative_eq!(section_modulus_solid(40.0), 2000.0 * PI, epsilon = 1e-6);
    }

    #[test]
    fn torsion_stress_of_a_40mm_shaft() {
        // T=1000 N·m sur Ø40 → τ = 1e6/12566,4 ≈ 79,58 MPa.
        let wt = polar_section_modulus_solid(40.0);
        assert_relative_eq!(torsional_shear_stress(1000.0, wt), 79.577, epsilon = 1e-2);
    }

    #[test]
    fn bending_stress_of_a_40mm_shaft() {
        // M=500 N·m sur Ø40 → σ = 5e5/6283,2 ≈ 79,58 MPa.
        let w = section_modulus_solid(40.0);
        assert_relative_eq!(bending_stress(500.0, w), 79.577, epsilon = 1e-2);
    }

    #[test]
    fn von_mises_combines_bending_and_torsion() {
        // M=500, T=1000, Ø40 : σ=τ≈79,58 → σ_eq = √(σ²+3τ²) = 2σ ≈ 159,15 MPa.
        let sigma = von_mises_solid(500.0, 1000.0, 40.0);
        assert_relative_eq!(sigma, 159.155, epsilon = 1e-2);
        // cohérence avec √(σ²+3τ²) reconstruit.
        let s = bending_stress(500.0, section_modulus_solid(40.0));
        let t = torsional_shear_stress(1000.0, polar_section_modulus_solid(40.0));
        assert_relative_eq!(sigma, (s * s + 3.0 * t * t).sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn hollow_section_is_weaker_than_solid_of_same_outer_diameter() {
        // Le perçage central retire de la matière → module plus faible.
        assert!(section_modulus_hollow(40.0, 20.0) < section_modulus_solid(40.0));
    }

    #[test]
    fn angle_of_twist_of_a_steel_shaft() {
        // T=1000, L=500, Ø40, G=81500 → θ ≈ 1,398°.
        assert_relative_eq!(
            angle_of_twist_deg(1000.0, 500.0, 40.0, 81_500.0),
            1.398,
            epsilon = 1e-2
        );
    }

    #[test]
    #[should_panic(expected = "extérieur")]
    fn hollow_requires_outer_bigger_than_inner() {
        section_modulus_hollow(20.0, 30.0);
    }
}
