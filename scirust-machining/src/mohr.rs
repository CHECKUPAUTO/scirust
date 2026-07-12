//! État de contrainte plan — cercle de **Mohr** : contraintes principales,
//! cisaillement maximal, rotation de repère, et critères de limite élastique de
//! **von Mises** et **Tresca**.
//!
//! ```text
//! contraintes principales  σ1,2 = (σx+σy)/2 ± R
//! rayon de Mohr            R = √( ((σx−σy)/2)² + τxy² )   (= τmax dans le plan)
//! direction principale     θp = ½·atan2(2·τxy, σx−σy)
//! rotation de repère       σx' = (σx+σy)/2 + (σx−σy)/2·cos2θ + τxy·sin2θ
//!                          τx'y' = −(σx−σy)/2·sin2θ + τxy·cos2θ
//! von Mises (plan)         σvm = √(σx² − σx·σy + σy² + 3·τxy²)
//! Tresca (plan, σ3=0)      σtr = max(|σ1|, |σ2|, |σ1−σ2|)
//! ```
//!
//! `σx, σy` contraintes normales, `τxy` cisaillement, angles en radians. En
//! contraintes planes la troisième contrainte principale est nulle (`σ3 = 0`),
//! ce dont Tresca tient compte.
//!
//! **Convention** : traction positive ; unités cohérentes de l'appelant (Pa ou
//! MPa). **Limite honnête** : état **plan** de contrainte, matériau élastique
//! isotrope ; pas de triaxialité générale (les fonctions von Mises/Tresca 3D à
//! partir des trois principales sont dans [`crate::mohr::von_mises_principal`]).

/// Contraintes principales `(σ1, σ2)` avec `σ1 ≥ σ2`, d'un état plan
/// `(σx, σy, τxy)`.
pub fn principal_stresses(sx: f64, sy: f64, txy: f64) -> (f64, f64) {
    let mean = (sx + sy) / 2.0;
    let r = mohr_radius(sx, sy, txy);
    (mean + r, mean - r)
}

/// Rayon du cercle de Mohr `R = √(((σx−σy)/2)² + τxy²)`, égal au cisaillement
/// maximal **dans le plan**.
pub fn mohr_radius(sx: f64, sy: f64, txy: f64) -> f64 {
    let d = (sx - sy) / 2.0;
    (d * d + txy * txy).sqrt()
}

/// Cisaillement maximal dans le plan `τmax = R` (alias de [`mohr_radius`]).
pub fn max_in_plane_shear(sx: f64, sy: f64, txy: f64) -> f64 {
    mohr_radius(sx, sy, txy)
}

/// Angle de la direction principale `θp = ½·atan2(2·τxy, σx−σy)` (rad).
pub fn principal_angle_rad(sx: f64, sy: f64, txy: f64) -> f64 {
    0.5 * (2.0 * txy).atan2(sx - sy)
}

/// Contrainte normale sur une facette tournée de `θ` :
/// `σx' = (σx+σy)/2 + (σx−σy)/2·cos2θ + τxy·sin2θ`.
pub fn normal_stress_rotated(sx: f64, sy: f64, txy: f64, theta_rad: f64) -> f64 {
    let mean = (sx + sy) / 2.0;
    let d = (sx - sy) / 2.0;
    let two = 2.0 * theta_rad;
    mean + d * two.cos() + txy * two.sin()
}

/// Cisaillement sur une facette tournée de `θ` :
/// `τx'y' = −(σx−σy)/2·sin2θ + τxy·cos2θ`.
pub fn shear_stress_rotated(sx: f64, sy: f64, txy: f64, theta_rad: f64) -> f64 {
    let d = (sx - sy) / 2.0;
    let two = 2.0 * theta_rad;
    -d * two.sin() + txy * two.cos()
}

/// Contrainte équivalente de **von Mises** en contraintes planes
/// `σvm = √(σx² − σx·σy + σy² + 3·τxy²)`.
pub fn von_mises_plane(sx: f64, sy: f64, txy: f64) -> f64 {
    (sx * sx - sx * sy + sy * sy + 3.0 * txy * txy).sqrt()
}

/// Contrainte équivalente de **von Mises** à partir des trois contraintes
/// principales `σvm = √(½·[(σ1−σ2)² + (σ2−σ3)² + (σ3−σ1)²])`.
pub fn von_mises_principal(s1: f64, s2: f64, s3: f64) -> f64 {
    let a = s1 - s2;
    let b = s2 - s3;
    let c = s3 - s1;
    (0.5 * (a * a + b * b + c * c)).sqrt()
}

/// Contrainte équivalente de **Tresca** en contraintes planes (`σ3 = 0`)
/// `σtr = max(|σ1|, |σ2|, |σ1−σ2|)`.
pub fn tresca_plane(sx: f64, sy: f64, txy: f64) -> f64 {
    let (s1, s2) = principal_stresses(sx, sy, txy);
    s1.abs().max(s2.abs()).max((s1 - s2).abs())
}

/// Coefficient de sécurité `n = σe / σeq` face à une contrainte équivalente.
///
/// Panique si `equivalent_stress <= 0`.
pub fn safety_factor(yield_stress: f64, equivalent_stress: f64) -> f64 {
    assert!(
        equivalent_stress > 0.0,
        "la contrainte équivalente doit être strictement positive"
    );
    yield_stress / equivalent_stress
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pure_shear_principal_stresses() {
        // Cisaillement pur τ=50 : σ1=+50, σ2=−50, à 45°.
        let (s1, s2) = principal_stresses(0.0, 0.0, 50.0);
        assert_relative_eq!(s1, 50.0, epsilon = 1e-12);
        assert_relative_eq!(s2, -50.0, epsilon = 1e-12);
        assert_relative_eq!(
            principal_angle_rad(0.0, 0.0, 50.0),
            core::f64::consts::FRAC_PI_4,
            epsilon = 1e-12
        );
    }

    #[test]
    fn uniaxial_state_is_recovered_by_principals() {
        // σx=100, σy=0, τ=0 → σ1=100, σ2=0, τmax=50.
        let (s1, s2) = principal_stresses(100.0, 0.0, 0.0);
        assert_relative_eq!(s1, 100.0, epsilon = 1e-12);
        assert_relative_eq!(s2, 0.0, epsilon = 1e-12);
        assert_relative_eq!(max_in_plane_shear(100.0, 0.0, 0.0), 50.0, epsilon = 1e-12);
    }

    #[test]
    fn rotation_at_principal_angle_zeroes_shear() {
        // En tournant du repère à θp, le cisaillement s'annule et σx' devient σ1.
        let (sx, sy, txy) = (80.0, 20.0, 30.0);
        let tp = principal_angle_rad(sx, sy, txy);
        assert_relative_eq!(shear_stress_rotated(sx, sy, txy, tp), 0.0, epsilon = 1e-10);
        let (s1, _) = principal_stresses(sx, sy, txy);
        assert_relative_eq!(normal_stress_rotated(sx, sy, txy, tp), s1, epsilon = 1e-10);
    }

    #[test]
    fn von_mises_pure_shear_relation() {
        // Cisaillement pur : σvm = √3·τ ; Tresca = 2·τ.
        let txy = 100.0;
        assert_relative_eq!(
            von_mises_plane(0.0, 0.0, txy),
            (3.0f64).sqrt() * txy,
            epsilon = 1e-9
        );
        assert_relative_eq!(tresca_plane(0.0, 0.0, txy), 2.0 * txy, epsilon = 1e-9);
    }

    #[test]
    fn von_mises_plane_and_principal_agree() {
        // Les deux expressions de von Mises doivent coïncider (σ3=0).
        let (sx, sy, txy) = (120.0, 40.0, 50.0);
        let (s1, s2) = principal_stresses(sx, sy, txy);
        assert_relative_eq!(
            von_mises_plane(sx, sy, txy),
            von_mises_principal(s1, s2, 0.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn tresca_is_more_conservative_than_von_mises() {
        // À iso-état, Tresca ≥ von Mises (critère plus prudent).
        let (sx, sy, txy) = (150.0, -60.0, 40.0);
        assert!(tresca_plane(sx, sy, txy) >= von_mises_plane(sx, sy, txy));
    }

    #[test]
    fn safety_factor_from_yield() {
        assert_relative_eq!(safety_factor(250.0, 125.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "contrainte équivalente")]
    fn zero_equivalent_stress_panics() {
        safety_factor(250.0, 0.0);
    }
}
