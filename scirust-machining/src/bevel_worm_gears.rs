//! Engrenages **coniques** et **roue et vis sans fin** — angles de cône, rapport
//! de réduction, angle d'hélice de la vis, rendement et irréversibilité.
//!
//! ```text
//! cône (arbres à 90°)  γ_pignon = atan(Z_p/Z_r)   Γ_roue = atan(Z_r/Z_p)
//! roue et vis          i = Z_roue/z_filets
//! avance de la vis     L = z·p_axial          tan λ = L/(π·d_vis)
//! rendement (menant)   η = tan λ / tan(λ + φ)      (φ angle de frottement)
//! irréversibilité      λ ≤ φ  → système auto-bloquant
//! ```
//!
//! `Z_p`/`Z_r` nombres de dents pignon/roue coniques, `Z_roue` dents de la roue,
//! `z` nombre de filets de la vis, `p_axial` pas axial, `d_vis` diamètre primitif
//! de la vis, `λ` angle d'hélice, `φ` angle de frottement (`tan φ = µ`).
//!
//! **Convention** : angles en rad, longueurs cohérentes. **Limite honnête** :
//! géométrie et rendement **idéaux** ; le rendement roue-vis n'inclut pas les
//! pertes de barbotage ni la variation de `µ` avec la vitesse de glissement. Les
//! arbres coniques sont supposés **orthogonaux**.

use core::f64::consts::PI;

/// Angle de cône primitif du **pignon** conique (arbres à 90°)
/// `γ = atan(Z_pignon/Z_roue)` (rad).
///
/// Panique si `gear_teeth == 0`.
pub fn bevel_pitch_angle_pinion(pinion_teeth: u32, gear_teeth: u32) -> f64 {
    assert!(gear_teeth > 0, "la roue doit avoir au moins une dent");
    (pinion_teeth as f64 / gear_teeth as f64).atan()
}

/// Angle de cône primitif de la **roue** conique `Γ = atan(Z_roue/Z_pignon)` (rad).
///
/// Panique si `pinion_teeth == 0`.
pub fn bevel_pitch_angle_gear(pinion_teeth: u32, gear_teeth: u32) -> f64 {
    assert!(pinion_teeth > 0, "le pignon doit avoir au moins une dent");
    (gear_teeth as f64 / pinion_teeth as f64).atan()
}

/// Rapport de réduction d'une roue et vis `i = Z_roue/z_filets`.
///
/// Panique si `worm_starts == 0`.
pub fn worm_gear_ratio(gear_teeth: u32, worm_starts: u32) -> f64 {
    assert!(worm_starts > 0, "la vis doit avoir au moins un filet");
    gear_teeth as f64 / worm_starts as f64
}

/// Angle d'hélice (avance) de la vis `tan λ = z·p_axial/(π·d_vis)` → `λ` (rad).
///
/// Panique si `worm_pitch_diameter <= 0`.
pub fn worm_lead_angle(worm_starts: u32, axial_pitch: f64, worm_pitch_diameter: f64) -> f64 {
    assert!(
        worm_pitch_diameter > 0.0,
        "le diamètre primitif de la vis doit être strictement positif"
    );
    let lead = worm_starts as f64 * axial_pitch;
    (lead / (PI * worm_pitch_diameter)).atan()
}

/// Rendement d'une roue et vis (vis **menante**) `η = tan λ/tan(λ + φ)`.
///
/// Panique si `λ + φ` atteint `π/2`.
pub fn worm_efficiency(lead_angle: f64, friction_angle: f64) -> f64 {
    let sum = lead_angle + friction_angle;
    assert!(
        sum < core::f64::consts::FRAC_PI_2,
        "λ + φ doit rester inférieur à π/2"
    );
    lead_angle.tan() / sum.tan()
}

/// Vrai si la roue et vis est **auto-bloquante** (irréversible) : `λ ≤ φ`.
pub fn worm_self_locking(lead_angle: f64, friction_angle: f64) -> bool {
    lead_angle <= friction_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bevel_cone_angles_are_complementary() {
        // Arbres à 90° : γ + Γ = π/2.
        let g = bevel_pitch_angle_pinion(20, 40);
        let gg = bevel_pitch_angle_gear(20, 40);
        assert_relative_eq!(g + gg, core::f64::consts::FRAC_PI_2, epsilon = 1e-12);
        // Rapport 1:2 → γ = atan(0,5) ≈ 26,57°.
        assert_relative_eq!(g, 0.5f64.atan(), epsilon = 1e-12);
    }

    #[test]
    fn worm_gives_large_reduction() {
        // Roue 40 dents, vis 1 filet → i = 40 (forte réduction en un étage).
        assert_relative_eq!(worm_gear_ratio(40, 1), 40.0, epsilon = 1e-12);
        assert_relative_eq!(worm_gear_ratio(40, 2), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn lead_angle_from_geometry() {
        // z=2, p=10 mm, d=40 mm → L=20, tanλ = 20/(π·40) → λ ≈ 9,04°.
        let lambda = worm_lead_angle(2, 0.010, 0.040);
        assert_relative_eq!(lambda, (0.020f64 / (PI * 0.040)).atan(), epsilon = 1e-12);
    }

    #[test]
    fn efficiency_and_self_locking() {
        // λ petit (5°), φ=6° (µ≈0,105) → auto-bloquant, rendement < 0,5.
        let lambda = 5.0_f64.to_radians();
        let phi = 6.0_f64.to_radians();
        assert!(worm_self_locking(lambda, phi));
        let eta = worm_efficiency(lambda, phi);
        assert!(eta > 0.0 && eta < 0.5);
        // λ grand (30°), φ=6° → non bloquant, bon rendement (tan30/tan36 ≈ 0,795).
        let lambda2 = 30.0_f64.to_radians();
        assert!(!worm_self_locking(lambda2, phi));
        let eta2 = worm_efficiency(lambda2, phi);
        assert!(eta2 > 0.78 && eta2 < 0.80);
        assert_relative_eq!(eta2, lambda2.tan() / (lambda2 + phi).tan(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "au moins un filet")]
    fn zero_starts_panics() {
        worm_gear_ratio(40, 0);
    }
}
