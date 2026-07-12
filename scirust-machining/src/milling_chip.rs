//! Usinage — **fraisage** : géométrie du copeau (avance par dent, angle
//! d'engagement, dents en prise, épaisseur de copeau instantanée).
//!
//! ```text
//! avance par dent    fz = vf/(N·z)
//! angle d'engagement φs = arccos(1 − 2·ae/D)
//! dents en prise      z_c = z·φs/(2π)
//! épaisseur de copeau h(θ) = fz·sin θ        (fraisage périphérique)
//! ```
//!
//! `vf` vitesse d'avance (mm/min), `N` fréquence de rotation (tr/min), `z` nombre
//! de dents, `fz` avance par dent (mm), `ae` engagement radial (mm), `D` diamètre
//! de la fraise (mm), `φs` arc de contact (rad), `θ` position angulaire de la
//! dent, `h` épaisseur de copeau non déformée. L'épaisseur varie de zéro à un
//! maximum au cours de l'engagement.
//!
//! **Convention** : unités de fiche outil (mm, tr/min, mm/min) ; angles en rad.
//! **Limite honnête** : fraisage **périphérique** idéalisé (trajectoire circulaire
//! approchée par l'avance) ; ne traite pas l'effet cycloïdal exact, ni le fraisage
//! en bout. Le débit de copeaux est dans [`crate::kinematics`].

use core::f64::consts::PI;

/// Avance par dent `fz = vf/(N·z)` (mm).
///
/// Panique si `rpm·teeth <= 0`.
pub fn feed_per_tooth(feed_velocity: f64, rpm: f64, teeth: u32) -> f64 {
    let denom = rpm * teeth as f64;
    assert!(denom > 0.0, "N·z doit être strictement positif");
    feed_velocity / denom
}

/// Arc de contact `φs = arccos(1 − 2·ae/D)` (rad).
///
/// Panique si `ae/D` sort de `[0, 1]`.
pub fn engagement_angle(radial_depth: f64, diameter: f64) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    let ratio = radial_depth / diameter;
    assert!(
        (0.0..=1.0).contains(&ratio),
        "l'engagement radial doit vérifier 0 ≤ ae ≤ D"
    );
    (1.0 - 2.0 * ratio).acos()
}

/// Nombre moyen de dents en prise `z_c = z·φs/(2π)`.
pub fn teeth_in_cut(engagement_angle: f64, teeth: u32) -> f64 {
    teeth as f64 * engagement_angle / (2.0 * PI)
}

/// Épaisseur de copeau non déformée `h(θ) = fz·sin θ` (mm).
pub fn chip_thickness_at_angle(feed_per_tooth: f64, angle: f64) -> f64 {
    feed_per_tooth * angle.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn feed_per_tooth_definition() {
        // vf=300 mm/min, N=1000 tr/min, z=3 → fz = 0,1 mm.
        assert_relative_eq!(feed_per_tooth(300.0, 1000.0, 3), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn slotting_engages_half_turn() {
        // Rainurage (ae = D) → φs = arccos(−1) = π (180°).
        assert_relative_eq!(engagement_angle(50.0, 50.0), PI, epsilon = 1e-12);
        // Engagement à mi-diamètre (ae = D/2) → arccos(0) = π/2.
        assert_relative_eq!(engagement_angle(25.0, 50.0), PI / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn teeth_in_cut_scales_with_engagement() {
        // Rainurage z=4, φs=π → z_c = 4·π/2π = 2 dents en prise.
        assert_relative_eq!(teeth_in_cut(PI, 4), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn chip_thickness_peaks_at_ninety_degrees() {
        // h(θ) = fz·sinθ : maximal à θ=90°, nul à θ=0.
        assert_relative_eq!(chip_thickness_at_angle(0.1, PI / 2.0), 0.1, epsilon = 1e-12);
        assert_relative_eq!(chip_thickness_at_angle(0.1, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "engagement radial")]
    fn excessive_radial_depth_panics() {
        engagement_angle(60.0, 50.0);
    }
}
