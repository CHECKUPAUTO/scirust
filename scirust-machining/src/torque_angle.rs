//! Serrage **à l'angle** (méthode tour-d'écrou) : au-delà de l'accostage, la
//! précharge est pilotée par l'angle de rotation via l'avance axiale du filet.
//!
//! ```text
//! avance axiale        s  = (theta/(2·PI))·p
//! raideur combinée     ke = (kb·kc)/(kb + kc)
//! précharge à l'angle  F  = s·ke = (theta/(2·PI))·p·(kb·kc)/(kb + kc)
//! ```
//!
//! `theta` angle de rotation au-delà de l'accostage (rad), `p` pas du filet (m),
//! `s` avance axiale (déplacement axial du filet, m), `kb` raideur du boulon
//! (N/m), `kc` raideur des pièces serrées (N/m), `ke` raideur combinée en série
//! (N/m), `F` précharge (N).
//!
//! **Convention** : SI cohérent (rad, m, N/m, N), avance et précharge positives.
//!
//! **Limite honnête** : méthode **angle** (contrôle de la rotation après
//! l'accostage *snug*) ; au-delà du couple d'accostage la précharge est
//! proportionnelle à l'avance axiale. Les raideurs du boulon `kb` et des pièces
//! serrées `kc` sont **FOURNIES par l'appelant** — ce module n'invente aucune
//! valeur « par défaut » de raideur, de pas ni d'angle d'accostage. Modèle
//! **élastique** : suppose que le boulon et les pièces restent dans leur domaine
//! linéaire (pas de plastification contrôlée, contrairement au serrage
//! angle-au-delà-de-la-limite-élastique).

use core::f64::consts::PI;

/// Avance axiale du filet `s = (theta/(2·PI))·p` (m).
///
/// `turn_angle_rad` = `theta` (rad, au-delà de l'accostage),
/// `thread_pitch` = `p` (m).
///
/// Panique si `turn_angle_rad < 0` ou si `thread_pitch <= 0`.
pub fn torqueangle_axial_advance(turn_angle_rad: f64, thread_pitch: f64) -> f64 {
    assert!(
        turn_angle_rad >= 0.0,
        "l'angle de rotation doit être positif ou nul"
    );
    assert!(
        thread_pitch > 0.0,
        "le pas du filet doit être strictement positif"
    );
    (turn_angle_rad / (2.0 * PI)) * thread_pitch
}

/// Raideur combinée en série `ke = (kb·kc)/(kb + kc)` (N/m).
///
/// `bolt_stiffness` = `kb` (N/m), `joint_stiffness` = `kc` (N/m).
///
/// Panique si `bolt_stiffness <= 0` ou si `joint_stiffness <= 0`.
pub fn torqueangle_combined_stiffness(bolt_stiffness: f64, joint_stiffness: f64) -> f64 {
    assert!(
        bolt_stiffness > 0.0,
        "la raideur du boulon doit être strictement positive"
    );
    assert!(
        joint_stiffness > 0.0,
        "la raideur des pièces serrées doit être strictement positive"
    );
    (bolt_stiffness * joint_stiffness) / (bolt_stiffness + joint_stiffness)
}

/// Précharge à l'angle `F = (theta/(2·PI))·p·(kb·kc)/(kb + kc)` (N).
///
/// `turn_angle_rad` = `theta` (rad, au-delà de l'accostage),
/// `thread_pitch` = `p` (m), `joint_stiffness` = `kc` (N/m),
/// `bolt_stiffness` = `kb` (N/m).
///
/// Panique si `turn_angle_rad < 0`, si `thread_pitch <= 0`, si
/// `joint_stiffness <= 0` ou si `bolt_stiffness <= 0`.
pub fn torqueangle_preload_from_angle(
    turn_angle_rad: f64,
    thread_pitch: f64,
    joint_stiffness: f64,
    bolt_stiffness: f64,
) -> f64 {
    let advance = torqueangle_axial_advance(turn_angle_rad, thread_pitch);
    let combined = torqueangle_combined_stiffness(bolt_stiffness, joint_stiffness);
    advance * combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn full_turn_advances_one_pitch() {
        // Cas limite : un tour complet (2·PI rad) avance exactement d'un pas.
        let pitch = 1.5e-3_f64;
        assert_relative_eq!(
            torqueangle_axial_advance(2.0 * PI, pitch),
            pitch,
            epsilon = 1e-12
        );
    }

    #[test]
    fn advance_scales_linearly_with_angle() {
        // Proportionnalité : doubler l'angle double l'avance axiale (pas fixé).
        let s1 = torqueangle_axial_advance(PI / 2.0, 2.0e-3);
        let s2 = torqueangle_axial_advance(PI, 2.0e-3);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-12);
    }

    #[test]
    fn combined_stiffness_of_equal_springs_is_half() {
        // Deux ressorts identiques k en série → k/2.
        let k = 4.0e8_f64;
        assert_relative_eq!(
            torqueangle_combined_stiffness(k, k),
            k / 2.0,
            epsilon = 1e-3
        );
    }

    #[test]
    fn combined_stiffness_is_symmetric() {
        // Symétrie : ke(kb, kc) = ke(kc, kb).
        let (kb, kc) = (1.0e9_f64, 3.0e9_f64);
        assert_relative_eq!(
            torqueangle_combined_stiffness(kb, kc),
            torqueangle_combined_stiffness(kc, kb),
            epsilon = 1e-3
        );
    }

    #[test]
    fn preload_from_angle_reference_case() {
        // theta=PI/2 (90°), p=2e-3 m → s = 0,25·2e-3 = 5e-4 m.
        // kb=kc=4e8 N/m → ke = 4e8·4e8/8e8 = 2e8 N/m.
        // F = 5e-4·2e8 = 1,0e5 N.
        assert_relative_eq!(
            torqueangle_preload_from_angle(PI / 2.0, 2.0e-3, 4.0e8, 4.0e8),
            1.0e5,
            epsilon = 1e-3
        );
    }

    #[test]
    fn preload_equals_advance_times_combined() {
        // Identité de composition : F = s·ke, avec s et ke calculés séparément.
        let (theta, pitch, kc, kb) = (0.8_f64, 1.75e-3_f64, 2.5e9_f64, 1.0e9_f64);
        let expected =
            torqueangle_axial_advance(theta, pitch) * torqueangle_combined_stiffness(kb, kc);
        assert_relative_eq!(
            torqueangle_preload_from_angle(theta, pitch, kc, kb),
            expected,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "le pas du filet doit être strictement positif")]
    fn zero_pitch_panics() {
        torqueangle_axial_advance(PI, 0.0);
    }
}
