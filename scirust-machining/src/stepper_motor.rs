//! **Moteur pas à pas** — résolution angulaire, cadence d'impulsions, vitesse et
//! résolution linéaire sur un axe entraîné par vis.
//!
//! ```text
//! pas par tour     s = 360/θ                            (θ angle de pas, °)
//! angle de pas     θ = 360/s
//! vitesse          N = 60·f/s                           (f cadence d'impulsions, Hz)
//! cadence          f = N·s/60
//! résolution lin.  r = p·θ/360 = p/s                    (p pas de vis)
//! ```
//!
//! `θ` angle de pas mécanique (°, ex. 1,8° → 200 pas/tour), `s` pas par tour
//! (éventuellement multiplié par le **micro-pas**), `f` fréquence des impulsions
//! (Hz = pas/s), `N` vitesse de rotation (tr/min), `p` pas de la vis (m), `r`
//! résolution linéaire (m/pas).
//!
//! **Convention** : angles en degrés, cadence en Hz, vitesse en tr/min.
//! **Limite honnête** : relations **cinématiques** (comptage de pas) valables
//! **sans perte de pas** ; le couple disponible chute avec la fréquence
//! (caractéristique couple-vitesse), non modélisée ici — cette courbe est une
//! donnée du moteur fournie par l'appelant. Voir [`crate::ball_screw`] pour la
//! conversion en course.

/// Pas par tour depuis l'angle de pas `s = 360/θ`.
///
/// Panique si `step_angle_deg <= 0`.
pub fn steps_per_revolution(step_angle_deg: f64) -> f64 {
    assert!(
        step_angle_deg > 0.0,
        "l'angle de pas doit être strictement positif"
    );
    360.0 / step_angle_deg
}

/// Angle de pas depuis le nombre de pas par tour `θ = 360/s`.
///
/// Panique si `steps_per_revolution <= 0`.
pub fn step_angle_deg(steps_per_revolution: f64) -> f64 {
    assert!(
        steps_per_revolution > 0.0,
        "le nombre de pas par tour doit être strictement positif"
    );
    360.0 / steps_per_revolution
}

/// Vitesse de rotation `N = 60·f/s` (tr/min) depuis la cadence d'impulsions.
///
/// Panique si `steps_per_revolution <= 0` ou `pulse_frequency_hz < 0`.
pub fn speed_from_pulse_rate(pulse_frequency_hz: f64, steps_per_revolution: f64) -> f64 {
    assert!(
        steps_per_revolution > 0.0 && pulse_frequency_hz >= 0.0,
        "s > 0 et f ≥ 0 requis"
    );
    60.0 * pulse_frequency_hz / steps_per_revolution
}

/// Cadence d'impulsions `f = N·s/60` (Hz) pour une vitesse cible.
///
/// Panique si `steps_per_revolution <= 0` ou `speed_rpm < 0`.
pub fn pulse_rate_for_speed(speed_rpm: f64, steps_per_revolution: f64) -> f64 {
    assert!(
        steps_per_revolution > 0.0 && speed_rpm >= 0.0,
        "s > 0 et N ≥ 0 requis"
    );
    speed_rpm * steps_per_revolution / 60.0
}

/// Résolution linéaire `r = p·θ/360` (m/pas) sur un axe à vis.
///
/// Panique si `step_angle_deg <= 0` ou `lead <= 0`.
pub fn linear_resolution(step_angle_deg: f64, lead: f64) -> f64 {
    assert!(
        step_angle_deg > 0.0 && lead > 0.0,
        "angle de pas et pas de vis > 0 requis"
    );
    lead * step_angle_deg / 360.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn standard_18_degree_motor_has_200_steps() {
        // 1,8° → 200 pas/tour, et réciproquement.
        assert_relative_eq!(steps_per_revolution(1.8), 200.0, epsilon = 1e-9);
        assert_relative_eq!(step_angle_deg(200.0), 1.8, epsilon = 1e-12);
    }

    #[test]
    fn speed_and_pulse_rate_are_inverse() {
        // 200 pas/tour : 1000 Hz → 300 tr/min ; conversion inverse redonne 1000 Hz.
        let n = speed_from_pulse_rate(1000.0, 200.0);
        assert_relative_eq!(n, 300.0, epsilon = 1e-9);
        assert_relative_eq!(pulse_rate_for_speed(n, 200.0), 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn linear_resolution_from_lead() {
        // 200 pas/tour (1,8°), vis pas 5 mm → 5/200 = 25 µm/pas.
        let r = linear_resolution(1.8, 0.005);
        assert_relative_eq!(r, 0.005 / 200.0, epsilon = 1e-12);
        assert_relative_eq!(r, 25e-6, epsilon = 1e-12);
    }

    #[test]
    fn microstepping_refines_resolution() {
        // 16 micro-pas : 3200 pas/tour → résolution 16× plus fine.
        let full = linear_resolution(step_angle_deg(200.0), 0.005);
        let micro = linear_resolution(step_angle_deg(3200.0), 0.005);
        assert_relative_eq!(full / micro, 16.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "angle de pas")]
    fn zero_step_angle_panics() {
        steps_per_revolution(0.0);
    }
}
