//! **Commande de moteur pas à pas** — géométrie de la commande en boucle
//! ouverte : angle de pas mécanique, résolution en micro-pas, vitesse de
//! rotation déduite de la fréquence des impulsions, nombre d'impulsions pour un
//! déplacement angulaire visé et couple utile décroissant avec la vitesse.
//!
//! ```text
//! angle de pas               θ_pas = 360 / N_pas
//! résolution micro-pas       θ_µ   = θ_pas / m
//! vitesse (tr/min)           n     = 60·f / N_pas
//! impulsions pour un angle   p     = θ_cible / θ_pas
//! couple utile               C_u   = C_h · k(n)
//! ```
//!
//! `θ_pas` angle d'un pas complet (degrés), `N_pas` nombre de pas complets par
//! tour, `m` nombre de divisions de micro-pas (entier fourni), `θ_µ` résolution
//! angulaire en micro-pas (degrés), `f` fréquence des impulsions de commande
//! (Hz, soit pas/s), `n` vitesse de rotation de l'arbre (tr/min), `θ_cible`
//! déplacement angulaire visé (degrés), `p` nombre d'impulsions correspondant,
//! `C_h` couple de maintien (N·m), `k(n)` facteur d'affaiblissement issu de la
//! courbe couple-vitesse (sans dimension, dans `[0, 1]`), `C_u` couple utile
//! disponible à la vitesse considérée (N·m).
//!
//! **Convention** : angles en **degrés** (usage moteur pas à pas), fréquence en
//! Hz, vitesse en tr/min, couples en N·m ; les nombres de pas et de micro-pas
//! sont des grandeurs réelles positives. **Limite honnête** : moteur commandé
//! en **boucle ouverte** ; le nombre de pas par tour `N_pas` et le nombre de
//! divisions de micro-pas `m` sont **fournis** par la fiche constructeur ; le
//! couple utile **décroît avec la vitesse** et le facteur `k(n)` de la courbe
//! couple-vitesse est **fourni par l'appelant** (le module ne modélise pas la
//! courbe elle-même) ; le **risque de perte de pas** au-delà de la charge ou de
//! la vitesse admissible n'est **pas modélisé**. Distinct du module
//! `stepper_motor` d'une autre crate.

/// Angle mécanique d'un pas complet `θ_pas = 360 / N_pas` (degrés).
///
/// Panique si `steps_per_revolution <= 0`.
pub fn stepdrv_step_angle(steps_per_revolution: f64) -> f64 {
    assert!(
        steps_per_revolution > 0.0,
        "le nombre de pas par tour N_pas doit être > 0"
    );
    360.0 / steps_per_revolution
}

/// Résolution angulaire en micro-pas `θ_µ = θ_pas / m` (degrés).
///
/// Panique si `full_step_angle <= 0` ou si `microstep_divisions <= 0`.
pub fn stepdrv_resolution_microstepping(full_step_angle: f64, microstep_divisions: f64) -> f64 {
    assert!(full_step_angle > 0.0, "l'angle de pas θ_pas doit être > 0");
    assert!(
        microstep_divisions > 0.0,
        "le nombre de divisions de micro-pas m doit être > 0"
    );
    full_step_angle / microstep_divisions
}

/// Vitesse de rotation de l'arbre `n = 60·f / N_pas` (tr/min) déduite de la
/// fréquence des impulsions de commande.
///
/// Panique si `step_frequency < 0` ou si `steps_per_revolution <= 0`.
pub fn stepdrv_speed_rpm(step_frequency: f64, steps_per_revolution: f64) -> f64 {
    assert!(
        step_frequency >= 0.0,
        "la fréquence des impulsions f doit être ≥ 0"
    );
    assert!(
        steps_per_revolution > 0.0,
        "le nombre de pas par tour N_pas doit être > 0"
    );
    60.0 * step_frequency / steps_per_revolution
}

/// Nombre d'impulsions pour un déplacement angulaire visé
/// `p = θ_cible / θ_pas` (sans dimension).
///
/// Panique si `target_angle < 0` ou si `step_angle <= 0`.
pub fn stepdrv_pulses_for_angle(target_angle: f64, step_angle: f64) -> f64 {
    assert!(target_angle >= 0.0, "l'angle visé θ_cible doit être ≥ 0");
    assert!(step_angle > 0.0, "l'angle de pas θ_pas doit être > 0");
    target_angle / step_angle
}

/// Couple utile disponible à la vitesse considérée `C_u = C_h · k(n)` (N·m), le
/// facteur `speed_factor` provenant de la courbe couple-vitesse du constructeur.
///
/// Panique si `holding_torque < 0` ou si `speed_factor` n'est pas dans `[0, 1]`.
pub fn stepdrv_holding_to_working_torque(holding_torque: f64, speed_factor: f64) -> f64 {
    assert!(
        holding_torque >= 0.0,
        "le couple de maintien C_h doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&speed_factor),
        "le facteur couple-vitesse k(n) doit être dans [0, 1]"
    );
    holding_torque * speed_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn step_angle_microstepping_chain() {
        // Moteur classique 200 pas/tour : θ_pas = 360/200 = 1,8°.
        let theta = stepdrv_step_angle(200.0);
        assert_relative_eq!(theta, 1.8, epsilon = 1e-12);
        // Micro-pas 1/16 : θ_µ = 1,8/16 = 0,1125°.
        let micro = stepdrv_resolution_microstepping(theta, 16.0);
        assert_relative_eq!(micro, 0.1125, epsilon = 1e-12);
    }

    #[test]
    fn microstepping_multiplies_effective_steps_per_rev() {
        // Sur un tour complet, le nombre de micro-pas = 360 / θ_µ doit valoir
        // N_pas · m : ici 200 · 16 = 3200 micro-pas par tour.
        let theta = stepdrv_step_angle(200.0);
        let micro = stepdrv_resolution_microstepping(theta, 16.0);
        assert_relative_eq!(360.0 / micro, 200.0 * 16.0, epsilon = 1e-9);
    }

    #[test]
    fn speed_proportional_to_frequency_and_inverse_to_steps() {
        // n ∝ f : doubler la fréquence double la vitesse.
        let n1 = stepdrv_speed_rpm(1000.0, 200.0);
        let n2 = stepdrv_speed_rpm(2000.0, 200.0);
        assert_relative_eq!(n2 / n1, 2.0, epsilon = 1e-12);
        // n ∝ 1/N_pas : doubler le nombre de pas par tour divise la vitesse par 2.
        let n3 = stepdrv_speed_rpm(1000.0, 400.0);
        assert_relative_eq!(n1 / n3, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn speed_realistic_case() {
        // Cas chiffré : f = 1000 Hz, N_pas = 200.
        //   n = 60·1000 / 200 = 60000 / 200 = 300 tr/min
        let n = stepdrv_speed_rpm(1000.0, 200.0);
        assert_relative_eq!(n, 300.0, epsilon = 1e-9);
    }

    #[test]
    fn pulses_invert_step_angle_over_full_turn() {
        // Un tour complet (360°) doit demander exactement N_pas impulsions.
        let theta = stepdrv_step_angle(200.0);
        let p = stepdrv_pulses_for_angle(360.0, theta);
        assert_relative_eq!(p, 200.0, epsilon = 1e-9);
        // Cas chiffré : θ_cible = 90°, θ_pas = 1,8° → p = 90/1,8 = 50 impulsions.
        let p90 = stepdrv_pulses_for_angle(90.0, 1.8);
        assert_relative_eq!(p90, 50.0, epsilon = 1e-9);
    }

    #[test]
    fn working_torque_bounds() {
        // À l'arrêt (k = 1) le couple utile égale le couple de maintien.
        assert_relative_eq!(
            stepdrv_holding_to_working_torque(1.2, 1.0),
            1.2,
            epsilon = 1e-12
        );
        // À grande vitesse le facteur chute : k = 0,25 → C_u = 1,2·0,25 = 0,3 N·m.
        assert_relative_eq!(
            stepdrv_holding_to_working_torque(1.2, 0.25),
            0.3,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "le facteur couple-vitesse k(n) doit être dans [0, 1]")]
    fn working_torque_rejects_factor_above_one() {
        let _ = stepdrv_holding_to_working_torque(1.2, 1.5);
    }
}
