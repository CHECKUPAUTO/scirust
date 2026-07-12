//! **Couple moteur** d'un axe — couple d'accélération, couple total et couple
//! **efficace (RMS)** sur un cycle, pour le dimensionnement thermique du moteur.
//!
//! ```text
//! accél. angulaire  α = Δω/Δt
//! couple d'accél.   C_a = J·α                           (J inertie totale à l'arbre)
//! couple total      C = C_a + C_frot + C_charge
//! couple RMS        C_rms = √( Σ Cᵢ²·tᵢ / Σ tᵢ )
//! ```
//!
//! `J` inertie totale ramenée à l'arbre (kg·m², moteur + charge, voir
//! [`crate::reflected_inertia`]), `α` accélération angulaire (rad/s²), `Δω`
//! variation de vitesse (rad/s), `C` couples (N·m), `C_rms` couple efficace sur
//! un cycle (segments `(couple, durée)`).
//!
//! **Convention** : SI (rad/s, rad/s², N·m, s). **Limite honnête** : le couple
//! d'accélération suppose une inertie **constante** et une accélération
//! **uniforme** sur le segment ; le couple **RMS** dimensionne l'échauffement
//! (à comparer au couple **nominal** continu), tandis que le couple de pointe
//! d'accélération se compare au couple **crête** — les deux limites moteur étant
//! des données fournies par l'appelant.

/// Accélération angulaire `α = Δω/Δt`.
///
/// Panique si `time <= 0`.
pub fn angular_acceleration(delta_omega: f64, time: f64) -> f64 {
    assert!(time > 0.0, "la durée doit être strictement positive");
    delta_omega / time
}

/// Couple d'accélération `C_a = J·α`.
///
/// Panique si `inertia < 0`.
pub fn acceleration_torque(inertia: f64, angular_accel: f64) -> f64 {
    assert!(inertia >= 0.0, "l'inertie doit être positive");
    inertia * angular_accel
}

/// Couple total `C = C_a + C_frot + C_charge`.
pub fn total_torque(acceleration_torque: f64, friction_torque: f64, load_torque: f64) -> f64 {
    acceleration_torque + friction_torque + load_torque
}

/// Couple **efficace (RMS)** sur un cycle `C_rms = √(Σ Cᵢ²·tᵢ / Σ tᵢ)`.
///
/// Chaque segment est un couple `(N·m)` maintenu pendant une durée `(s)`.
/// Panique si la liste est vide, la durée totale nulle ou une durée négative.
pub fn rms_torque(segments: &[(f64, f64)]) -> f64 {
    assert!(!segments.is_empty(), "au moins un segment est requis");
    let mut sum_sq_t = 0.0;
    let mut sum_t = 0.0;
    for &(torque, dt) in segments
    {
        assert!(dt >= 0.0, "les durées doivent être positives");
        sum_sq_t += torque * torque * dt;
        sum_t += dt;
    }
    assert!(
        sum_t > 0.0,
        "la durée totale du cycle doit être strictement positive"
    );
    (sum_sq_t / sum_t).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn acceleration_torque_from_inertia() {
        // J=0,01 kg·m² accéléré de 0 à 314 rad/s en 0,2 s → α=1570, C=15,7 N·m.
        let alpha = angular_acceleration(314.0, 0.2);
        let c = acceleration_torque(0.01, alpha);
        assert_relative_eq!(alpha, 1570.0, epsilon = 1e-9);
        assert_relative_eq!(c, 15.7, epsilon = 1e-9);
    }

    #[test]
    fn total_sums_components() {
        assert_relative_eq!(total_torque(15.7, 0.5, 2.0), 18.2, epsilon = 1e-9);
    }

    #[test]
    fn rms_of_constant_is_that_value() {
        // Couple constant sur tout le cycle → RMS = ce couple.
        assert_relative_eq!(rms_torque(&[(4.0, 1.0), (4.0, 3.0)]), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn rms_between_min_and_max() {
        // Cycle accél/repos : RMS entre le couple mini et maxi du cycle.
        let segs = [(20.0, 0.2), (2.0, 0.6), (20.0, 0.2)];
        let c_rms = rms_torque(&segs);
        // √((400·0,2 + 4·0,6 + 400·0,2)/1,0) = √162,4 ≈ 12,74.
        assert_relative_eq!(c_rms, (162.4_f64).sqrt(), epsilon = 1e-9);
        assert!(c_rms > 2.0 && c_rms < 20.0);
    }

    #[test]
    #[should_panic(expected = "au moins un segment")]
    fn empty_cycle_panics() {
        rms_torque(&[]);
    }
}
