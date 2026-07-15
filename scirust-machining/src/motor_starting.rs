//! **Démarrage d'un moteur** — couple accélérateur, temps de montée en
//! vitesse à couple constant et courant appelé au démarrage direct ou en
//! étoile-triangle.
//!
//! ```text
//! couple accélérateur   T_acc = T_moteur − T_charge
//! temps d'accélération  t     = I · Δω / T_acc
//! courant démarrage DOL I_dol = I_nom · k
//! courant étoile-triangle I_yd = I_dol / 3
//! ```
//!
//! `T_moteur` couple moteur développé (N·m), `T_charge` couple résistant de la
//! charge (N·m), `T_acc` couple accélérateur disponible (N·m), `I` inertie
//! totale ramenée à l'arbre (kg·m²), `Δω` variation de vitesse angulaire à
//! atteindre (rad/s), `t` temps d'accélération (s), `I_nom` courant nominal
//! (A), `k` ratio de courant de démarrage direct (sans dimension, ~6–7),
//! `I_dol` courant de démarrage direct (A), `I_yd` courant de démarrage
//! étoile-triangle (A).
//!
//! **Convention** : SI ; couples en N·m ; inertie en kg·m² ; vitesse
//! angulaire en rad/s ; courants en ampères. **Limite honnête** : les couples
//! moteur et résistant sont supposés **constants** sur la plage de vitesse
//! (approximation), l'inertie totale ramenée à l'arbre est **fournie** par
//! l'appelant, et le ratio de courant de démarrage `k` est **fourni** par le
//! constructeur (plaque signalétique) ; on **néglige** la variation du couple
//! avec le glissement. Aucune valeur « par défaut » n'est inventée.

/// Couple accélérateur disponible `T_acc = T_moteur − T_charge`.
///
/// Le résultat peut être négatif : le moteur ne peut alors pas démarrer la
/// charge (couple résistant supérieur au couple moteur).
///
/// Panique si `motor_torque` ou `load_torque` n'est pas fini, ou si l'un des
/// deux couples est négatif.
pub fn motor_acceleration_torque(motor_torque: f64, load_torque: f64) -> f64 {
    assert!(motor_torque.is_finite(), "T_moteur doit être fini");
    assert!(load_torque.is_finite(), "T_charge doit être fini");
    assert!(motor_torque >= 0.0, "T_moteur ≥ 0 requis");
    assert!(load_torque >= 0.0, "T_charge ≥ 0 requis");
    motor_torque - load_torque
}

/// Temps d'accélération à couple constant `t = I · Δω / T_acc`.
///
/// Panique si `total_inertia <= 0`, si `speed_change_rad <= 0` ou si
/// `acceleration_torque <= 0` (le couple accélérateur doit être positif pour
/// que le moteur monte en vitesse).
pub fn motor_starting_time(
    total_inertia: f64,
    speed_change_rad: f64,
    acceleration_torque: f64,
) -> f64 {
    assert!(total_inertia > 0.0, "I > 0 requis");
    assert!(speed_change_rad > 0.0, "Δω > 0 requis");
    assert!(
        acceleration_torque > 0.0,
        "T_acc > 0 requis (le moteur doit accélérer)"
    );
    total_inertia * speed_change_rad / acceleration_torque
}

/// Courant de démarrage direct (DOL) `I_dol = I_nom · k`.
///
/// Le ratio `k` est fourni par le constructeur (typiquement ~6–7).
///
/// Panique si `rated_current <= 0` ou si `starting_current_ratio < 1`
/// (un démarrage direct appelle au moins le courant nominal).
pub fn motor_dol_starting_current(rated_current: f64, starting_current_ratio: f64) -> f64 {
    assert!(rated_current > 0.0, "I_nom > 0 requis");
    assert!(
        starting_current_ratio >= 1.0,
        "k ≥ 1 requis (courant de démarrage ≥ courant nominal)"
    );
    rated_current * starting_current_ratio
}

/// Courant de démarrage étoile-triangle `I_yd = I_dol / 3`.
///
/// Le couplage étoile divise par 3 le courant de ligne appelé (et le couple)
/// par rapport au démarrage direct en triangle.
///
/// Panique si `dol_starting_current <= 0`.
pub fn motor_star_delta_starting_current(dol_starting_current: f64) -> f64 {
    assert!(dol_starting_current > 0.0, "I_dol > 0 requis");
    dol_starting_current / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn acceleration_torque_is_the_difference() {
        // T_acc = T_moteur − T_charge.
        assert_relative_eq!(motor_acceleration_torque(50.0, 20.0), 30.0);
    }

    #[test]
    fn acceleration_torque_can_be_negative_when_load_dominates() {
        // Charge plus forte que le moteur : couple accélérateur négatif.
        assert_relative_eq!(motor_acceleration_torque(15.0, 40.0), -25.0);
    }

    #[test]
    fn starting_time_realistic_case() {
        // I = 2 kg·m², Δω = 60 rad/s, T_acc = 30 N·m
        // t = 2 · 60 / 30 = 4 s.
        assert_relative_eq!(motor_starting_time(2.0, 60.0, 30.0), 4.0);
    }

    #[test]
    fn starting_time_is_proportional_to_inertia() {
        // Doubler l'inertie double le temps d'accélération (Δω, T_acc fixes).
        let t1 = motor_starting_time(2.0, 60.0, 30.0);
        let t2 = motor_starting_time(4.0, 60.0, 30.0);
        assert_relative_eq!(t2, 2.0 * t1);
    }

    #[test]
    fn starting_time_from_impulse_momentum_identity() {
        // Identité impulsion angulaire : T_acc · t = I · Δω.
        let inertia = 3.5_f64;
        let d_omega = 157.08_f64;
        let t_acc = 42.0_f64;
        let t = motor_starting_time(inertia, d_omega, t_acc);
        assert_relative_eq!(t_acc * t, inertia * d_omega, epsilon = 1e-9);
    }

    #[test]
    fn star_delta_thirds_the_dol_current() {
        // I_dol = 10 · 6 = 60 A ; I_yd = 60 / 3 = 20 A.
        let i_dol = motor_dol_starting_current(10.0, 6.0);
        assert_relative_eq!(i_dol, 60.0);
        assert_relative_eq!(motor_star_delta_starting_current(i_dol), 20.0);
    }

    #[test]
    #[should_panic(expected = "T_acc > 0 requis")]
    fn starting_time_panics_when_no_accelerating_torque() {
        let _ = motor_starting_time(2.0, 60.0, 0.0);
    }
}
