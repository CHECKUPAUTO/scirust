//! **Moteur asynchrone triphasé** — vitesse de synchronisme, glissement, vitesse
//! rotorique et fréquence des courants rotoriques.
//!
//! ```text
//! vitesse de synchronisme   Ns = 60·f/p                   (tr/min)
//! glissement                s  = (Ns − Nr)/Ns             (sans dimension)
//! vitesse rotorique         Nr = Ns·(1 − s)               (tr/min)
//! fréquence rotorique       fr = s·f                      (Hz)
//! ```
//!
//! `f` fréquence d'alimentation (Hz), `p` nombre de **paires de pôles** (entier
//! ≥ 1, ex. 2 paires = 4 pôles), `Ns` vitesse du champ tournant (tr/min), `Nr`
//! vitesse mécanique du rotor (tr/min), `s` glissement (0 = synchronisme, 1 =
//! rotor à l'arrêt), `fr` fréquence des courants induits au rotor (Hz).
//!
//! **Convention** : fréquences en Hz, vitesses en tr/min, glissement en fraction
//! (0 à 1 en fonctionnement moteur).
//! **Limite honnête** : relations du **régime permanent** sous alimentation
//! **sinusoïdale triphasée équilibrée** ; la fréquence et le nombre de paires de
//! pôles sont des données de la machine **fournies par l'appelant** (aucune
//! valeur « par défaut » inventée). Les pertes fer, mécaniques et le rendement ne
//! sont **pas** modélisés ici — voir [`crate::motor_torque`] pour le couple et un
//! module de rendement dédié pour les pertes.

/// Vitesse de synchronisme `Ns = 60·f/p` (tr/min) du champ tournant.
///
/// Panique si `supply_frequency < 0` ou `pole_pairs < 1`.
pub fn induction_synchronous_speed_rpm(supply_frequency: f64, pole_pairs: f64) -> f64 {
    assert!(
        supply_frequency >= 0.0,
        "la fréquence d'alimentation doit être positive ou nulle"
    );
    assert!(
        pole_pairs >= 1.0,
        "le nombre de paires de pôles doit être supérieur ou égal à 1"
    );
    60.0 * supply_frequency / pole_pairs
}

/// Glissement `s = (Ns − Nr)/Ns` (sans dimension).
///
/// Panique si `synchronous_speed <= 0`.
pub fn induction_slip(synchronous_speed: f64, rotor_speed: f64) -> f64 {
    assert!(
        synchronous_speed > 0.0,
        "la vitesse de synchronisme doit être strictement positive"
    );
    (synchronous_speed - rotor_speed) / synchronous_speed
}

/// Vitesse rotorique `Nr = Ns·(1 − s)` (tr/min) depuis le glissement.
///
/// Panique si `synchronous_speed < 0`.
pub fn induction_rotor_speed_rpm(synchronous_speed: f64, slip: f64) -> f64 {
    assert!(
        synchronous_speed >= 0.0,
        "la vitesse de synchronisme doit être positive ou nulle"
    );
    synchronous_speed * (1.0 - slip)
}

/// Fréquence des courants rotoriques `fr = s·f` (Hz).
///
/// Panique si `supply_frequency < 0`.
pub fn induction_rotor_frequency(supply_frequency: f64, slip: f64) -> f64 {
    assert!(
        supply_frequency >= 0.0,
        "la fréquence d'alimentation doit être positive ou nulle"
    );
    slip * supply_frequency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn four_pole_50hz_gives_1500_rpm() {
        // 4 pôles = 2 paires, 50 Hz → Ns = 60·50/2 = 1500 tr/min.
        assert_relative_eq!(
            induction_synchronous_speed_rpm(50.0, 2.0),
            1500.0,
            epsilon = 1e-9
        );
        // 2 pôles = 1 paire, 60 Hz → 3600 tr/min.
        assert_relative_eq!(
            induction_synchronous_speed_rpm(60.0, 1.0),
            3600.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn slip_and_rotor_speed_are_inverse() {
        // Ns = 1500, Nr = 1455 → s = 45/1500 = 0,03 ; réciproquement Nr = 1500·0,97.
        let s = induction_slip(1500.0, 1455.0);
        assert_relative_eq!(s, 0.03, epsilon = 1e-12);
        assert_relative_eq!(induction_rotor_speed_rpm(1500.0, s), 1455.0, epsilon = 1e-9);
    }

    #[test]
    fn synchronism_and_standstill_limits() {
        // Rotor au synchronisme → glissement nul ; rotor à l'arrêt → glissement 1.
        assert_relative_eq!(induction_slip(1500.0, 1500.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(induction_slip(1500.0, 0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rotor_frequency_tracks_slip() {
        // 50 Hz, s = 0,03 → fr = 1,5 Hz ; au démarrage (s = 1) fr = f.
        assert_relative_eq!(induction_rotor_frequency(50.0, 0.03), 1.5, epsilon = 1e-12);
        assert_relative_eq!(induction_rotor_frequency(50.0, 1.0), 50.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_operating_point_is_consistent() {
        // Machine 4 pôles, 50 Hz, glissement nominal 3 % : chaîne complète cohérente.
        let ns = induction_synchronous_speed_rpm(50.0, 2.0);
        let nr = induction_rotor_speed_rpm(ns, 0.03);
        assert_relative_eq!(nr, 1455.0, epsilon = 1e-9);
        // Le glissement recalculé depuis Nr redonne 3 %.
        assert_relative_eq!(induction_slip(ns, nr), 0.03, epsilon = 1e-12);
        // Fréquence rotorique associée : 1,5 Hz.
        assert_relative_eq!(induction_rotor_frequency(50.0, 0.03), 1.5, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "paires de pôles")]
    fn zero_pole_pairs_panics() {
        induction_synchronous_speed_rpm(50.0, 0.0);
    }
}
