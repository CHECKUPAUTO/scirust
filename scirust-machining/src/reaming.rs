//! Usinage — **alésage à l'alésoir** (reaming) : conditions de coupe (vitesse
//! périphérique, avance en mm/min, temps d'usinage et régime réciproque).
//!
//! ```text
//! vitesse de coupe   v  = π·d·N/60          (m/s, N en tr/min)
//! avance             vf = f·N               (mm/min)
//! temps d'usinage    t  = L/(f·N)           (min)
//! régime (réciproque) n = v/(π·d)           (tr/s)
//! ```
//!
//! `d` diamètre de l'alésoir (m), `N` fréquence de rotation (tr/min), `v`
//! vitesse de coupe périphérique (m/s), `f` avance par tour (mm/tr), `vf` avance
//! (mm/min), `L` longueur alésée (mm), `t` temps (min), `n` régime (tr/s).
//!
//! **Convention** : SI mixte de fiche outil — diamètre et vitesse en unités SI
//! (m, m/s), longueur/avance en mm et régime en tr/min ; `π·d·N/60` sort en m/s
//! car `N/60` est un régime en tr/s. **Limite honnête** : coupe idéale à régime
//! établi ; la vitesse et l'avance sont des conditions de coupe **fournies** par
//! l'appelant (couple outil/matière). Ignore l'approche/dégagement et la
//! surépaisseur d'alésage, elles aussi **fournies** par l'appelant ; aucune
//! valeur « par défaut » n'est inventée ici.

use core::f64::consts::PI;

/// Vitesse de coupe périphérique `v = π·d·N/60` (m/s, `d` en m, `N` en tr/min).
///
/// Panique si `diameter <= 0` ou `rotational_speed_rpm <= 0`.
pub fn reaming_cutting_speed(diameter: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    assert!(
        rotational_speed_rpm > 0.0,
        "la fréquence de rotation doit être strictement positive"
    );
    PI * diameter * rotational_speed_rpm / 60.0
}

/// Avance en mm/min `vf = f·N` (`f` en mm/tr, `N` en tr/min).
///
/// Panique si `feed_per_revolution <= 0` ou `rotational_speed_rpm <= 0`.
pub fn reaming_feed_rate(feed_per_revolution: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(
        feed_per_revolution > 0.0,
        "l'avance par tour doit être strictement positive"
    );
    assert!(
        rotational_speed_rpm > 0.0,
        "la fréquence de rotation doit être strictement positive"
    );
    feed_per_revolution * rotational_speed_rpm
}

/// Temps d'usinage `t = L/(f·N)` (min, `L` en mm, `f` en mm/tr, `N` en tr/min).
///
/// Panique si `length <= 0`, `feed_per_revolution <= 0` ou
/// `rotational_speed_rpm <= 0`.
pub fn reaming_machining_time(
    length: f64,
    feed_per_revolution: f64,
    rotational_speed_rpm: f64,
) -> f64 {
    assert!(
        length > 0.0,
        "la longueur alésée doit être strictement positive"
    );
    assert!(
        feed_per_revolution > 0.0,
        "l'avance par tour doit être strictement positive"
    );
    assert!(
        rotational_speed_rpm > 0.0,
        "la fréquence de rotation doit être strictement positive"
    );
    length / (feed_per_revolution * rotational_speed_rpm)
}

/// Régime réciproque `n = v/(π·d)` (tr/s, `v` en m/s, `d` en m).
///
/// Panique si `cutting_speed <= 0` ou `diameter <= 0`.
pub fn reaming_spindle_speed(cutting_speed: f64, diameter: f64) -> f64 {
    assert!(
        cutting_speed > 0.0,
        "la vitesse de coupe doit être strictement positive"
    );
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    cutting_speed / (PI * diameter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cutting_speed_realistic_case() {
        // Alésoir Ø10 mm à 300 tr/min : v = π·0,010·300/60 = π·0,05 ≈ 0,15708 m/s.
        let v = reaming_cutting_speed(0.010, 300.0);
        assert_relative_eq!(v, PI * 0.05, epsilon = 1e-12);
        assert_relative_eq!(v, 0.157_079_632_679, epsilon = 1e-9);
    }

    #[test]
    fn spindle_speed_is_reciprocal_of_cutting_speed() {
        // n(v(d,N), d) = N/60 : la réciproque redonne le régime en tr/s.
        let d = 0.012;
        let rpm = 420.0;
        let v = reaming_cutting_speed(d, rpm);
        assert_relative_eq!(reaming_spindle_speed(v, d), rpm / 60.0, epsilon = 1e-12);
    }

    #[test]
    fn cutting_speed_scales_linearly_with_diameter() {
        // v ∝ d à régime fixe : doubler le diamètre double la vitesse de coupe.
        let v1 = reaming_cutting_speed(0.008, 500.0);
        let v2 = reaming_cutting_speed(0.016, 500.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn feed_rate_definition() {
        // f=0,3 mm/tr, N=300 tr/min → vf = 90 mm/min.
        assert_relative_eq!(reaming_feed_rate(0.3, 300.0), 90.0, epsilon = 1e-9);
    }

    #[test]
    fn machining_time_matches_length_over_feed_rate() {
        // t = L/vf : L=30 mm, vf = 0,3·300 = 90 mm/min → t = 1/3 min.
        let f = 0.3;
        let rpm = 300.0;
        let t = reaming_machining_time(30.0, f, rpm);
        assert_relative_eq!(t, 30.0 / reaming_feed_rate(f, rpm), epsilon = 1e-12);
        assert_relative_eq!(t, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn machining_time_halves_when_speed_doubles() {
        // t ∝ 1/N : doubler le régime divise le temps par deux.
        let t1 = reaming_machining_time(50.0, 0.25, 200.0);
        let t2 = reaming_machining_time(50.0, 0.25, 400.0);
        assert_relative_eq!(t1 / t2, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le diamètre doit être strictement positif")]
    fn cutting_speed_rejects_nonpositive_diameter() {
        reaming_cutting_speed(0.0, 300.0);
    }
}
