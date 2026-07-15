//! **Contrôle non destructif par ultrasons** (CND / UT) — grandeurs de base pour
//! le sondage d'une pièce par une onde ultrasonore émise par un traducteur.
//!
//! ```text
//! longueur d'onde        lambda = c / f
//! profondeur de défaut   d      = c · t / 2        (temps de vol aller-retour)
//! longueur de champ proche  N   = D² / (4 · lambda)
//! atténuation            A      = alpha · x
//! ```
//!
//! `lambda` longueur d'onde ultrasonore (m), `c` vitesse du son dans le matériau
//! (m/s), `f` fréquence du traducteur (Hz), `d` profondeur du réflecteur sous la
//! surface d'entrée (m), `t` temps de vol aller-retour mesuré (s), `N` longueur
//! du champ proche (zone de Fresnel) du traducteur (m), `D` diamètre du cristal
//! actif du traducteur (m), `A` atténuation cumulée sur le trajet (dB), `alpha`
//! coefficient d'atténuation du matériau (dB/m), `x` distance parcourue (m).
//!
//! **Convention** : SI (fréquence en Hz, distances en m, temps en s) ;
//! l'atténuation est exprimée en décibels (dB) et `alpha` en dB/m. **Limite
//! honnête** : matériau supposé **homogène et isotrope**, **incidence normale**
//! de l'onde, propagation en **onde plane** ; la **diffraction**, la dispersion
//! et les conversions de mode sont **négligées**. La vitesse du son `c`, la
//! fréquence `f`, le diamètre `D` et le coefficient d'atténuation `alpha` sont
//! des **données de l'appelant** (ils dépendent du matériau, du mode d'onde et
//! du traducteur) et ne sont jamais supposés par défaut.

/// Longueur d'onde ultrasonore `lambda = c / f` (m).
///
/// Rapport de la vitesse du son dans le matériau à la fréquence d'émission ;
/// fixe la résolution axiale et le pouvoir de détection des petits défauts.
///
/// Panique si `sound_velocity <= 0` ou `frequency <= 0`.
pub fn ut_wavelength(sound_velocity: f64, frequency: f64) -> f64 {
    assert!(sound_velocity > 0.0, "la vitesse du son c doit être > 0");
    assert!(frequency > 0.0, "la fréquence f doit être > 0");
    sound_velocity / frequency
}

/// Profondeur d'un défaut `d = c · t / 2` (m) à partir du temps de vol.
///
/// Le facteur `1/2` traduit le trajet **aller-retour** de l'écho : l'onde
/// parcourt deux fois la profondeur entre l'émission et la réception.
///
/// Panique si `sound_velocity <= 0` ou `time_of_flight < 0`.
pub fn ut_defect_depth(sound_velocity: f64, time_of_flight: f64) -> f64 {
    assert!(sound_velocity > 0.0, "la vitesse du son c doit être > 0");
    assert!(time_of_flight >= 0.0, "le temps de vol t doit être ≥ 0");
    sound_velocity * time_of_flight / 2.0
}

/// Longueur du champ proche `N = D² / (4 · lambda)` (m).
///
/// Étendue de la zone de Fresnel d'un traducteur circulaire : le faisceau y est
/// convergent et l'amplitude irrégulière ; au-delà commence le champ lointain.
///
/// Panique si `transducer_diameter <= 0` ou `wavelength <= 0`.
pub fn ut_near_field_length(transducer_diameter: f64, wavelength: f64) -> f64 {
    assert!(
        transducer_diameter > 0.0,
        "le diamètre du traducteur D doit être > 0"
    );
    assert!(wavelength > 0.0, "la longueur d'onde lambda doit être > 0");
    transducer_diameter.powi(2) / (4.0 * wavelength)
}

/// Atténuation cumulée `A = alpha · x` (dB) sur un trajet de longueur `x`.
///
/// Perte d'amplitude, en décibels, subie par l'onde sur la distance parcourue,
/// proportionnelle au coefficient d'atténuation du matériau.
///
/// Panique si `attenuation_coefficient < 0` ou `distance < 0`.
pub fn ut_attenuation_db(attenuation_coefficient: f64, distance: f64) -> f64 {
    assert!(
        attenuation_coefficient >= 0.0,
        "le coefficient d'atténuation alpha doit être ≥ 0"
    );
    assert!(distance >= 0.0, "la distance x doit être ≥ 0");
    attenuation_coefficient * distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn wavelength_reciprocity_with_velocity() {
        // Identité de définition : c = lambda · f.
        let c = 5920.0_f64;
        let f = 5.0e6_f64;
        let lambda = ut_wavelength(c, f);
        assert_relative_eq!(lambda * f, c, epsilon = 1e-9);
    }

    #[test]
    fn defect_depth_proportional_to_time() {
        // d ∝ t à vitesse fixée : doubler le temps de vol double la profondeur.
        let c = 5920.0_f64;
        let d1 = ut_defect_depth(c, 10.0e-6);
        let d2 = ut_defect_depth(c, 20.0e-6);
        assert_relative_eq!(d2, 2.0 * d1, epsilon = 1e-12);
    }

    #[test]
    fn near_field_scales_with_frequency() {
        // En combinant N = D²/(4·lambda) et lambda = c/f, on a N = D²·f/(4·c) :
        // à c et D fixés, N est proportionnel à f.
        let c = 5920.0_f64;
        let d = 0.01_f64;
        let n_low = ut_near_field_length(d, ut_wavelength(c, 2.5e6));
        let n_high = ut_near_field_length(d, ut_wavelength(c, 5.0e6));
        assert_relative_eq!(n_high, 2.0 * n_low, epsilon = 1e-9);
    }

    #[test]
    fn attenuation_is_linear_in_distance() {
        // A ∝ x à coefficient fixé : l'atténuation triple avec un trajet triple.
        let alpha = 10.0_f64;
        let a1 = ut_attenuation_db(alpha, 0.2);
        let a3 = ut_attenuation_db(alpha, 0.6);
        assert_relative_eq!(a3, 3.0 * a1, epsilon = 1e-12);
    }

    #[test]
    fn steel_probe_realistic_case() {
        // Acier : c = 5920 m/s, traducteur f = 5 MHz, D = 10 mm.
        // lambda = 5920 / 5e6 = 1,184e-3 m (1,184 mm).
        let c = 5920.0_f64;
        let lambda = ut_wavelength(c, 5.0e6);
        assert_relative_eq!(lambda, 1.184e-3, epsilon = 1e-9);
        // N = D²/(4·lambda) = (0,01)² / (4·1,184e-3)
        //   = 1e-4 / 4,736e-3 = 0,021114865 m ≈ 21,1 mm.
        let n = ut_near_field_length(0.01, lambda);
        assert_relative_eq!(n, 0.021_114_865, epsilon = 1e-6);
        // Écho reçu à t = 20 µs → d = 5920·20e-6/2 = 0,0592 m (59,2 mm).
        let d = ut_defect_depth(c, 20.0e-6);
        assert_relative_eq!(d, 0.0592, epsilon = 1e-9);
        // Atténuation sur ce trajet aller-retour (2·d = 0,1184 m) à alpha = 20 dB/m
        // → A = 20·0,1184 = 2,368 dB.
        let a = ut_attenuation_db(20.0, 2.0 * d);
        assert_relative_eq!(a, 2.368, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la fréquence f doit être > 0")]
    fn zero_frequency_panics() {
        ut_wavelength(5920.0, 0.0);
    }
}
