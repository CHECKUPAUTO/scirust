//! **Analyse d'ordres** — conversion entre vitesse d'arbre, fréquence et ordre
//! pour le diagnostic vibratoire des machines tournantes (rééchantillonnage
//! angulaire, suivi d'ordres).
//!
//! ```text
//! fréquence de l'ordre k     f_k = k · rpm / 60
//! numéro d'ordre             k   = f · 60 / rpm
//! fréquence de rotation      f_r = rpm / 60          (= f_1, l'ordre 1)
//! Shannon angulaire          Spr = 2 · k_max
//! ordre max représentable    k_max = Spr / 2
//! ```
//!
//! `rpm` vitesse de rotation de l'arbre (tr/min), `k` numéro d'ordre
//! (sans dimension, multiple de la fréquence de rotation), `f`, `f_k`, `f_r`
//! fréquences (Hz), `Spr` nombre d'échantillons par tour (—). L'ordre est le
//! rapport d'une fréquence à la fréquence de rotation de l'arbre : l'ordre 1
//! correspond à un tour d'arbre.
//!
//! **Convention** : SI, fréquences en Hz, vitesses en tr/min, angles implicites
//! en tours. **Limite honnête** : la vitesse d'arbre est supposée **connue**
//! (mesurée par tachymètre / codeur), et le rééchantillonnage angulaire est
//! supposé **idéal** (échantillonnage synchrone parfait, sans jitter ni erreur
//! d'interpolation). Aucune vitesse, aucun nombre d'échantillons par tour n'est
//! inventé par défaut : ces valeurs sont **fournies par l'appelant** (chaîne
//! d'acquisition). L'interprétation physique des ordres (ordre 1 =
//! déséquilibre, harmoniques = défauts d'alignement, d'engrènement, etc.) est à
//! la **charge de l'appelant** ; ce module ne fait que la conversion
//! cinématique. Voir [`crate::bearing_defect_frequencies`] et
//! [`crate::balancing`].

/// Fréquence (Hz) de l'ordre `k` pour une vitesse d'arbre donnée :
/// `f_k = k · rpm / 60`.
///
/// Convertit un numéro d'ordre en fréquence physique. L'ordre `k = 1` redonne
/// la fréquence de rotation de l'arbre.
///
/// Panique si `shaft_speed_rpm < 0` ou `order < 0`.
pub fn order_frequency(shaft_speed_rpm: f64, order: f64) -> f64 {
    assert!(
        shaft_speed_rpm >= 0.0,
        "la vitesse d'arbre rpm doit être positive"
    );
    assert!(order >= 0.0, "le numéro d'ordre k doit être positif");
    order * shaft_speed_rpm / 60.0
}

/// Numéro d'ordre (—) correspondant à une fréquence pour une vitesse d'arbre
/// donnée : `k = f · 60 / rpm`.
///
/// Opération réciproque de [`order_frequency`] : rapporte une raie fréquentielle
/// à la fréquence de rotation de l'arbre.
///
/// Panique si `frequency_hz < 0` ou `shaft_speed_rpm <= 0`.
pub fn order_number(frequency_hz: f64, shaft_speed_rpm: f64) -> f64 {
    assert!(frequency_hz >= 0.0, "la fréquence f doit être positive");
    assert!(
        shaft_speed_rpm > 0.0,
        "la vitesse d'arbre rpm doit être strictement positive"
    );
    frequency_hz * 60.0 / shaft_speed_rpm
}

/// Fréquence de rotation de l'arbre (Hz) : `f_r = rpm / 60`.
///
/// Il s'agit de la fréquence de l'ordre 1, `order_frequency(rpm, 1)`.
///
/// Panique si `shaft_speed_rpm < 0`.
pub fn order_shaft_frequency(shaft_speed_rpm: f64) -> f64 {
    assert!(
        shaft_speed_rpm >= 0.0,
        "la vitesse d'arbre rpm doit être positive"
    );
    shaft_speed_rpm / 60.0
}

/// Nombre minimal d'échantillons par tour pour représenter les ordres jusqu'à
/// `max_order` (critère de Shannon en domaine angulaire) : `Spr = 2 · k_max`.
///
/// L'échantillonnage angulaire synchrone joue le rôle de l'échantillonnage
/// temporel : l'ordre maximal représentable sans repliement est la moitié du
/// nombre d'échantillons par tour.
///
/// Panique si `max_order <= 0`.
pub fn order_samples_per_rev_for_max_order(max_order: f64) -> f64 {
    assert!(
        max_order > 0.0,
        "l'ordre maximal k_max doit être strictement positif"
    );
    2.0 * max_order
}

/// Ordre maximal représentable sans repliement pour un nombre d'échantillons
/// par tour donné : `k_max = Spr / 2`.
///
/// Opération réciproque de [`order_samples_per_rev_for_max_order`] : c'est
/// l'« ordre de Nyquist » du rééchantillonnage angulaire.
///
/// Panique si `samples_per_rev <= 0`.
pub fn order_max_from_samples_per_rev(samples_per_rev: f64) -> f64 {
    assert!(
        samples_per_rev > 0.0,
        "le nombre d'échantillons par tour Spr doit être strictement positif"
    );
    samples_per_rev / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence réaliste : arbre à 1800 tr/min → f_r = 30 Hz.
    //   ordre 1 : f_1 = 1·1800/60  = 30 Hz
    //   ordre 3 : f_3 = 3·1800/60  = 90 Hz
    //   ordre inverse : k = 90·60/1800 = 3
    const RPM: f64 = 1800.0;

    #[test]
    fn reference_case_values() {
        assert_relative_eq!(order_frequency(RPM, 1.0), 30.0, epsilon = 1e-9);
        assert_relative_eq!(order_frequency(RPM, 3.0), 90.0, epsilon = 1e-9);
        assert_relative_eq!(order_shaft_frequency(RPM), 30.0, epsilon = 1e-9);
        assert_relative_eq!(order_number(90.0, RPM), 3.0, epsilon = 1e-9);
    }

    #[test]
    fn frequency_and_number_are_reciprocal() {
        // order_number(order_frequency(rpm, k), rpm) == k pour tout k.
        for &k in &[0.5_f64, 1.0, 2.5, 7.0]
        {
            let f = order_frequency(RPM, k);
            assert_relative_eq!(order_number(f, RPM), k, epsilon = 1e-9);
        }
    }

    #[test]
    fn order_one_equals_shaft_frequency() {
        // L'ordre 1 est, par définition, la fréquence de rotation de l'arbre.
        assert_relative_eq!(
            order_frequency(RPM, 1.0),
            order_shaft_frequency(RPM),
            epsilon = 1e-12
        );
    }

    #[test]
    fn frequency_is_bilinear_in_order_and_speed() {
        // f_k = k·rpm/60 : linéaire en k et en rpm séparément.
        let a = 2.0;
        let b = 3.0;
        assert_relative_eq!(
            order_frequency(RPM, a * 4.0),
            a * order_frequency(RPM, 4.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            order_frequency(b * RPM, 4.0),
            b * order_frequency(RPM, 4.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn shannon_conversions_are_reciprocal() {
        // Spr = 2·k_max et k_max = Spr/2 sont inverses l'une de l'autre.
        for &kmax in &[1.0_f64, 5.0, 12.5, 40.0]
        {
            let spr = order_samples_per_rev_for_max_order(kmax);
            assert_relative_eq!(spr, 2.0 * kmax, epsilon = 1e-12);
            assert_relative_eq!(order_max_from_samples_per_rev(spr), kmax, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "la vitesse d'arbre rpm doit être strictement positive")]
    fn zero_speed_in_order_number_panics() {
        order_number(90.0, 0.0);
    }
}
