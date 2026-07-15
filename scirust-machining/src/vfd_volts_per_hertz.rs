//! **Variateur de vitesse — loi V/f (Volts par Hertz)** — commande scalaire d'un
//! moteur asynchrone maintenant le flux quasi constant en asservissant la tension
//! proportionnellement à la fréquence, avec un *boost* de tension à bas régime.
//!
//! ```text
//! rapport V/f nominal   k   = Vn / fn                    (V/Hz)
//! tension appliquée     V   = k·f + Vboost               (V)
//! flux relatif          φr  = (V/f) / (Vn/fn)            (sans dimension)
//! ```
//!
//! `Vn` tension nominale (V), `fn` fréquence nominale (Hz), `k` rapport
//! Volts-par-Hertz nominal (V/Hz), `f` fréquence de consigne (Hz), `Vboost`
//! tension de compensation ajoutée à basse fréquence (V), `V` tension de sortie
//! du variateur (V), `φr` flux relatif rapporté au flux nominal (1 = flux
//! nominal, > 1 = surflux, < 1 = affaiblissement).
//!
//! **Convention** : tensions en volts (V), fréquences en hertz (Hz), rapport en
//! V/Hz, flux relatif sans dimension.
//! **Limite honnête** : commande **scalaire** V/f à flux **supposé constant** ;
//! la tension et la fréquence nominales sont des données de la machine
//! **fournies par l'appelant** (aucune valeur « par défaut » inventée), de même
//! que le *boost* bas régime qui compense la chute résistive statorique — chute
//! **négligée** ici au-delà de ce terme de boost. Ce module ne modélise ni le
//! couple (voir [`crate::motor_torque`]), ni le glissement (voir
//! [`crate::induction_motor`]), ni la zone d'affaiblissement de champ au-dessus
//! de la fréquence nominale.

/// Rapport V/f nominal `k = Vn / fn` (V/Hz).
///
/// Panique si `rated_voltage < 0` ou `rated_frequency <= 0`.
pub fn vf_ratio(rated_voltage: f64, rated_frequency: f64) -> f64 {
    assert!(
        rated_voltage >= 0.0,
        "la tension nominale doit être positive ou nulle"
    );
    assert!(
        rated_frequency > 0.0,
        "la fréquence nominale doit être strictement positive"
    );
    rated_voltage / rated_frequency
}

/// Tension de sortie `V = k·f + Vboost` (V) pour la loi V/f avec boost bas régime.
///
/// Panique si `vf_ratio < 0`, `frequency < 0` ou `boost_voltage < 0`.
pub fn vf_voltage_for_frequency(vf_ratio: f64, frequency: f64, boost_voltage: f64) -> f64 {
    assert!(vf_ratio >= 0.0, "le rapport V/f doit être positif ou nul");
    assert!(
        frequency >= 0.0,
        "la fréquence de consigne doit être positive ou nulle"
    );
    assert!(
        boost_voltage >= 0.0,
        "la tension de boost doit être positive ou nulle"
    );
    vf_ratio * frequency + boost_voltage
}

/// Flux relatif `φr = (V/f) / (Vn/fn)` (sans dimension), quasi constant en loi V/f.
///
/// Panique si `applied_frequency <= 0`, `rated_voltage < 0`, `rated_frequency <= 0`
/// ou `applied_voltage < 0`.
pub fn vf_flux_ratio(
    applied_voltage: f64,
    applied_frequency: f64,
    rated_voltage: f64,
    rated_frequency: f64,
) -> f64 {
    assert!(
        applied_voltage >= 0.0,
        "la tension appliquée doit être positive ou nulle"
    );
    assert!(
        applied_frequency > 0.0,
        "la fréquence appliquée doit être strictement positive"
    );
    assert!(
        rated_voltage >= 0.0,
        "la tension nominale doit être positive ou nulle"
    );
    assert!(
        rated_frequency > 0.0,
        "la fréquence nominale doit être strictement positive"
    );
    (applied_voltage / applied_frequency) / (rated_voltage / rated_frequency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rated_ratio_is_voltage_over_frequency() {
        // Machine 400 V / 50 Hz → k = 400/50 = 8 V/Hz.
        assert_relative_eq!(vf_ratio(400.0, 50.0), 8.0, epsilon = 1e-12);
        // 230 V / 50 Hz → 4,6 V/Hz.
        assert_relative_eq!(vf_ratio(230.0, 50.0), 4.6, epsilon = 1e-12);
    }

    #[test]
    fn voltage_law_recovers_rated_point_without_boost() {
        // Sans boost, la loi V/f rendue à fn redonne exactement Vn (réciprocité).
        let k = vf_ratio(400.0, 50.0);
        assert_relative_eq!(
            vf_voltage_for_frequency(k, 50.0, 0.0),
            400.0,
            epsilon = 1e-9
        );
        // À moitié de fréquence, tension moitié : 8·25 = 200 V.
        assert_relative_eq!(
            vf_voltage_for_frequency(k, 25.0, 0.0),
            200.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn boost_adds_offset_at_zero_frequency() {
        // À fréquence nulle, seule subsiste la tension de boost.
        let k = vf_ratio(400.0, 50.0);
        assert_relative_eq!(
            vf_voltage_for_frequency(k, 0.0, 15.0),
            15.0,
            epsilon = 1e-12
        );
        // Le boost est un simple décalage additif : 8·25 + 10 = 210 V.
        assert_relative_eq!(
            vf_voltage_for_frequency(k, 25.0, 10.0),
            210.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn flux_is_unity_along_pure_vf_line() {
        // Sur la droite V/f sans boost, le flux relatif reste à 1 (flux constant).
        let k = vf_ratio(400.0, 50.0);
        let v = vf_voltage_for_frequency(k, 30.0, 0.0);
        assert_relative_eq!(vf_flux_ratio(v, 30.0, 400.0, 50.0), 1.0, epsilon = 1e-12);
        // Au point nominal exact, flux relatif = 1 par définition.
        assert_relative_eq!(
            vf_flux_ratio(400.0, 50.0, 400.0, 50.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn boost_raises_flux_at_low_speed() {
        // 400 V/50 Hz, marche à 25 Hz avec boost 10 V → V = 210 V.
        // V/f appliqué = 210/25 = 8,4 V/Hz ; nominal = 8 V/Hz ; φr = 8,4/8 = 1,05.
        let k = vf_ratio(400.0, 50.0);
        let v = vf_voltage_for_frequency(k, 25.0, 10.0);
        assert_relative_eq!(v, 210.0, epsilon = 1e-9);
        assert_relative_eq!(vf_flux_ratio(v, 25.0, 400.0, 50.0), 1.05, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "fréquence nominale")]
    fn zero_rated_frequency_panics() {
        vf_ratio(400.0, 0.0);
    }
}
