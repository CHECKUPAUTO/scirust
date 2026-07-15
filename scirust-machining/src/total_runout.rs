//! Métrologie — **battement** (runout) : dépouillement d'un relevé au comparateur
//! (battement total indiqué, battement simple, contrôle de tolérance).
//!
//! ```text
//! TIR (total indicated runout) TIR = max(readings) − min(readings)
//! battement simple (circular)  CR  = max_reading − min_reading
//! conformité                   ok  = TIR ≤ tolérance
//! ```
//!
//! `readings` relevés successifs du palpeur du comparateur (m, SI), `max`/`min`
//! extrêmes du relevé (m), `TIR`/`CR` étendue lue = battement (m), `tolérance`
//! battement admissible (m), `ok` conformité (booléen).
//!
//! **Convention** : toutes les longueurs en **mètres** (SI) ; l'appelant convertit
//! les centièmes de millimètre du comparateur (`1 c/100 = 1·10⁻⁵ m`). **Limite
//! honnête** : le battement **additionne** les défauts de forme et d'excentricité —
//! il ne les sépare pas ; le relevé (nombre de points, résolution du comparateur au
//! centième, référentiel de rotation) est **fourni** par l'appelant, aucune valeur
//! de tolérance ou de précision n'est inventée ici.

/// Battement total indiqué `TIR = max(readings) − min(readings)`.
///
/// Étendue (m) d'un relevé complet au comparateur sur un tour ou un balayage.
///
/// Panique si `readings` est vide.
pub fn tir_total_indicated_runout(readings: &[f64]) -> f64 {
    assert!(
        !readings.is_empty(),
        "le relevé au comparateur ne doit pas être vide"
    );
    let max = readings.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = readings.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

/// Battement simple `CR = max_reading − min_reading` à partir des deux extrêmes.
///
/// Panique si `max_reading < min_reading`.
pub fn runout_circular(max_reading: f64, min_reading: f64) -> f64 {
    assert!(
        max_reading >= min_reading,
        "la lecture maximale doit être supérieure ou égale à la minimale"
    );
    max_reading - min_reading
}

/// Conformité `ok = TIR ≤ tolérance` du battement d'un relevé.
///
/// Panique si `readings` est vide ou si `tolerance < 0`.
pub fn runout_is_within(readings: &[f64], tolerance: f64) -> bool {
    assert!(
        tolerance >= 0.0,
        "la tolérance de battement doit être positive ou nulle"
    );
    tir_total_indicated_runout(readings) <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tir_equals_span_of_readings() {
        // TIR = max − min, indépendant de l'ordre des points.
        let readings = [0.010e-3, 0.032e-3, 0.021e-3, 0.005e-3];
        assert_relative_eq!(
            tir_total_indicated_runout(&readings),
            0.032e-3 - 0.005e-3,
            max_relative = 1e-12
        );
    }

    #[test]
    fn tir_and_circular_agree_on_extremes() {
        // Le battement simple des extrêmes reproduit le TIR du relevé.
        let readings = [0.008e-3, 0.041e-3, 0.017e-3];
        assert_relative_eq!(
            tir_total_indicated_runout(&readings),
            runout_circular(0.041e-3, 0.008e-3),
            max_relative = 1e-12
        );
    }

    #[test]
    fn constant_reading_gives_zero_runout() {
        // Un relevé parfaitement constant → battement nul (cas limite).
        assert_relative_eq!(
            tir_total_indicated_runout(&[0.015e-3; 8]),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn offset_does_not_change_runout() {
        // Le battement est une étendue : invariant par translation du relevé.
        let base = [0.004e-3, 0.030e-3, 0.012e-3];
        let shifted: Vec<f64> = base.iter().map(|r| r + 0.100e-3).collect();
        assert_relative_eq!(
            tir_total_indicated_runout(&base),
            tir_total_indicated_runout(&shifted),
            max_relative = 1e-12
        );
    }

    #[test]
    fn tolerance_check_realistic_case() {
        // Comparateur au centième : relevé de 25 c/100 (0,25 mm) d'étendue.
        // Étendue 0,25 mm > tolérance 0,20 mm → non conforme ;
        //                  ≤ tolérance 0,30 mm → conforme.
        let readings = [0.05e-3, 0.30e-3, 0.18e-3];
        assert!(!runout_is_within(&readings, 0.20e-3));
        assert!(runout_is_within(&readings, 0.30e-3));
    }

    #[test]
    #[should_panic(expected = "ne doit pas être vide")]
    fn empty_readings_panics() {
        tir_total_indicated_runout(&[]);
    }
}
