//! Loi du déplacement de **Wien** — position du maximum d'émission d'un corps
//! noir en fonction de sa température absolue (base de la pyrométrie optique).
//!
//! ```text
//! longueur d'onde du pic   lambda_max = b / T
//! température (réciproque)  T          = b / lambda_max
//! fréquence du pic          nu_max     = b_freq · T
//! ```
//!
//! `T` température **absolue** (K), `lambda_max` longueur d'onde du maximum
//! spectral (m), `nu_max` fréquence du maximum spectral (Hz), `b` constante de
//! déplacement de Wien (m·K), `b_freq` constante de Wien en fréquence (Hz/K).
//!
//! **Convention** : températures en **kelvin**, unités **SI**. **Limite
//! honnête** : ces relations décrivent le **corps noir** (loi de Planck) ; le
//! maximum en longueur d'onde et le maximum en fréquence ne correspondent PAS à
//! la même raie (`lambda_max·nu_max ≠ c`). L'émissivité réelle d'un matériau
//! décale l'application en pyrométrie : la correction d'émissivité est à la
//! charge de l'appelant. Complète [`crate::radiation`].

/// Constante de déplacement de Wien `b` (m·K).
pub const WIEN_DISPLACEMENT_CONSTANT: f64 = 2.897_771_955e-3;

/// Constante de Wien en fréquence `b_freq` (Hz/K).
pub const WIEN_FREQUENCY_CONSTANT: f64 = 5.878_925_757e10;

/// Longueur d'onde du maximum d'émission `lambda_max = b / T` (m).
///
/// Panique si `temperature_kelvin <= 0`.
pub fn wien_peak_wavelength(temperature_kelvin: f64) -> f64 {
    assert!(
        temperature_kelvin > 0.0,
        "la température absolue doit être strictement positive (kelvin)"
    );
    WIEN_DISPLACEMENT_CONSTANT / temperature_kelvin
}

/// Température déduite du pic spectral `T = b / lambda_max` (K), réciproque de
/// [`wien_peak_wavelength`] utilisée en pyrométrie.
///
/// Panique si `peak_wavelength <= 0`.
pub fn wien_temperature_from_peak(peak_wavelength: f64) -> f64 {
    assert!(
        peak_wavelength > 0.0,
        "la longueur d'onde du pic doit être strictement positive (m)"
    );
    WIEN_DISPLACEMENT_CONSTANT / peak_wavelength
}

/// Fréquence du maximum d'émission `nu_max = b_freq · T` (Hz).
///
/// Panique si `temperature_kelvin <= 0`.
pub fn wien_peak_frequency(temperature_kelvin: f64) -> f64 {
    assert!(
        temperature_kelvin > 0.0,
        "la température absolue doit être strictement positive (kelvin)"
    );
    WIEN_FREQUENCY_CONSTANT * temperature_kelvin
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn wavelength_at_1000k_equals_constant_shifted() {
        // lambda_max = b / 1000 = 2,897771955e-3 / 1000 = 2,897771955e-6 m
        // (≈ 2,9 µm, infrarouge moyen — cohérent physiquement).
        assert_relative_eq!(
            wien_peak_wavelength(1000.0),
            2.897_771_955e-6,
            epsilon = 1e-18
        );
    }

    #[test]
    fn temperature_is_reciprocal_of_wavelength() {
        // T = b / lambda_max doit inverser lambda_max = b / T.
        let t = 5772.0;
        assert_relative_eq!(
            wien_temperature_from_peak(wien_peak_wavelength(t)),
            t,
            epsilon = 1e-9
        );
    }

    #[test]
    fn wavelength_scales_inversely_with_temperature() {
        // Doubler T divise lambda_max par deux (loi de déplacement).
        assert_relative_eq!(
            wien_peak_wavelength(600.0),
            2.0_f64 * wien_peak_wavelength(1200.0),
            epsilon = 1e-15
        );
    }

    #[test]
    fn peak_frequency_scales_linearly_with_temperature() {
        // nu_max ∝ T : le rapport à deux températures vaut le rapport des T.
        assert_relative_eq!(
            wien_peak_frequency(3000.0),
            3.0_f64 * wien_peak_frequency(1000.0),
            epsilon = 1e-3
        );
    }

    #[test]
    fn peak_frequency_at_1000k() {
        // nu_max = b_freq · 1000 = 5,878925757e10 · 1000 = 5,878925757e13 Hz.
        assert_relative_eq!(wien_peak_frequency(1000.0), 5.878_925_757e13, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "la température absolue doit être strictement positive")]
    fn zero_temperature_panics() {
        let _ = wien_peak_wavelength(0.0);
    }
}
