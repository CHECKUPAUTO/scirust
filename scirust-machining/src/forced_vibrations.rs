//! Vibrations **forcées** d'un système à 1 ddl — réponse en régime permanent à
//! une excitation harmonique : facteur d'amplification, phase, transmissibilité
//! et réponse au balourd tournant.
//!
//! ```text
//! rapport de fréquence   r = ω/ωn
//! amplification (force)  M = 1/√((1−r²)² + (2ζr)²)
//! phase                  φ = atan2(2ζr, 1−r²)
//! transmissibilité       Tr = √(1 + (2ζr)²) / √((1−r²)² + (2ζr)²)
//! réponse au balourd     X/(m_b·e/m) = r² / √((1−r²)² + (2ζr)²)
//! pic de résonance       r_pic = √(1 − 2ζ²)   (ζ < 1/√2)
//! ```
//!
//! `ω` pulsation d'excitation, `ωn` pulsation propre, `r` rapport de fréquence,
//! `ζ` taux d'amortissement. `M` multiplie la flèche statique `F0/k` pour donner
//! l'amplitude dynamique ; `Tr` est le rapport force transmise/force appliquée
//! (isolation vibratoire).
//!
//! **Convention** : sans dimension sauf indication. **Limite honnête** : système
//! **linéaire** à 1 ddl en régime **harmonique permanent** établi ; pas de
//! transitoire, de non-linéarité, ni de plusieurs modes. Voir [`crate::vibrations`]
//! pour le régime libre.

/// Rapport de fréquence `r = ω/ωn` (sans dimension).
///
/// Panique si `omega_n <= 0`.
pub fn frequency_ratio(omega_rad_s: f64, omega_n_rad_s: f64) -> f64 {
    assert!(
        omega_n_rad_s > 0.0,
        "la pulsation propre doit être strictement positive"
    );
    omega_rad_s / omega_n_rad_s
}

/// Facteur d'amplification dynamique `M = 1/√((1−r²)² + (2ζr)²)`.
///
/// L'amplitude vaut `M·(F0/k)`. Panique si `zeta < 0`.
pub fn magnification_factor(r: f64, zeta: f64) -> f64 {
    assert!(zeta >= 0.0, "le taux d'amortissement doit être ≥ 0");
    let a = 1.0 - r * r;
    let b = 2.0 * zeta * r;
    1.0 / (a * a + b * b).sqrt()
}

/// Déphasage réponse/excitation `φ = atan2(2ζr, 1−r²)` (rad, dans `[0, π]`).
pub fn phase_lag_rad(r: f64, zeta: f64) -> f64 {
    (2.0 * zeta * r).atan2(1.0 - r * r)
}

/// Transmissibilité `Tr = √(1 + (2ζr)²)/√((1−r²)² + (2ζr)²)` (sans dimension).
///
/// Rapport de la force transmise au support sur la force appliquée. Panique si
/// `zeta < 0`.
pub fn transmissibility(r: f64, zeta: f64) -> f64 {
    assert!(zeta >= 0.0, "le taux d'amortissement doit être ≥ 0");
    let b = 2.0 * zeta * r;
    let a = 1.0 - r * r;
    (1.0 + b * b).sqrt() / (a * a + b * b).sqrt()
}

/// Amplitude adimensionnée de la réponse à un **balourd tournant**
/// `X·m/(m_b·e) = r²/√((1−r²)² + (2ζr)²)`.
///
/// `m` masse totale, `m_b` masse balourd, `e` excentricité ; multiplier le
/// résultat par `m_b·e/m` pour l'amplitude physique. Panique si `zeta < 0`.
pub fn rotating_unbalance_response(r: f64, zeta: f64) -> f64 {
    assert!(zeta >= 0.0, "le taux d'amortissement doit être ≥ 0");
    let a = 1.0 - r * r;
    let b = 2.0 * zeta * r;
    r * r / (a * a + b * b).sqrt()
}

/// Rapport de fréquence au **pic de résonance** en amplitude
/// `r_pic = √(1 − 2ζ²)`, défini pour `ζ < 1/√2`.
///
/// Panique si `ζ ≥ 1/√2` (pas de pic : réponse monotone décroissante).
pub fn resonance_peak_ratio(zeta: f64) -> f64 {
    let arg = 1.0 - 2.0 * zeta * zeta;
    assert!(arg > 0.0, "pas de pic de résonance pour ζ ≥ 1/√2");
    arg.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_2;

    #[test]
    fn magnification_equals_q_at_resonance_ratio_one() {
        // À r=1 : M = 1/(2ζ) = Q. Pour ζ=0,1 → M=5.
        assert_relative_eq!(magnification_factor(1.0, 0.1), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn static_limit_and_high_frequency() {
        // r→0 : M→1 (réponse quasi statique). r≫1 : M→0.
        assert_relative_eq!(magnification_factor(0.0, 0.1), 1.0, epsilon = 1e-12);
        assert!(magnification_factor(5.0, 0.1) < 0.05);
    }

    #[test]
    fn phase_is_ninety_degrees_at_resonance() {
        // À r=1 la phase vaut exactement π/2 quel que soit ζ.
        assert_relative_eq!(phase_lag_rad(1.0, 0.05), FRAC_PI_2, epsilon = 1e-12);
        assert_relative_eq!(phase_lag_rad(1.0, 0.5), FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn transmissibility_unit_at_root_two() {
        // À r=√2, Tr=1 pour tout ζ (point de croisement des courbes d'isolation).
        assert_relative_eq!(transmissibility(2.0f64.sqrt(), 0.1), 1.0, epsilon = 1e-12);
        assert_relative_eq!(transmissibility(2.0f64.sqrt(), 0.4), 1.0, epsilon = 1e-12);
        // Isolation (Tr<1) seulement au-delà de r=√2.
        assert!(transmissibility(3.0, 0.1) < 1.0);
    }

    #[test]
    fn unbalance_response_grows_then_saturates() {
        // r→0 : réponse →0 ; r≫1 : réponse →1 (amplitude → m_b·e/m).
        assert_relative_eq!(rotating_unbalance_response(0.0, 0.1), 0.0, epsilon = 1e-12);
        assert!(rotating_unbalance_response(10.0, 0.1) > 0.98);
    }

    #[test]
    fn resonance_peak_below_natural_frequency() {
        // ζ=0,1 → r_pic = √(1−0,02) ≈ 0,99 < 1.
        let rp = resonance_peak_ratio(0.1);
        assert!(rp < 1.0);
        assert_relative_eq!(rp, (1.0f64 - 0.02).sqrt(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pas de pic")]
    fn no_peak_when_overdamped_enough() {
        resonance_peak_ratio(0.8); // > 1/√2 ≈ 0,707
    }
}
