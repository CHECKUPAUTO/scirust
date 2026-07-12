//! Diagramme de **Bode** d'un premier ordre `K/(1 + jωτ)` — gain (dB), phase et
//! pulsation de coupure.
//!
//! ```text
//! gain          |G| = K/√(1 + (ωτ)²)
//! gain (dB)     G_dB = 20·log₁₀(K) − 10·log₁₀(1 + (ωτ)²)
//! phase         φ = −arctan(ωτ)          (rad puis converti en °)
//! coupure       ω_c = 1/τ    →    G_dB(ω_c) = 20·log₁₀(K) − 3,01 dB, φ = −45°
//! ```
//!
//! `K` gain statique, `τ` constante de temps (s), `ω` pulsation (rad/s), `G_dB`
//! gain en décibels, `φ` phase (°). À la coupure `ω_c = 1/τ`, le gain chute de
//! **3 dB** et la phase vaut **−45°**.
//!
//! **Convention** : pulsations en rad/s, phase en degrés. **Limite honnête** :
//! premier ordre **stable** à minimum de phase (`τ > 0`) ; ne couvre ni les
//! zéros, ni les retards purs (phase supplémentaire `−ωL`), ni le second ordre.
//! `K` et `τ` sont fournis par l'appelant.

/// Conversion linéaire → décibels `20·log₁₀(x)`.
///
/// Panique si `ratio <= 0`.
pub fn decibels(ratio: f64) -> f64 {
    assert!(ratio > 0.0, "le rapport doit être strictement positif");
    20.0 * ratio.log10()
}

/// Gain en décibels `G_dB = 20·log₁₀(K) − 10·log₁₀(1 + (ωτ)²)`.
///
/// Panique si `gain <= 0`, `tau <= 0` ou `omega < 0`.
pub fn magnitude_db(gain: f64, tau: f64, omega: f64) -> f64 {
    assert!(
        gain > 0.0 && tau > 0.0 && omega >= 0.0,
        "K, τ > 0 et ω ≥ 0 requis"
    );
    let wt = omega * tau;
    20.0 * gain.log10() - 10.0 * (1.0 + wt * wt).log10()
}

/// Phase `φ = −arctan(ωτ)` (degrés).
///
/// Panique si `tau <= 0` ou `omega < 0`.
pub fn phase_deg(tau: f64, omega: f64) -> f64 {
    assert!(tau > 0.0 && omega >= 0.0, "τ > 0 et ω ≥ 0 requis");
    -(omega * tau).atan().to_degrees()
}

/// Pulsation de coupure `ω_c = 1/τ` (rad/s).
///
/// Panique si `tau <= 0`.
pub fn corner_frequency_rad(tau: f64) -> f64 {
    assert!(tau > 0.0, "τ doit être strictement positif");
    1.0 / tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dc_gain_in_db() {
        // À ω=0 : G_dB = 20·log₁₀(K). K=10 → 20 dB.
        assert_relative_eq!(magnitude_db(10.0, 0.5, 0.0), 20.0, epsilon = 1e-9);
        assert_relative_eq!(decibels(10.0), 20.0, epsilon = 1e-9);
    }

    #[test]
    fn three_db_drop_at_corner() {
        // À ω = ω_c = 1/τ : gain chute de 3,01 dB sous le gain continu.
        let tau = 0.5;
        let wc = corner_frequency_rad(tau);
        let drop = magnitude_db(1.0, tau, 0.0) - magnitude_db(1.0, tau, wc);
        assert_relative_eq!(drop, 10.0 * 2.0_f64.log10(), epsilon = 1e-9);
        assert!(drop > 3.0 && drop < 3.02);
    }

    #[test]
    fn phase_is_minus_45_at_corner() {
        // À la coupure, la phase vaut exactement −45°.
        let tau = 0.5;
        assert_relative_eq!(
            phase_deg(tau, corner_frequency_rad(tau)),
            -45.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn phase_tends_to_minus_90() {
        // ω ≫ ω_c → φ → −90°.
        assert!(phase_deg(0.5, 1e6) < -89.9);
    }

    #[test]
    fn high_frequency_rolloff_20db_per_decade() {
        // Bien au-dessus de la coupure, l'asymptote est de −20 dB par décade.
        let tau = 0.5;
        let wc = corner_frequency_rad(tau);
        let g1 = magnitude_db(1.0, tau, 100.0 * wc);
        let g2 = magnitude_db(1.0, tau, 1000.0 * wc);
        assert_relative_eq!(g1 - g2, 20.0, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "strictement positif")]
    fn zero_gain_db_panics() {
        decibels(0.0);
    }
}
