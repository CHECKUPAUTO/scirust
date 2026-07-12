//! Réponse d'un système du **premier ordre** — réponse indicielle, constante de
//! temps, temps de réponse et fréquence de coupure.
//!
//! ```text
//! réponse indicielle y(t) = K·u·(1 − e^{−t/τ})
//! temps à x %        t = −τ·ln(1 − x)          (95 % → 3τ, 98 % → 4τ)
//! pulsation coupure  ω_c = 1/τ
//! fréquence coupure  f_c = 1/(2π·τ)
//! ```
//!
//! `K` gain statique, `u` amplitude de l'échelon d'entrée, `τ` constante de temps
//! (s), `t` instant (s), `x` fraction de la valeur finale (`∈ ]0, 1[`), `ω_c`
//! pulsation de coupure (rad/s), `f_c` fréquence de coupure (Hz). À `t = τ` la
//! sortie atteint 63,2 % de sa valeur finale.
//!
//! **Convention** : SI (temps en s, pulsations en rad/s). **Limite honnête** :
//! système **linéaire** du premier ordre (une seule constante de temps), sans
//! retard pur ni saturation ; `K` et `τ` proviennent d'une identification fournie
//! par l'appelant. Pour le second ordre, voir [`crate::second_order_response`].

use core::f64::consts::PI;

/// Réponse indicielle `y(t) = K·u·(1 − e^{−t/τ})`.
///
/// Panique si `tau <= 0` ou `time < 0`.
pub fn step_response(gain: f64, tau: f64, step_input: f64, time: f64) -> f64 {
    assert!(tau > 0.0 && time >= 0.0, "τ > 0 et t ≥ 0 requis");
    gain * step_input * (1.0 - (-time / tau).exp())
}

/// Temps pour atteindre une fraction `x` de la valeur finale
/// `t = −τ·ln(1 − x)`.
///
/// Panique si `tau <= 0` ou `fraction` hors `]0, 1[`.
pub fn time_to_fraction(tau: f64, fraction: f64) -> f64 {
    assert!(tau > 0.0, "τ doit être strictement positif");
    assert!(
        fraction > 0.0 && fraction < 1.0,
        "la fraction doit être dans ]0, 1["
    );
    -tau * (1.0 - fraction).ln()
}

/// Pulsation de coupure `ω_c = 1/τ` (rad/s).
///
/// Panique si `tau <= 0`.
pub fn corner_frequency_rad(tau: f64) -> f64 {
    assert!(tau > 0.0, "τ doit être strictement positif");
    1.0 / tau
}

/// Fréquence de coupure `f_c = 1/(2π·τ)` (Hz).
///
/// Panique si `tau <= 0`.
pub fn cutoff_frequency_hz(tau: f64) -> f64 {
    assert!(tau > 0.0, "τ doit être strictement positif");
    1.0 / (2.0 * PI * tau)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reaches_632_percent_at_one_tau() {
        // À t = τ, la sortie vaut 63,2 % de K·u.
        let y = step_response(2.0, 0.5, 10.0, 0.5);
        assert_relative_eq!(
            y,
            2.0 * 10.0 * (1.0 - 1.0 / core::f64::consts::E),
            epsilon = 1e-9
        );
        assert!(y > 0.63 * 20.0 && y < 0.633 * 20.0);
    }

    #[test]
    fn settling_times_are_multiples_of_tau() {
        // 95 % → ≈ 3τ ; 98 % → ≈ 4τ.
        let tau = 0.5;
        assert_relative_eq!(
            time_to_fraction(tau, 0.95),
            tau * (0.05_f64).ln().abs(),
            epsilon = 1e-9
        );
        assert!(time_to_fraction(tau, 0.95) > 2.9 * tau && time_to_fraction(tau, 0.95) < 3.1 * tau);
        assert!(time_to_fraction(tau, 0.98) > 3.9 * tau && time_to_fraction(tau, 0.98) < 4.0 * tau);
    }

    #[test]
    fn corner_and_cutoff_relate_by_two_pi() {
        // ω_c = 2π·f_c.
        let tau = 0.02;
        assert_relative_eq!(
            corner_frequency_rad(tau),
            2.0 * PI * cutoff_frequency_hz(tau),
            epsilon = 1e-9
        );
    }

    #[test]
    fn output_saturates_at_gain_times_input() {
        // t ≫ τ → y → K·u.
        assert!(step_response(2.0, 0.5, 10.0, 10.0) > 0.999 * 20.0);
    }

    #[test]
    #[should_panic(expected = "fraction")]
    fn fraction_one_panics() {
        time_to_fraction(0.5, 1.0);
    }
}
