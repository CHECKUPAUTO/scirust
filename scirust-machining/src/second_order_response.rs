//! Réponse d'un système du **second ordre** (sous-amorti) — pulsation amortie,
//! dépassement, temps de pic et temps de réponse.
//!
//! ```text
//! pulsation amortie  ω_d = ω_n·√(1 − ζ²)
//! dépassement        D = exp(−π·ζ/√(1 − ζ²))            (fraction, 0 ≤ ζ < 1)
//! temps de pic       t_p = π/ω_d
//! temps de réponse   t_s ≈ 4/(ζ·ω_n)                     (bande ±2 %)
//! ```
//!
//! `ω_n` pulsation propre non amortie (rad/s), `ζ` coefficient d'amortissement
//! (`0 ≤ ζ < 1` en régime sous-amorti), `ω_d` pulsation propre amortie (rad/s),
//! `D` dépassement relatif (fraction du régime établi), `t_p` instant du premier
//! pic (s), `t_s` temps de réponse à ±2 % (s).
//!
//! **Convention** : SI. **Limite honnête** : régime **sous-amorti** (`ζ < 1`,
//! sinon pas de dépassement ni d'oscillation) ; `t_s ≈ 4/(ζ·ω_n)` est
//! l'approximation usuelle de l'enveloppe exponentielle (valable pour `ζ`
//! modéré). `ω_n` et `ζ` proviennent d'une identification fournie par
//! l'appelant. Voir aussi [`crate::vibrations`] (mécanique) et
//! [`crate::first_order_response`].

use core::f64::consts::PI;

/// Pulsation propre amortie `ω_d = ω_n·√(1 − ζ²)`.
///
/// Panique si `natural_frequency <= 0` ou `damping_ratio` hors `[0, 1[`.
pub fn damped_frequency(natural_frequency: f64, damping_ratio: f64) -> f64 {
    assert!(natural_frequency > 0.0, "ω_n doit être strictement positif");
    assert!(
        (0.0..1.0).contains(&damping_ratio),
        "ζ doit être dans [0, 1["
    );
    natural_frequency * (1.0 - damping_ratio * damping_ratio).sqrt()
}

/// Dépassement relatif `D = exp(−π·ζ/√(1 − ζ²))` (fraction du régime établi).
///
/// Panique si `damping_ratio` hors `[0, 1[`.
pub fn overshoot(damping_ratio: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&damping_ratio),
        "ζ doit être dans [0, 1["
    );
    (-PI * damping_ratio / (1.0 - damping_ratio * damping_ratio).sqrt()).exp()
}

/// Temps de pic `t_p = π/ω_d`.
///
/// Panique si `natural_frequency <= 0` ou `damping_ratio` hors `[0, 1[`.
pub fn peak_time(natural_frequency: f64, damping_ratio: f64) -> f64 {
    PI / damped_frequency(natural_frequency, damping_ratio)
}

/// Temps de réponse à ±2 % `t_s ≈ 4/(ζ·ω_n)`.
///
/// Panique si `natural_frequency <= 0` ou `damping_ratio` hors `]0, 1[`.
pub fn settling_time_2pct(natural_frequency: f64, damping_ratio: f64) -> f64 {
    assert!(natural_frequency > 0.0, "ω_n doit être strictement positif");
    assert!(
        damping_ratio > 0.0 && damping_ratio < 1.0,
        "ζ doit être dans ]0, 1["
    );
    4.0 / (damping_ratio * natural_frequency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn undamped_has_full_overshoot_and_equal_frequencies() {
        // ζ=0 : dépassement = 1 (100 %) et ω_d = ω_n.
        assert_relative_eq!(overshoot(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(damped_frequency(10.0, 0.0), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn classic_damping_gives_known_overshoot() {
        // ζ=0,5 → dépassement ≈ 16,3 % ; ζ=0,707 → ≈ 4,3 %.
        assert!(overshoot(0.5) > 0.162 && overshoot(0.5) < 0.164);
        assert!(overshoot(0.707) > 0.04 && overshoot(0.707) < 0.045);
    }

    #[test]
    fn more_damping_reduces_overshoot() {
        assert!(overshoot(0.7) < overshoot(0.3));
    }

    #[test]
    fn peak_and_settling_times() {
        // ω_n=10, ζ=0,5 : ω_d=8,66 → t_p=π/8,66≈0,363 s ; t_s=4/(0,5·10)=0,8 s.
        assert_relative_eq!(
            peak_time(10.0, 0.5),
            PI / (10.0 * (0.75_f64).sqrt()),
            epsilon = 1e-9
        );
        assert_relative_eq!(settling_time_2pct(10.0, 0.5), 0.8, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ζ doit être dans [0, 1[")]
    fn overdamped_overshoot_panics() {
        overshoot(1.2);
    }
}
