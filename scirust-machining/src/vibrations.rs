//! Vibrations d'un système à **1 degré de liberté** (masse-ressort-amortisseur) —
//! pulsation propre, amortissement, régime libre et forcé.
//!
//! ```text
//! pulsation propre       ωn = √(k/m)          fn = ωn/2π
//! amortissement critique cc = 2·√(k·m) = 2·m·ωn
//! taux d'amortissement   ζ = c/cc
//! pulsation amortie      ωd = ωn·√(1 − ζ²)      (ζ < 1)
//! décrément logarithmique δ = 2π·ζ/√(1 − ζ²)
//! facteur d'amplification Q = 1/(2ζ)            (à la résonance)
//! ```
//!
//! `m` masse (kg), `k` raideur (N/m), `c` amortissement (N·s/m), `ζ` sans
//! dimension. `ζ < 1` sous-amorti (oscillant), `ζ = 1` critique, `ζ > 1`
//! sur-amorti.
//!
//! **Convention** : SI. **Limite honnête** : système linéaire à 1 ddl, paramètres
//! constants ; pas de non-linéarité, de plusieurs modes, ni d'amortissement
//! non visqueux.

use core::f64::consts::PI;

/// Pulsation propre non amortie `ωn = √(k/m)` (rad/s).
///
/// Panique si `m <= 0` ou `k < 0`.
pub fn natural_frequency_rad(k_n_m: f64, m_kg: f64) -> f64 {
    assert!(m_kg > 0.0 && k_n_m >= 0.0, "m > 0 et k ≥ 0 requis");
    (k_n_m / m_kg).sqrt()
}

/// Fréquence propre `fn = ωn/2π` (Hz).
pub fn natural_frequency_hz(k_n_m: f64, m_kg: f64) -> f64 {
    natural_frequency_rad(k_n_m, m_kg) / (2.0 * PI)
}

/// Amortissement critique `cc = 2·√(k·m)` (N·s/m).
pub fn critical_damping(k_n_m: f64, m_kg: f64) -> f64 {
    assert!(m_kg > 0.0 && k_n_m >= 0.0, "m > 0 et k ≥ 0 requis");
    2.0 * (k_n_m * m_kg).sqrt()
}

/// Taux d'amortissement `ζ = c / cc` (sans dimension).
pub fn damping_ratio(c_ns_m: f64, k_n_m: f64, m_kg: f64) -> f64 {
    c_ns_m / critical_damping(k_n_m, m_kg)
}

/// Pulsation propre amortie `ωd = ωn·√(1 − ζ²)` (rad/s), régime sous-amorti.
///
/// Panique si `ζ >= 1` (non oscillant).
pub fn damped_frequency_rad(omega_n_rad_s: f64, zeta: f64) -> f64 {
    assert!(zeta < 1.0, "système non oscillant (ζ ≥ 1)");
    omega_n_rad_s * (1.0 - zeta * zeta).sqrt()
}

/// Décrément logarithmique `δ = 2π·ζ/√(1 − ζ²)` (sans dimension).
///
/// Panique si `ζ >= 1`.
pub fn log_decrement(zeta: f64) -> f64 {
    assert!(zeta < 1.0, "décrément logarithmique défini pour ζ < 1");
    2.0 * PI * zeta / (1.0 - zeta * zeta).sqrt()
}

/// Facteur d'amplification à la résonance `Q = 1/(2ζ)` (sans dimension).
///
/// Panique si `ζ <= 0`.
pub fn quality_factor(zeta: f64) -> f64 {
    assert!(zeta > 0.0, "le facteur de qualité exige ζ > 0");
    1.0 / (2.0 * zeta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn natural_frequency_of_a_unit_system() {
        // k=100 N/m, m=1 kg → ωn = 10 rad/s, fn = 10/2π ≈ 1,592 Hz.
        assert_relative_eq!(natural_frequency_rad(100.0, 1.0), 10.0, epsilon = 1e-12);
        assert_relative_eq!(
            natural_frequency_hz(100.0, 1.0),
            10.0 / (2.0 * PI),
            epsilon = 1e-12
        );
    }

    #[test]
    fn damping_ratio_from_critical() {
        // k=100, m=1 → cc = 2·10 = 20 N·s/m. c=5 → ζ = 0,25.
        assert_relative_eq!(critical_damping(100.0, 1.0), 20.0, epsilon = 1e-12);
        assert_relative_eq!(damping_ratio(5.0, 100.0, 1.0), 0.25, epsilon = 1e-12);
    }

    #[test]
    fn damped_frequency_below_natural() {
        // ωn=10, ζ=0,25 → ωd = 10·√(1−0,0625) ≈ 9,682 rad/s.
        let wd = damped_frequency_rad(10.0, 0.25);
        assert!(wd < 10.0);
        assert_relative_eq!(wd, 10.0 * (1.0f64 - 0.0625).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn log_decrement_and_quality_factor() {
        assert_relative_eq!(
            log_decrement(0.25),
            2.0 * PI * 0.25 / (1.0f64 - 0.0625).sqrt(),
            epsilon = 1e-12
        );
        // Q = 1/(2·0,25) = 2.
        assert_relative_eq!(quality_factor(0.25), 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "non oscillant")]
    fn overdamped_has_no_damped_frequency() {
        damped_frequency_rad(10.0, 1.5);
    }
}
