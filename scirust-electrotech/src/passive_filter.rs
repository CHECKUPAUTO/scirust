//! **Filtre passif du premier ordre (RC/RL)** — module de fréquence de coupure,
//! module du gain, gain en décibels et déphasage d'un filtre passe-bas passif
//! non chargé du premier ordre.
//!
//! ```text
//! coupure RC        f_c = 1 / (2·π·R·C)
//! coupure RL        f_c = R / (2·π·L)
//! module du gain    |H| = 1 / √(1 + (f/f_c)²)
//! gain en décibels  G_dB = 20·log₁₀(|H|)
//! déphasage         φ   = −atan(f/f_c)
//! ```
//!
//! `R` résistance (Ω), `C` capacité (F), `L` inductance (H), `f_c` fréquence de
//! coupure à −3 dB (Hz), `f` fréquence de travail (Hz), `|H|` module du gain en
//! tension (sans dimension, ∈ ]0 ; 1]), `G_dB` gain exprimé en décibels (dB),
//! `φ` déphasage entrée→sortie (radians, négatif pour un passe-bas : la sortie
//! est en retard sur l'entrée).
//!
//! **Convention** : SI ; résistances en Ω, capacités en F, inductances en H,
//! fréquences en Hz, angle `φ` en **radians**. **Limite honnête** : filtre
//! **passif du premier ordre non chargé** (impédance de charge supposée
//! **infinie**, aucun effet de source) ; le passe-haut échange les rôles
//! entrée/sortie et reste **à la charge de l'appelant** ; la fréquence de
//! coupure est définie **à −3 dB** et le roll-off asymptotique vaut **20 dB par
//! décade**. Les valeurs de composants (`R`, `L`, `C`) et la fréquence de
//! travail `f` sont **fournies par l'appelant** (fiches composant, mesures) —
//! aucune valeur « par défaut » n'est inventée.

/// Fréquence de coupure d'un filtre RC `f_c = 1 / (2·π·R·C)` (Hz).
///
/// Panique si `resistance <= 0` ou si `capacitance <= 0`.
pub fn filt_rc_cutoff_frequency(resistance: f64, capacitance: f64) -> f64 {
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    1.0 / (2.0 * core::f64::consts::PI * resistance * capacitance)
}

/// Fréquence de coupure d'un filtre RL `f_c = R / (2·π·L)` (Hz).
///
/// Panique si `resistance <= 0` ou si `inductance <= 0`.
pub fn filt_rl_cutoff_frequency(resistance: f64, inductance: f64) -> f64 {
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    resistance / (2.0 * core::f64::consts::PI * inductance)
}

/// Module du gain d'un passe-bas du premier ordre
/// `|H| = 1 / √(1 + (f/f_c)²)` (sans dimension).
///
/// Panique si `frequency < 0` ou si `cutoff_frequency <= 0`.
pub fn filt_first_order_gain(frequency: f64, cutoff_frequency: f64) -> f64 {
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0");
    assert!(
        cutoff_frequency > 0.0,
        "la fréquence de coupure f_c doit être strictement positive"
    );
    let ratio = frequency / cutoff_frequency;
    1.0 / (1.0 + ratio.powi(2)).sqrt()
}

/// Gain exprimé en décibels `G_dB = 20·log₁₀(|H|)` (dB).
///
/// Panique si `gain <= 0` (le logarithme n'est pas défini).
pub fn filt_gain_decibels(gain: f64) -> f64 {
    assert!(
        gain > 0.0,
        "le module du gain doit être strictement positif"
    );
    20.0 * gain.log10()
}

/// Déphasage d'un passe-bas RC `φ = −atan(f/f_c)` (radians).
///
/// Panique si `frequency < 0` ou si `cutoff_frequency <= 0`.
pub fn filt_phase_shift(frequency: f64, cutoff_frequency: f64) -> f64 {
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0");
    assert!(
        cutoff_frequency > 0.0,
        "la fréquence de coupure f_c doit être strictement positive"
    );
    -(frequency / cutoff_frequency).atan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rc_and_rl_cutoff_coincide_when_l_equals_r2c() {
        // Identité de réciprocité : f_c(RC) = 1/(2πRC) et f_c(RL) = R/(2πL)
        // coïncident dès que L = R²·C (car 1/(RC) = R/L ⇔ L = R²·C).
        let r = 1000.0_f64;
        let c = 1.0e-6_f64;
        let l = r * r * c; // 0,1 H
        let f_rc = filt_rc_cutoff_frequency(r, c);
        let f_rl = filt_rl_cutoff_frequency(r, l);
        assert_relative_eq!(f_rc, f_rl, epsilon = 1e-9);
    }

    #[test]
    fn gain_is_one_over_sqrt2_at_cutoff() {
        // Cas limite : à f = f_c le module du gain vaut 1/√2 ≈ 0,707 106 781.
        let f_c = 500.0_f64;
        let gain = filt_first_order_gain(f_c, f_c);
        assert_relative_eq!(gain, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn gain_decibels_is_minus_three_at_cutoff() {
        // La coupure est définie à −3 dB : 20·log₁₀(1/√2) = −10·log₁₀(2)
        //   = −10·0,301 029 995… = −3,010 299 957… dB.
        let f_c = 1000.0_f64;
        let gain = filt_first_order_gain(f_c, f_c);
        let g_db = filt_gain_decibels(gain);
        assert_relative_eq!(g_db, -3.010_299_957, epsilon = 1e-6);
    }

    #[test]
    fn rolloff_is_twenty_db_per_decade_asymptotically() {
        // Roll-off asymptotique : à une décade au-dessus de la coupure
        //   |H| = 1/√(1 + 10²) = 1/√101 → 20·log₁₀(1/√101)
        //   = −10·log₁₀(101) = −20,043 213 7… dB (≈ −20 dB/décade).
        let f_c = 200.0_f64;
        let gain = filt_first_order_gain(10.0 * f_c, f_c);
        let g_db = filt_gain_decibels(gain);
        assert_relative_eq!(g_db, -20.043_213_7, epsilon = 1e-4);
    }

    #[test]
    fn phase_shift_is_minus_quarter_pi_at_cutoff() {
        // À f = f_c le déphasage vaut −atan(1) = −π/4 ≈ −0,785 398 163 rad.
        let f_c = 750.0_f64;
        let phi = filt_phase_shift(f_c, f_c);
        assert_relative_eq!(phi, -core::f64::consts::FRAC_PI_4, epsilon = 1e-12);
    }

    #[test]
    fn realistic_rc_low_pass_case() {
        // Cas chiffré réaliste, R = 1 kΩ, C = 1 µF :
        //   f_c = 1/(2π·1000·1e-6) = 1/(2π·1e-3) ≈ 159,154 943 Hz.
        // À la coupure, gain = 1/√2 et déphasage = −π/4.
        let r = 1000.0_f64;
        let c = 1.0e-6_f64;
        let f_c = filt_rc_cutoff_frequency(r, c);
        assert_relative_eq!(f_c, 159.154_943, epsilon = 1e-3);
        assert_relative_eq!(
            filt_first_order_gain(f_c, f_c),
            core::f64::consts::FRAC_1_SQRT_2,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            filt_phase_shift(f_c, f_c),
            -core::f64::consts::FRAC_PI_4,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "la capacité C doit être strictement positive")]
    fn zero_capacitance_cutoff_panics() {
        filt_rc_cutoff_frequency(1000.0, 0.0);
    }
}
