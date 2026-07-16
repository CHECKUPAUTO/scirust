//! **Onduleur de tension (VSI, pont monophasé)** — valeurs efficaces d'un
//! créneau, efficace de son fondamental, crête du fondamental en MLI
//! sinus-triangle et indice de modulation.
//!
//! ```text
//! efficace créneau (pont complet)   V_rms      = V_dc
//! efficace du fondamental (créneau) V_1,rms    = 4 · V_dc / (π · √2)
//! crête du fondamental (MLI, m≤1)   V̂_1        = m · V_dc / 2
//! indice de modulation              m          = V̂_ref / V̂_carrier
//! ```
//!
//! `V_dc` tension du bus continu (V), `V_rms` valeur efficace de la tension de
//! sortie en pleine onde (créneau, V), `V_1,rms` valeur efficace du fondamental
//! du créneau (V), `V̂_1` amplitude (crête) du fondamental en MLI sinus-triangle
//! (V), `m` indice de modulation (sans dimension, `0 ≤ m ≤ 1` en zone linéaire),
//! `V̂_ref` amplitude de la modulante sinusoïdale (V), `V̂_carrier` amplitude de
//! la porteuse triangulaire (V). Réciprocité : la crête du fondamental d'un
//! créneau vaut `4 · V_dc / π`, donc son efficace `V_1,rms = (4 · V_dc / π) / √2`.
//!
//! **Convention** : SI ; tensions en V, indices sans dimension. Angles en
//! radians (aucun n'apparaît explicitement ici). **Limite honnête** : onduleur
//! de **tension** en pont, interrupteurs **idéaux** (chutes directes, temps
//! morts et pertes de commutation négligés) ; la MLI sinus-triangle est décrite
//! en **zone linéaire** (indice de modulation `m ≤ 1`) — au-delà (surmodulation),
//! ces formules ne tiennent plus. La tension de bus `V_dc`, l'indice de
//! modulation `m` et les amplitudes de modulante/porteuse sont **fournis par
//! l'appelant** (réseau continu, commande, dimensionnement) — aucune valeur
//! « par défaut » n'est inventée.

use core::f64::consts::{PI, SQRT_2};

/// Valeur efficace de la tension de sortie d'un pont complet en pleine onde
/// (créneau) `V_rms = V_dc` (V).
///
/// Panique si `dc_voltage < 0`.
pub fn inv_square_wave_rms(dc_voltage: f64) -> f64 {
    assert!(
        dc_voltage >= 0.0,
        "la tension de bus continu V_dc doit être ≥ 0"
    );
    dc_voltage
}

/// Valeur efficace du fondamental d'un créneau (pont complet)
/// `V_1,rms = 4 · V_dc / (π · √2)` (V).
///
/// Panique si `dc_voltage < 0`.
pub fn inv_square_wave_fundamental_rms(dc_voltage: f64) -> f64 {
    assert!(
        dc_voltage >= 0.0,
        "la tension de bus continu V_dc doit être ≥ 0"
    );
    4.0 * dc_voltage / (PI * SQRT_2)
}

/// Amplitude (crête) du fondamental en MLI sinus-triangle sur un demi-pont, en
/// zone linéaire `V̂_1 = m · V_dc / 2` (V).
///
/// Panique si `dc_voltage < 0` ou si `modulation_index` n'est pas dans `[0, 1]`
/// (zone linéaire ; au-delà, surmodulation).
pub fn inv_sine_pwm_fundamental_peak(dc_voltage: f64, modulation_index: f64) -> f64 {
    assert!(
        dc_voltage >= 0.0,
        "la tension de bus continu V_dc doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&modulation_index),
        "l'indice de modulation m doit être dans [0, 1] (zone linéaire)"
    );
    modulation_index * dc_voltage / 2.0
}

/// Indice de modulation `m = V̂_ref / V̂_carrier` (sans dimension).
///
/// Panique si `reference_peak < 0` ou si `carrier_peak <= 0`.
pub fn inv_modulation_index(reference_peak: f64, carrier_peak: f64) -> f64 {
    assert!(
        reference_peak >= 0.0,
        "l'amplitude de la modulante V̂_ref doit être ≥ 0"
    );
    assert!(
        carrier_peak > 0.0,
        "l'amplitude de la porteuse V̂_carrier doit être strictement positive"
    );
    reference_peak / carrier_peak
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn square_wave_rms_equals_dc_voltage() {
        // Identité : en pleine onde (pont complet), la valeur efficace du
        // créneau vaut exactement V_dc.
        let v_dc = 311.0_f64;
        assert_relative_eq!(inv_square_wave_rms(v_dc), v_dc, epsilon = 1e-12);
    }

    #[test]
    fn fundamental_rms_relates_to_fundamental_peak() {
        // Réciprocité crête/efficace : la crête du fondamental d'un créneau
        // vaut 4·V_dc/π, donc V_1,rms · √2 = 4·V_dc/π.
        let v_dc = 200.0_f64;
        let peak = 4.0 * v_dc / PI;
        assert_relative_eq!(
            inv_square_wave_fundamental_rms(v_dc) * SQRT_2,
            peak,
            epsilon = 1e-9
        );
    }

    #[test]
    fn fundamental_rms_reference_case() {
        // Cas chiffré : V_dc = 100 V.
        //   V_1,rms = 4 · 100 / (π · √2)
        //           = 400 / (3,141_592_653_6 · 1,414_213_562_4)
        //           = 400 / 4,442_882_938_2
        //           = 90,031_631_6 V.
        assert_relative_eq!(
            inv_square_wave_fundamental_rms(100.0),
            90.031_631_6,
            epsilon = 1e-6
        );
    }

    #[test]
    fn sine_pwm_peak_scales_linearly() {
        // Proportionnalité : à V_dc fixé, doubler m double la crête du
        // fondamental.
        let v1 = inv_sine_pwm_fundamental_peak(400.0, 0.4);
        let v2 = inv_sine_pwm_fundamental_peak(400.0, 0.8);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn sine_pwm_peak_from_modulation_index() {
        // Enchaînement cohérent : m = V̂_ref / V̂_carrier = 8/10 = 0,8, puis
        //   V̂_1 = m · V_dc / 2 = 0,8 · 400 / 2 = 160 V.
        let m = inv_modulation_index(8.0, 10.0);
        assert_relative_eq!(m, 0.8, epsilon = 1e-12);
        assert_relative_eq!(
            inv_sine_pwm_fundamental_peak(400.0, m),
            160.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn full_modulation_reaches_half_bus() {
        // Cas limite : m = 1 (bord de la zone linéaire) donne V̂_1 = V_dc / 2.
        let v_dc = 540.0_f64;
        assert_relative_eq!(
            inv_sine_pwm_fundamental_peak(v_dc, 1.0),
            v_dc / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "l'indice de modulation m doit être dans [0, 1] (zone linéaire)")]
    fn overmodulation_panics() {
        inv_sine_pwm_fundamental_peak(400.0, 1.2);
    }
}
