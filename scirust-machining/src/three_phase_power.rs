//! Électrotechnique — **puissances d'un système triphasé équilibré** :
//! puissances active, apparente et réactive, et courant de ligne, à partir des
//! tensions et courants **de ligne** et du facteur de puissance.
//!
//! ```text
//! puissance active       P  = √3 · U_L · I_L · cos φ            [W]
//! puissance apparente    S  = √3 · U_L · I_L                    [VA]
//! puissance réactive     Q  = √3 · U_L · I_L · sin(acos cosφ)   [var]
//! courant de ligne       I_L = P / (√3 · U_L · cos φ)           [A]
//! (identité)             S² = P² + Q²
//! ```
//!
//! `U_L` tension composée (entre lignes) [V], `I_L` courant de ligne [A],
//! `cos φ` facteur de puissance [sans dimension, 0…1], `P` puissance active
//! [W], `S` puissance apparente [VA], `Q` puissance réactive [var]. Le `√3`
//! provient du rapport tension composée / tension simple d'un système
//! triphasé équilibré.
//!
//! **Limite honnête** : ces formules supposent un système **triphasé
//! équilibré** et **sinusoïdal** (régime permanent), exprimé en grandeurs
//! **de ligne** ; le facteur de puissance `cos φ` est **fourni par l'appelant**
//! et n'est jamais supposé par défaut. Les harmoniques, le déséquilibre de
//! charge et les régimes transitoires sont **négligés** ; en présence
//! d'harmoniques, `S² = P² + Q²` cesse d'être vérifiée (puissance déformante).

/// Puissance active `P = √3 · U_L · I_L · cos φ` [W].
///
/// `line_voltage` tension composée en volts (V), `line_current` courant de
/// ligne en ampères (A), `power_factor` facteur de puissance `cos φ` (sans
/// dimension, 0…1) ; le résultat est en watts (W).
///
/// Panique si `line_voltage < 0`, `line_current < 0`, ou si `power_factor`
/// n'est pas dans l'intervalle `[0, 1]`.
pub fn three_phase_active_power(line_voltage: f64, line_current: f64, power_factor: f64) -> f64 {
    assert!(
        line_voltage >= 0.0,
        "la tension de ligne doit être positive ou nulle (V)"
    );
    assert!(
        line_current >= 0.0,
        "le courant de ligne doit être positif ou nul (A)"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance doit être dans [0, 1]"
    );
    3.0_f64.sqrt() * line_voltage * line_current * power_factor
}

/// Puissance apparente `S = √3 · U_L · I_L` [VA].
///
/// `line_voltage` tension composée en volts (V), `line_current` courant de
/// ligne en ampères (A) ; le résultat est en volt-ampères (VA).
///
/// Panique si `line_voltage < 0` ou `line_current < 0`.
pub fn three_phase_apparent_power(line_voltage: f64, line_current: f64) -> f64 {
    assert!(
        line_voltage >= 0.0,
        "la tension de ligne doit être positive ou nulle (V)"
    );
    assert!(
        line_current >= 0.0,
        "le courant de ligne doit être positif ou nul (A)"
    );
    3.0_f64.sqrt() * line_voltage * line_current
}

/// Puissance réactive `Q = √3 · U_L · I_L · sin(acos cos φ)` [var].
///
/// `line_voltage` tension composée en volts (V), `line_current` courant de
/// ligne en ampères (A), `power_factor` facteur de puissance `cos φ` (sans
/// dimension, 0…1) ; le résultat est en var. Convention : signe positif
/// (charge inductive), le facteur de puissance ne portant pas le signe.
///
/// Panique si `line_voltage < 0`, `line_current < 0`, ou si `power_factor`
/// n'est pas dans l'intervalle `[0, 1]`.
pub fn three_phase_reactive_power(line_voltage: f64, line_current: f64, power_factor: f64) -> f64 {
    assert!(
        line_voltage >= 0.0,
        "la tension de ligne doit être positive ou nulle (V)"
    );
    assert!(
        line_current >= 0.0,
        "le courant de ligne doit être positif ou nul (A)"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance doit être dans [0, 1]"
    );
    // sin(acos cosφ) = √(1 − cos²φ), robuste et sans appel trigonométrique.
    let sin_phi = (1.0 - power_factor * power_factor).sqrt();
    3.0_f64.sqrt() * line_voltage * line_current * sin_phi
}

/// Courant de ligne `I_L = P / (√3 · U_L · cos φ)` [A] (inverse de
/// [`three_phase_active_power`] par rapport au courant).
///
/// `active_power` puissance active en watts (W), `line_voltage` tension
/// composée en volts (V), `power_factor` facteur de puissance `cos φ` (sans
/// dimension, `]0, 1]`) ; le résultat est en ampères (A).
///
/// Panique si `active_power < 0`, `line_voltage <= 0`, ou si `power_factor`
/// n'est pas dans l'intervalle `]0, 1]` (facteur nul interdit : division).
pub fn three_phase_line_current(active_power: f64, line_voltage: f64, power_factor: f64) -> f64 {
    assert!(
        active_power >= 0.0,
        "la puissance active doit être positive ou nulle (W)"
    );
    assert!(
        line_voltage > 0.0,
        "la tension de ligne doit être strictement positive (V)"
    );
    assert!(
        power_factor > 0.0 && power_factor <= 1.0,
        "le facteur de puissance doit être dans ]0, 1]"
    );
    active_power / (3.0_f64.sqrt() * line_voltage * power_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_400v_case() {
        // Charge triphasée 400 V, 10 A, cos φ = 0,8 (cas d'atelier typique).
        // S = √3·400·10 = 6928,2032… VA
        // P = S·0,8 = 5542,5626… W ; Q = S·0,6 = 4156,9219… var
        let (u, i, pf) = (400.0, 10.0, 0.8);
        assert_relative_eq!(
            three_phase_apparent_power(u, i),
            6_928.203_230_275_509,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            three_phase_active_power(u, i, pf),
            5_542.562_584_220_407,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            three_phase_reactive_power(u, i, pf),
            4_156.921_938_165_305,
            epsilon = 1e-9
        );
    }

    #[test]
    fn power_triangle_identity() {
        // S² = P² + Q² pour tout (U, I, cos φ) valide.
        let (u, i, pf) = (690.0, 42.0, 0.87);
        let s = three_phase_apparent_power(u, i);
        let p = three_phase_active_power(u, i, pf);
        let q = three_phase_reactive_power(u, i, pf);
        assert_relative_eq!(p * p + q * q, s * s, epsilon = 1e-6);
    }

    #[test]
    fn active_is_apparent_times_power_factor() {
        // P = S · cos φ par construction.
        let (u, i, pf) = (230.0, 16.0, 0.92);
        let s = three_phase_apparent_power(u, i);
        assert_relative_eq!(three_phase_active_power(u, i, pf), s * pf, epsilon = 1e-9);
    }

    #[test]
    fn current_is_reciprocal_of_active_power() {
        // three_phase_line_current ∘ three_phase_active_power = identité (I).
        let (u, i, pf) = (400.0, 12.5, 0.85);
        let p = three_phase_active_power(u, i, pf);
        assert_relative_eq!(three_phase_line_current(p, u, pf), i, epsilon = 1e-9);
    }

    #[test]
    fn unity_power_factor_zeroes_reactive() {
        // cos φ = 1 ⇒ Q = 0 et P = S.
        let (u, i) = (400.0, 10.0);
        assert_relative_eq!(three_phase_reactive_power(u, i, 1.0), 0.0, epsilon = 1e-9);
        assert_relative_eq!(
            three_phase_active_power(u, i, 1.0),
            three_phase_apparent_power(u, i),
            epsilon = 1e-9
        );
    }

    #[test]
    fn active_power_scales_linearly_with_current() {
        // P ∝ I_L à tension et cos φ fixés : doubler I double P.
        let (u, pf) = (400.0, 0.8);
        let p1 = three_phase_active_power(u, 10.0, pf);
        let p2 = three_phase_active_power(u, 20.0, pf);
        assert_relative_eq!(p2, 2.0 * p1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "facteur de puissance doit être dans [0, 1]")]
    fn power_factor_above_one_panics() {
        three_phase_active_power(400.0, 10.0, 1.5);
    }
}
