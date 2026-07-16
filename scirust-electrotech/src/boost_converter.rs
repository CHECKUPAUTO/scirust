//! **Hacheur parallèle (boost, élévateur) en conduction continue** — tension de
//! sortie moyenne, rapport cyclique requis, ondulation du courant d'inductance
//! et courant d'entrée moyen (conservation de puissance idéale).
//!
//! ```text
//! tension de sortie (moy.)   V_out = V_in / (1 − D)
//! rapport cyclique requis    D     = 1 − V_in / V_out
//! ondulation de courant      ΔI_L  = V_in · D / (L · f_sw)
//! courant d'entrée (moy.)    I_in  = I_out / (1 − D)
//! ```
//!
//! `V_in` tension d'entrée (V), `V_out` tension de sortie moyenne (V), `D`
//! rapport cyclique (sans dimension, `0 < D < 1`), `ΔI_L` ondulation crête-à-crête
//! du courant dans l'inductance (A), `L` inductance de stockage (H), `f_sw`
//! fréquence de découpage (Hz), `I_out` courant de sortie moyen (A), `I_in`
//! courant d'entrée moyen (A). En régime permanent, `V_out = V_in / (1 − D)` et
//! `D = 1 − V_in / V_out` sont réciproques ; la conservation de puissance idéale
//! `V_in · I_in = V_out · I_out` donne `I_in = I_out / (1 − D)`.
//!
//! **Convention** : SI ; tensions en V, courants en A, inductance en H, fréquence
//! en Hz. **Limite honnête** : conduction **continue** (CCM), **régime
//! permanent**, interrupteur, diode (ou redresseur synchrone), inductance et
//! condensateur supposés **idéaux** (résistances série, chutes directes et pertes
//! de commutation négligées ; rendement unitaire). La tension de sortie tend vers
//! l'infini quand `D → 1` (comportement non physique, borné en pratique par les
//! pertes réelles). La tension d'entrée `V_in`, le rapport cyclique `D`
//! (`0 < D < 1`), les valeurs de composants (`L`, `f_sw`) et le courant de sortie
//! `I_out` sont **fournis par l'appelant** (réseau, commande, choix de
//! dimensionnement) — aucune valeur « par défaut » n'est inventée.

/// Tension de sortie moyenne d'un hacheur parallèle `V_out = V_in / (1 − D)` (V).
///
/// Panique si `input_voltage < 0` ou si `duty_cycle` n'est pas dans `]0, 1[`.
pub fn boost_output_voltage(input_voltage: f64, duty_cycle: f64) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    input_voltage / (1.0 - duty_cycle)
}

/// Rapport cyclique requis pour obtenir `V_out` à partir de `V_in`,
/// `D = 1 − V_in / V_out` (sans dimension).
///
/// Panique si `input_voltage <= 0`, si `output_voltage <= 0` ou si
/// `output_voltage <= input_voltage` (un élévateur ne peut abaisser la tension,
/// `D > 0` requiert `V_out > V_in`).
pub fn boost_duty_for_output(input_voltage: f64, output_voltage: f64) -> f64 {
    assert!(
        input_voltage > 0.0,
        "la tension d'entrée V_in doit être strictement positive"
    );
    assert!(
        output_voltage > 0.0,
        "la tension de sortie V_out doit être strictement positive"
    );
    assert!(
        output_voltage > input_voltage,
        "élévateur : V_out > V_in requis (0 < D < 1)"
    );
    1.0 - input_voltage / output_voltage
}

/// Ondulation crête-à-crête du courant d'inductance
/// `ΔI_L = V_in · D / (L · f_sw)` (A).
///
/// Panique si `input_voltage < 0`, si `duty_cycle` n'est pas dans `]0, 1[`, si
/// `inductance <= 0` ou si `switching_frequency <= 0`.
pub fn boost_inductor_ripple_current(
    input_voltage: f64,
    duty_cycle: f64,
    inductance: f64,
    switching_frequency: f64,
) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    assert!(
        switching_frequency > 0.0,
        "la fréquence de découpage f_sw doit être strictement positive"
    );
    input_voltage * duty_cycle / (inductance * switching_frequency)
}

/// Courant d'entrée moyen d'un hacheur parallèle `I_in = I_out / (1 − D)` (A),
/// issu de la conservation de puissance idéale `V_in · I_in = V_out · I_out`.
///
/// Panique si `output_current < 0` ou si `duty_cycle` n'est pas dans `]0, 1[`.
pub fn boost_input_current(output_current: f64, duty_cycle: f64) -> f64 {
    assert!(
        output_current >= 0.0,
        "le courant de sortie I_out doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    output_current / (1.0 - duty_cycle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_and_duty_are_reciprocal() {
        // Réciprocité : D = 1 − V_in / V_out inverse V_out = V_in / (1 − D).
        let v_in = 12.0_f64;
        let duty = 0.6_f64;
        let v_out = boost_output_voltage(v_in, duty);
        assert_relative_eq!(boost_duty_for_output(v_in, v_out), duty, epsilon = 1e-12);
    }

    #[test]
    fn output_voltage_scales_linearly_with_input() {
        // Proportionnalité : à D fixé, doubler V_in double V_out.
        let v1 = boost_output_voltage(10.0, 0.5);
        let v2 = boost_output_voltage(20.0, 0.5);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn output_voltage_reference_case() {
        // Cas chiffré : V_in = 12 V, D = 0,6, 1 − D = 0,4.
        //   V_out = 12 / 0,4 = 30 V.
        let v_out = boost_output_voltage(12.0, 0.6);
        assert_relative_eq!(v_out, 30.0, epsilon = 1e-12);
    }

    #[test]
    fn power_is_conserved_ideally() {
        // Conservation de puissance idéale : V_in · I_in = V_out · I_out.
        //   V_in = 12 V, D = 0,6 → V_out = 30 V ; I_out = 2 A → I_in = 2 / 0,4 = 5 A.
        //   P_in = 12 · 5 = 60 W ; P_out = 30 · 2 = 60 W.
        let v_in = 12.0_f64;
        let duty = 0.6_f64;
        let i_out = 2.0_f64;
        let v_out = boost_output_voltage(v_in, duty);
        let i_in = boost_input_current(i_out, duty);
        assert_relative_eq!(i_in, 5.0, epsilon = 1e-12);
        assert_relative_eq!(v_in * i_in, v_out * i_out, epsilon = 1e-9);
    }

    #[test]
    fn inductor_ripple_reference_case() {
        // Cas chiffré : V_in = 12 V, D = 0,6, L = 100 µH = 100e-6 H,
        //   f_sw = 100 kHz = 100e3 Hz. L · f_sw = 100e-6 · 100e3 = 10.
        //   ΔI_L = 12 · 0,6 / 10 = 7,2 / 10 = 0,72 A.
        let ripple = boost_inductor_ripple_current(12.0, 0.6, 100e-6, 100e3);
        assert_relative_eq!(ripple, 0.72, epsilon = 1e-9);
    }

    #[test]
    fn output_voltage_grows_as_duty_approaches_one() {
        // Cas limite : quand D → 1, V_out = V_in / (1 − D) → +∞ (non physique).
        let v_out = boost_output_voltage(12.0, 0.99);
        assert!(v_out > 1000.0, "V_out doit diverger quand D → 1");
    }

    #[test]
    #[should_panic(expected = "élévateur : V_out > V_in requis (0 < D < 1)")]
    fn duty_for_lower_output_panics() {
        boost_duty_for_output(12.0, 5.0);
    }
}
