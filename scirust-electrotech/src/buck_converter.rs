//! **Hacheur série (buck, abaisseur) en conduction continue** — tension de
//! sortie moyenne, rapport cyclique requis, ondulation du courant d'inductance
//! et ondulation résiduelle de la tension de sortie.
//!
//! ```text
//! tension de sortie (moy.)   V_out = D · V_in
//! rapport cyclique requis    D     = V_out / V_in
//! ondulation de courant      ΔI_L  = V_out · (1 − D) / (L · f_sw)
//! ondulation de tension      ΔV_out = ΔI_L / (8 · C · f_sw)
//! ```
//!
//! `V_in` tension d'entrée (V), `V_out` tension de sortie moyenne (V), `D`
//! rapport cyclique (sans dimension, `0 < D < 1`), `ΔI_L` ondulation crête-à-crête
//! du courant dans l'inductance (A), `L` inductance de filtrage (H), `f_sw`
//! fréquence de découpage (Hz), `ΔV_out` ondulation crête-à-crête de la tension
//! de sortie (V), `C` capacité du condensateur de sortie (F). En régime
//! permanent, `V_out = D · V_in` et donc `D = V_out / V_in` sont réciproques.
//!
//! **Convention** : SI ; tensions en V, courants en A, inductance en H, capacité
//! en F, fréquence en Hz. **Limite honnête** : conduction **continue** (CCM),
//! **régime permanent**, interrupteur, diode (ou redresseur synchrone),
//! inductance et condensateur supposés **idéaux** (résistances série, chutes
//! directes et pertes de commutation négligées ; ESR nulle pour l'ondulation de
//! tension). La tension d'entrée `V_in`, le rapport cyclique `D` (`0 < D < 1`) et
//! les valeurs de composants (`L`, `C`, `f_sw`) sont **fournis par l'appelant**
//! (réseau, commande, choix de dimensionnement) — aucune valeur « par défaut »
//! n'est inventée.

/// Tension de sortie moyenne d'un hacheur série `V_out = D · V_in` (V).
///
/// Panique si `input_voltage < 0` ou si `duty_cycle` n'est pas dans `]0, 1[`.
pub fn buck_output_voltage(input_voltage: f64, duty_cycle: f64) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    duty_cycle * input_voltage
}

/// Rapport cyclique requis pour obtenir `V_out` à partir de `V_in`,
/// `D = V_out / V_in` (sans dimension).
///
/// Panique si `input_voltage <= 0`, si `output_voltage < 0` ou si
/// `output_voltage > input_voltage` (un abaisseur ne peut élever la tension,
/// `D ≤ 1`).
pub fn buck_duty_for_output(output_voltage: f64, input_voltage: f64) -> f64 {
    assert!(
        input_voltage > 0.0,
        "la tension d'entrée V_in doit être strictement positive"
    );
    assert!(
        output_voltage >= 0.0,
        "la tension de sortie V_out doit être ≥ 0"
    );
    assert!(
        output_voltage <= input_voltage,
        "abaisseur : V_out ≤ V_in requis (D ≤ 1)"
    );
    output_voltage / input_voltage
}

/// Ondulation crête-à-crête du courant d'inductance
/// `ΔI_L = V_out · (1 − D) / (L · f_sw)` (A).
///
/// Panique si `output_voltage < 0`, si `duty_cycle` n'est pas dans `]0, 1[`, si
/// `inductance <= 0` ou si `switching_frequency <= 0`.
pub fn buck_inductor_ripple_current(
    output_voltage: f64,
    duty_cycle: f64,
    inductance: f64,
    switching_frequency: f64,
) -> f64 {
    assert!(
        output_voltage >= 0.0,
        "la tension de sortie V_out doit être ≥ 0"
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
    output_voltage * (1.0 - duty_cycle) / (inductance * switching_frequency)
}

/// Ondulation crête-à-crête de la tension de sortie
/// `ΔV_out = ΔI_L / (8 · C · f_sw)` (V), condensateur idéal (ESR nulle).
///
/// Panique si `inductor_ripple_current < 0`, si `capacitance <= 0` ou si
/// `switching_frequency <= 0`.
pub fn buck_output_ripple_voltage(
    inductor_ripple_current: f64,
    capacitance: f64,
    switching_frequency: f64,
) -> f64 {
    assert!(
        inductor_ripple_current >= 0.0,
        "l'ondulation de courant ΔI_L doit être ≥ 0"
    );
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    assert!(
        switching_frequency > 0.0,
        "la fréquence de découpage f_sw doit être strictement positive"
    );
    inductor_ripple_current / (8.0 * capacitance * switching_frequency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_and_duty_are_reciprocal() {
        // Réciprocité : D = V_out / V_in inverse V_out = D · V_in.
        let v_in = 12.0_f64;
        let duty = 0.375_f64;
        let v_out = buck_output_voltage(v_in, duty);
        assert_relative_eq!(buck_duty_for_output(v_out, v_in), duty, epsilon = 1e-12);
    }

    #[test]
    fn output_voltage_scales_linearly_with_input() {
        // Proportionnalité : à D fixé, doubler V_in double V_out.
        let v1 = buck_output_voltage(10.0, 0.4);
        let v2 = buck_output_voltage(20.0, 0.4);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn duty_for_five_from_twelve_volts() {
        // Cas chiffré : abaisser 12 V vers 5 V demande D = 5/12 ≈ 0,4166667.
        let duty = buck_duty_for_output(5.0, 12.0);
        assert_relative_eq!(duty, 0.416_666_666_7, epsilon = 1e-9);
    }

    #[test]
    fn inductor_ripple_reference_case() {
        // Cas chiffré : V_out = 6 V (12 V, D = 0,5), 1 − D = 0,5,
        //   L = 100 µH = 100e-6 H, f_sw = 100 kHz = 100e3 Hz.
        //   L · f_sw = 100e-6 · 100e3 = 10.
        //   ΔI_L = 6 · 0,5 / 10 = 3 / 10 = 0,3 A.
        let v_out = buck_output_voltage(12.0, 0.5);
        assert_relative_eq!(v_out, 6.0, epsilon = 1e-12);
        let ripple = buck_inductor_ripple_current(v_out, 0.5, 100e-6, 100e3);
        assert_relative_eq!(ripple, 0.3, epsilon = 1e-9);
    }

    #[test]
    fn output_ripple_voltage_reference_case() {
        // Suite du cas précédent : ΔI_L = 0,3 A, C = 100 µF = 100e-6 F,
        //   f_sw = 100 kHz. 8 · C · f_sw = 8 · 100e-6 · 100e3 = 8 · 10 = 80.
        //   ΔV_out = 0,3 / 80 = 0,003 75 V = 3,75 mV.
        let ripple_v = buck_output_ripple_voltage(0.3, 100e-6, 100e3);
        assert_relative_eq!(ripple_v, 0.003_75, epsilon = 1e-9);
    }

    #[test]
    fn ripple_current_vanishes_as_duty_approaches_one() {
        // Cas limite : quand D → 1, le facteur (1 − D) → 0 et ΔI_L → 0.
        let ripple = buck_inductor_ripple_current(6.0, 0.999, 100e-6, 100e3);
        assert!(
            ripple > 0.0 && ripple < 1e-2,
            "ΔI_L doit être faible et positif"
        );
    }

    #[test]
    #[should_panic(expected = "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)")]
    fn duty_out_of_range_panics() {
        buck_output_voltage(12.0, 1.5);
    }
}
