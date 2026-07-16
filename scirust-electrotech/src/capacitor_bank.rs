//! **Batterie de condensateurs (compensation)** — module de puissance réactive
//! fournie par une batterie de condensateurs shunt, de capacité nécessaire pour
//! une puissance réactive visée, de nombre de gradins d'une batterie
//! automatique et de résistance de décharge de sécurité (constante de temps RC).
//!
//! ```text
//! puissance réactive       Q  = 2·π·f·C·V²
//! capacité pour un Q visé   C  = Q / (2·π·f·V²)
//! nombre de gradins         n  = Q_total / Q_gradin
//! résistance de décharge    R  = t / (C·ln(V / V_res))
//! ```
//!
//! `Q` puissance réactive fournie par la batterie (var), `f` fréquence du réseau
//! (Hz), `C` capacité de la batterie (F), `V` tension aux bornes (V),
//! `Q_total` puissance réactive totale à compenser (var), `Q_gradin` puissance
//! réactive d'un gradin élémentaire (var), `n` nombre de gradins (sans
//! dimension), `t` temps de décharge visé (s), `V_res` tension résiduelle
//! admissible après décharge (V), `R` résistance de décharge (Ω).
//!
//! **Convention** : SI ; puissances réactives en var, capacités en F, tensions
//! en V, fréquences en Hz, temps en s, résistances en Ω ; l'argument du
//! logarithme est **sans dimension**. **Limite honnête** : condensateurs de
//! **compensation en régime sinusoïdal** — la capacité, la tension et la
//! fréquence sont **fournies par l'appelant** (fiche batterie, mesure réseau) ;
//! le nombre de gradins est un **réel** que l'appelant **arrondit à l'entier**
//! selon sa stratégie de régulation ; la résistance de décharge dimensionne la
//! **mise à la terre de sécurité** via la constante de temps RC ; les relations
//! sont **monophasées** (le triphasé s'adapte selon le couplage étoile ou
//! triangle). Aucune valeur « par défaut » n'est inventée.

/// Puissance réactive fournie par la batterie `Q = 2·π·f·C·V²` (var).
///
/// Formule : `Q = 2·π·f·C·V²`, puissance réactive capacitive absorbée par un
/// condensateur en régime sinusoïdal monophasé.
///
/// Panique si `capacitance < 0` ou si `frequency < 0` (grandeurs sans sens
/// physique).
pub fn capbank_reactive_power(capacitance: f64, voltage: f64, frequency: f64) -> f64 {
    assert!(capacitance >= 0.0, "la capacité C doit être ≥ 0");
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0");
    2.0 * core::f64::consts::PI * frequency * capacitance * voltage * voltage
}

/// Capacité nécessaire pour une puissance réactive visée
/// `C = Q / (2·π·f·V²)` (F).
///
/// Formule : inverse de [`capbank_reactive_power`] à tension et fréquence
/// fixées ; capacité de batterie fournissant `Q` var sous `V` volts.
///
/// Panique si `reactive_power < 0`, si `frequency <= 0` ou si `voltage == 0`
/// (division par zéro).
pub fn capbank_capacitance_for_kvar(reactive_power: f64, voltage: f64, frequency: f64) -> f64 {
    assert!(
        reactive_power >= 0.0,
        "la puissance réactive Q doit être ≥ 0"
    );
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    assert!(
        voltage != 0.0,
        "la tension V doit être non nulle (division par V²)"
    );
    reactive_power / (2.0 * core::f64::consts::PI * frequency * voltage * voltage)
}

/// Nombre de gradins d'une batterie automatique
/// `n = Q_total / Q_gradin` (sans dimension).
///
/// Formule : rapport de la puissance réactive totale à compenser sur la
/// puissance réactive d'un gradin élémentaire. Le résultat est un **réel** ;
/// l'appelant l'arrondit à l'entier selon sa stratégie de régulation.
///
/// Panique si `total_reactive_power < 0` ou si `step_reactive_power <= 0`
/// (division par zéro).
pub fn capbank_number_of_steps(total_reactive_power: f64, step_reactive_power: f64) -> f64 {
    assert!(
        total_reactive_power >= 0.0,
        "la puissance réactive totale Q_total doit être ≥ 0"
    );
    assert!(
        step_reactive_power > 0.0,
        "la puissance réactive d'un gradin Q_gradin doit être strictement positive"
    );
    total_reactive_power / step_reactive_power
}

/// Résistance de décharge de sécurité `R = t / (C·ln(V / V_res))` (Ω).
///
/// Formule : décharge d'un condensateur à travers une résistance en un temps
/// `t` pour ramener la tension de `V` à la tension résiduelle admissible
/// `V_res` (constante de temps RC : `v(t) = V·e^(−t/RC)`).
///
/// Panique si `capacitance <= 0`, si `discharge_time <= 0`, si `voltage <= 0`,
/// si `residual_voltage <= 0` ou si `residual_voltage >= voltage` (le
/// logarithme `ln(V / V_res)` doit être strictement positif).
pub fn capbank_discharge_resistor(
    capacitance: f64,
    voltage: f64,
    discharge_time: f64,
    residual_voltage: f64,
) -> f64 {
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    assert!(
        discharge_time > 0.0,
        "le temps de décharge t doit être strictement positif"
    );
    assert!(voltage > 0.0, "la tension V doit être strictement positive");
    assert!(
        residual_voltage > 0.0,
        "la tension résiduelle V_res doit être strictement positive"
    );
    assert!(
        residual_voltage < voltage,
        "la tension résiduelle V_res doit être strictement inférieure à V"
    );
    discharge_time / (capacitance * (voltage / residual_voltage).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reactive_power_and_capacitance_are_reciprocal() {
        // Réciprocité : la capacité calculée pour un Q visé, réinjectée dans la
        // formule de puissance réactive, redonne exactement ce Q (mêmes V, f).
        let q = 1_661.9_f64;
        let v = 230.0_f64;
        let f = 50.0_f64;
        let c = capbank_capacitance_for_kvar(q, v, f);
        assert_relative_eq!(capbank_reactive_power(c, v, f), q, epsilon = 1e-6);
    }

    #[test]
    fn reactive_power_is_quadratic_in_voltage() {
        // Proportionnalité : Q ∝ V² à C et f fixés ; doubler la tension
        // quadruple la puissance réactive fournie.
        let c = 1.0e-4_f64;
        let f = 50.0_f64;
        let q1 = capbank_reactive_power(c, 230.0, f);
        let q2 = capbank_reactive_power(c, 460.0, f);
        assert_relative_eq!(q2, 4.0 * q1, epsilon = 1e-9);
    }

    #[test]
    fn realistic_reactive_power_case() {
        // Cas chiffré réaliste, C = 100 µF sous V = 230 V à f = 50 Hz :
        //   Q = 2π·50·1e-4·230²
        //     = 314,159 265·1e-4·52 900
        //     = 0,031 415 926 5·52 900
        //     ≈ 1 661,90 var
        let q = capbank_reactive_power(1.0e-4, 230.0, 50.0);
        assert_relative_eq!(q, 1_661.90, epsilon = 1e-1);
    }

    #[test]
    fn number_of_steps_is_linear() {
        // Proportionnalité : n = Q_total / Q_gradin est linéaire en Q_total ;
        // cas chiffré simple 500 var / 50 var = 10 gradins.
        let n = capbank_number_of_steps(500.0, 50.0);
        assert_relative_eq!(n, 10.0, epsilon = 1e-12);
        let n2 = capbank_number_of_steps(1_000.0, 50.0);
        assert_relative_eq!(n2, 2.0 * n, epsilon = 1e-12);
    }

    #[test]
    fn discharge_resistor_satisfies_rc_identity() {
        // Identité RC : la résistance calculée vérifie t = R·C·ln(V / V_res),
        // c'est-à-dire v(t) = V·e^(−t/RC) = V_res à l'instant t visé.
        let c = 1.0e-4_f64;
        let v = 400.0_f64;
        let t = 60.0_f64;
        let v_res = 50.0_f64;
        let r = capbank_discharge_resistor(c, v, t, v_res);
        let recovered = r * c * (v / v_res).ln();
        assert_relative_eq!(recovered, t, epsilon = 1e-9);
    }

    #[test]
    fn realistic_discharge_resistor_case() {
        // Cas chiffré réaliste, C = 100 µF, V = 400 V, V_res = 50 V, t = 60 s :
        //   ln(400/50) = ln(8) ≈ 2,079 441 542
        //   R = 60 / (1e-4 · 2,079 441 542)
        //     = 60 / 2,079 441 542e-4
        //     ≈ 288 539,0 Ω
        let r = capbank_discharge_resistor(1.0e-4, 400.0, 60.0, 50.0);
        assert_relative_eq!(r, 288_539.0, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "la tension résiduelle V_res doit être strictement inférieure à V")]
    fn residual_not_below_voltage_panics() {
        capbank_discharge_resistor(1.0e-4, 400.0, 60.0, 400.0);
    }
}
