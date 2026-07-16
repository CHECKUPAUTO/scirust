//! **Condensateur idéal** — module d'énergie stockée, de charge électrique, de
//! réactance capacitive et de régime transitoire de charge d'un condensateur
//! linéaire dans un circuit RC du premier ordre.
//!
//! ```text
//! énergie stockée      E   = 0,5·C·V²
//! charge électrique    Q   = C·V
//! réactance capacitive X_C = 1 / (2·π·f·C)
//! constante de temps   τ   = R·C
//! tension de charge     v(t) = V_s·(1 − e^(−t/τ))
//! ```
//!
//! `C` capacité (F), `V` tension aux bornes (V), `E` énergie électrostatique
//! stockée (J), `Q` charge accumulée (C), `f` fréquence du régime sinusoïdal
//! (Hz), `X_C` réactance capacitive (Ω), `R` résistance série de charge (Ω),
//! `τ` constante de temps du circuit RC (s), `t` temps écoulé depuis le début
//! de la charge (s), `V_s` tension d'alimentation (V), `v(t)` tension aux
//! bornes du condensateur pendant la charge (V).
//!
//! **Convention** : SI ; capacités en F, tensions en V, énergies en J, charges
//! en C, fréquences en Hz, réactances en Ω, temps et constantes de temps en s ;
//! l'argument des exponentielles est **sans dimension**. **Limite honnête** :
//! condensateur **linéaire idéal** (capacité constante, **sans fuite** ni
//! résistance série équivalente ESR), circuit **RC du premier ordre** pour le
//! transitoire, et **régime sinusoïdal permanent** pour la réactance. Les
//! grandeurs réseau (fréquence, tension), les valeurs de composant (`C`, `R`)
//! et les grandeurs matériau sont **fournies par l'appelant** (mesures, fiches
//! composant) — aucune valeur « par défaut » n'est inventée.

/// Énergie électrostatique stockée `E = 0,5·C·V²` (J).
///
/// Panique si `capacitance < 0` (une capacité négative n'a pas de sens
/// physique).
pub fn cap_energy(capacitance: f64, voltage: f64) -> f64 {
    assert!(capacitance >= 0.0, "la capacité C doit être ≥ 0");
    0.5 * capacitance * voltage * voltage
}

/// Charge électrique accumulée `Q = C·V` (C).
///
/// Panique si `capacitance < 0` (une capacité négative n'a pas de sens
/// physique).
pub fn cap_charge(capacitance: f64, voltage: f64) -> f64 {
    assert!(capacitance >= 0.0, "la capacité C doit être ≥ 0");
    capacitance * voltage
}

/// Réactance capacitive en régime sinusoïdal `X_C = 1 / (2·π·f·C)` (Ω).
///
/// Panique si `capacitance <= 0` ou si `frequency <= 0` (division par zéro).
pub fn cap_reactance(capacitance: f64, frequency: f64) -> f64 {
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    1.0 / (2.0 * core::f64::consts::PI * frequency * capacitance)
}

/// Constante de temps du circuit RC `τ = R·C` (s).
///
/// Panique si `resistance < 0` ou si `capacitance < 0`.
pub fn cap_rc_time_constant(resistance: f64, capacitance: f64) -> f64 {
    assert!(resistance >= 0.0, "la résistance R doit être ≥ 0");
    assert!(capacitance >= 0.0, "la capacité C doit être ≥ 0");
    resistance * capacitance
}

/// Tension aux bornes pendant la charge à travers une résistance
/// `v(t) = V_s·(1 − e^(−t/τ))` (V).
///
/// Panique si `time < 0` ou si `time_constant <= 0` (division par zéro).
pub fn cap_charging_voltage(supply_voltage: f64, time: f64, time_constant: f64) -> f64 {
    assert!(time >= 0.0, "le temps t doit être ≥ 0");
    assert!(
        time_constant > 0.0,
        "la constante de temps τ doit être strictement positive"
    );
    supply_voltage * (1.0 - (-time / time_constant).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn energy_equals_half_charge_times_voltage() {
        // Identité : E = 0,5·C·V² = 0,5·(C·V)·V = 0,5·Q·V. On relie l'énergie
        // à la charge pour les mêmes composants.
        let c = 4.7e-6_f64;
        let v = 25.0_f64;
        let q = cap_charge(c, v);
        assert_relative_eq!(cap_energy(c, v), 0.5 * q * v, epsilon = 1e-15);
    }

    #[test]
    fn charge_is_linear_in_voltage() {
        // Proportionnalité : Q = C·V est linéaire en V ; doubler la tension
        // double la charge accumulée.
        let c = 1.0e-3_f64;
        let q1 = cap_charge(c, 10.0);
        let q2 = cap_charge(c, 20.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-15);
    }

    #[test]
    fn realistic_energy_and_charge_case() {
        // Cas chiffré réaliste, C = 100 µF sous V = 100 V :
        //   Q = C·V   = 1e-4 · 100      = 1,0e-2 C
        //   E = 0,5·C·V² = 0,5·1e-4·1e4 = 0,5 J
        let c = 1.0e-4_f64;
        let v = 100.0_f64;
        assert_relative_eq!(cap_charge(c, v), 1.0e-2, epsilon = 1e-12);
        assert_relative_eq!(cap_energy(c, v), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn realistic_reactance_case() {
        // Cas chiffré réaliste, C = 1 µF à f = 50 Hz :
        //   X_C = 1/(2π·50·1e-6) = 1/(3,141 592 65e-4) ≈ 3183,098 9 Ω
        let x = cap_reactance(1.0e-6, 50.0);
        assert_relative_eq!(x, 3_183.098_9, epsilon = 1e-3);
    }

    #[test]
    fn charging_hits_known_fractions() {
        // Cas limites du transitoire RC : v(0) = 0 ; à t = τ la tension atteint
        // (1 − 1/e)·V_s ≈ 0,632 120 6·V_s.
        let vs = 12.0_f64;
        let tau = 0.47_f64;
        assert_relative_eq!(cap_charging_voltage(vs, 0.0, tau), 0.0, epsilon = 1e-15);
        let expected = vs * (1.0 - core::f64::consts::E.recip());
        assert_relative_eq!(
            cap_charging_voltage(vs, tau, tau),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn charging_approaches_supply_and_uses_time_constant() {
        // Cohérence : τ = R·C, et pour t ≫ τ la tension tend vers V_s. À
        // t = 5·τ la charge atteint ≈ 99,33 % de V_s.
        let r = 1.0e3_f64;
        let c = 4.7e-4_f64;
        let tau = cap_rc_time_constant(r, c);
        assert_relative_eq!(tau, 0.47, epsilon = 1e-12);
        let vs = 12.0_f64;
        let v5 = cap_charging_voltage(vs, 5.0 * tau, tau);
        assert!(v5 < vs, "la tension de charge reste sous V_s");
        assert_relative_eq!(v5, vs * (1.0 - (-5.0_f64).exp()), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la fréquence f doit être strictement positive")]
    fn zero_frequency_reactance_panics() {
        cap_reactance(1.0e-6, 0.0);
    }
}
