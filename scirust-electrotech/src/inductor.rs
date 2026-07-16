//! **Bobine idéale** — module d'énergie magnétique stockée, de réactance
//! inductive, de régime transitoire d'établissement du courant et de force
//! électromotrice auto-induite d'une inductance linéaire dans un circuit RL du
//! premier ordre.
//!
//! ```text
//! énergie stockée      E    = 0,5·L·I²
//! réactance inductive  X_L  = 2·π·f·L
//! constante de temps   τ    = L/R
//! courant d'établissement i(t) = I_f·(1 − e^(−t/τ))
//! f.é.m. auto-induite  e    = |L·(di/dt)|
//! ```
//!
//! `L` inductance (H), `I` courant traversant la bobine (A), `E` énergie
//! magnétique stockée (J), `f` fréquence du régime sinusoïdal (Hz), `X_L`
//! réactance inductive (Ω), `R` résistance série du circuit (Ω), `τ` constante
//! de temps du circuit RL (s), `t` temps écoulé depuis la fermeture du circuit
//! (s), `I_f` courant final établi (A), `i(t)` courant dans la bobine pendant
//! l'établissement (A), `di/dt` taux de variation du courant (A/s), `e` force
//! électromotrice auto-induite (V).
//!
//! **Convention** : SI ; inductances en H, courants en A, énergies en J,
//! fréquences en Hz, réactances en Ω, résistances en Ω, temps et constantes de
//! temps en s, f.é.m. en V ; l'argument des exponentielles est **sans
//! dimension** et les angles sont en **radians**. **Limite honnête** :
//! inductance **linéaire idéale** (inductance constante, **sans saturation du
//! noyau** ni hystérésis, résistance de bobinage traitée séparément), circuit
//! **RL du premier ordre** pour le transitoire, et **régime sinusoïdal
//! permanent** pour la réactance. Les grandeurs réseau (fréquence, courant),
//! les valeurs de composant (`L`, `R`) et les grandeurs matériau sont
//! **fournies par l'appelant** (mesures, fiches composant) — aucune valeur
//! « par défaut » n'est inventée.

/// Énergie magnétique stockée `E = 0,5·L·I²` (J).
///
/// Panique si `inductance < 0` (une inductance négative n'a pas de sens
/// physique).
pub fn ind_energy(inductance: f64, current: f64) -> f64 {
    assert!(inductance >= 0.0, "l'inductance L doit être ≥ 0");
    0.5 * inductance * current * current
}

/// Réactance inductive en régime sinusoïdal `X_L = 2·π·f·L` (Ω).
///
/// Panique si `inductance < 0` ou si `frequency < 0`.
pub fn ind_reactance(inductance: f64, frequency: f64) -> f64 {
    assert!(inductance >= 0.0, "l'inductance L doit être ≥ 0");
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0");
    2.0 * core::f64::consts::PI * frequency * inductance
}

/// Constante de temps du circuit RL `τ = L/R` (s).
///
/// Panique si `inductance < 0` ou si `resistance <= 0` (division par zéro).
pub fn ind_rl_time_constant(inductance: f64, resistance: f64) -> f64 {
    assert!(inductance >= 0.0, "l'inductance L doit être ≥ 0");
    assert!(
        resistance > 0.0,
        "la résistance R doit être strictement positive"
    );
    inductance / resistance
}

/// Courant d'établissement dans un circuit RL
/// `i(t) = I_f·(1 − e^(−t/τ))` (A).
///
/// Panique si `time < 0` ou si `time_constant <= 0` (division par zéro).
pub fn ind_current_rise(final_current: f64, time: f64, time_constant: f64) -> f64 {
    assert!(time >= 0.0, "le temps t doit être ≥ 0");
    assert!(
        time_constant > 0.0,
        "la constante de temps τ doit être strictement positive"
    );
    final_current * (1.0 - (-time / time_constant).exp())
}

/// Force électromotrice auto-induite `e = |L·(di/dt)|` (V).
///
/// Panique si `inductance < 0` (une inductance négative n'a pas de sens
/// physique).
pub fn ind_induced_emf(inductance: f64, current_rate_of_change: f64) -> f64 {
    assert!(inductance >= 0.0, "l'inductance L doit être ≥ 0");
    (inductance * current_rate_of_change).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn energy_is_quadratic_in_current() {
        // Proportionnalité : E = 0,5·L·I² est quadratique en I ; doubler le
        // courant quadruple l'énergie magnétique stockée.
        let l = 0.1_f64;
        let e1 = ind_energy(l, 5.0);
        let e2 = ind_energy(l, 10.0);
        assert_relative_eq!(e2, 4.0 * e1, epsilon = 1e-15);
    }

    #[test]
    fn reactance_is_linear_in_frequency() {
        // Proportionnalité : X_L = 2·π·f·L est linéaire en f ; X_L(0) = 0 et
        // doubler la fréquence double la réactance.
        let l = 22.0e-3_f64;
        assert_relative_eq!(ind_reactance(l, 0.0), 0.0, epsilon = 1e-15);
        let x1 = ind_reactance(l, 50.0);
        let x2 = ind_reactance(l, 100.0);
        assert_relative_eq!(x2, 2.0 * x1, epsilon = 1e-15);
    }

    #[test]
    fn realistic_energy_case() {
        // Cas chiffré réaliste, L = 0,1 H parcourue par I = 10 A :
        //   E = 0,5·L·I² = 0,5·0,1·100 = 5,0 J
        assert_relative_eq!(ind_energy(0.1, 10.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_reactance_case() {
        // Cas chiffré réaliste, L = 10 mH à f = 50 Hz :
        //   X_L = 2π·50·0,01 = π ≈ 3,141 592 65 Ω
        let x = ind_reactance(10.0e-3, 50.0);
        assert_relative_eq!(x, core::f64::consts::PI, epsilon = 1e-12);
    }

    #[test]
    fn current_rise_hits_known_fractions() {
        // Cas limites du transitoire RL : i(0) = 0 ; à t = τ le courant atteint
        // (1 − 1/e)·I_f ≈ 0,632 120 6·I_f.
        let i_f = 2.0_f64;
        let tau = 0.01_f64;
        assert_relative_eq!(ind_current_rise(i_f, 0.0, tau), 0.0, epsilon = 1e-15);
        let expected = i_f * (1.0 - core::f64::consts::E.recip());
        assert_relative_eq!(ind_current_rise(i_f, tau, tau), expected, epsilon = 1e-12);
    }

    #[test]
    fn time_constant_and_induced_emf_case() {
        // Cohérence : τ = L/R, et f.é.m. auto-induite pour un courant qui décroît.
        //   τ = 0,1/10 = 0,01 s
        //   e = |L·(di/dt)| = |0,5·(−100)| = 50 V (la valeur absolue efface le signe)
        let tau = ind_rl_time_constant(0.1, 10.0);
        assert_relative_eq!(tau, 0.01, epsilon = 1e-15);
        assert_relative_eq!(ind_induced_emf(0.5, -100.0), 50.0, epsilon = 1e-12);
        assert_relative_eq!(ind_induced_emf(0.5, 100.0), 50.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la résistance R doit être strictement positive")]
    fn zero_resistance_time_constant_panics() {
        ind_rl_time_constant(0.1, 0.0);
    }
}
