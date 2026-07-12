//! Régulateur **PID** — sortie parallèle, conversion forme standard → parallèle
//! et réglage de **Ziegler-Nichols** (méthode du gain critique).
//!
//! ```text
//! sortie parallèle  u = Kp·e + Ki·∫e·dt + Kd·(de/dt)
//! standard→parallèle Ki = Kp/Ti     Kd = Kp·Td
//! Ziegler-Nichols PID  Kp = 0,6·Ku   Ti = 0,5·Tu   Td = 0,125·Tu
//! ```
//!
//! `Kp` gain proportionnel, `Ki` gain intégral, `Kd` gain dérivé, `e` erreur,
//! `∫e` intégrale de l'erreur, `de/dt` dérivée de l'erreur, `Ti`/`Td` temps
//! intégral/dérivé (s) de la forme standard, `Ku` gain critique (au seuil
//! d'oscillation), `Tu` période des oscillations entretenues (s).
//!
//! **Convention** : forme **parallèle** pour la sortie ; temps en s. **Limite
//! honnête** : PID **idéal** (sans filtrage de la dérivée ni anti-emballement de
//! l'intégrale) ; les gains de Ziegler-Nichols sont un **point de départ**
//! agressif (fort dépassement), à affiner. `Ku`, `Tu` sont mesurés par
//! l'appelant.

/// Sortie du PID parallèle `u = Kp·e + Ki·∫e + Kd·(de/dt)`.
pub fn pid_output(kp: f64, ki: f64, kd: f64, error: f64, integral: f64, derivative: f64) -> f64 {
    kp * error + ki * integral + kd * derivative
}

/// Gain intégral parallèle depuis la forme standard `Ki = Kp/Ti`.
///
/// Panique si `integral_time <= 0`.
pub fn integral_gain(kp: f64, integral_time: f64) -> f64 {
    assert!(
        integral_time > 0.0,
        "le temps intégral doit être strictement positif"
    );
    kp / integral_time
}

/// Gain dérivé parallèle depuis la forme standard `Kd = Kp·Td`.
///
/// Panique si `derivative_time < 0`.
pub fn derivative_gain(kp: f64, derivative_time: f64) -> f64 {
    assert!(derivative_time >= 0.0, "le temps dérivé doit être positif");
    kp * derivative_time
}

/// Gain proportionnel de Ziegler-Nichols (PID) `Kp = 0,6·Ku`.
///
/// Panique si `ultimate_gain <= 0`.
pub fn ziegler_nichols_kp(ultimate_gain: f64) -> f64 {
    assert!(
        ultimate_gain > 0.0,
        "le gain critique doit être strictement positif"
    );
    0.6 * ultimate_gain
}

/// Temps intégral de Ziegler-Nichols (PID) `Ti = 0,5·Tu`.
///
/// Panique si `ultimate_period <= 0`.
pub fn ziegler_nichols_ti(ultimate_period: f64) -> f64 {
    assert!(
        ultimate_period > 0.0,
        "la période critique doit être strictement positive"
    );
    0.5 * ultimate_period
}

/// Temps dérivé de Ziegler-Nichols (PID) `Td = 0,125·Tu`.
///
/// Panique si `ultimate_period <= 0`.
pub fn ziegler_nichols_td(ultimate_period: f64) -> f64 {
    assert!(
        ultimate_period > 0.0,
        "la période critique doit être strictement positive"
    );
    0.125 * ultimate_period
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_sums_three_actions() {
        // Kp=2, Ki=0,5, Kd=0,1 ; e=3, ∫e=4, de=−1 → 6 + 2 − 0,1 = 7,9.
        assert_relative_eq!(
            pid_output(2.0, 0.5, 0.1, 3.0, 4.0, -1.0),
            7.9,
            epsilon = 1e-9
        );
    }

    #[test]
    fn pure_proportional_when_gains_zero() {
        assert_relative_eq!(
            pid_output(2.0, 0.0, 0.0, 3.0, 100.0, 100.0),
            6.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn standard_to_parallel_conversion() {
        // Kp=3, Ti=2 → Ki=1,5 ; Td=0,5 → Kd=1,5.
        assert_relative_eq!(integral_gain(3.0, 2.0), 1.5, epsilon = 1e-9);
        assert_relative_eq!(derivative_gain(3.0, 0.5), 1.5, epsilon = 1e-9);
    }

    #[test]
    fn ziegler_nichols_ratios() {
        // Ku=4, Tu=2 s → Kp=2,4 ; Ti=1 s ; Td=0,25 s.
        assert_relative_eq!(ziegler_nichols_kp(4.0), 2.4, epsilon = 1e-9);
        assert_relative_eq!(ziegler_nichols_ti(2.0), 1.0, epsilon = 1e-9);
        assert_relative_eq!(ziegler_nichols_td(2.0), 0.25, epsilon = 1e-9);
        // Td = Ti/4 pour le réglage PID classique.
        assert_relative_eq!(
            ziegler_nichols_td(2.0),
            ziegler_nichols_ti(2.0) / 4.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "gain critique")]
    fn zero_ultimate_gain_panics() {
        ziegler_nichols_kp(0.0);
    }
}
