//! Fiabilité — modèle **exponentiel** (taux de défaillance constant) : fiabilité,
//! probabilité de défaillance, MTBF et estimation par essai.
//!
//! ```text
//! fiabilité         R(t) = e^{−λ·t}
//! défaillance       F(t) = 1 − e^{−λ·t}
//! MTBF              MTBF = 1/λ
//! λ depuis MTBF     λ = 1/MTBF
//! MTBF par essai    MTBF = T_cumulé/n_défaillances
//! ```
//!
//! `λ` taux de défaillance (1/temps), `t` durée (temps), `R` fiabilité (probabilité
//! de survie), `F` probabilité de défaillance, `MTBF` temps moyen entre
//! défaillances. Le modèle exponentiel correspond au fond de la « courbe en
//! baignoire » (défaillances aléatoires, taux constant).
//!
//! **Convention** : temps cohérents. **Limite honnête** : taux de défaillance
//! **constant** (sans usure ni jeunesse) ; pour un taux variable, voir
//! [`crate::weibull`]. `λ`/MTBF sont des données de fiabilité fournies par
//! l'appelant.

/// Fiabilité `R(t) = e^{−λ·t}`.
///
/// Panique si `failure_rate < 0` ou `time < 0`.
pub fn exponential_reliability(failure_rate: f64, time: f64) -> f64 {
    assert!(failure_rate >= 0.0 && time >= 0.0, "λ ≥ 0 et t ≥ 0 requis");
    (-failure_rate * time).exp()
}

/// Probabilité de défaillance `F(t) = 1 − e^{−λ·t}`.
///
/// Panique si `failure_rate < 0` ou `time < 0`.
pub fn probability_of_failure(failure_rate: f64, time: f64) -> f64 {
    1.0 - exponential_reliability(failure_rate, time)
}

/// Taux de défaillance depuis le MTBF `λ = 1/MTBF`.
///
/// Panique si `mtbf <= 0`.
pub fn failure_rate_from_mtbf(mtbf: f64) -> f64 {
    assert!(mtbf > 0.0, "le MTBF doit être strictement positif");
    1.0 / mtbf
}

/// MTBF depuis le taux de défaillance `MTBF = 1/λ`.
///
/// Panique si `failure_rate <= 0`.
pub fn mtbf_from_failure_rate(failure_rate: f64) -> f64 {
    assert!(
        failure_rate > 0.0,
        "le taux de défaillance doit être strictement positif"
    );
    1.0 / failure_rate
}

/// MTBF estimé par essai `MTBF = T_cumulé/n` (n défaillances).
///
/// Panique si `failures == 0`.
pub fn mtbf_from_test(total_operating_time: f64, failures: u32) -> f64 {
    assert!(
        failures > 0,
        "au moins une défaillance est requise pour estimer le MTBF"
    );
    total_operating_time / failures as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reliability_starts_at_one() {
        // R(0) = 1 ; R décroît avec le temps.
        assert_relative_eq!(exponential_reliability(1e-4, 0.0), 1.0, epsilon = 1e-12);
        assert!(exponential_reliability(1e-4, 5000.0) < 1.0);
    }

    #[test]
    fn reliability_at_mtbf_is_one_over_e() {
        // À t = MTBF (λt = 1) → R = 1/e ≈ 0,368.
        let lambda = failure_rate_from_mtbf(10_000.0);
        assert_relative_eq!(
            exponential_reliability(lambda, 10_000.0),
            1.0 / core::f64::consts::E,
            epsilon = 1e-9
        );
    }

    #[test]
    fn reliability_and_failure_sum_to_one() {
        let (l, t) = (2e-4, 3000.0);
        assert_relative_eq!(
            exponential_reliability(l, t) + probability_of_failure(l, t),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn mtbf_and_rate_are_inverse() {
        assert_relative_eq!(
            mtbf_from_failure_rate(failure_rate_from_mtbf(8000.0)),
            8000.0,
            epsilon = 1e-9
        );
        // essai : 50000 h cumulées, 5 défaillances → MTBF = 10000 h.
        assert_relative_eq!(mtbf_from_test(50_000.0, 5), 10_000.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "MTBF doit être")]
    fn zero_mtbf_panics() {
        failure_rate_from_mtbf(0.0);
    }
}
