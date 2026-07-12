//! Fiabilité — distribution de **Weibull** (taux de défaillance variable) :
//! fiabilité, taux de hasard et durées de vie `Bx`.
//!
//! ```text
//! fiabilité      R(t) = e^{−(t/η)^β}
//! taux de hasard h(t) = (β/η)·(t/η)^{β−1}
//! vie Bx         t = η·(−ln R)^{1/β}      (Bx : R = 1 − x/100)
//! ```
//!
//! `η` paramètre d'échelle (**vie caractéristique**, `R(η) = 1/e ≈ 63,2 % de
//! défaillances), `β` paramètre de forme (`β < 1` mortalité infantile, `β = 1`
//! aléatoire/exponentiel, `β > 1` usure), `t` durée, `h` taux de hasard
//! instantané. La vie `B10` est la durée à laquelle 10 % des éléments ont
//! défailli (`R = 0,9`).
//!
//! **Convention** : temps cohérents. **Limite honnête** : distribution à **deux
//! paramètres** (paramètre de position nul) ; `η` et `β` proviennent d'un
//! ajustement (papier de Weibull, MLE) fourni par l'appelant. `β = 1` redonne le
//! modèle exponentiel de [`crate::reliability`].

/// Fiabilité de Weibull `R(t) = e^{−(t/η)^β}`.
///
/// Panique si `scale <= 0`, `shape <= 0` ou `time < 0`.
pub fn weibull_reliability(time: f64, scale: f64, shape: f64) -> f64 {
    assert!(
        scale > 0.0 && shape > 0.0 && time >= 0.0,
        "η > 0, β > 0 et t ≥ 0 requis"
    );
    (-(time / scale).powf(shape)).exp()
}

/// Taux de hasard `h(t) = (β/η)·(t/η)^{β−1}`.
///
/// Panique si `scale <= 0`, `shape <= 0` ou `time < 0`.
pub fn weibull_hazard_rate(time: f64, scale: f64, shape: f64) -> f64 {
    assert!(
        scale > 0.0 && shape > 0.0 && time >= 0.0,
        "η > 0, β > 0 et t ≥ 0 requis"
    );
    (shape / scale) * (time / scale).powf(shape - 1.0)
}

/// Durée de vie `Bx` `t = η·(−ln R)^{1/β}` pour une fiabilité `R` cible.
///
/// Panique si `reliability` n'est pas dans `]0, 1]`.
pub fn weibull_b_life(scale: f64, shape: f64, reliability: f64) -> f64 {
    assert!(
        reliability > 0.0 && reliability <= 1.0,
        "la fiabilité cible doit être dans ]0, 1]"
    );
    scale * (-reliability.ln()).powf(1.0 / shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn beta_one_reduces_to_exponential() {
        // β=1 → R = e^{−t/η}, soit exponentiel de taux λ = 1/η.
        let (eta, t) = (10_000.0, 3000.0);
        assert_relative_eq!(
            weibull_reliability(t, eta, 1.0),
            (-t / eta).exp(),
            epsilon = 1e-12
        );
        // taux de hasard constant = 1/η.
        assert_relative_eq!(weibull_hazard_rate(t, eta, 1.0), 1.0 / eta, epsilon = 1e-12);
    }

    #[test]
    fn characteristic_life_gives_63_percent_failure() {
        // À t = η, R = 1/e quelle que soit β.
        for &beta in &[0.5, 1.0, 2.5]
        {
            assert_relative_eq!(
                weibull_reliability(5000.0, 5000.0, beta),
                1.0 / core::f64::consts::E,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn wear_out_has_increasing_hazard() {
        // β>1 (usure) : le taux de hasard croît avec le temps.
        assert!(
            weibull_hazard_rate(8000.0, 10_000.0, 2.0) > weibull_hazard_rate(2000.0, 10_000.0, 2.0)
        );
        // β<1 (jeunesse) : il décroît.
        assert!(
            weibull_hazard_rate(8000.0, 10_000.0, 0.5) < weibull_hazard_rate(2000.0, 10_000.0, 0.5)
        );
    }

    #[test]
    fn b10_life_below_characteristic() {
        // B10 (R=0,9) est bien inférieur à la vie caractéristique η.
        let b10 = weibull_b_life(10_000.0, 1.5, 0.9);
        assert!(b10 < 10_000.0);
        // vérifie la cohérence : R(B10) = 0,9.
        assert_relative_eq!(
            weibull_reliability(b10, 10_000.0, 1.5),
            0.9,
            max_relative = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "η > 0")]
    fn zero_scale_panics() {
        weibull_reliability(1000.0, 0.0, 2.0);
    }
}
