//! Relaxation de contrainte en **viscoélasticité** selon le modèle de **Maxwell**
//! (un seul temps de relaxation).
//!
//! ```text
//! contrainte relaxée    σ(t) = σ0 · exp(−t / τ)
//! temps de relaxation   τ    = η / E
//! fraction résiduelle   f(t) = σ(t) / σ0 = exp(−t / τ)
//! ```
//!
//! `σ0` contrainte initiale (Pa), `t` temps écoulé (s), `τ` temps de relaxation
//! (s), `η` viscosité dynamique (Pa·s), `E` module d'Young (Pa), `f` fraction de
//! contrainte restante (sans dimension, dans `[0, 1]`).
//!
//! **Convention** : unités SI (Pa, s, Pa·s). **Limite honnête** : modèle de
//! Maxwell à **un seul** temps de relaxation, adapté aux cas idéalisés ; les
//! matériaux réels suivent souvent un spectre de temps de relaxation. La
//! viscosité `η` et le module `E` sont des **données matériau fournies** par
//! l'appelant, jamais des valeurs « par défaut » inventées. À distinguer du
//! fluage [`crate::creep`] (déformation croissante à contrainte constante), qui
//! est le phénomène dual.

/// Contrainte restante après relaxation `σ(t) = σ0 · exp(−t / τ)` (Pa).
///
/// Panique si `initial_stress < 0`, `time < 0` ou `relaxation_time <= 0`.
pub fn relaxation_stress(initial_stress: f64, time: f64, relaxation_time: f64) -> f64 {
    assert!(
        initial_stress >= 0.0,
        "la contrainte initiale doit être positive"
    );
    assert!(time >= 0.0, "le temps doit être positif");
    assert!(
        relaxation_time > 0.0,
        "le temps de relaxation doit être strictement positif"
    );
    initial_stress * (-time / relaxation_time).exp()
}

/// Temps de relaxation du modèle de Maxwell `τ = η / E` (s).
///
/// Panique si `viscosity < 0` ou `youngs_modulus <= 0`.
pub fn relaxation_time_constant(viscosity: f64, youngs_modulus: f64) -> f64 {
    assert!(viscosity >= 0.0, "la viscosité doit être positive");
    assert!(
        youngs_modulus > 0.0,
        "le module d'Young doit être strictement positif"
    );
    viscosity / youngs_modulus
}

/// Fraction de contrainte restante `f(t) = exp(−t / τ)` (sans dimension).
///
/// Panique si `time < 0` ou `relaxation_time <= 0`.
pub fn relaxation_remaining_stress_fraction(time: f64, relaxation_time: f64) -> f64 {
    assert!(time >= 0.0, "le temps doit être positif");
    assert!(
        relaxation_time > 0.0,
        "le temps de relaxation doit être strictement positif"
    );
    (-time / relaxation_time).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stress_at_zero_time_equals_initial() {
        // À t = 0, exp(0) = 1 : la contrainte vaut exactement σ0.
        assert_relative_eq!(
            relaxation_stress(200.0e6, 0.0, 50.0),
            200.0e6,
            max_relative = 1e-12
        );
    }

    #[test]
    fn one_time_constant_leaves_reciprocal_of_e() {
        // À t = τ, la fraction résiduelle vaut exp(−1) = 1/e ≈ 0.367879441.
        let f = relaxation_remaining_stress_fraction(50.0, 50.0);
        assert_relative_eq!(f, core::f64::consts::E.recip(), max_relative = 1e-12);
    }

    #[test]
    fn stress_is_fraction_times_initial() {
        // Identité : σ(t) = σ0 · f(t) pour tout t.
        let sigma0 = 150.0e6;
        let t = 30.0;
        let tau = 80.0;
        let sigma = relaxation_stress(sigma0, t, tau);
        let f = relaxation_remaining_stress_fraction(t, tau);
        assert_relative_eq!(sigma, sigma0 * f, max_relative = 1e-12);
    }

    #[test]
    fn time_constant_is_viscosity_over_modulus() {
        // τ = η / E : η = 2.1e12 Pa·s, E = 2.1e11 Pa → τ = 10 s.
        let tau = relaxation_time_constant(2.1e12, 2.1e11);
        assert_relative_eq!(tau, 10.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_worked_case() {
        // σ0 = 100 MPa, E = 200 GPa, η = 4.0e11 Pa·s ⇒ τ = η/E = 2.0 s.
        // À t = 2τ = 4 s : σ = 100 MPa · exp(−2) ≈ 100e6 · 0.1353352832
        //                    ≈ 13.53352832 MPa.
        let tau = relaxation_time_constant(4.0e11, 200.0e9);
        assert_relative_eq!(tau, 2.0, max_relative = 1e-12);
        let sigma = relaxation_stress(100.0e6, 4.0, tau);
        assert_relative_eq!(sigma, 100.0e6 * (-2.0_f64).exp(), max_relative = 1e-12);
        assert_relative_eq!(sigma, 13_533_528.32, max_relative = 1e-6);
    }

    #[test]
    fn fraction_is_monotonically_decreasing() {
        // La fraction résiduelle décroît strictement avec le temps.
        let f1 = relaxation_remaining_stress_fraction(10.0, 50.0);
        let f2 = relaxation_remaining_stress_fraction(20.0, 50.0);
        assert!(f2 < f1 && f1 <= 1.0);
    }

    #[test]
    #[should_panic(expected = "le temps de relaxation doit être strictement positif")]
    fn zero_relaxation_time_panics() {
        relaxation_stress(200.0e6, 10.0, 0.0);
    }
}
