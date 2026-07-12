//! Fiabilité — **essais de vie accélérée** : facteur d'accélération d'**Arrhenius**
//! (température), règle des « 10 °C » et déclassement (derating).
//!
//! ```text
//! Arrhenius   AF = exp[ (Ea/k)·(1/T_usage − 1/T_stress) ]
//! règle 10°C  AF = 2^{ΔT/10}        (la vitesse double / la vie halve par +10 °C)
//! déclassement σ_admissible = σ_nominale · f_derating
//! ```
//!
//! `Ea` énergie d'activation (eV), `k` constante de Boltzmann (8,617·10⁻⁵ eV/K),
//! `T` températures **absolues** (K), `AF` facteur d'accélération (rapport des
//! durées de vie usage/essai), `ΔT` écart de température (°C), `f_derating`
//! facteur de déclassement (`< 1`).
//!
//! **Convention** : températures en **kelvin** pour Arrhenius, écart en °C pour la
//! règle empirique. **Limite honnête** : modèle d'**Arrhenius** (un seul mécanisme
//! de défaillance thermiquement activé) ; `Ea` est une donnée du mécanisme fournie
//! par l'appelant. La règle des 10 °C est une approximation (`Ea ≈ 0,5`–`0,7 eV`).

/// Constante de Boltzmann en eV/K.
pub const BOLTZMANN_EV_K: f64 = 8.617_333e-5;

/// Facteur d'accélération d'Arrhenius
/// `AF = exp[(Ea/k)·(1/T_usage − 1/T_stress)]`.
///
/// Panique si une température `<= 0`.
pub fn arrhenius_acceleration_factor(
    activation_energy_ev: f64,
    temp_use_k: f64,
    temp_stress_k: f64,
) -> f64 {
    assert!(
        temp_use_k > 0.0 && temp_stress_k > 0.0,
        "les températures absolues doivent être strictement positives"
    );
    ((activation_energy_ev / BOLTZMANN_EV_K) * (1.0 / temp_use_k - 1.0 / temp_stress_k)).exp()
}

/// Facteur d'accélération par la **règle des 10 °C** `AF = 2^{ΔT/10}`.
pub fn ten_degree_rule_factor(delta_t_celsius: f64) -> f64 {
    2.0_f64.powf(delta_t_celsius / 10.0)
}

/// Contrainte admissible après **déclassement** `σ_adm = σ_nom·f`.
///
/// Panique si `derating_factor` sort de `]0, 1]`.
pub fn derated_value(rated_value: f64, derating_factor: f64) -> f64 {
    assert!(
        derating_factor > 0.0 && derating_factor <= 1.0,
        "le facteur de déclassement doit être dans ]0, 1]"
    );
    rated_value * derating_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn higher_stress_temperature_accelerates() {
        // T_stress > T_usage → AF > 1 (l'essai est accéléré).
        let af = arrhenius_acceleration_factor(0.7, 300.0, 400.0);
        assert!(af > 1.0);
        assert_relative_eq!(
            af,
            ((0.7 / BOLTZMANN_EV_K) * (1.0 / 300.0 - 1.0 / 400.0)).exp(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn equal_temperatures_give_unity() {
        // T_usage = T_stress → AF = 1 (aucune accélération).
        assert_relative_eq!(
            arrhenius_acceleration_factor(0.7, 350.0, 350.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ten_degree_rule_doubles_per_step() {
        // +10 °C → ×2 ; +20 °C → ×4.
        assert_relative_eq!(ten_degree_rule_factor(10.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(ten_degree_rule_factor(20.0), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn derating_reduces_allowable() {
        // Déclasser à 50 % → contrainte admissible divisée par deux.
        assert_relative_eq!(derated_value(400e6, 0.5), 200e6, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "températures absolues")]
    fn zero_temperature_panics() {
        arrhenius_acceleration_factor(0.7, 0.0, 400.0);
    }
}
