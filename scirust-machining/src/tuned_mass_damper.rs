//! Amortisseur à masse accordée (TMD) — **optimisation de Den Hartog** pour une
//! structure principale **non amortie** sous excitation harmonique : accord et
//! amortissement optimaux de l'absorbeur en fonction du rapport de masse.
//!
//! ```text
//! rapport de fréquence optimal   f_opt = 1 / (1 + µ)
//! amortissement optimal          ζ_opt = √( 3·µ / (8·(1 + µ)³) )
//! masse de l'absorbeur           m_a   = µ · m_p
//! raideur de l'absorbeur         k_a   = m_a · (f_opt · ω_p)²
//! ```
//!
//! `µ` rapport de masse m_a/m_p (sans dimension, > 0), `m_p` masse principale
//! (kg), `m_a` masse de l'absorbeur (kg), `ω_p` pulsation propre de la structure
//! principale (rad/s), `f_opt` rapport d'accord f_a/f_p optimal (sans dimension),
//! `ζ_opt` taux d'amortissement optimal de l'absorbeur (sans dimension),
//! `k_a` raideur de l'absorbeur (N/m). `f_opt · ω_p` est la pulsation propre
//! accordée de l'absorbeur (rad/s).
//!
//! **Limite honnête** : formules de Den Hartog valables pour une structure
//! principale **linéaire à 2 ddl, non amortie**, sous **excitation harmonique**
//! (force appliquée sur la masse principale). Le rapport de masse `µ`, la masse
//! principale et la pulsation propre sont **fournis par l'appelant** — aucune
//! valeur « par défaut » de masse, de raideur ou d'amortissement n'est inventée
//! ici. Complète l'isolation à 1 ddl de [`crate::vibration_isolation`].

/// Rapport d'accord optimal `f_opt = 1 / (1 + µ)` (sans dimension).
///
/// Rapport optimal de la fréquence propre de l'absorbeur sur celle de la
/// structure principale (accord de Den Hartog). Tend vers 1 quand `µ → 0`.
///
/// Panique si `mass_ratio <= 0`.
pub fn tmd_optimal_frequency_ratio(mass_ratio: f64) -> f64 {
    assert!(
        mass_ratio > 0.0,
        "le rapport de masse doit être strictement positif"
    );
    1.0 / (1.0 + mass_ratio)
}

/// Amortissement optimal `ζ_opt = √( 3·µ / (8·(1 + µ)³) )` (sans dimension).
///
/// Taux d'amortissement optimal de l'absorbeur (Den Hartog). Croît comme `√µ`
/// pour les petits rapports de masse et tend vers 0 quand `µ → 0`.
///
/// Panique si `mass_ratio <= 0`.
pub fn tmd_optimal_damping_ratio(mass_ratio: f64) -> f64 {
    assert!(
        mass_ratio > 0.0,
        "le rapport de masse doit être strictement positif"
    );
    let one_plus = 1.0 + mass_ratio;
    (3.0 * mass_ratio / (8.0 * one_plus.powi(3))).sqrt()
}

/// Masse de l'absorbeur `m_a = µ · m_p` (kg).
///
/// `mass_ratio` = µ (sans dimension), `primary_mass` = m_p (kg).
///
/// Panique si `mass_ratio <= 0` ou `primary_mass <= 0`.
pub fn tmd_absorber_mass(mass_ratio: f64, primary_mass: f64) -> f64 {
    assert!(
        mass_ratio > 0.0,
        "le rapport de masse doit être strictement positif"
    );
    assert!(
        primary_mass > 0.0,
        "la masse principale doit être strictement positive"
    );
    mass_ratio * primary_mass
}

/// Raideur de l'absorbeur `k_a = m_a · (f_opt · ω_p)²` (N/m).
///
/// `absorber_mass` = m_a (kg), `primary_natural_frequency` = ω_p pulsation propre
/// de la structure principale (rad/s), `optimal_frequency_ratio` = f_opt (sans
/// dimension). `f_opt · ω_p` est la pulsation propre accordée de l'absorbeur.
///
/// Panique si `absorber_mass <= 0`, `primary_natural_frequency <= 0`
/// ou `optimal_frequency_ratio <= 0`.
pub fn tmd_absorber_stiffness(
    absorber_mass: f64,
    primary_natural_frequency: f64,
    optimal_frequency_ratio: f64,
) -> f64 {
    assert!(
        absorber_mass > 0.0,
        "la masse de l'absorbeur doit être strictement positive"
    );
    assert!(
        primary_natural_frequency > 0.0,
        "la pulsation propre principale doit être strictement positive"
    );
    assert!(
        optimal_frequency_ratio > 0.0,
        "le rapport d'accord doit être strictement positif"
    );
    let tuned = optimal_frequency_ratio * primary_natural_frequency;
    absorber_mass * tuned * tuned
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn frequency_ratio_identity() {
        // f_opt·(1 + µ) = 1 par définition, pour tout µ > 0.
        for &mu in &[0.01, 0.05, 0.2, 1.0]
        {
            let f_opt = tmd_optimal_frequency_ratio(mu);
            assert_relative_eq!(f_opt * (1.0 + mu), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn frequency_ratio_tends_to_one() {
        // Un rapport de masse minuscule accorde l'absorbeur ~ sur la structure.
        assert!(tmd_optimal_frequency_ratio(1e-6) < 1.0);
        assert_relative_eq!(tmd_optimal_frequency_ratio(1e-6), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn damping_scales_as_sqrt_mu_for_small_ratio() {
        // Pour µ ≪ 1, ζ_opt ≈ √(3µ/8) ; doubler µ multiplie ζ par ~√2.
        let z1 = tmd_optimal_damping_ratio(1e-4);
        let z2 = tmd_optimal_damping_ratio(2e-4);
        assert_relative_eq!(z2 / z1, 2.0_f64.sqrt(), epsilon = 1e-3);
        // valeur asymptotique √(3µ/8) pour µ = 1e-4
        assert_relative_eq!(z1, (3.0e-4_f64 / 8.0).sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn realistic_case_mass_ratio_five_percent() {
        // µ = 5 %, m_p = 1000 kg, ω_p = 10 rad/s.
        let mu = 0.05;
        let f_opt = tmd_optimal_frequency_ratio(mu);
        assert_relative_eq!(f_opt, 1.0 / 1.05, epsilon = 1e-12);

        // ζ_opt = √(0,15 / (8·1,05³)) = √(0,15 / 9,261) = 0,127267…
        let z_opt = tmd_optimal_damping_ratio(mu);
        assert_relative_eq!(z_opt, (0.15_f64 / 9.261).sqrt(), epsilon = 1e-12);
        assert_relative_eq!(z_opt, 0.127_267_2, epsilon = 1e-6);

        // m_a = 0,05 · 1000 = 50 kg
        let m_a = tmd_absorber_mass(mu, 1000.0);
        assert_relative_eq!(m_a, 50.0, epsilon = 1e-9);

        // k_a = 50 · (f_opt·10)² ; f_opt·10 = 10/1,05 = 200/21 rad/s
        let k_a = tmd_absorber_stiffness(m_a, 10.0, f_opt);
        assert_relative_eq!(k_a, 50.0 * (200.0_f64 / 21.0).powi(2), epsilon = 1e-9);
    }

    #[test]
    fn stiffness_proportional_to_mass_and_frequency_squared() {
        // k_a ∝ m_a et k_a ∝ ω_p² à rapport d'accord fixé.
        let f_opt = tmd_optimal_frequency_ratio(0.1);
        let k1 = tmd_absorber_stiffness(20.0, 8.0, f_opt);
        let k_double_mass = tmd_absorber_stiffness(40.0, 8.0, f_opt);
        let k_double_freq = tmd_absorber_stiffness(20.0, 16.0, f_opt);
        assert_relative_eq!(k_double_mass, 2.0 * k1, epsilon = 1e-9);
        assert_relative_eq!(k_double_freq, 4.0 * k1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le rapport de masse doit être strictement positif")]
    fn zero_mass_ratio_panics() {
        tmd_optimal_damping_ratio(0.0);
    }
}
