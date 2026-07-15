//! Isolation vibratoire par plots élastiques — **méthode de la flèche statique**
//! pour un système à 1 ddl **non amorti** : fréquence propre déduite de
//! l'affaissement des plots, transmissibilité et efficacité d'isolation.
//!
//! ```text
//! fréquence propre    fn = (1/(2·π))·√(g/δ)
//! rapport de fréquence r  = f/fn
//! transmissibilité    T  = 1/|r² − 1|          (amortissement négligé)
//! efficacité          η  = 1 − T
//! isolation           r > √2  ⇒  T < 1  ⇒  η > 0
//! ```
//!
//! `δ` flèche statique des plots sous la charge (m), `g` accélération de la
//! pesanteur (m/s²), `fn` fréquence propre (Hz), `f` fréquence d'excitation (Hz),
//! `r` rapport de fréquence (sans dimension), `T` transmissibilité (sans
//! dimension, force transmise/force appliquée), `η` efficacité d'isolation (sans
//! dimension, ∈ ]0, 1[ dans la zone utile).
//!
//! **Limite honnête** : modèle **linéaire à 1 ddl non amorti** ; l'amortissement
//! est **négligé** (T diverge à la résonance `r = 1` et reste surestimée près de
//! celle-ci). L'isolation n'est effective que pour `r > √2`. Les constantes
//! physiques (`g`), les propriétés des plots et la charge sont **fournies par
//! l'appelant** — aucune valeur « par défaut » n'est inventée ici. Pour le modèle
//! amorti complet, voir [`crate::forced_vibrations`].

/// Fréquence propre par la méthode de la flèche statique
/// `fn = (1/(2·π))·√(g/δ)` (Hz).
///
/// `static_deflection` = flèche statique des plots δ (m), `gravity` = g (m/s²) ;
/// l'appelant fournit `gravity` (p. ex. 9,80665 pour la pesanteur normale).
///
/// Panique si `static_deflection <= 0` ou `gravity <= 0`.
pub fn iso_natural_frequency_from_deflection(static_deflection: f64, gravity: f64) -> f64 {
    assert!(
        static_deflection > 0.0,
        "la flèche statique doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur doit être strictement positive"
    );
    (gravity / static_deflection).sqrt() / core::f64::consts::TAU
}

/// Transmissibilité non amortie `T = 1/|r² − 1|` (sans dimension).
///
/// `r = f/fn`. Pour `r < 1` (raide, T > 1 avec amplification) comme pour
/// `r > √2` (souple, T < 1, isolation), la valeur absolue garde `T > 0`.
///
/// Panique si `r < 0` ou si `r == 1` (résonance : T diverge sans amortissement).
pub fn iso_transmissibility(frequency_ratio: f64) -> f64 {
    assert!(
        frequency_ratio >= 0.0,
        "le rapport de fréquence doit être ≥ 0"
    );
    let denom = (frequency_ratio * frequency_ratio - 1.0).abs();
    assert!(
        denom > 0.0,
        "résonance r = 1 : transmissibilité infinie sans amortissement"
    );
    1.0 / denom
}

/// Efficacité d'isolation `η = 1 − T` (sans dimension).
///
/// Positive (isolation utile) seulement lorsque `T < 1`, c.-à-d. `r > √2`.
/// Négative si `T > 1` (amplification en deçà de la résonance).
///
/// Panique si `transmissibility < 0`.
pub fn iso_efficiency(transmissibility: f64) -> f64 {
    assert!(transmissibility >= 0.0, "la transmissibilité doit être ≥ 0");
    1.0 - transmissibility
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn transmissibility_unit_at_root_two() {
        // À r = √2 : |r² − 1| = 1 donc T = 1 (frontière isolation/amplification).
        let r = core::f64::consts::SQRT_2;
        assert_relative_eq!(iso_transmissibility(r), 1.0, epsilon = 1e-12);
        // Et donc η = 0 exactement à cette frontière.
        assert_relative_eq!(
            iso_efficiency(iso_transmissibility(r)),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn worked_case_ratio_two() {
        // r = 2 : T = 1/|4 − 1| = 1/3 ; η = 1 − 1/3 = 2/3.
        let t = iso_transmissibility(2.0);
        assert_relative_eq!(t, 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(iso_efficiency(t), 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_and_transmissibility_are_complementary() {
        // Identité η + T = 1 pour tout r admissible (r ≠ 1).
        for &r in &[0.5_f64, 1.5, 2.0, 3.0, 5.0]
        {
            let t = iso_transmissibility(r);
            assert_relative_eq!(iso_efficiency(t) + t, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn natural_frequency_worked_case() {
        // δ = 0,01 m, g = 9,81 m/s² : fn = √(981)/(2π) ≈ 4,9849 Hz.
        let f = iso_natural_frequency_from_deflection(0.01, 9.81);
        let expected = (9.81_f64 / 0.01).sqrt() / core::f64::consts::TAU;
        assert_relative_eq!(f, expected, epsilon = 1e-12);
        assert_relative_eq!(f, 4.984_879_f64, epsilon = 1e-6);
    }

    #[test]
    fn natural_frequency_scales_as_inverse_sqrt_deflection() {
        // fn ∝ 1/√δ : quadrupler la flèche divise la fréquence propre par 2.
        let f1 = iso_natural_frequency_from_deflection(0.01, 9.81);
        let f4 = iso_natural_frequency_from_deflection(0.04, 9.81);
        assert_relative_eq!(f1 / f4, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "résonance")]
    fn transmissibility_panics_at_resonance() {
        iso_transmissibility(1.0);
    }
}
