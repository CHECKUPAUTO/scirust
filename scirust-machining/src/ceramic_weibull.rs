//! Statistique de rupture des matériaux fragiles (céramiques) — modèle de
//! **Weibull** du maillon le plus faible.
//!
//! ```text
//! survie          Ps(σ) = exp(−(σ/σ0)^m)
//! effet d'échelle σ2 = σ1·(V1/V2)^(1/m)
//! contrainte      σ = σ0·(−ln Ps)^(1/m)   (inverse de la survie)
//! ```
//!
//! `σ` contrainte appliquée (Pa), `σ0` contrainte caractéristique (Pa,
//! `Ps(σ0) = 1/e ≈ 36,8 % de survie`), `m` module de Weibull (sans dimension,
//! dispersion de la résistance : `m` élevé = matériau homogène), `Ps`
//! probabilité de survie (sans dimension, dans `]0, 1]`), `V1`/`V2` volumes (ou
//! surfaces) effectivement sollicités (même unité, m³).
//!
//! **Convention** : unités SI cohérentes (`σ` et `σ0` dans la même unité ; `V1`
//! et `V2` dans la même unité). **Limite honnête** : Weibull à **deux
//! paramètres** (contrainte seuil nulle), hypothèse du maillon le plus faible et
//! sollicitation uniforme ; le module `m` et la contrainte caractéristique `σ0`
//! proviennent d'essais fournis par l'appelant (aucune valeur matériau « par
//! défaut » n'est inventée). Distinct de [`crate::weibull`] qui traite la
//! fiabilité temporelle (durée de vie).

/// Probabilité de survie `Ps(σ) = exp(−(σ/σ0)^m)` d'une pièce fragile sous la
/// contrainte `stress`.
///
/// `stress` et `characteristic_strength` dans la même unité (Pa) ; résultat
/// dans `]0, 1]`.
///
/// Panique si `stress < 0`, `characteristic_strength <= 0` ou
/// `weibull_modulus <= 0`.
pub fn ceramic_weibull_survival_probability(
    stress: f64,
    characteristic_strength: f64,
    weibull_modulus: f64,
) -> f64 {
    assert!(
        stress >= 0.0,
        "la contrainte appliquée doit être positive ou nulle"
    );
    assert!(
        characteristic_strength > 0.0,
        "la contrainte caractéristique σ0 doit être strictement positive"
    );
    assert!(
        weibull_modulus > 0.0,
        "le module de Weibull m doit être strictement positif"
    );
    (-(stress / characteristic_strength).powf(weibull_modulus)).exp()
}

/// Probabilité de rupture `Pf(σ) = 1 − Ps(σ) = 1 − exp(−(σ/σ0)^m)`.
///
/// `stress` et `characteristic_strength` dans la même unité (Pa) ; résultat
/// dans `[0, 1[`.
///
/// Panique si `stress < 0`, `characteristic_strength <= 0` ou
/// `weibull_modulus <= 0`.
pub fn ceramic_weibull_failure_probability(
    stress: f64,
    characteristic_strength: f64,
    weibull_modulus: f64,
) -> f64 {
    1.0 - ceramic_weibull_survival_probability(stress, characteristic_strength, weibull_modulus)
}

/// Effet d'échelle de Weibull `σ2 = σ1·(V1/V2)^(1/m)` : résistance attendue d'un
/// volume `volume2` à partir de la résistance mesurée `strength1` sur un volume
/// `volume1` (un volume plus grand contient plus de défauts, donc résiste moins).
///
/// `volume1` et `volume2` dans la même unité (m³) ; `strength1` en Pa.
///
/// Panique si `strength1 < 0`, `volume1 <= 0`, `volume2 <= 0` ou
/// `weibull_modulus <= 0`.
pub fn ceramic_weibull_size_effect(
    strength1: f64,
    volume1: f64,
    volume2: f64,
    weibull_modulus: f64,
) -> f64 {
    assert!(
        strength1 >= 0.0,
        "la résistance de référence doit être positive ou nulle"
    );
    assert!(
        volume1 > 0.0 && volume2 > 0.0,
        "les volumes sollicités V1 et V2 doivent être strictement positifs"
    );
    assert!(
        weibull_modulus > 0.0,
        "le module de Weibull m doit être strictement positif"
    );
    strength1 * (volume1 / volume2).powf(1.0 / weibull_modulus)
}

/// Contrainte admissible pour une survie cible `σ = σ0·(−ln Ps)^(1/m)` (inverse
/// de [`ceramic_weibull_survival_probability`]).
///
/// `characteristic_strength` en Pa ; `survival_probability` dans `]0, 1]` ;
/// résultat en Pa (`Ps = 1` donne `σ = 0`).
///
/// Panique si `characteristic_strength <= 0`, `weibull_modulus <= 0` ou
/// `survival_probability` hors de `]0, 1]`.
pub fn ceramic_weibull_stress_for_probability(
    characteristic_strength: f64,
    weibull_modulus: f64,
    survival_probability: f64,
) -> f64 {
    assert!(
        characteristic_strength > 0.0,
        "la contrainte caractéristique σ0 doit être strictement positive"
    );
    assert!(
        weibull_modulus > 0.0,
        "le module de Weibull m doit être strictement positif"
    );
    assert!(
        survival_probability > 0.0 && survival_probability <= 1.0,
        "la probabilité de survie doit être dans ]0, 1]"
    );
    characteristic_strength * (-survival_probability.ln()).powf(1.0 / weibull_modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    /// À `σ = σ0`, la survie vaut `1/e ≈ 36,8 %` (définition de σ0).
    #[test]
    fn survival_at_characteristic_strength_is_one_over_e() {
        let ps = ceramic_weibull_survival_probability(350.0e6, 350.0e6, 10.0);
        assert_relative_eq!(ps, 1.0 / E, epsilon = 1e-12);
    }

    /// Survie et contrainte admissible sont réciproques (aller-retour).
    #[test]
    fn survival_and_stress_are_inverse() {
        let sigma0 = 400.0e6;
        let m = 12.0;
        let sigma = 310.0e6;
        let ps = ceramic_weibull_survival_probability(sigma, sigma0, m);
        let sigma_back = ceramic_weibull_stress_for_probability(sigma0, m, ps);
        assert_relative_eq!(sigma_back, sigma, epsilon = 1e-3);
    }

    /// Cas chiffré : σ/σ0 = 6/7, m = 10 ⇒ (6/7)^10 = 0,2140599,
    /// Ps = exp(−0,2140599) = 0,8073013.
    #[test]
    fn survival_numeric_case() {
        let ps = ceramic_weibull_survival_probability(300.0e6, 350.0e6, 10.0);
        assert_relative_eq!(ps, 0.807_301_3, epsilon = 1e-6);
    }

    /// Effet d'échelle : volumes égaux ⇒ résistance inchangée.
    #[test]
    fn size_effect_equal_volumes_is_identity() {
        let s = ceramic_weibull_size_effect(500.0e6, 2.0e-6, 2.0e-6, 8.0);
        assert_relative_eq!(s, 500.0e6, epsilon = 1e-6);
    }

    /// Effet d'échelle réciproque : V1→V2 puis V2→V1 redonne σ1.
    #[test]
    fn size_effect_is_reversible() {
        let s1 = 350.0e6;
        let s2 = ceramic_weibull_size_effect(s1, 1.0e-6, 1.0e-3, 10.0);
        let s1_back = ceramic_weibull_size_effect(s2, 1.0e-3, 1.0e-6, 10.0);
        assert_relative_eq!(s1_back, s1, epsilon = 1e-3);
    }

    /// Cas chiffré effet d'échelle : V1/V2 = 1e-3, m = 10 ⇒ (1e-3)^0,1 = 10^−0,3
    /// = 0,5011872 ⇒ σ2 = 350·0,5011872 = 175,4155 MPa.
    #[test]
    fn size_effect_numeric_case() {
        let s2 = ceramic_weibull_size_effect(350.0e6, 1.0e-6, 1.0e-3, 10.0);
        assert_relative_eq!(s2, 175.4155e6, epsilon = 1.0e2);
    }

    /// Survie et rupture sont complémentaires.
    #[test]
    fn survival_plus_failure_is_one() {
        let ps = ceramic_weibull_survival_probability(280.0e6, 350.0e6, 9.0);
        let pf = ceramic_weibull_failure_probability(280.0e6, 350.0e6, 9.0);
        assert_relative_eq!(ps + pf, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le module de Weibull m doit être strictement positif")]
    fn negative_modulus_panics() {
        let _ = ceramic_weibull_survival_probability(300.0e6, 350.0e6, -1.0);
    }
}
