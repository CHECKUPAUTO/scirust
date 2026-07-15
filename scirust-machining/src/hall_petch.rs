//! Renforcement par taille de grain — relation empirique de **Hall-Petch** :
//! limite d'élasticité, taille de grain réciproque et identification du coefficient.
//!
//! ```text
//! limite élastique     sigma_y = sigma0 + k·d^(-1/2)
//! taille de grain      d       = (k/(sigma_y - sigma0))^2        (réciproque)
//! coefficient k        k       = (sigma2 - sigma1)/(d2^(-1/2) - d1^(-1/2))
//! ```
//!
//! `sigma_y` limite d'élasticité (Pa), `sigma0` contrainte de friction du réseau
//! (Pa), `k` coefficient de Hall-Petch (Pa·m^(1/2)), `d` diamètre moyen de grain
//! (m). Les indices `1`/`2` désignent deux états de grain mesurés servant à
//! identifier `k`.
//!
//! **Convention** : SI ; contraintes en Pa, diamètre en m, `k` en Pa·m^(1/2).
//! **Limite honnête** : relation **empirique** de Hall-Petch ; la contrainte de
//! friction `sigma0` et le coefficient `k` sont des propriétés du **matériau**
//! FOURNIES par l'appelant (issues d'essais), jamais de valeur « par défaut »
//! inventée. Le modèle est invalide pour les grains **nanométriques**, où l'effet
//! s'inverse (Hall-Petch inverse) faute de mécanisme d'empilement de dislocations.

/// Limite d'élasticité `sigma_y = sigma0 + k·d^(-1/2)`.
///
/// Panique si `friction_stress < 0`, `strengthening_coefficient < 0`
/// ou `grain_diameter <= 0`.
pub fn hall_petch_yield_strength(
    friction_stress: f64,
    strengthening_coefficient: f64,
    grain_diameter: f64,
) -> f64 {
    assert!(
        friction_stress >= 0.0 && strengthening_coefficient >= 0.0 && grain_diameter > 0.0,
        "sigma0 ≥ 0, k ≥ 0 et d > 0 requis"
    );
    friction_stress + strengthening_coefficient * grain_diameter.powf(-0.5)
}

/// Diamètre de grain requis pour une limite visée
/// `d = (k/(target_yield - sigma0))^2` (réciproque de [`hall_petch_yield_strength`]).
///
/// Panique si `strengthening_coefficient <= 0`, `friction_stress < 0`
/// ou `target_yield <= friction_stress`.
pub fn hall_petch_grain_size_for_yield(
    friction_stress: f64,
    strengthening_coefficient: f64,
    target_yield: f64,
) -> f64 {
    assert!(
        friction_stress >= 0.0 && strengthening_coefficient > 0.0 && target_yield > friction_stress,
        "sigma0 ≥ 0, k > 0 et sigma_y > sigma0 requis"
    );
    (strengthening_coefficient / (target_yield - friction_stress)).powi(2)
}

/// Coefficient de Hall-Petch identifié à partir de deux états mesurés
/// `k = (sigma2 - sigma1)/(d2^(-1/2) - d1^(-1/2))`.
///
/// Panique si un diamètre `<= 0` ou si `grain1 == grain2`
/// (dénominateur nul, `k` indéterminé).
pub fn hall_petch_strengthening_coefficient(
    yield1: f64,
    yield2: f64,
    grain1: f64,
    grain2: f64,
) -> f64 {
    assert!(grain1 > 0.0 && grain2 > 0.0, "d1 > 0 et d2 > 0 requis");
    assert!(
        grain1 != grain2,
        "d1 ≠ d2 requis (dénominateur nul, k indéterminé)"
    );
    (yield2 - yield1) / (grain2.powf(-0.5) - grain1.powf(-0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn grain_size_is_reciprocal_of_yield() {
        // d obtenu pour une limite visée redonne exactement cette limite.
        let (sigma0, k, target) = (70.0e6, 0.31e6, 1.2e8);
        let d = hall_petch_grain_size_for_yield(sigma0, k, target);
        assert_relative_eq!(
            hall_petch_yield_strength(sigma0, k, d),
            target,
            max_relative = 1e-12
        );
    }

    #[test]
    fn coefficient_recovers_the_generating_k() {
        // Deux points générés avec (sigma0, k) connus : l'identification retrouve k.
        let (sigma0, k) = (50.0e6, 0.5e6);
        let (d1, d2) = (100.0e-6, 25.0e-6);
        let y1 = hall_petch_yield_strength(sigma0, k, d1);
        let y2 = hall_petch_yield_strength(sigma0, k, d2);
        assert_relative_eq!(
            hall_petch_strengthening_coefficient(y1, y2, d1, d2),
            k,
            max_relative = 1e-12
        );
    }

    #[test]
    fn strengthening_scales_as_inverse_sqrt() {
        // Terme de renfort k·d^(-1/2) : diviser d par 4 double le renfort.
        let (sigma0, k) = (40.0e6, 0.3e6);
        let coarse = hall_petch_yield_strength(sigma0, k, 8.0e-5) - sigma0;
        let fine = hall_petch_yield_strength(sigma0, k, 2.0e-5) - sigma0;
        assert_relative_eq!(fine / coarse, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_iron_yield_strength() {
        // Fer : sigma0 = 70 MPa, k = 0,31 MPa·m^(1/2), d = 50 µm.
        // d^(-1/2) = 1/sqrt(5e-5) = 141,4213562 m^(-1/2)
        // sigma_y = 70e6 + 0,31e6·141,4213562 = 70e6 + 43,84062044e6
        //         = 113,84062044 MPa.
        assert_relative_eq!(
            hall_petch_yield_strength(70.0e6, 0.31e6, 50.0e-6),
            1.1384062044e8,
            max_relative = 1e-9
        );
    }

    #[test]
    fn finer_grain_raises_yield_strength() {
        // Grain plus fin ⇒ limite d'élasticité plus élevée (monotonie).
        let (sigma0, k) = (60.0e6, 0.25e6);
        let coarse = hall_petch_yield_strength(sigma0, k, 1.0e-4);
        let fine = hall_petch_yield_strength(sigma0, k, 1.0e-5);
        assert!(fine > coarse);
    }

    #[test]
    #[should_panic(expected = "sigma_y > sigma0")]
    fn target_below_friction_panics() {
        hall_petch_grain_size_for_yield(70.0e6, 0.31e6, 50.0e6);
    }
}
