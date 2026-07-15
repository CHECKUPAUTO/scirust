//! Voilement (flambage) d'une plaque mince rectangulaire comprimée dans son plan
//! — contrainte critique, charge critique, élancement de plaque et coefficient
//! de voilement d'une plaque simplement appuyée.
//!
//! ```text
//! contrainte critique  σcr = k · π²·E / (12·(1 − ν²)) · (t/b)²
//! charge critique      Ncr = σcr · t · b
//! élancement de plaque λp  = √(σe / σcr)
//! coefficient (appuyée) k = (m·b/a + a/(m·b))²   (m demi-ondes suivant a)
//! ```
//!
//! `k` coefficient de voilement (sans dimension, dépend des conditions de bord et
//! du rapport d'aspect `a/b`), `E` module de Young (Pa), `ν` coefficient de
//! Poisson (sans dimension), `t` épaisseur de la plaque (m), `b` largeur chargée
//! (m), `a` longueur (m), `σcr` contrainte critique de voilement (Pa), `Ncr`
//! charge critique totale sur le bord chargé (N), `σe` limite élastique (Pa),
//! `m` nombre de demi-ondes longitudinales (entier ≥ 1).
//!
//! **Convention** : SI cohérent. **Limite honnête** : plaque **mince**
//! rectangulaire, **élastique** linéaire, comprimée uniformément sur un bord ; le
//! coefficient `k` est **fourni par l'appelant** selon les conditions de bord
//! réelles et le rapport d'aspect (seul le cas académique « simplement appuyée
//! sur les quatre bords » est offert par [`plate_buckling_coefficient_simply_supported`]).
//! Aucune valeur de matériau, de `k` ou de procédé n'est inventée par défaut.
//! Complète [`crate::buckling`] (colonnes d'Euler) : ici la ruine par instabilité
//! concerne une **surface** et non une **poutre**.

use core::f64::consts::PI;

/// Contrainte critique de voilement `σcr = k·π²·E / (12·(1 − ν²)) · (t/b)²` (Pa).
///
/// Panique si `poisson_ratio` sort de `]-1 ; 0,5]`, ou si l'un de
/// `buckling_coefficient`, `youngs_modulus`, `thickness`, `width` n'est pas
/// strictement positif.
pub fn plate_critical_stress(
    buckling_coefficient: f64,
    youngs_modulus: f64,
    poisson_ratio: f64,
    thickness: f64,
    width: f64,
) -> f64 {
    assert!(
        buckling_coefficient > 0.0,
        "le coefficient de voilement k doit être strictement positif"
    );
    assert!(
        youngs_modulus > 0.0,
        "le module de Young doit être strictement positif"
    );
    assert!(
        thickness > 0.0 && width > 0.0,
        "l'épaisseur et la largeur doivent être strictement positives"
    );
    assert!(
        poisson_ratio > -1.0 && poisson_ratio <= 0.5,
        "le coefficient de Poisson doit être dans ]-1 ; 0,5]"
    );
    let ratio = thickness / width;
    buckling_coefficient * PI * PI * youngs_modulus / (12.0 * (1.0 - poisson_ratio * poisson_ratio))
        * ratio
        * ratio
}

/// Charge critique de voilement totale `Ncr = σcr·t·b` (N), obtenue en
/// intégrant la contrainte critique sur la section chargée `t·b`.
///
/// Panique si `critical_stress`, `thickness` ou `width` n'est pas strictement
/// positif.
pub fn plate_critical_load(critical_stress: f64, thickness: f64, width: f64) -> f64 {
    assert!(
        critical_stress > 0.0,
        "la contrainte critique doit être strictement positive"
    );
    assert!(
        thickness > 0.0 && width > 0.0,
        "l'épaisseur et la largeur doivent être strictement positives"
    );
    critical_stress * thickness * width
}

/// Élancement réduit de plaque `λp = √(σe / σcr)` (sans dimension) : au-delà de
/// `λp = 1` la plaque voile avant d'atteindre la limite élastique.
///
/// Panique si `yield_stress` ou `critical_stress` n'est pas strictement positif.
pub fn plate_buckling_slenderness(yield_stress: f64, critical_stress: f64) -> f64 {
    assert!(
        yield_stress > 0.0,
        "la limite élastique doit être strictement positive"
    );
    assert!(
        critical_stress > 0.0,
        "la contrainte critique doit être strictement positive"
    );
    (yield_stress / critical_stress).sqrt()
}

/// Coefficient de voilement `k = (m·b/a + a/(m·b))²` d'une plaque **simplement
/// appuyée sur ses quatre bords** en compression uniaxiale, pour `m` demi-ondes
/// longitudinales, `aspect_ratio = a/b` étant le rapport longueur/largeur.
///
/// Panique si `aspect_ratio <= 0` ou si `half_waves == 0`.
pub fn plate_buckling_coefficient_simply_supported(aspect_ratio: f64, half_waves: u32) -> f64 {
    assert!(
        aspect_ratio > 0.0,
        "le rapport d'aspect a/b doit être strictement positif"
    );
    assert!(
        half_waves >= 1,
        "le nombre de demi-ondes doit être au moins 1"
    );
    let m = f64::from(half_waves);
    let term = m / aspect_ratio + aspect_ratio / m;
    term * term
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn load_recovers_stress_over_loaded_section() {
        // Réciprocité : Ncr / (t·b) doit redonner σcr.
        let (sigma, t, b) = (3.0e8, 0.006, 0.3);
        let ncr = plate_critical_load(sigma, t, b);
        assert_relative_eq!(ncr / (t * b), sigma, epsilon = 1e-6);
    }

    #[test]
    fn stress_scales_as_thickness_over_width_squared() {
        // σcr ∝ (t/b)² : doubler t/b quadruple la contrainte critique.
        let base = plate_critical_stress(4.0, 210e9, 0.3, 0.006, 0.3);
        let doubled = plate_critical_stress(4.0, 210e9, 0.3, 0.012, 0.3);
        assert_relative_eq!(doubled / base, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn simply_supported_coefficient_minimum_is_four() {
        // Plaque carrée (a/b = 1) avec une demi-onde : k = (1 + 1)² = 4.
        assert_relative_eq!(
            plate_buckling_coefficient_simply_supported(1.0, 1),
            4.0,
            epsilon = 1e-12
        );
        // a/b = 2 avec 2 demi-ondes retombe aussi sur k = 4 (ondes carrées).
        assert_relative_eq!(
            plate_buckling_coefficient_simply_supported(2.0, 2),
            4.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn slenderness_is_unity_at_the_yield_threshold() {
        // Si σcr = σe alors λp = 1 (voilement et plastification simultanés).
        assert_relative_eq!(
            plate_buckling_slenderness(355e6, 355e6),
            1.0,
            epsilon = 1e-12
        );
        // Réciprocité : σcr déduit de λp = √(σe/σcr) redonne l'entrée.
        let sigma_cr = 3.0e8;
        let lambda = plate_buckling_slenderness(355e6, sigma_cr);
        assert_relative_eq!(355e6 / (lambda * lambda), sigma_cr, epsilon = 1.0);
    }

    #[test]
    fn worked_case_steel_plate() {
        // Cas chiffré : plaque acier appuyée, k=4, E=210 GPa, ν=0,3,
        // t=6 mm, b=300 mm.
        // σcr = 4·π²·210e9 / (12·(1−0,09)) · (0,006/0,3)²
        //     = 4·9,8696044·210e9 / 10,92 · 4e-4 ≈ 3,0368e8 Pa (≈ 303,7 MPa).
        let sigma = plate_critical_stress(4.0, 210e9, 0.3, 0.006, 0.3);
        assert_relative_eq!(sigma, 3.036801e8, epsilon = 1e2);
        // Vérification indépendante contre l'expression explicite.
        let expected =
            4.0_f64 * PI * PI * 210e9 / (12.0 * (1.0 - 0.3_f64 * 0.3)) * (0.006_f64 / 0.3).powi(2);
        assert_relative_eq!(sigma, expected, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient de Poisson doit être dans ]-1 ; 0,5]")]
    fn rejects_impossible_poisson_ratio() {
        let _ = plate_critical_stress(4.0, 210e9, 0.75, 0.006, 0.3);
    }
}
