//! Loi de comportement **élasto-plastique** de **Ramberg-Osgood** (courbe
//! monotone contrainte-déformation, forme à déformation totale).
//!
//! ```text
//! déformation élastique  ε_e = σ / E
//! déformation plastique  ε_p = (σ / K)^(1/n)
//! déformation totale     ε   = ε_e + ε_p = σ/E + (σ/K)^(1/n)
//! module sécant          E_s = σ / ε
//! ```
//!
//! `σ` contrainte (Pa), `E` module d'Young (Pa), `K` coefficient de résistance
//! (Pa), `n` exposant d'écrouissage (sans dimension, `0 < n <= 1`), `ε`/`ε_e`/`ε_p`
//! déformations totale/élastique/plastique (sans dimension), `E_s` module sécant (Pa).
//!
//! **Convention** : SI cohérent, traction positive (`σ >= 0`).
//! **Limite honnête** : courbe **monotone** de Ramberg-Osgood ; le module `E`,
//! le coefficient de résistance `K` et l'exposant d'écrouissage `n` sont des
//! **données du matériau fournies par l'appelant** (aucune valeur « par défaut »
//! inventée). Distinct de [`crate::true_stress_strain`] (Hollomon, plastique pur).

/// Déformation totale de Ramberg-Osgood `ε = σ/E + (σ/K)^(1/n)` (sans dimension).
///
/// Panique si `σ < 0`, `E <= 0`, `K <= 0`, ou `n` hors de `]0, 1]`.
pub fn rambergosgood_total_strain(
    stress: f64,
    youngs_modulus: f64,
    strength_coefficient: f64,
    hardening_exponent: f64,
) -> f64 {
    rambergosgood_elastic_strain(stress, youngs_modulus)
        + rambergosgood_plastic_strain(stress, strength_coefficient, hardening_exponent)
}

/// Déformation élastique `ε_e = σ/E` (sans dimension).
///
/// Panique si `σ < 0` ou `E <= 0`.
pub fn rambergosgood_elastic_strain(stress: f64, youngs_modulus: f64) -> f64 {
    assert!(stress >= 0.0, "la contrainte doit être positive (traction)");
    assert!(
        youngs_modulus > 0.0,
        "le module d'Young doit être strictement positif"
    );
    stress / youngs_modulus
}

/// Déformation plastique `ε_p = (σ/K)^(1/n)` (sans dimension).
///
/// Panique si `σ < 0`, `K <= 0`, ou `n` hors de `]0, 1]`.
pub fn rambergosgood_plastic_strain(
    stress: f64,
    strength_coefficient: f64,
    hardening_exponent: f64,
) -> f64 {
    assert!(stress >= 0.0, "la contrainte doit être positive (traction)");
    assert!(
        strength_coefficient > 0.0,
        "le coefficient de résistance doit être strictement positif"
    );
    assert!(
        hardening_exponent > 0.0 && hardening_exponent <= 1.0,
        "l'exposant d'écrouissage doit vérifier 0 < n <= 1"
    );
    (stress / strength_coefficient).powf(1.0 / hardening_exponent)
}

/// Module sécant `E_s = σ/ε` (Pa).
///
/// Panique si `σ < 0` ou `total_strain <= 0`.
pub fn rambergosgood_secant_modulus(stress: f64, total_strain: f64) -> f64 {
    assert!(stress >= 0.0, "la contrainte doit être positive (traction)");
    assert!(
        total_strain > 0.0,
        "la déformation totale doit être strictement positive"
    );
    stress / total_strain
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn total_is_sum_of_elastic_and_plastic() {
        // Additivité : ε = ε_e + ε_p (identité de définition).
        let (stress, e, k, n) = (400.0e6_f64, 200.0e9_f64, 600.0e6_f64, 0.15_f64);
        let ee = rambergosgood_elastic_strain(stress, e);
        let ep = rambergosgood_plastic_strain(stress, k, n);
        assert_relative_eq!(
            rambergosgood_total_strain(stress, e, k, n),
            ee + ep,
            epsilon = 1e-15
        );
    }

    #[test]
    fn elastic_strain_is_hookean() {
        // ε_e = σ/E ; contrainte nulle => déformation nulle.
        let (stress, e) = (300.0e6_f64, 210.0e9_f64);
        assert_relative_eq!(
            rambergosgood_elastic_strain(stress, e),
            stress / e,
            epsilon = 1e-15
        );
        assert_relative_eq!(rambergosgood_elastic_strain(0.0, e), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn plastic_strain_reaches_unit_at_coefficient() {
        // σ = K  =>  ε_p = (1)^(1/n) = 1, quel que soit n.
        let (k, n) = (600.0e6_f64, 0.15_f64);
        assert_relative_eq!(rambergosgood_plastic_strain(k, k, n), 1.0, epsilon = 1e-12);
        // écrouissage : ε_p croît avec la contrainte.
        assert!(
            rambergosgood_plastic_strain(0.9 * k, k, n) < rambergosgood_plastic_strain(k, k, n)
        );
    }

    #[test]
    fn secant_modulus_recovers_stress() {
        // Réciprocité : E_s·ε = σ.
        let (stress, e, k, n) = (400.0e6_f64, 200.0e9_f64, 600.0e6_f64, 0.15_f64);
        let eps = rambergosgood_total_strain(stress, e, k, n);
        let es = rambergosgood_secant_modulus(stress, eps);
        assert_relative_eq!(es * eps, stress, epsilon = 1.0);
        // Le module sécant est inférieur au module d'Young (présence de plasticité).
        assert!(es < e);
    }

    #[test]
    fn worked_example() {
        // σ=400 MPa, E=200 GPa, K=600 MPa, n=0,15.
        // ε_e = 400e6/200e9 = 0,002.
        // ε_p = (400/600)^(1/0,15) = (2/3)^6,6667 ≈ 0,066991.
        // ε   ≈ 0,002 + 0,066991 = 0,068991.
        let (stress, e, k, n) = (400.0e6_f64, 200.0e9_f64, 600.0e6_f64, 0.15_f64);
        assert_relative_eq!(
            rambergosgood_elastic_strain(stress, e),
            0.002,
            epsilon = 1e-9
        );
        let ep = (2.0_f64 / 3.0).powf(1.0 / 0.15);
        assert_relative_eq!(
            rambergosgood_plastic_strain(stress, k, n),
            ep,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            rambergosgood_total_strain(stress, e, k, n),
            0.002 + ep,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "0 < n <= 1")]
    fn hardening_exponent_out_of_range_panics() {
        rambergosgood_plastic_strain(400.0e6, 600.0e6, 1.5);
    }
}
