//! Contrainte et déformation **vraies** (rationnelles) — conversion depuis les
//! grandeurs nominales (ingénieur) et loi d'écrouissage de **Hollomon**.
//!
//! ```text
//! contrainte vraie   σ_v = σ_n·(1 + ε_n)
//! déformation vraie  ε_v = ln(1 + ε_n)
//! Hollomon           σ_v = K·ε_v^n        (K coefficient, n exposant d'écrouissage)
//! ```
//!
//! `σ_n`/`ε_n` contrainte/déformation **nominales** (rapportées à la section et
//! à la longueur initiales), `σ_v`/`ε_v` grandeurs **vraies** (section et longueur
//! instantanées), `K` coefficient de résistance (Pa), `n` exposant d'écrouissage.
//! Les relations supposent la **conservation du volume** (déformation plastique).
//!
//! **Convention** : SI cohérent, traction positive. **Limite honnête** : valables
//! **avant striction** (déformation homogène, volume constant) ; au-delà de la
//! striction, la contrainte vraie exige une correction (Bridgman). `K` et `n`
//! sont des données du matériau fournies par l'appelant.

/// Contrainte vraie `σ_v = σ_n·(1 + ε_n)` (Pa).
pub fn true_stress(engineering_stress: f64, engineering_strain: f64) -> f64 {
    engineering_stress * (1.0 + engineering_strain)
}

/// Déformation vraie `ε_v = ln(1 + ε_n)`.
///
/// Panique si `1 + ε_n <= 0`.
pub fn true_strain(engineering_strain: f64) -> f64 {
    assert!(
        1.0 + engineering_strain > 0.0,
        "1 + ε_n doit être strictement positif"
    );
    (1.0 + engineering_strain).ln()
}

/// Contrainte d'écrouissage de Hollomon `σ_v = K·ε_v^n` (Pa).
///
/// Panique si `true_strain < 0`.
pub fn hollomon_stress(
    strength_coefficient: f64,
    true_strain: f64,
    hardening_exponent: f64,
) -> f64 {
    assert!(
        true_strain >= 0.0,
        "la déformation vraie doit être positive"
    );
    strength_coefficient * true_strain.powf(hardening_exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn true_exceeds_engineering_in_tension() {
        // ε_n=0,2 : σ_v = 1,2·σ_n > σ_n ; ε_v = ln1,2 < ε_n.
        assert_relative_eq!(true_stress(300e6, 0.2), 360e6, epsilon = 1e-3);
        let ev = true_strain(0.2);
        assert_relative_eq!(ev, 1.2f64.ln(), epsilon = 1e-12);
        assert!(ev < 0.2);
    }

    #[test]
    fn small_strains_converge() {
        // Pour ε_n petit, ε_v ≈ ε_n (ln(1+x) ≈ x).
        let ev = true_strain(0.001);
        assert_relative_eq!(ev, 0.001, max_relative = 1e-3);
    }

    #[test]
    fn hollomon_hardening() {
        // K=800 MPa, n=0,2 : σ_v(0,1) = 800·0,1^0,2.
        assert_relative_eq!(
            hollomon_stress(800e6, 0.1, 0.2),
            800e6 * 0.1f64.powf(0.2),
            epsilon = 1e-3
        );
        // écrouissage : σ croît avec la déformation.
        assert!(hollomon_stress(800e6, 0.2, 0.2) > hollomon_stress(800e6, 0.1, 0.2));
    }

    #[test]
    #[should_panic(expected = "1 + ε_n")]
    fn compression_beyond_full_shortening_panics() {
        true_strain(-1.5);
    }
}
