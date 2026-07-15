//! Flambement des poteaux — formule empirique de **Rankine-Gordon** couvrant les
//! poteaux courts à moyens (transition écrasement ↔ flambement d'Euler).
//!
//! ```text
//! charge de ruine        Pc = σc·A / (1 + a·λ²)
//! contrainte de ruine    σr = σc / (1 + a·λ²)
//! constante de Rankine   a  = σc / (π²·E)     (reliant écrasement et Euler)
//! ```
//!
//! `σc` contrainte d'écrasement du matériau (Pa), `A` aire de section (m²),
//! `E` module de Young (Pa), `a` constante empirique de Rankine (sans
//! dimension), `λ = Le/k` élancement (sans dimension, `Le` longueur de
//! flambement, `k` rayon de giration). `Pc` en N, `σr` en Pa.
//!
//! **Convention** : SI cohérent. **Limite honnête** : formule **empirique** de
//! Rankine-Gordon interpolant entre l'écrasement plastique (poteaux trapus,
//! `λ → 0`, `σr → σc`) et le flambement élastique d'Euler (poteaux élancés,
//! `λ → ∞`, `σr → σc/(a·λ²) = π²E/λ²`). La contrainte d'écrasement `σc` et la
//! constante `a` sont **fournies** par le matériau/procédé (jamais inventées) ;
//! l'élancement `λ` est **fourni**. Complète [`crate::buckling`] (Euler pur).

use core::f64::consts::PI;

/// Charge de ruine de Rankine-Gordon `Pc = σc·A / (1 + a·λ²)` (N).
///
/// Interpole entre l'écrasement (`λ = 0` → `Pc = σc·A`) et Euler (`λ` grand).
///
/// Panique si `crushing_stress <= 0`, `area <= 0`, `rankine_constant < 0` ou
/// `slenderness_ratio < 0`.
pub fn rankine_crippling_load(
    crushing_stress_pa: f64,
    area_m2: f64,
    rankine_constant: f64,
    slenderness_ratio: f64,
) -> f64 {
    assert!(
        crushing_stress_pa > 0.0,
        "la contrainte d'écrasement doit être strictement positive"
    );
    assert!(
        area_m2 > 0.0,
        "l'aire de section doit être strictement positive"
    );
    assert!(
        rankine_constant >= 0.0,
        "la constante de Rankine doit être positive ou nulle"
    );
    assert!(
        slenderness_ratio >= 0.0,
        "l'élancement doit être positif ou nul"
    );
    crushing_stress_pa * area_m2 / (1.0 + rankine_constant * slenderness_ratio * slenderness_ratio)
}

/// Contrainte de ruine de Rankine-Gordon `σr = σc / (1 + a·λ²)` (Pa).
///
/// Panique si `crushing_stress <= 0`, `rankine_constant < 0` ou
/// `slenderness_ratio < 0`.
pub fn rankine_crippling_stress(
    crushing_stress_pa: f64,
    rankine_constant: f64,
    slenderness_ratio: f64,
) -> f64 {
    assert!(
        crushing_stress_pa > 0.0,
        "la contrainte d'écrasement doit être strictement positive"
    );
    assert!(
        rankine_constant >= 0.0,
        "la constante de Rankine doit être positive ou nulle"
    );
    assert!(
        slenderness_ratio >= 0.0,
        "l'élancement doit être positif ou nul"
    );
    crushing_stress_pa / (1.0 + rankine_constant * slenderness_ratio * slenderness_ratio)
}

/// Constante de Rankine `a = σc / (π²·E)` déduite de la cohérence avec Euler
/// (sans dimension).
///
/// Assure qu'aux grands élancements `σr → π²E/λ²`, la contrainte critique
/// d'Euler.
///
/// Panique si `crushing_stress <= 0` ou `youngs_modulus <= 0`.
pub fn rankine_constant_from_euler(crushing_stress_pa: f64, youngs_modulus_pa: f64) -> f64 {
    assert!(
        crushing_stress_pa > 0.0,
        "la contrainte d'écrasement doit être strictement positive"
    );
    assert!(
        youngs_modulus_pa > 0.0,
        "le module de Young doit être strictement positif"
    );
    crushing_stress_pa / (PI * PI * youngs_modulus_pa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn load_equals_stress_times_area() {
        // Identité : Pc = σr·A pour les mêmes σc, a, λ.
        let (sigma_c, area, a, lambda) = (550e6, 1.5e-3, 1.0 / 1600.0, 90.0);
        let load = rankine_crippling_load(sigma_c, area, a, lambda);
        let stress = rankine_crippling_stress(sigma_c, a, lambda);
        assert_relative_eq!(load, stress * area, epsilon = 1e-6);
    }

    #[test]
    fn short_column_recovers_crushing() {
        // λ = 0 : poteau trapu → aucune réduction, σr = σc et Pc = σc·A.
        let (sigma_c, area, a) = (550e6, 1.5e-3, 1.0 / 1600.0);
        assert_relative_eq!(
            rankine_crippling_stress(sigma_c, a, 0.0),
            sigma_c,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            rankine_crippling_load(sigma_c, area, a, 0.0),
            sigma_c * area,
            epsilon = 1e-3
        );
    }

    #[test]
    fn slender_limit_matches_euler_stress() {
        // Avec a = σc/(π²E), aux grands λ : σr = σc/(1+a·λ²) → π²E/λ² (Euler).
        let (sigma_c, e, lambda) = (550e6, 100e9, 300.0);
        let a = rankine_constant_from_euler(sigma_c, e);
        let sigma_r = rankine_crippling_stress(sigma_c, a, lambda);
        let euler = PI * PI * e / (lambda * lambda);
        // À λ = 300, le terme « 1 » du dénominateur est négligeable (~2 %).
        assert_relative_eq!(sigma_r, euler, max_relative = 0.03);
    }

    #[test]
    fn rankine_constant_definition() {
        // a = σc/(π²E) : fonte σc = 550 MPa, E = 100 GPa.
        let a = rankine_constant_from_euler(550e6, 100e9);
        assert_relative_eq!(a, 550e6 / (PI * PI * 100e9), epsilon = 1e-15);
    }

    #[test]
    fn realistic_cast_iron_column() {
        // Fonte : σc = 550 MPa, a = 1/1600, λ = 100, A = 1000 mm² = 1e-3 m².
        // 1 + a·λ² = 1 + 10000/1600 = 1 + 6,25 = 7,25.
        // σr = 550e6/7,25 ≈ 75,862 MPa ; Pc = σr·A ≈ 75 862 N.
        let (sigma_c, area, a, lambda) = (550e6, 1e-3, 1.0 / 1600.0, 100.0);
        let denom = 1.0 + a * lambda * lambda;
        assert_relative_eq!(denom, 7.25, epsilon = 1e-12);
        let stress = rankine_crippling_stress(sigma_c, a, lambda);
        assert_relative_eq!(stress, 550e6 / 7.25, epsilon = 1e-3);
        let load = rankine_crippling_load(sigma_c, area, a, lambda);
        assert_relative_eq!(load, 550e6 * area / 7.25, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "contrainte d'écrasement")]
    fn zero_crushing_stress_panics() {
        rankine_crippling_stress(0.0, 1.0 / 1600.0, 100.0);
    }
}
