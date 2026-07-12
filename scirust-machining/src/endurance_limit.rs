//! Limite d'endurance et courbe **S-N** — correction de la limite d'endurance
//! par les facteurs de **Marin** et construction de la droite de fatigue à
//! nombre fini de cycles (approche de Shigley/Basquin).
//!
//! ```text
//! limite corrigée   Se = ka·kb·kc·kd·ke · Se'
//! estimation acier  Se' ≈ 0,5·Su  (plafonnée)      (flexion rotative)
//! droite S-N        Sf(N) = a·N^b
//!   a = (f·Su)²/Se        b = −(1/3)·log10(f·Su/Se)   (entre 10³ et 10⁶ cycles)
//! ```
//!
//! `Se'` limite d'endurance de l'éprouvette lisse, `ka…ke` facteurs de Marin
//! (état de surface, taille, charge, température, fiabilité), `Su` résistance à
//! la rupture, `f` fraction de résistance à 10³ cycles (~0,9). `Sf(N)` est la
//! résistance en fatigue à `N` cycles.
//!
//! **Convention** : contraintes cohérentes (MPa conseillé), `Su` et le plafond
//! dans la même unité. **Limite honnête** : **estimations** d'ingénieur
//! (Shigley) ; `Se'`, les facteurs de Marin et `f` proviennent d'abaques/essais
//! fournis par l'appelant. Le cumul de dommage relève de `scirust-fatigue`.

/// Limite d'endurance corrigée `Se = ka·kb·kc·kd·ke·Se'`.
pub fn corrected_endurance_limit(se_prime: f64, factors: &[f64]) -> f64 {
    factors.iter().fold(se_prime, |acc, &k| acc * k)
}

/// Estimation de la limite d'endurance en flexion rotative d'un **acier**
/// `Se' ≈ 0,5·Su`, plafonnée à `cap` (p. ex. 700 MPa).
///
/// Panique si `su < 0` ou `cap <= 0`.
pub fn steel_endurance_estimate(su: f64, cap: f64) -> f64 {
    assert!(su >= 0.0 && cap > 0.0, "Su ≥ 0 et plafond > 0 requis");
    (0.5 * su).min(cap)
}

/// Coefficients `(a, b)` de la droite S-N `Sf = a·N^b` entre 10³ et 10⁶ cycles,
/// `f` fraction de résistance à 10³ cycles.
///
/// Panique si `se <= 0` ou `f*su <= 0`.
pub fn sn_coefficients(su: f64, se: f64, f: f64) -> (f64, f64) {
    assert!(se > 0.0 && f * su > 0.0, "Se > 0 et f·Su > 0 requis");
    let fsu = f * su;
    let a = fsu * fsu / se;
    let b = -(1.0 / 3.0) * (fsu / se).log10();
    (a, b)
}

/// Résistance en fatigue à `N` cycles `Sf(N) = a·N^b`.
///
/// Panique si `cycles <= 0`.
pub fn fatigue_strength_at_cycles(a: f64, b: f64, cycles: f64) -> f64 {
    assert!(
        cycles > 0.0,
        "le nombre de cycles doit être strictement positif"
    );
    a * cycles.powf(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn marin_factors_multiply() {
        // Se' = 350, ka=0,8, kb=0,9, kc=1, → Se = 350·0,8·0,9 = 252.
        assert_relative_eq!(
            corrected_endurance_limit(350.0, &[0.8, 0.9, 1.0]),
            252.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn steel_estimate_is_capped() {
        // Su=800 → 0,5·800 = 400 (< plafond 700).
        assert_relative_eq!(
            steel_endurance_estimate(800.0, 700.0),
            400.0,
            epsilon = 1e-9
        );
        // Su=1600 → 800 plafonné à 700.
        assert_relative_eq!(
            steel_endurance_estimate(1600.0, 700.0),
            700.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn sn_curve_passes_through_its_anchor_points() {
        // À N=10³ : Sf doit valoir f·Su ; à N=10⁶ : Sf doit valoir Se.
        let (su, se, f) = (700.0, 300.0, 0.9);
        let (a, b) = sn_coefficients(su, se, f);
        assert_relative_eq!(
            fatigue_strength_at_cycles(a, b, 1e3),
            f * su,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            fatigue_strength_at_cycles(a, b, 1e6),
            se,
            max_relative = 1e-9
        );
    }

    #[test]
    fn strength_decreases_with_cycles() {
        let (a, b) = sn_coefficients(700.0, 300.0, 0.9);
        assert!(fatigue_strength_at_cycles(a, b, 1e4) > fatigue_strength_at_cycles(a, b, 1e5));
    }

    #[test]
    #[should_panic(expected = "Se > 0")]
    fn zero_endurance_coefficients_panic() {
        sn_coefficients(700.0, 0.0, 0.9);
    }
}
