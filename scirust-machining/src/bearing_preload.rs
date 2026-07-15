//! **Précharge de roulement** — flèche, rigidité effective et effort axial
//! d'une paire de roulements montée en précharge.
//!
//! ```text
//! flèche          δ = F / k                       (déplacement sous précharge)
//! rigidité        k_eff = k_nom · factor          (la précharge rigidifie)
//! effort axial    F = k · offset                  (précharge par calage/offset)
//! ```
//!
//! `F` effort de précharge (N), `k` raideur du roulement (N/m), `δ` flèche
//! (m), `k_nom` raideur nominale (N/m), `factor` coefficient de rigidification
//! adimensionnel (`≥ 1`), `offset` calage/décalage axial imposé (m).
//!
//! **Convention** : SI. **Limite honnête** : la raideur de roulement `k` (et le
//! coefficient `factor`) sont des **données fournies par l'appelant** ; aucune
//! valeur « par défaut » n'est inventée. La raideur d'un roulement est en réalité
//! **non linéaire** (loi de Hertz, `k ∝ F^{1/3}` en charge) ; on la **linéarise
//! ici autour du point de précharge**. Voir [`crate::bearings`] et
//! [`crate::hertz`].

/// Flèche sous précharge `δ = F / k` (m).
///
/// Panique si `bearing_stiffness <= 0`.
pub fn preload_deflection(preload_force: f64, bearing_stiffness: f64) -> f64 {
    assert!(
        bearing_stiffness > 0.0,
        "la raideur du roulement doit être strictement positive"
    );
    preload_force / bearing_stiffness
}

/// Raideur effective rigidifiée par la précharge `k_eff = k_nom · factor` (N/m).
///
/// Panique si `nominal_stiffness <= 0` ou `preload_factor < 1` (la précharge ne
/// peut qu'augmenter la rigidité).
pub fn preloaded_stiffness(nominal_stiffness: f64, preload_factor: f64) -> f64 {
    assert!(
        nominal_stiffness > 0.0,
        "la raideur nominale doit être strictement positive"
    );
    assert!(
        preload_factor >= 1.0,
        "le coefficient de précharge doit être ≥ 1 (la précharge rigidifie)"
    );
    nominal_stiffness * preload_factor
}

/// Effort axial de précharge issu d'un calage imposé `F = k · offset` (N).
///
/// Panique si `stiffness <= 0` ou `offset < 0`.
pub fn axial_preload_from_offset(offset: f64, stiffness: f64) -> f64 {
    assert!(stiffness > 0.0, "la raideur doit être strictement positive");
    assert!(offset >= 0.0, "le calage axial doit être positif ou nul");
    stiffness * offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn deflection_and_offset_are_reciprocal() {
        // δ = F/k puis F = k·δ doit rendre l'effort de départ.
        let k = 2.5e8;
        let f = 1200.0;
        let delta = preload_deflection(f, k);
        assert_relative_eq!(axial_preload_from_offset(delta, k), f, epsilon = 1e-9);
    }

    #[test]
    fn deflection_is_inversely_proportional_to_stiffness() {
        // Doubler la raideur divise la flèche par deux (même effort).
        let d1 = preload_deflection(1000.0, 1.0e8);
        let d2 = preload_deflection(1000.0, 2.0e8);
        assert_relative_eq!(d1 / d2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn unit_factor_leaves_stiffness_unchanged() {
        // factor = 1 → pas de rigidification.
        assert_relative_eq!(preloaded_stiffness(3.0e8, 1.0), 3.0e8, epsilon = 1e-3);
    }

    #[test]
    fn preload_factor_scales_stiffness_linearly() {
        // factor = 1,4 → +40 % de raideur.
        assert_relative_eq!(preloaded_stiffness(2.0e8, 1.4), 2.8e8, epsilon = 1e-3);
    }

    #[test]
    fn realistic_offset_gives_expected_force() {
        // k = 3e8 N/m, calage 5 µm → F = 1500 N.
        assert_relative_eq!(
            axial_preload_from_offset(5.0e-6, 3.0e8),
            1500.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "le coefficient de précharge doit être ≥ 1")]
    fn factor_below_one_panics() {
        preloaded_stiffness(1.0e8, 0.9);
    }
}
