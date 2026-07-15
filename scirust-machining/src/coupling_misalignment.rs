//! Désalignement d'accouplement — géométrie du défaut et efforts radiaux induits
//! sur l'arbre et les paliers.
//!
//! ```text
//! désalignement angulaire  θ = atan(offset / distance)   (rad, converti en °)
//! réaction (offset parallèle) F = k · offset               (effort radial)
//! désalignement combiné    δ_eq = hypot(δ_ang, δ_par) = √(δ_ang² + δ_par²)
//! ```
//!
//! `offset` décalage radial des deux demi-arbres (m), `distance` entre-axe des
//! plans de mesure (m), `θ` angle de désalignement (°), `k` raideur radiale de
//! l'accouplement (N/m), `F` effort radial de réaction (N), `δ_ang`/`δ_par`
//! contributions angulaire et parallèle exprimées dans la **même** unité (N ou m),
//! `δ_eq` résultante quadratique. Un accouplement désaligné impose sur chaque
//! demi-arbre un effort radial cyclique qui charge les paliers voisins.
//!
//! **Convention** : SI cohérent (m, N, N/m ; angles rendus en degrés). **Limite
//! honnête** : la raideur radiale `k` de l'accouplement et les valeurs de
//! matériau/procédé sont **fournies par l'appelant** (jamais de « valeur par
//! défaut » inventée) ; modèle quasi statique valable pour de **petits
//! désalignements** — la dynamique (balourd induit, jeu, amortissement, effets
//! de vitesse) est **ignorée**.

use core::f64::consts::PI;

/// Désalignement angulaire `θ = atan(offset / distance)` rendu en degrés.
///
/// `offset` (m) est le décalage radial mesuré entre deux plans séparés de
/// `distance` (m). Le résultat est l'angle entre les axes des deux demi-arbres.
///
/// Panique si `distance <= 0` ou `offset < 0`.
pub fn misalign_angular_deg(offset: f64, distance: f64) -> f64 {
    assert!(
        distance > 0.0,
        "l'entre-axe des plans de mesure doit être strictement positif"
    );
    assert!(offset >= 0.0, "le décalage radial doit être positif ou nul");
    (offset / distance).atan() * 180.0 / PI
}

/// Effort radial de réaction dû à un offset parallèle `F = k · offset` (N).
///
/// `k` (N/m) est la raideur radiale de l'accouplement fournie par l'appelant,
/// `offset` (m) le décalage parallèle des deux axes.
///
/// Panique si `coupling_stiffness < 0` ou `offset < 0`.
pub fn coupling_parallel_offset_reaction(offset: f64, coupling_stiffness: f64) -> f64 {
    assert!(
        coupling_stiffness >= 0.0,
        "la raideur d'accouplement doit être positive ou nulle"
    );
    assert!(
        offset >= 0.0,
        "le décalage parallèle doit être positif ou nul"
    );
    coupling_stiffness * offset
}

/// Désalignement combiné `δ_eq = √(δ_ang² + δ_par²)` (résultante quadratique).
///
/// Les deux contributions doivent être exprimées dans la **même** unité (par
/// exemple deux efforts en N, ou deux décalages en m) ; le résultat conserve
/// cette unité. Combinaison quadratique de deux composantes orthogonales.
///
/// Panique si `angular < 0` ou `parallel < 0`.
pub fn misalign_combined(angular: f64, parallel: f64) -> f64 {
    assert!(
        angular >= 0.0,
        "la contribution angulaire doit être positive ou nulle"
    );
    assert!(
        parallel >= 0.0,
        "la contribution parallèle doit être positive ou nulle"
    );
    angular.hypot(parallel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn angular_forty_five_degrees() {
        // offset = distance → atan(1) = 45°.
        assert_relative_eq!(misalign_angular_deg(0.010, 0.010), 45.0, epsilon = 1e-9);
    }

    #[test]
    fn angular_zero_when_aligned() {
        // Pas de décalage → aucun désalignement angulaire.
        assert_relative_eq!(misalign_angular_deg(0.0, 0.100), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn parallel_reaction_is_linear_in_offset() {
        // F = k·offset : doubler l'offset double l'effort (proportionnalité).
        let k = 2.5e6_f64;
        let f1 = coupling_parallel_offset_reaction(0.0002, k);
        let f2 = coupling_parallel_offset_reaction(0.0004, k);
        assert_relative_eq!(f1, 500.0, epsilon = 1e-9);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn combined_is_pythagorean() {
        // 3-4-5 : √(3² + 4²) = 5 (triplet exact).
        assert_relative_eq!(misalign_combined(3.0, 4.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn combined_reduces_to_single_component() {
        // Une composante nulle → la résultante vaut l'autre composante.
        assert_relative_eq!(misalign_combined(750.0, 0.0), 750.0, epsilon = 1e-12);
        assert_relative_eq!(misalign_combined(0.0, 750.0), 750.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case() {
        // offset 0,2 mm sur 200 mm → θ = atan(0,001) ≈ 0,0573°.
        let theta = misalign_angular_deg(0.0002, 0.200);
        assert_relative_eq!(theta, 0.001_f64.atan() * 180.0 / PI, epsilon = 1e-12);
        assert_relative_eq!(theta, 0.057_295_760, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "entre-axe des plans de mesure")]
    fn zero_distance_panics() {
        misalign_angular_deg(0.001, 0.0);
    }
}
