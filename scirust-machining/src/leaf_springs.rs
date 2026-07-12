//! Ressorts à **lames** — lame(s) en console (forme quart-elliptique) : contrainte
//! de flexion, flèche et raideur d'un empilage de `n` lames.
//!
//! ```text
//! contrainte    σ = 6·F·L/(n·b·t²)
//! flèche        δ = 6·F·L³/(n·b·t³·E)
//! raideur       k = F/δ = n·b·t³·E/(6·L³)
//! ```
//!
//! `F` effort en bout (N), `L` longueur de la lame (m), `n` nombre de lames, `b`
//! largeur (m), `t` épaisseur d'une lame (m), `E` module de Young (Pa). Le modèle
//! est celui d'une **poutre-console** de section `n·b×t`, hypothèse d'égale
//! contrainte le long des lames.
//!
//! **Convention** : SI cohérent. **Limite honnête** : ressort à lames idéalisé
//! en console d'égale contrainte (lames de même épaisseur, effort en bout) ; ne
//! traite ni le frottement inter-lames (amortissement), ni la précontrainte de
//! cambrure, ni la lame maîtresse à œillets.

/// Contrainte de flexion maximale `σ = 6·F·L/(n·b·t²)` (Pa).
///
/// Panique si `n·b·t² <= 0`.
pub fn bending_stress(force: f64, length: f64, leaves: u32, width: f64, thickness: f64) -> f64 {
    let denom = leaves as f64 * width * thickness * thickness;
    assert!(denom > 0.0, "n·b·t² doit être strictement positif");
    6.0 * force * length / denom
}

/// Flèche en bout `δ = 6·F·L³/(n·b·t³·E)` (m).
///
/// Panique si `n·b·t³·E <= 0`.
pub fn deflection(
    force: f64,
    length: f64,
    leaves: u32,
    width: f64,
    thickness: f64,
    youngs_modulus: f64,
) -> f64 {
    let denom = leaves as f64 * width * thickness.powi(3) * youngs_modulus;
    assert!(denom > 0.0, "n·b·t³·E doit être strictement positif");
    6.0 * force * length.powi(3) / denom
}

/// Raideur `k = n·b·t³·E/(6·L³)` (N/m).
///
/// Panique si `length <= 0`.
pub fn rate(length: f64, leaves: u32, width: f64, thickness: f64, youngs_modulus: f64) -> f64 {
    assert!(length > 0.0, "la longueur doit être strictement positive");
    leaves as f64 * width * thickness.powi(3) * youngs_modulus / (6.0 * length.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_is_force_over_deflection() {
        // k doit égaler F/δ pour toute charge.
        let (f, l, n, b, t, e) = (1000.0, 0.5, 3u32, 0.06, 0.008, 210e9);
        let d = deflection(f, l, n, b, t, e);
        let k = rate(l, n, b, t, e);
        assert_relative_eq!(k, f / d, max_relative = 1e-9);
    }

    #[test]
    fn more_leaves_reduce_stress_and_deflection() {
        // Ajouter des lames (n↑) diminue contrainte et flèche à charge égale.
        let s3 = bending_stress(1000.0, 0.5, 3, 0.06, 0.008);
        let s6 = bending_stress(1000.0, 0.5, 6, 0.06, 0.008);
        assert_relative_eq!(s3 / s6, 2.0, epsilon = 1e-9);
        let d3 = deflection(1000.0, 0.5, 3, 0.06, 0.008, 210e9);
        let d6 = deflection(1000.0, 0.5, 6, 0.06, 0.008, 210e9);
        assert_relative_eq!(d3 / d6, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn thickness_dominates_deflection() {
        // δ ∝ 1/t³ : doubler l'épaisseur divise la flèche par 8.
        let d1 = deflection(1000.0, 0.5, 3, 0.06, 0.008, 210e9);
        let d2 = deflection(1000.0, 0.5, 3, 0.06, 0.016, 210e9);
        assert_relative_eq!(d1 / d2, 8.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "n·b·t²")]
    fn zero_leaves_panics() {
        bending_stress(1000.0, 0.5, 0, 0.06, 0.008);
    }
}
