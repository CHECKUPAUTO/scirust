//! Centroïdes de surfaces composées — position du centre de gravité d'une section
//! assemblée à partir de surfaces élémentaires (positives ou négatives pour les
//! évidements).
//!
//! ```text
//! aire totale     A = Σ Ai
//! centroïde       x̄ = Σ(Ai·xi) / Σ Ai        (par axe)
//! ```
//!
//! `Ai` aires élémentaires (m² ; **négatives** pour les trous/évidements), `xi`
//! positions de leurs centroïdes sur l'axe considéré. Appliquer une fois par axe
//! (`x̄`, `ȳ`) avec les positions correspondantes.
//!
//! **Convention** : unités cohérentes de l'appelant. **Limite honnête** :
//! centroïde **géométrique** d'aire (surfaces homogènes) ; pour un centre de
//! masse, remplacer les aires par des masses. Les aires et positions des éléments
//! sont fournies par l'appelant (décomposition de la section).

/// Aire totale `A = Σ Ai` (les évidements comptent en négatif).
pub fn total_area(areas: &[f64]) -> f64 {
    areas.iter().sum()
}

/// Position du centroïde sur un axe `x̄ = Σ(Ai·xi)/Σ Ai`.
///
/// Panique si `areas` et `positions` diffèrent en longueur, sont vides, ou si
/// l'aire totale est nulle.
pub fn composite_centroid(areas: &[f64], positions: &[f64]) -> f64 {
    assert!(
        areas.len() == positions.len() && !areas.is_empty(),
        "areas et positions doivent avoir la même longueur non nulle"
    );
    let total: f64 = areas.iter().sum();
    assert!(total != 0.0, "l'aire totale ne doit pas être nulle");
    let moment: f64 = areas.iter().zip(positions).map(|(a, x)| a * x).sum();
    moment / total
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn centroid_of_two_equal_areas_is_midpoint() {
        // Deux aires égales en x=0 et x=10 → centroïde à 5.
        assert_relative_eq!(
            composite_centroid(&[4.0, 4.0], &[0.0, 10.0]),
            5.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn weighted_towards_larger_area() {
        // Aire 3× plus grande en x=0 : centroïde tiré vers 0.
        // x̄ = (30·0 + 10·10)/40 = 2,5.
        assert_relative_eq!(
            composite_centroid(&[30.0, 10.0], &[0.0, 10.0]),
            2.5,
            epsilon = 1e-9
        );
    }

    #[test]
    fn hole_shifts_centroid_away() {
        // Plaque 100 en x=5 avec un trou −20 en x=8 → x̄ = (500−160)/80 = 4,25.
        let x = composite_centroid(&[100.0, -20.0], &[5.0, 8.0]);
        assert_relative_eq!(x, (500.0 - 160.0) / 80.0, epsilon = 1e-9);
        assert!(x < 5.0); // le trou décentré éloigne le centroïde
    }

    #[test]
    fn total_area_sums_with_holes() {
        assert_relative_eq!(total_area(&[100.0, -20.0, -5.0]), 75.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "aire totale ne doit pas être nulle")]
    fn zero_total_area_panics() {
        composite_centroid(&[10.0, -10.0], &[1.0, 2.0]);
    }
}
