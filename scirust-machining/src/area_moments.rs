//! Moments quadratiques de sections composées — théorème de **Huygens** (axes
//! parallèles), assemblage composé, rayon de giration et théorème des axes
//! perpendiculaires.
//!
//! ```text
//! transport (Huygens)   I = I_G + A·d²
//! section composée      I_tot = Σ (I_Gi + Ai·di²)
//! rayon de giration     i = √(I/A)
//! axes perpendiculaires I_p = Ix + Iy      (moment polaire)
//! ```
//!
//! `I_G` moment quadratique autour de l'axe centroïdal de l'élément, `A` aire,
//! `d` distance entre l'axe de l'élément et l'axe de référence, `Ix`/`Iy` moments
//! quadratiques autour des axes propres. Les évidements se traitent avec des
//! aires (et moments) **négatifs**.
//!
//! **Convention** : unités cohérentes de l'appelant (m⁴, mm⁴). **Limite
//! honnête** : moments quadratiques d'**aire** (RDM) ; le théorème des axes
//! perpendiculaires ne vaut que pour une **section plane**. Les moments
//! centroïdaux élémentaires proviennent de [`crate::beams`] ou de tables.

/// Transport d'un moment quadratique par le théorème de Huygens `I = I_G + A·d²`.
pub fn parallel_axis(i_centroidal: f64, area: f64, distance: f64) -> f64 {
    i_centroidal + area * distance * distance
}

/// Moment quadratique d'une **section composée** `I_tot = Σ(I_Gi + Ai·di²)`,
/// chaque élément donné par `(I_G, aire, distance à l'axe de référence)`.
pub fn composite_second_moment(components: &[(f64, f64, f64)]) -> f64 {
    components
        .iter()
        .map(|&(ig, a, d)| parallel_axis(ig, a, d))
        .sum()
}

/// Rayon de giration d'une aire `i = √(I/A)`.
///
/// Panique si `area <= 0` ou `i < 0`.
pub fn radius_of_gyration(i: f64, area: f64) -> f64 {
    assert!(area > 0.0 && i >= 0.0, "A > 0 et I ≥ 0 requis");
    (i / area).sqrt()
}

/// Moment quadratique polaire par le théorème des axes perpendiculaires
/// `I_p = Ix + Iy`.
pub fn polar_second_moment(ix: f64, iy: f64) -> f64 {
    ix + iy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn parallel_axis_increases_inertia() {
        // I augmente de A·d² en s'éloignant de l'axe centroïdal.
        assert_relative_eq!(
            parallel_axis(100.0, 20.0, 3.0),
            100.0 + 20.0 * 9.0,
            epsilon = 1e-9
        );
        assert!(parallel_axis(100.0, 20.0, 3.0) > 100.0);
    }

    #[test]
    fn composite_of_two_flanges() {
        // Deux semelles identiques (I_G=50, A=10) à d=±5 de l'axe.
        // I = 2·(50 + 10·25) = 2·300 = 600.
        let i = composite_second_moment(&[(50.0, 10.0, 5.0), (50.0, 10.0, -5.0)]);
        assert_relative_eq!(i, 600.0, epsilon = 1e-9);
    }

    #[test]
    fn radius_of_gyration_definition() {
        // I=800, A=200 → i = √4 = 2.
        assert_relative_eq!(radius_of_gyration(800.0, 200.0), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn perpendicular_axis_for_a_square() {
        // Carré : Ix = Iy → I_p = 2·Ix.
        assert_relative_eq!(
            polar_second_moment(45000.0, 45000.0),
            90000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_area_radius_panics() {
        radius_of_gyration(800.0, 0.0);
    }
}
