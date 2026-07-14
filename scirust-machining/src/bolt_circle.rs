//! Cercle de perçage (PCD, *pitch circle diameter*) — géométrie d'un ensemble de
//! `n` trous régulièrement espacés sur un cercle de diamètre `D`.
//!
//! ```text
//! rayon du cercle          R = D/2
//! angle du trou i          θ_i = 2·π·i / n
//! position du trou i       (x, y) = (R·cos θ_i, R·sin θ_i)
//! corde entre trous voisins c = D·sin(π/n)
//! ```
//!
//! `D` diamètre du cercle de perçage (m), `n` nombre de trous (`n ≥ 1`, et
//! `n ≥ 2` pour la corde), `i` indice du trou (0-based), `θ_i` angle du trou (rad,
//! mesuré depuis l'axe `+x`), `(x, y)` position du centre du trou (m), `c` longueur
//! de corde reliant deux trous adjacents (m).
//!
//! **Convention** : angles en rad, longueurs en m (SI cohérent, tout facteur
//! d'échelle est conservé). Le trou `i = 0` est placé sur l'axe `+x`.
//!
//! **Limite honnête** : géométrie exacte d'un cercle de perçage idéal à trous
//! régulièrement espacés (pas de tolérance, pas de décalage angulaire de la
//! première position, trous ponctuels). Le diamètre `D` et le nombre de trous `n`
//! sont FOURNIS par l'appelant ; ce module n'invente aucune valeur « par défaut »
//! de PCD, de diamètre de trou ni de standard de bride.

use core::f64::consts::PI;

/// Angle du trou d'indice `index` (0-based) parmi `n_holes` : `θ = 2·π·index / n` (rad).
///
/// Angle mesuré depuis l'axe `+x`, dans le sens trigonométrique.
///
/// Panique si `n_holes == 0` ou si `index >= n_holes`.
pub fn bolt_hole_angle_rad(n_holes: u32, index: u32) -> f64 {
    assert!(
        n_holes >= 1,
        "le cercle de perçage doit compter au moins 1 trou"
    );
    assert!(
        index < n_holes,
        "l'indice du trou doit être strictement inférieur au nombre de trous"
    );
    2.0 * PI * index as f64 / n_holes as f64
}

/// Position `(x, y)` du centre du trou `index` (0-based) sur un cercle de perçage.
///
/// `(x, y) = (R·cos θ, R·sin θ)` avec `R = pcd_diameter/2` et `θ = 2·π·index/n` (m).
///
/// Panique si `pcd_diameter < 0`, si `n_holes == 0` ou si `index >= n_holes`.
pub fn bolt_hole_position(pcd_diameter: f64, n_holes: u32, index: u32) -> (f64, f64) {
    assert!(
        pcd_diameter >= 0.0,
        "le diamètre du cercle de perçage doit être positif ou nul"
    );
    let theta = bolt_hole_angle_rad(n_holes, index);
    let radius = pcd_diameter / 2.0;
    (radius * theta.cos(), radius * theta.sin())
}

/// Longueur de corde entre deux trous adjacents : `c = D·sin(π/n)` (m).
///
/// Distance rectiligne (centre à centre) séparant deux trous voisins du cercle.
///
/// Panique si `pcd_diameter < 0` ou si `n_holes < 2`.
pub fn chord_between_holes(pcd_diameter: f64, n_holes: u32) -> f64 {
    assert!(
        pcd_diameter >= 0.0,
        "le diamètre du cercle de perçage doit être positif ou nul"
    );
    assert!(
        n_holes >= 2,
        "une corde entre trous voisins nécessite au moins 2 trous"
    );
    pcd_diameter * (PI / n_holes as f64).sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn first_hole_sits_on_positive_x_axis() {
        // Le trou 0 est à l'angle 0 : position (R, 0), quel que soit n.
        for &n in &[1u32, 3, 4, 6, 12]
        {
            assert_relative_eq!(bolt_hole_angle_rad(n, 0), 0.0, epsilon = 1e-12);
            let (x, y) = bolt_hole_position(100.0, n, 0);
            assert_relative_eq!(x, 50.0, epsilon = 1e-12);
            assert_relative_eq!(y, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn angles_are_evenly_spaced_and_span_full_turn() {
        // Écart angulaire constant = 2π/n ; le dernier trou est à 2π(n−1)/n.
        let n = 8u32;
        let step = 2.0 * PI / n as f64;
        for i in 0..n
        {
            assert_relative_eq!(bolt_hole_angle_rad(n, i), step * i as f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn all_holes_lie_on_the_pitch_circle() {
        // Chaque trou est à distance R = D/2 du centre.
        let d = 250.0_f64;
        let n = 6u32;
        for i in 0..n
        {
            let (x, y) = bolt_hole_position(d, n, i);
            assert_relative_eq!((x * x + y * y).sqrt(), d / 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn chord_equals_euclidean_distance_between_neighbours() {
        // Identité : la formule D·sin(π/n) doit égaler la distance calculée
        // entre les positions des trous 0 et 1.
        let d = 180.0_f64;
        for &n in &[2u32, 3, 4, 5, 6, 12]
        {
            let (x0, y0) = bolt_hole_position(d, n, 0);
            let (x1, y1) = bolt_hole_position(d, n, 1);
            let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            assert_relative_eq!(chord_between_holes(d, n), dist, epsilon = 1e-9);
        }
    }

    #[test]
    fn square_pattern_chord_is_diameter_over_sqrt_two() {
        // n=4 : corde = D·sin(45°) = D/√2 ≈ 0,7071·D.
        let d = 100.0_f64;
        assert_relative_eq!(
            chord_between_holes(d, 4),
            d / 2.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn chord_scales_linearly_with_diameter() {
        // Proportionnalité : doubler D double la corde (n fixé).
        let n = 5u32;
        let c1 = chord_between_holes(80.0, n);
        let c2 = chord_between_holes(160.0, n);
        assert_relative_eq!(c2, 2.0 * c1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins 2 trous")]
    fn single_hole_has_no_chord() {
        chord_between_holes(120.0, 1);
    }
}
