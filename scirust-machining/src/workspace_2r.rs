//! Espace de travail (anneau atteignable) d'un bras robotisé planaire **2R**
//! (deux segments rigides reliés par deux liaisons rotoïdes).
//!
//! ```text
//! portée maximale   r_max = l1 + l2
//! portée minimale   r_min = |l1 − l2|
//! atteignable       r_min ≤ d ≤ r_max
//! aire de l'anneau  A = π·(r_max² − r_min²) = π·((l1+l2)² − (l1−l2)²) = 4·π·l1·l2
//! ```
//!
//! `l1`, `l2` longueurs des deux segments (m, ≥ 0), `r_max` rayon extérieur de
//! l'espace de travail (m), `r_min` rayon intérieur / trou central (m),
//! `d` distance radiale de la base au point visé (m, ≥ 0), `A` aire de la couronne
//! balayée par l'outil dans le plan (m²).
//!
//! **Convention** : longueurs en m, aire en m² (SI cohérent, tout facteur d'échelle
//! est conservé). L'origine est la base du bras.
//!
//! **Limite honnête** : anneau de travail géométrique idéal — segments parfaitement
//! rigides, mouvement plan, aucune butée articulaire ni obstacle, l'outil balaie la
//! couronne complète. Les longueurs `l1`, `l2` et la distance cible `d` sont FOURNIES
//! par l'appelant ; ce module n'invente aucune valeur « par défaut » de géométrie de
//! bras, de course articulaire ni de limite mécanique.

use core::f64::consts::PI;

/// Portée maximale de l'outil : `r_max = l1 + l2` (m).
///
/// Atteinte lorsque les deux segments sont alignés (bras déployé).
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn ws2r_max_reach(l1: f64, l2: f64) -> f64 {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    l1 + l2
}

/// Portée minimale de l'outil : `r_min = |l1 − l2|` (m).
///
/// Atteinte lorsque le second segment se replie sur le premier ; rayon du trou
/// central non atteignable (nul si `l1 == l2`).
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn ws2r_min_reach(l1: f64, l2: f64) -> f64 {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    (l1 - l2).abs()
}

/// Teste si une cible à distance radiale `d` est atteignable : `r_min ≤ d ≤ r_max`.
///
/// Renvoie `true` si `|l1 − l2| ≤ d ≤ l1 + l2`, `false` sinon (cible trop proche
/// dans le trou central, ou trop lointaine au-delà du bras déployé).
///
/// Panique si `l1 < 0`, `l2 < 0` ou `target_distance < 0`.
pub fn ws2r_is_reachable(l1: f64, l2: f64, target_distance: f64) -> bool {
    assert!(
        l1 >= 0.0,
        "la longueur du premier segment doit être positive ou nulle"
    );
    assert!(
        l2 >= 0.0,
        "la longueur du second segment doit être positive ou nulle"
    );
    assert!(
        target_distance >= 0.0,
        "la distance cible doit être positive ou nulle"
    );
    let r_min = (l1 - l2).abs();
    let r_max = l1 + l2;
    (r_min..=r_max).contains(&target_distance)
}

/// Aire de l'anneau de travail : `A = π·(r_max² − r_min²) = 4·π·l1·l2` (m²).
///
/// Surface de la couronne balayée par l'outil entre les rayons `r_min` et `r_max`.
///
/// Panique si `l1 < 0` ou si `l2 < 0`.
pub fn ws2r_workspace_area(l1: f64, l2: f64) -> f64 {
    let r_max = ws2r_max_reach(l1, l2);
    let r_min = ws2r_min_reach(l1, l2);
    PI * (r_max * r_max - r_min * r_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn max_reach_is_sum_min_reach_is_abs_difference() {
        // Identités de base des deux portées.
        let (l1, l2) = (0.30_f64, 0.20_f64);
        assert_relative_eq!(ws2r_max_reach(l1, l2), 0.50, epsilon = 1e-12);
        assert_relative_eq!(ws2r_min_reach(l1, l2), 0.10, epsilon = 1e-12);
        // r_min est symétrique en l1 ↔ l2, r_max aussi.
        assert_relative_eq!(ws2r_max_reach(l2, l1), 0.50, epsilon = 1e-12);
        assert_relative_eq!(ws2r_min_reach(l2, l1), 0.10, epsilon = 1e-12);
    }

    #[test]
    fn equal_segments_close_the_central_hole() {
        // l1 = l2 : le trou central disparaît (r_min = 0) et l'anneau devient un disque.
        let l = 0.25_f64;
        assert_relative_eq!(ws2r_min_reach(l, l), 0.0, epsilon = 1e-12);
        // A = π·r_max² = π·(2l)² = 4·π·l².
        assert_relative_eq!(
            ws2r_workspace_area(l, l),
            PI * (2.0 * l) * (2.0 * l),
            epsilon = 1e-12
        );
        assert!(ws2r_is_reachable(l, l, 0.0));
    }

    #[test]
    fn area_equals_four_pi_l1_l2_identity() {
        // Identité algébrique : π·(r_max² − r_min²) = 4·π·l1·l2.
        for &(l1, l2) in &[(0.30_f64, 0.20_f64), (0.42, 0.27), (0.15, 0.55)]
        {
            assert_relative_eq!(
                ws2r_workspace_area(l1, l2),
                4.0 * PI * l1 * l2,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn reachability_matches_ring_bounds() {
        // Les bornes exactes sont atteignables, un pas au-delà ne l'est pas.
        let (l1, l2) = (0.35_f64, 0.20_f64);
        let r_min = ws2r_min_reach(l1, l2); // 0.15
        let r_max = ws2r_max_reach(l1, l2); // 0.55
        assert!(ws2r_is_reachable(l1, l2, r_min));
        assert!(ws2r_is_reachable(l1, l2, r_max));
        assert!(ws2r_is_reachable(l1, l2, 0.5 * (r_min + r_max)));
        assert!(!ws2r_is_reachable(l1, l2, r_min - 1e-6));
        assert!(!ws2r_is_reachable(l1, l2, r_max + 1e-6));
    }

    #[test]
    fn area_scales_quadratically_with_uniform_scaling() {
        // Homothétie de facteur k sur les deux segments : l'aire est multipliée par k².
        let (l1, l2, k) = (0.28_f64, 0.19_f64, 2.5_f64);
        assert_relative_eq!(
            ws2r_workspace_area(k * l1, k * l2),
            k * k * ws2r_workspace_area(l1, l2),
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_scara_case() {
        // Bras type SCARA : l1 = 0.325 m, l2 = 0.275 m.
        let (l1, l2) = (0.325_f64, 0.275_f64);
        assert_relative_eq!(ws2r_max_reach(l1, l2), 0.600, epsilon = 1e-12);
        assert_relative_eq!(ws2r_min_reach(l1, l2), 0.050, epsilon = 1e-12);
        // A = 4·π·0.325·0.275 = 1.122548... m².
        assert_relative_eq!(
            ws2r_workspace_area(l1, l2),
            4.0 * PI * 0.325 * 0.275,
            epsilon = 1e-12
        );
        assert!(ws2r_is_reachable(l1, l2, 0.40));
        assert!(!ws2r_is_reachable(l1, l2, 0.01));
    }

    #[test]
    #[should_panic(expected = "premier segment doit être positive")]
    fn negative_first_segment_panics() {
        ws2r_workspace_area(-0.1, 0.2);
    }
}
