//! Cintrage de tube — géométrie de la fibre extérieure : déformation de la peau
//! externe, rayon de cintrage minimal admissible et longueur de la fibre neutre.
//!
//! ```text
//! déformation fibre externe   ε = OD / (2·R)
//! rayon minimal admissible    R_min = OD / (2·ε_adm)   (réciproque de la précédente)
//! longueur de fibre neutre    L = R · θ
//! ```
//!
//! `OD` diamètre extérieur du tube (m), `R` rayon de cintrage mesuré à l'axe du
//! tube (m), `ε` déformation d'ingénierie (sans dimension) de la fibre externe à
//! l'extrados, `ε_adm` déformation maximale admissible (sans dimension), `θ`
//! angle de cintrage (rad), `L` longueur d'arc de la fibre neutre (m).
//! `outer_fiber_strain` et `minimum_bend_radius` sont **réciproques** :
//! `outer_fiber_strain(OD, minimum_bend_radius(OD, ε)) = ε`.
//!
//! **Convention** : SI cohérent (longueurs en m, angles en rad) ; il suffit que
//! `OD` et `R` partagent la même unité de longueur pour un ratio correct.
//! **Limite honnête** : la fibre neutre est supposée **au centre** du tube
//! (approximation ; en pratique elle se décale vers l'intrados sous l'effet de
//! l'aplatissement et de l'amincissement de paroi, ce qui surestime légèrement
//! `ε`). Le modèle est purement géométrique : il n'inclut ni le **retour
//! élastique** (springback), ni l'ovalisation, ni l'amincissement, ni l'effort
//! de cintrage. Aucune déformation admissible ni propriété matière n'est imposée :
//! `ε_adm` est **fournie par l'appelant** d'après le couple matière/procédé.

/// Déformation d'ingénierie de la fibre externe `ε = OD / (2·R)` (sans
/// dimension), fibre neutre supposée au centre du tube.
///
/// Panique si `tube_outer_diameter < 0` ou si `bend_radius <= 0`.
pub fn outer_fiber_strain(tube_outer_diameter: f64, bend_radius: f64) -> f64 {
    assert!(
        tube_outer_diameter >= 0.0,
        "le diamètre extérieur OD ne peut être négatif"
    );
    assert!(
        bend_radius > 0.0,
        "le rayon de cintrage R doit être strictement positif"
    );
    tube_outer_diameter / (2.0 * bend_radius)
}

/// Rayon de cintrage minimal admissible `R_min = OD / (2·ε_adm)` (m) pour ne pas
/// dépasser la déformation `max_allowable_strain` sur la fibre externe.
///
/// Panique si `tube_outer_diameter < 0` ou si `max_allowable_strain <= 0`.
pub fn minimum_bend_radius(tube_outer_diameter: f64, max_allowable_strain: f64) -> f64 {
    assert!(
        tube_outer_diameter >= 0.0,
        "le diamètre extérieur OD ne peut être négatif"
    );
    assert!(
        max_allowable_strain > 0.0,
        "la déformation admissible ε_adm doit être strictement positive"
    );
    tube_outer_diameter / (2.0 * max_allowable_strain)
}

/// Longueur d'arc de la fibre neutre `L = R · θ` (m), fibre neutre supposée au
/// centre du tube.
///
/// Panique si `bend_radius < 0` ou si `bend_angle_rad < 0`.
pub fn neutral_axis_length(bend_radius: f64, bend_angle_rad: f64) -> f64 {
    assert!(
        bend_radius >= 0.0,
        "le rayon de cintrage R ne peut être négatif"
    );
    assert!(
        bend_angle_rad >= 0.0,
        "l'angle de cintrage θ ne peut être négatif"
    );
    bend_radius * bend_angle_rad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn strain_and_radius_are_reciprocal() {
        // Réciprocité : strain(OD, minimum_bend_radius(OD, ε)) = ε.
        let od = 0.025_4_f64;
        for &eps in &[0.02_f64, 0.05, 0.10, 0.15]
        {
            let r_min = minimum_bend_radius(od, eps);
            assert_relative_eq!(outer_fiber_strain(od, r_min), eps, max_relative = 1e-12);
        }
    }

    #[test]
    fn strain_equals_half_when_radius_equals_diameter() {
        // R = OD → ε = OD/(2·OD) = 1/2, indépendant de l'unité de longueur.
        let od = 0.05_f64;
        assert_relative_eq!(outer_fiber_strain(od, od), 0.5, max_relative = 1e-12);
    }

    #[test]
    fn strain_is_inversely_proportional_to_radius() {
        // Doubler le rayon de cintrage divise la déformation par deux.
        let od = 0.030_f64;
        let tight = outer_fiber_strain(od, 0.100);
        let loose = outer_fiber_strain(od, 0.200);
        assert_relative_eq!(tight, 2.0 * loose, max_relative = 1e-12);
    }

    #[test]
    fn neutral_length_scales_with_angle_and_radius() {
        // L = R·θ : demi-tour (θ = π) d'un rayon 0,150 m → arc = 0,150·π.
        let r = 0.150_f64;
        assert_relative_eq!(neutral_axis_length(r, PI), r * PI, max_relative = 1e-12);
        // Angle nul → longueur nulle.
        assert_relative_eq!(neutral_axis_length(r, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_one_inch_tube_case() {
        // Tube 1" (OD = 0,025 4 m) cintré à R = 0,0762 m (rayon 3·OD) :
        // ε = 0,025 4 / (2·0,076 2) = 1/6 ≈ 0,166 67.
        let od = 0.025_4_f64;
        let r = 0.076_2_f64;
        let eps = outer_fiber_strain(od, r);
        assert_relative_eq!(eps, od / (2.0 * r), max_relative = 1e-12);
        assert_relative_eq!(eps, 1.0 / 6.0, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le rayon de cintrage R doit être strictement positif")]
    fn zero_radius_panics() {
        outer_fiber_strain(0.025, 0.0);
    }
}
