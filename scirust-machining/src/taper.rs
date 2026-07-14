//! Cônes en tournage — géométrie d'un cône droit à génératrice rectiligne
//! (conicité, angle au sommet, diamètre courant le long de l'axe).
//!
//! Un cône est défini par son grand diamètre `D`, son petit diamètre `d` et la
//! longueur `L` séparant les deux sections. La **conicité** `C` (taper ratio)
//! est la variation de diamètre par unité de longueur ; l'angle au sommet `α`
//! (angle inclus, plein angle) en découle par la génératrice :
//!
//! ```text
//! C  = (D − d) / L                 (conicité, sans dimension)
//! α  = 2·atan(C / 2)               (angle inclus, rad)
//! d(x) = d + C·x                   (diamètre à la distance x du petit bout)
//! ```
//!
//! Légende (unités SI cohérentes — ici tout en mètres pour les longueurs) :
//! - `D`, `d` : grand et petit diamètres (m), `D ≥ d`.
//! - `L` : longueur axiale entre les deux sections (m), `L > 0`.
//! - `C` : conicité (m/m, sans dimension). Ex. cône Morse ≈ 0,05.
//! - `α` : angle inclus total du cône (rad) ; la demi-conicité vaut `α/2`.
//! - `x` : distance axiale mesurée depuis la section de petit diamètre (m).
//!
//! **Limite honnête** : ce module suppose un **cône droit idéal à génératrice
//! rectiligne** (surface conique parfaite, axe droit, sections circulaires).
//! Il ne modélise ni les défauts de forme, ni la reprise élastique, ni les
//! rayons de raccordement. Aucune valeur matériau ou tolérance n'est supposée :
//! les diamètres, longueurs et conicités normalisées (Morse, métrique, etc.)
//! sont **fournis par l'appelant** ; ce module n'invente aucune constante « par
//! défaut ».

/// Conicité `C = (D − d) / L` (sans dimension) d'un cône droit, à partir du
/// grand diamètre `large_diameter`, du petit diamètre `small_diameter` et de la
/// longueur axiale `length`. Longueurs en unités cohérentes (m).
///
/// Panique si `length <= 0`, si un diamètre est négatif, ou si
/// `large_diameter < small_diameter`.
pub fn taper_ratio(large_diameter: f64, small_diameter: f64, length: f64) -> f64 {
    assert!(
        small_diameter >= 0.0 && large_diameter >= 0.0,
        "les diamètres doivent être positifs ou nuls"
    );
    assert!(
        large_diameter >= small_diameter,
        "le grand diamètre doit être supérieur ou égal au petit diamètre"
    );
    assert!(length > 0.0, "la longueur doit être strictement positive");
    (large_diameter - small_diameter) / length
}

/// Angle inclus (plein angle au sommet) `α = 2·atan(C / 2)` (rad) d'un cône de
/// conicité `taper_ratio`.
///
/// Panique si `taper_ratio < 0`.
pub fn taper_included_angle_rad(taper_ratio: f64) -> f64 {
    assert!(
        taper_ratio >= 0.0,
        "la conicité doit être positive ou nulle"
    );
    2.0 * (taper_ratio / 2.0).atan()
}

/// Demi-angle au sommet `α/2 = atan(C / 2)` (rad), angle entre la génératrice et
/// l'axe du cône de conicité `taper_ratio`.
///
/// Panique si `taper_ratio < 0`.
pub fn taper_half_angle_rad(taper_ratio: f64) -> f64 {
    assert!(
        taper_ratio >= 0.0,
        "la conicité doit être positive ou nulle"
    );
    (taper_ratio / 2.0).atan()
}

/// Conicité `C = 2·tan(α / 2)` (sans dimension) reconstruite depuis l'angle
/// inclus `included_angle` (rad) ; inverse de [`taper_included_angle_rad`].
///
/// Panique si `included_angle` n'est pas dans `[0, π[`.
pub fn taper_ratio_from_included_angle(included_angle: f64) -> f64 {
    assert!(
        (0.0..core::f64::consts::PI).contains(&included_angle),
        "l'angle inclus doit être dans [0, π["
    );
    2.0 * (included_angle / 2.0).tan()
}

/// Diamètre `d(x) = d + C·x` (m) à la distance axiale `distance` mesurée depuis
/// la section de petit diamètre `small_diameter`, pour une conicité `taper_ratio`.
///
/// Panique si `small_diameter < 0`, `taper_ratio < 0` ou `distance < 0`.
pub fn taper_diameter_at_distance(small_diameter: f64, taper_ratio: f64, distance: f64) -> f64 {
    assert!(
        small_diameter >= 0.0,
        "le petit diamètre doit être positif ou nul"
    );
    assert!(
        taper_ratio >= 0.0,
        "la conicité doit être positive ou nulle"
    );
    assert!(distance >= 0.0, "la distance doit être positive ou nulle");
    small_diameter + taper_ratio * distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn taper_ratio_matches_definition() {
        // D=30 mm, d=20 mm, L=100 mm → C = 10/100 = 0,1.
        assert_relative_eq!(taper_ratio(0.030, 0.020, 0.100), 0.1, epsilon = 1e-12);
    }

    #[test]
    fn included_angle_and_ratio_are_reciprocal() {
        // Aller-retour C → α → C neutre (inversion exacte des deux formules).
        let c = 0.15_f64;
        let alpha = taper_included_angle_rad(c);
        assert_relative_eq!(taper_ratio_from_included_angle(alpha), c, epsilon = 1e-12);
        // L'angle inclus est le double du demi-angle.
        assert_relative_eq!(alpha, 2.0 * taper_half_angle_rad(c), epsilon = 1e-12);
    }

    #[test]
    fn diameter_grows_linearly_and_returns_endpoints() {
        // À x=0 on retrouve le petit diamètre ; à x=L le grand diamètre.
        let (d, big, l) = (0.020_f64, 0.030_f64, 0.100_f64);
        let c = taper_ratio(big, d, l);
        assert_relative_eq!(taper_diameter_at_distance(d, c, 0.0), d, epsilon = 1e-12);
        assert_relative_eq!(taper_diameter_at_distance(d, c, l), big, epsilon = 1e-12);
        // Linéarité : le diamètre à mi-longueur est la moyenne des extrêmes.
        assert_relative_eq!(
            taper_diameter_at_distance(d, c, l / 2.0),
            (d + big) / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ninety_degree_included_angle_gives_unit_ratio_component() {
        // α = 90° → tan(45°) = 1 → C = 2·1 = 2.
        assert_relative_eq!(
            taper_ratio_from_included_angle(PI / 2.0),
            2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn cylinder_has_zero_taper_and_zero_angle() {
        // D = d → conicité nulle → angle nul → diamètre constant.
        let c = taper_ratio(0.025, 0.025, 0.100);
        assert_relative_eq!(c, 0.0, epsilon = 1e-12);
        assert_relative_eq!(taper_included_angle_rad(c), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            taper_diameter_at_distance(0.025, c, 0.050),
            0.025,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "grand diamètre")]
    fn inverted_diameters_panic() {
        taper_ratio(0.010, 0.020, 0.100);
    }
}
