//! **Poussée hydrostatique** sur une surface plane immergée dans un fluide au
//! repos — force résultante, pression manométrique et position du centre de poussée.
//!
//! ```text
//! pression manométrique     p = ρ·g·z
//! force résultante          F = ρ·g·h_c·A = p(h_c)·A
//! centre de poussée         y_cp = y_c + I_c / (y_c·A)
//! plaque verticale (rect.)  F = ρ·g·b·(z_bas² − z_haut²) / 2
//! ```
//!
//! `ρ` masse volumique du fluide (kg/m³), `g` accélération de la pesanteur (m/s²),
//! `z` profondeur verticale sous la surface libre (m), `p` pression manométrique
//! (Pa = N/m²), `h_c` profondeur verticale du centre de gravité de la surface (m),
//! `A` aire de la surface immergée (m²), `F` force résultante normale à la surface
//! (N), `y_c` distance du centre de gravité **mesurée le long du plan** de la
//! surface depuis la surface libre (m), `I_c` moment quadratique de la surface
//! autour de son axe centroïdal parallèle à la surface libre (m⁴), `y_cp` position
//! du centre de poussée **le long du plan** de la surface (m), `b` largeur du
//! rectangle vertical (m), `z_haut`/`z_bas` profondeurs des arêtes supérieure et
//! inférieure du rectangle (m).
//!
//! **Convention** : SI cohérent — masses volumiques en kg/m³, longueurs en m, aires
//! en m², moments quadratiques en m⁴, pressions en Pa, forces en N. Pour une surface
//! **verticale**, `y_c = h_c` ; pour une surface **inclinée**, `y_c = h_c / sin(α)`
//! (conversion à la charge de l'appelant).
//!
//! **Limite honnête** : hydrostatique pure d'un **fluide incompressible au repos**
//! contre une **surface plane**. Pression **manométrique** (surface libre supposée à
//! la pression atmosphérique, donc l'atmosphère agit également de l'autre côté et se
//! compense). La masse volumique `ρ` et l'accélération `g` sont **fournies par
//! l'appelant** — aucune valeur de fluide ni de pesanteur « par défaut » n'est
//! inventée ici. Le centre de poussée est calculé **le long du plan** de la surface :
//! si celle-ci est inclinée, l'appelant convertit ses profondeurs inclinées en
//! profondeurs verticales (et inversement) via l'angle d'inclinaison.

/// Pression manométrique `p = ρ·g·z` à la profondeur `depth` (Pa).
///
/// Pression relative à l'atmosphère régnant sur la surface libre ; croît
/// linéairement avec la profondeur verticale.
///
/// Panique si `fluid_density < 0`, `gravity < 0` ou `depth < 0`.
pub fn submerged_pressure_at_depth(fluid_density: f64, gravity: f64, depth: f64) -> f64 {
    assert!(fluid_density >= 0.0, "la masse volumique ρ doit être ≥ 0");
    assert!(gravity >= 0.0, "l'accélération g doit être ≥ 0");
    assert!(depth >= 0.0, "la profondeur z doit être ≥ 0");
    fluid_density * gravity * depth
}

/// Force résultante `F = ρ·g·h_c·A` sur une surface plane immergée (N).
///
/// Égale à la pression au centre de gravité multipliée par l'aire ; s'applique
/// quelle que soit l'orientation de la surface, `centroid_depth` étant la
/// profondeur **verticale** du centre de gravité.
///
/// Panique si `fluid_density < 0`, `gravity < 0`, `centroid_depth < 0` ou `area < 0`.
pub fn submerged_resultant_force(
    fluid_density: f64,
    gravity: f64,
    centroid_depth: f64,
    area: f64,
) -> f64 {
    assert!(fluid_density >= 0.0, "la masse volumique ρ doit être ≥ 0");
    assert!(gravity >= 0.0, "l'accélération g doit être ≥ 0");
    assert!(
        centroid_depth >= 0.0,
        "la profondeur du centre de gravité h_c doit être ≥ 0"
    );
    assert!(area >= 0.0, "l'aire A doit être ≥ 0");
    fluid_density * gravity * centroid_depth * area
}

/// Centre de poussée `y_cp = y_c + I_c / (y_c·A)` le long du plan de la surface (m).
///
/// Position du point d'application de la force résultante, mesurée le long du plan
/// de la surface depuis la surface libre. Le terme correctif `I_c/(y_c·A)` est
/// strictement positif : le centre de poussée est **toujours plus bas** que le
/// centre de gravité (sauf `I_c = 0`, où ils coïncident).
///
/// Panique si `centroid_depth_along_surface <= 0`, `second_moment_about_centroid < 0`
/// ou `area <= 0`.
pub fn submerged_center_of_pressure(
    centroid_depth_along_surface: f64,
    second_moment_about_centroid: f64,
    area: f64,
) -> f64 {
    assert!(
        centroid_depth_along_surface > 0.0,
        "la profondeur centroïdale le long du plan y_c doit être > 0"
    );
    assert!(
        second_moment_about_centroid >= 0.0,
        "le moment quadratique I_c doit être ≥ 0"
    );
    assert!(area > 0.0, "l'aire A doit être > 0");
    centroid_depth_along_surface
        + second_moment_about_centroid / (centroid_depth_along_surface * area)
}

/// Force sur un rectangle **vertical** `F = ρ·g·b·(z_bas² − z_haut²)/2` (N).
///
/// Intègre la pression manométrique sur un rectangle vertical de largeur `width`
/// dont les arêtes horizontales sont aux profondeurs `top_depth` et `bottom_depth`.
/// Équivaut à [`submerged_resultant_force`] avec `h_c = (z_haut + z_bas)/2` et
/// `A = b·(z_bas − z_haut)`.
///
/// Panique si `fluid_density < 0`, `gravity < 0`, `width < 0`, `top_depth < 0` ou
/// `bottom_depth < top_depth`.
pub fn submerged_force_on_vertical_rectangle(
    fluid_density: f64,
    gravity: f64,
    width: f64,
    top_depth: f64,
    bottom_depth: f64,
) -> f64 {
    assert!(fluid_density >= 0.0, "la masse volumique ρ doit être ≥ 0");
    assert!(gravity >= 0.0, "l'accélération g doit être ≥ 0");
    assert!(width >= 0.0, "la largeur b doit être ≥ 0");
    assert!(
        top_depth >= 0.0,
        "la profondeur supérieure z_haut doit être ≥ 0"
    );
    assert!(
        bottom_depth >= top_depth,
        "z_bas doit être ≥ z_haut (arête inférieure plus profonde)"
    );
    fluid_density * gravity * width * (bottom_depth * bottom_depth - top_depth * top_depth) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn resultant_force_equals_centroid_pressure_times_area() {
        // Identité de définition : F = p(h_c)·A avec p(h_c) = ρ·g·h_c.
        let (rho, g, h_c, a) = (1000.0_f64, 9.81_f64, 2.0_f64, 4.0_f64);
        let f = submerged_resultant_force(rho, g, h_c, a);
        let p = submerged_pressure_at_depth(rho, g, h_c);
        assert_relative_eq!(f, p * a, epsilon = 1e-9);
    }

    #[test]
    fn vertical_rectangle_matches_generic_resultant() {
        // Pour un rectangle vertical, la formule dédiée et la formule générique
        // (avec h_c = (z_haut+z_bas)/2 et A = b·(z_bas−z_haut)) coïncident.
        let (rho, g, b, top, bottom) = (1000.0_f64, 9.81_f64, 2.0_f64, 1.0_f64, 3.0_f64);
        let f_rect = submerged_force_on_vertical_rectangle(rho, g, b, top, bottom);
        let h_c = (top + bottom) / 2.0;
        let area = b * (bottom - top);
        let f_gen = submerged_resultant_force(rho, g, h_c, area);
        assert_relative_eq!(f_rect, f_gen, epsilon = 1e-9);
    }

    #[test]
    fn pressure_is_linear_in_depth() {
        // p ∝ z à ρ et g fixés : tripler la profondeur triple la pression.
        let (rho, g) = (998.0_f64, 9.81_f64);
        let p1 = submerged_pressure_at_depth(rho, g, 1.5);
        let p3 = submerged_pressure_at_depth(rho, g, 4.5);
        assert_relative_eq!(p3, 3.0 * p1, epsilon = 1e-9);
    }

    #[test]
    fn center_of_pressure_lies_below_centroid() {
        // Le terme I_c/(y_c·A) > 0 : le centre de poussée est plus bas que y_c ;
        // il coïncide avec le centroïde lorsque I_c = 0.
        let (y_c, i_c, a) = (2.0_f64, 1.5_f64, 4.0_f64);
        let y_cp = submerged_center_of_pressure(y_c, i_c, a);
        assert!(
            y_cp > y_c,
            "le centre de poussée doit être sous le centroïde"
        );
        assert_relative_eq!(y_cp - y_c, i_c / (y_c * a), epsilon = 1e-12);
        assert_relative_eq!(
            submerged_center_of_pressure(y_c, 0.0, a),
            y_c,
            epsilon = 1e-12
        );
    }

    #[test]
    fn vertical_gate_realistic_case() {
        // Vanne rectangulaire verticale dans l'eau : ρ = 1000 kg/m³, g = 9,81 m/s²,
        // largeur b = 2 m, arête haute z_haut = 1 m, arête basse z_bas = 3 m.
        // A = 2·(3−1) = 4 m², h_c = (1+3)/2 = 2 m.
        // F = ρ·g·h_c·A = 1000·9,81·2·4 = 78 480 N.
        let f = submerged_resultant_force(1000.0, 9.81, 2.0, 4.0);
        assert_relative_eq!(f, 78_480.0, epsilon = 1e-6);
        // Formule dédiée : F = 1000·9,81·2·(3²−1²)/2 = 1000·9,81·2·8/2 = 78 480 N.
        let f_rect = submerged_force_on_vertical_rectangle(1000.0, 9.81, 2.0, 1.0, 3.0);
        assert_relative_eq!(f_rect, 78_480.0, epsilon = 1e-6);
        // Centre de poussée : I_c = b·H³/12 = 2·2³/12 = 4/3 m⁴, y_c = 2 m, A = 4 m².
        // y_cp = 2 + (4/3)/(2·4) = 2 + 1/6 = 2,166 666… m.
        let y_cp = submerged_center_of_pressure(2.0, 4.0 / 3.0, 4.0);
        assert_relative_eq!(y_cp, 2.0 + 1.0 / 6.0, epsilon = 1e-12);
        assert_relative_eq!(y_cp, 2.166_666_666_666_667, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "l'aire A doit être > 0")]
    fn zero_area_center_of_pressure_panics() {
        submerged_center_of_pressure(2.0, 1.0, 0.0);
    }
}
