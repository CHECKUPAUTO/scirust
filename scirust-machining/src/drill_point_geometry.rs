//! Géométrie de la pointe conique d'un foret (**drill point geometry**) —
//! hauteur du cône de pointe, longueur d'arête et course supplémentaire de
//! débouchage, distinctes des efforts de perçage.
//!
//! ```text
//! hauteur de pointe   h = (d/2) / tan(σ/2)           (hauteur du cône)
//! longueur d'arête    l = (d/2) / sin(σ/2)           (arête tranchante)
//! course de débouchage t = h                          (dépassement pour percer de part en part)
//! ```
//!
//! `d` diamètre du foret (m), `σ` angle de pointe total (rad, 118° ≈ 2,06 rad
//! usuel), `h` hauteur du cône de pointe (m), `l` longueur de l'arête
//! tranchante du sommet au bord (m), `t` course axiale supplémentaire nécessaire
//! pour que la pointe traverse complètement la pièce (m). Par construction du
//! triangle rectangle (rayon, hauteur, arête) : `l² = h² + (d/2)²` et
//! `h = l · cos(σ/2)`.
//!
//! **Convention** : SI cohérent (longueurs en m, angles en rad). Pointe conique
//! **symétrique** d'angle total `σ ∈ ]0, π[` ; l'angle est mesuré au sommet
//! entre les deux arêtes.
//!
//! **Limite honnête** : modèle purement **géométrique** d'un cône symétrique
//! parfait (ni amincissement d'âme, ni affûtage particulier, ni usure). Le
//! diamètre `d` et l'angle de pointe `σ` sont **fournis par l'appelant** ;
//! aucune valeur « par défaut » de foret ou de matériau n'est supposée.

use core::f64::consts::PI;

/// Hauteur du cône de pointe `h = (d/2) / tan(σ/2)` (m), du sommet au plan
/// passant par les becs, pour un foret de diamètre `diameter` et d'angle de
/// pointe total `point_angle_rad`.
///
/// Panique si `diameter <= 0` ou si `point_angle_rad` sort de `]0, π[`.
pub fn drill_point_length(diameter: f64, point_angle_rad: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre d doit être strictement positif"
    );
    assert!(
        point_angle_rad > 0.0 && point_angle_rad < PI,
        "l'angle de pointe σ doit être compris dans ]0, π[ rad"
    );
    (diameter / 2.0) / (point_angle_rad / 2.0).tan()
}

/// Longueur de l'arête tranchante `l = (d/2) / sin(σ/2)` (m), du sommet au bec,
/// pour un foret de diamètre `diameter` et d'angle de pointe total
/// `point_angle_rad`.
///
/// Panique si `diameter <= 0` ou si `point_angle_rad` sort de `]0, π[`.
pub fn drill_point_lip_length(diameter: f64, point_angle_rad: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre d doit être strictement positif"
    );
    assert!(
        point_angle_rad > 0.0 && point_angle_rad < PI,
        "l'angle de pointe σ doit être compris dans ]0, π[ rad"
    );
    (diameter / 2.0) / (point_angle_rad / 2.0).sin()
}

/// Course axiale supplémentaire `t = h = (d/2) / tan(σ/2)` (m) à ajouter à
/// l'épaisseur de la pièce pour que la pointe conique débouche complètement,
/// pour un foret de diamètre `diameter` et d'angle de pointe `point_angle_rad`.
///
/// Panique si `diameter <= 0` ou si `point_angle_rad` sort de `]0, π[`.
pub fn drill_point_extra_travel(diameter: f64, point_angle_rad: f64) -> f64 {
    drill_point_length(diameter, point_angle_rad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn right_triangle_identity_holds() {
        // Triangle rectangle (rayon, hauteur, arête) : l² = h² + (d/2)².
        let d = 12.0e-3_f64;
        for &sigma in &[0.5_f64, 1.0, 2.06, 3.0]
        {
            let h = drill_point_length(d, sigma);
            let l = drill_point_lip_length(d, sigma);
            assert_relative_eq!(l * l, h * h + (d / 2.0).powi(2), max_relative = 1e-12);
        }
    }

    #[test]
    fn height_is_lip_projected_on_axis() {
        // Projection : h = l · cos(σ/2).
        let d = 8.0e-3_f64;
        let sigma = 2.06_f64;
        let h = drill_point_length(d, sigma);
        let l = drill_point_lip_length(d, sigma);
        assert_relative_eq!(h, l * (sigma / 2.0).cos(), max_relative = 1e-12);
    }

    #[test]
    fn extra_travel_equals_point_length() {
        // La course de débouchage est identiquement la hauteur de pointe.
        let d = 10.0e-3_f64;
        let sigma = 118.0_f64.to_radians();
        assert_relative_eq!(
            drill_point_extra_travel(d, sigma),
            drill_point_length(d, sigma),
            max_relative = 1e-12
        );
    }

    #[test]
    fn geometry_is_proportional_to_diameter() {
        // h et l sont linéaires en d : doubler le diamètre double les longueurs.
        let sigma = 1.7_f64;
        let h1 = drill_point_length(5.0e-3, sigma);
        let h2 = drill_point_length(10.0e-3, sigma);
        let l1 = drill_point_lip_length(5.0e-3, sigma);
        let l2 = drill_point_lip_length(10.0e-3, sigma);
        assert_relative_eq!(h2, 2.0 * h1, max_relative = 1e-12);
        assert_relative_eq!(l2, 2.0 * l1, max_relative = 1e-12);
    }

    #[test]
    fn right_angle_point_gives_half_diameter_height() {
        // Cas limite chiffré : σ = 90° ⇒ tan(45°) = 1 ⇒ h = d/2.
        let d = 10.0e-3_f64;
        let h = drill_point_length(d, PI / 2.0);
        assert_relative_eq!(h, d / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_118_degree_drill_case() {
        // Foret standard d = 10 mm, angle de pointe 118°.
        // h = 5 / tan(59°) = 5 / 1,66428… ≈ 3,00432 mm.
        let d = 10.0e-3_f64;
        let sigma = 118.0_f64.to_radians();
        let h = drill_point_length(d, sigma);
        assert_relative_eq!(h, 3.004_32e-3, max_relative = 1e-5);
    }

    #[test]
    #[should_panic(expected = "l'angle de pointe σ doit être compris dans ]0, π[")]
    fn flat_point_angle_panics() {
        drill_point_length(10.0e-3, PI);
    }
}
