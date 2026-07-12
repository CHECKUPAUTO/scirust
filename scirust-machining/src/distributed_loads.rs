//! Charges réparties — résultante (effort équivalent) et sa position pour les
//! répartitions usuelles : uniforme, triangulaire et trapézoïdale.
//!
//! ```text
//! uniforme       R = w·L            à L/2
//! triangulaire   R = ½·w_max·L      à 2L/3 du côté nul (L/3 du côté max)
//! trapézoïdale   R = (w1 + w2)/2·L
//! ```
//!
//! `w`/`w_max` intensités linéiques (N/m), `w1`/`w2` intensités aux extrémités,
//! `L` longueur chargée (m). La résultante est l'**aire** du diagramme de charge,
//! appliquée au **centroïde** de ce diagramme.
//!
//! **Convention** : SI cohérent, position mesurée depuis l'origine indiquée.
//! **Limite honnête** : réduction statique d'une charge répartie à sa
//! résultante (calcul de réactions/équilibre) ; ne donne pas le diagramme des
//! efforts internes le long de la poutre.

/// Résultante d'une charge **uniforme** `R = w·L` (N).
pub fn uniform_resultant(intensity: f64, length: f64) -> f64 {
    intensity * length
}

/// Position de la résultante d'une charge uniforme `L/2` (depuis une extrémité).
pub fn uniform_centroid(length: f64) -> f64 {
    length / 2.0
}

/// Résultante d'une charge **triangulaire** `R = ½·w_max·L` (N).
pub fn triangular_resultant(peak_intensity: f64, length: f64) -> f64 {
    0.5 * peak_intensity * length
}

/// Position de la résultante d'une charge triangulaire, mesurée depuis le côté
/// d'intensité **nulle** : `2L/3`.
pub fn triangular_centroid_from_zero(length: f64) -> f64 {
    2.0 * length / 3.0
}

/// Résultante d'une charge **trapézoïdale** `R = (w1 + w2)/2·L` (N).
pub fn trapezoidal_resultant(intensity_start: f64, intensity_end: f64, length: f64) -> f64 {
    0.5 * (intensity_start + intensity_end) * length
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn uniform_load_resultant_and_position() {
        // w=500 N/m sur 4 m → R=2000 N à 2 m.
        assert_relative_eq!(uniform_resultant(500.0, 4.0), 2000.0, epsilon = 1e-9);
        assert_relative_eq!(uniform_centroid(4.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn triangular_load_resultant_and_position() {
        // w_max=600 N/m sur 3 m → R=900 N à 2 m du côté nul.
        assert_relative_eq!(triangular_resultant(600.0, 3.0), 900.0, epsilon = 1e-9);
        assert_relative_eq!(triangular_centroid_from_zero(3.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn trapezoid_reduces_to_uniform_and_triangle() {
        // w1=w2=w : trapèze = uniforme. w1=0 : trapèze = triangle.
        assert_relative_eq!(
            trapezoidal_resultant(500.0, 500.0, 4.0),
            uniform_resultant(500.0, 4.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            trapezoidal_resultant(0.0, 600.0, 3.0),
            triangular_resultant(600.0, 3.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn trapezoid_is_average_times_length() {
        // w1=200, w2=800, L=5 → R = 500·5 = 2500 N.
        assert_relative_eq!(
            trapezoidal_resultant(200.0, 800.0, 5.0),
            2500.0,
            epsilon = 1e-9
        );
    }
}
