//! Alignement d'arbres par la méthode du comparateur à cadran (rim-and-face).
//!
//! ```text
//! désalignement angulaire   s = face_reading / dial_diameter   (pente, sans dimension)
//! offset parallèle          δ = rim_reading / 2                 (le radial lit 2×)
//! correction de cale        c = s · foot_distance               (m)
//! ```
//!
//! `face_reading` écart total lu par le comparateur axial (face) entre deux
//! positions diamétralement opposées (m), `dial_diameter` diamètre du cercle
//! balayé par la touche du comparateur axial (m), `s` pente angulaire du défaut
//! (m/m, sans dimension ; l'angle vaut `atan(s)`), `rim_reading` écart total lu
//! par le comparateur radial (rim) sur un tour (m), `δ` décalage réel des axes
//! (m), `foot_distance` distance axiale entre le plan de mesure et le pied à
//! caler (m), `c` épaisseur de cale à ajouter/retirer sous ce pied (m).
//!
//! **Convention** : SI cohérent (m pour toutes les longueurs ; pente rendue sans
//! dimension). Le comparateur radial lit **deux fois** le décalage réel des axes
//! (la barre parcourt l'écart de part et d'autre), d'où le facteur 1/2. **Limite
//! honnête** : méthode du comparateur **idéalisée** — la flèche (sag) de la barre
//! support, le jeu, l'excentricité de montage et les erreurs de lecture sont
//! **négligés** ; les valeurs de lecture, de diamètre et de géométrie sont
//! **fournies par l'appelant** (jamais de « valeur par défaut » inventée).

/// Désalignement angulaire `s = face_reading / dial_diameter` (pente sans dimension).
///
/// `face_reading` (m) est l'écart total lu par le comparateur axial entre deux
/// points diamétralement opposés, `dial_diameter` (m) le diamètre du cercle
/// balayé. Le résultat est la pente du défaut angulaire (m par m) ; l'angle
/// correspondant vaut `atan(s)`.
///
/// Panique si `dial_diameter <= 0`.
pub fn alignment_angular_misalignment(face_reading: f64, dial_diameter: f64) -> f64 {
    assert!(
        dial_diameter > 0.0,
        "le diamètre du cadran doit être strictement positif"
    );
    face_reading / dial_diameter
}

/// Offset parallèle réel des axes `δ = rim_reading / 2` (m).
///
/// `rim_reading` (m) est l'écart total (Total Indicator Reading) lu par le
/// comparateur radial sur un tour complet. Comme la touche parcourt le décalage
/// des deux côtés de l'arbre, la lecture vaut **deux fois** le décalage réel,
/// d'où la division par 2.
///
/// Panique si `rim_reading` n'est pas fini.
pub fn rim_parallel_offset(rim_reading: f64) -> f64 {
    assert!(
        rim_reading.is_finite(),
        "la lecture du comparateur radial doit être finie"
    );
    rim_reading / 2.0
}

/// Correction de cale `c = s · foot_distance` (m).
///
/// `angular_slope` (sans dimension) est la pente du défaut angulaire (issue de
/// [`alignment_angular_misalignment`]), `foot_distance` (m) la distance axiale
/// entre le plan de mesure et le pied à caler. Le résultat est l'épaisseur de
/// cale à placer sous ce pied pour annuler le défaut angulaire.
///
/// Panique si `foot_distance < 0`.
pub fn alignment_shim_correction(angular_slope: f64, foot_distance: f64) -> f64 {
    assert!(
        foot_distance >= 0.0,
        "la distance au pied à caler doit être positive ou nulle"
    );
    angular_slope * foot_distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn angular_slope_is_reading_over_diameter() {
        // face_reading = dial_diameter → pente unitaire (angle de 45°).
        assert_relative_eq!(
            alignment_angular_misalignment(0.050, 0.050),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn angular_slope_zero_when_faces_parallel() {
        // Aucun écart de face → aucun défaut angulaire.
        assert_relative_eq!(
            alignment_angular_misalignment(0.0, 0.100),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn rim_reading_is_twice_the_offset() {
        // Réciprocité : le décalage réel doit être la moitié de la lecture radiale.
        let true_offset = 0.000_15_f64;
        let rim_reading = 2.0 * true_offset;
        assert_relative_eq!(
            rim_parallel_offset(rim_reading),
            true_offset,
            epsilon = 1e-15
        );
    }

    #[test]
    fn shim_is_linear_in_distance() {
        // c = s·L : doubler la distance double la cale (proportionnalité).
        let s = alignment_angular_misalignment(0.000_10, 0.080);
        let c1 = alignment_shim_correction(s, 0.300);
        let c2 = alignment_shim_correction(s, 0.600);
        assert_relative_eq!(c2 / c1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case() {
        // Face 0,10 mm sur cadran Ø80 mm → pente = 1,25e-3.
        // Pied à 300 mm → cale = 1,25e-3 · 0,300 = 0,375 mm.
        let s = alignment_angular_misalignment(0.000_10, 0.080);
        assert_relative_eq!(s, 1.25e-3, epsilon = 1e-12);
        let shim = alignment_shim_correction(s, 0.300);
        assert_relative_eq!(shim, 0.000_375, epsilon = 1e-12);
    }

    #[test]
    fn shim_cancels_measured_face_gap_at_dial_radius() {
        // Cohérence : à la distance = diamètre du cadran, la cale reconstitue
        // exactement la lecture de face (c = (f/d)·d = f).
        let face = 0.000_08_f64;
        let diameter = 0.090_f64;
        let s = alignment_angular_misalignment(face, diameter);
        assert_relative_eq!(
            alignment_shim_correction(s, diameter),
            face,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "diamètre du cadran")]
    fn zero_diameter_panics() {
        alignment_angular_misalignment(0.001, 0.0);
    }
}
