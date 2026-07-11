//! Clavetages à clavette parallèle (ISO 773 / DIN 6885) — vérification d'une
//! liaison arbre-moyeu transmettant un couple : effort tangentiel, cisaillement
//! de la clavette et pression de matage sur les flancs.
//!
//! Le couple `T` (N·m) transmis par un arbre de diamètre `d` (mm) engendre à sa
//! surface un effort tangentiel :
//!
//! ```text
//! F = 2·T / d           (T en N·m, d en m ⇒ F en N)
//! ```
//!
//! La clavette de largeur `b`, hauteur `h` et longueur portante `l` (mm) est
//! sollicitée de deux façons :
//!
//! ```text
//! cisaillement   τ = F / (b·l)              (section cisaillée b·l)
//! matage         p = F / ((h/2)·l)          (demi-hauteur en appui)
//! ```
//!
//! Le matage (pression sur les flancs) est en général le critère dimensionnant.
//!
//! **Convention d'unités** : `T` en N·m, dimensions en mm, `F` en N, contraintes
//! et pressions en MPa (les conversions N·m → N·mm sont internes).
//!
//! **Limite honnête** : modèle usuel de dimensionnement statique à répartition
//! uniforme sur la demi-hauteur. Il ne tient pas compte de la concentration de
//! contrainte en fond de rainure, du partage réel de charge sur plusieurs
//! clavettes, ni de la fatigue — la pression admissible dépend du matériau du
//! moyeu et du régime (choc, alternance) et reste à la charge de l'appelant.

/// Effort tangentiel `F = 2·T/d` (N) à la surface d'un arbre de diamètre
/// `shaft_diameter` (mm) transmettant un couple `torque` (N·m).
///
/// Panique si `shaft_diameter <= 0`.
pub fn tangential_force(torque_nm: f64, shaft_diameter_mm: f64) -> f64 {
    assert!(
        shaft_diameter_mm > 0.0,
        "le diamètre d'arbre doit être strictement positif"
    );
    2.0 * torque_nm * 1000.0 / shaft_diameter_mm
}

/// Contrainte de cisaillement de la clavette `τ = F / (b·l)` (MPa), pour un
/// couple `torque` (N·m), un diamètre `shaft_diameter` (mm), une largeur
/// `width` (mm) et une longueur portante `length` (mm).
///
/// Panique si une dimension est non strictement positive.
pub fn key_shear_stress(
    torque_nm: f64,
    shaft_diameter_mm: f64,
    width_mm: f64,
    length_mm: f64,
) -> f64 {
    assert!(
        width_mm > 0.0 && length_mm > 0.0,
        "largeur et longueur de clavette doivent être strictement positives"
    );
    tangential_force(torque_nm, shaft_diameter_mm) / (width_mm * length_mm)
}

/// Pression de matage sur les flancs `p = F / ((h/2)·l)` (MPa), pour un couple
/// `torque` (N·m), un diamètre `shaft_diameter` (mm), une hauteur `height` (mm)
/// et une longueur portante `length` (mm).
///
/// Panique si une dimension est non strictement positive.
pub fn key_bearing_pressure(
    torque_nm: f64,
    shaft_diameter_mm: f64,
    height_mm: f64,
    length_mm: f64,
) -> f64 {
    assert!(
        height_mm > 0.0 && length_mm > 0.0,
        "hauteur et longueur de clavette doivent être strictement positives"
    );
    tangential_force(torque_nm, shaft_diameter_mm) / (height_mm / 2.0 * length_mm)
}

/// Longueur portante minimale `l` (mm) pour ne pas dépasser une pression de
/// matage admissible `allowable_pressure` (MPa) :
/// `l = F / ((h/2)·p_adm)`.
///
/// Panique si `height <= 0` ou `allowable_pressure <= 0`.
pub fn required_length_for_bearing(
    torque_nm: f64,
    shaft_diameter_mm: f64,
    height_mm: f64,
    allowable_pressure_mpa: f64,
) -> f64 {
    assert!(
        height_mm > 0.0 && allowable_pressure_mpa > 0.0,
        "hauteur et pression admissible doivent être strictement positives"
    );
    tangential_force(torque_nm, shaft_diameter_mm) / (height_mm / 2.0 * allowable_pressure_mpa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangential_force_from_torque_and_diameter() {
        // T=200 N·m, d=40 mm → F = 2·200000/40 = 10000 N.
        assert_relative_eq!(tangential_force(200.0, 40.0), 10_000.0, epsilon = 1e-9);
    }

    #[test]
    fn shear_stress_of_a_parallel_key() {
        // clavette 12×8, l=50 → τ = 10000/(12·50) ≈ 16,67 MPa.
        assert_relative_eq!(
            key_shear_stress(200.0, 40.0, 12.0, 50.0),
            10_000.0 / 600.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bearing_pressure_uses_half_height() {
        // p = 10000/((8/2)·50) = 10000/200 = 50 MPa.
        assert_relative_eq!(
            key_bearing_pressure(200.0, 40.0, 8.0, 50.0),
            50.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn bearing_dominates_shear_for_a_standard_key() {
        // Pour une clavette normalisée, le matage est plus sévère que le cisaillement.
        let tau = key_shear_stress(200.0, 40.0, 12.0, 50.0);
        let p = key_bearing_pressure(200.0, 40.0, 8.0, 50.0);
        assert!(p > tau);
    }

    #[test]
    fn required_length_inverts_the_bearing_formula() {
        // Longueur pour p_adm=50 MPa → doit redonner p=50 à cette longueur.
        let l = required_length_for_bearing(200.0, 40.0, 8.0, 50.0);
        assert_relative_eq!(l, 50.0, epsilon = 1e-9);
        assert_relative_eq!(
            key_bearing_pressure(200.0, 40.0, 8.0, l),
            50.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "diamètre d'arbre")]
    fn zero_diameter_panics() {
        tangential_force(200.0, 0.0);
    }
}
