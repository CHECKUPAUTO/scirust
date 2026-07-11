//! Rugosité théorique — état de surface géométrique laissé par la trace de
//! l'outil en tournage, indépendamment des défauts dynamiques (vibrations,
//! arête rapportée, usure).
//!
//! Pour un outil à **rayon de bec** `r` (mm) avançant de `f` (mm/tr), la
//! hauteur maximale du profil et la rugosité arithmétique valent :
//!
//! ```text
//! Rt ≈ f² / (8·r)              (hauteur crête-à-creux)
//! Ra ≈ f² / (18·√3·r)         (rugosité arithmétique, ≈ f²/32r)
//! ```
//!
//! Pour un outil à **arête vive** (bec négligeable) d'angles de direction
//! `κr` et `κr'`, ce sont les deux arêtes qui tracent les sillons :
//!
//! ```text
//! Rt = f / (cot κr + cot κr')
//! ```
//!
//! **Limite honnête** : il s'agit de la rugosité **géométrique idéale**. La
//! rugosité mesurée est toujours supérieure (facteur ~1,5 à plusieurs fois) du
//! fait de l'arête rapportée à basse vitesse, des vibrations, du rodage et de
//! l'usure — effets non déterministes que ce module ne modélise pas. Ces
//! formules servent à choisir un couple `(f, r)` de départ, pas à garantir un
//! `Ra` mesuré.

use core::f64::consts::PI;

/// Rugosité arithmétique théorique `Ra` (µm) en tournage avec rayon de bec,
/// pour une avance `feed` (mm/tr) et un rayon `nose_radius` (mm) :
/// `Ra = f² / (18·√3·r)`. `feed` et `nose_radius` en mm, résultat en **µm**.
///
/// Panique si `nose_radius <= 0`.
pub fn theoretical_ra_turning(feed_mm: f64, nose_radius_mm: f64) -> f64 {
    assert!(
        nose_radius_mm > 0.0,
        "le rayon de bec doit être strictement positif"
    );
    // f²/(18√3·r) donne des mm ; ×1000 → µm.
    feed_mm * feed_mm / (18.0 * 3f64.sqrt() * nose_radius_mm) * 1000.0
}

/// Hauteur maximale théorique `Rt` (µm) en tournage avec rayon de bec :
/// `Rt = f² / (8·r)`. `feed` et `nose_radius` en mm, résultat en **µm**.
///
/// Panique si `nose_radius <= 0`.
pub fn theoretical_rt_turning(feed_mm: f64, nose_radius_mm: f64) -> f64 {
    assert!(
        nose_radius_mm > 0.0,
        "le rayon de bec doit être strictement positif"
    );
    feed_mm * feed_mm / (8.0 * nose_radius_mm) * 1000.0
}

/// Avance `f` (mm/tr) atteignant une rugosité arithmétique cible `ra` (µm)
/// avec un rayon de bec `nose_radius` (mm), en inversant [`theoretical_ra_turning`] :
/// `f = √(Ra · 18·√3·r)`.
///
/// Panique si `ra < 0` ou `nose_radius <= 0`.
pub fn feed_for_target_ra(ra_um: f64, nose_radius_mm: f64) -> f64 {
    assert!(
        ra_um >= 0.0,
        "la rugosité cible doit être positive ou nulle"
    );
    assert!(
        nose_radius_mm > 0.0,
        "le rayon de bec doit être strictement positif"
    );
    (ra_um / 1000.0 * 18.0 * 3f64.sqrt() * nose_radius_mm).sqrt()
}

/// Hauteur maximale théorique `Rt` (µm) d'un outil à arête vive d'angles de
/// direction `kappa` (κr) et `kappa_prime` (κr'), en degrés :
/// `Rt = f / (cot κr + cot κr')`. `feed` en mm, résultat en **µm**.
///
/// Panique si `feed <= 0` ou si un angle n'est pas dans `]0°, 90°]`.
pub fn theoretical_rt_sharp(feed_mm: f64, kappa_deg: f64, kappa_prime_deg: f64) -> f64 {
    assert!(feed_mm > 0.0, "l'avance doit être strictement positive");
    assert!(
        kappa_deg > 0.0 && kappa_deg <= 90.0 && kappa_prime_deg > 0.0 && kappa_prime_deg <= 90.0,
        "les angles de direction doivent être dans ]0°, 90°]"
    );
    let cot = |deg: f64| 1.0 / (deg * PI / 180.0).tan();
    feed_mm / (cot(kappa_deg) + cot(kappa_prime_deg)) * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rt_turning_matches_the_f2_over_8r_law() {
        // f=0,2 mm, r=0,8 mm → Rt = 0,04/6,4 mm = 6,25e-3 mm = 6,25 µm.
        assert_relative_eq!(theoretical_rt_turning(0.2, 0.8), 6.25, epsilon = 1e-6);
    }

    #[test]
    fn ra_is_smaller_than_rt() {
        // Ra ≈ Rt/(8·18√3/8)… simplement Ra < Rt pour le même (f, r).
        let ra = theoretical_ra_turning(0.2, 0.8);
        let rt = theoretical_rt_turning(0.2, 0.8);
        assert!(ra < rt);
        // Ra = f²/(18√3·r)·1000 = 0,04/(31,1769·0,8)·1000 ≈ 1,6037 µm.
        assert_relative_eq!(ra, 1.60375, epsilon = 1e-4);
    }

    #[test]
    fn feed_for_target_ra_inverts_the_forward_law() {
        // Un aller-retour Ra → f → Ra doit être neutre.
        let f = feed_for_target_ra(1.6, 0.8);
        assert_relative_eq!(theoretical_ra_turning(f, 0.8), 1.6, epsilon = 1e-9);
    }

    #[test]
    fn sharp_tool_at_45_degrees() {
        // κr = κr' = 45° → cot = 1 chacun → Rt = f/2. f=0,2 → 100 µm.
        assert_relative_eq!(theoretical_rt_sharp(0.2, 45.0, 45.0), 100.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "rayon de bec")]
    fn zero_nose_radius_panics() {
        theoretical_ra_turning(0.2, 0.0);
    }
}
