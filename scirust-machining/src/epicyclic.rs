//! Trains épicycloïdaux (planétaires) — équation de **Willis** reliant les
//! vitesses du planétaire (soleil), de la couronne et du porte-satellites.
//!
//! ```text
//! nombre de dents couronne  N_R = N_S + 2·N_P
//! équation de Willis        (ω_S − ω_C)/(ω_R − ω_C) = −N_R/N_S
//! réducteur (couronne fixe) ω_S/ω_C = 1 + N_R/N_S   (soleil→porte-satellites)
//! ```
//!
//! `ω_S` vitesse du soleil, `ω_R` de la couronne, `ω_C` du porte-satellites
//! (carrier) — rad/s ou tr/min, unité cohérente. `N_S, N_P, N_R` nombres de
//! dents (soleil, satellite, couronne). Le rapport `−N_R/N_S` est la valeur du
//! train à porte-satellites bloqué.
//!
//! **Convention** : vitesses algébriques (signe = sens de rotation), même unité.
//! **Limite honnête** : cinématique **exacte** d'un train planétaire simple
//! (un étage soleil-satellite-couronne) ; ne traite ni les trains composés à
//! plusieurs étages, ni la répartition de couple entre satellites, ni le
//! rendement.

/// Nombre de dents de la couronne d'un train planétaire coaxial
/// `N_R = N_S + 2·N_P`.
pub fn ring_teeth(sun_teeth: u32, planet_teeth: u32) -> u32 {
    sun_teeth + 2 * planet_teeth
}

/// Valeur du train à porte-satellites bloqué `k = −N_R/N_S`
/// (rapport `(ω_R − ω_C)/(ω_S − ω_C)` de Willis).
///
/// Panique si `sun_teeth == 0`.
pub fn willis_ratio(sun_teeth: u32, ring_teeth: u32) -> f64 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    -(ring_teeth as f64) / (sun_teeth as f64)
}

/// Vitesse du porte-satellites déduite du soleil et de la couronne (Willis).
///
/// `ω_C = (ω_S − k·ω_R)/(1 − k)` avec `k = −N_R/N_S`. Panique si `sun_teeth == 0`.
pub fn carrier_speed(omega_sun: f64, omega_ring: f64, sun_teeth: u32, ring_teeth: u32) -> f64 {
    let k = willis_ratio(sun_teeth, ring_teeth);
    (omega_sun - k * omega_ring) / (1.0 - k)
}

/// Vitesse du soleil déduite du porte-satellites et de la couronne (Willis).
///
/// `ω_S = ω_C + k·(ω_R − ω_C)`.
pub fn sun_speed(omega_carrier: f64, omega_ring: f64, sun_teeth: u32, ring_teeth: u32) -> f64 {
    let k = willis_ratio(sun_teeth, ring_teeth);
    omega_carrier + k * (omega_ring - omega_carrier)
}

/// Vitesse de la couronne déduite du porte-satellites et du soleil (Willis).
///
/// `ω_R = ω_C + (ω_S − ω_C)/k`.
pub fn ring_speed(omega_carrier: f64, omega_sun: f64, sun_teeth: u32, ring_teeth: u32) -> f64 {
    let k = willis_ratio(sun_teeth, ring_teeth);
    omega_carrier + (omega_sun - omega_carrier) / k
}

/// Rapport de réduction **soleil → porte-satellites, couronne fixe**
/// `i = ω_S/ω_C = 1 + N_R/N_S`.
///
/// Panique si `sun_teeth == 0`.
pub fn reduction_ratio_ring_fixed(sun_teeth: u32, ring_teeth: u32) -> f64 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    1.0 + (ring_teeth as f64) / (sun_teeth as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ring_teeth_from_coaxial_condition() {
        // N_S=20, N_P=15 → N_R = 20 + 30 = 50.
        assert_eq!(ring_teeth(20, 15), 50);
    }

    #[test]
    fn reduction_ratio_when_ring_fixed() {
        // Couronne fixe, soleil entrée, porte-satellites sortie : i = 1 + 50/20 = 3,5.
        assert_relative_eq!(reduction_ratio_ring_fixed(20, 50), 3.5, epsilon = 1e-12);
    }

    #[test]
    fn carrier_fixed_reverses_and_scales() {
        // Porte-satellites bloqué (ω_C=0) : ω_R = −ω_S·N_S/N_R (inversion de sens).
        let wr = ring_speed(0.0, 100.0, 20, 50);
        assert_relative_eq!(wr, -100.0 * 20.0 / 50.0, epsilon = 1e-9);
    }

    #[test]
    fn willis_relations_are_self_consistent() {
        // Partant de (ω_S, ω_C) on calcule ω_R, puis on doit retrouver ω_C.
        let (ws, wc) = (300.0, 80.0);
        let wr = ring_speed(wc, ws, 20, 50);
        assert_relative_eq!(carrier_speed(ws, wr, 20, 50), wc, epsilon = 1e-9);
        // et retrouver ω_S depuis (ω_C, ω_R).
        assert_relative_eq!(sun_speed(wc, wr, 20, 50), ws, epsilon = 1e-9);
    }

    #[test]
    fn ring_fixed_ratio_matches_willis() {
        // Couronne fixe (ω_R=0) : ω_S/ω_C doit valoir 1 + N_R/N_S.
        let ws = sun_speed(100.0, 0.0, 20, 50);
        assert_relative_eq!(
            ws / 100.0,
            reduction_ratio_ring_fixed(20, 50),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "au moins une dent")]
    fn zero_sun_teeth_panics() {
        willis_ratio(0, 50);
    }
}
