//! Paliers lisses hydrodynamiques — charge unitaire, nombre de **Sommerfeld**,
//! frottement de **Petroff** et épaisseur minimale de film.
//!
//! ```text
//! charge unitaire        p = W/(L·d)
//! nombre de Sommerfeld   S = (r/c)²·µ·N/p
//! frottement (Petroff)   f = 2π²·(µ·N/p)·(r/c)      (palier peu chargé, centré)
//! couple de Petroff      C = 4π²·µ·N·r³·L/c
//! épaisseur mini de film h_min = c·(1 − ε)
//! ```
//!
//! `W` charge radiale (N), `L` longueur (m), `d`/`r` diamètre/rayon du tourillon
//! (m), `c` jeu radial (m), `µ` viscosité dynamique (Pa·s), `N` fréquence de
//! rotation (tr/s), `p` charge unitaire (Pa), `ε` excentricité relative (`0` centré,
//! `1` contact).
//!
//! **Convention** : SI cohérent, `N` en **tours par seconde**. **Limite
//! honnête** : Petroff suppose un palier **concentrique** peu chargé (borne du
//! frottement) ; le nombre de Sommerfeld caractérise le fonctionnement réel mais
//! son exploitation (abaques de Raimondi-Boyd) reste à la charge de l'appelant.
//! `µ` et le régime hydrodynamique établi sont fournis/supposés.

use core::f64::consts::PI;

/// Charge unitaire (pression projetée) `p = W/(L·d)` (Pa).
///
/// Panique si `length*diameter <= 0`.
pub fn unit_load(radial_load: f64, length: f64, diameter: f64) -> f64 {
    assert!(length * diameter > 0.0, "L·d doit être strictement positif");
    radial_load / (length * diameter)
}

/// Nombre de Sommerfeld `S = (r/c)²·µ·N/p` (sans dimension).
///
/// Panique si `clearance <= 0` ou `unit_load <= 0`.
pub fn sommerfeld_number(
    radius: f64,
    clearance: f64,
    viscosity: f64,
    speed_rev_s: f64,
    unit_load: f64,
) -> f64 {
    assert!(clearance > 0.0 && unit_load > 0.0, "c > 0 et p > 0 requis");
    let ratio = radius / clearance;
    ratio * ratio * viscosity * speed_rev_s / unit_load
}

/// Coefficient de frottement de Petroff `f = 2π²·(µ·N/p)·(r/c)`.
///
/// Panique si `clearance <= 0` ou `unit_load <= 0`.
pub fn petroff_friction_coefficient(
    viscosity: f64,
    speed_rev_s: f64,
    unit_load: f64,
    radius: f64,
    clearance: f64,
) -> f64 {
    assert!(clearance > 0.0 && unit_load > 0.0, "c > 0 et p > 0 requis");
    2.0 * PI * PI * (viscosity * speed_rev_s / unit_load) * (radius / clearance)
}

/// Couple de frottement de Petroff `C = 4π²·µ·N·r³·L/c` (N·m).
///
/// Panique si `clearance <= 0`.
pub fn petroff_torque(
    viscosity: f64,
    speed_rev_s: f64,
    radius: f64,
    length: f64,
    clearance: f64,
) -> f64 {
    assert!(
        clearance > 0.0,
        "le jeu radial doit être strictement positif"
    );
    4.0 * PI * PI * viscosity * speed_rev_s * radius.powi(3) * length / clearance
}

/// Épaisseur minimale du film `h_min = c·(1 − ε)` (m).
///
/// Panique si `eccentricity_ratio` sort de `[0, 1)`.
pub fn minimum_film_thickness(clearance: f64, eccentricity_ratio: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&eccentricity_ratio),
        "l'excentricité relative doit être dans [0, 1)"
    );
    clearance * (1.0 - eccentricity_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_load_is_projected_pressure() {
        // W=5000 N, L=50 mm, d=50 mm → p = 5000/(0,05·0,05) = 2 MPa.
        assert_relative_eq!(unit_load(5000.0, 0.05, 0.05), 2e6, epsilon = 1.0);
    }

    #[test]
    fn petroff_torque_and_friction_are_consistent() {
        // Le couple de Petroff doit valoir f·W·r avec W = p·L·d.
        let (visc, n, r, l, c) = (0.02, 30.0, 0.025, 0.050, 25e-6);
        let p = unit_load(5000.0, l, 2.0 * r);
        let f = petroff_friction_coefficient(visc, n, p, r, c);
        let torque = petroff_torque(visc, n, r, l, c);
        // C = f·W·r (W = p·L·d = p·L·2r).
        let w = p * l * 2.0 * r;
        assert_relative_eq!(torque, f * w * r, max_relative = 1e-9);
    }

    #[test]
    fn sommerfeld_scales_with_clearance_ratio_squared() {
        // Halver le jeu quadruple S.
        let s1 = sommerfeld_number(0.025, 25e-6, 0.02, 30.0, 2e6);
        let s2 = sommerfeld_number(0.025, 12.5e-6, 0.02, 30.0, 2e6);
        assert_relative_eq!(s2 / s1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn film_thickness_vanishes_as_eccentricity_grows() {
        // ε=0 : h=c ; ε=0,6 : h=0,4c.
        assert_relative_eq!(minimum_film_thickness(25e-6, 0.0), 25e-6, epsilon = 1e-12);
        assert_relative_eq!(minimum_film_thickness(25e-6, 0.6), 10e-6, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "excentricité")]
    fn full_eccentricity_panics() {
        minimum_film_thickness(25e-6, 1.0);
    }
}
