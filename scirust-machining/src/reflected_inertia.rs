//! **Inertie ramenée** à l'arbre moteur — la donnée qui conditionne le
//! dimensionnement d'un moteur d'axe et son **ratio d'inertie**.
//!
//! ```text
//! réducteur      J_ram = J_charge / i²                 (i = ω_moteur/ω_charge)
//! masse linéaire J_ram = m·(p/2π)²                     (vis à billes, pas p)
//! vis pleine     J = (π/32)·ρ·D⁴·L                     (cylindre plein)
//! ratio          R = J_charge_ramenée / J_moteur
//! ```
//!
//! `J` moment d'inertie (kg·m²), `i` rapport de réduction (`> 1` en réduction),
//! `m` masse en translation (kg), `p` pas de la vis (m), `ρ` masse volumique
//! (kg/m³), `D` diamètre, `L` longueur (m), `R` ratio d'inertie
//! (charge ramenée / moteur).
//!
//! **Convention** : SI. **Limite honnête** : la réduction est supposée **sans
//! jeu** et de rendement unitaire pour la réflexion d'inertie (l'inertie propre
//! du réducteur/vis s'ajoute séparément). Un ratio `R` de l'ordre de **1 à 5**
//! est visé en pratique (10 maxi) pour un asservissement stable ; le seuil est
//! une donnée de conception fournie par l'appelant. Voir [`crate::ball_screw`] et
//! [`crate::motor_torque`].

use core::f64::consts::PI;

/// Inertie d'une charge ramenée à travers un réducteur `J_ram = J_charge/i²`.
///
/// Panique si `gear_ratio <= 0`.
pub fn inertia_through_gear(load_inertia: f64, gear_ratio: f64) -> f64 {
    assert!(
        gear_ratio > 0.0,
        "le rapport de réduction doit être strictement positif"
    );
    load_inertia / (gear_ratio * gear_ratio)
}

/// Inertie d'une masse en translation ramenée à la vis `J = m·(p/2π)²`.
///
/// Panique si `mass < 0` ou `lead <= 0`.
pub fn ballscrew_load_inertia(mass: f64, lead: f64) -> f64 {
    assert!(mass >= 0.0 && lead > 0.0, "masse ≥ 0 et pas > 0 requis");
    let r = lead / (2.0 * PI);
    mass * r * r
}

/// Inertie propre d'une vis pleine (cylindre) `J = (π/32)·ρ·D⁴·L`.
///
/// Panique si un paramètre `<= 0`.
pub fn screw_inertia_solid(density: f64, diameter: f64, length: f64) -> f64 {
    assert!(
        density > 0.0 && diameter > 0.0 && length > 0.0,
        "ρ, D et L strictement positifs requis"
    );
    PI / 32.0 * density * diameter.powi(4) * length
}

/// Ratio d'inertie `R = J_charge_ramenée / J_moteur`.
///
/// Panique si `motor_inertia <= 0`.
pub fn inertia_ratio(reflected_load_inertia: f64, motor_inertia: f64) -> f64 {
    assert!(
        motor_inertia > 0.0,
        "l'inertie moteur doit être strictement positive"
    );
    reflected_load_inertia / motor_inertia
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reduction_shrinks_inertia_by_square() {
        // Réduction 10:1 → inertie ramenée divisée par 100.
        assert_relative_eq!(inertia_through_gear(2.0, 10.0), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn ballscrew_inertia_scales_with_lead_squared() {
        // Doubler le pas quadruple l'inertie ramenée de la masse.
        let j1 = ballscrew_load_inertia(200.0, 0.010);
        let j2 = ballscrew_load_inertia(200.0, 0.020);
        assert_relative_eq!(j2 / j1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn ballscrew_inertia_matches_formula() {
        // m=200 kg, p=10 mm → J = 200·(0,01/2π)² ≈ 5,07e-4 kg·m².
        let j = ballscrew_load_inertia(200.0, 0.010);
        assert_relative_eq!(j, 200.0 * (0.010_f64 / (2.0 * PI)).powi(2), epsilon = 1e-15);
        assert!(j > 5.0e-4 && j < 5.1e-4);
    }

    #[test]
    fn solid_screw_inertia_scales_with_diameter_fourth() {
        // ×2 sur le diamètre → ×16 sur l'inertie.
        let j1 = screw_inertia_solid(7850.0, 0.020, 1.0);
        let j2 = screw_inertia_solid(7850.0, 0.040, 1.0);
        assert_relative_eq!(j2 / j1, 16.0, epsilon = 1e-9);
    }

    #[test]
    fn inertia_ratio_is_dimensionless_quotient() {
        assert_relative_eq!(inertia_ratio(3.0e-4, 1.0e-4), 3.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "inertie moteur")]
    fn zero_motor_inertia_panics() {
        inertia_ratio(1.0e-4, 0.0);
    }
}
