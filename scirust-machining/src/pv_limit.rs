//! **Facteur PV des paliers lisses secs/autolubrifiants** — pression projetée,
//! vitesse de glissement, produit PV et vitesse admissible pour une pression donnée.
//!
//! ```text
//! pression projetée      P = F/A_proj      (A_proj = d·L, surface diamétrale)
//! vitesse de glissement  V = π·d·n/60      (n régime en tr/min)
//! facteur PV             PV = P·V
//! vitesse admissible     V_max = PV_lim/P  (pour une pression P imposée)
//! ```
//!
//! `F` charge radiale (N), `A_proj` surface projetée du coussinet `A_proj = d·L`
//! (m²), `d` diamètre de l'arbre (m), `L` longueur portante (m), `P` pression
//! projetée (Pa), `n` régime de rotation (tr/min), `V` vitesse de glissement
//! périphérique (m/s), `PV` facteur PV (Pa·m/s ≡ W/m²), `PV_lim` facteur PV
//! admissible du matériau de coussinet (Pa·m/s), `V_max` vitesse de glissement
//! admissible sous la pression `P` (m/s).
//!
//! **Convention.** SI cohérent : longueurs en m, forces en N, pressions en Pa,
//! vitesses en m/s, régimes en tr/min.
//!
//! **Limite honnête.** Le facteur PV **admissible** du matériau de coussinet
//! (`PV_lim`, limite thermique/usure du fabricant) est **fourni par l'appelant** ;
//! la pression est **projetée** (surface diamétrale `d·L`, pas la surface réelle
//! de contact) et le régime de frottement est supposé **sec ou limite** (coussinets
//! secs/autolubrifiants). Ce module ne modélise **pas** l'échauffement transitoire
//! ni la montée en température : aucune constante matériau/procédé n'est inventée.

use core::f64::consts::PI;

/// Pression projetée `P = F/A_proj`, avec `A_proj = d·L` la surface diamétrale
/// (projection du coussinet sur un plan perpendiculaire à la charge).
///
/// Panique si `load < 0` ou `projected_area <= 0`.
pub fn pv_bearing_pressure(load: f64, projected_area: f64) -> f64 {
    assert!(load >= 0.0, "la charge F doit être positive");
    assert!(
        projected_area > 0.0,
        "la surface projetée A_proj doit être strictement positive"
    );
    load / projected_area
}

/// Vitesse de glissement périphérique `V = π·d·n/60`, avec `d` diamètre de l'arbre
/// (m) et `n` régime de rotation (tr/min).
///
/// Panique si `shaft_diameter < 0` ou `rotational_speed_rpm < 0`.
pub fn pv_sliding_velocity(shaft_diameter: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(
        shaft_diameter >= 0.0,
        "le diamètre de l'arbre d doit être positif"
    );
    assert!(rotational_speed_rpm >= 0.0, "le régime n doit être positif");
    PI * shaft_diameter * rotational_speed_rpm / 60.0
}

/// Facteur PV `PV = P·V` (produit pression projetée × vitesse de glissement).
///
/// Panique si `pressure < 0` ou `sliding_velocity < 0`.
pub fn pv_factor(pressure: f64, sliding_velocity: f64) -> f64 {
    assert!(pressure >= 0.0, "la pression P doit être positive");
    assert!(
        sliding_velocity >= 0.0,
        "la vitesse de glissement V doit être positive"
    );
    pressure * sliding_velocity
}

/// Vitesse de glissement admissible sous une pression imposée
/// `V_max = PV_lim/P`, où `PV_lim` est le facteur PV admissible du matériau.
///
/// Panique si `pv_limit_value < 0` ou `pressure <= 0`.
pub fn pv_max_speed_for_load(pv_limit_value: f64, pressure: f64) -> f64 {
    assert!(
        pv_limit_value >= 0.0,
        "le facteur PV admissible PV_lim doit être positif"
    );
    assert!(
        pressure > 0.0,
        "la pression P doit être strictement positive"
    );
    pv_limit_value / pressure
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pressure_realistic_case() {
        // F=1200 N sur A_proj = d·L = 0,02·0,03 = 6e-4 m² → P = 1200/6e-4 = 2e6 Pa.
        let p = pv_bearing_pressure(1200.0, 6.0e-4);
        assert_relative_eq!(p, 2.0e6, epsilon = 1e-6);
    }

    #[test]
    fn pressure_scales_linearly_with_load() {
        // P ∝ F : doubler la charge double la pression projetée.
        let a = pv_bearing_pressure(800.0, 5.0e-4);
        let b = pv_bearing_pressure(1600.0, 5.0e-4);
        assert_relative_eq!(b, 2.0 * a, epsilon = 1e-9);
    }

    #[test]
    fn sliding_velocity_realistic_case() {
        // d=0,05 m ; n=600 tr/min → V = π·0,05·600/60 = π·0,5 = π/2 ≈ 1,5708 m/s.
        let v = pv_sliding_velocity(0.05, 600.0);
        assert_relative_eq!(v, PI * 0.5, epsilon = 1e-12);
    }

    #[test]
    fn sliding_velocity_scales_linearly_with_speed() {
        // V ∝ n : doubler le régime double la vitesse de glissement.
        let low = pv_sliding_velocity(0.04, 500.0);
        let high = pv_sliding_velocity(0.04, 1000.0);
        assert_relative_eq!(high, 2.0 * low, epsilon = 1e-12);
    }

    #[test]
    fn pv_factor_realistic_case() {
        // P=2e6 Pa, V=1,5 m/s → PV = 3e6 Pa·m/s.
        let pv = pv_factor(2.0e6, 1.5);
        assert_relative_eq!(pv, 3.0e6, epsilon = 1e-6);
    }

    #[test]
    fn max_speed_is_inverse_of_pv_factor() {
        // Réciprocité : V_max(PV(P,V), P) = P·V/P = V pour tout P > 0.
        let pressure = 2.0e6_f64;
        let velocity = 1.5_f64;
        let pv = pv_factor(pressure, velocity);
        let recovered = pv_max_speed_for_load(pv, pressure);
        assert_relative_eq!(recovered, velocity, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "A_proj doit être strictement positive")]
    fn zero_area_panics() {
        pv_bearing_pressure(1200.0, 0.0);
    }
}
