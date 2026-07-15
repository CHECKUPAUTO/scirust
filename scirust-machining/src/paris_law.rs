//! Propagation de fissure en **fatigue** — loi de **Paris** (régime linéaire
//! du diagramme log-log da/dN vs ΔK).
//!
//! ```text
//! amplitude d'intensité   ΔK = Y·Δσ·√(π·a)
//! vitesse de fissuration  da/dN = C·ΔK^m
//! ΔK à partir de da/dN     ΔK = (da/dN / C)^(1/m)
//! ```
//!
//! `ΔK` amplitude du facteur d'intensité de contrainte sur un cycle (Pa·√m),
//! `Y` facteur de forme géométrique (sans dimension, ~1,12 pour une fissure de
//! bord), `Δσ` amplitude de contrainte du cycle (Pa), `a` longueur de fissure
//! (m), `da/dN` vitesse de propagation par cycle (m/cycle), `C` coefficient de
//! Paris (unités SI : (m/cycle)/(Pa·√m)^m), `m` exposant de Paris (sans
//! dimension).
//!
//! **Convention** : SI cohérent (`ΔK` en Pa·√m, `da/dN` en m/cycle).
//! **Limite honnête** : la loi ne vaut que dans le **régime de Paris** (zone II,
//! linéaire en log-log), loin du seuil ΔK_th et de la ténacité K_Ic ; les
//! constantes matériau `C` et `m` sont **fournies par l'appelant** (issues
//! d'essais), jamais de valeur « par défaut » inventée. Complète
//! [`crate::fracture`] (ténacité, rupture brutale) et scirust-fatigue
//! (comptage de cycles).

use core::f64::consts::PI;

/// Amplitude du facteur d'intensité de contrainte `ΔK = Y·Δσ·√(π·a)` (Pa·√m).
///
/// Panique si `crack_length < 0`.
pub fn paris_stress_intensity_range(
    geometry_factor: f64,
    stress_range: f64,
    crack_length: f64,
) -> f64 {
    assert!(
        crack_length >= 0.0,
        "la longueur de fissure doit être positive"
    );
    geometry_factor * stress_range * (PI * crack_length).sqrt()
}

/// Vitesse de propagation de fissure `da/dN = C·ΔK^m` (m/cycle).
///
/// Panique si `paris_c <= 0` ou `delta_k < 0`.
pub fn paris_crack_growth_rate(paris_c: f64, paris_m: f64, delta_k: f64) -> f64 {
    assert!(paris_c > 0.0, "le coefficient de Paris C doit être positif");
    assert!(
        delta_k >= 0.0,
        "l'amplitude d'intensité ΔK doit être positive"
    );
    paris_c * delta_k.powf(paris_m)
}

/// Amplitude d'intensité `ΔK = (da/dN / C)^(1/m)` reconstituée depuis la
/// vitesse de fissuration (Pa·√m).
///
/// Panique si `paris_c <= 0`, `paris_m == 0` ou `growth_rate < 0`.
pub fn paris_delta_k_from_rate(paris_c: f64, paris_m: f64, growth_rate: f64) -> f64 {
    assert!(paris_c > 0.0, "le coefficient de Paris C doit être positif");
    assert!(paris_m != 0.0, "l'exposant de Paris m doit être non nul");
    assert!(
        growth_rate >= 0.0,
        "la vitesse de fissuration doit être positive"
    );
    (growth_rate / paris_c).powf(1.0_f64 / paris_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn intensity_range_scales_with_sqrt_crack() {
        // ΔK ∝ √a : quadrupler la fissure double ΔK.
        let k1 = paris_stress_intensity_range(1.0, 100e6, 0.001);
        let k2 = paris_stress_intensity_range(1.0, 100e6, 0.004);
        assert_relative_eq!(k2 / k1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn intensity_range_linear_in_stress() {
        // ΔK ∝ Δσ à géométrie et fissure fixées.
        let k1 = paris_stress_intensity_range(1.12, 50e6, 0.002);
        let k2 = paris_stress_intensity_range(1.12, 150e6, 0.002);
        assert_relative_eq!(k2 / k1, 3.0, epsilon = 1e-9);
    }

    #[test]
    fn intensity_range_numeric_value() {
        // Y=1,12 ; Δσ=100 MPa ; a=1 mm.
        // ΔK = 1,12·100e6·√(π·0,001) = 1,12e8·0,0560499121...
        //    = 6 277 590,16... Pa·√m.
        let dk = paris_stress_intensity_range(1.12, 100e6, 0.001);
        let expected = 1.12_f64 * 100e6 * (PI * 0.001_f64).sqrt();
        assert_relative_eq!(dk, expected, max_relative = 1e-12);
        assert_relative_eq!(dk, 6_277_590.16, max_relative = 1e-6);
    }

    #[test]
    fn growth_rate_power_law_in_delta_k() {
        // da/dN = C·ΔK^m : doubler ΔK multiplie la vitesse par 2^m.
        let (c, m) = (1e-11, 3.0);
        let r1 = paris_crack_growth_rate(c, m, 5e6);
        let r2 = paris_crack_growth_rate(c, m, 10e6);
        assert_relative_eq!(r2 / r1, 2.0_f64.powf(m), max_relative = 1e-12);
    }

    #[test]
    fn delta_k_round_trips_through_rate() {
        // Réciprocité : ΔK → da/dN → ΔK doit redonner ΔK.
        let (c, m, dk) = (1e-11_f64, 3.2_f64, 8.5e6_f64);
        let rate = paris_crack_growth_rate(c, m, dk);
        let recovered = paris_delta_k_from_rate(c, m, rate);
        assert_relative_eq!(recovered, dk, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "coefficient de Paris C")]
    fn zero_coefficient_growth_rate_panics() {
        paris_crack_growth_rate(0.0, 3.0, 5e6);
    }
}
