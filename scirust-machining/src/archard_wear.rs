//! Usure adhésive/abrasive — loi d'**Archard** : volume usé, profondeur et taux
//! d'usure spécifique.
//!
//! ```text
//! volume usé        V = k·F·s/H
//! taux spécifique   k_s = V/(F·s) = k/H
//! profondeur        h = V/A_app = k_s·p·s        (p = F/A_app pression apparente)
//! distance visée    s = h/(k_s·p)
//! ```
//!
//! `V` volume de matière enlevé (m³), `k` coefficient d'usure sans dimension, `F`
//! charge normale (N), `s` distance de glissement (m), `H` dureté du matériau le
//! plus tendre (Pa), `k_s` taux d'usure **spécifique** (m³·N⁻¹·m⁻¹ = Pa⁻¹), `A_app`
//! aire apparente de contact (m²), `p` pression apparente (Pa), `h` profondeur usée.
//!
//! **Convention** : SI ; dureté en Pa (1 HV ≈ 9,81 MPa). **Limite honnête** :
//! modèle d'**Archard** (usure proportionnelle au produit charge × distance), à
//! coefficient `k` **constant** — il regroupe toute la physique du couple de
//! frottement et provient d'essais fournis par l'appelant ; ne modélise ni le
//! rodage initial, ni les transitions de régime d'usure.

/// Volume usé `V = k·F·s/H`.
///
/// Panique si un paramètre `< 0` ou `hardness <= 0`.
pub fn worn_volume(wear_coefficient: f64, load: f64, sliding_distance: f64, hardness: f64) -> f64 {
    assert!(
        wear_coefficient >= 0.0 && load >= 0.0 && sliding_distance >= 0.0 && hardness > 0.0,
        "k, F, s ≥ 0 et H > 0 requis"
    );
    wear_coefficient * load * sliding_distance / hardness
}

/// Taux d'usure **spécifique** `k_s = k/H` (m³·N⁻¹·m⁻¹).
///
/// Panique si `wear_coefficient < 0` ou `hardness <= 0`.
pub fn specific_wear_rate(wear_coefficient: f64, hardness: f64) -> f64 {
    assert!(
        wear_coefficient >= 0.0 && hardness > 0.0,
        "k ≥ 0 et H > 0 requis"
    );
    wear_coefficient / hardness
}

/// Profondeur usée `h = V/A_app`.
///
/// Panique si `worn_volume < 0` ou `apparent_area <= 0`.
pub fn wear_depth(worn_volume: f64, apparent_area: f64) -> f64 {
    assert!(
        worn_volume >= 0.0 && apparent_area > 0.0,
        "V ≥ 0 et A > 0 requis"
    );
    worn_volume / apparent_area
}

/// Distance de glissement pour atteindre une profondeur `h`
/// `s = h/(k_s·p)`.
///
/// Panique si `depth < 0`, `specific_wear_rate <= 0` ou `apparent_pressure <= 0`.
pub fn sliding_distance_for_depth(
    depth: f64,
    specific_wear_rate: f64,
    apparent_pressure: f64,
) -> f64 {
    assert!(
        depth >= 0.0 && specific_wear_rate > 0.0 && apparent_pressure > 0.0,
        "h ≥ 0, k_s > 0 et p > 0 requis"
    );
    depth / (specific_wear_rate * apparent_pressure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn worn_volume_proportional_to_load_and_distance() {
        // V ∝ F·s : doubler la charge ou la distance double le volume.
        let v1 = worn_volume(1e-4, 100.0, 1000.0, 2e9);
        let v2 = worn_volume(1e-4, 200.0, 1000.0, 2e9);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn specific_rate_is_k_over_hardness() {
        assert_relative_eq!(
            specific_wear_rate(1e-4, 2e9),
            1e-4 / 2e9,
            max_relative = 1e-12
        );
    }

    #[test]
    fn depth_from_volume_and_area() {
        // V=1 mm³ sur 100 mm² → h = 1e-9/1e-4 = 10 µm.
        assert_relative_eq!(wear_depth(1e-9, 1e-4), 1e-5, epsilon = 1e-15);
    }

    #[test]
    fn distance_and_depth_are_consistent() {
        // h = k_s·p·s ⇒ s = h/(k_s·p) redonne la distance.
        let (ks, p, s) = (5e-14, 2e6, 1e4);
        let h = ks * p * s;
        assert_relative_eq!(sliding_distance_for_depth(h, ks, p), s, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "H > 0")]
    fn zero_hardness_panics() {
        worn_volume(1e-4, 100.0, 1000.0, 0.0);
    }
}
