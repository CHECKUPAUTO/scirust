//! **Rendement d'un engrenage cylindrique** — pertes par frottement de glissement
//! moyennées sur l'engrènement, puissance perdue et transmise.
//!
//! ```text
//! facteur de glissement   S = (H_a² + H_r²)/(H_a + H_r)   (arcs d'approche/retraite)
//! fraction de perte       L = π·f·(1/z1 + 1/z2)·S / cos φ
//! rendement               η = 1 − L
//! puissance perdue        P_perte = P·(1 − η)
//! puissance transmise     P_out   = P·η
//! ```
//!
//! `f` coefficient de frottement de glissement des flancs (sans dimension), `z1`
//! nombre de dents du pignon, `z2` nombre de dents de la roue, `φ` angle de
//! pression de fonctionnement (rad), `S` facteur de glissement moyen (sans
//! dimension, issu des arcs d'approche `H_a` et de retraite `H_r`), `L` fraction de
//! perte (sans dimension), `η` rendement (sans dimension), `P` puissance d'entrée
//! (W), `P_perte`/`P_out` puissances (W).
//!
//! **Convention** : SI ; angle de pression en radians ; nombres de dents et
//! facteur de glissement sans dimension. **Limite honnête** : modèle de perte par
//! glissement **moyennée** sur l'engrènement (approximation de type Buckingham,
//! denture développante à profils standard) ; le coefficient de frottement `f`, le
//! facteur de glissement `S` et l'angle de pression sont **fournis par l'appelant**
//! (issus d'essais, de la lubrification et de la géométrie réelle) — aucune valeur
//! « par défaut » n'est inventée. Distinct de [`crate::gears`] (géométrie) et de la
//! norme de résistance [`crate::iso6336`].

use core::f64::consts::PI;

/// Facteur de glissement moyen `S = (H_a² + H_r²)/(H_a + H_r)` construit à partir
/// des longueurs (ou fractions) des arcs d'approche `H_a` et de retraite `H_r`.
///
/// Panique si `approach_arc < 0`, `recess_arc < 0` ou `H_a + H_r == 0`.
pub fn mesh_sliding_factor_from_arcs(approach_arc: f64, recess_arc: f64) -> f64 {
    assert!(
        approach_arc >= 0.0 && recess_arc >= 0.0,
        "H_a ≥ 0 et H_r ≥ 0 requis"
    );
    let sum = approach_arc + recess_arc;
    assert!(sum > 0.0, "H_a + H_r > 0 requis");
    (approach_arc * approach_arc + recess_arc * recess_arc) / sum
}

/// Fraction de perte par glissement `L = π·f·(1/z1 + 1/z2)·S / cos φ`.
///
/// Panique si `friction_coefficient < 0`, `pinion_teeth <= 0`, `gear_teeth <= 0`,
/// `sliding_factor < 0` ou si `pressure_angle_rad` n'est pas dans `]0, π/2[`.
pub fn mesh_sliding_loss_fraction(
    friction_coefficient: f64,
    pinion_teeth: f64,
    gear_teeth: f64,
    pressure_angle_rad: f64,
    sliding_factor: f64,
) -> f64 {
    assert!(friction_coefficient >= 0.0, "f ≥ 0 requis");
    assert!(
        pinion_teeth > 0.0 && gear_teeth > 0.0,
        "z1 > 0 et z2 > 0 requis"
    );
    assert!(sliding_factor >= 0.0, "S ≥ 0 requis");
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "0 < φ < π/2 requis"
    );
    PI * friction_coefficient * (1.0 / pinion_teeth + 1.0 / gear_teeth) * sliding_factor
        / pressure_angle_rad.cos()
}

/// Rendement d'un engrenage cylindrique droit
/// `η = 1 − π·f·(1/z1 + 1/z2)·S / cos φ`.
///
/// Panique si `friction_coefficient < 0`, `pinion_teeth <= 0`, `gear_teeth <= 0`,
/// `sliding_factor < 0` ou si `pressure_angle_rad` n'est pas dans `]0, π/2[`.
pub fn gear_eff_spur_efficiency(
    friction_coefficient: f64,
    pinion_teeth: f64,
    gear_teeth: f64,
    pressure_angle_rad: f64,
    sliding_factor: f64,
) -> f64 {
    1.0 - mesh_sliding_loss_fraction(
        friction_coefficient,
        pinion_teeth,
        gear_teeth,
        pressure_angle_rad,
        sliding_factor,
    )
}

/// Puissance perdue dans l'engrènement `P_perte = P·(1 − η)`.
///
/// Panique si `input_power < 0` ou si `efficiency` n'est pas dans `[0, 1]`.
pub fn gear_eff_power_loss(input_power: f64, efficiency: f64) -> f64 {
    assert!(input_power >= 0.0, "P ≥ 0 requis");
    assert!((0.0..=1.0).contains(&efficiency), "η ∈ [0, 1] requis");
    input_power * (1.0 - efficiency)
}

/// Puissance transmise en sortie `P_out = P·η`.
///
/// Panique si `input_power < 0` ou si `efficiency` n'est pas dans `[0, 1]`.
pub fn gear_eff_output_power(input_power: f64, efficiency: f64) -> f64 {
    assert!(input_power >= 0.0, "P ≥ 0 requis");
    assert!((0.0..=1.0).contains(&efficiency), "η ∈ [0, 1] requis");
    input_power * efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn frictionless_mesh_is_perfect() {
        // f = 0 : aucune perte de glissement → η = 1, L = 0.
        assert_relative_eq!(
            mesh_sliding_loss_fraction(0.0, 20.0, 40.0, 20.0_f64.to_radians(), 1.0),
            0.0,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            gear_eff_spur_efficiency(0.0, 20.0, 40.0, 20.0_f64.to_radians(), 1.0),
            1.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn loss_symmetric_in_teeth() {
        // 1/z1 + 1/z2 est symétrique : échanger pignon et roue ne change rien.
        let a = mesh_sliding_loss_fraction(0.05, 20.0, 40.0, 20.0_f64.to_radians(), 1.2);
        let b = mesh_sliding_loss_fraction(0.05, 40.0, 20.0, 20.0_f64.to_radians(), 1.2);
        assert_relative_eq!(a, b, epsilon = 1e-15);
    }

    #[test]
    fn efficiency_complements_loss() {
        // η = 1 − L par construction.
        let l = mesh_sliding_loss_fraction(0.06, 18.0, 54.0, 25.0_f64.to_radians(), 1.1);
        let eta = gear_eff_spur_efficiency(0.06, 18.0, 54.0, 25.0_f64.to_radians(), 1.1);
        assert_relative_eq!(eta + l, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn power_split_conserves_input() {
        // P_out + P_perte = P (conservation de la puissance).
        let p = 5_000.0;
        let eta = 0.985;
        assert_relative_eq!(
            gear_eff_output_power(p, eta) + gear_eff_power_loss(p, eta),
            p,
            epsilon = 1e-9
        );
    }

    #[test]
    fn realistic_spur_pair() {
        // f=0,05 ; z1=20 ; z2=40 ; φ=20° ; S=1 → L = π·0,05·0,075/cos20°.
        let expected_loss = PI * 0.05 * (1.0 / 20.0 + 1.0 / 40.0) / 20.0_f64.to_radians().cos();
        let eta = gear_eff_spur_efficiency(0.05, 20.0, 40.0, 20.0_f64.to_radians(), 1.0);
        assert_relative_eq!(eta, 1.0 - expected_loss, epsilon = 1e-12);
        // Rendement d'engrènement réaliste, supérieur à 98 %.
        assert!(eta > 0.98 && eta < 1.0);
    }

    #[test]
    fn sliding_factor_equal_arcs() {
        // H_a = H_r = h → S = (2h²)/(2h) = h.
        assert_relative_eq!(
            mesh_sliding_factor_from_arcs(0.7, 0.7),
            0.7,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "0 < φ < π/2")]
    fn right_angle_pressure_panics() {
        mesh_sliding_loss_fraction(0.05, 20.0, 40.0, PI / 2.0, 1.0);
    }
}
