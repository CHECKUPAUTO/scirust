//! Ressort **ondulé** plat multi-tours (*wave spring*, type Smalley) — raideur,
//! flèche et contrainte de flexion d'un empilage ondulé « crête contre crête ».
//!
//! ```text
//! raideur      k = E·b·t³·N⁴ / (C_r · D_m³ · Z)        (C_r = 2.4)
//! flèche       δ = F / k
//! effort       F = k · δ
//! contrainte   σ = 3·π·F·D_m / (4·b·t²·N²)
//! ```
//!
//! `E` module de Young (Pa), `b` largeur radiale de la bande (m), `t` épaisseur
//! de la bande (m), `N` nombre d'ondes (crêtes) par tour (sans dimension), `D_m`
//! diamètre moyen du ressort (m), `Z` nombre de tours actifs (sans dimension),
//! `C_r = 2.4` constante empirique de raideur (sans dimension), `k` raideur
//! (N/m), `δ` flèche (m), `F` effort axial (N), `σ` contrainte de flexion (Pa).
//!
//! **Convention** : SI cohérent (m, N, Pa, N/m). **Limite honnête** : ressort
//! ondulé plat, multi-tours, en régime **élastique** et **petites déflexions**
//! (loi linéaire de Smalley) ; la constante `C_r = 2.4` et la géométrie sont
//! celles de cette forme classique. Le module de Young, les dimensions et le
//! nombre d'ondes/tours sont **fournis par l'appelant** : ce module ne compose
//! que ces primitives et n'invente aucune constante de matériau ni de procédé.

use core::f64::consts::PI;

/// Constante empirique de raideur `C_r` de la loi de Smalley (sans dimension).
pub const WAVE_SPRING_RATE_CONSTANT: f64 = 2.4;

/// Raideur d'un ressort ondulé plat multi-tours (loi de Smalley) :
/// `k = E·b·t³·N⁴ / (C_r · D_m³ · Z)` (N/m).
///
/// `number_of_waves` = ondes (crêtes) par tour, `turns` = tours actifs.
///
/// Panique si l'un des arguments est négatif, si `mean_diameter <= 0`,
/// si `number_of_waves <= 0` ou si `turns <= 0`.
pub fn wave_spring_rate(
    youngs_modulus: f64,
    mean_diameter: f64,
    radial_width: f64,
    thickness: f64,
    number_of_waves: f64,
    turns: f64,
) -> f64 {
    assert!(youngs_modulus >= 0.0, "youngs_modulus doit être ≥ 0");
    assert!(radial_width >= 0.0, "radial_width doit être ≥ 0");
    assert!(thickness >= 0.0, "thickness doit être ≥ 0");
    assert!(mean_diameter > 0.0, "mean_diameter doit être > 0");
    assert!(number_of_waves > 0.0, "number_of_waves doit être > 0");
    assert!(turns > 0.0, "turns doit être > 0");
    let numerator = youngs_modulus * radial_width * thickness.powi(3) * number_of_waves.powi(4);
    let denominator = WAVE_SPRING_RATE_CONSTANT * mean_diameter.powi(3) * turns;
    numerator / denominator
}

/// Flèche axiale d'un ressort ondulé sous un effort donné : `δ = F / k` (m).
///
/// Panique si `rate <= 0` ou si `force < 0`.
pub fn wave_spring_deflection(force: f64, rate: f64) -> f64 {
    assert!(force >= 0.0, "force doit être ≥ 0");
    assert!(rate > 0.0, "rate doit être > 0");
    force / rate
}

/// Effort axial d'un ressort ondulé pour une flèche donnée : `F = k · δ` (N).
///
/// Réciproque de [`wave_spring_deflection`].
///
/// Panique si `rate < 0` ou si `deflection < 0`.
pub fn wave_spring_load(rate: f64, deflection: f64) -> f64 {
    assert!(rate >= 0.0, "rate doit être ≥ 0");
    assert!(deflection >= 0.0, "deflection doit être ≥ 0");
    rate * deflection
}

/// Contrainte de flexion d'un ressort ondulé plat (loi de Smalley) :
/// `σ = 3·π·F·D_m / (4·b·t²·N²)` (Pa).
///
/// `waves` = ondes (crêtes) par tour.
///
/// Panique si `force < 0`, si `mean_diameter < 0`, si `radial_width <= 0`,
/// si `thickness <= 0` ou si `waves <= 0`.
pub fn wave_spring_stress(
    force: f64,
    mean_diameter: f64,
    waves: f64,
    radial_width: f64,
    thickness: f64,
) -> f64 {
    assert!(force >= 0.0, "force doit être ≥ 0");
    assert!(mean_diameter >= 0.0, "mean_diameter doit être ≥ 0");
    assert!(radial_width > 0.0, "radial_width doit être > 0");
    assert!(thickness > 0.0, "thickness doit être > 0");
    assert!(waves > 0.0, "waves doit être > 0");
    3.0 * PI * force * mean_diameter / (4.0 * radial_width * thickness.powi(2) * waves.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_matches_hand_computed_case() {
        // Cas chiffré réaliste (acier) calculé à la main :
        // E = 200e9, D_m = 0.05, b = 0.005, t = 0.0003, N = 4, Z = 3.
        // num = 200e9·0.005·(3e-4)³·4⁴ = 1e9·2.7e-11·256 = 6.912
        // den = 2.4·(0.05)³·3 = 2.4·1.25e-4·3 = 9.0e-4
        // k = 6.912 / 9.0e-4 = 7680 N/m
        let k = wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 4.0, 3.0);
        assert_relative_eq!(k, 7680.0, max_relative = 1e-12);
    }

    #[test]
    fn deflection_and_load_are_reciprocal() {
        // δ = F/k puis F = k·δ doit redonner l'effort de départ.
        let k = wave_spring_rate(210.0e9, 0.04, 0.006, 0.00025, 3.0, 4.0);
        let force = 12.5_f64;
        let delta = wave_spring_deflection(force, k);
        assert_relative_eq!(wave_spring_load(k, delta), force, max_relative = 1e-12);
    }

    #[test]
    fn rate_scales_with_waves_to_the_fourth() {
        // k ∝ N⁴ : doubler le nombre d'ondes multiplie la raideur par 16.
        let base = wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 4.0, 3.0);
        let doubled = wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 8.0, 3.0);
        assert_relative_eq!(doubled, 16.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn rate_is_inversely_proportional_to_turns() {
        // k ∝ 1/Z : tripler les tours divise la raideur par 3.
        let base = wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 4.0, 2.0);
        let tripled_turns = wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 4.0, 6.0);
        assert_relative_eq!(tripled_turns, base / 3.0, max_relative = 1e-12);
    }

    #[test]
    fn stress_is_linear_in_force() {
        // σ ∝ F à géométrie fixée : doubler l'effort double la contrainte.
        let s1 = wave_spring_stress(10.0, 0.05, 4.0, 0.005, 0.0003);
        let s2 = wave_spring_stress(20.0, 0.05, 4.0, 0.005, 0.0003);
        assert_relative_eq!(s2, 2.0 * s1, max_relative = 1e-12);
    }

    #[test]
    fn stress_matches_hand_computed_case() {
        // σ = 3π·F·D_m / (4·b·t²·N²) avec F = 7.68, D_m = 0.05, N = 4,
        // b = 0.005, t = 0.0003.
        // num = 3π·7.68·0.05 = 1.152·π
        // den = 4·0.005·(3e-4)²·16 = 0.02·9e-8·16 = 2.88e-8
        let sigma = wave_spring_stress(7.68, 0.05, 4.0, 0.005, 0.0003);
        let expected = 1.152 * PI / 2.88e-8;
        assert_relative_eq!(sigma, expected, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "turns doit être > 0")]
    fn zero_turns_panics() {
        wave_spring_rate(200.0e9, 0.05, 0.005, 0.0003, 4.0, 0.0);
    }
}
