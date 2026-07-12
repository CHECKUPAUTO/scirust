//! Mécanisme à **croix de Malte** (Geneva) externe — indexeur transformant la
//! rotation continue d'une manivelle en rotation intermittente d'une roue à `n`
//! rainures.
//!
//! ```text
//! rapport manivelle/entraxe  a/c = sin(π/n)      m = c/a = 1/sin(π/n)
//! angle de la roue menée     β = atan2(sinα, m − cosα)
//! rapport de vitesses        ω2/ω1 = (m·cosα − 1)/(1 − 2m·cosα + m²)
//! angle d'indexage (mvt)     Δα = π − 2π/n
//! angle de repos (dwell)     π + 2π/n
//! ```
//!
//! `n` nombre de rainures (`n ≥ 3`), `a` rayon de manivelle, `c` entraxe, `α`
//! angle de manivelle compté depuis la ligne des centres (`α = 0` au milieu de
//! l'indexage). La roue avance de `2π/n` par tour de manivelle ; la vitesse est
//! maximale au passage central (`α = 0`).
//!
//! **Convention** : angles en rad. **Limite honnête** : croix de Malte
//! **externe** idéale, entrée du pion sans choc (condition de tangence
//! `a/c = sin(π/n)`) ; ne modélise pas l'accélération d'entrée réelle, le jeu,
//! ni la variante interne.

use core::f64::consts::PI;

fn check(n: u32) {
    assert!(n >= 3, "une croix de Malte possède au moins 3 rainures");
}

/// Rapport rayon de manivelle / entraxe `a/c = sin(π/n)` (condition de tangence).
pub fn crank_ratio(n: u32) -> f64 {
    check(n);
    (PI / n as f64).sin()
}

/// Rapport entraxe / rayon de manivelle `m = c/a = 1/sin(π/n)`.
pub fn center_distance_ratio(n: u32) -> f64 {
    check(n);
    1.0 / (PI / n as f64).sin()
}

/// Angle de la roue menée `β = atan2(sinα, m − cosα)` (rad).
pub fn driven_angle(n: u32, alpha_rad: f64) -> f64 {
    let m = center_distance_ratio(n);
    alpha_rad.sin().atan2(m - alpha_rad.cos())
}

/// Rapport de vitesses instantané `ω2/ω1 = (m·cosα − 1)/(1 − 2m·cosα + m²)`.
pub fn velocity_ratio(n: u32, alpha_rad: f64) -> f64 {
    let m = center_distance_ratio(n);
    let c = alpha_rad.cos();
    (m * c - 1.0) / (1.0 - 2.0 * m * c + m * m)
}

/// Angle de manivelle pendant l'**indexage** (roue en mouvement) `Δα = π − 2π/n`.
pub fn indexing_crank_angle(n: u32) -> f64 {
    check(n);
    PI - 2.0 * PI / n as f64
}

/// Angle de manivelle pendant le **repos** (roue immobile) `π + 2π/n`.
pub fn dwell_crank_angle(n: u32) -> f64 {
    check(n);
    PI + 2.0 * PI / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tangency_ratio_for_four_slots() {
        // n=4 : a/c = sin45° = √2/2 ; c/a = √2.
        assert_relative_eq!(crank_ratio(4), 2.0f64.sqrt() / 2.0, epsilon = 1e-12);
        assert_relative_eq!(center_distance_ratio(4), 2.0f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn indexing_and_dwell_fill_a_full_turn() {
        // n=4 : indexage 90°, repos 270°, somme = 360°.
        assert_relative_eq!(indexing_crank_angle(4), PI / 2.0, epsilon = 1e-12);
        assert_relative_eq!(dwell_crank_angle(4), 3.0 * PI / 2.0, epsilon = 1e-12);
        assert_relative_eq!(
            indexing_crank_angle(4) + dwell_crank_angle(4),
            2.0 * PI,
            epsilon = 1e-12
        );
    }

    #[test]
    fn driven_wheel_indexes_by_two_pi_over_n() {
        // À l'entrée (α = α1 = π/2 − π/n) la roue est à β = π/n (demi-index).
        for &n in &[3u32, 4, 6, 8]
        {
            let alpha1 = PI / 2.0 - PI / n as f64;
            assert_relative_eq!(driven_angle(n, alpha1), PI / n as f64, epsilon = 1e-9);
            // au centre, β = 0.
            assert_relative_eq!(driven_angle(n, 0.0), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn peak_velocity_ratio_at_center() {
        // À α=0 : ω2/ω1 = 1/(m−1). Pour n=4, m=√2 → √2+1 ≈ 2,414.
        let m = center_distance_ratio(4);
        assert_relative_eq!(velocity_ratio(4, 0.0), 1.0 / (m - 1.0), epsilon = 1e-9);
        assert_relative_eq!(velocity_ratio(4, 0.0), 2.0f64.sqrt() + 1.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins 3 rainures")]
    fn too_few_slots_panics() {
        crank_ratio(2);
    }
}
