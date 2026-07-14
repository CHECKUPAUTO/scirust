//! Barre sinus (**sine bar**) — métrologie d'angle par conversion entre la
//! hauteur des cales sous un rouleau et l'angle d'inclinaison de la barre.
//!
//! ```text
//! hauteur de cale   h = L · sin(θ)        (montée du rouleau au-dessus du plan)
//! angle             θ = asin(h / L)       (réciproque de la précédente)
//! ```
//!
//! `L` entraxe des deux rouleaux de la barre (m), `h` hauteur des cales
//! empilées sous le rouleau haut (m), `θ` angle d'inclinaison de la barre
//! par rapport au marbre (rad). Les deux fonctions sont **réciproques** :
//! `sine_bar_angle_rad(L, sine_bar_gauge_height(L, θ)) = θ`.
//!
//! **Convention** : SI cohérent (longueurs en m, angles en rad) ; il suffit
//! que `L` et `h` partagent la même unité de longueur pour un ratio correct.
//! **Limite honnête** : la barre est supposée **parfaite** (entraxe `L` exact,
//! rouleaux de même diamètre, marbre plan) et le contact ponctuel. Comme
//! `sin` s'aplatit près de 90°, la **sensibilité chute au-delà de ~45–60°**
//! (une même incertitude sur `h` produit une erreur d'angle croissante), et la
//! barre sinus n'y est plus recommandée. Aucune valeur de `L` ni de tolérance
//! n'est imposée : l'entraxe nominal et les incertitudes sont **fournis par
//! l'appelant**.

use core::f64::consts::PI;

/// Hauteur de cale `h = L · sin(θ)` (m) à placer sous le rouleau haut pour
/// incliner de l'angle `angle_rad` une barre sinus d'entraxe `bar_length`.
///
/// Panique si `bar_length <= 0` ou si `angle_rad` sort de `[0, π/2]`.
pub fn sine_bar_gauge_height(bar_length: f64, angle_rad: f64) -> f64 {
    assert!(
        bar_length > 0.0,
        "l'entraxe L de la barre sinus doit être strictement positif"
    );
    assert!(
        (0.0..=PI / 2.0).contains(&angle_rad),
        "l'angle θ doit être compris dans [0, π/2] rad"
    );
    bar_length * angle_rad.sin()
}

/// Angle d'inclinaison `θ = asin(h / L)` (rad) d'une barre sinus d'entraxe
/// `bar_length` calée d'une hauteur `gauge_height`.
///
/// Panique si `bar_length <= 0`, si `gauge_height < 0` ou si
/// `gauge_height > bar_length` (rapport hors du domaine de `asin`).
pub fn sine_bar_angle_rad(bar_length: f64, gauge_height: f64) -> f64 {
    assert!(
        bar_length > 0.0,
        "l'entraxe L de la barre sinus doit être strictement positif"
    );
    assert!(
        gauge_height >= 0.0,
        "la hauteur de cale h ne peut être négative"
    );
    assert!(
        gauge_height <= bar_length,
        "la hauteur de cale h ne peut dépasser l'entraxe L (rapport hors de asin)"
    );
    (gauge_height / bar_length).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn height_at_thirty_degrees_is_half_length() {
        // sin(30°) = 1/2 : la cale vaut exactement la moitié de l'entraxe.
        let l = 0.100_f64;
        let h = sine_bar_gauge_height(l, PI / 6.0);
        assert_relative_eq!(h, l / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn flat_bar_needs_no_gauge() {
        // Barre à plat (θ = 0) → hauteur nulle, et réciproquement.
        assert_relative_eq!(sine_bar_gauge_height(0.250, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(sine_bar_angle_rad(0.250, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn vertical_bar_gauge_equals_length() {
        // À 90°, la montée du rouleau égale l'entraxe complet.
        let l = 0.200_f64;
        assert_relative_eq!(sine_bar_gauge_height(l, PI / 2.0), l, max_relative = 1e-12);
    }

    #[test]
    fn height_and_angle_are_reciprocal() {
        // Réciprocité : angle(height(θ)) = θ sur une plage utile.
        let l = 0.150_f64;
        for &theta in &[0.05_f64, 0.30, 0.6, 1.0, 1.2]
        {
            let h = sine_bar_gauge_height(l, theta);
            assert_relative_eq!(sine_bar_angle_rad(l, h), theta, max_relative = 1e-12);
        }
    }

    #[test]
    fn realistic_five_inch_bar_case() {
        // Barre 5" (0,127 m) inclinée à 20° : h = 0,127·sin(20°).
        let l = 0.127_f64;
        let theta = 20.0_f64.to_radians();
        let h = sine_bar_gauge_height(l, theta);
        assert_relative_eq!(h, l * theta.sin(), max_relative = 1e-12);
        // Contrôle chiffré indépendant : 0,127·0,342020… ≈ 0,043437 m.
        assert_relative_eq!(h, 0.043_436_6, max_relative = 1e-5);
    }

    #[test]
    fn low_angle_sensitivity_exceeds_high_angle() {
        // Identité physique : dh/dθ = L·cos(θ) décroît quand θ croît, donc une
        // même variation de hauteur correspond à un pas d'angle plus grand
        // (sensibilité dégradée) près de la verticale.
        let l = 0.100_f64;
        let dh = 1.0e-6_f64;
        // Pas d'angle induit par dh autour de 10° puis de 60°.
        let step_low = sine_bar_angle_rad(l, sine_bar_gauge_height(l, 10.0_f64.to_radians()) + dh)
            - 10.0_f64.to_radians();
        let step_high = sine_bar_angle_rad(l, sine_bar_gauge_height(l, 60.0_f64.to_radians()) + dh)
            - 60.0_f64.to_radians();
        assert!(step_high > step_low);
    }

    #[test]
    #[should_panic(expected = "ne peut dépasser l'entraxe L")]
    fn gauge_above_length_panics() {
        sine_bar_angle_rad(0.100, 0.150);
    }
}
