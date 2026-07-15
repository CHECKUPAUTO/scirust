//! **Jeu de denture (backlash) d'un engrenage cylindrique droit** — jeu
//! circonférentiel dû à un accroissement d'entraxe ou à l'amincissement des dents,
//! jeu angulaire au pignon et entraxe requis pour un jeu visé.
//!
//! ```text
//! jeu circonférentiel (entraxe)   j_t = 2·ΔC·tan(α)
//! jeu angulaire (au pignon)       j_θ = j_t / r
//! jeu circonférentiel (amincis.)  j_t = 2·Δs
//! entraxe pour jeu visé           ΔC  = j_t / (2·tan(α))
//! ```
//!
//! `ΔC` accroissement d'entraxe par rapport à l'entraxe théorique (m), `α` angle de
//! pression de fonctionnement (rad), `j_t` jeu circonférentiel mesuré sur le cercle
//! primitif (m), `r` rayon primitif du pignon (m), `j_θ` jeu angulaire au pignon
//! (rad), `Δs` réduction d'épaisseur de dent appliquée à chacune des deux roues (m).
//!
//! **Convention** : SI ; longueurs en mètres, angles en radians (toute unité de
//! longueur cohérente convient puisque les relations sont homogènes). **Limite
//! honnête** : engrenage cylindrique droit à denture développante, denture supposée
//! rigide (aucune déformation élastique sous charge). L'angle de pression `α` est
//! **fourni par l'appelant** ; le jeu recommandé selon le module, la classe de
//! qualité ou la dilatation thermique est **fourni par l'appelant** — aucune valeur
//! « par défaut » n'est inventée ici. Distinct de [`crate::gears`] (géométrie) et de
//! [`crate::gear_efficiency`] (pertes).

/// Jeu circonférentiel créé par un accroissement d'entraxe
/// `j_t = 2·ΔC·tan(α)`.
///
/// Panique si `center_distance_change < 0` ou si `pressure_angle_rad` n'est pas dans
/// `]0, π/2[`.
pub fn gear_backlash_circular(center_distance_change: f64, pressure_angle_rad: f64) -> f64 {
    assert!(center_distance_change >= 0.0, "ΔC ≥ 0 requis");
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < core::f64::consts::FRAC_PI_2,
        "0 < α < π/2 requis"
    );
    2.0 * center_distance_change * pressure_angle_rad.tan()
}

/// Jeu angulaire au niveau du pignon `j_θ = j_t / r` (radians).
///
/// Panique si `circular_backlash < 0` ou si `pitch_radius <= 0`.
pub fn gear_backlash_angular(circular_backlash: f64, pitch_radius: f64) -> f64 {
    assert!(circular_backlash >= 0.0, "j_t ≥ 0 requis");
    assert!(pitch_radius > 0.0, "r > 0 requis");
    circular_backlash / pitch_radius
}

/// Jeu circonférentiel dû à l'amincissement symétrique des dents des deux roues
/// `j_t = 2·Δs`.
///
/// Panique si `tooth_thickness_reduction < 0`.
pub fn gear_backlash_from_tooth_thinning(tooth_thickness_reduction: f64) -> f64 {
    assert!(tooth_thickness_reduction >= 0.0, "Δs ≥ 0 requis");
    2.0 * tooth_thickness_reduction
}

/// Accroissement d'entraxe produisant un jeu circonférentiel visé (réciproque de
/// [`gear_backlash_circular`]) `ΔC = j_t / (2·tan(α))`.
///
/// Panique si `target_backlash < 0` ou si `pressure_angle_rad` n'est pas dans
/// `]0, π/2[`.
pub fn gear_backlash_center_distance_for_backlash(
    target_backlash: f64,
    pressure_angle_rad: f64,
) -> f64 {
    assert!(target_backlash >= 0.0, "j_t ≥ 0 requis");
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < core::f64::consts::FRAC_PI_2,
        "0 < α < π/2 requis"
    );
    target_backlash / (2.0 * pressure_angle_rad.tan())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn circular_and_center_distance_are_reciprocal() {
        // ΔC → j_t → ΔC doit restituer l'accroissement d'entraxe initial.
        let alpha = 20.0_f64.to_radians();
        let delta_c = 0.000_15;
        let j_t = gear_backlash_circular(delta_c, alpha);
        assert_relative_eq!(
            gear_backlash_center_distance_for_backlash(j_t, alpha),
            delta_c,
            epsilon = 1e-15
        );
    }

    #[test]
    fn circular_backlash_proportional_to_center_distance() {
        // j_t ∝ ΔC : doubler l'accroissement d'entraxe double le jeu.
        let alpha = 20.0_f64.to_radians();
        let single = gear_backlash_circular(0.0001, alpha);
        let double = gear_backlash_circular(0.0002, alpha);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-15);
    }

    #[test]
    fn tooth_thinning_matches_center_distance_route() {
        // Un amincissement Δs équivaut au jeu d'un accroissement d'entraxe
        // ΔC = Δs/tan(α), car 2·Δs = 2·(Δs/tan α)·tan α.
        let alpha = 20.0_f64.to_radians();
        let reduction = 0.00005;
        let via_thinning = gear_backlash_from_tooth_thinning(reduction);
        let via_center = gear_backlash_circular(reduction / alpha.tan(), alpha);
        assert_relative_eq!(via_thinning, via_center, epsilon = 1e-15);
    }

    #[test]
    fn angular_backlash_scales_inversely_with_radius() {
        // j_θ = j_t/r : un rayon deux fois plus grand halve le jeu angulaire.
        let j_t = 0.00008;
        let small = gear_backlash_angular(j_t, 0.025);
        let large = gear_backlash_angular(j_t, 0.050);
        assert_relative_eq!(large, small / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn realistic_spur_gear_pair() {
        // ΔC = 0,1 mm ; α = 20° → j_t = 2·1e-4·tan20° = 2e-4·0,363970... .
        let alpha = 20.0_f64.to_radians();
        let j_t = gear_backlash_circular(0.0001, alpha);
        assert_relative_eq!(j_t, 2.0e-4 * 20.0_f64.to_radians().tan(), epsilon = 1e-15);
        // Jeu circonférentiel de l'ordre de 0,073 mm, réaliste pour un tel écart.
        assert!(j_t > 7.0e-5 && j_t < 7.5e-5);
        // Sur un pignon de rayon primitif 30 mm : j_θ = j_t/0,030 rad.
        let j_theta = gear_backlash_angular(j_t, 0.030);
        assert_relative_eq!(j_theta, j_t / 0.030, epsilon = 1e-15);
        assert!(j_theta > 0.002 && j_theta < 0.0025);
    }

    #[test]
    #[should_panic(expected = "0 < α < π/2")]
    fn right_angle_pressure_panics() {
        gear_backlash_circular(0.0001, core::f64::consts::FRAC_PI_2);
    }
}
