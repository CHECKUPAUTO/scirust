//! Mécanisme **bielle-manivelle** (slider-crank) — cinématique du piston en
//! fonction de l'angle de manivelle : course, vitesse et accélération.
//!
//! ```text
//! rapport d'obliquité   λ = r/l
//! déplacement (PMH)     s = r·(1 − cosθ) + l·(1 − √(1 − λ²·sin²θ))
//! vitesse (exacte)      v = r·ω·[sinθ + λ·sinθ·cosθ/√(1 − λ²·sin²θ)]
//! accélération (approx) a ≈ r·ω²·(cosθ + λ·cos2θ)
//! ```
//!
//! `r` rayon de manivelle, `l` longueur de bielle, `θ` angle de manivelle mesuré
//! depuis le **point mort haut** (PMH), `ω` vitesse angulaire de la manivelle
//! (rad/s, supposée constante). Le déplacement `s` est compté depuis le PMH ; la
//! course totale vaut `2r` (atteinte au point mort bas, `θ = π`).
//!
//! **Convention** : longueurs cohérentes, angles en rad. **Limite honnête** :
//! déplacement et vitesse **exacts** ; l'accélération est l'approximation à deux
//! termes usuelle (développement de la racine), valable pour `λ` modéré
//! (`l ≳ 3r`). `ω` constante ; pas de dynamique de bielle ni de jeu.

/// Rapport d'obliquité `λ = r/l` (sans dimension).
///
/// Panique si `l <= 0`.
pub fn obliquity_ratio(crank_radius: f64, rod_length: f64) -> f64 {
    assert!(
        rod_length > 0.0,
        "la longueur de bielle doit être strictement positive"
    );
    crank_radius / rod_length
}

/// Déplacement du piston depuis le PMH `s = r(1−cosθ) + l(1 − √(1−λ²sin²θ))`.
///
/// Panique si `l <= 0` ou si `λ·sinθ ≥ 1` (bielle plus courte que la manivelle).
pub fn piston_displacement(crank_radius: f64, rod_length: f64, theta_rad: f64) -> f64 {
    let lambda = obliquity_ratio(crank_radius, rod_length);
    let s = theta_rad.sin();
    let radicand = 1.0 - lambda * lambda * s * s;
    assert!(radicand > 0.0, "géométrie impossible : λ·sinθ ≥ 1");
    crank_radius * (1.0 - theta_rad.cos()) + rod_length * (1.0 - radicand.sqrt())
}

/// Vitesse **exacte** du piston `v = r·ω·[sinθ + λ·sinθ·cosθ/√(1−λ²sin²θ)]`
/// (dérivée du déplacement depuis le PMH).
///
/// Panique si `l <= 0` ou si `λ·sinθ ≥ 1`.
pub fn piston_velocity(
    crank_radius: f64,
    rod_length: f64,
    theta_rad: f64,
    omega_rad_s: f64,
) -> f64 {
    let lambda = obliquity_ratio(crank_radius, rod_length);
    let (s, c) = (theta_rad.sin(), theta_rad.cos());
    let radicand = 1.0 - lambda * lambda * s * s;
    assert!(radicand > 0.0, "géométrie impossible : λ·sinθ ≥ 1");
    crank_radius * omega_rad_s * (s + lambda * s * c / radicand.sqrt())
}

/// Accélération **approchée** du piston `a ≈ r·ω²·(cosθ + λ·cos2θ)`.
pub fn piston_acceleration_approx(
    crank_radius: f64,
    rod_length: f64,
    theta_rad: f64,
    omega_rad_s: f64,
) -> f64 {
    let lambda = obliquity_ratio(crank_radius, rod_length);
    crank_radius * omega_rad_s * omega_rad_s * (theta_rad.cos() + lambda * (2.0 * theta_rad).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn stroke_is_twice_crank_radius() {
        // PMH (θ=0) : s=0 ; PMB (θ=π) : s = 2r.
        assert_relative_eq!(piston_displacement(0.05, 0.2, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(piston_displacement(0.05, 0.2, PI), 0.10, epsilon = 1e-12);
    }

    #[test]
    fn velocity_zero_at_dead_centers() {
        // Vitesse nulle aux deux points morts (sinθ = 0).
        assert_relative_eq!(piston_velocity(0.05, 0.2, 0.0, 100.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(piston_velocity(0.05, 0.2, PI, 100.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_at_quarter_turn_equals_r_omega() {
        // À θ=π/2 : sinθ·cosθ = 0 → v = r·ω exactement.
        assert_relative_eq!(
            piston_velocity(0.05, 0.2, FRAC_PI_2, 100.0),
            0.05 * 100.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn acceleration_extremes_at_dead_centers() {
        // θ=0 : a = rω²(1+λ) ; θ=π : a = rω²(λ−1) ; θ=π/2 : a = −rω²·λ.
        let (r, l, w) = (0.05, 0.2, 100.0);
        let lambda = r / l;
        assert_relative_eq!(
            piston_acceleration_approx(r, l, 0.0, w),
            r * w * w * (1.0 + lambda),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            piston_acceleration_approx(r, l, PI, w),
            r * w * w * (lambda - 1.0),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            piston_acceleration_approx(r, l, FRAC_PI_2, w),
            -r * w * w * lambda,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "géométrie impossible")]
    fn rod_shorter_than_crank_panics() {
        // λ = 2 > 1 : à θ=π/2, λ·sinθ = 2 ≥ 1.
        piston_displacement(0.4, 0.2, FRAC_PI_2);
    }
}
