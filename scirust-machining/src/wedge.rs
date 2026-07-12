//! Coins (wedges) — effort d'entrée pour soulever une charge, avantage mécanique
//! idéal et condition d'auto-blocage, avec frottement de Coulomb sur les faces.
//!
//! ```text
//! effort moteur (levage)  P = W·tan(α + 2φ)
//! effort d'extraction     P' = W·tan(2φ − α)   (négatif ⇒ le coin sort seul)
//! avantage idéal          MA = 1/tan(α)        (sans frottement)
//! auto-blocage            α < 2φ               (le coin reste en place)
//! ```
//!
//! `W` charge soulevée (N), `α` angle du coin (rad), `φ` angle de frottement
//! (`tan φ = µ`, identique sur les deux faces de contact). Le facteur `2φ`
//! traduit le frottement sur les **deux** surfaces (l'une horizontale, l'autre
//! inclinée). Un coin auto-bloquant reste sous charge une fois enfoncé.
//!
//! **Convention** : SI cohérent, angles en rad. **Limite honnête** : coin de
//! poids négligeable, entraînement horizontal, charge verticale, frottement `φ`
//! identique sur les deux plans (le résultat `tan(α+2φ)` est **exact** pour cette
//! configuration). Autres géométries : à recomposer par l'appelant.

use core::f64::consts::FRAC_PI_2;

/// Effort moteur pour **enfoncer** le coin et soulever la charge
/// `P = W·tan(α + 2φ)` (N).
///
/// Panique si `α + 2φ ≥ π/2` (blocage géométrique).
pub fn driving_force(load: f64, wedge_angle: f64, friction_angle: f64) -> f64 {
    let arg = wedge_angle + 2.0 * friction_angle;
    assert!(arg < FRAC_PI_2, "α + 2φ doit rester inférieur à π/2");
    load * arg.tan()
}

/// Effort pour **extraire** le coin `P' = W·tan(2φ − α)` (N).
///
/// Positif si un effort est requis (coin auto-bloquant), négatif si le coin
/// ressort seul sous la charge.
pub fn extraction_force(load: f64, wedge_angle: f64, friction_angle: f64) -> f64 {
    load * (2.0 * friction_angle - wedge_angle).tan()
}

/// Avantage mécanique idéal (sans frottement) `MA = 1/tan(α)`.
///
/// Panique si `tan(α) <= 0`.
pub fn ideal_mechanical_advantage(wedge_angle: f64) -> f64 {
    let t = wedge_angle.tan();
    assert!(t > 0.0, "l'angle du coin doit vérifier 0 < α < π/2");
    1.0 / t
}

/// Vrai si le coin est **auto-bloquant** (`α < 2φ` : il reste en place).
pub fn self_locking(wedge_angle: f64, friction_angle: f64) -> bool {
    wedge_angle < 2.0 * friction_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn frictionless_wedge_needs_load_times_tan_alpha() {
        // φ=0 → P = W·tan(α) ; MA = 1/tan(α).
        let alpha = 10.0_f64.to_radians();
        assert_relative_eq!(
            driving_force(1000.0, alpha, 0.0),
            1000.0 * alpha.tan(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            ideal_mechanical_advantage(alpha),
            1.0 / alpha.tan(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn friction_raises_the_driving_force() {
        // Le frottement augmente l'effort moteur.
        let alpha = 10.0_f64.to_radians();
        let phi = 8.0_f64.to_radians();
        assert!(driving_force(1000.0, alpha, phi) > driving_force(1000.0, alpha, 0.0));
    }

    #[test]
    fn self_locking_when_angle_small() {
        // α=10°, φ=8° → 2φ=16° > α → auto-bloquant ; extraction exige un effort > 0.
        let (alpha, phi) = (10.0_f64.to_radians(), 8.0_f64.to_radians());
        assert!(self_locking(alpha, phi));
        assert!(extraction_force(1000.0, alpha, phi) > 0.0);
        // α=20° > 2φ=16° → non bloquant, le coin ressort seul (effort < 0).
        let alpha2 = 20.0_f64.to_radians();
        assert!(!self_locking(alpha2, phi));
        assert!(extraction_force(1000.0, alpha2, phi) < 0.0);
    }

    #[test]
    #[should_panic(expected = "α + 2φ")]
    fn geometric_lock_panics() {
        // α=60°, φ=20° → α+2φ=100° ≥ 90°.
        driving_force(1000.0, 60.0_f64.to_radians(), 20.0_f64.to_radians());
    }
}
