//! Joint de **Cardan** (joint universel de Hooke) — irrégularité de transmission
//! entre deux arbres formant un angle `β` : rapport de vitesses instantané,
//! angle de sortie et bornes de fluctuation.
//!
//! ```text
//! angle de sortie      tan θ2 = tan θ1 / cosβ
//! rapport de vitesses  ω2/ω1 = cosβ / (1 − sin²β·cos²θ1)
//! rapport maximal      (ω2/ω1)_max = 1/cosβ     (θ1 = 0, π)
//! rapport minimal      (ω2/ω1)_min = cosβ        (θ1 = π/2, 3π/2)
//! ```
//!
//! `β` angle entre les arbres (rad, `0 ≤ β < π/2`), `θ1` angle de rotation de
//! l'arbre menant (rad). La vitesse de sortie oscille deux fois par tour entre
//! `cosβ·ω1` et `ω1/cosβ` ; deux joints en opposition (double cardan) annulent
//! cette irrégularité.
//!
//! **Convention** : angles en rad, vitesses algébriques. **Limite honnête** :
//! cinématique **exacte** d'un joint de Cardan simple ; ne modélise ni les
//! efforts, ni le double cardan, ni le joint homocinétique (tripode, Rzeppa).

use core::f64::consts::FRAC_PI_2;

/// Angle de l'arbre mené `θ2` tel que `tan θ2 = tan θ1 / cosβ`, renvoyé dans le
/// bon quadrant via `atan2`.
///
/// Panique si `β ≥ π/2`.
pub fn output_angle(beta_rad: f64, theta1_rad: f64) -> f64 {
    assert!(
        beta_rad.abs() < FRAC_PI_2,
        "l'angle entre arbres doit vérifier β < π/2"
    );
    (theta1_rad.sin()).atan2(theta1_rad.cos() * beta_rad.cos())
}

/// Rapport de vitesses instantané `ω2/ω1 = cosβ/(1 − sin²β·cos²θ1)`.
///
/// Panique si `β ≥ π/2`.
pub fn velocity_ratio(beta_rad: f64, theta1_rad: f64) -> f64 {
    assert!(
        beta_rad.abs() < FRAC_PI_2,
        "l'angle entre arbres doit vérifier β < π/2"
    );
    let sb = beta_rad.sin();
    let c1 = theta1_rad.cos();
    beta_rad.cos() / (1.0 - sb * sb * c1 * c1)
}

/// Rapport de vitesses **maximal** sur un tour `1/cosβ` (arbre mené le plus
/// rapide, à `θ1 = 0` ou `π`).
///
/// Panique si `β ≥ π/2`.
pub fn max_velocity_ratio(beta_rad: f64) -> f64 {
    assert!(
        beta_rad.abs() < FRAC_PI_2,
        "l'angle entre arbres doit vérifier β < π/2"
    );
    1.0 / beta_rad.cos()
}

/// Rapport de vitesses **minimal** sur un tour `cosβ` (arbre mené le plus lent,
/// à `θ1 = π/2` ou `3π/2`).
pub fn min_velocity_ratio(beta_rad: f64) -> f64 {
    beta_rad.cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn aligned_shafts_transmit_uniformly() {
        // β=0 : ω2/ω1 = 1 pour tout θ1 (transmission parfaite).
        assert_relative_eq!(velocity_ratio(0.0, 0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(velocity_ratio(0.0, 1.234), 1.0, epsilon = 1e-12);
        assert_relative_eq!(output_angle(0.0, 0.7), 0.7, epsilon = 1e-12);
    }

    #[test]
    fn ratio_extremes_match_closed_forms() {
        // β=30° : max=1/cos30≈1,1547 à θ1=0 ; min=cos30≈0,8660 à θ1=π/2.
        let beta = PI / 6.0;
        assert_relative_eq!(
            velocity_ratio(beta, 0.0),
            max_velocity_ratio(beta),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            velocity_ratio(beta, FRAC_PI_2),
            min_velocity_ratio(beta),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            max_velocity_ratio(beta),
            1.0 / (beta.cos()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn min_below_one_below_max() {
        // La fluctuation encadre 1 : min < 1 < max pour β>0.
        let beta = PI / 5.0;
        assert!(min_velocity_ratio(beta) < 1.0);
        assert!(max_velocity_ratio(beta) > 1.0);
        // Produit min·max = 1 (symétrie du cardan).
        assert_relative_eq!(
            min_velocity_ratio(beta) * max_velocity_ratio(beta),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn output_angle_leads_or_lags_but_matches_at_cardinals() {
        // Aux angles cardinaux, θ2 = θ1 (les arbres se recroisent).
        let beta = PI / 6.0;
        assert_relative_eq!(output_angle(beta, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(output_angle(beta, FRAC_PI_2), FRAC_PI_2, epsilon = 1e-12);
        assert_relative_eq!(output_angle(beta, PI), PI, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "β < π/2")]
    fn right_angle_panics() {
        velocity_ratio(FRAC_PI_2, 0.0);
    }
}
