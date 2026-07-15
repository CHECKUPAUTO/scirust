//! **Glissement d'une courroie** — perte de vitesse par glissement, rapport de
//! transmission effectif corrigé et fluage élastique (creep) issu de la tension.
//!
//! ```text
//! perte de vitesse   s = (v1 − v2)/v1                (v1 vitesse au menant, v2 au mené)
//! rapport effectif   i_eff = (D1/D2)·(1 − s)         (D1 menant, D2 mené)
//! fluage (creep)     ε = (T1 − T2)/(A·E)             (T1 brin tendu, T2 brin mou)
//! ```
//!
//! `s` fraction de glissement total (sans dimension, `0 ≤ s < 1`), `v1`, `v2`
//! vitesses tangentielles (linéaires) de la courroie au menant et au mené (m/s),
//! `i_eff` rapport de transmission effectif (sans dimension), `D1`, `D2` diamètres
//! primitifs des poulies menante et menée (m), `ε` fluage élastique relatif (sans
//! dimension), `T1`, `T2` tensions des brins tendu et mou (N), `A` section de la
//! courroie (m²), `E` module d'Young du matériau de courroie (Pa).
//!
//! **Convention** : SI cohérent. **Limite honnête** : glissement et fluage
//! supposés **petits** (régime établi, courroie non emballée) ; le module `E` et
//! la section `A` du matériau de courroie, comme les tensions et diamètres,
//! sont **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée.
//! Distinct du dimensionnement par [`crate::belts`] (Euler-Eytelwein).

/// Perte de vitesse par glissement `s = (v1 − v2)/v1`, avec `v1` vitesse
/// tangentielle de la courroie au menant (idéale) et `v2` au mené (réelle).
///
/// Panique si `driver_speed <= 0` ou `driven_speed < 0` ou `driven_speed > driver_speed`.
pub fn belt_slip_speed_loss(driver_speed: f64, driven_speed: f64) -> f64 {
    assert!(driver_speed > 0.0, "v1 > 0 requis");
    assert!(
        (0.0..=driver_speed).contains(&driven_speed),
        "0 ≤ v2 ≤ v1 requis (glissement positif)"
    );
    (driver_speed - driven_speed) / driver_speed
}

/// Rapport de transmission effectif corrigé du glissement
/// `i_eff = (D1/D2)·(1 − s)`.
///
/// Panique si un diamètre `<= 0` ou si `slip_fraction` ∉ `[0, 1)`.
pub fn belt_slip_effective_velocity_ratio(
    driver_diameter: f64,
    driven_diameter: f64,
    slip_fraction: f64,
) -> f64 {
    assert!(
        driver_diameter > 0.0 && driven_diameter > 0.0,
        "D1 > 0 et D2 > 0 requis"
    );
    assert!(
        (0.0..1.0).contains(&slip_fraction),
        "0 ≤ s < 1 requis (glissement petit)"
    );
    (driver_diameter / driven_diameter) * (1.0 - slip_fraction)
}

/// Fluage élastique relatif (creep) issu de la différence de tension
/// `ε = (T1 − T2)/(A·E)`.
///
/// Panique si `tight_tension < slack_tension`, si une tension `< 0`,
/// ou si `area <= 0` ou `youngs_modulus <= 0`.
pub fn belt_slip_creep_from_tension(
    tight_tension: f64,
    slack_tension: f64,
    youngs_modulus: f64,
    area: f64,
) -> f64 {
    assert!(
        slack_tension >= 0.0 && tight_tension >= slack_tension,
        "T1 ≥ T2 ≥ 0 requis (brin tendu ≥ brin mou)"
    );
    assert!(youngs_modulus > 0.0 && area > 0.0, "E > 0 et A > 0 requis");
    (tight_tension - slack_tension) / (area * youngs_modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn no_slip_when_speeds_equal() {
        // v2 = v1 → glissement nul.
        assert_relative_eq!(belt_slip_speed_loss(10.0, 10.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn speed_loss_reciprocity() {
        // Si v2 = v1·(1 − s), on doit retrouver exactement s.
        let v1 = 12.5_f64;
        let s = 0.03_f64;
        let v2 = v1 * (1.0 - s);
        assert_relative_eq!(belt_slip_speed_loss(v1, v2), s, epsilon = 1e-12);
    }

    #[test]
    fn effective_ratio_reduces_to_geometric_without_slip() {
        // À s = 0, i_eff = D1/D2 (rapport purement géométrique).
        assert_relative_eq!(
            belt_slip_effective_velocity_ratio(0.3, 0.15, 0.0),
            0.3 / 0.15,
            epsilon = 1e-12
        );
    }

    #[test]
    fn slip_reduces_effective_ratio() {
        // Le glissement diminue le rapport effectif d'exactement le facteur (1 − s).
        let full = belt_slip_effective_velocity_ratio(0.3, 0.15, 0.0);
        let slipped = belt_slip_effective_velocity_ratio(0.3, 0.15, 0.04);
        assert_relative_eq!(slipped, full * (1.0 - 0.04), epsilon = 1e-12);
        assert!(slipped < full);
    }

    #[test]
    fn creep_scales_with_tension_and_inverse_of_stiffness() {
        // ε est proportionnel à (T1 − T2) et inversement proportionnel à A.
        let base = belt_slip_creep_from_tension(600.0, 200.0, 100.0e9, 1.0e-4);
        let double_delta = belt_slip_creep_from_tension(1000.0, 200.0, 100.0e9, 1.0e-4);
        let double_area = belt_slip_creep_from_tension(600.0, 200.0, 100.0e9, 2.0e-4);
        assert_relative_eq!(double_delta, 2.0 * base, epsilon = 1e-15);
        assert_relative_eq!(double_area, base / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn creep_realistic_value() {
        // T1=600 N, T2=200 N, E=100 GPa, A=1e-4 m² → ε = 400/(1e-4·1e11) = 4e-5.
        assert_relative_eq!(
            belt_slip_creep_from_tension(600.0, 200.0, 100.0e9, 1.0e-4),
            4.0e-5,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "0 ≤ s < 1 requis")]
    fn slip_out_of_range_panics() {
        belt_slip_effective_velocity_ratio(0.3, 0.15, 1.0);
    }
}
