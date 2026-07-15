//! **Roue libre / antidévireur** (à cames ou galets) — effort de coincement,
//! couple transmissible par adhérence et condition d'auto-coincement (arc-boutement).
//!
//! ```text
//! effort normal      N = F_t / tan(α)                (coincement d'un galet/came)
//! couple transmis    C = μ · N · R · n               (adhérence sur les éléments)
//! auto-coincement    tan(α) ≤ 2·μ                    (condition d'arc-boutement)
//! angle limite       α_max = atan(2·μ)               (angle de came maximal)
//! ```
//!
//! `F_t` effort tangentiel appliqué à l'élément (N), `α` angle de came (ou angle de
//! coincement du galet) mesuré depuis la tangente à la piste (rad, `0 < α < π/2`),
//! `N` effort normal de coincement sur la piste (N), `μ` coefficient de frottement
//! d'adhérence galet/piste (sans dimension), `R` rayon de la piste de roulement (m),
//! `n` nombre d'éléments actifs (galets ou cames, sans dimension), `C` couple
//! transmissible (N·m), `α_max` angle de came limite (rad).
//!
//! **Convention** : SI cohérent. **Limite honnête** : coincement par **adhérence**
//! (galets/cames) ; le coefficient de frottement `μ` et l'angle de came `α` sont
//! **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée.
//! L'auto-coincement exige `tan(α) ≤ 2·μ` (arc-boutement), sinon la roue libre
//! **patine** ; l'effort est supposé **réparti idéalement** sur les `n` éléments.

use core::f64::consts::FRAC_PI_2;

/// Effort normal de coincement d'un galet/came `N = F_t / tan(α)`, avec `F_t`
/// effort tangentiel et `α` angle de came depuis la tangente à la piste.
///
/// Panique si `tangential_force < 0` ou si `wedge_angle_rad` ∉ `(0, π/2)`.
pub fn freewheel_normal_force(tangential_force: f64, wedge_angle_rad: f64) -> f64 {
    assert!(tangential_force >= 0.0, "F_t ≥ 0 requis");
    assert!(
        wedge_angle_rad > 0.0 && wedge_angle_rad < FRAC_PI_2,
        "0 < α < π/2 requis (angle de came)"
    );
    tangential_force / wedge_angle_rad.tan()
}

/// Couple transmissible par adhérence `C = μ · N · R · n`, réparti idéalement sur
/// `n` éléments actifs de rayon de piste `R`.
///
/// Panique si `friction_coefficient < 0`, si `normal_force < 0`,
/// si `race_radius <= 0` ou si `element_count < 1`.
pub fn freewheel_torque_capacity(
    friction_coefficient: f64,
    normal_force: f64,
    race_radius: f64,
    element_count: f64,
) -> f64 {
    assert!(friction_coefficient >= 0.0, "μ ≥ 0 requis");
    assert!(normal_force >= 0.0, "N ≥ 0 requis");
    assert!(race_radius > 0.0, "R > 0 requis");
    assert!(element_count >= 1.0, "n ≥ 1 requis (au moins un élément)");
    friction_coefficient * normal_force * race_radius * element_count
}

/// Condition d'auto-coincement (arc-boutement) `tan(α) ≤ 2·μ` : renvoie `true` si
/// la roue libre se coince, `false` si elle patine.
///
/// Panique si `friction_coefficient < 0` ou si `wedge_angle_rad` ∉ `(0, π/2)`.
pub fn freewheel_self_locking(friction_coefficient: f64, wedge_angle_rad: f64) -> bool {
    assert!(friction_coefficient >= 0.0, "μ ≥ 0 requis");
    assert!(
        wedge_angle_rad > 0.0 && wedge_angle_rad < FRAC_PI_2,
        "0 < α < π/2 requis (angle de came)"
    );
    wedge_angle_rad.tan() <= 2.0 * friction_coefficient
}

/// Angle de came limite de coincement `α_max = atan(2·μ)` (rad) : au-delà, la roue
/// libre patine au lieu de se coincer.
///
/// Panique si `friction_coefficient < 0`.
pub fn freewheel_max_wedge_angle(friction_coefficient: f64) -> f64 {
    assert!(friction_coefficient >= 0.0, "μ ≥ 0 requis");
    (2.0 * friction_coefficient).atan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normal_force_reciprocity() {
        // Par définition, N · tan(α) doit redonner exactement F_t.
        let ft = 1000.0_f64;
        let alpha = 0.15_f64;
        let n = freewheel_normal_force(ft, alpha);
        assert_relative_eq!(n * alpha.tan(), ft, epsilon = 1e-9);
    }

    #[test]
    fn normal_force_realistic_value() {
        // tan(α) = 0.1, F_t = 1000 N → N = 1000/0.1 = 10000 N.
        let alpha = 0.1_f64.atan();
        assert_relative_eq!(
            freewheel_normal_force(1000.0, alpha),
            10_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn torque_realistic_value() {
        // μ=0.05, N=10000 N, R=0.03 m, n=6 → C = 0.05·10000·0.03·6 = 90 N·m.
        assert_relative_eq!(
            freewheel_torque_capacity(0.05, 10_000.0, 0.03, 6.0),
            90.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn torque_proportional_to_element_count() {
        // Le couple est proportionnel au nombre d'éléments actifs.
        let one = freewheel_torque_capacity(0.05, 10_000.0, 0.03, 1.0);
        let six = freewheel_torque_capacity(0.05, 10_000.0, 0.03, 6.0);
        assert_relative_eq!(six, 6.0 * one, epsilon = 1e-9);
    }

    #[test]
    fn max_wedge_angle_matches_locking_boundary() {
        // À α_max, tan(α_max) = 2μ exactement, et le coincement est encore atteint (≤).
        let mu = 0.08_f64;
        let amax = freewheel_max_wedge_angle(mu);
        assert_relative_eq!(amax.tan(), 2.0 * mu, epsilon = 1e-12);
        assert!(freewheel_self_locking(mu, amax));
        // Un angle légèrement plus grand fait patiner la roue libre.
        assert!(!freewheel_self_locking(mu, amax + 0.01));
    }

    #[test]
    fn max_wedge_angle_realistic_value() {
        // μ = 0.08 → α_max = atan(0.16), donc tan(α_max) = 0.16.
        let amax = freewheel_max_wedge_angle(0.08);
        assert_relative_eq!(amax.tan(), 0.16, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < α < π/2 requis")]
    fn normal_force_zero_angle_panics() {
        freewheel_normal_force(1000.0, 0.0);
    }
}
