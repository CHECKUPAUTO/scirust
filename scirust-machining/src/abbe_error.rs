//! Erreur d'**Abbe** — biais du premier ordre d'une mesure dont l'axe de
//! mesure n'est pas colinéaire avec l'échelle : un défaut angulaire `α` du
//! chariot, multiplié par le décalage d'Abbe `d` (distance entre l'axe mesuré
//! et l'axe de la règle), produit une erreur de longueur.
//!
//! ```text
//! erreur d'Abbe        e = d · tan(α)          (modèle exact du 1er ordre)
//! approx. petit angle  e ≈ d · α               (tan α ≈ α, α → 0)
//! décalage max toléré  d_max = e_adm / tan(α)  (réciproque, pour α ∈ (0, π/2))
//! ```
//!
//! `d` décalage d'Abbe (m), `α` défaut angulaire de guidage (rad), `e` erreur
//! d'Abbe résultante (m), `e_adm` erreur admissible (m). `abbe_error` est
//! **réciproque** de `abbe_max_offset_for_error` : pour un même angle,
//! `abbe_max_offset_for_error(abbe_error(d, α), α) = d`.
//!
//! **Convention** : SI cohérent — longueurs en mètres (`d`, `e`, `e_adm` dans
//! une même unité), angles en radians. Défauts angulaires restreints à
//! `[0, π/2)` (guidage quasi rectiligne).
//!
//! **Limite honnête** : modèle d'erreur du **premier ordre** dû au seul
//! non-respect du principe d'Abbe (axe de mesure non colinéaire) pour un petit
//! défaut angulaire ; distinct de l'erreur cosinus [`crate::cosine_error`]
//! (désalignement de l'axe de palpage). Aucun budget d'erreur, aucun décalage
//! ni aucun défaut angulaire « par défaut » n'est imposé : toutes ces valeurs
//! sont **fournies par l'appelant**.

use core::f64::consts::PI;

/// Erreur d'Abbe exacte du premier ordre `e = d · tan(α)` (même unité que
/// `offset_distance`), pour un décalage d'Abbe `offset_distance` et un défaut
/// angulaire de guidage `angular_error_rad`.
///
/// Panique si `offset_distance < 0` ou si `angular_error_rad` sort de `[0, π/2)`.
pub fn abbe_error(offset_distance: f64, angular_error_rad: f64) -> f64 {
    assert!(
        offset_distance >= 0.0,
        "le décalage d'Abbe d ne peut être négatif"
    );
    assert!(
        (0.0..PI / 2.0).contains(&angular_error_rad),
        "le défaut angulaire α doit être compris dans [0, π/2) rad"
    );
    offset_distance * angular_error_rad.tan()
}

/// Approximation petit angle de l'erreur d'Abbe `e ≈ d · α` (même unité que
/// `offset_distance`), pour un décalage `offset_distance` et un défaut angulaire
/// `angular_error_rad`.
///
/// Panique si `offset_distance < 0` ou si `angular_error_rad` sort de `[0, π/2)`.
pub fn abbe_error_small_angle(offset_distance: f64, angular_error_rad: f64) -> f64 {
    assert!(
        offset_distance >= 0.0,
        "le décalage d'Abbe d ne peut être négatif"
    );
    assert!(
        (0.0..PI / 2.0).contains(&angular_error_rad),
        "le défaut angulaire α doit être compris dans [0, π/2) rad"
    );
    offset_distance * angular_error_rad
}

/// Décalage d'Abbe maximal `d_max = e_adm / tan(α)` (même unité que
/// `allowable_error`) tolérable pour que l'erreur d'Abbe reste sous
/// `allowable_error` avec un défaut angulaire `angular_error_rad`.
///
/// Panique si `allowable_error < 0` ou si `angular_error_rad` sort de `(0, π/2)`
/// (l'angle nul annule `tan α` et rend le décalage non borné).
pub fn abbe_max_offset_for_error(allowable_error: f64, angular_error_rad: f64) -> f64 {
    assert!(
        allowable_error >= 0.0,
        "l'erreur admissible e_adm ne peut être négative"
    );
    assert!(
        angular_error_rad > 0.0 && angular_error_rad < PI / 2.0,
        "le défaut angulaire α doit être compris dans (0, π/2) rad"
    );
    allowable_error / angular_error_rad.tan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn max_offset_is_reciprocal_of_error() {
        // Réciprocité : pour un même angle, d → e → d reproduit le décalage.
        let alpha = 5.0e-4_f64;
        for &d in &[0.0_f64, 0.01, 0.1, 0.5, 2.0]
        {
            let e = abbe_error(d, alpha);
            assert_relative_eq!(abbe_max_offset_for_error(e, alpha), d, max_relative = 1e-12);
        }
    }

    #[test]
    fn no_offset_or_no_angle_means_no_error() {
        // Principe d'Abbe respecté (d = 0) ou guidage parfait (α = 0) : e = 0.
        assert_relative_eq!(abbe_error(0.0, 1.0e-3), 0.0, epsilon = 1e-15);
        assert_relative_eq!(abbe_error(0.2, 0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(abbe_error_small_angle(0.0, 1.0e-3), 0.0, epsilon = 1e-15);
        assert_relative_eq!(abbe_error_small_angle(0.2, 0.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn error_is_proportional_to_offset() {
        // e = d·tan α est linéaire en d : doubler le décalage double l'erreur.
        let alpha = 2.0e-3_f64;
        let e1 = abbe_error(0.05, alpha);
        let e2 = abbe_error(0.10, alpha);
        assert_relative_eq!(e2, 2.0 * e1, max_relative = 1e-12);
    }

    #[test]
    fn small_angle_matches_exact_for_tiny_angle() {
        // À petit angle, tan α ≈ α : l'approximation colle au modèle exact.
        let d = 0.15_f64;
        let alpha = 1.0e-4_f64;
        assert_relative_eq!(
            abbe_error_small_angle(d, alpha),
            abbe_error(d, alpha),
            max_relative = 1e-6
        );
    }

    #[test]
    fn realistic_worked_value() {
        // Cas chiffré : décalage d = 0,1 m, défaut α = 1 mrad = 1e-3 rad.
        // tan(1e-3) = 1e-3 + (1e-3)^3/3 + ... = 1.000000333333e-3, donc
        // e = 0,1 · 1.000000333333e-3 = 1.000000333333e-4 m ≈ 100,00003 µm.
        let d = 0.1_f64;
        let alpha = 1.0e-3_f64;
        let expected = d * alpha.tan();
        assert_relative_eq!(abbe_error(d, alpha), expected, max_relative = 1e-12);
        // L'approximation petit angle vaut exactement d·α = 1.0e-4 m = 100 µm.
        assert_relative_eq!(
            abbe_error_small_angle(d, alpha),
            1.0e-4,
            max_relative = 1e-12
        );
        // Écart relatif exact/approx ≈ α²/3 ≈ 3.33e-7 (négligeable ici).
        let rel = (abbe_error(d, alpha) - 1.0e-4) / 1.0e-4;
        assert_relative_eq!(rel, alpha * alpha / 3.0, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le défaut angulaire α doit être compris dans [0, π/2) rad")]
    fn error_panics_on_right_angle() {
        let _ = abbe_error(0.1, PI / 2.0);
    }
}
