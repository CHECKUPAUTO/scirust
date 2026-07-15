//! Profondeur de fraisage d'un lamage conique (**countersink**) obtenue à partir
//! du diamètre du lamage, du diamètre du trou et de l'angle inclus du cône.
//!
//! ```text
//! profondeur     h   = (D_cs − D_hole) / (2 · tan(α/2))
//! diamètre cône  D_cs = D_hole + 2 · h · tan(α/2)     (réciproque)
//! ```
//!
//! `D_cs` diamètre de face du lamage conique (m), `D_hole` diamètre du trou de
//! passage (m), `α` angle inclus du cône (rad, 90° = π/2 usuel pour tête
//! fraisée), `h` profondeur axiale du cône mesurée du bord de face jusqu'à la
//! jonction avec le trou (m). Par construction `countersink_diameter_from_depth`
//! est **réciproque** de `countersink_depth` : injecter `h` redonne `D_cs`.
//!
//! **Convention** : SI cohérent (longueurs dans une même unité, angle en rad).
//! Cône **symétrique** coaxial au trou ; `α` est l'angle **total** inclus, si
//! bien que le demi-angle vaut `α/2`. On se restreint à `α ∈ (0, π)` (demi-angle
//! dans `(0, π/2)`, `tan` fini et positif) et `D_cs ≥ D_hole` (profondeur ≥ 0).
//!
//! **Limite honnête** : géométrie idéale d'un cône parfait ; on **ignore** le
//! rayon de bec / la géométrie réelle de l'outil, le débouchage et les
//! chanfreins d'entrée. Aucun angle « par défaut », aucun diamètre normalisé ni
//! aucune tolérance n'est imposé : diamètres et angle inclus sont **fournis par
//! l'appelant** (par ex. 90° pour une vis à tête fraisée, 82° en usage impérial).

use core::f64::consts::PI;

/// Profondeur axiale `h = (D_cs − D_hole) / (2 · tan(α/2))` (m) du cône d'un
/// lamage de diamètre de face `countersink_diameter` percé sur un trou de
/// diamètre `hole_diameter`, pour un angle inclus `included_angle_rad`.
///
/// Panique si `included_angle_rad` sort de `(0, π)`, si `hole_diameter <= 0` ou
/// si `countersink_diameter < hole_diameter`.
pub fn countersink_depth(
    countersink_diameter: f64,
    hole_diameter: f64,
    included_angle_rad: f64,
) -> f64 {
    assert!(
        included_angle_rad > 0.0 && included_angle_rad < PI,
        "l'angle inclus α doit être strictement compris dans (0, π) rad"
    );
    assert!(
        hole_diameter > 0.0,
        "le diamètre du trou D_hole doit être strictement positif"
    );
    assert!(
        countersink_diameter >= hole_diameter,
        "le diamètre du lamage D_cs ne peut être inférieur au diamètre du trou D_hole"
    );
    (countersink_diameter - hole_diameter) / (2.0 * (included_angle_rad / 2.0).tan())
}

/// Diamètre de face `D_cs = D_hole + 2 · h · tan(α/2)` (m) du lamage conique
/// atteint en fraisant à la profondeur `depth` sur un trou de diamètre
/// `hole_diameter`, pour un angle inclus `included_angle_rad`.
///
/// Panique si `included_angle_rad` sort de `(0, π)`, si `hole_diameter <= 0` ou
/// si `depth < 0`.
pub fn countersink_diameter_from_depth(
    hole_diameter: f64,
    depth: f64,
    included_angle_rad: f64,
) -> f64 {
    assert!(
        included_angle_rad > 0.0 && included_angle_rad < PI,
        "l'angle inclus α doit être strictement compris dans (0, π) rad"
    );
    assert!(
        hole_diameter > 0.0,
        "le diamètre du trou D_hole doit être strictement positif"
    );
    assert!(depth >= 0.0, "la profondeur h ne peut être négative");
    hole_diameter + 2.0 * depth * (included_angle_rad / 2.0).tan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn depth_and_diameter_are_reciprocal() {
        // Réciprocité : injecter h = countersink_depth(D_cs, …) redonne D_cs.
        let d_hole = 6.0_f64;
        let alpha = 90.0_f64.to_radians();
        for &d_cs in &[6.0_f64, 8.0, 11.0, 20.0]
        {
            let h = countersink_depth(d_cs, d_hole, alpha);
            let back = countersink_diameter_from_depth(d_hole, h, alpha);
            assert_relative_eq!(back, d_cs, max_relative = 1e-12);
        }
    }

    #[test]
    fn equal_diameters_give_zero_depth() {
        // Cas limite : D_cs = D_hole ⇒ aucun cône à usiner, h = 0.
        let h = countersink_depth(9.0, 9.0, 82.0_f64.to_radians());
        assert_relative_eq!(h, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn depth_is_proportional_to_diametral_difference() {
        // h ∝ (D_cs − D_hole) à angle fixé : doubler l'excès diamétral double h.
        let d_hole = 5.0_f64;
        let alpha = 100.0_f64.to_radians();
        let h1 = countersink_depth(d_hole + 2.0, d_hole, alpha);
        let h2 = countersink_depth(d_hole + 4.0, d_hole, alpha);
        assert_relative_eq!(h2, 2.0 * h1, max_relative = 1e-12);
    }

    #[test]
    fn right_angle_cone_depth_equals_radial_step() {
        // α = 90° ⇒ tan(α/2) = 1 ⇒ h = (D_cs − D_hole)/2 = pas radial.
        let d_cs = 12.4_f64;
        let d_hole = 6.4_f64;
        let h = countersink_depth(d_cs, d_hole, PI / 2.0);
        assert_relative_eq!(h, (d_cs - d_hole) / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_flat_head_screw_case() {
        // Vis M6 à tête fraisée 90° : trou Ø6,4 mm, lamage Ø12,4 mm.
        // h = (12,4 − 6,4)/(2·tan45°) = 6,0/2 = 3,0 mm.
        let h = countersink_depth(12.4, 6.4, 90.0_f64.to_radians());
        assert_relative_eq!(h, 3.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "ne peut être inférieur au diamètre du trou")]
    fn countersink_smaller_than_hole_panics() {
        countersink_depth(5.0, 8.0, PI / 2.0);
    }
}
