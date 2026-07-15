//! Fluotournage / repoussage conique (**metal spinning / flow forming**) —
//! prédiction de l'épaisseur de paroi d'un cône par la **loi du sinus**.
//!
//! ```text
//! épaisseur finale      t_f = t_0 · sin(α)          (loi du sinus)
//! amincissement         Δt  = t_0 · (1 − sin(α))    (réduction absolue)
//! rapport d'épaisseur    r   = t_f / t_0 = sin(α)    (final / initial)
//! angle limite          α   = asin(r)               (réciproque du rapport)
//! ```
//!
//! `t_0` épaisseur initiale du flan (m), `t_f` épaisseur finale de la paroi
//! du cône (m), `Δt` amincissement absolu (m), `α` **demi-angle au sommet** du
//! cône (rad, mesuré entre l'axe et la génératrice), `r` rapport épaisseur
//! finale / initiale (sans dimension, dans `(0, 1]`).
//!
//! **Convention** : SI cohérent ; `t_0`, `t_f` et `Δt` partagent la même unité
//! de longueur. Le demi-angle `α` est pris dans `(0, π/2]` : `α → π/2` (cône
//! plat, paroi presque perpendiculaire à l'axe) conserve l'épaisseur
//! (`sin → 1`), tandis qu'un `α` petit (cône élancé, pointu) amincit fortement.
//!
//! **Limite honnête** : modèle de fluotournage conique **idéal** suivant la
//! seule loi du sinus (déformation homogène, volume conservé, **sans**
//! allongement axial du métal). Il ignore l'écrouissage, le retour élastique,
//! l'échauffement et les **limites de formabilité** (amincissement maximal
//! admissible, nombre de passes) : ces bornes matériaux/procédé, comme le
//! demi-angle `α` lui-même, sont **fournies par l'appelant**. Aucune valeur
//! d'angle, de matériau ou de réduction « par défaut » n'est supposée.

use core::f64::consts::PI;

/// Épaisseur finale `t_f = t_0 · sin(α)` (m) de la paroi d'un cône obtenu par
/// fluotournage à partir d'un flan d'épaisseur `initial_thickness`, pour un
/// demi-angle au sommet `half_cone_angle_rad` (loi du sinus).
///
/// Panique si `initial_thickness <= 0` ou si `half_cone_angle_rad` sort de
/// `(0, π/2]`.
pub fn spinning_sine_law_thickness(initial_thickness: f64, half_cone_angle_rad: f64) -> f64 {
    assert!(
        initial_thickness > 0.0,
        "l'épaisseur initiale t_0 doit être strictement positive"
    );
    assert!(
        half_cone_angle_rad > 0.0 && half_cone_angle_rad <= PI / 2.0,
        "le demi-angle au sommet α doit être dans (0, π/2] rad"
    );
    initial_thickness * half_cone_angle_rad.sin()
}

/// Amincissement absolu `Δt = t_0 · (1 − sin(α))` (m) subi par la paroi lors du
/// fluotournage conique d'un flan d'épaisseur `initial_thickness` au demi-angle
/// `half_cone_angle_rad`.
///
/// Panique si `initial_thickness <= 0` ou si `half_cone_angle_rad` sort de
/// `(0, π/2]`.
pub fn spinning_thickness_reduction(initial_thickness: f64, half_cone_angle_rad: f64) -> f64 {
    assert!(
        initial_thickness > 0.0,
        "l'épaisseur initiale t_0 doit être strictement positive"
    );
    assert!(
        half_cone_angle_rad > 0.0 && half_cone_angle_rad <= PI / 2.0,
        "le demi-angle au sommet α doit être dans (0, π/2] rad"
    );
    initial_thickness * (1.0 - half_cone_angle_rad.sin())
}

/// Rapport d'épaisseur `r = t_f / t_0 = sin(α)` (sans dimension) imposé par la
/// loi du sinus pour un demi-angle au sommet `half_cone_angle_rad`.
///
/// Panique si `half_cone_angle_rad` sort de `(0, π/2]`.
pub fn spinning_reduction_ratio(half_cone_angle_rad: f64) -> f64 {
    assert!(
        half_cone_angle_rad > 0.0 && half_cone_angle_rad <= PI / 2.0,
        "le demi-angle au sommet α doit être dans (0, π/2] rad"
    );
    half_cone_angle_rad.sin()
}

/// Demi-angle au sommet limite `α = asin(r)` (rad) réalisant un rapport
/// d'épaisseur cible `target_reduction_ratio = t_f / t_0` (réciproque de
/// [`spinning_reduction_ratio`]).
///
/// Panique si `target_reduction_ratio` sort de `(0, 1]` (hors du domaine
/// physique du rapport et de `asin`).
pub fn spinning_max_half_angle_for_reduction(target_reduction_ratio: f64) -> f64 {
    assert!(
        target_reduction_ratio > 0.0 && target_reduction_ratio <= 1.0,
        "le rapport d'épaisseur cible r doit être dans (0, 1]"
    );
    target_reduction_ratio.asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn thickness_plus_reduction_equals_initial() {
        // Identité de conservation : t_f + Δt = t_0 pour tout α admissible.
        let t0 = 4.0e-3_f64;
        for &alpha in &[0.20_f64, 0.5, PI / 6.0, 1.0, 1.4]
        {
            let tf = spinning_sine_law_thickness(t0, alpha);
            let dt = spinning_thickness_reduction(t0, alpha);
            assert_relative_eq!(tf + dt, t0, max_relative = 1e-12);
        }
    }

    #[test]
    fn ratio_matches_thickness_over_initial() {
        // r = t_f / t_0 : cohérence entre le rapport et l'épaisseur finale.
        let t0 = 6.0e-3_f64;
        for &alpha in &[0.15_f64, 0.7, 1.1, PI / 2.0]
        {
            let tf = spinning_sine_law_thickness(t0, alpha);
            let r = spinning_reduction_ratio(alpha);
            assert_relative_eq!(r, tf / t0, max_relative = 1e-12);
        }
    }

    #[test]
    fn flat_cone_preserves_thickness() {
        // α = π/2 (paroi perpendiculaire à l'axe) : sin = 1, aucune réduction.
        let t0 = 3.0e-3_f64;
        assert_relative_eq!(
            spinning_sine_law_thickness(t0, PI / 2.0),
            t0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            spinning_thickness_reduction(t0, PI / 2.0),
            0.0,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            spinning_reduction_ratio(PI / 2.0),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn angle_and_ratio_are_reciprocal() {
        // Réciprocité : max_half_angle_for_reduction(reduction_ratio(α)) = α.
        for &alpha in &[0.10_f64, 0.4, 0.9, 1.3, PI / 2.0]
        {
            let r = spinning_reduction_ratio(alpha);
            assert_relative_eq!(
                spinning_max_half_angle_for_reduction(r),
                alpha,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn realistic_thirty_degree_cone_halves_thickness() {
        // Cône de demi-angle 30° : sin(30°) = 1/2 exactement.
        // Flan de 4 mm → paroi 2 mm, amincissement 2 mm, angle limite = 30°.
        let t0 = 4.0e-3_f64;
        let alpha = PI / 6.0;
        assert_relative_eq!(
            spinning_sine_law_thickness(t0, alpha),
            2.0e-3_f64,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            spinning_thickness_reduction(t0, alpha),
            2.0e-3_f64,
            max_relative = 1e-12
        );
        assert_relative_eq!(spinning_reduction_ratio(alpha), 0.5, max_relative = 1e-12);
        assert_relative_eq!(
            spinning_max_half_angle_for_reduction(0.5),
            alpha,
            max_relative = 1e-12
        );
    }

    #[test]
    fn smaller_angle_thins_more() {
        // Monotonie : sin croissant sur (0, π/2] ⇒ l'épaisseur finale croît avec
        // α, donc un cône plus pointu (α plus petit) amincit davantage.
        let t0 = 5.0e-3_f64;
        assert!(
            spinning_sine_law_thickness(t0, 0.3) < spinning_sine_law_thickness(t0, 0.9),
            "un demi-angle plus petit doit donner une paroi plus fine"
        );
    }

    #[test]
    #[should_panic(expected = "rapport d'épaisseur cible r doit être dans (0, 1]")]
    fn ratio_above_one_panics() {
        spinning_max_half_angle_for_reduction(1.2);
    }
}
