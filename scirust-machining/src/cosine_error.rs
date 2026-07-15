//! Erreur cosinus (**cosine error**) — biais d'un instrument de mesure dont
//! l'axe de palpage est incliné de `θ` par rapport à la direction de la
//! grandeur mesurée.
//!
//! ```text
//! valeur vraie      v = m · cos(θ)           (projection de la lecture)
//! erreur cosinus    e = m · (1 − cos(θ)) = m − v   (excès de la lecture)
//! angle max toléré  θ_max = acos(1 − e/m)     (réciproque, pour e/m ∈ [0, 1])
//! ```
//!
//! `m` lecture de l'instrument (m, ou toute longueur), `v` valeur vraie
//! projetée (même unité), `θ` désalignement angulaire de l'axe de palpage
//! (rad), `e` erreur cosinus absolue (même unité que `m`). Par construction
//! `v + e = m`, et `cosine_error` est **réciproque** de `alignment_max_angle_for_error` :
//! `cosine_error(m, alignment_max_angle_for_error(m, e)) = e`.
//!
//! **Convention** : SI cohérent (longueurs dans une même unité, angles en rad) ;
//! seul le rapport `e/m` importe pour l'angle, donc `m` et `e` doivent partager
//! l'unité de longueur. On se restreint aux petits désalignements `θ ∈ [0, π/2]`,
//! où la lecture **surestime** toujours la grandeur (`e ≥ 0`).
//!
//! **Limite honnête** : modèle d'erreur du **premier ordre** dû au seul
//! désalignement angulaire, contact supposé **ponctuel** (ni jeu, ni flèche du
//! palpeur, ni erreur d'Abbe). Aucune tolérance, aucune valeur de lecture ni
//! aucun budget d'erreur n'est imposé : la lecture `m` et l'erreur admissible
//! `e` sont **fournies par l'appelant**.

use core::f64::consts::PI;

/// Valeur vraie `v = m · cos(θ)` (même unité que `measured_value`) obtenue en
/// projetant la lecture `measured_value` d'un instrument désaligné de
/// `misalignment_angle_rad`.
///
/// Panique si `misalignment_angle_rad` sort de `[0, π/2]`.
pub fn cosine_true_value_from_reading(measured_value: f64, misalignment_angle_rad: f64) -> f64 {
    assert!(
        (0.0..=PI / 2.0).contains(&misalignment_angle_rad),
        "le désalignement θ doit être compris dans [0, π/2] rad"
    );
    measured_value * misalignment_angle_rad.cos()
}

/// Erreur cosinus absolue `e = m · (1 − cos(θ))` (même unité que
/// `measured_value`) : excès de la lecture `measured_value` sur la valeur vraie,
/// dû au désalignement `misalignment_angle_rad`.
///
/// Panique si `misalignment_angle_rad` sort de `[0, π/2]`.
pub fn cosine_error(measured_value: f64, misalignment_angle_rad: f64) -> f64 {
    assert!(
        (0.0..=PI / 2.0).contains(&misalignment_angle_rad),
        "le désalignement θ doit être compris dans [0, π/2] rad"
    );
    measured_value * (1.0 - misalignment_angle_rad.cos())
}

/// Désalignement maximal `θ_max = acos(1 − e/m)` (rad) tolérable pour que
/// l'erreur cosinus reste sous `allowable_error` sur une lecture `measured_value`.
///
/// Panique si `measured_value <= 0`, si `allowable_error < 0` ou si
/// `allowable_error > measured_value` (rapport `e/m` hors de `[0, 1]`, argument
/// de `acos` hors domaine).
pub fn alignment_max_angle_for_error(measured_value: f64, allowable_error: f64) -> f64 {
    assert!(
        measured_value > 0.0,
        "la lecture m doit être strictement positive"
    );
    assert!(
        allowable_error >= 0.0,
        "l'erreur admissible e ne peut être négative"
    );
    assert!(
        allowable_error <= measured_value,
        "l'erreur admissible e ne peut dépasser la lecture m (rapport hors de acos)"
    );
    (1.0 - allowable_error / measured_value).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn true_value_and_error_sum_to_reading() {
        // Identité de décomposition : v + e = m pour tout désalignement.
        let m = 50.0_f64;
        for &theta in &[0.0_f64, 0.02, 0.1, 0.5, PI / 2.0]
        {
            let v = cosine_true_value_from_reading(m, theta);
            let e = cosine_error(m, theta);
            assert_relative_eq!(v + e, m, max_relative = 1e-12);
        }
    }

    #[test]
    fn perfect_alignment_has_no_error() {
        // θ = 0 : la lecture est exacte, erreur nulle et valeur vraie = lecture.
        assert_relative_eq!(cosine_error(120.0, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            cosine_true_value_from_reading(120.0, 0.0),
            120.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn error_is_proportional_to_reading() {
        // e = m·(1−cos θ) est linéaire en m : doubler la lecture double l'erreur.
        let theta = 0.15_f64;
        let e1 = cosine_error(10.0, theta);
        let e2 = cosine_error(20.0, theta);
        assert_relative_eq!(e2, 2.0 * e1, max_relative = 1e-12);
    }

    #[test]
    fn max_angle_is_reciprocal_of_error() {
        // Réciprocité : l'angle rendu par θ_max reproduit exactement l'erreur e.
        let m = 80.0_f64;
        for &e in &[0.0_f64, 0.001, 0.05, 5.0, 40.0]
        {
            let theta = alignment_max_angle_for_error(m, e);
            assert_relative_eq!(cosine_error(m, theta), e, max_relative = 1e-9);
        }
    }

    #[test]
    fn realistic_gauge_misalignment_case() {
        // Comparateur mesurant m = 25 mm avec un axe désaligné de 3°.
        // e = 25·(1 − cos 3°) = 25·(1 − 0,9986295…) ≈ 0,0342616 mm.
        let m = 25.0_f64;
        let theta = 3.0_f64.to_radians();
        let e = cosine_error(m, theta);
        assert_relative_eq!(e, m * (1.0 - theta.cos()), max_relative = 1e-12);
        assert_relative_eq!(e, 0.034_261_6, max_relative = 1e-5);
    }

    #[test]
    fn half_reading_error_gives_sixty_degrees() {
        // Cas limite chiffré : e = m/2 ⇒ 1 − e/m = 1/2 ⇒ θ_max = acos(1/2) = 60°.
        let theta = alignment_max_angle_for_error(30.0, 15.0);
        assert_relative_eq!(theta, PI / 3.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "ne peut dépasser la lecture m")]
    fn error_above_reading_panics() {
        alignment_max_angle_for_error(10.0, 12.0);
    }
}
