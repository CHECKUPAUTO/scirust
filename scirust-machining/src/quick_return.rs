//! Mécanisme à retour rapide (**quick return** : étau-limeur, Whitworth) —
//! rapports de temps aller/retour déduits des seuls angles de manivelle,
//! la manivelle tournant à vitesse angulaire constante.
//!
//! ```text
//! ω = cste                       (manivelle à vitesse angulaire constante)
//! temps balayé      t = θ / ω    (le temps est proportionnel à l'angle)
//! rapport de temps  r = t_a / t_b = θ_a / θ_b
//! retour rapide     R = t_coupe / t_retour = α / β
//! fraction de coupe f = α / θ_tot           (θ_tot = 2π sur un tour)
//! angle de retour   β = 2π − α              (sur un tour complet)
//! ```
//!
//! `α` (`cutting_stroke_angle_rad`) angle de manivelle de la course de coupe
//! (rad), `β` (`return_stroke_angle_rad`) angle de la course de retour (rad),
//! `θ_tot` (`total_angle_rad`) angle total d'un cycle (rad, un tour = 2π),
//! `R` rapport de retour rapide (sans dimension, ≥ 1 pour un mécanisme
//! réellement à retour rapide), `f` fraction du cycle passée en coupe
//! (sans dimension, ∈ ]0, 1[), `ω` vitesse angulaire (rad/s), `t` temps (s).
//! À `ω` constante, les rapports de temps se réduisent à des rapports d'angles :
//! [`quickreturn_ratio`] n'est que [`quickreturn_time_ratio_from_angles`]
//! appliqué à `(α, β)`, et [`quickreturn_return_stroke_angle`] est réciproque
//! de son propre complément à `2π`.
//!
//! **Convention** : SI cohérent (angles en radians, temps en secondes) ; les
//! angles sont ceux effectivement balayés par la manivelle et doivent être
//! strictement positifs.
//!
//! **Limite honnête** : modèle purement cinématique idéalisé — vitesse
//! angulaire de manivelle **constante**, donc temps strictement proportionnel
//! à l'angle balayé. Les angles d'aller `α` et de retour `β` résultent de la
//! **géométrie du mécanisme** (excentration, entraxes) et sont **fournis par
//! l'appelant** ; aucune loi de mouvement, aucune inertie ni aucune valeur
//! d'angle « par défaut » n'est supposée ici.

use core::f64::consts::PI;

/// Rapport des temps `r = t_a / t_b = θ_a / θ_b` de deux phases balayées à
/// vitesse angulaire constante, déduit de leurs angles de manivelle
/// `first_angle_rad` et `second_angle_rad` (le temps étant proportionnel à
/// l'angle). Grandeur sans dimension.
///
/// Panique si l'un des angles n'est pas strictement positif et fini.
pub fn quickreturn_time_ratio_from_angles(first_angle_rad: f64, second_angle_rad: f64) -> f64 {
    assert!(
        first_angle_rad.is_finite() && first_angle_rad > 0.0,
        "l'angle θ_a doit être strictement positif et fini"
    );
    assert!(
        second_angle_rad.is_finite() && second_angle_rad > 0.0,
        "l'angle θ_b doit être strictement positif et fini"
    );
    first_angle_rad / second_angle_rad
}

/// Rapport de retour rapide `R = t_coupe / t_retour = α / β` (sans dimension),
/// obtenu à `ω` constante depuis l'angle de coupe `cutting_stroke_angle_rad` et
/// l'angle de retour `return_stroke_angle_rad`. `R > 1` traduit un retour plus
/// rapide que la coupe.
///
/// Panique si l'un des angles n'est pas strictement positif et fini.
pub fn quickreturn_ratio(cutting_stroke_angle_rad: f64, return_stroke_angle_rad: f64) -> f64 {
    quickreturn_time_ratio_from_angles(cutting_stroke_angle_rad, return_stroke_angle_rad)
}

/// Fraction du cycle passée en coupe `f = α / θ_tot` (sans dimension), depuis
/// l'angle de coupe `cutting_stroke_angle_rad` et l'angle total du cycle
/// `total_angle_rad` (un tour complet vaut `2π`).
///
/// Panique si `total_angle_rad` n'est pas strictement positif et fini, si
/// `cutting_stroke_angle_rad` n'est pas strictement positif, ou s'il dépasse
/// `total_angle_rad` (fraction hors de ]0, 1]).
pub fn quickreturn_cutting_time_fraction(
    cutting_stroke_angle_rad: f64,
    total_angle_rad: f64,
) -> f64 {
    assert!(
        total_angle_rad.is_finite() && total_angle_rad > 0.0,
        "l'angle total θ_tot doit être strictement positif et fini"
    );
    assert!(
        cutting_stroke_angle_rad.is_finite() && cutting_stroke_angle_rad > 0.0,
        "l'angle de coupe α doit être strictement positif et fini"
    );
    assert!(
        cutting_stroke_angle_rad <= total_angle_rad,
        "l'angle de coupe α ne peut dépasser l'angle total θ_tot"
    );
    cutting_stroke_angle_rad / total_angle_rad
}

/// Angle de retour `β = 2π − α` (rad) complétant l'angle de coupe
/// `cutting_stroke_angle_rad` sur un tour complet de manivelle.
///
/// Panique si `cutting_stroke_angle_rad` n'est pas strictement positif et fini
/// ou s'il atteint ou dépasse `2π` (aucun angle ne resterait pour le retour).
pub fn quickreturn_return_stroke_angle(cutting_stroke_angle_rad: f64) -> f64 {
    assert!(
        cutting_stroke_angle_rad.is_finite() && cutting_stroke_angle_rad > 0.0,
        "l'angle de coupe α doit être strictement positif et fini"
    );
    assert!(
        cutting_stroke_angle_rad < 2.0 * PI,
        "l'angle de coupe α doit être strictement inférieur à 2π"
    );
    2.0 * PI - cutting_stroke_angle_rad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn time_ratio_is_reciprocal_when_angles_swap() {
        // r(θ_a, θ_b) · r(θ_b, θ_a) = 1 : échanger les angles inverse le rapport.
        let a = 3.9_f64;
        let b = 2.4_f64;
        let r = quickreturn_time_ratio_from_angles(a, b);
        let r_inv = quickreturn_time_ratio_from_angles(b, a);
        assert_relative_eq!(r * r_inv, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn ratio_matches_generic_time_ratio() {
        // quickreturn_ratio n'est que le rapport de temps appliqué à (α, β).
        let alpha = 220.0_f64.to_radians();
        let beta = 140.0_f64.to_radians();
        assert_relative_eq!(
            quickreturn_ratio(alpha, beta),
            quickreturn_time_ratio_from_angles(alpha, beta),
            max_relative = 1e-12
        );
    }

    #[test]
    fn return_angle_completes_full_turn() {
        // α + β = 2π : l'angle de retour complète bien le tour.
        let alpha = 1.7_f64;
        let beta = quickreturn_return_stroke_angle(alpha);
        assert_relative_eq!(alpha + beta, 2.0 * PI, max_relative = 1e-12);
    }

    #[test]
    fn cutting_and_return_fractions_sum_to_one() {
        // Sur un tour, f_coupe + f_retour = 1 (α + β = 2π).
        let alpha = 200.0_f64.to_radians();
        let beta = quickreturn_return_stroke_angle(alpha);
        let total = 2.0 * PI;
        let f_cut = quickreturn_cutting_time_fraction(alpha, total);
        let f_ret = quickreturn_cutting_time_fraction(beta, total);
        assert_relative_eq!(f_cut + f_ret, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn ratio_is_scale_invariant() {
        // R = α/β ne dépend que du rapport des angles : mise à l'échelle sans effet.
        let alpha = 2.2_f64;
        let beta = 1.4_f64;
        let k = 0.5_f64;
        assert_relative_eq!(
            quickreturn_ratio(alpha, beta),
            quickreturn_ratio(k * alpha, k * beta),
            max_relative = 1e-12
        );
    }

    #[test]
    fn realistic_whitworth_case() {
        // Étau-limeur : coupe sur 220° de manivelle, retour sur 140°.
        // R = 220/140 = 11/7 ≈ 1,571428…  ;  f = 220/360 = 11/18 ≈ 0,611111…
        let alpha = 220.0_f64.to_radians();
        let beta = 140.0_f64.to_radians();
        assert_relative_eq!(
            quickreturn_ratio(alpha, beta),
            11.0 / 7.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            quickreturn_cutting_time_fraction(alpha, 2.0 * PI),
            11.0 / 18.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "ne peut dépasser l'angle total")]
    fn cutting_angle_above_total_panics() {
        quickreturn_cutting_time_fraction(7.0, 2.0 * PI);
    }
}
