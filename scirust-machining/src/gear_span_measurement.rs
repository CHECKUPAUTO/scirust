//! Cote sur k dents (**base tangent length**, cote de Wildhaber) — mesure d'une
//! denture droite au pied à coulisse à becs plats enjambant plusieurs dents.
//!
//! ```text
//! fonction développante   inv(α) = tan(α) − α
//! cote sur k dents        W = m · cos(α) · ( π·(k − ½) + z · inv(α) )
//! ```
//!
//! `m` module réel de la denture (m ; SI, mais toute unité de longueur donne
//! `W` dans la même unité), `α` angle de pression de fonctionnement (rad),
//! `z` nombre total de dents de la roue, `k` = `span_teeth` nombre de dents
//! enjambées par les becs, `W` cote sur k dents (même unité que `m`). La
//! quantité `inv(α)` est la **fonction développante** (involute) qui relie
//! l'angle de pression à la géométrie du profil en développante de cercle.
//!
//! **Convention** : SI cohérent (module en m, angle en rad, cote en m) ; seul
//! le partage de la même unité de longueur entre `m` et `W` importe.
//! **Limite honnête** : formule valable pour une **denture droite normale à
//! profil en développante, NON déportée** (déport nul) et supposée parfaite ;
//! elle ignore l'épaisseur des becs, le jeu et les défauts. Le nombre de dents
//! `k` à enjamber doit être choisi pour que les becs portent sur les flancs
//! (typiquement `k ≈ z·α/π + 0,5`) : ce choix relève du métier. Aucun angle de
//! pression, module ni matériau « par défaut » n'est supposé : l'angle de
//! pression de fonctionnement est **fourni par l'appelant**.

use core::f64::consts::PI;

/// Fonction développante `inv(α) = tan(α) − α` (rad) associée à l'angle de
/// pression `pressure_angle_rad`.
///
/// Panique si `pressure_angle_rad` sort de `]0, π/2[` (tan divergent en π/2,
/// développante non définie hors de cette plage utile).
pub fn involute_function(pressure_angle_rad: f64) -> f64 {
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "l'angle de pression α doit être dans ]0, π/2[ rad"
    );
    pressure_angle_rad.tan() - pressure_angle_rad
}

/// Cote sur k dents `W = m · cos(α) · (π·(k − ½) + z · inv(α))` (même unité que
/// `module_length`) d'une denture droite non déportée à `teeth` dents mesurée
/// en enjambant `span_teeth` dents avec des becs plats.
///
/// Panique si `module_length <= 0`, si `pressure_angle_rad` sort de `]0, π/2[`,
/// si `teeth == 0` ou si `span_teeth == 0`.
pub fn base_tangent_length(
    module_length: f64,
    pressure_angle_rad: f64,
    teeth: u32,
    span_teeth: u32,
) -> f64 {
    assert!(
        module_length > 0.0,
        "le module m doit être strictement positif"
    );
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "l'angle de pression α doit être dans ]0, π/2[ rad"
    );
    assert!(teeth >= 1, "le nombre de dents z doit être au moins 1");
    assert!(
        span_teeth >= 1,
        "le nombre de dents enjambées k doit être au moins 1"
    );
    let inv = involute_function(pressure_angle_rad);
    module_length
        * pressure_angle_rad.cos()
        * (PI * (f64::from(span_teeth) - 0.5) + f64::from(teeth) * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn involute_of_twenty_degrees_is_standard_value() {
        // Valeur tabulée classique : inv(20°) ≈ 0,014904 rad.
        let inv = involute_function(20.0_f64.to_radians());
        assert_relative_eq!(inv, 0.014_904_4, max_relative = 1e-5);
    }

    #[test]
    fn involute_grows_with_pressure_angle() {
        // inv(α) = tan(α) − α est strictement croissante sur ]0, π/2[.
        let a = involute_function(15.0_f64.to_radians());
        let b = involute_function(20.0_f64.to_radians());
        let c = involute_function(25.0_f64.to_radians());
        assert!(a < b && b < c);
    }

    #[test]
    fn span_is_proportional_to_module() {
        // W est linéaire en m : doubler le module double la cote (α, z, k fixés).
        let alpha = 20.0_f64.to_radians();
        let w1 = base_tangent_length(1.0, alpha, 40, 5);
        let w2 = base_tangent_length(2.0, alpha, 40, 5);
        assert_relative_eq!(w2, 2.0 * w1, max_relative = 1e-12);
    }

    #[test]
    fn extra_span_tooth_adds_base_pitch() {
        // Enjamber une dent de plus ajoute exactement un pas de base
        // pb = π·m·cos(α), indépendamment de z (identité de la développante).
        let m = 3.0_f64;
        let alpha = 20.0_f64.to_radians();
        let base_pitch = PI * m * alpha.cos();
        let w5 = base_tangent_length(m, alpha, 30, 5);
        let w6 = base_tangent_length(m, alpha, 30, 6);
        assert_relative_eq!(w6 - w5, base_pitch, max_relative = 1e-12);
    }

    #[test]
    fn realistic_spur_gear_case() {
        // m = 2 mm, α = 20°, z = 40, k = 5 dents enjambées.
        // W = 2·cos20°·(π·4,5 + 40·inv20°) ≈ 27,690 mm.
        let m = 2.0_f64;
        let alpha = 20.0_f64.to_radians();
        let w = base_tangent_length(m, alpha, 40, 5);
        let inv = alpha.tan() - alpha;
        let expected = m * alpha.cos() * (PI * 4.5 + 40.0 * inv);
        assert_relative_eq!(w, expected, max_relative = 1e-12);
        assert_relative_eq!(w, 27.6896, max_relative = 1e-4);
    }

    #[test]
    #[should_panic(expected = "l'angle de pression α doit être dans")]
    fn involute_at_zero_panics() {
        involute_function(0.0);
    }
}
