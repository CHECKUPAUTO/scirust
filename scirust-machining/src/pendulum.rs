//! **Pendule simple et pendule composé (pesant)** — période des petites
//! oscillations libres.
//!
//! ```text
//! pendule simple       T = 2·π·√(L/g)
//! pendule composé      T = 2·π·√(I/(m·g·d))
//! longueur synchrone   L_eq = I/(m·d)
//! rayon de giration    k = √(I/m)
//! ```
//!
//! `T` période d'oscillation (s), `L` longueur du pendule simple (m), `g`
//! accélération de la pesanteur (m/s²), `I` moment d'inertie par rapport à l'axe
//! de rotation (kg·m²), `m` masse du corps (kg), `d` distance du pivot au centre
//! de gravité (m), `L_eq` longueur du pendule simple synchrone du pendule composé
//! (m), `k` rayon de giration par rapport au centre de gravité ou à l'axe selon
//! `I` fourni (m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : théorie des **petites
//! oscillations** (sin θ ≈ θ, isochronisme), pivot **sans frottement**,
//! **amortissement de l'air négligé**. L'accélération de la pesanteur `g`, le
//! moment d'inertie `I`, la masse `m` et la géométrie `d` sont des **données de
//! l'appelant** : aucune valeur « par défaut » n'est supposée, `g` variant avec
//! le lieu et `I` dépendant de la répartition des masses.

use core::f64::consts::PI;

/// Période du **pendule simple** `T = 2·π·√(L/g)` (s).
///
/// Masse ponctuelle suspendue à un fil inextensible sans masse, dans le régime
/// des petites oscillations.
///
/// Panique si `length <= 0` ou `gravity <= 0`.
pub fn pendulum_simple_period(length: f64, gravity: f64) -> f64 {
    assert!(length > 0.0, "la longueur L doit être strictement positive");
    assert!(
        gravity > 0.0,
        "la pesanteur g doit être strictement positive"
    );
    2.0 * PI * (length / gravity).sqrt()
}

/// Période du **pendule composé (pesant)** `T = 2·π·√(I/(m·g·d))` (s).
///
/// Corps rigide oscillant autour d'un axe horizontal fixe ne passant pas par son
/// centre de gravité, dans le régime des petites oscillations.
///
/// Panique si `moment_of_inertia <= 0`, `mass <= 0`, `pivot_distance <= 0` ou
/// `gravity <= 0`.
pub fn pendulum_compound_period(
    moment_of_inertia: f64,
    mass: f64,
    pivot_distance: f64,
    gravity: f64,
) -> f64 {
    assert!(
        moment_of_inertia > 0.0,
        "le moment d'inertie I doit être strictement positif"
    );
    assert!(mass > 0.0, "la masse m doit être strictement positive");
    assert!(
        pivot_distance > 0.0,
        "la distance pivot–centre de gravité d doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "la pesanteur g doit être strictement positive"
    );
    2.0 * PI * (moment_of_inertia / (mass * gravity * pivot_distance)).sqrt()
}

/// Longueur du pendule simple **synchrone** `L_eq = I/(m·d)` (m).
///
/// Longueur d'un pendule simple ayant la même période que le pendule composé
/// donné : `pendulum_compound_period(...)` vaut alors
/// `pendulum_simple_period(L_eq, g)`.
///
/// Panique si `moment_of_inertia <= 0`, `mass <= 0` ou `pivot_distance <= 0`.
pub fn pendulum_equivalent_length(moment_of_inertia: f64, mass: f64, pivot_distance: f64) -> f64 {
    assert!(
        moment_of_inertia > 0.0,
        "le moment d'inertie I doit être strictement positif"
    );
    assert!(mass > 0.0, "la masse m doit être strictement positive");
    assert!(
        pivot_distance > 0.0,
        "la distance pivot–centre de gravité d doit être strictement positive"
    );
    moment_of_inertia / (mass * pivot_distance)
}

/// Rayon de giration `k = √(I/m)` (m).
///
/// Distance équivalente concentrant toute la masse `m` pour donner le moment
/// d'inertie `I` par rapport au même axe.
///
/// Panique si `moment_of_inertia < 0` ou `mass <= 0`.
pub fn pendulum_radius_of_gyration(moment_of_inertia: f64, mass: f64) -> f64 {
    assert!(
        moment_of_inertia >= 0.0,
        "le moment d'inertie I doit être positif"
    );
    assert!(mass > 0.0, "la masse m doit être strictement positive");
    (moment_of_inertia / mass).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn compound_matches_simple_of_equivalent_length() {
        // Identité de synchronisme : le pendule composé oscille comme un pendule
        // simple de longueur L_eq = I/(m·d).
        let i = 0.42_f64;
        let m = 3.0_f64;
        let d = 0.35_f64;
        let g = 9.81_f64;
        let l_eq = pendulum_equivalent_length(i, m, d);
        assert_relative_eq!(
            pendulum_compound_period(i, m, d, g),
            pendulum_simple_period(l_eq, g),
            epsilon = 1e-12
        );
    }

    #[test]
    fn equivalent_length_equals_gyration_squared_over_distance() {
        // L_eq = I/(m·d) = k²/d, avec k² = I/m.
        let i = 0.42_f64;
        let m = 3.0_f64;
        let d = 0.35_f64;
        let k = pendulum_radius_of_gyration(i, m);
        assert_relative_eq!(
            pendulum_equivalent_length(i, m, d),
            k * k / d,
            epsilon = 1e-12
        );
    }

    #[test]
    fn simple_period_scales_as_sqrt_length() {
        // T ∝ √L : quadrupler la longueur double la période.
        let g = 9.81_f64;
        let t1 = pendulum_simple_period(1.0, g);
        let t4 = pendulum_simple_period(4.0, g);
        assert_relative_eq!(t4, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn simple_pendulum_one_meter_realistic_case() {
        // Pendule d'une seconde (battant) approché : L = 1 m, g = 9,81 m/s².
        // T = 2·π·√(1/9,81) = 2·π·0,319275… = 2,006066 s.
        let t = pendulum_simple_period(1.0, 9.81);
        assert_relative_eq!(t, 2.006_066, epsilon = 1e-6);
    }

    #[test]
    fn uniform_rod_compound_period_realistic_case() {
        // Barre homogène pivotée à une extrémité : ℓ = 1 m, m = 2 kg.
        // I = (1/3)·m·ℓ² = 0,666667 kg·m², d = ℓ/2 = 0,5 m, g = 9,81 m/s².
        // T = 2·π·√(0,666667/(2·9,81·0,5)) = 2·π·√0,0679579 = 1,637947 s.
        let i = 2.0_f64 * 1.0_f64.powi(2) / 3.0;
        let t = pendulum_compound_period(i, 2.0, 0.5, 9.81);
        assert_relative_eq!(t, 1.637_946_6, epsilon = 1e-6);
        // Longueur synchrone d'une barre à l'extrémité : L_eq = (2/3)·ℓ.
        assert_relative_eq!(
            pendulum_equivalent_length(i, 2.0, 0.5),
            2.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn radius_of_gyration_exact_value() {
        // k = √(I/m) = √(2/8) = √0,25 = 0,5 m.
        assert_relative_eq!(pendulum_radius_of_gyration(2.0, 8.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la pesanteur g doit être strictement positive")]
    fn zero_gravity_panics() {
        pendulum_simple_period(1.0, 0.0);
    }
}
