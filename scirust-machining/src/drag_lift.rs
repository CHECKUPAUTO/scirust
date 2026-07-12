//! Traînée et portance — efforts aérodynamiques/hydrodynamiques sur un corps,
//! puissance de traînée et vitesse limite de chute.
//!
//! ```text
//! traînée          Fd = ½·ρ·v²·Cd·A
//! portance         Fl = ½·ρ·v²·Cl·A
//! puissance        P  = Fd·v = ½·ρ·v³·Cd·A
//! vitesse limite   v∞ = √( 2·m·g / (ρ·Cd·A) )     (poids = traînée)
//! ```
//!
//! `ρ` masse volumique du fluide (kg/m³), `v` vitesse relative (m/s), `Cd`/`Cl`
//! coefficients de traînée/portance (sans dimension), `A` aire de référence (m²),
//! `m` masse du corps (kg), `g` pesanteur (m/s²). À la vitesse limite, la traînée
//! équilibre exactement le poids.
//!
//! **Convention** : SI cohérent. **Limite honnête** : `Cd`/`Cl` dépendent du
//! nombre de Reynolds et de la forme — ce sont des **données** fournies par
//! l'appelant (abaques, souffleries) ; l'aire de référence doit être cohérente
//! avec la définition de `Cd`. Régime incompressible établi.

/// Force de traînée `Fd = ½·ρ·v²·Cd·A` (N).
pub fn drag_force(rho: f64, velocity: f64, drag_coefficient: f64, area: f64) -> f64 {
    0.5 * rho * velocity * velocity * drag_coefficient * area
}

/// Force de portance `Fl = ½·ρ·v²·Cl·A` (N).
pub fn lift_force(rho: f64, velocity: f64, lift_coefficient: f64, area: f64) -> f64 {
    0.5 * rho * velocity * velocity * lift_coefficient * area
}

/// Puissance dissipée par la traînée `P = Fd·v` (W).
pub fn drag_power(rho: f64, velocity: f64, drag_coefficient: f64, area: f64) -> f64 {
    drag_force(rho, velocity, drag_coefficient, area) * velocity
}

/// Vitesse limite de chute `v∞ = √(2·m·g/(ρ·Cd·A))` (m/s).
///
/// Panique si `ρ·Cd·A <= 0`.
pub fn terminal_velocity(mass: f64, g: f64, rho: f64, drag_coefficient: f64, area: f64) -> f64 {
    let denom = rho * drag_coefficient * area;
    assert!(denom > 0.0, "ρ·Cd·A doit être strictement positif");
    (2.0 * mass * g / denom).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn drag_scales_with_velocity_squared() {
        // Doubler la vitesse quadruple la traînée.
        let f1 = drag_force(1.225, 10.0, 0.4, 2.0);
        let f2 = drag_force(1.225, 20.0, 0.4, 2.0);
        assert_relative_eq!(f2 / f1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn lift_definition() {
        // ρ=1,225, v=50, Cl=1,2, A=15 → Fl = 0,5·1,225·2500·1,2·15.
        assert_relative_eq!(
            lift_force(1.225, 50.0, 1.2, 15.0),
            0.5 * 1.225 * 2500.0 * 1.2 * 15.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn power_grows_with_cube_of_velocity() {
        // P = Fd·v ∝ v³ : doubler v multiplie la puissance par 8.
        let p1 = drag_power(1.225, 10.0, 0.4, 2.0);
        let p2 = drag_power(1.225, 20.0, 0.4, 2.0);
        assert_relative_eq!(p2 / p1, 8.0, epsilon = 1e-9);
    }

    #[test]
    fn terminal_velocity_balances_weight() {
        // À v∞, la traînée doit égaler le poids m·g.
        let (m, g, rho, cd, a) = (80.0, 9.81, 1.225, 1.0, 0.7);
        let vt = terminal_velocity(m, g, rho, cd, a);
        assert_relative_eq!(drag_force(rho, vt, cd, a), m * g, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ρ·Cd·A")]
    fn zero_area_terminal_panics() {
        terminal_velocity(80.0, 9.81, 1.225, 1.0, 0.0);
    }
}
