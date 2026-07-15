//! Sédimentation d'une sphère rigide isolée en régime de Stokes (écoulement
//! rampant, `Re < ~1`) : vitesse limite de chute, nombre de Reynolds et traînée.
//!
//! ```text
//! vitesse limite   v∞  = (ρp - ρf)·g·d² / (18·μ)
//! nombre Reynolds  Re  = ρf·v·d / μ                (Stokes valide si Re < ~1)
//! traînée Stokes   Fd  = 3·π·μ·d·v
//! ```
//!
//! `d` diamètre de la particule (m), `ρp` masse volumique de la particule
//! (kg/m³), `ρf` masse volumique du fluide (kg/m³), `μ` viscosité dynamique
//! (Pa·s), `g` pesanteur (m/s²), `v` vitesse relative sphère/fluide (m/s),
//! `v∞` vitesse limite de chute (m/s), `Fd` force de traînée (N). À la vitesse
//! limite, la traînée de Stokes équilibre exactement le poids déjaugé
//! `(ρp - ρf)·g·(π/6)·d³`.
//!
//! **Convention** : SI cohérent. **Limite honnête** : le régime de Stokes
//! suppose un écoulement laminaire rampant (`Re < ~1`), une sphère **rigide,
//! isolée** (ni interaction, ni paroi, ni concentration), et un fluide newtonien.
//! La pesanteur, les masses volumiques et la viscosité sont des **données**
//! fournies par l'appelant (tables du fluide/matériau, conditions du procédé) —
//! aucune valeur « par défaut » n'est inventée ici. Au-delà de `Re ≈ 1`, il faut
//! corréler avec un coefficient de traînée `Cd(Re)` (loi de Newton, abaques).

use core::f64::consts::PI;

/// Vitesse limite de chute de Stokes `v∞ = (ρp - ρf)·g·d²/(18·μ)` (m/s).
///
/// Positive si la particule est plus dense que le fluide (elle sédimente),
/// négative si elle est plus légère (elle remonte).
///
/// Panique si `particle_diameter <= 0`, `dynamic_viscosity <= 0` ou `gravity < 0`.
pub fn stokes_terminal_velocity(
    particle_diameter: f64,
    particle_density: f64,
    fluid_density: f64,
    dynamic_viscosity: f64,
    gravity: f64,
) -> f64 {
    assert!(
        particle_diameter > 0.0,
        "le diamètre de la particule doit être strictement positif"
    );
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    assert!(gravity >= 0.0, "la pesanteur ne peut pas être négative");
    (particle_density - fluid_density) * gravity * particle_diameter * particle_diameter
        / (18.0 * dynamic_viscosity)
}

/// Nombre de Reynolds particulaire `Re = ρf·v·d/μ` (sans dimension).
///
/// Le régime de Stokes n'est valide que si `Re < ~1` ; au-delà la traînée
/// n'est plus linéaire en `v`.
///
/// Panique si `diameter <= 0` ou `dynamic_viscosity <= 0`.
pub fn stokes_reynolds_number(
    velocity: f64,
    diameter: f64,
    fluid_density: f64,
    dynamic_viscosity: f64,
) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    fluid_density * velocity * diameter / dynamic_viscosity
}

/// Force de traînée de Stokes `Fd = 3·π·μ·d·v` (N).
///
/// Traînée linéaire en vitesse, valable seulement en écoulement rampant
/// (`Re < ~1`) autour d'une sphère rigide isolée.
///
/// Panique si `diameter <= 0` ou `dynamic_viscosity <= 0`.
pub fn stokes_drag_force(velocity: f64, diameter: f64, dynamic_viscosity: f64) -> f64 {
    assert!(diameter > 0.0, "le diamètre doit être strictement positif");
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    3.0 * PI * dynamic_viscosity * diameter * velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn terminal_velocity_matches_explicit_formula() {
        // Grain de silice (ρp=2650) de 100 µm dans l'eau (ρf=1000, μ=1e-3), g=9,81.
        let d = 100.0e-6_f64;
        let (rho_p, rho_f, mu, g) = (2650.0, 1000.0, 1.0e-3, 9.81);
        let v = stokes_terminal_velocity(d, rho_p, rho_f, mu, g);
        let expected = (rho_p - rho_f) * g * d * d / (18.0 * mu);
        assert_relative_eq!(v, expected, epsilon = 1e-12);
        // Valeur chiffrée attendue ≈ 8,9925e-3 m/s.
        assert_relative_eq!(v, 8.992_5e-3, max_relative = 1e-4);
    }

    #[test]
    fn terminal_velocity_scales_with_diameter_squared() {
        // v∞ ∝ d² : doubler le diamètre quadruple la vitesse limite.
        let base = stokes_terminal_velocity(50.0e-6, 2650.0, 1000.0, 1.0e-3, 9.81);
        let big = stokes_terminal_velocity(100.0e-6, 2650.0, 1000.0, 1.0e-3, 9.81);
        assert_relative_eq!(big / base, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn lighter_particle_rises() {
        // ρp < ρf → vitesse de sédimentation négative (remontée).
        let v = stokes_terminal_velocity(80.0e-6, 800.0, 1000.0, 1.0e-3, 9.81);
        assert!(
            v < 0.0,
            "une particule plus légère que le fluide doit remonter"
        );
    }

    #[test]
    fn drag_balances_submerged_weight_at_terminal_velocity() {
        // À v∞, la traînée de Stokes égale le poids déjaugé (ρp-ρf)·g·(π/6)·d³.
        let d = 100.0e-6_f64;
        let (rho_p, rho_f, mu, g) = (2650.0, 1000.0, 1.0e-3, 9.81);
        let v = stokes_terminal_velocity(d, rho_p, rho_f, mu, g);
        let drag = stokes_drag_force(v, d, mu);
        let submerged_weight = (rho_p - rho_f) * g * (PI / 6.0) * d * d * d;
        assert_relative_eq!(drag, submerged_weight, max_relative = 1e-9);
    }

    #[test]
    fn reynolds_stays_below_one_in_stokes_regime() {
        // Le cas chiffré ci-dessus doit bien vérifier Re < 1.
        let d = 100.0e-6_f64;
        let (rho_p, rho_f, mu, g) = (2650.0, 1000.0, 1.0e-3, 9.81);
        let v = stokes_terminal_velocity(d, rho_p, rho_f, mu, g);
        let re = stokes_reynolds_number(v, d, rho_f, mu);
        let expected = rho_f * v * d / mu;
        assert_relative_eq!(re, expected, epsilon = 1e-12);
        assert!(re < 1.0, "le régime de Stokes exige Re < ~1, ici Re = {re}");
    }

    #[test]
    fn drag_is_linear_in_velocity() {
        // Fd = 3·π·μ·d·v est linéaire en v : tripler v triple la traînée.
        let f1 = stokes_drag_force(0.01, 100.0e-6, 1.0e-3);
        let f3 = stokes_drag_force(0.03, 100.0e-6, 1.0e-3);
        assert_relative_eq!(f3 / f1, 3.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "diamètre")]
    fn zero_diameter_panics() {
        stokes_terminal_velocity(0.0, 2650.0, 1000.0, 1.0e-3, 9.81);
    }
}
