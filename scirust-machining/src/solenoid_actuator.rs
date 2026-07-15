//! Actionneur électromagnétique à **noyau plongeur** (solénoïde) : force de
//! **réluctance** attirant le noyau dans l'entrefer d'un circuit magnétique.
//!
//! ```text
//! force d'attraction   F   = µ0·(N·I)²·A/(2·g²)   (N)
//! induction entrefer   B   = µ0·(N·I)/g           (T)
//! ampères-tours requis NI  = g·sqrt(2·F/(µ0·A))   (A, réciproque de F)
//! inductance (entrefer)L   = µ0·N²·A/g            (H)
//! ```
//!
//! `N·I` ampères-tours (force magnétomotrice, A), `A` aire de pôle en regard du
//! noyau (m²), `g` longueur de l'entrefer (m), `µ0` perméabilité du vide
//! (H·m⁻¹ = T·m·A⁻¹), `F` force axiale d'attraction du noyau (N), `B` induction
//! dans l'entrefer (T), `L` inductance de la bobine (H), `N` nombre de spires.
//!
//! **Convention** : unités SI strictes.
//! **Limite honnête** : circuit magnétique **dominé par l'entrefer** — la
//! réluctance du fer et la **saturation** sont négligées, le flux est supposé
//! **uniforme** dans l'entrefer, sans fuites ; l'aire de pôle `A`, l'entrefer
//! `g`, les ampères-tours `N·I` et la perméabilité `µ0` sont **fournis par
//! l'appelant** — aucune valeur « par défaut » n'est inventée. La force croît en
//! `1/g²` (fortement **non linéaire**), la force réelle est donc inférieure à
//! ces bornes idéales. Modèle exprimé en `N·I` et `g` ; distinct de
//! [`crate::magnetic_holding`], exprimé en induction `B`.

/// Perméabilité magnétique du vide `µ0` (H·m⁻¹), valeur physique de référence.
pub const SOLENOID_MU0: f64 = 1.256_637_061_4e-6;

/// Force axiale d'attraction du noyau `F = µ0·(N·I)²·A/(2·g²)` (N).
///
/// Panique si `ampere_turns < 0`, `pole_area <= 0`, `air_gap <= 0`
/// ou `permeability_vacuum <= 0`.
pub fn solenoid_force(
    ampere_turns: f64,
    pole_area: f64,
    air_gap: f64,
    permeability_vacuum: f64,
) -> f64 {
    assert!(ampere_turns >= 0.0, "N·I ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(air_gap > 0.0, "g > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    permeability_vacuum * ampere_turns * ampere_turns * pole_area / (2.0 * air_gap * air_gap)
}

/// Induction magnétique dans l'entrefer `B = µ0·(N·I)/g` (T),
/// circuit fer idéal (réluctance du fer négligée).
///
/// Panique si `ampere_turns < 0`, `air_gap <= 0` ou `permeability_vacuum <= 0`.
pub fn solenoid_flux_density(ampere_turns: f64, air_gap: f64, permeability_vacuum: f64) -> f64 {
    assert!(ampere_turns >= 0.0, "N·I ≥ 0 requis");
    assert!(air_gap > 0.0, "g > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    permeability_vacuum * ampere_turns / air_gap
}

/// Ampères-tours requis pour une force cible `N·I = g·sqrt(2·F/(µ0·A))` (A) —
/// réciproque de [`solenoid_force`].
///
/// Panique si `target_force < 0`, `pole_area <= 0`, `air_gap <= 0`
/// ou `permeability_vacuum <= 0`.
pub fn solenoid_ampere_turns_for_force(
    target_force: f64,
    pole_area: f64,
    air_gap: f64,
    permeability_vacuum: f64,
) -> f64 {
    assert!(target_force >= 0.0, "F ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(air_gap > 0.0, "g > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    air_gap * (2.0_f64 * target_force / (permeability_vacuum * pole_area)).sqrt()
}

/// Inductance de la bobine dominée par l'entrefer `L = µ0·N²·A/g` (H).
///
/// Panique si `turns < 0`, `pole_area <= 0`, `air_gap <= 0`
/// ou `permeability_vacuum <= 0`.
pub fn solenoid_inductance(
    turns: f64,
    pole_area: f64,
    air_gap: f64,
    permeability_vacuum: f64,
) -> f64 {
    assert!(turns >= 0.0, "N ≥ 0 requis");
    assert!(pole_area > 0.0, "A > 0 requis");
    assert!(air_gap > 0.0, "g > 0 requis");
    assert!(permeability_vacuum > 0.0, "µ0 > 0 requis");
    permeability_vacuum * turns * turns * pole_area / air_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn force_realistic_case() {
        // Cas chiffré : N·I = 1000 A, A = 0.001 m² (10 cm²), g = 0.002 m (2 mm).
        // F = µ0·1000²·0.001/(2·0.002²) = µ0·1.25e8 = 4π·1e-7·1.25e8 = 50·π.
        let f = solenoid_force(1000.0, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(f, 157.079_632_7, max_relative = 1e-6);
    }

    #[test]
    fn ampere_turns_is_reciprocal_of_force() {
        // NI → F → NI : réciprocité exacte des deux formules.
        let ni = 1000.0;
        let f = solenoid_force(ni, 0.001, 0.002, SOLENOID_MU0);
        let ni_back = solenoid_ampere_turns_for_force(f, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(ni_back, ni, max_relative = 1e-12);
    }

    #[test]
    fn force_inverse_square_in_gap() {
        // F ∝ 1/g² : diviser l'entrefer par deux quadruple la force.
        let f_large = solenoid_force(800.0, 0.001, 0.004, SOLENOID_MU0);
        let f_small = solenoid_force(800.0, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(f_small / f_large, 4.0, max_relative = 1e-12);
    }

    #[test]
    fn force_quadratic_in_ampere_turns() {
        // F ∝ (N·I)² : doubler les ampères-tours quadruple la force.
        let f1 = solenoid_force(500.0, 0.001, 0.002, SOLENOID_MU0);
        let f2 = solenoid_force(1000.0, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(f2 / f1, 4.0, max_relative = 1e-12);
    }

    #[test]
    fn force_matches_maxwell_pull_via_flux_density() {
        // Identité : F = B²·A/(2·µ0) avec B = µ0·(N·I)/g (traction de Maxwell).
        let b = solenoid_flux_density(1000.0, 0.002, SOLENOID_MU0);
        let f_maxwell = b * b * 0.001 / (2.0 * SOLENOID_MU0);
        let f_direct = solenoid_force(1000.0, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(f_direct, f_maxwell, max_relative = 1e-12);
    }

    #[test]
    fn inductance_quadratic_in_turns() {
        // L ∝ N² : doubler le nombre de spires quadruple l'inductance.
        let l1 = solenoid_inductance(500.0, 0.001, 0.002, SOLENOID_MU0);
        let l2 = solenoid_inductance(1000.0, 0.001, 0.002, SOLENOID_MU0);
        assert_relative_eq!(l2 / l1, 4.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "g > 0")]
    fn zero_air_gap_panics() {
        solenoid_force(1000.0, 0.001, 0.0, SOLENOID_MU0);
    }
}
