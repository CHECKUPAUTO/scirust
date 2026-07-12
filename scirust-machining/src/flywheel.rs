//! Volant d'inertie — régularisation de la vitesse d'une machine alternative :
//! coefficient de fluctuation, énergie à emmagasiner et inertie requise.
//!
//! ```text
//! vitesse moyenne         ω_moy = (ω_max + ω_min)/2
//! coeff. de fluctuation   Cs = (ω_max − ω_min)/ω_moy
//! fluctuation d'énergie    ΔE = I·ω_moy²·Cs
//! inertie requise         I = ΔE/(ω_moy²·Cs)
//! coeff. de fluctuation d'énergie  Ce = ΔE/E_travail
//! ```
//!
//! `ω_max, ω_min` vitesses extrêmes sur un cycle (rad/s), `Cs` coefficient de
//! fluctuation de vitesse (donnée de conception : ~0,002 pour un alternateur,
//! ~0,2 pour une pompe), `ΔE` variation d'énergie cinétique max sur le cycle (J),
//! `I` moment d'inertie du volant (kg·m²).
//!
//! **Convention** : SI cohérent. **Limite honnête** : bilan d'énergie cinétique
//! sur un cycle en régime établi, vitesse moyenne constante ; `Cs` et le
//! diagramme couple-angle (d'où `ΔE`) sont des données du problème fournies par
//! l'appelant, non calculées ici.

/// Vitesse moyenne `ω_moy = (ω_max + ω_min)/2` (rad/s).
pub fn mean_speed(omega_max: f64, omega_min: f64) -> f64 {
    (omega_max + omega_min) / 2.0
}

/// Coefficient de fluctuation de vitesse `Cs = (ω_max − ω_min)/ω_moy`.
///
/// Panique si `ω_moy <= 0`.
pub fn coefficient_of_fluctuation(omega_max: f64, omega_min: f64) -> f64 {
    let mean = mean_speed(omega_max, omega_min);
    assert!(
        mean > 0.0,
        "la vitesse moyenne doit être strictement positive"
    );
    (omega_max - omega_min) / mean
}

/// Fluctuation d'énergie cinétique `ΔE = I·ω_moy²·Cs` (J).
pub fn energy_fluctuation(inertia_kg_m2: f64, mean_speed_rad_s: f64, cs: f64) -> f64 {
    inertia_kg_m2 * mean_speed_rad_s * mean_speed_rad_s * cs
}

/// Moment d'inertie requis `I = ΔE/(ω_moy²·Cs)` (kg·m²) pour tenir un `Cs` donné.
///
/// Panique si `ω_moy² · Cs <= 0`.
pub fn required_inertia(energy_fluctuation_j: f64, mean_speed_rad_s: f64, cs: f64) -> f64 {
    let denom = mean_speed_rad_s * mean_speed_rad_s * cs;
    assert!(denom > 0.0, "ω_moy² · Cs doit être strictement positif");
    energy_fluctuation_j / denom
}

/// Énergie cinétique de rotation stockée `E = ½·I·ω²` (J).
pub fn stored_energy(inertia_kg_m2: f64, omega_rad_s: f64) -> f64 {
    0.5 * inertia_kg_m2 * omega_rad_s * omega_rad_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fluctuation_coefficient_and_mean() {
        // ω_max=105, ω_min=95 → ω_moy=100, Cs=10/100=0,1.
        assert_relative_eq!(mean_speed(105.0, 95.0), 100.0, epsilon = 1e-12);
        assert_relative_eq!(
            coefficient_of_fluctuation(105.0, 95.0),
            0.1,
            epsilon = 1e-12
        );
    }

    #[test]
    fn energy_and_inertia_are_inverse() {
        // ΔE = I·ω²·Cs et I = ΔE/(ω²·Cs) doivent se composer en identité.
        let (i, wm, cs) = (2.0, 100.0, 0.05);
        let de = energy_fluctuation(i, wm, cs);
        assert_relative_eq!(required_inertia(de, wm, cs), i, epsilon = 1e-9);
    }

    #[test]
    fn design_a_flywheel() {
        // ΔE=1000 J à ω_moy=100 rad/s, Cs=0,02 → I = 1000/(10000·0,02) = 5 kg·m².
        assert_relative_eq!(required_inertia(1000.0, 100.0, 0.02), 5.0, epsilon = 1e-9);
    }

    #[test]
    fn stored_energy_is_half_i_omega_squared() {
        // I=5, ω=100 → E = 0,5·5·10000 = 25000 J.
        assert_relative_eq!(stored_energy(5.0, 100.0), 25_000.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "ω_moy² · Cs")]
    fn zero_cs_panics() {
        required_inertia(1000.0, 100.0, 0.0);
    }
}
