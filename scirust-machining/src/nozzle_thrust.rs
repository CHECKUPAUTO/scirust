//! Tuyère — débit massique, poussée idéale et vitesse d'éjection par Bernoulli.
//!
//! ```text
//! débit massique      ṁ = ρ·v·A
//! poussée idéale      F = ṁ·v_e
//! vitesse d'éjection  v_e = √(2·ΔP/ρ)
//! ```
//!
//! `ρ` masse volumique du fluide (kg/m³), `v` vitesse au col (m/s), `A` section
//! du col (m²), `ṁ` débit massique (kg/s), `v_e` vitesse d'éjection (m/s), `F`
//! poussée (N), `ΔP` chute de pression au travers de la tuyère (Pa).
//!
//! **Convention** : SI cohérent. **Limite honnête** : la vitesse d'éjection
//! suppose un écoulement **idéal incompressible** (Bernoulli, sans pertes) ; la
//! poussée est donnée à **pression de sortie adaptée** (le terme de pression
//! `(p_e − p_a)·A_e` est négligé). Toutes les propriétés (`ρ`, sections,
//! vitesses, `ΔP`) sont **fournies par l'appelant** — aucune valeur matériau ou
//! fluide n'est supposée par défaut.

/// Débit massique au col `ṁ = ρ·v·A` (kg/s).
///
/// Panique si `density < 0`, `throat_area < 0` ou `velocity < 0`.
pub fn nozzle_mass_flow(density: f64, velocity: f64, throat_area: f64) -> f64 {
    assert!(
        density >= 0.0,
        "la masse volumique doit être positive ou nulle"
    );
    assert!(velocity >= 0.0, "la vitesse doit être positive ou nulle");
    assert!(
        throat_area >= 0.0,
        "la section du col doit être positive ou nulle"
    );
    density * velocity * throat_area
}

/// Poussée idéale à pression de sortie adaptée `F = ṁ·v_e` (N).
///
/// Panique si `mass_flow < 0` ou `exit_velocity < 0`.
pub fn nozzle_thrust(mass_flow: f64, exit_velocity: f64) -> f64 {
    assert!(
        mass_flow >= 0.0,
        "le débit massique doit être positif ou nul"
    );
    assert!(
        exit_velocity >= 0.0,
        "la vitesse d'éjection doit être positive ou nulle"
    );
    mass_flow * exit_velocity
}

/// Vitesse d'éjection par Bernoulli incompressible `v_e = √(2·ΔP/ρ)` (m/s).
///
/// Panique si `pressure_drop < 0` ou `density <= 0`.
pub fn nozzle_exit_velocity_bernoulli(pressure_drop: f64, density: f64) -> f64 {
    assert!(
        pressure_drop >= 0.0,
        "la chute de pression doit être positive ou nulle"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    (2.0 * pressure_drop / density).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mass_flow_reference_case() {
        // ρ=1000 kg/m³, v=3 m/s, A=2e-3 m² → ṁ = 1000·3·2e-3 = 6 kg/s.
        assert_relative_eq!(nozzle_mass_flow(1000.0, 3.0, 2e-3), 6.0, epsilon = 1e-9);
    }

    #[test]
    fn mass_flow_is_proportional_to_area() {
        // Doubler la section double le débit (linéarité en A).
        let base = nozzle_mass_flow(1.225, 50.0, 1e-4);
        let doubled = nozzle_mass_flow(1.225, 50.0, 2e-4);
        assert_relative_eq!(doubled, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    fn thrust_reference_case() {
        // ṁ=6 kg/s, v_e=20 m/s → F = 120 N.
        assert_relative_eq!(nozzle_thrust(6.0, 20.0), 120.0, epsilon = 1e-9);
    }

    #[test]
    fn exit_velocity_reference_case() {
        // ΔP=20000 Pa, ρ=1000 kg/m³ → v_e = √(2·20000/1000) = √40 = 6,3246 m/s.
        assert_relative_eq!(
            nozzle_exit_velocity_bernoulli(20_000.0, 1000.0),
            40.0_f64.sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn bernoulli_inverts_dynamic_pressure() {
        // Réciprocité : ΔP = ½·ρ·v_e² doit redonner v_e via Bernoulli.
        let (rho, v_e) = (1.225_f64, 30.0_f64);
        let dp = 0.5 * rho * v_e * v_e;
        assert_relative_eq!(nozzle_exit_velocity_bernoulli(dp, rho), v_e, epsilon = 1e-9);
    }

    #[test]
    fn thrust_chained_from_flow_and_velocity() {
        // Chaîne cohérente : ṁ puis F, cas chiffré vérifié à la main.
        // ρ=1000, v=2, A=1e-3 → ṁ=2 kg/s. v_e=√(2·5000/1000)=√10.
        // F = 2·√10 ≈ 6,3246 N.
        let m_dot = nozzle_mass_flow(1000.0, 2.0, 1e-3);
        assert_relative_eq!(m_dot, 2.0, epsilon = 1e-12);
        let v_e = nozzle_exit_velocity_bernoulli(5000.0, 1000.0);
        assert_relative_eq!(
            nozzle_thrust(m_dot, v_e),
            2.0 * 10.0_f64.sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_density_panics() {
        nozzle_exit_velocity_bernoulli(1000.0, 0.0);
    }
}
