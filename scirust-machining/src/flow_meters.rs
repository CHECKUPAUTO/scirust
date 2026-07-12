//! Débitmètres déprimogènes — plaque à **orifice**, **Venturi** et tuyère :
//! débit à partir de la perte de pression et facteur de vitesse d'approche.
//!
//! ```text
//! rapport de diamètres  β = d/D
//! débit                 Q = (Cd/√(1 − β⁴))·A_col·√(2·ΔP/ρ)
//! perte de pression     ΔP = ρ/2·[ Q·√(1 − β⁴)/(Cd·A_col) ]²
//! ```
//!
//! `β` rapport diamètre col/conduite, `Cd` coefficient de décharge (~0,6 orifice,
//! ~0,98 Venturi), `A_col` aire au col (m²), `ΔP` perte de pression mesurée (Pa),
//! `ρ` masse volumique (kg/m³), `Q` débit volumique (m³/s). Le facteur
//! `1/√(1−β⁴)` corrige la vitesse d'approche amont.
//!
//! **Convention** : SI cohérent. **Limite honnête** : fluide **incompressible**,
//! régime permanent ; `Cd` dépend du type d'élément, du nombre de Reynolds et de
//! la prise de pression (norme ISO 5167) — c'est une **donnée** de l'appelant.

/// Rapport de diamètres `β = d/D`.
///
/// Panique si `pipe_diameter <= 0` ou `throat_diameter >= pipe_diameter`.
pub fn beta_ratio(throat_diameter: f64, pipe_diameter: f64) -> f64 {
    assert!(
        pipe_diameter > 0.0 && throat_diameter < pipe_diameter,
        "0 < d < D requis"
    );
    throat_diameter / pipe_diameter
}

/// Débit volumique `Q = (Cd/√(1−β⁴))·A_col·√(2·ΔP/ρ)` (m³/s).
///
/// Panique si `ρ <= 0`, `β` hors `[0, 1[`, ou `ΔP < 0`.
pub fn flow_rate(discharge_coeff: f64, throat_area: f64, beta: f64, delta_p: f64, rho: f64) -> f64 {
    assert!(rho > 0.0 && delta_p >= 0.0, "ρ > 0 et ΔP ≥ 0 requis");
    assert!((0.0..1.0).contains(&beta), "β doit être dans [0, 1[");
    let approach = 1.0 / (1.0 - beta.powi(4)).sqrt();
    discharge_coeff * approach * throat_area * (2.0 * delta_p / rho).sqrt()
}

/// Perte de pression correspondant à un débit `Q` (Pa) — inverse de [`flow_rate`].
///
/// Panique si `Cd·A_col <= 0` ou `β` hors `[0, 1[`.
pub fn pressure_drop_for_flow(
    discharge_coeff: f64,
    throat_area: f64,
    beta: f64,
    flow_rate: f64,
    rho: f64,
) -> f64 {
    assert!(
        discharge_coeff * throat_area > 0.0,
        "Cd·A_col doit être strictement positif"
    );
    assert!((0.0..1.0).contains(&beta), "β doit être dans [0, 1[");
    let v = flow_rate * (1.0 - beta.powi(4)).sqrt() / (discharge_coeff * throat_area);
    0.5 * rho * v * v
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn beta_of_a_typical_orifice() {
        // d=50 mm dans D=100 mm → β = 0,5.
        assert_relative_eq!(beta_ratio(0.050, 0.100), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn flow_rises_with_root_pressure_drop() {
        // Q ∝ √ΔP : quadrupler ΔP double le débit.
        let q1 = flow_rate(0.61, 2e-3, 0.5, 1000.0, 1000.0);
        let q2 = flow_rate(0.61, 2e-3, 0.5, 4000.0, 1000.0);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn flow_and_pressure_drop_are_inverse() {
        // pressure_drop_for_flow doit redonner le ΔP initial.
        let (cd, a, beta, rho) = (0.61, 2e-3, 0.5, 1000.0);
        let q = flow_rate(cd, a, beta, 1500.0, rho);
        assert_relative_eq!(
            pressure_drop_for_flow(cd, a, beta, q, rho),
            1500.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn approach_factor_increases_flow_for_large_beta() {
        // Le facteur 1/√(1−β⁴) augmente le débit quand β grandit.
        let small = flow_rate(0.61, 2e-3, 0.2, 1000.0, 1000.0);
        let large = flow_rate(0.61, 2e-3, 0.7, 1000.0, 1000.0);
        assert!(large > small);
    }

    #[test]
    #[should_panic(expected = "0 < d < D")]
    fn throat_larger_than_pipe_panics() {
        beta_ratio(0.120, 0.100);
    }
}
