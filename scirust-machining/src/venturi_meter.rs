//! Débitmètre à **Venturi** : débit volumique déduit de la dépression au col
//! par l'équation de Bernoulli couplée à la continuité (fluide incompressible).
//!
//! ```text
//! rapport de diamètres  β  = d_col / d_conduite
//! débit volumique       Q  = Cd·A_col·√( 2·ΔP / (ρ·(1 − β⁴)) )
//! perte de pression     ΔP = ρ·(1 − β⁴)/2 · ( Q / (Cd·A_col) )²
//! ```
//!
//! `β` rapport diamètre col/conduite (sans dimension, `< 1`), `Cd` coefficient de
//! décharge (sans dimension), `A_col` aire de la section au col (m²), `ΔP` perte de
//! pression amont→col (Pa), `ρ` masse volumique du fluide (kg/m³), `Q` débit
//! volumique (m³/s). Le facteur `1/√(1 − β⁴)` corrige la vitesse d'approche amont.
//!
//! **Convention** : SI cohérent (m, m², Pa, kg/m³, m³/s). **Limite honnête** :
//! écoulement **incompressible** et **permanent**, compressibilité négligée ; le
//! coefficient de décharge `Cd` provient de l'**étalonnage** ou de la **norme**
//! (ISO 5167) — c'est une **donnée** de l'appelant, jamais une valeur inventée.

/// Rapport de diamètres du Venturi `β = d_col / d_conduite` (sans dimension).
///
/// Panique si `pipe_diameter <= 0` ou si `throat_diameter` n'est pas dans `]0, pipe_diameter[`.
pub fn venturi_beta_ratio(throat_diameter: f64, pipe_diameter: f64) -> f64 {
    assert!(pipe_diameter > 0.0, "le diamètre de conduite doit être > 0");
    assert!(
        throat_diameter > 0.0 && throat_diameter < pipe_diameter,
        "0 < d_col < d_conduite requis"
    );
    throat_diameter / pipe_diameter
}

/// Débit volumique `Q = Cd·A_col·√( 2·ΔP / (ρ·(1 − β⁴)) )` (m³/s).
///
/// Panique si `ρ <= 0`, `throat_area < 0`, `pressure_drop < 0`, ou `beta_ratio` hors `[0, 1[`.
pub fn venturi_flow_rate(
    discharge_coefficient: f64,
    throat_area: f64,
    beta_ratio: f64,
    pressure_drop: f64,
    density: f64,
) -> f64 {
    assert!(density > 0.0, "la masse volumique doit être > 0");
    assert!(throat_area >= 0.0, "l'aire au col doit être ≥ 0");
    assert!(pressure_drop >= 0.0, "la perte de pression doit être ≥ 0");
    assert!((0.0..1.0).contains(&beta_ratio), "β doit être dans [0, 1[");
    let denom = density * (1.0 - beta_ratio.powi(4));
    discharge_coefficient * throat_area * (2.0 * pressure_drop / denom).sqrt()
}

/// Perte de pression `ΔP = ρ·(1 − β⁴)/2 · ( Q / (Cd·A_col) )²` (Pa) — inverse de [`venturi_flow_rate`].
///
/// Panique si `density < 0`, `Cd·A_col <= 0`, ou `beta_ratio` hors `[0, 1[`.
pub fn venturi_pressure_drop(
    flow_rate: f64,
    discharge_coefficient: f64,
    throat_area: f64,
    beta_ratio: f64,
    density: f64,
) -> f64 {
    assert!(density >= 0.0, "la masse volumique doit être ≥ 0");
    assert!(
        discharge_coefficient * throat_area > 0.0,
        "Cd·A_col doit être strictement positif"
    );
    assert!((0.0..1.0).contains(&beta_ratio), "β doit être dans [0, 1[");
    let velocity_head = flow_rate / (discharge_coefficient * throat_area);
    0.5 * density * (1.0 - beta_ratio.powi(4)) * velocity_head * velocity_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn beta_of_a_standard_venturi() {
        // Col d_col = 50 mm dans une conduite d_conduite = 100 mm → β = 0,5.
        assert_relative_eq!(venturi_beta_ratio(0.050, 0.100), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn flow_rises_with_root_pressure_drop() {
        // Q ∝ √ΔP : quadrupler ΔP double le débit.
        let q1 = venturi_flow_rate(0.98, 2e-3, 0.5, 3000.0, 1000.0);
        let q2 = venturi_flow_rate(0.98, 2e-3, 0.5, 12000.0, 1000.0);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn flow_and_pressure_drop_are_inverse() {
        // venturi_pressure_drop doit redonner le ΔP initial.
        let (cd, area, beta, rho) = (0.98, 2e-3, 0.5, 1000.0);
        let q = venturi_flow_rate(cd, area, beta, 5000.0, rho);
        assert_relative_eq!(
            venturi_pressure_drop(q, cd, area, beta, rho),
            5000.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn realistic_water_venturi_flow() {
        // Eau (ρ = 1000 kg/m³), Cd = 0,98, col d = 50 mm dans D = 100 mm (β = 0,5),
        // ΔP = 5000 Pa.
        //   A_col = π/4·0,05² = 1,963495408e-3 m²
        //   1 − β⁴ = 1 − 0,0625 = 0,9375
        //   Q = 0,98·A_col·√(2·5000 / (1000·0,9375))
        //     = 0,98·1,963495408e-3·√(10,66666667)
        //     = 0,98·1,963495408e-3·3,265986324 ≈ 6,284494e-3 m³/s
        let area = PI / 4.0 * 0.05_f64.powi(2);
        let beta = venturi_beta_ratio(0.050, 0.100);
        let q = venturi_flow_rate(0.98, area, beta, 5000.0, 1000.0);
        assert_relative_eq!(q, 6.284494e-3, max_relative = 1e-5);
    }

    #[test]
    fn flow_scales_linearly_with_throat_area() {
        // À ΔP, ρ, β, Cd fixés, Q ∝ A_col.
        let q1 = venturi_flow_rate(0.98, 1e-3, 0.4, 4000.0, 998.0);
        let q2 = venturi_flow_rate(0.98, 3e-3, 0.4, 4000.0, 998.0);
        assert_relative_eq!(q2 / q1, 3.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "0 < d_col < d_conduite")]
    fn throat_larger_than_pipe_panics() {
        venturi_beta_ratio(0.120, 0.100);
    }
}
