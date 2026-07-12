//! Réseaux de **résistances thermiques** — analogie électrique : résistances de
//! convection, associations série/parallèle et coefficient global d'échange.
//!
//! ```text
//! résistance convection  R = 1/(h·A)
//! série                  R_tot = Σ Ri
//! parallèle              1/R_tot = Σ 1/Ri
//! flux                   Q = ΔT/R_tot
//! coeff. global          U = 1/(R_tot·A)
//! ```
//!
//! `h` coefficient de convection (W/(m²·K)), `A` aire (m²), `Ri` résistances
//! thermiques (K/W), `ΔT` écart de température global (K), `U` coefficient global
//! d'échange (W/(m²·K)). Les couches (paroi, convection, contact) se composent en
//! **série** ; les chemins parallèles (ailettes + surface nue) en **parallèle**.
//!
//! **Convention** : SI cohérent. **Limite honnête** : régime **permanent**, 1D,
//! propriétés constantes. Les résistances de **conduction** de paroi
//! (`e/(λ·A)`) sont fournies par [`crate::thermal::thermal_resistance`] ; ce
//! module assemble le réseau et en tire le flux et le coefficient global.

/// Résistance thermique de **convection** `R = 1/(h·A)` (K/W).
///
/// Panique si `h·A <= 0`.
pub fn convection_resistance(h: f64, area: f64) -> f64 {
    assert!(h * area > 0.0, "h·A doit être strictement positif");
    1.0 / (h * area)
}

/// Résistance équivalente d'associations en **série** `R_tot = Σ Ri` (K/W).
pub fn series_resistance(resistances: &[f64]) -> f64 {
    resistances.iter().sum()
}

/// Résistance équivalente d'associations en **parallèle** `1/R_tot = Σ 1/Ri` (K/W).
///
/// Panique si la liste est vide ou contient une résistance `<= 0`.
pub fn parallel_resistance(resistances: &[f64]) -> f64 {
    assert!(
        !resistances.is_empty(),
        "au moins une résistance est requise"
    );
    let sum_inv: f64 = resistances
        .iter()
        .map(|&r| {
            assert!(r > 0.0, "chaque résistance doit être strictement positive");
            1.0 / r
        })
        .sum();
    1.0 / sum_inv
}

/// Flux à travers un réseau `Q = ΔT/R_tot` (W).
///
/// Panique si `total_resistance <= 0`.
pub fn heat_flow(delta_t: f64, total_resistance: f64) -> f64 {
    assert!(
        total_resistance > 0.0,
        "la résistance totale doit être strictement positive"
    );
    delta_t / total_resistance
}

/// Coefficient global d'échange `U = 1/(R_tot·A)` (W/(m²·K)).
///
/// Panique si `total_resistance·area <= 0`.
pub fn overall_heat_transfer_coefficient(total_resistance: f64, area: f64) -> f64 {
    assert!(
        total_resistance * area > 0.0,
        "R_tot·A doit être strictement positif"
    );
    1.0 / (total_resistance * area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn convection_resistance_definition() {
        // h=25, A=2 → R = 1/50 = 0,02 K/W.
        assert_relative_eq!(convection_resistance(25.0, 2.0), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn series_adds_parallel_reduces() {
        // Série : R = 0,02+0,1+0,02 = 0,14.
        assert_relative_eq!(series_resistance(&[0.02, 0.1, 0.02]), 0.14, epsilon = 1e-12);
        // Parallèle de deux chemins égaux (0,1) → 0,05, sous le plus petit.
        let rp = parallel_resistance(&[0.1, 0.1]);
        assert_relative_eq!(rp, 0.05, epsilon = 1e-12);
        assert!(rp < 0.1);
    }

    #[test]
    fn heat_flow_and_u_are_consistent() {
        // Mur composite : ΔT=30 K, R_tot=0,14 K/W → Q ≈ 214 W.
        let rtot = series_resistance(&[0.02, 0.1, 0.02]);
        let q = heat_flow(30.0, rtot);
        assert_relative_eq!(q, 30.0 / 0.14, epsilon = 1e-9);
        // U·A·ΔT doit redonner le même flux.
        let area = 2.0;
        let u = overall_heat_transfer_coefficient(rtot, area);
        assert_relative_eq!(u * area * 30.0, q, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins une résistance")]
    fn empty_parallel_panics() {
        parallel_resistance(&[]);
    }
}
