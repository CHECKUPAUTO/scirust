//! Thermique — dilatation, conduction (Fourier), convection, chaleur sensible
//! et contrainte d'origine thermique.
//!
//! ```text
//! dilatation linéaire   ΔL = α·L·ΔT
//! conduction (paroi)    Q = λ·A·ΔT/e        résistance R = e/(λ·A)
//! convection            Q = h·A·ΔT
//! chaleur sensible      Q = m·c·ΔT
//! contrainte thermique  σ = E·α·ΔT          (dilatation empêchée)
//! ```
//!
//! `α` coefficient de dilatation (1/K), `λ` conductivité (W/(m·K)), `h`
//! coefficient de convection (W/(m²·K)), `A` surface (m²), `e` épaisseur (m),
//! `m` masse (kg), `c` chaleur massique (J/(kg·K)), `E` module de Young (Pa).
//!
//! **Convention** : SI cohérent ; `ΔT` en K (ou °C, un écart de température est
//! identique). **Limite honnête** : régime **permanent** 1D, propriétés
//! constantes, paroi simple ; pas de régime transitoire, de rayonnement, ni de
//! contact/ailettes.

/// Allongement par dilatation linéaire `ΔL = α·L·ΔT`.
pub fn linear_expansion(alpha_per_k: f64, length: f64, delta_t_k: f64) -> f64 {
    alpha_per_k * length * delta_t_k
}

/// Flux de chaleur conductif à travers une paroi (Fourier) `Q = λ·A·ΔT/e` (W).
///
/// Panique si `thickness <= 0`.
pub fn conduction_heat_flow(
    conductivity_w_mk: f64,
    area_m2: f64,
    delta_t_k: f64,
    thickness_m: f64,
) -> f64 {
    assert!(
        thickness_m > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    conductivity_w_mk * area_m2 * delta_t_k / thickness_m
}

/// Résistance thermique de conduction d'une paroi `R = e/(λ·A)` (K/W).
///
/// Panique si `conductivity*area <= 0`.
pub fn thermal_resistance(conductivity_w_mk: f64, area_m2: f64, thickness_m: f64) -> f64 {
    assert!(
        conductivity_w_mk * area_m2 > 0.0,
        "conductivité et surface doivent être strictement positives"
    );
    thickness_m / (conductivity_w_mk * area_m2)
}

/// Flux de chaleur convectif `Q = h·A·ΔT` (W).
pub fn convection_heat_flow(h_w_m2k: f64, area_m2: f64, delta_t_k: f64) -> f64 {
    h_w_m2k * area_m2 * delta_t_k
}

/// Chaleur sensible `Q = m·c·ΔT` (J).
pub fn sensible_heat(mass_kg: f64, specific_heat_j_kgk: f64, delta_t_k: f64) -> f64 {
    mass_kg * specific_heat_j_kgk * delta_t_k
}

/// Contrainte thermique (dilatation totalement empêchée) `σ = E·α·ΔT` (Pa).
pub fn thermal_stress(youngs_modulus_pa: f64, alpha_per_k: f64, delta_t_k: f64) -> f64 {
    youngs_modulus_pa * alpha_per_k * delta_t_k
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn steel_bar_expansion() {
        // α=12e-6 /K, L=1 m, ΔT=100 K → ΔL = 1,2e-3 m = 1,2 mm.
        assert_relative_eq!(linear_expansion(12e-6, 1.0, 100.0), 1.2e-3, epsilon = 1e-12);
    }

    #[test]
    fn conduction_and_resistance_are_consistent() {
        // λ=0,04, A=10 m², ΔT=20 K, e=0,1 m → Q = 0,04·10·20/0,1 = 80 W.
        let q = conduction_heat_flow(0.04, 10.0, 20.0, 0.1);
        assert_relative_eq!(q, 80.0, epsilon = 1e-9);
        // Q = ΔT/R doit redonner le même flux.
        let r = thermal_resistance(0.04, 10.0, 0.1);
        assert_relative_eq!(20.0 / r, q, epsilon = 1e-9);
    }

    #[test]
    fn convection_and_sensible_heat() {
        // Q_conv = 25·2·15 = 750 W.
        assert_relative_eq!(convection_heat_flow(25.0, 2.0, 15.0), 750.0, epsilon = 1e-9);
        // Q = m·c·ΔT : 2 kg d'eau (c=4186), ΔT=10 → 83720 J.
        assert_relative_eq!(sensible_heat(2.0, 4186.0, 10.0), 83_720.0, epsilon = 1e-6);
    }

    #[test]
    fn thermal_stress_of_constrained_steel() {
        // E=210e9, α=12e-6, ΔT=50 → σ = 210e9·12e-6·50 = 126 MPa.
        assert_relative_eq!(thermal_stress(210e9, 12e-6, 50.0), 126e6, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "épaisseur")]
    fn zero_thickness_panics() {
        conduction_heat_flow(0.04, 10.0, 20.0, 0.0);
    }
}
