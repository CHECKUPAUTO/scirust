//! Pompes centrifuges — puissance hydraulique et absorbée, **NPSH disponible**,
//! lois de similitude (affinité) et vitesse spécifique.
//!
//! ```text
//! puissance hydraulique   P_h = ρ·g·Q·H
//! puissance absorbée      P_a = P_h/η
//! NPSH disponible (m)     NPSHd = (p_abs − p_vap)/(ρ·g) + v²/(2g)
//! lois d'affinité         Q₂/Q₁ = N₂/N₁   H₂/H₁ = (N₂/N₁)²   P₂/P₁ = (N₂/N₁)³
//! vitesse spécifique      Ns = N·√Q / H^{3/4}
//! ```
//!
//! `ρ` masse volumique (kg/m³), `g` pesanteur (m/s²), `Q` débit (m³/s), `H`
//! hauteur manométrique (m), `η` rendement, `p_abs` pression absolue à
//! l'aspiration, `p_vap` pression de vapeur saturante, `N` vitesse de rotation.
//! Le NPSH disponible doit rester supérieur au NPSH requis de la pompe pour
//! éviter la cavitation.
//!
//! **Convention** : SI cohérent (sauf `Ns` dont les unités suivent celles de
//! l'appelant). **Limite honnête** : pompe hydraulique idéalisée, similitude
//! valable à rendement constant sur une plage limitée ; le NPSH requis et la
//! courbe caractéristique sont des données du constructeur, non calculées ici.

/// Puissance hydraulique `P_h = ρ·g·Q·H` (W).
pub fn hydraulic_power(rho: f64, g: f64, flow_m3_s: f64, head_m: f64) -> f64 {
    rho * g * flow_m3_s * head_m
}

/// Puissance mécanique absorbée `P_a = P_h/η` (W).
///
/// Panique si `efficiency` n'est pas dans `]0, 1]`.
pub fn shaft_power(hydraulic_power_w: f64, efficiency: f64) -> f64 {
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans ]0, 1]"
    );
    hydraulic_power_w / efficiency
}

/// NPSH disponible `NPSHd = (p_abs − p_vap)/(ρ·g) + v²/(2g)` (m).
///
/// Panique si `ρ·g <= 0`.
pub fn npsh_available(
    inlet_abs_pressure: f64,
    vapor_pressure: f64,
    rho: f64,
    g: f64,
    inlet_velocity: f64,
) -> f64 {
    assert!(rho * g > 0.0, "ρ·g doit être strictement positif");
    (inlet_abs_pressure - vapor_pressure) / (rho * g) + inlet_velocity * inlet_velocity / (2.0 * g)
}

/// Débit après changement de vitesse (affinité) `Q₂ = Q₁·N₂/N₁`.
///
/// Panique si `n1 <= 0`.
pub fn affinity_flow(q1: f64, n1: f64, n2: f64) -> f64 {
    assert!(
        n1 > 0.0,
        "la vitesse initiale doit être strictement positive"
    );
    q1 * n2 / n1
}

/// Hauteur après changement de vitesse (affinité) `H₂ = H₁·(N₂/N₁)²`.
///
/// Panique si `n1 <= 0`.
pub fn affinity_head(h1: f64, n1: f64, n2: f64) -> f64 {
    assert!(
        n1 > 0.0,
        "la vitesse initiale doit être strictement positive"
    );
    let r = n2 / n1;
    h1 * r * r
}

/// Puissance après changement de vitesse (affinité) `P₂ = P₁·(N₂/N₁)³`.
///
/// Panique si `n1 <= 0`.
pub fn affinity_power(p1: f64, n1: f64, n2: f64) -> f64 {
    assert!(
        n1 > 0.0,
        "la vitesse initiale doit être strictement positive"
    );
    let r = n2 / n1;
    p1 * r * r * r
}

/// Vitesse spécifique `Ns = N·√Q / H^{3/4}` (unités de l'appelant).
///
/// Panique si `head <= 0` ou `flow < 0`.
pub fn specific_speed(rotational_speed: f64, flow: f64, head: f64) -> f64 {
    assert!(head > 0.0 && flow >= 0.0, "H > 0 et Q ≥ 0 requis");
    rotational_speed * flow.sqrt() / head.powf(0.75)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_and_shaft_power() {
        // ρ=1000, g=9,81, Q=0,05 m³/s, H=30 m → P_h = 14715 W.
        let ph = hydraulic_power(1000.0, 9.81, 0.05, 30.0);
        assert_relative_eq!(ph, 14_715.0, epsilon = 1e-6);
        // η=0,75 → P_a = 19620 W.
        assert_relative_eq!(shaft_power(ph, 0.75), ph / 0.75, epsilon = 1e-9);
    }

    #[test]
    fn npsh_available_drops_with_vapor_pressure() {
        // p_abs=101325, p_vap=2339 (eau 20°C), ρ=998, g=9,81, v=1,5.
        let n = npsh_available(101_325.0, 2339.0, 998.0, 9.81, 1.5);
        assert_relative_eq!(
            n,
            (101_325.0 - 2339.0) / (998.0 * 9.81) + 1.5 * 1.5 / (2.0 * 9.81),
            epsilon = 1e-9
        );
        assert!(n > 0.0);
    }

    #[test]
    fn affinity_laws_scale_as_powers_of_speed() {
        // Doubler la vitesse : Q×2, H×4, P×8.
        assert_relative_eq!(affinity_flow(10.0, 1450.0, 2900.0), 20.0, epsilon = 1e-9);
        assert_relative_eq!(affinity_head(30.0, 1450.0, 2900.0), 120.0, epsilon = 1e-9);
        assert_relative_eq!(
            affinity_power(5000.0, 1450.0, 2900.0),
            40_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn specific_speed_definition() {
        // N=1450, Q=0,05, H=30 → Ns = 1450·√0,05/30^0,75.
        assert_relative_eq!(
            specific_speed(1450.0, 0.05, 30.0),
            1450.0 * 0.05f64.sqrt() / 30.0f64.powf(0.75),
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "rendement")]
    fn efficiency_above_one_panics() {
        shaft_power(1000.0, 1.5);
    }
}
