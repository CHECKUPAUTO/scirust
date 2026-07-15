//! Palier hydrostatique (patin à poche) — capacité de charge, débit d'alimentation,
//! puissance de pompage et raideur du film.
//!
//! ```text
//! capacité de charge   W = ps·Ae·β
//! débit d'alimentation Q = ps·h³·k / µ
//! puissance de pompage P = ps·Q
//! raideur du film      S = 3·W / h        (compensateur idéal)
//! ```
//!
//! `ps` pression d'alimentation (Pa), `Ae` aire effective du patin (m²), `β` rapport
//! de pression `p_poche/ps` (sans dimension, `0`…`1`), `h` épaisseur du film (m),
//! `µ` viscosité dynamique (Pa·s), `k` facteur géométrique de la portée (sans
//! dimension), `Q` débit (m³/s), `W` charge (N), `P` puissance (W), `S` raideur (N/m).
//!
//! **Limite honnête** : écoulement **laminaire** et **film mince** supposés ;
//! l'aire effective `Ae` et le facteur géométrique `k` de la portée du patin sont
//! **fournis par l'appelant** (aucune valeur « par défaut » inventée) ; la raideur
//! `S = 3·W/h` correspond à un **compensateur idéal** à pression constante. Le
//! choix du compensateur réel (capillaire, orifice, valve) reste à la charge de
//! l'appelant.

/// Capacité de charge du patin `W = ps·Ae·β` (N).
///
/// Panique si `supply_pressure < 0`, `effective_area < 0`, ou si le rapport de
/// pression `pressure_ratio` sort de `[0, 1]`.
pub fn hydrostatic_load_capacity(
    supply_pressure: f64,
    effective_area: f64,
    pressure_ratio: f64,
) -> f64 {
    assert!(
        supply_pressure >= 0.0,
        "la pression d'alimentation doit être positive"
    );
    assert!(effective_area >= 0.0, "l'aire effective doit être positive");
    assert!(
        (0.0..=1.0).contains(&pressure_ratio),
        "le rapport de pression doit être dans [0, 1]"
    );
    supply_pressure * effective_area * pressure_ratio
}

/// Débit d'alimentation `Q = ps·h³·k / µ` (m³/s), variant en `h³`.
///
/// Panique si `viscosity <= 0`, `supply_pressure < 0`, `film_thickness < 0`, ou
/// `land_geometry_factor < 0`.
pub fn hydrostatic_flow_rate(
    supply_pressure: f64,
    film_thickness: f64,
    viscosity: f64,
    land_geometry_factor: f64,
) -> f64 {
    assert!(
        viscosity > 0.0,
        "la viscosité doit être strictement positive"
    );
    assert!(
        supply_pressure >= 0.0,
        "la pression d'alimentation doit être positive"
    );
    assert!(
        film_thickness >= 0.0,
        "l'épaisseur du film doit être positive"
    );
    assert!(
        land_geometry_factor >= 0.0,
        "le facteur géométrique doit être positif"
    );
    supply_pressure * film_thickness.powi(3) * land_geometry_factor / viscosity
}

/// Puissance de pompage `P = ps·Q` (W).
///
/// Panique si `supply_pressure < 0` ou `flow_rate < 0`.
pub fn hydrostatic_pumping_power(supply_pressure: f64, flow_rate: f64) -> f64 {
    assert!(
        supply_pressure >= 0.0,
        "la pression d'alimentation doit être positive"
    );
    assert!(flow_rate >= 0.0, "le débit doit être positif");
    supply_pressure * flow_rate
}

/// Raideur du film pour un compensateur idéal `S = 3·W / h` (N/m).
///
/// Panique si `film_thickness <= 0`.
pub fn hydrostatic_stiffness(load_capacity: f64, film_thickness: f64) -> f64 {
    assert!(
        film_thickness > 0.0,
        "l'épaisseur du film doit être strictement positive"
    );
    3.0 * load_capacity / film_thickness
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn load_capacity_realistic_value() {
        // ps = 5 MPa, Ae = 0,01 m², β = 0,5 → W = 5e6·0,01·0,5 = 25 000 N.
        assert_relative_eq!(
            hydrostatic_load_capacity(5e6, 0.01, 0.5),
            25_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn flow_rate_scales_as_thickness_cubed() {
        // Doubler l'épaisseur multiplie le débit par 8.
        let q1 = hydrostatic_flow_rate(5e6, 50e-6, 0.04, 0.1);
        let q2 = hydrostatic_flow_rate(5e6, 100e-6, 0.04, 0.1);
        assert_relative_eq!(q2 / q1, 8.0, epsilon = 1e-9);
    }

    #[test]
    fn flow_rate_realistic_value() {
        // ps=5e6, h=5e-5, µ=0,04, k=0,1 :
        // Q = 5e6·(5e-5)³·0,1/0,04 = 5e6·1,25e-13·0,1/0,04 = 1,5625e-6 m³/s.
        assert_relative_eq!(
            hydrostatic_flow_rate(5e6, 50e-6, 0.04, 0.1),
            1.562_5e-6,
            epsilon = 1e-12
        );
    }

    #[test]
    fn pumping_power_matches_pressure_times_flow() {
        // P = ps·Q ; avec Q ci-dessus : P = 5e6·1,5625e-6 = 7,8125 W.
        let q = hydrostatic_flow_rate(5e6, 50e-6, 0.04, 0.1);
        let p = hydrostatic_pumping_power(5e6, q);
        assert_relative_eq!(p, 5e6 * q, max_relative = 1e-12);
        assert_relative_eq!(p, 7.812_5, epsilon = 1e-9);
    }

    #[test]
    fn stiffness_is_inversely_proportional_to_film() {
        // S = 3W/h ; halver h double la raideur.
        let s1 = hydrostatic_stiffness(25_000.0, 50e-6);
        let s2 = hydrostatic_stiffness(25_000.0, 25e-6);
        assert_relative_eq!(s2 / s1, 2.0, epsilon = 1e-9);
        // Valeur chiffrée : 3·25000/5e-5 = 1,5e9 N/m.
        assert_relative_eq!(s1, 1.5e9, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "rapport de pression")]
    fn load_capacity_rejects_ratio_above_one() {
        hydrostatic_load_capacity(5e6, 0.01, 1.5);
    }
}
