//! Découpe au jet d'eau abrasif — **vitesse et puissance du jet** : vitesse
//! théorique du jet par Bernoulli, indice empirique de vitesse de coupe et
//! puissance hydraulique du jet.
//!
//! ```text
//! vitesse du jet      v = sqrt(2·P / rho)                  (Bernoulli idéal)
//! indice de coupe     i = k · sqrt(P) · d² / (t · M)       (forme empirique)
//! puissance du jet    W = P · Q
//! ```
//!
//! `P` pression de service de la pompe (Pa), `rho` masse volumique de l'eau
//! (kg/m³), `v` vitesse théorique du jet à la sortie de l'orifice (m/s), `k`
//! coefficient empirique du procédé (unité telle que `i` sorte en m/s), `d`
//! diamètre de l'orifice / buse (m), `t` épaisseur de la pièce (m), `M` facteur
//! d'usinabilité du matériau (sans dimension, plus grand = plus difficile),
//! `i` indice de vitesse de coupe (m/s), `Q` débit volumique d'eau (m³/s), `W`
//! puissance hydraulique du jet (W). La vitesse du jet découle d'un bilan de
//! Bernoulli sans perte : toute la pression se convertit en énergie cinétique.
//!
//! **Convention** : SI cohérent (Pa, kg/m³, m/s, m, m³/s, W). **Limite
//! honnête** : la vitesse du jet est celle d'un jet parfait de Bernoulli (aucun
//! coefficient de décharge ni perte visqueuse) ; l'indice de vitesse de coupe
//! est une **corrélation empirique** dont le coefficient `k` et le facteur
//! d'usinabilité `M` sont **fournis par l'appelant**. Aucune valeur de
//! matériau, de procédé ou de rendement n'est inventée ici.

/// Vitesse théorique du jet `v = sqrt(2·P / rho)` (m/s).
///
/// Vitesse de sortie d'un jet parfait obtenue par conversion intégrale de la
/// pression en énergie cinétique (équation de Bernoulli sans perte).
///
/// Panique si `pressure < 0` ou `water_density <= 0`.
pub fn waterjet_jet_velocity(pressure: f64, water_density: f64) -> f64 {
    assert!(pressure >= 0.0, "la pression doit être positive");
    assert!(
        water_density > 0.0,
        "la masse volumique de l'eau doit être strictement positive"
    );
    (2.0_f64 * pressure / water_density).sqrt()
}

/// Indice empirique de vitesse de coupe `i = k · sqrt(P) · d² / (t · M)` (m/s).
///
/// Corrélation empirique : la vitesse de coupe admissible croît avec la racine
/// de la pression et le carré du diamètre de buse, et décroît avec l'épaisseur
/// et le facteur d'usinabilité du matériau.
///
/// Panique si `coefficient < 0`, `pressure < 0`, `orifice_diameter < 0`,
/// `thickness <= 0` ou `machinability <= 0`.
pub fn waterjet_cutting_speed_index(
    coefficient: f64,
    pressure: f64,
    orifice_diameter: f64,
    thickness: f64,
    machinability: f64,
) -> f64 {
    assert!(
        coefficient >= 0.0,
        "le coefficient empirique doit être positif"
    );
    assert!(pressure >= 0.0, "la pression doit être positive");
    assert!(
        orifice_diameter >= 0.0,
        "le diamètre d'orifice doit être positif"
    );
    assert!(
        thickness > 0.0,
        "l'épaisseur de la pièce doit être strictement positive"
    );
    assert!(
        machinability > 0.0,
        "le facteur d'usinabilité doit être strictement positif"
    );
    coefficient * pressure.sqrt() * orifice_diameter.powi(2) / (thickness * machinability)
}

/// Puissance hydraulique du jet `W = P · Q` (W).
///
/// Puissance transportée par le jet, produit de la pression de service par le
/// débit volumique d'eau.
///
/// Panique si `pressure < 0` ou `flow_rate < 0`.
pub fn waterjet_jet_power(pressure: f64, flow_rate: f64) -> f64 {
    assert!(pressure >= 0.0, "la pression doit être positive");
    assert!(flow_rate >= 0.0, "le débit volumique doit être positif");
    pressure * flow_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn jet_velocity_realistic_value() {
        // Pompe THP 350 MPa, eau à 1000 kg/m³ :
        // v = sqrt(2·3,5e8 / 1000) = sqrt(7e5) ≈ 836,66 m/s (jet supersonique).
        let v = waterjet_jet_velocity(3.5e8, 1000.0);
        assert_relative_eq!(v, 836.660_026_534_1, epsilon = 1e-6);
    }

    #[test]
    fn jet_velocity_scales_with_sqrt_pressure() {
        // v ∝ sqrt(P) : quadrupler la pression double la vitesse.
        let v1 = waterjet_jet_velocity(1e8, 1000.0);
        let v2 = waterjet_jet_velocity(4e8, 1000.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn cutting_index_scales_with_square_of_diameter() {
        // i ∝ d² : doubler le diamètre de buse quadruple l'indice.
        let i1 = waterjet_cutting_speed_index(1.0, 3e8, 2.5e-4, 0.01, 2.0);
        let i2 = waterjet_cutting_speed_index(1.0, 3e8, 5.0e-4, 0.01, 2.0);
        assert_relative_eq!(i2 / i1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn cutting_index_scales_inversely_with_thickness() {
        // i ∝ 1/t : doubler l'épaisseur divise l'indice par deux.
        let i1 = waterjet_cutting_speed_index(1.0, 3e8, 3e-4, 0.005, 2.0);
        let i2 = waterjet_cutting_speed_index(1.0, 3e8, 3e-4, 0.010, 2.0);
        assert_relative_eq!(i1 / i2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn jet_power_realistic_value() {
        // P = 300 MPa, Q = 2 L/min = 2e-3/60 m³/s ≈ 3,333e-5 m³/s :
        // W = 3e8 · 3,333e-5 = 1e4 W = 10 kW.
        let q = 2.0e-3_f64 / 60.0;
        let w = waterjet_jet_power(3.0e8, q);
        assert_relative_eq!(w, 3.0e8 * q, epsilon = 1e-6);
        assert_relative_eq!(w, 10_000.0, epsilon = 1e-6);
    }

    #[test]
    fn jet_power_is_bilinear() {
        // W = P·Q : doubler l'un OU l'autre double la puissance.
        let base = waterjet_jet_power(3e8, 3e-5);
        let double_p = waterjet_jet_power(6e8, 3e-5);
        let double_q = waterjet_jet_power(3e8, 6e-5);
        assert_relative_eq!(double_p / base, 2.0, epsilon = 1e-12);
        assert_relative_eq!(double_q / base, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "facteur d'usinabilité")]
    fn zero_machinability_panics() {
        waterjet_cutting_speed_index(1.0, 3e8, 3e-4, 0.01, 0.0);
    }
}
