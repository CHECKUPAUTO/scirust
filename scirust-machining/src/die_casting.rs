//! Fonderie **sous pression** (die casting) : vitesse en attaque, temps de
//! remplissage de l'empreinte et effort de verrouillage de la presse.
//!
//! ```text
//! vitesse en attaque   v = Q/Ag              (Q débit, Ag section d'attaque)
//! temps de remplissage t = V/Q               (V volume d'empreinte)
//! effort de verrouillage F = A·p·S           (A surface projetée, p pression, S coeff.)
//! ```
//!
//! `Q` débit volumique injecté (m³/s), `Ag` section d'attaque (gate) (m²), `v`
//! vitesse du métal en attaque (m/s), `V` volume de l'empreinte (m³), `t` temps
//! de remplissage (s), `A` surface projetée sur le plan de joint (m²), `p`
//! pression d'injection (Pa), `S` coefficient de sécurité (–), `F` effort de
//! verrouillage requis (N).
//!
//! **Convention** : SI cohérent. **Limite honnête** : remplissage idéalisé,
//! pertes de charge et compressibilité négligées. La pression d'injection, la
//! surface projetée, le débit et le coefficient de sécurité sont **fournis par
//! l'appelant** ; aucune valeur physique/matériau n'est présumée par défaut.

/// Vitesse du métal en attaque `v = Q/Ag` (m/s).
///
/// Panique si `gate_area <= 0`.
pub fn diecast_gate_velocity(flow_rate: f64, gate_area: f64) -> f64 {
    assert!(
        gate_area > 0.0,
        "la section d'attaque doit être strictement positive"
    );
    flow_rate / gate_area
}

/// Temps de remplissage de l'empreinte `t = V/Q` (s).
///
/// Panique si `flow_rate <= 0`.
pub fn diecast_fill_time(cavity_volume: f64, flow_rate: f64) -> f64 {
    assert!(
        flow_rate > 0.0,
        "le débit volumique doit être strictement positif"
    );
    cavity_volume / flow_rate
}

/// Effort de verrouillage de la presse `F = A·p·S` (N).
///
/// Panique si `projected_area < 0`, `injection_pressure < 0` ou
/// `safety_factor < 1`.
pub fn diecast_locking_force(
    projected_area: f64,
    injection_pressure: f64,
    safety_factor: f64,
) -> f64 {
    assert!(
        projected_area >= 0.0,
        "la surface projetée doit être positive"
    );
    assert!(
        injection_pressure >= 0.0,
        "la pression d'injection doit être positive"
    );
    assert!(
        safety_factor >= 1.0,
        "le coefficient de sécurité doit être supérieur ou égal à 1"
    );
    projected_area * injection_pressure * safety_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gate_velocity_and_fill_time_are_consistent() {
        // Avec v = Q/Ag et t = V/Q, on a V = v·Ag·t : réciprocité des définitions.
        let (q, ag, vol) = (2.0e-3_f64, 5.0e-5_f64, 1.0e-4_f64);
        let v = diecast_gate_velocity(q, ag);
        let t = diecast_fill_time(vol, q);
        assert_relative_eq!(vol, v * ag * t, max_relative = 1e-12);
    }

    #[test]
    fn gate_velocity_realistic_case() {
        // Q = 1 L/s = 1e-3 m³/s, Ag = 100 mm² = 1e-4 m² → v = 10 m/s.
        assert_relative_eq!(diecast_gate_velocity(1.0e-3, 1.0e-4), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn smaller_gate_gives_higher_velocity() {
        // À débit constant, réduire la section d'attaque augmente la vitesse.
        let q = 1.0e-3_f64;
        assert!(diecast_gate_velocity(q, 5.0e-5) > diecast_gate_velocity(q, 1.0e-4));
    }

    #[test]
    fn fill_time_inversely_proportional_to_flow() {
        // Doubler le débit divise par deux le temps de remplissage.
        let vol = 2.0e-4_f64;
        let t1 = diecast_fill_time(vol, 1.0e-3);
        let t2 = diecast_fill_time(vol, 2.0e-3);
        assert_relative_eq!(t1, 2.0 * t2, max_relative = 1e-12);
    }

    #[test]
    fn locking_force_realistic_case() {
        // A = 0,02 m² (200 cm²), p = 80 MPa, S = 1,2 → F = 0,02·8e7·1,2 = 1,92 MN.
        assert_relative_eq!(
            diecast_locking_force(0.02, 80.0e6, 1.2),
            1.92e6,
            max_relative = 1e-12
        );
    }

    #[test]
    fn locking_force_scales_with_projected_area() {
        // À pression et sécurité fixées, F est proportionnel à la surface projetée.
        let f1 = diecast_locking_force(0.01, 50.0e6, 1.5);
        let f2 = diecast_locking_force(0.03, 50.0e6, 1.5);
        assert_relative_eq!(f2, 3.0 * f1, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "coefficient de sécurité")]
    fn locking_force_rejects_safety_factor_below_one() {
        diecast_locking_force(0.02, 80.0e6, 0.9);
    }
}
