//! Rodage libre / polissage — loi de **Preston** : taux d'enlèvement de matière
//! proportionnel au produit pression × vitesse relative.
//!
//! ```text
//! taux d'enlèvement   MRR = K_p·P·v          (m/s)
//! matière enlevée     e   = K_p·P·v·t        (m)
//! temps pour un stock t   = stock/(K_p·P·v)  (s)
//! pression            P   = F/A              (Pa)
//! ```
//!
//! `MRR` taux d'enlèvement exprimé en épaisseur par unité de temps (m/s), `K_p`
//! coefficient de Preston (Pa⁻¹, dépend de l'abrasif, du matériau et du lubrifiant),
//! `P` pression de contact (Pa), `v` vitesse relative outil/pièce (m/s), `t` durée
//! du procédé (s), `e` épaisseur enlevée (m), `stock` surépaisseur à enlever (m),
//! `F` charge normale appliquée (N), `A` aire de contact (m²).
//!
//! **Convention** : SI cohérent. **Limite honnête** : loi de **Preston**
//! (enlèvement ∝ pression × vitesse) à coefficient `K_p` **constant** ; ce
//! coefficient regroupe toute la physique du couple abrasif/matériau/lubrifiant et
//! est **fourni par l'appelant** — aucune valeur « par défaut » n'est inventée ici.
//! Pression et vitesse relative sont supposées **uniformes** sur la zone de contact.

/// Taux d'enlèvement de matière `MRR = K_p·P·v` (m/s), loi de Preston.
///
/// Panique si `preston_coefficient < 0`, `pressure < 0` ou `relative_velocity < 0`.
pub fn lapping_removal_rate(
    preston_coefficient: f64,
    pressure: f64,
    relative_velocity: f64,
) -> f64 {
    assert!(
        preston_coefficient >= 0.0 && pressure >= 0.0 && relative_velocity >= 0.0,
        "K_p ≥ 0, P ≥ 0 et v ≥ 0 requis"
    );
    preston_coefficient * pressure * relative_velocity
}

/// Épaisseur de matière enlevée `e = K_p·P·v·t` (m).
///
/// Panique si `preston_coefficient < 0`, `pressure < 0`, `relative_velocity < 0`
/// ou `time < 0`.
pub fn lapping_material_removed(
    preston_coefficient: f64,
    pressure: f64,
    relative_velocity: f64,
    time: f64,
) -> f64 {
    assert!(
        preston_coefficient >= 0.0 && pressure >= 0.0 && relative_velocity >= 0.0 && time >= 0.0,
        "K_p ≥ 0, P ≥ 0, v ≥ 0 et t ≥ 0 requis"
    );
    preston_coefficient * pressure * relative_velocity * time
}

/// Temps requis pour enlever une surépaisseur `stock` : `t = stock/(K_p·P·v)` (s).
///
/// Panique si `stock < 0`, `preston_coefficient <= 0`, `pressure <= 0`
/// ou `relative_velocity <= 0`.
pub fn lapping_time_for_stock(
    stock: f64,
    preston_coefficient: f64,
    pressure: f64,
    relative_velocity: f64,
) -> f64 {
    assert!(
        stock >= 0.0 && preston_coefficient > 0.0 && pressure > 0.0 && relative_velocity > 0.0,
        "stock ≥ 0, K_p > 0, P > 0 et v > 0 requis"
    );
    stock / (preston_coefficient * pressure * relative_velocity)
}

/// Pression de contact `P = F/A` (Pa).
///
/// Panique si `load < 0` ou `contact_area <= 0`.
pub fn lapping_pressure(load: f64, contact_area: f64) -> f64 {
    assert!(load >= 0.0 && contact_area > 0.0, "F ≥ 0 et A > 0 requis");
    load / contact_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn removal_rate_proportional_to_pressure_and_velocity() {
        // MRR ∝ P·v : doubler la pression ou la vitesse double le taux.
        let r1 = lapping_removal_rate(1e-13, 5e4, 0.5);
        let r2 = lapping_removal_rate(1e-13, 1e5, 0.5);
        let r3 = lapping_removal_rate(1e-13, 5e4, 1.0);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-9);
        assert_relative_eq!(r3 / r1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn material_removed_is_rate_times_time() {
        // e = MRR·t : cohérence entre les deux fonctions.
        let (kp, p, v, t) = (2e-13, 4e4, 0.8, 120.0);
        let e = lapping_material_removed(kp, p, v, t);
        assert_relative_eq!(e, lapping_removal_rate(kp, p, v) * t, max_relative = 1e-12);
    }

    #[test]
    fn time_for_stock_is_inverse_of_removal() {
        // t = stock/MRR : réciprocité avec le taux d'enlèvement.
        let (kp, p, v) = (1.5e-13, 6e4, 0.6);
        let stock = 3e-6;
        let t = lapping_time_for_stock(stock, kp, p, v);
        assert_relative_eq!(
            lapping_material_removed(kp, p, v, t),
            stock,
            max_relative = 1e-9
        );
    }

    #[test]
    fn pressure_is_load_over_area() {
        // F = 200 N sur A = 0,01 m² → P = 20 000 Pa.
        assert_relative_eq!(lapping_pressure(200.0, 0.01), 2e4, epsilon = 1e-9);
    }

    #[test]
    fn realistic_removal_case() {
        // Cas chiffré : K_p = 1e-13 Pa⁻¹, F = 150 N sur A = 0,005 m² → P = 30 000 Pa,
        // v = 0,5 m/s. MRR = 1e-13 · 30000 · 0,5 = 1,5e-9 m/s = 1,5 nm/s.
        // En t = 600 s : e = 1,5e-9 · 600 = 9e-7 m = 0,9 µm.
        let p = lapping_pressure(150.0, 0.005);
        assert_relative_eq!(p, 3e4, epsilon = 1e-9);
        let mrr = lapping_removal_rate(1e-13, p, 0.5);
        assert_relative_eq!(mrr, 1.5e-9, max_relative = 1e-12);
        let e = lapping_material_removed(1e-13, p, 0.5, 600.0);
        assert_relative_eq!(e, 9e-7, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_area_panics() {
        lapping_pressure(150.0, 0.0);
    }
}
