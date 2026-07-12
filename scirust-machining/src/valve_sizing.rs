//! Dimensionnement des **vannes** — coefficients de débit `Kv` (métrique) et
//! `Cv` (impérial) pour un liquide.
//!
//! ```text
//! débit liquide   Q = Kv·√(Δp/SG)          Q [m³/h], Δp [bar]
//! Kv requis       Kv = Q·√(SG/Δp)
//! conversions     Kv = 0,865·Cv     Cv = 1,156·Kv
//! ```
//!
//! `Kv` débit d'eau (m³/h) sous `Δp = 1 bar`, `Cv` son équivalent impérial
//! (gpm US sous 1 psi), `Q` débit volumique (m³/h), `Δp` perte de pression (bar),
//! `SG` densité relative (eau = 1).
//!
//! **Convention** : `Q` en m³/h, `Δp` en bar (unités de catalogue robinetterie).
//! **Limite honnête** : formule **liquide** (incompressible), régime turbulent
//! **non cavitant** ; ne couvre ni le débit gazeux compressible (voir
//! [`crate::air_flow`]), ni la correction de viscosité pour les fluides
//! visqueux. `Kv`/`Cv` sont des données de la vanne fournies par l'appelant.

/// Conversion `Kv = 0,865·Cv`.
pub fn kv_from_cv(cv: f64) -> f64 {
    0.865 * cv
}

/// Conversion `Cv = 1,156·Kv`.
pub fn cv_from_kv(kv: f64) -> f64 {
    1.156 * kv
}

/// Débit liquide `Q = Kv·√(Δp/SG)`.
///
/// Panique si `pressure_drop_bar <= 0` ou `specific_gravity <= 0`.
pub fn liquid_flow(kv: f64, pressure_drop_bar: f64, specific_gravity: f64) -> f64 {
    assert!(
        pressure_drop_bar > 0.0 && specific_gravity > 0.0,
        "Δp et densité relative strictement positifs requis"
    );
    kv * (pressure_drop_bar / specific_gravity).sqrt()
}

/// `Kv` requis pour un débit cible `Kv = Q·√(SG/Δp)`.
///
/// Panique si `pressure_drop_bar <= 0` ou `specific_gravity <= 0`.
pub fn required_kv(flow_m3h: f64, pressure_drop_bar: f64, specific_gravity: f64) -> f64 {
    assert!(
        pressure_drop_bar > 0.0 && specific_gravity > 0.0,
        "Δp et densité relative strictement positifs requis"
    );
    flow_m3h * (specific_gravity / pressure_drop_bar).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn kv_cv_round_trip() {
        // Kv≈0,865·Cv et Cv≈1,156·Kv sont quasi réciproques (0,865·1,156 ≈ 1).
        assert_relative_eq!(kv_from_cv(cv_from_kv(10.0)), 10.0, max_relative = 2e-3);
    }

    #[test]
    fn flow_at_unit_drop_equals_kv() {
        // Par définition, sous Δp = 1 bar et eau (SG=1) : Q = Kv.
        assert_relative_eq!(liquid_flow(25.0, 1.0, 1.0), 25.0, epsilon = 1e-12);
    }

    #[test]
    fn flow_scales_with_sqrt_of_drop() {
        // ×4 sur Δp → ×2 sur le débit.
        let q1 = liquid_flow(25.0, 1.0, 1.0);
        let q4 = liquid_flow(25.0, 4.0, 1.0);
        assert_relative_eq!(q4 / q1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn required_kv_inverts_flow() {
        // Kv requis pour Q = flow(Kv) redonne Kv.
        let kv = required_kv(liquid_flow(16.0, 2.5, 0.9), 2.5, 0.9);
        assert_relative_eq!(kv, 16.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "Δp et densité")]
    fn zero_drop_panics() {
        liquid_flow(25.0, 0.0, 1.0);
    }
}
