//! Coefficient de débit d'une vanne (**Kv** / **Cv**) : dimensionnement d'une
//! vanne de régulation en **écoulement liquide** à partir du débit, de la chute
//! de pression et de la densité relative.
//!
//! ```text
//! débit liquide     Q  = Kv·√(ΔP/SG)                 (m³/h)
//! Kv nécessaire     Kv = Q/√(ΔP/SG)                  (m³·h⁻¹·bar^{-1/2})
//! chute de pression ΔP = SG·(Q/Kv)²                  (bar)
//! conversion        Cv = 1,156·Kv                    (impérial ↔ métrique)
//!                   Kv = Cv/1,156
//! ```
//!
//! `Q` débit volumique (m³/h) ; `Kv` coefficient de débit **métrique** (débit
//! d'eau en m³/h pour ΔP = 1 bar) ; `Cv` coefficient de débit **impérial**
//! (gpm US pour ΔP = 1 psi) ; `ΔP` chute de pression au travers de la vanne
//! (bar) ; `SG` densité relative du liquide (adimensionnelle, eau ≈ 1).
//!
//! **Convention** : débit en m³/h, pression en **bar**, `SG` adimensionnel ; le
//! facteur numérique de la relation liquide est cohérent avec ces unités.
//!
//! **Limite honnête** : écoulement **liquide turbulent non cavitant** et non
//! flashant, régime permanent. Ne couvre **ni les gaz/vapeurs** (compressibles,
//! qui demandent une formulation en pression absolue et température), **ni la
//! cavitation/le flashing** (qui bornent le débit sous ΔP croissant). La densité
//! relative `SG` est une **donnée fournie par l'appelant** ; aucune valeur de
//! fluide, de vanne ou de procédé n'est supposée « par défaut » ici.

/// Facteur de conversion `Cv = 1,156·Kv` (coefficient impérial / métrique).
pub const VALVE_KV_TO_CV_FACTOR: f64 = 1.156;

/// Débit liquide `Q = Kv·√(ΔP/SG)` (m³/h).
///
/// `pressure_drop_bar` est ΔP en bar, `specific_gravity` est `SG` (eau ≈ 1).
///
/// Panique si `flow_coefficient_kv < 0`, `pressure_drop_bar < 0` ou
/// `specific_gravity <= 0`.
pub fn valve_flow_rate_from_kv(
    flow_coefficient_kv: f64,
    pressure_drop_bar: f64,
    specific_gravity: f64,
) -> f64 {
    assert!(
        flow_coefficient_kv >= 0.0,
        "le coefficient de débit Kv ne peut pas être négatif"
    );
    assert!(
        pressure_drop_bar >= 0.0,
        "la chute de pression ΔP ne peut pas être négative"
    );
    assert!(
        specific_gravity > 0.0,
        "la densité relative SG doit être strictement positive"
    );
    flow_coefficient_kv * (pressure_drop_bar / specific_gravity).sqrt()
}

/// Coefficient de débit `Kv = Q/√(ΔP/SG)` nécessaire pour passer le débit `Q`
/// (m³·h⁻¹·bar^{-1/2}) — inverse de [`valve_flow_rate_from_kv`].
///
/// Panique si `flow_rate_m3h < 0`, `pressure_drop_bar <= 0` ou
/// `specific_gravity <= 0`.
pub fn valve_kv_required(flow_rate_m3h: f64, pressure_drop_bar: f64, specific_gravity: f64) -> f64 {
    assert!(flow_rate_m3h >= 0.0, "le débit Q ne peut pas être négatif");
    assert!(
        pressure_drop_bar > 0.0,
        "la chute de pression ΔP doit être strictement positive"
    );
    assert!(
        specific_gravity > 0.0,
        "la densité relative SG doit être strictement positive"
    );
    flow_rate_m3h / (pressure_drop_bar / specific_gravity).sqrt()
}

/// Chute de pression `ΔP = SG·(Q/Kv)²` (bar) produite par un débit `Q` au travers
/// d'une vanne de coefficient `Kv` — inverse de [`valve_flow_rate_from_kv`].
///
/// Panique si `flow_rate_m3h < 0`, `flow_coefficient_kv <= 0` ou
/// `specific_gravity <= 0`.
pub fn valve_pressure_drop_bar(
    flow_rate_m3h: f64,
    flow_coefficient_kv: f64,
    specific_gravity: f64,
) -> f64 {
    assert!(flow_rate_m3h >= 0.0, "le débit Q ne peut pas être négatif");
    assert!(
        flow_coefficient_kv > 0.0,
        "le coefficient de débit Kv doit être strictement positif"
    );
    assert!(
        specific_gravity > 0.0,
        "la densité relative SG doit être strictement positive"
    );
    let ratio = flow_rate_m3h / flow_coefficient_kv;
    specific_gravity * ratio * ratio
}

/// Conversion vers le coefficient **impérial** `Cv = 1,156·Kv`.
///
/// Panique si `flow_coefficient_kv < 0`.
pub fn valve_cv_from_kv(flow_coefficient_kv: f64) -> f64 {
    assert!(
        flow_coefficient_kv >= 0.0,
        "le coefficient de débit Kv ne peut pas être négatif"
    );
    VALVE_KV_TO_CV_FACTOR * flow_coefficient_kv
}

/// Conversion vers le coefficient **métrique** `Kv = Cv/1,156` — inverse de
/// [`valve_cv_from_kv`].
///
/// Panique si `flow_coefficient_cv < 0`.
pub fn valve_kv_from_cv(flow_coefficient_cv: f64) -> f64 {
    assert!(
        flow_coefficient_cv >= 0.0,
        "le coefficient de débit Cv ne peut pas être négatif"
    );
    flow_coefficient_cv / VALVE_KV_TO_CV_FACTOR
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn debit_cas_chiffre() {
        // Kv = 10, ΔP = 4 bar, SG = 1 → Q = 10·√(4/1) = 10·2 = 20 m³/h.
        let q = valve_flow_rate_from_kv(10.0, 4.0, 1.0);
        assert_relative_eq!(q, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn reciprocite_debit_kv() {
        // Kv → Q → Kv doit redonner le Kv de départ.
        let (kv, dp, sg) = (12.5, 2.5, 0.85);
        let q = valve_flow_rate_from_kv(kv, dp, sg);
        let kv_back = valve_kv_required(q, dp, sg);
        assert_relative_eq!(kv_back, kv, epsilon = 1e-12);
    }

    #[test]
    fn reciprocite_debit_chute_pression() {
        // ΔP = SG·(Q/Kv)² doit être l'inverse de Q = Kv·√(ΔP/SG).
        let (kv, dp, sg) = (8.0, 3.2, 1.15);
        let q = valve_flow_rate_from_kv(kv, dp, sg);
        let dp_back = valve_pressure_drop_bar(q, kv, sg);
        assert_relative_eq!(dp_back, dp, epsilon = 1e-12);
    }

    #[test]
    fn proportionnalite_kv() {
        // Q est linéaire en Kv à ΔP et SG fixés : doubler Kv double Q.
        let q1 = valve_flow_rate_from_kv(5.0, 1.5, 0.9);
        let q2 = valve_flow_rate_from_kv(10.0, 1.5, 0.9);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn conversion_cv_reciproque() {
        // Cv = 1,156·Kv puis retour ; et valeur chiffrée Kv = 10 → Cv = 11,56.
        assert_relative_eq!(valve_cv_from_kv(10.0), 11.56, epsilon = 1e-12);
        let kv = 17.3;
        assert_relative_eq!(valve_kv_from_cv(valve_cv_from_kv(kv)), kv, epsilon = 1e-12);
    }

    #[test]
    fn eau_unitaire() {
        // Définition du Kv : eau (SG = 1) sous ΔP = 1 bar → Q = Kv numériquement.
        let kv = 42.0;
        assert_relative_eq!(valve_flow_rate_from_kv(kv, 1.0, 1.0), kv, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "densité relative SG doit être strictement positive")]
    fn sg_nul_panique() {
        let _ = valve_flow_rate_from_kv(10.0, 4.0, 0.0);
    }
}
