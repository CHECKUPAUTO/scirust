//! Échangeur de chaleur — méthode de la différence de température logarithmique
//! moyenne (DTLM). Dimensionnement et bilan d'un échangeur en régime permanent :
//! DTLM entre les deux fluides, puissance échangée `Q = U·A·DTLM·F`, aire de
//! transfert requise et température de sortie côté froid par bilan enthalpique.
//!
//! ```text
//! différence de température logarithmique moyenne (DTLM)
//!   DTLM = (ΔT₁ − ΔT₂) / ln(ΔT₁/ΔT₂)        [K]   (ΔT₁ ≠ ΔT₂)
//!   DTLM = ΔT₁                               [K]   (ΔT₁ = ΔT₂, limite)
//! puissance échangée
//!   Q = U · A · DTLM · F                     [W]
//! aire de transfert requise
//!   A = Q / (U · DTLM · F)                    [m²]
//! température de sortie côté froid (bilan)
//!   T_out = T_in + Q / (m·c_p)               [K]
//! ```
//!
//! `ΔT₁`, `ΔT₂` écarts de température aux deux extrémités de l'échangeur [K],
//! `DTLM` différence de température logarithmique moyenne [K], `U` coefficient
//! global d'échange rapporté à l'aire `A` [W·m⁻²·K⁻¹], `A` aire de transfert
//! [m²], `F` facteur de correction de configuration [sans dimension, 0 < F ≤ 1],
//! `Q` puissance thermique échangée [W], `T_in`/`T_out` températures d'entrée et
//! de sortie du courant [K], `m` débit massique du courant [kg·s⁻¹], `c_p`
//! capacité thermique massique [J·kg⁻¹·K⁻¹].
//!
//! **Limite honnête** : modèle à l'échelle des **opérations unitaires**. Le
//! **coefficient global d'échange** `U`, le **facteur de correction** `F`
//! (fixé par la configuration multi-passes / à courants croisés, lu sur abaque
//! ou corrélation), les **capacités thermiques** `c_p` et les **propriétés
//! physiques** des fluides sont **FOURNIS** par l'appelant — jamais inventés ni
//! supposés « par défaut ». La DTLM n'est rigoureuse que pour un `U` **constant**
//! le long de l'échangeur et un écoulement **co-courant ou contre-courant pur**
//! (`F = 1`) ; `F < 1` corrige les faisceaux et configurations non idéales.
//! **Régime permanent** ; aucune corrélation de convection, aucun encrassement
//! ni aucune propriété d'état n'est calculé par ce module.

/// Différence de température logarithmique moyenne (DTLM)
/// `DTLM = (ΔT₁ − ΔT₂)/ln(ΔT₁/ΔT₂)` (K), ramenée à `ΔT₁` lorsque les deux
/// écarts sont égaux (limite continue, forme indéterminée levée).
///
/// `delta_t_1` (ΔT₁) et `delta_t_2` (ΔT₂) écarts de température aux deux
/// extrémités de l'échangeur [K], strictement positifs (même sens de transfert).
///
/// Panique si `ΔT₁ ≤ 0` ou si `ΔT₂ ≤ 0`.
pub fn lmtd_log_mean(delta_t_1: f64, delta_t_2: f64) -> f64 {
    assert!(
        delta_t_1 > 0.0,
        "ΔT₁ > 0 requis (écart de température à l'extrémité 1)"
    );
    assert!(
        delta_t_2 > 0.0,
        "ΔT₂ > 0 requis (écart de température à l'extrémité 2)"
    );
    if (delta_t_1 - delta_t_2).abs() < 1e-9
    {
        delta_t_1
    }
    else
    {
        (delta_t_1 - delta_t_2) / (delta_t_1 / delta_t_2).ln()
    }
}

/// Puissance thermique échangée
/// `Q = U·A·DTLM·F` (W), produit du coefficient global d'échange, de l'aire de
/// transfert, de la DTLM et du facteur de correction de configuration.
///
/// `overall_coefficient` (U) coefficient global d'échange [W·m⁻²·K⁻¹], `area`
/// (A) aire de transfert [m²], `log_mean_delta` (DTLM) [K], `correction_factor`
/// (F) facteur de correction [sans dimension, 0 < F ≤ 1].
///
/// Panique si `U < 0`, si `A < 0`, si `DTLM < 0`, ou si `F` hors de `]0, 1]`.
pub fn lmtd_duty(
    overall_coefficient: f64,
    area: f64,
    log_mean_delta: f64,
    correction_factor: f64,
) -> f64 {
    assert!(
        overall_coefficient >= 0.0,
        "U ≥ 0 requis (coefficient global d'échange)"
    );
    assert!(area >= 0.0, "A ≥ 0 requis (aire de transfert)");
    assert!(
        log_mean_delta >= 0.0,
        "DTLM ≥ 0 requis (différence de température logarithmique moyenne)"
    );
    assert!(
        correction_factor > 0.0 && correction_factor <= 1.0,
        "0 < F ≤ 1 requis (facteur de correction de configuration)"
    );
    overall_coefficient * area * log_mean_delta * correction_factor
}

/// Aire de transfert requise
/// `A = Q/(U·DTLM·F)` (m²), inverse de [`lmtd_duty`] : aire nécessaire pour
/// échanger une puissance `Q` donnée sous une DTLM et un `U` fixés.
///
/// `duty` (Q) puissance à échanger [W], `overall_coefficient` (U)
/// [W·m⁻²·K⁻¹], `log_mean_delta` (DTLM) [K], `correction_factor` (F)
/// facteur de correction [sans dimension, 0 < F ≤ 1].
///
/// Panique si `Q < 0`, si `U ≤ 0`, si `DTLM ≤ 0`, ou si `F` hors de `]0, 1]`.
pub fn lmtd_required_area(
    duty: f64,
    overall_coefficient: f64,
    log_mean_delta: f64,
    correction_factor: f64,
) -> f64 {
    assert!(duty >= 0.0, "Q ≥ 0 requis (puissance à échanger)");
    assert!(
        overall_coefficient > 0.0,
        "U > 0 requis (coefficient global d'échange)"
    );
    assert!(
        log_mean_delta > 0.0,
        "DTLM > 0 requis (différence de température logarithmique moyenne)"
    );
    assert!(
        correction_factor > 0.0 && correction_factor <= 1.0,
        "0 < F ≤ 1 requis (facteur de correction de configuration)"
    );
    duty / (overall_coefficient * log_mean_delta * correction_factor)
}

/// Température de sortie côté froid par bilan enthalpique
/// `T_out = T_in + Q/(m·c_p)` (K), élévation de température d'un courant qui
/// reçoit la puissance `Q` en régime permanent.
///
/// `inlet_temp` (T_in) température d'entrée du courant [K], `duty` (Q) puissance
/// reçue par le courant [W], `mass_flow` (m) débit massique [kg·s⁻¹],
/// `heat_capacity` (c_p) capacité thermique massique [J·kg⁻¹·K⁻¹].
///
/// Panique si `T_in ≤ 0` (température absolue), si `m ≤ 0`, ou si `c_p ≤ 0`.
pub fn lmtd_outlet_temp_from_duty(
    inlet_temp: f64,
    duty: f64,
    mass_flow: f64,
    heat_capacity: f64,
) -> f64 {
    assert!(
        inlet_temp > 0.0,
        "T_in > 0 requis (température absolue d'entrée en K)"
    );
    assert!(mass_flow > 0.0, "m > 0 requis (débit massique)");
    assert!(
        heat_capacity > 0.0,
        "c_p > 0 requis (capacité thermique massique)"
    );
    inlet_temp + duty / (mass_flow * heat_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn log_mean_is_symmetric_in_its_two_end_differences() {
        // La DTLM est symétrique : échanger ΔT₁ et ΔT₂ ne change pas le résultat.
        let a = lmtd_log_mean(50.0, 20.0);
        let b = lmtd_log_mean(20.0, 50.0);
        assert_relative_eq!(a, b, epsilon = 1e-9);
    }

    #[test]
    fn log_mean_reduces_to_end_difference_when_equal() {
        // Limite ΔT₁ = ΔT₂ : la forme indéterminée est levée à DTLM = ΔT₁.
        let l = lmtd_log_mean(25.0, 25.0);
        assert_relative_eq!(l, 25.0, epsilon = 1e-9);
        // La DTLM est toujours encadrée par ΔT₂ ≤ DTLM ≤ ΔT₁ (moyenne).
        let m = lmtd_log_mean(50.0, 20.0);
        assert!((20.0..=50.0).contains(&m));
    }

    #[test]
    fn log_mean_numeric_case() {
        // ΔT₁ = 50 K, ΔT₂ = 20 K : DTLM = (50 − 20)/ln(50/20) = 30/ln(2.5).
        // ln(2.5) = 0.916290731874..., 30/0.916290731874 = 32.7407000381...
        let l = lmtd_log_mean(50.0, 20.0);
        assert_relative_eq!(l, 32.740_700_038_118_74, epsilon = 1e-3);
    }

    #[test]
    fn duty_and_required_area_are_inverse_operations() {
        // A → Q → A : réciprocité de lmtd_duty et lmtd_required_area.
        let (u, area, lmtd, f) = (500.0, 12.0, 30.0, 0.9);
        let q = lmtd_duty(u, area, lmtd, f);
        let back = lmtd_required_area(q, u, lmtd, f);
        assert_relative_eq!(back, area, epsilon = 1e-9);
        // Cas chiffré : Q = 500·12·30·0.9 = 162 000 W.
        assert_relative_eq!(q, 162_000.0, epsilon = 1e-6);
    }

    #[test]
    fn duty_is_proportional_to_area() {
        // Q = U·A·DTLM·F : doubler l'aire double la puissance échangée.
        let q1 = lmtd_duty(400.0, 5.0, 25.0, 1.0);
        let q2 = lmtd_duty(400.0, 10.0, 25.0, 1.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-6);
        // Cas chiffré : Q₁ = 400·5·25·1 = 50 000 W.
        assert_relative_eq!(q1, 50_000.0, epsilon = 1e-6);
    }

    #[test]
    fn outlet_temperature_rises_by_duty_over_capacity_rate() {
        // T_out = T_in + Q/(m·c_p). Eau : m = 2 kg/s, c_p = 4180 J·kg⁻¹·K⁻¹.
        // Q = 501 600 W ⇒ ΔT = 501 600/(2·4180) = 60 K ⇒ T_out = 300 + 60 = 360 K.
        let t_out = lmtd_outlet_temp_from_duty(300.0, 501_600.0, 2.0, 4180.0);
        assert_relative_eq!(t_out, 360.0, epsilon = 1e-6);
        // Puissance nulle : la température ne change pas.
        let t_same = lmtd_outlet_temp_from_duty(300.0, 0.0, 2.0, 4180.0);
        assert_relative_eq!(t_same, 300.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ΔT₁ > 0")]
    fn log_mean_rejects_non_positive_end_difference() {
        lmtd_log_mean(0.0, 20.0);
    }
}
