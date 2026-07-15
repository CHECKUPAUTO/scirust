//! Écart de température logarithmique moyen (DTLM) pour le dimensionnement des
//! échangeurs de chaleur, en dispositions **contre-courant** et **co-courant**.
//!
//! ```text
//! contre-courant   ΔT1 = T_ch,e − T_fr,s     ΔT2 = T_ch,s − T_fr,e
//! co-courant       ΔT1 = T_ch,e − T_fr,e     ΔT2 = T_ch,s − T_fr,s
//! DTLM             ΔT_lm = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)
//!   (ΔT1 = ΔT2)    ΔT_lm = ΔT1                 (limite continue)
//! flux échangé     Q = U·A·F·ΔT_lm
//! ```
//!
//! `T_ch,e`/`T_ch,s` températures d'entrée/sortie du fluide **chaud** (K ou °C,
//! au choix mais cohérent), `T_fr,e`/`T_fr,s` idem pour le fluide **froid**,
//! `ΔT1`/`ΔT2` écarts de température aux deux extrémités (K), `ΔT_lm` DTLM (K),
//! `U` coefficient global d'échange (W/(m²·K)), `A` surface d'échange (m²),
//! `F` facteur de correction adimensionnel (multi-passes, `0 < F ≤ 1`),
//! `Q` flux thermique (W).
//!
//! **Limite honnête** : régime permanent, `U` supposé constant sur toute la
//! surface, débits et capacités thermiques constants, pas de changement de phase
//! ni de pertes vers l'extérieur ; le facteur de correction `F` (dispositions
//! multi-passes / courants croisés) est **fourni par l'appelant** — aucune valeur
//! de `U`, `A` ou `F` n'est inventée ici. Les températures doivent être
//! cohérentes : le croisement de températures (écart négatif ou nul aux
//! extrémités) est interdit.

/// Moyenne logarithmique de deux écarts de température strictement positifs,
/// `(ΔT1 − ΔT2)/ln(ΔT1/ΔT2)`, avec la limite continue `ΔT1` quand `ΔT1 ≈ ΔT2`.
fn log_mean_delta(delta_t1: f64, delta_t2: f64) -> f64 {
    if (delta_t1 / delta_t2 - 1.0).abs() < 1e-9
    {
        return 0.5 * (delta_t1 + delta_t2);
    }
    (delta_t1 - delta_t2) / (delta_t1 / delta_t2).ln()
}

/// DTLM en disposition **contre-courant** :
/// `ΔT1 = hot_in − cold_out`, `ΔT2 = hot_out − cold_in`,
/// `ΔT_lm = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)` (K). Renvoie `ΔT1` quand `ΔT1 ≈ ΔT2`.
///
/// Panique si un écart d'extrémité est `≤ 0` (croisement de températures).
pub fn lmtd_counterflow(hot_in: f64, hot_out: f64, cold_in: f64, cold_out: f64) -> f64 {
    let delta_t1 = hot_in - cold_out;
    let delta_t2 = hot_out - cold_in;
    assert!(
        delta_t1 > 0.0 && delta_t2 > 0.0,
        "écarts d'extrémité positifs requis (pas de croisement de températures)"
    );
    log_mean_delta(delta_t1, delta_t2)
}

/// DTLM en disposition **co-courant** :
/// `ΔT1 = hot_in − cold_in`, `ΔT2 = hot_out − cold_out`,
/// `ΔT_lm = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)` (K). Renvoie `ΔT1` quand `ΔT1 ≈ ΔT2`.
///
/// Panique si un écart d'extrémité est `≤ 0` (croisement de températures).
pub fn lmtd_parallelflow(hot_in: f64, hot_out: f64, cold_in: f64, cold_out: f64) -> f64 {
    let delta_t1 = hot_in - cold_in;
    let delta_t2 = hot_out - cold_out;
    assert!(
        delta_t1 > 0.0 && delta_t2 > 0.0,
        "écarts d'extrémité positifs requis (pas de croisement de températures)"
    );
    log_mean_delta(delta_t1, delta_t2)
}

/// Flux thermique échangé `Q = U·A·F·ΔT_lm` (W), avec `U` coefficient global
/// (W/(m²·K)), `A` surface (m²), `F` facteur de correction (`0 < F ≤ 1`) et
/// `ΔT_lm` la DTLM (K).
///
/// Panique si `overall_coefficient < 0`, `area < 0`, `lmtd < 0`, ou si
/// `correction_factor` sort de `]0, 1]`.
pub fn lmtd_heat_duty(
    overall_coefficient: f64,
    area: f64,
    lmtd: f64,
    correction_factor: f64,
) -> f64 {
    assert!(
        overall_coefficient >= 0.0,
        "le coefficient global U doit être positif"
    );
    assert!(area >= 0.0, "la surface d'échange A doit être positive");
    assert!(lmtd >= 0.0, "la DTLM doit être positive");
    assert!(
        correction_factor > 0.0 && correction_factor <= 1.0,
        "le facteur de correction F doit vérifier 0 < F ≤ 1"
    );
    overall_coefficient * area * correction_factor * lmtd
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn counterflow_matches_closed_form() {
        // T_ch : 150→90, T_fr : 40→80. ΔT1 = 150−80 = 70, ΔT2 = 90−40 = 50.
        // ΔT_lm = (70−50)/ln(70/50) = 20/ln(1,4) ≈ 59,4403 K.
        let value = lmtd_counterflow(150.0, 90.0, 40.0, 80.0);
        assert_relative_eq!(value, 20.0 / (70.0f64 / 50.0).ln(), epsilon = 1e-9);
        assert_relative_eq!(value, 59.4402682, epsilon = 1e-6);
    }

    #[test]
    fn parallelflow_matches_closed_form() {
        // Mêmes températures. ΔT1 = 150−40 = 110, ΔT2 = 90−80 = 10.
        // ΔT_lm = (110−10)/ln(110/10) = 100/ln(11) ≈ 41,7027 K.
        let value = lmtd_parallelflow(150.0, 90.0, 40.0, 80.0);
        assert_relative_eq!(value, 100.0 / 11.0f64.ln(), epsilon = 1e-9);
        assert_relative_eq!(value, 41.7032391, epsilon = 1e-6);
    }

    #[test]
    fn counterflow_beats_parallelflow_for_same_streams() {
        // Pour des mêmes températures d'entrée/sortie, la DTLM contre-courant
        // est supérieure à la DTLM co-courant (échangeur mieux exploité).
        let cf = lmtd_counterflow(150.0, 90.0, 40.0, 80.0);
        let pf = lmtd_parallelflow(150.0, 90.0, 40.0, 80.0);
        assert!(cf > pf);
    }

    #[test]
    fn equal_end_differences_give_that_common_value() {
        // ΔT1 = 100−70 = 30, ΔT2 = 60−30 = 30 → limite continue ΔT_lm = 30 K.
        assert_relative_eq!(
            lmtd_counterflow(100.0, 60.0, 30.0, 70.0),
            30.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn lmtd_is_bounded_by_the_two_end_differences() {
        // La moyenne logarithmique est toujours comprise entre ΔT2 et ΔT1.
        let (dt1, dt2) = (70.0, 50.0);
        let value = lmtd_counterflow(150.0, 90.0, 40.0, 80.0);
        assert!(dt2 <= value && value <= dt1);
    }

    #[test]
    fn heat_duty_is_linear_in_each_factor() {
        // Q = U·A·F·ΔT_lm. Cas chiffré : 500·10·1·40 = 200 000 W.
        let q = lmtd_heat_duty(500.0, 10.0, 40.0, 1.0);
        assert_relative_eq!(q, 200_000.0, epsilon = 1e-6);
        // Doubler U double le flux ; F = 0,9 le réduit d'exactement 10 %.
        assert_relative_eq!(
            lmtd_heat_duty(1000.0, 10.0, 40.0, 1.0),
            2.0 * q,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            lmtd_heat_duty(500.0, 10.0, 40.0, 0.9),
            0.9 * q,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "croisement de températures")]
    fn temperature_crossing_panics() {
        // hot_out (60) < cold_in (70) : ΔT2 = 60−70 < 0 → interdit.
        lmtd_counterflow(150.0, 60.0, 70.0, 80.0);
    }
}
