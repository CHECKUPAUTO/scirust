//! Échangeurs de chaleur — dimensionnement par la **DTLM** (différence de
//! température logarithmique moyenne) et par la méthode **NUT-efficacité**
//! (ε-NTU) pour les configurations co-courant et contre-courant.
//!
//! ```text
//! DTLM                 ΔT_lm = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)
//! flux (DTLM)          Q = U·A·ΔT_lm
//! NUT                  NTU = U·A/C_min
//! rapport de capacités Cr = C_min/C_max
//! efficacité contre-courant ε = (1 − e^{−NTU(1−Cr)})/(1 − Cr·e^{−NTU(1−Cr)})
//!   (Cr = 1)               ε = NTU/(1 + NTU)
//! efficacité co-courant     ε = (1 − e^{−NTU(1+Cr)})/(1 + Cr)
//! flux réel            Q = ε·C_min·(T_ch,e − T_fr,e)
//! ```
//!
//! `U` coefficient global d'échange (W/(m²·K)), `A` surface (m²), `C = ṁ·cp`
//! débit de capacité thermique (W/K), `ΔT1, ΔT2` écarts de température aux deux
//! extrémités. `C_min`/`C_max` sont le plus petit/grand des deux débits de
//! capacité.
//!
//! **Convention** : SI cohérent, écarts de température en K. **Limite honnête** :
//! régime **permanent**, `U` constant, pas de changement de phase ni de pertes
//! vers l'extérieur ; les corrélations d'efficacité valent pour les deux
//! dispositions simples co-courant et contre-courant.

/// Différence de température logarithmique moyenne
/// `ΔT_lm = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)` (K).
///
/// Renvoie la limite `ΔT1` quand `ΔT1 ≈ ΔT2`. Panique si un écart est `≤ 0`.
pub fn lmtd(delta_t1: f64, delta_t2: f64) -> f64 {
    assert!(
        delta_t1 > 0.0 && delta_t2 > 0.0,
        "les écarts de température doivent être strictement positifs"
    );
    if (delta_t1 / delta_t2 - 1.0).abs() < 1e-6
    {
        return 0.5 * (delta_t1 + delta_t2);
    }
    (delta_t1 - delta_t2) / (delta_t1 / delta_t2).ln()
}

/// Flux échangé par la DTLM `Q = U·A·ΔT_lm` (W).
pub fn heat_duty_lmtd(u_w_m2k: f64, area_m2: f64, lmtd_k: f64) -> f64 {
    u_w_m2k * area_m2 * lmtd_k
}

/// Nombre d'unités de transfert `NTU = U·A/C_min`.
///
/// Panique si `c_min <= 0`.
pub fn ntu(u_w_m2k: f64, area_m2: f64, c_min_w_k: f64) -> f64 {
    assert!(c_min_w_k > 0.0, "C_min doit être strictement positif");
    u_w_m2k * area_m2 / c_min_w_k
}

/// Rapport des débits de capacité `Cr = C_min/C_max` (dans `[0, 1]`).
///
/// Panique si `c_max <= 0` ou si `c_min > c_max`.
pub fn capacity_ratio(c_min_w_k: f64, c_max_w_k: f64) -> f64 {
    assert!(c_max_w_k > 0.0, "C_max doit être strictement positif");
    assert!(c_min_w_k <= c_max_w_k, "C_min ne peut pas dépasser C_max");
    c_min_w_k / c_max_w_k
}

/// Efficacité d'un échangeur **contre-courant** par la méthode ε-NTU.
///
/// Panique si `cr` sort de `[0, 1]` ou `ntu < 0`.
pub fn effectiveness_counterflow(ntu: f64, cr: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&cr) && ntu >= 0.0,
        "0 ≤ Cr ≤ 1 et NTU ≥ 0 requis"
    );
    if (cr - 1.0).abs() < 1e-12
    {
        return ntu / (1.0 + ntu);
    }
    let e = (-ntu * (1.0 - cr)).exp();
    (1.0 - e) / (1.0 - cr * e)
}

/// Efficacité d'un échangeur **co-courant** par la méthode ε-NTU
/// `ε = (1 − e^{−NTU(1+Cr)})/(1 + Cr)`.
///
/// Panique si `cr` sort de `[0, 1]` ou `ntu < 0`.
pub fn effectiveness_parallel_flow(ntu: f64, cr: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&cr) && ntu >= 0.0,
        "0 ≤ Cr ≤ 1 et NTU ≥ 0 requis"
    );
    (1.0 - (-ntu * (1.0 + cr)).exp()) / (1.0 + cr)
}

/// Flux réellement échangé `Q = ε·C_min·(T_ch,e − T_fr,e)` (W), à partir des
/// températures d'**entrée** chaude et froide.
pub fn actual_heat_transfer(
    effectiveness: f64,
    c_min_w_k: f64,
    t_hot_in_k: f64,
    t_cold_in_k: f64,
) -> f64 {
    effectiveness * c_min_w_k * (t_hot_in_k - t_cold_in_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lmtd_of_unequal_ends() {
        // ΔT1=50, ΔT2=30 → ΔT_lm = 20/ln(5/3) ≈ 39,15 K.
        assert_relative_eq!(
            lmtd(50.0, 30.0),
            20.0 / (50.0f64 / 30.0).ln(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn lmtd_equal_ends_is_the_common_value() {
        // ΔT1 = ΔT2 = 40 → ΔT_lm = 40 (limite continue).
        assert_relative_eq!(lmtd(40.0, 40.0), 40.0, epsilon = 1e-9);
    }

    #[test]
    fn ntu_and_capacity_ratio() {
        // U=500, A=10, C_min=2000 → NTU = 2,5 ; Cr = 2000/4000 = 0,5.
        assert_relative_eq!(ntu(500.0, 10.0, 2000.0), 2.5, epsilon = 1e-12);
        assert_relative_eq!(capacity_ratio(2000.0, 4000.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn counterflow_beats_parallel_flow() {
        // À NTU et Cr égaux, le contre-courant est plus efficace.
        let (n, cr) = (2.5, 0.5);
        assert!(effectiveness_counterflow(n, cr) > effectiveness_parallel_flow(n, cr));
    }

    #[test]
    fn balanced_counterflow_special_case() {
        // Cr=1 : ε = NTU/(1+NTU) = 2,5/3,5.
        assert_relative_eq!(
            effectiveness_counterflow(2.5, 1.0),
            2.5 / 3.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn phase_change_limit_both_configs_agree() {
        // Cr=0 (condenseur/évaporateur) : ε = 1 − e^{−NTU} dans les deux cas.
        let n = 1.5_f64;
        let target = 1.0 - (-n).exp();
        assert_relative_eq!(effectiveness_counterflow(n, 0.0), target, epsilon = 1e-12);
        assert_relative_eq!(effectiveness_parallel_flow(n, 0.0), target, epsilon = 1e-12);
    }

    #[test]
    fn actual_heat_uses_inlet_temperature_difference() {
        // ε=0,7, C_min=2000, ΔT_max = 80−20 = 60 → Q = 84000 W.
        assert_relative_eq!(
            actual_heat_transfer(0.7, 2000.0, 80.0, 20.0),
            84_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "C_min ne peut pas dépasser")]
    fn swapped_capacities_panic() {
        capacity_ratio(4000.0, 2000.0);
    }
}
