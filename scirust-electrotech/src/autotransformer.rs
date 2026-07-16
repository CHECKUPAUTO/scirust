//! **Autotransformateur idéal** — rapport de tension, puissance totale transitée,
//! puissance réellement transformée par les enroulements (« puissance propre ») et
//! économie de cuivre par rapport à un transformateur à deux enroulements.
//!
//! ```text
//! rapport de tension        r      = (N_s + N_c) / N_c
//! puissance transitée       S_th   = V_out · I_out
//! puissance des enroulements S_w    = S_th · (1 − 1/r)
//! économie de cuivre        k_cu   = 1/r
//! ```
//!
//! `N_s` nombre de spires de l'enroulement **série** et `N_c` de l'enroulement
//! **commun** (sans dimension, `> 0`), `r` rapport de tension élévateur (sans
//! dimension, `> 1` pour un élévateur, `= 1` à la limite d'un rapport unité),
//! `V_out` tension de sortie (V), `I_out` courant de sortie (A), `S_th` puissance
//! apparente totale transitée par la ligne (VA), `S_w` puissance apparente
//! réellement transformée par couplage magnétique des enroulements (VA), `k_cu`
//! économie de cuivre relative (sans dimension, `∈ ]0, 1]`).
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances apparentes en
//! VA ; grandeurs efficaces en **régime sinusoïdal permanent**. **Limite
//! honnête** : **autotransformateur idéal** (pas de pertes, magnétisation
//! linéaire) ; les nombres de spires **série/commune** sont **fournis par
//! l'appelant** (construction de la machine), aucune valeur n'est inventée.
//! L'économie de cuivre et de matière est **d'autant plus grande que le rapport
//! est proche de 1** (`S_w` et `k_cu` tendent alors vers 0 et 1). **Limite de
//! sécurité** : un autotransformateur n'assure **aucune isolation galvanique**
//! entre primaire et secondaire (enroulement commun partagé).

/// Rapport de tension `r = (N_s + N_c) / N_c` (élévateur, spires série + commune
/// sur commune).
///
/// Panique si `series_turns < 0` ou si `common_turns <= 0`.
pub fn autoxfmr_voltage_ratio(series_turns: f64, common_turns: f64) -> f64 {
    assert!(series_turns >= 0.0, "N_s ≥ 0 requis");
    assert!(common_turns > 0.0, "N_c > 0 requis");
    (series_turns + common_turns) / common_turns
}

/// Puissance apparente totale transitée `S_th = V_out · I_out` (VA).
///
/// Panique si `output_voltage < 0` ou si `output_current < 0`.
pub fn autoxfmr_throughput_power(output_voltage: f64, output_current: f64) -> f64 {
    assert!(output_voltage >= 0.0, "V_out ≥ 0 requis");
    assert!(output_current >= 0.0, "I_out ≥ 0 requis");
    output_voltage * output_current
}

/// Puissance propre des enroulements `S_w = S_th · (1 − 1/r)` (VA) — part
/// réellement transformée par couplage magnétique.
///
/// Panique si `throughput_power < 0` ou si `voltage_ratio < 1`.
pub fn autoxfmr_winding_power(throughput_power: f64, voltage_ratio: f64) -> f64 {
    assert!(throughput_power >= 0.0, "S_th ≥ 0 requis");
    assert!(voltage_ratio >= 1.0, "r ≥ 1 requis");
    throughput_power * (1.0 - 1.0 / voltage_ratio)
}

/// Économie de cuivre relative `k_cu = 1/r` par rapport à un transformateur à deux
/// enroulements (proche de 1 quand le rapport tend vers 1).
///
/// Panique si `voltage_ratio < 1`.
pub fn autoxfmr_copper_saving(voltage_ratio: f64) -> f64 {
    assert!(voltage_ratio >= 1.0, "r ≥ 1 requis");
    1.0 / voltage_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unit_ratio_when_no_series_winding() {
        // Cas limite : pas de spire série → r = 1, aucune puissance propre,
        // économie de cuivre maximale (k_cu = 1).
        let r = autoxfmr_voltage_ratio(0.0, 100.0);
        assert_relative_eq!(r, 1.0, epsilon = 1e-15);
        assert_relative_eq!(autoxfmr_winding_power(12_000.0, r), 0.0, epsilon = 1e-12);
        assert_relative_eq!(autoxfmr_copper_saving(r), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn winding_and_copper_saving_are_complementary() {
        // Identité : S_w/S_th = 1 − k_cu, car (1 − 1/r) + 1/r = 1.
        let r = autoxfmr_voltage_ratio(30.0, 120.0); // r = 1,25
        let s_th = autoxfmr_throughput_power(400.0, 25.0);
        let s_w = autoxfmr_winding_power(s_th, r);
        let k_cu = autoxfmr_copper_saving(r);
        assert_relative_eq!(s_w / s_th + k_cu, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn winding_power_shrinks_as_ratio_nears_one() {
        // Proportionnalité : plus le rapport est proche de 1, plus S_w est faible.
        let s_th = 10_000.0_f64;
        let s_near = autoxfmr_winding_power(s_th, 1.1);
        let s_far = autoxfmr_winding_power(s_th, 2.0);
        assert!(s_near < s_far, "S_w doit décroître vers un rapport unité");
    }

    #[test]
    fn realistic_step_up_autotransformer() {
        // Cas chiffré : série 20 spires, commune 100 spires → r = 120/100 = 1,2.
        let r = autoxfmr_voltage_ratio(20.0, 100.0);
        assert_relative_eq!(r, 1.2, epsilon = 1e-12);
        // Sortie 240 V, 50 A → S_th = 240 · 50 = 12 000 VA.
        let s_th = autoxfmr_throughput_power(240.0, 50.0);
        assert_relative_eq!(s_th, 12_000.0, epsilon = 1e-9);
        // Puissance propre : 12 000 · (1 − 1/1,2) = 12 000 · (1/6) = 2 000 VA.
        assert_relative_eq!(autoxfmr_winding_power(s_th, r), 2_000.0, epsilon = 1e-6);
        // Économie de cuivre : 1/1,2 = 0,833333…
        assert_relative_eq!(autoxfmr_copper_saving(r), 1.0 / 1.2, epsilon = 1e-12);
    }

    #[test]
    fn ratio_two_transforms_half_the_power() {
        // Cas chiffré : série 100, commune 100 → r = 2, S_w = S_th/2, k_cu = 0,5.
        let r = autoxfmr_voltage_ratio(100.0, 100.0);
        assert_relative_eq!(r, 2.0, epsilon = 1e-12);
        assert_relative_eq!(autoxfmr_winding_power(8_000.0, r), 4_000.0, epsilon = 1e-9);
        assert_relative_eq!(autoxfmr_copper_saving(r), 0.5, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "r ≥ 1 requis")]
    fn ratio_below_one_panics() {
        autoxfmr_copper_saving(0.8);
    }
}
