//! **Rendement global d'un réducteur multi-étages** — produit des rendements
//! d'étage, puissance transmise/perdue et nombre d'étages nécessaires pour un
//! rapport de réduction donné.
//!
//! ```text
//! rendement global      η = η_1 · η_2 · … · η_n = Π η_i
//! puissance de sortie   P_out   = P · η
//! puissance perdue      P_perte = P · (1 − η)
//! nombre d'étages       n = ⌈ ln(R) / ln(r_max) ⌉
//! ```
//!
//! `η_i` rendement de l'étage `i` (sans dimension, `∈ [0, 1]`), `η` rendement
//! global (sans dimension), `P` puissance d'entrée (W), `P_out`/`P_perte`
//! puissances de sortie/perdue (W), `R` rapport de réduction total (sans
//! dimension, `≥ 1`), `r_max` rapport de réduction maximal par étage (sans
//! dimension, `> 1`), `n` nombre d'étages (entier).
//!
//! **Convention** : SI ; puissances en watts ; rendements et rapports sans
//! dimension. **Limite honnête** : le modèle suppose des étages **indépendants**
//! dont les rendements se multiplient ; les rendements d'étage `η_i ∈ [0, 1]`, le
//! rapport total `R` et le rapport maximal par étage `r_max` sont **fournis par
//! l'appelant** (issus d'essais, de la lubrification, de la charge et de la
//! cinématique réelles) — aucune valeur « par défaut » n'est inventée. Complète
//! [`crate::gear_efficiency`] qui traite un **seul** engrènement.

/// Rendement global d'un train multi-étages `η = Π η_i`, produit des rendements
/// d'étage supposés indépendants.
///
/// Panique si `stage_efficiencies` est vide ou si un rendement d'étage n'est pas
/// dans `[0, 1]`.
pub fn gearbox_overall_efficiency(stage_efficiencies: &[f64]) -> f64 {
    assert!(
        !stage_efficiencies.is_empty(),
        "au moins un étage requis (tranche non vide)"
    );
    let mut product = 1.0_f64;
    for &eta in stage_efficiencies
    {
        assert!(
            (0.0..=1.0).contains(&eta),
            "chaque rendement d'étage η_i ∈ [0, 1] requis"
        );
        product *= eta;
    }
    product
}

/// Puissance transmise en sortie du réducteur `P_out = P · η`.
///
/// Panique si `input_power < 0` ou si `overall_efficiency` n'est pas dans `[0, 1]`.
pub fn gearbox_output_power(input_power: f64, overall_efficiency: f64) -> f64 {
    assert!(input_power >= 0.0, "P ≥ 0 requis");
    assert!(
        (0.0..=1.0).contains(&overall_efficiency),
        "η ∈ [0, 1] requis"
    );
    input_power * overall_efficiency
}

/// Puissance dissipée dans le réducteur `P_perte = P · (1 − η)`.
///
/// Panique si `input_power < 0` ou si `overall_efficiency` n'est pas dans `[0, 1]`.
pub fn gearbox_power_loss(input_power: f64, overall_efficiency: f64) -> f64 {
    assert!(input_power >= 0.0, "P ≥ 0 requis");
    assert!(
        (0.0..=1.0).contains(&overall_efficiency),
        "η ∈ [0, 1] requis"
    );
    input_power * (1.0 - overall_efficiency)
}

/// Nombre minimal d'étages `n = ⌈ ln(R) / ln(r_max) ⌉` pour atteindre le rapport
/// total `R` sans dépasser le rapport `r_max` par étage.
///
/// Panique si `total_ratio < 1` ou si `max_ratio_per_stage <= 1`.
pub fn gearbox_stages_for_ratio(total_ratio: f64, max_ratio_per_stage: f64) -> u32 {
    assert!(total_ratio >= 1.0, "R ≥ 1 requis");
    assert!(max_ratio_per_stage > 1.0, "r_max > 1 requis");
    // Tolérance relative pour que R = r_max^n (puissance exacte) ne soit pas
    // sur-comptée par l'erreur d'arrondi de ln (ex. ln(125)/ln(5) ≈ 3 + 4e-16).
    let raw = total_ratio.ln() / max_ratio_per_stage.ln();
    let count = (raw - 1e-9).ceil();
    // R == 1 → ceil(0) == 0, mais un réducteur réel comporte au moins un étage.
    (count as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overall_efficiency_is_the_product() {
        // Identité : le produit de rendements identiques vaut la puissance.
        let eta = 0.98_f64;
        assert_relative_eq!(
            gearbox_overall_efficiency(&[eta, eta, eta]),
            eta.powi(3),
            epsilon = 1e-15
        );
    }

    #[test]
    fn perfect_stages_give_unit_efficiency() {
        // Cas limite : tous les étages parfaits → η = 1.
        assert_relative_eq!(
            gearbox_overall_efficiency(&[1.0, 1.0, 1.0, 1.0]),
            1.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn output_plus_loss_conserves_power() {
        // Conservation : P_out + P_perte = P quel que soit η.
        let power = 7_500.0_f64;
        let eta = 0.94_f64;
        assert_relative_eq!(
            gearbox_output_power(power, eta) + gearbox_power_loss(power, eta),
            power,
            epsilon = 1e-9
        );
    }

    #[test]
    fn output_power_scales_realistic_case() {
        // Cas chiffré : réducteur à deux étages 0.97 et 0.96 sur 10 kW.
        let eta = gearbox_overall_efficiency(&[0.97, 0.96]);
        assert_relative_eq!(eta, 0.9312, epsilon = 1e-12);
        assert_relative_eq!(gearbox_output_power(10_000.0, eta), 9_312.0, epsilon = 1e-9);
        assert_relative_eq!(gearbox_power_loss(10_000.0, eta), 688.0, epsilon = 1e-9);
    }

    #[test]
    fn stages_cover_the_ratio() {
        // Réciprocité : n étages à r_max doivent couvrir R = r_max^n exactement.
        let r_max = 5.0_f64;
        assert_eq!(gearbox_stages_for_ratio(r_max.powi(3), r_max), 3);
        // Un dépassement d'un cran impose un étage de plus.
        assert_eq!(gearbox_stages_for_ratio(r_max.powi(3) + 1.0, r_max), 4);
        // R = 1 : au moins un étage physique.
        assert_eq!(gearbox_stages_for_ratio(1.0, r_max), 1);
    }

    #[test]
    #[should_panic(expected = "η ∈ [0, 1] requis")]
    fn efficiency_above_one_panics() {
        gearbox_output_power(1_000.0, 1.2);
    }
}
