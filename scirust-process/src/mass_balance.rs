//! Bilan matière en **régime permanent** — conservation de la masse sur une
//! opération unitaire, un nœud de mélange, un recyclage, une purge ou un
//! diviseur de courant.
//!
//! ```text
//! bilan global      ΔM = Σ(entrées) − Σ(sorties)             [kg/s ou mol/s]
//! débit partiel     ṁᵢ = F·xᵢ − Σ(autres sorties du const.)  [kg/s ou mol/s]
//! taux de recyclage R  = Ṙ / Ḟ                               [sans dimension]
//! débit de purge    Ṗ  = (Ṙ+Ṗ) − Ṙ                          [kg/s ou mol/s]
//! fraction de split φ  = ṁ_branche / ṁ_total                 [sans dimension]
//! ```
//!
//! `ΔM` déséquilibre du bilan (nul à l'équilibre), `Σ(entrées)`/`Σ(sorties)`
//! sommes des débits massiques ou molaires [kg/s ou mol/s], `F` débit d'entrée
//! [kg/s ou mol/s], `xᵢ` fraction (massique ou molaire) du constituant dans
//! l'entrée [sans dimension], `ṁᵢ` débit partiel du constituant obtenu par
//! différence [kg/s ou mol/s], `Ṙ` débit de recyclage [kg/s ou mol/s], `Ḟ`
//! débit d'alimentation fraîche [kg/s ou mol/s], `R` taux de recyclage [sans
//! dimension], `Ṗ` débit de purge [kg/s ou mol/s], `φ` fraction dérivée par une
//! branche d'un diviseur [sans dimension].
//!
//! **Limite honnête** : ces relations traduisent la **conservation de la masse
//! en régime permanent, sans accumulation ni réaction chimique** (un bilan de
//! comptage). Les débits et les fractions sont **fournis par l'appelant** (d'après
//! des mesures ou des spécifications procédé) ; aucune propriété n'est inventée.
//! Pour un procédé **réactif**, il faut employer un bilan par constituant intégrant
//! le **taux de conversion fourni** (production/consommation), que ces fonctions
//! ne modélisent pas. Elles donnent les **relations élémentaires** : elles ne
//! **résolvent pas** les systèmes d'équations couplés d'un schéma avec recyclage.

/// Déséquilibre du bilan matière global `ΔM = Σ(entrées) − Σ(sorties)`
/// [kg/s ou mol/s], **nul** en régime permanent conservatif.
///
/// `inputs` débits entrants, `outputs` débits sortants (mêmes unités, kg/s ou
/// mol/s). Une valeur positive signale un excès d'entrée (accumulation), une
/// valeur négative un excès de sortie.
///
/// Panique si un débit est négatif ou non fini.
pub fn massbal_overall_steady(inputs: &[f64], outputs: &[f64]) -> f64 {
    for &m in inputs.iter().chain(outputs.iter())
    {
        assert!(
            m.is_finite() && m >= 0.0,
            "chaque débit doit être fini et positif ou nul (kg/s ou mol/s)"
        );
    }
    let sum_in: f64 = inputs.iter().sum();
    let sum_out: f64 = outputs.iter().sum();
    sum_in - sum_out
}

/// Débit partiel d'un constituant obtenu **par différence**
/// `ṁᵢ = F·xᵢ − Σ(autres sorties du constituant)` [kg/s ou mol/s].
///
/// `input_flow` `F` débit total d'entrée [kg/s ou mol/s], `input_fraction` `xᵢ`
/// fraction (massique ou molaire) du constituant dans l'entrée [sans dimension,
/// dans `[0, 1]`], `other_outputs_component` débits partiels **du même
/// constituant** déjà connus dans les autres courants sortants [kg/s ou mol/s].
/// Le résultat ferme le bilan du constituant sur le courant restant.
///
/// Panique si `input_flow` est négatif ou non fini, si `input_fraction` sort de
/// `[0, 1]`, ou si un débit partiel fourni est négatif ou non fini.
pub fn massbal_component_output(
    input_flow: f64,
    input_fraction: f64,
    other_outputs_component: &[f64],
) -> f64 {
    assert!(
        input_flow.is_finite() && input_flow >= 0.0,
        "le débit d'entrée doit être fini et positif ou nul (kg/s ou mol/s)"
    );
    assert!(
        (0.0..=1.0).contains(&input_fraction),
        "la fraction d'entrée doit être comprise dans [0, 1] (sans dimension)"
    );
    for &m in other_outputs_component
    {
        assert!(
            m.is_finite() && m >= 0.0,
            "chaque débit partiel sortant doit être fini et positif ou nul (kg/s ou mol/s)"
        );
    }
    let others: f64 = other_outputs_component.iter().sum();
    input_flow * input_fraction - others
}

/// Taux de recyclage `R = Ṙ / Ḟ` [sans dimension].
///
/// `recycle_flow` `Ṙ` débit recyclé [kg/s ou mol/s], `fresh_feed_flow` `Ḟ` débit
/// d'alimentation fraîche [kg/s ou mol/s]. `R = 2` signifie qu'on recycle deux
/// fois le débit frais.
///
/// Panique si `recycle_flow` est négatif ou non fini, ou si `fresh_feed_flow`
/// n'est pas strictement positif (division).
pub fn massbal_recycle_ratio(recycle_flow: f64, fresh_feed_flow: f64) -> f64 {
    assert!(
        recycle_flow.is_finite() && recycle_flow >= 0.0,
        "le débit de recyclage doit être fini et positif ou nul (kg/s ou mol/s)"
    );
    assert!(
        fresh_feed_flow > 0.0,
        "le débit d'alimentation fraîche doit être strictement positif (kg/s ou mol/s)"
    );
    recycle_flow / fresh_feed_flow
}

/// Débit de purge `Ṗ = (Ṙ+Ṗ) − Ṙ` [kg/s ou mol/s], soustrait à un courant de
/// recyclage-plus-purge pour isoler la fraction purgée.
///
/// `recycle_plus_purge_flow` `Ṙ+Ṗ` débit total avant séparation purge/recyclage
/// [kg/s ou mol/s], `recycle_flow` `Ṙ` débit effectivement recyclé
/// [kg/s ou mol/s]. Le reste part en purge.
///
/// Panique si un débit est négatif ou non fini, ou si le débit recyclé dépasse
/// le débit total (purge négative impossible).
pub fn massbal_purge_flow(recycle_plus_purge_flow: f64, recycle_flow: f64) -> f64 {
    assert!(
        recycle_plus_purge_flow.is_finite() && recycle_plus_purge_flow >= 0.0,
        "le débit recyclage-plus-purge doit être fini et positif ou nul (kg/s ou mol/s)"
    );
    assert!(
        recycle_flow.is_finite() && recycle_flow >= 0.0,
        "le débit de recyclage doit être fini et positif ou nul (kg/s ou mol/s)"
    );
    assert!(
        recycle_flow <= recycle_plus_purge_flow,
        "le débit recyclé ne peut pas dépasser le débit recyclage-plus-purge"
    );
    recycle_plus_purge_flow - recycle_flow
}

/// Fraction dérivée par une branche d'un diviseur de courant
/// `φ = ṁ_branche / ṁ_total` [sans dimension].
///
/// `branch_flow` `ṁ_branche` débit de la branche considérée [kg/s ou mol/s],
/// `total_flow` `ṁ_total` débit entrant dans le diviseur [kg/s ou mol/s]. Un
/// diviseur ne modifie pas la composition : `φ` s'applique identiquement à
/// chaque constituant.
///
/// Panique si `total_flow` n'est pas strictement positif, si `branch_flow` est
/// négatif ou non fini, ou si la branche dépasse le débit total (`φ > 1`).
pub fn massbal_splitter_fraction(branch_flow: f64, total_flow: f64) -> f64 {
    assert!(
        total_flow > 0.0,
        "le débit total doit être strictement positif (kg/s ou mol/s)"
    );
    assert!(
        branch_flow.is_finite() && branch_flow >= 0.0,
        "le débit de branche doit être fini et positif ou nul (kg/s ou mol/s)"
    );
    assert!(
        branch_flow <= total_flow,
        "le débit de branche ne peut pas dépasser le débit total"
    );
    branch_flow / total_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overall_balance_closes_at_steady_state() {
        // Entrées 10 + 5 = 15 ; sorties 8 + 7 = 15 → déséquilibre nul.
        let inputs = [10.0, 5.0];
        let outputs = [8.0, 7.0];
        assert_relative_eq!(
            massbal_overall_steady(&inputs, &outputs),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn overall_balance_is_antisymmetric() {
        // Échanger entrées et sorties change le signe du déséquilibre.
        let a = [12.0, 3.0, 1.0];
        let b = [4.0, 4.0];
        let forward = massbal_overall_steady(&a, &b);
        let backward = massbal_overall_steady(&b, &a);
        assert_relative_eq!(forward, -backward, epsilon = 1e-12);
        // Ici : (16) − (8) = 8, et (8) − (16) = −8.
        assert_relative_eq!(forward, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn component_output_by_difference_known_case() {
        // F = 100, x = 0,4 → constituant à l'entrée = 40 mol/s.
        // Autres sorties du constituant = 15 + 10 = 25 → reste = 40 − 25 = 15.
        let others = [15.0, 10.0];
        assert_relative_eq!(
            massbal_component_output(100.0, 0.4, &others),
            15.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn component_output_is_zero_when_others_absorb_all() {
        // Si les autres sorties emportent tout le constituant (F·x = Σ),
        // le courant restant en contient zéro.
        let feed_component = 60.0 * 0.5; // = 30
        let others = [20.0, 10.0]; // Σ = 30
        assert_relative_eq!(
            massbal_component_output(60.0, 0.5, &others),
            feed_component - 30.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            massbal_component_output(60.0, 0.5, &others),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn recycle_ratio_and_purge_are_consistent() {
        // Recyclage 300, alimentation fraîche 100 → R = 3.
        assert_relative_eq!(massbal_recycle_ratio(300.0, 100.0), 3.0, epsilon = 1e-12);
        // Courant recyclage-plus-purge 320, recyclé 300 → purge 20.
        assert_relative_eq!(massbal_purge_flow(320.0, 300.0), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn splitter_fractions_sum_to_one() {
        // Un diviseur partage un débit total de 10 en deux branches 4 et 6.
        let total = 10.0;
        let phi_a = massbal_splitter_fraction(4.0, total);
        let phi_b = massbal_splitter_fraction(6.0, total);
        assert_relative_eq!(phi_a, 0.4, epsilon = 1e-12);
        assert_relative_eq!(phi_a + phi_b, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positif")]
    fn zero_fresh_feed_panics() {
        // Division par un débit d'alimentation fraîche nul : rejet.
        massbal_recycle_ratio(300.0, 0.0);
    }
}
