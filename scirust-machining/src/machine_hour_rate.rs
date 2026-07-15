//! Coût horaire d'une machine-outil — somme des postes de charge (amortissement,
//! énergie, maintenance, main-d'œuvre, frais généraux) rapportés à l'heure de
//! fonctionnement, tel qu'utilisé en devis et en calcul de prix de revient.
//!
//! ```text
//! Amortissement (€/h) :  d = (C − S) / H
//! Énergie       (€/h) :  e = P · lf · p_e
//! Taux horaire  (€/h) :  R = d + e + m + w + o
//! ```
//!
//! Légende :
//! - `C`   coût d'acquisition de la machine (€)
//! - `S`   valeur résiduelle / de revente en fin de vie (€), `0 ≤ S ≤ C`
//! - `H`   durée de vie amortissable (h), `H > 0`
//! - `P`   puissance installée (kW)
//! - `lf`  taux de charge moyen (facteur d'utilisation de la puissance), `0 ≤ lf ≤ 1`
//! - `p_e` prix de l'énergie (€/kWh)
//! - `d`   amortissement horaire (€/h)
//! - `e`   coût énergétique horaire (€/h)
//! - `m`   maintenance horaire (€/h)
//! - `w`   main-d'œuvre horaire (€/h)
//! - `o`   frais généraux (overhead) horaires (€/h)
//! - `R`   taux horaire machine (€/h)
//!
//! **Limite honnête** : l'amortissement est supposé **linéaire** (répartition
//! uniforme sur `H`). Le taux de charge `lf`, le prix de l'énergie `p_e` et tous
//! les postes de coût (`C`, `S`, `m`, `w`, `o`) sont **fournis par l'appelant** :
//! ce module n'invente aucune valeur « par défaut » de coût, de tarif ou de
//! rendement. L'unité monétaire (ici notée €) est arbitraire mais doit rester
//! cohérente entre tous les arguments.

/// Amortissement horaire linéaire `d = (C − S) / H` (€/h).
///
/// `purchase_cost` `C` et `salvage_value` `S` sont en unité monétaire cohérente,
/// `life_hours` `H` en heures.
///
/// Panique si `life_hours <= 0`, si `purchase_cost < 0`, si `salvage_value < 0`
/// ou si `salvage_value > purchase_cost`.
pub fn machine_depreciation_per_hour(
    purchase_cost: f64,
    salvage_value: f64,
    life_hours: f64,
) -> f64 {
    assert!(
        purchase_cost >= 0.0,
        "le coût d'acquisition doit être positif ou nul"
    );
    assert!(
        salvage_value >= 0.0,
        "la valeur résiduelle doit être positive ou nulle"
    );
    assert!(
        salvage_value <= purchase_cost,
        "la valeur résiduelle ne peut pas dépasser le coût d'acquisition"
    );
    assert!(
        life_hours > 0.0,
        "la durée de vie amortissable doit être strictement positive"
    );
    (purchase_cost - salvage_value) / life_hours
}

/// Coût énergétique horaire `e = P · lf · p_e` (€/h).
///
/// `power_kw` `P` est la puissance installée (kW), `load_factor` `lf` le taux de
/// charge moyen (fraction sans dimension entre 0 et 1) et `energy_price_per_kwh`
/// `p_e` le prix de l'énergie (€/kWh).
///
/// Panique si `power_kw < 0`, si `load_factor` n'est pas dans `[0, 1]` ou si
/// `energy_price_per_kwh < 0`.
pub fn machine_power_cost_per_hour(
    power_kw: f64,
    load_factor: f64,
    energy_price_per_kwh: f64,
) -> f64 {
    assert!(
        power_kw >= 0.0,
        "la puissance installée doit être positive ou nulle"
    );
    assert!(
        (0.0..=1.0).contains(&load_factor),
        "le taux de charge doit être compris entre 0 et 1"
    );
    assert!(
        energy_price_per_kwh >= 0.0,
        "le prix de l'énergie doit être positif ou nul"
    );
    power_kw * load_factor * energy_price_per_kwh
}

/// Taux horaire machine `R = d + e + m + w + o` (€/h), somme des postes de charge
/// déjà exprimés par heure de fonctionnement.
///
/// `depreciation_ph` `d`, `power_ph` `e`, `maintenance_ph` `m`, `labour_ph` `w`
/// et `overhead_ph` `o` sont tous en €/h.
///
/// Panique si l'un des postes est strictement négatif.
pub fn machine_hour_rate(
    depreciation_ph: f64,
    power_ph: f64,
    maintenance_ph: f64,
    labour_ph: f64,
    overhead_ph: f64,
) -> f64 {
    assert!(
        depreciation_ph >= 0.0,
        "l'amortissement horaire doit être positif ou nul"
    );
    assert!(
        power_ph >= 0.0,
        "le coût énergétique horaire doit être positif ou nul"
    );
    assert!(
        maintenance_ph >= 0.0,
        "la maintenance horaire doit être positive ou nulle"
    );
    assert!(
        labour_ph >= 0.0,
        "la main-d'œuvre horaire doit être positive ou nulle"
    );
    assert!(
        overhead_ph >= 0.0,
        "les frais généraux horaires doivent être positifs ou nuls"
    );
    depreciation_ph + power_ph + maintenance_ph + labour_ph + overhead_ph
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn depreciation_recovers_capital_over_life() {
        // d · H doit restituer exactement le capital amortissable (C − S).
        let (c, s, h) = (120_000.0_f64, 20_000.0_f64, 40_000.0_f64);
        let d = machine_depreciation_per_hour(c, s, h);
        assert_relative_eq!(d * h, c - s, epsilon = 1e-9);
    }

    #[test]
    fn depreciation_is_zero_when_salvage_equals_purchase() {
        // Rien à amortir si la machine est revendue à son prix d'achat.
        assert_relative_eq!(
            machine_depreciation_per_hour(50_000.0, 50_000.0, 10_000.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn power_cost_is_linear_in_load_factor() {
        // e ∝ lf : doubler le taux de charge double le coût énergétique.
        let e1 = machine_power_cost_per_hour(30.0, 0.4, 0.15);
        let e2 = machine_power_cost_per_hour(30.0, 0.8, 0.15);
        assert_relative_eq!(e2, 2.0 * e1, epsilon = 1e-9);
    }

    #[test]
    fn power_cost_matches_hand_computation() {
        // 30 kW · 0,5 · 0,20 €/kWh = 3,00 €/h.
        assert_relative_eq!(
            machine_power_cost_per_hour(30.0, 0.5, 0.20),
            3.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn hour_rate_is_the_sum_of_its_posts() {
        // R est additif : la somme des cinq postes.
        let d = machine_depreciation_per_hour(120_000.0, 20_000.0, 40_000.0); // 2,5 €/h
        let e = machine_power_cost_per_hour(30.0, 0.5, 0.20); // 3,0 €/h
        let (m, w, o) = (4.0_f64, 25.0_f64, 6.0_f64);
        let r = machine_hour_rate(d, e, m, w, o);
        assert_relative_eq!(r, d + e + m + w + o, epsilon = 1e-9);
        assert_relative_eq!(r, 40.5, epsilon = 1e-9);
    }

    #[test]
    fn hour_rate_reduces_to_zero_with_no_charges() {
        // Cas limite dégénéré : aucun poste ⇒ taux nul.
        assert_relative_eq!(
            machine_hour_rate(0.0, 0.0, 0.0, 0.0, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "durée de vie amortissable doit être strictement positive")]
    fn zero_life_panics() {
        machine_depreciation_per_hour(100_000.0, 10_000.0, 0.0);
    }
}
