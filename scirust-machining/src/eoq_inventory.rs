//! Gestion de stock — quantité économique de commande (modèle de **Wilson**),
//! coût total, point de commande et nombre de commandes annuel.
//!
//! ```text
//! quantité économique   Q* = sqrt(2 · D · S / H)
//! coût total de stock    C(Q) = D · S / Q + Q · H / 2
//! point de commande      ROP = d · L
//! nombre de commandes    N = D / Q
//! ```
//!
//! `D` demande annuelle (unités/an), `S` coût fixe par commande (€/commande),
//! `H` coût de possession par unité et par an (€/(unité·an)), `Q` quantité
//! commandée (unités), `d` demande journalière (unités/jour), `L` délai de
//! réapprovisionnement (jours), `ROP` niveau de stock déclenchant une commande
//! (unités), `N` fréquence de commande (commandes/an). Le coût `C(Q)` est en
//! €/an. À l'optimum `Q*`, coût de commande et coût de possession s'égalisent.
//!
//! **Convention** : unités cohérentes (mêmes bases de temps entre `D`, `H` et
//! `N`, entre `d` et `L`). **Limite honnête** : hypothèses de Wilson — demande
//! constante et connue, réapprovisionnement instantané, pas de rupture ni de
//! remise sur quantité. Les coûts `S`, `H`, la demande `D`/`d` et le délai `L`
//! sont FOURNIS par l'appelant ; aucune valeur « par défaut » n'est inventée.

/// Quantité économique de commande de Wilson `Q* = sqrt(2 · D · S / H)`.
///
/// Panique si `annual_demand < 0`, `order_cost < 0` ou `holding_cost_per_unit <= 0`.
pub fn eoq_economic_order_quantity(
    annual_demand: f64,
    order_cost: f64,
    holding_cost_per_unit: f64,
) -> f64 {
    assert!(
        annual_demand >= 0.0,
        "la demande annuelle doit être positive ou nulle"
    );
    assert!(
        order_cost >= 0.0,
        "le coût de commande doit être positif ou nul"
    );
    assert!(
        holding_cost_per_unit > 0.0,
        "le coût de possession doit être strictement positif"
    );
    (2.0 * annual_demand * order_cost / holding_cost_per_unit).sqrt()
}

/// Coût total annuel de stock `C(Q) = D · S / Q + Q · H / 2`.
///
/// Somme du coût de commande `D · S / Q` et du coût de possession `Q · H / 2`.
///
/// Panique si `order_quantity <= 0`, ou si `annual_demand`, `order_cost` ou
/// `holding_cost_per_unit` est négatif.
pub fn inventory_total_cost(
    annual_demand: f64,
    order_cost: f64,
    holding_cost_per_unit: f64,
    order_quantity: f64,
) -> f64 {
    assert!(
        annual_demand >= 0.0,
        "la demande annuelle doit être positive ou nulle"
    );
    assert!(
        order_cost >= 0.0,
        "le coût de commande doit être positif ou nul"
    );
    assert!(
        holding_cost_per_unit >= 0.0,
        "le coût de possession doit être positif ou nul"
    );
    assert!(
        order_quantity > 0.0,
        "la quantité commandée doit être strictement positive"
    );
    annual_demand * order_cost / order_quantity + order_quantity * holding_cost_per_unit / 2.0
}

/// Point de commande `ROP = d · L`.
///
/// Niveau de stock qui déclenche une nouvelle commande, égal à la demande
/// écoulée pendant le délai de réapprovisionnement.
///
/// Panique si `daily_demand < 0` ou `lead_time_days < 0`.
pub fn inventory_reorder_point(daily_demand: f64, lead_time_days: f64) -> f64 {
    assert!(
        daily_demand >= 0.0,
        "la demande journalière doit être positive ou nulle"
    );
    assert!(
        lead_time_days >= 0.0,
        "le délai de réapprovisionnement doit être positif ou nul"
    );
    daily_demand * lead_time_days
}

/// Nombre de commandes par an `N = D / Q`.
///
/// Panique si `order_quantity <= 0` ou `annual_demand < 0`.
pub fn eoq_number_of_orders(annual_demand: f64, order_quantity: f64) -> f64 {
    assert!(
        annual_demand >= 0.0,
        "la demande annuelle doit être positive ou nulle"
    );
    assert!(
        order_quantity > 0.0,
        "la quantité commandée doit être strictement positive"
    );
    annual_demand / order_quantity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn eoq_realistic_case() {
        // D = 800 unités/an, S = 100 €/commande, H = 4 €/(unité·an).
        // Q* = sqrt(2·800·100/4) = sqrt(40000) = 200 unités.
        let q = eoq_economic_order_quantity(800.0, 100.0, 4.0);
        assert_relative_eq!(q, 200.0, epsilon = 1e-9);
    }

    #[test]
    fn at_optimum_ordering_equals_holding_cost() {
        // À Q*, coût de commande D·S/Q* = coût de possession Q*·H/2.
        let (d, s, h) = (1200.0, 80.0, 6.0);
        let q = eoq_economic_order_quantity(d, s, h);
        let ordering = d * s / q;
        let holding = q * h / 2.0;
        assert_relative_eq!(ordering, holding, epsilon = 1e-9);
    }

    #[test]
    fn total_cost_minimal_at_eoq() {
        // Le coût total en Q* vaut sqrt(2·D·S·H) et minore toute autre quantité.
        let (d, s, h) = (1000.0, 50.0, 4.0);
        let q = eoq_economic_order_quantity(d, s, h);
        let c_star = inventory_total_cost(d, s, h, q);
        assert_relative_eq!(c_star, (2.0 * d * s * h).sqrt(), epsilon = 1e-9);
        // Toute quantité voisine coûte davantage.
        assert!(inventory_total_cost(d, s, h, q * 0.5) > c_star);
        assert!(inventory_total_cost(d, s, h, q * 2.0) > c_star);
    }

    #[test]
    fn eoq_scales_as_sqrt_of_demand() {
        // Q* ∝ sqrt(D) : quadrupler la demande double la quantité économique.
        let q1 = eoq_economic_order_quantity(500.0, 50.0, 4.0);
        let q4 = eoq_economic_order_quantity(2000.0, 50.0, 4.0);
        assert_relative_eq!(q4, 2.0 * q1, epsilon = 1e-9);
    }

    #[test]
    fn number_of_orders_is_reciprocal_of_cycle() {
        // N = D/Q, et à Q* le nombre de commandes redonne D/Q*.
        let (d, s, h) = (1000.0, 50.0, 4.0);
        let q = eoq_economic_order_quantity(d, s, h);
        assert_relative_eq!(eoq_number_of_orders(d, q), d / q, epsilon = 1e-9);
        // Cohérence : N · Q = D.
        assert_relative_eq!(eoq_number_of_orders(d, q) * q, d, epsilon = 1e-9);
    }

    #[test]
    fn reorder_point_is_demand_over_lead_time() {
        // d = 20 unités/jour, L = 7 jours → ROP = 140 unités.
        assert_relative_eq!(inventory_reorder_point(20.0, 7.0), 140.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "coût de possession doit être strictement positif")]
    fn zero_holding_cost_panics() {
        eoq_economic_order_quantity(1000.0, 50.0, 0.0);
    }
}
