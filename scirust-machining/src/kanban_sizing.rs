//! Dimensionnement d'une boucle Kanban en flux tiré — nombre de cartes,
//! point de recommande, stock de sécurité et en-cours maximal d'une boucle.
//!
//! ```text
//! nombre de cartes    N   = D · L · (1 + k) / Q
//! point de recommande ROP = D · L + Ss
//! en-cours maximal    Imax = N · Q
//! stock de sécurité   Ss  = D · L · k
//! ```
//!
//! `D` demande moyenne (taux de consommation, p. ex. pièces/h), `L` délai de
//! réapprovisionnement moyen (lead time, même unité de temps que `D`, p. ex. h),
//! `k` coefficient de sécurité (sans dimension, ≥ 0), `Q` taille de conteneur
//! (nombre de pièces par carte/conteneur), `N` nombre de cartes ou conteneurs
//! (sans dimension), `ROP` point de recommande (en pièces), `Ss` stock de
//! sécurité (en pièces), `Imax` en-cours maximal de la boucle (en pièces).
//!
//! **Convention** : unités de temps cohérentes entre `D` et `L` (si `D` est en
//! pièces/h alors `L` est en heures) ; `D · L` a alors la dimension d'un nombre
//! de pièces.
//!
//! **Limite honnête** : demande et délai de réapprovisionnement sont supposés
//! STABLES, leurs MOYENNES étant FOURNIES par l'appelant (mesures ou objectifs) ;
//! le coefficient de sécurité est FOURNI par la politique d'atelier. Le modèle est
//! déterministe : il n'inclut aucune valeur « par défaut » inventée et ne rend pas
//! compte de la variabilité réelle de la demande ou des délais, laquelle exige une
//! analyse statistique (lois de distribution, niveau de service, etc.).

/// Nombre de cartes/conteneurs `N = D · L · (1 + k) / Q` d'une boucle Kanban.
///
/// Couverture de la demande sur le délai de réapprovisionnement, majorée du
/// coefficient de sécurité, rapportée à la taille de conteneur.
///
/// Panique si `demand_rate <= 0`, `lead_time <= 0`, `safety_factor < 0` ou
/// `container_size <= 0`.
pub fn kanban_card_count(
    demand_rate: f64,
    lead_time: f64,
    safety_factor: f64,
    container_size: f64,
) -> f64 {
    assert!(
        demand_rate > 0.0,
        "la demande doit être strictement positive"
    );
    assert!(
        lead_time > 0.0,
        "le délai de réapprovisionnement doit être strictement positif"
    );
    assert!(
        safety_factor >= 0.0,
        "le coefficient de sécurité doit être positif ou nul"
    );
    assert!(
        container_size > 0.0,
        "la taille de conteneur doit être strictement positive"
    );
    demand_rate * lead_time * (1.0 + safety_factor) / container_size
}

/// Point de recommande `ROP = D · L + Ss` d'une boucle Kanban.
///
/// Niveau de stock déclenchant le réapprovisionnement : consommation pendant le
/// délai augmentée du stock de sécurité.
///
/// Panique si `demand_rate <= 0`, `lead_time <= 0` ou `safety_stock < 0`.
pub fn kanban_reorder_point(demand_rate: f64, lead_time: f64, safety_stock: f64) -> f64 {
    assert!(
        demand_rate > 0.0,
        "la demande doit être strictement positive"
    );
    assert!(
        lead_time > 0.0,
        "le délai de réapprovisionnement doit être strictement positif"
    );
    assert!(
        safety_stock >= 0.0,
        "le stock de sécurité doit être positif ou nul"
    );
    demand_rate * lead_time + safety_stock
}

/// En-cours maximal `Imax = N · Q` d'une boucle Kanban.
///
/// Quantité de pièces plafond dans la boucle : nombre de cartes multiplié par la
/// taille de conteneur.
///
/// Panique si `card_count <= 0` ou `container_size <= 0`.
pub fn kanban_max_inventory(card_count: f64, container_size: f64) -> f64 {
    assert!(
        card_count > 0.0,
        "le nombre de cartes doit être strictement positif"
    );
    assert!(
        container_size > 0.0,
        "la taille de conteneur doit être strictement positive"
    );
    card_count * container_size
}

/// Stock de sécurité `Ss = D · L · k` d'une boucle Kanban.
///
/// Couverture supplémentaire dimensionnée par le coefficient de sécurité appliqué
/// à la consommation pendant le délai de réapprovisionnement.
///
/// Panique si `demand_rate <= 0`, `lead_time <= 0` ou `safety_factor < 0`.
pub fn kanban_safety_stock(demand_rate: f64, lead_time: f64, safety_factor: f64) -> f64 {
    assert!(
        demand_rate > 0.0,
        "la demande doit être strictement positive"
    );
    assert!(
        lead_time > 0.0,
        "le délai de réapprovisionnement doit être strictement positif"
    );
    assert!(
        safety_factor >= 0.0,
        "le coefficient de sécurité doit être positif ou nul"
    );
    demand_rate * lead_time * safety_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_card_count_case() {
        // Ligne : D = 50 pièces/h, L = 4 h, k = 0,2, Q = 25 pièces/conteneur.
        // N = 50 · 4 · (1 + 0,2) / 25 = 200 · 1,2 / 25 = 240 / 25 = 9,6.
        let n = kanban_card_count(50.0, 4.0, 0.2, 25.0);
        assert_relative_eq!(n, 9.6, epsilon = 1e-9);
    }

    #[test]
    fn max_inventory_reconstructs_covered_demand() {
        // Imax = N · Q = D · L · (1 + k) : l'en-cours plafond retrouve la
        // couverture majorée, indépendamment de la taille de conteneur.
        let d = 50.0;
        let l = 4.0;
        let k = 0.2;
        let q = 25.0;
        let n = kanban_card_count(d, l, k, q);
        let imax = kanban_max_inventory(n, q);
        assert_relative_eq!(imax, d * l * (1.0 + k), epsilon = 1e-9);
    }

    #[test]
    fn reorder_point_with_safety_stock_equals_max_inventory() {
        // Cohérence des deux chemins : ROP = D·L + Ss avec Ss = D·L·k vaut
        // D·L·(1 + k), soit exactement l'en-cours maximal de la boucle.
        let d = 50.0;
        let l = 4.0;
        let k = 0.2;
        let q = 25.0;
        let ss = kanban_safety_stock(d, l, k);
        let rop = kanban_reorder_point(d, l, ss);
        let n = kanban_card_count(d, l, k, q);
        let imax = kanban_max_inventory(n, q);
        assert_relative_eq!(rop, imax, epsilon = 1e-9);
    }

    #[test]
    fn safety_stock_proportional_to_safety_factor() {
        // Ss ∝ k à demande et délai constants : doubler le coefficient double le
        // stock de sécurité.
        let d = 30.0;
        let l = 5.0;
        let s1 = kanban_safety_stock(d, l, 0.15);
        let s2 = kanban_safety_stock(d, l, 0.30);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-12);
    }

    #[test]
    fn card_count_proportional_to_demand_rate() {
        // N ∝ D à délai, coefficient et conteneur constants : tripler la demande
        // triple le nombre de cartes.
        let l = 3.0;
        let k = 0.1;
        let q = 12.0;
        let n1 = kanban_card_count(20.0, l, k, q);
        let n2 = kanban_card_count(60.0, l, k, q);
        assert_relative_eq!(n2, 3.0 * n1, epsilon = 1e-12);
    }

    #[test]
    fn zero_safety_factor_reduces_to_bare_coverage() {
        // Cas limite k = 0 : plus de stock de sécurité, ROP = D·L et
        // N = D·L / Q (couverture nue du délai).
        let d = 40.0;
        let l = 2.0;
        let q = 16.0;
        assert_relative_eq!(kanban_safety_stock(d, l, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(kanban_reorder_point(d, l, 0.0), d * l, epsilon = 1e-9);
        assert_relative_eq!(kanban_card_count(d, l, 0.0, q), d * l / q, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la taille de conteneur doit être strictement positive")]
    fn non_positive_container_size_panics() {
        kanban_card_count(50.0, 4.0, 0.2, 0.0);
    }
}
