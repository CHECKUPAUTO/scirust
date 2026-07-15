//! Coût d'usinage par pièce — modèle additif linéaire décomposant le prix de
//! revient en coupe, amortissement de la mise en train (« setup ») et outillage.
//!
//! ```text
//! coût outillage/pièce  C_out  = C_t / N_e
//! coût par pièce        C_p    = t_m·R + t_s·R / n_b + C_t / N_e
//! coût du lot           C_lot  = C_p · n_b
//! ```
//!
//! `t_m` temps d'usinage par pièce (min), `R` taux machine + opérateur (€/min),
//! `t_s` temps de mise en train du lot (min), `n_b` taille du lot (pièces),
//! `C_t` coût d'une arête d'outil (€), `N_e` nombre de pièces produites par
//! arête (pièces), `C_out` part outillage par pièce (€), `C_p` coût par pièce
//! (€), `C_lot` coût total du lot (€).
//!
//! **Convention** : temps en minutes, unités monétaires cohérentes (mêmes €
//! partout), taux en € par minute. **Limite honnête** : modèle additif linéaire
//! à taux et coûts unitaires constants (pas d'effet d'apprentissage, ni de
//! remise, ni de saut de charge machine). Les constantes de coût — taux machine
//! `R`, coût d'arête `C_t`, pièces par arête `N_e` — ainsi que les temps sont
//! FOURNIS par l'appelant ; aucune valeur « par défaut » n'est inventée. Ce
//! module complète [`crate::economics`] (vitesses optimales de Gilbert), qui
//! fournit temps et durée de vie d'arête optimisés vis-à-vis de la loi de Taylor.

/// Part outillage du coût par pièce `C_out = C_t / N_e`.
///
/// Coût d'arête d'outil réparti sur les pièces qu'une arête permet d'usiner.
///
/// Panique si `tool_cost_per_edge < 0` ou `parts_per_edge <= 0`.
pub fn machining_cost_tooling_per_part(tool_cost_per_edge: f64, parts_per_edge: f64) -> f64 {
    assert!(
        tool_cost_per_edge >= 0.0,
        "le coût d'une arête d'outil doit être positif ou nul"
    );
    assert!(
        parts_per_edge > 0.0,
        "le nombre de pièces par arête doit être strictement positif"
    );
    tool_cost_per_edge / parts_per_edge
}

/// Coût par pièce `C_p = t_m·R + t_s·R / n_b + C_t / N_e`.
///
/// Somme de la coupe (`t_m·R`), de la mise en train amortie sur le lot
/// (`t_s·R / n_b`) et de l'outillage (`C_t / N_e`).
///
/// Panique si `machining_time < 0`, `machine_rate < 0`, `setup_time < 0`,
/// `batch_size <= 0`, `tool_cost_per_edge < 0` ou `parts_per_edge <= 0`.
pub fn machining_cost_per_part(
    machining_time: f64,
    machine_rate: f64,
    setup_time: f64,
    batch_size: f64,
    tool_cost_per_edge: f64,
    parts_per_edge: f64,
) -> f64 {
    assert!(
        machining_time >= 0.0,
        "le temps d'usinage doit être positif ou nul"
    );
    assert!(
        machine_rate >= 0.0,
        "le taux machine doit être positif ou nul"
    );
    assert!(
        setup_time >= 0.0,
        "le temps de mise en train doit être positif ou nul"
    );
    assert!(
        batch_size > 0.0,
        "la taille du lot doit être strictement positive"
    );
    let cutting = machining_time * machine_rate;
    let setup = setup_time * machine_rate / batch_size;
    let tooling = machining_cost_tooling_per_part(tool_cost_per_edge, parts_per_edge);
    cutting + setup + tooling
}

/// Coût total du lot `C_lot = C_p · n_b`.
///
/// Reconstitue la dépense de l'ensemble du lot à partir du coût par pièce.
///
/// Panique si `cost_per_part < 0` ou `batch_size <= 0`.
pub fn machining_cost_total_batch(cost_per_part: f64, batch_size: f64) -> f64 {
    assert!(
        cost_per_part >= 0.0,
        "le coût par pièce doit être positif ou nul"
    );
    assert!(
        batch_size > 0.0,
        "la taille du lot doit être strictement positive"
    );
    cost_per_part * batch_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_cost_per_part_case() {
        // t_m = 4 min, R = 1,5 €/min, t_s = 30 min, n_b = 50, C_t = 8 €, N_e = 20.
        // coupe = 4·1,5 = 6 € ; setup = 30·1,5/50 = 0,9 € ; outillage = 8/20 = 0,4 €.
        // C_p = 6 + 0,9 + 0,4 = 7,3 €.
        let cp = machining_cost_per_part(4.0, 1.5, 30.0, 50.0, 8.0, 20.0);
        assert_relative_eq!(cp, 7.3, epsilon = 1e-9);
    }

    #[test]
    fn tooling_share_matches_direct_formula() {
        // Identité : la part outillage vaut bien C_t / N_e.
        assert_relative_eq!(
            machining_cost_tooling_per_part(8.0, 20.0),
            8.0 / 20.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn batch_total_equals_sum_of_parts() {
        // Réciprocité : C_lot / n_b redonne C_p.
        let n_b = 50.0;
        let cp = machining_cost_per_part(4.0, 1.5, 30.0, n_b, 8.0, 20.0);
        let total = machining_cost_total_batch(cp, n_b);
        assert_relative_eq!(total / n_b, cp, epsilon = 1e-9);
    }

    #[test]
    fn setup_amortization_vanishes_for_large_batches() {
        // La part setup ∝ 1/n_b : décupler le lot divise sa contribution par dix.
        let cp_small = machining_cost_per_part(4.0, 1.5, 30.0, 50.0, 8.0, 20.0);
        let cp_large = machining_cost_per_part(4.0, 1.5, 30.0, 500.0, 8.0, 20.0);
        let setup_small = 30.0 * 1.5 / 50.0;
        let setup_large = 30.0 * 1.5 / 500.0;
        assert_relative_eq!(
            cp_small - cp_large,
            setup_small - setup_large,
            epsilon = 1e-9
        );
        assert_relative_eq!(setup_large, setup_small / 10.0, epsilon = 1e-9);
    }

    #[test]
    fn total_batch_is_proportional_to_batch_size() {
        // C_lot ∝ n_b à coût par pièce fixé : doubler le lot double le coût total.
        let cp = 7.3;
        let t1 = machining_cost_total_batch(cp, 50.0);
        let t2 = machining_cost_total_batch(cp, 100.0);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn cost_per_part_decomposes_into_three_terms() {
        // C_p est la somme exacte coupe + setup amorti + outillage.
        let (t_m, r, t_s, n_b, c_t, n_e) = (4.0, 1.5, 30.0, 50.0, 8.0, 20.0);
        let cp = machining_cost_per_part(t_m, r, t_s, n_b, c_t, n_e);
        let expected = t_m * r + t_s * r / n_b + machining_cost_tooling_per_part(c_t, n_e);
        assert_relative_eq!(cp, expected, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "taille du lot doit être strictement positive")]
    fn zero_batch_size_panics() {
        machining_cost_per_part(4.0, 1.5, 30.0, 0.0, 8.0, 20.0);
    }
}
