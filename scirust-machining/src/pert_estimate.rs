//! Estimation **PERT** de durées d'activités probabilistes (approximation
//! bêta-PERT à pondération 1-4-1).
//!
//! ```text
//! durée espérée      te    = (o + 4·m + p) / 6
//! variance           σ²    = ((p − o) / 6)²
//! écart-type         σ     = (p − o) / 6
//! score normal réduit z     = (T − te) / σ
//! ```
//!
//! `o` durée optimiste, `m` durée la plus probable (mode), `p` durée pessimiste,
//! `te` durée espérée, `σ` écart-type, `σ²` variance, `T` durée cible, `z` score
//! normal réduit ; toutes les durées dans la même unité de temps (s, min, h ou
//! jours, au choix cohérent de l'appelant), `z` sans dimension.
//!
//! **Limite honnête** : approximation bêta-PERT (pondération 1-4-1) ; les
//! estimations `o`/`m`/`p` sont FOURNIES par l'expertise (jamais inventées ici) ;
//! sommer les variances le long d'un chemin suppose l'indépendance des activités ;
//! le score `z` se convertit en probabilité d'achèvement via une table de la loi
//! normale centrée réduite, NON incluse et FOURNIE par l'appelant.

/// Durée espérée `te = (o + 4·m + p) / 6` (espérance de la loi bêta-PERT).
///
/// `optimistic` `o`, `most_likely` `m` (mode), `pessimistic` `p`, même unité de temps.
///
/// Panique si l'ordre `o <= m <= p` n'est pas respecté.
pub fn pert_expected_time(optimistic: f64, most_likely: f64, pessimistic: f64) -> f64 {
    assert!(
        optimistic <= most_likely,
        "la durée optimiste doit être <= la durée la plus probable"
    );
    assert!(
        most_likely <= pessimistic,
        "la durée la plus probable doit être <= la durée pessimiste"
    );
    (optimistic + 4.0 * most_likely + pessimistic) / 6.0
}

/// Variance `σ² = ((p − o) / 6)²` de la durée d'activité.
///
/// `optimistic` `o`, `pessimistic` `p`, même unité de temps (variance en temps²).
///
/// Panique si `pessimistic < optimistic`.
pub fn pert_variance(optimistic: f64, pessimistic: f64) -> f64 {
    assert!(
        pessimistic >= optimistic,
        "la durée pessimiste doit être >= la durée optimiste"
    );
    let range = (pessimistic - optimistic) / 6.0;
    range * range
}

/// Écart-type `σ = (p − o) / 6` de la durée d'activité.
///
/// `optimistic` `o`, `pessimistic` `p`, même unité de temps.
///
/// Panique si `pessimistic < optimistic`.
pub fn pert_standard_deviation(optimistic: f64, pessimistic: f64) -> f64 {
    assert!(
        pessimistic >= optimistic,
        "la durée pessimiste doit être >= la durée optimiste"
    );
    (pessimistic - optimistic) / 6.0
}

/// Score normal réduit `z = (T − te) / σ` pour une durée cible `T`.
///
/// `target_time` `T`, `expected_time` `te`, `standard_deviation` `σ` (> 0), mêmes
/// unités de temps ; `z` sans dimension, à convertir en probabilité par une table
/// de la loi normale centrée réduite (FOURNIE par l'appelant).
///
/// Panique si `standard_deviation <= 0`.
pub fn pert_z_score(target_time: f64, expected_time: f64, standard_deviation: f64) -> f64 {
    assert!(
        standard_deviation > 0.0,
        "l'écart-type doit être strictement positif"
    );
    (target_time - expected_time) / standard_deviation
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn worked_example_o2_m5_p14() {
        // o=2, m=5, p=14 : te=(2+20+14)/6=36/6=6 ; σ=(14−2)/6=2 ; σ²=4.
        assert_relative_eq!(pert_expected_time(2.0, 5.0, 14.0), 6.0, epsilon = 1e-12);
        assert_relative_eq!(pert_standard_deviation(2.0, 14.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(pert_variance(2.0, 14.0), 4.0, epsilon = 1e-12);
        // Cible T=8 : z=(8−6)/2=1.
        assert_relative_eq!(pert_z_score(8.0, 6.0, 2.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn variance_is_squared_standard_deviation() {
        // Identité σ² = σ² : cohérence entre les deux fonctions.
        let (o, p) = (3.0, 21.0);
        let sigma = pert_standard_deviation(o, p);
        assert_relative_eq!(pert_variance(o, p), sigma * sigma, epsilon = 1e-12);
    }

    #[test]
    fn symmetric_estimates_give_mode_as_expected_time() {
        // Si m est au milieu de [o, p], la durée espérée vaut m (loi symétrique).
        let (o, m, p) = (4.0, 10.0, 16.0);
        assert_relative_eq!(pert_expected_time(o, m, p), m, epsilon = 1e-12);
    }

    #[test]
    fn zero_uncertainty_gives_zero_spread() {
        // o = p : aucune incertitude → σ = 0 et σ² = 0.
        assert_relative_eq!(pert_standard_deviation(7.0, 7.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(pert_variance(7.0, 7.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn z_score_at_expected_time_is_zero() {
        // Cible = durée espérée → z = 0 (probabilité d'achèvement 50 %).
        assert_relative_eq!(pert_z_score(6.0, 6.0, 2.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn path_variance_adds_under_independence() {
        // Le long d'un chemin, les variances s'additionnent (activités indépendantes).
        let v1 = pert_variance(2.0, 14.0); // 4
        let v2 = pert_variance(3.0, 9.0); // 1
        let sigma_path = (v1 + v2).sqrt();
        assert_relative_eq!(sigma_path, 5.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "durée pessimiste doit être >= la durée optimiste")]
    fn variance_with_inverted_range_panics() {
        pert_variance(10.0, 2.0);
    }
}
