//! Taux de rendement synthétique (**TRS / OEE**) — disponibilité, performance,
//! qualité et leur produit.
//!
//! ```text
//! disponibilité  D = temps de fonctionnement / temps requis
//! performance    P = (temps de cycle idéal · quantité produite) / temps de fonctionnement
//! qualité        Q = pièces bonnes / quantité produite
//! TRS (OEE)      TRS = D · P · Q
//! ```
//!
//! `temps requis` temps d'ouverture planifié, `temps de fonctionnement` temps
//! réel de marche (hors arrêts), `temps de cycle idéal` cadence nominale, `D`,
//! `P`, `Q` fractions dans `[0, 1]`. Le TRS est la fraction de temps réellement
//! productive et conforme.
//!
//! **Convention** : temps cohérents, ratios sans dimension. **Limite honnête** :
//! définition standard du TRS (produit des trois taux) ; la ventilation des
//! pertes (arrêts, micro-arrêts, rebuts) et la classification des temps sont à la
//! charge de l'appelant.

/// Taux de **disponibilité** `D = temps de fonctionnement / temps requis`.
///
/// Panique si `required_time <= 0`.
pub fn availability(operating_time: f64, required_time: f64) -> f64 {
    assert!(
        required_time > 0.0,
        "le temps requis doit être strictement positif"
    );
    operating_time / required_time
}

/// Taux de **performance** `P = (T_cycle_idéal · quantité) / temps de fonctionnement`.
///
/// Panique si `operating_time <= 0`.
pub fn performance(ideal_cycle_time: f64, total_count: f64, operating_time: f64) -> f64 {
    assert!(
        operating_time > 0.0,
        "le temps de fonctionnement doit être strictement positif"
    );
    ideal_cycle_time * total_count / operating_time
}

/// Taux de **qualité** `Q = pièces bonnes / quantité produite`.
///
/// Panique si `total_count <= 0`.
pub fn quality(good_count: f64, total_count: f64) -> f64 {
    assert!(
        total_count > 0.0,
        "la quantité produite doit être strictement positive"
    );
    good_count / total_count
}

/// TRS (OEE) `= D · P · Q`.
pub fn oee(availability: f64, performance: f64, quality: f64) -> f64 {
    availability * performance * quality
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn three_rates_and_their_product() {
        // Exemple classique : D=0,9, P=0,95, Q=0,99 → TRS ≈ 0,846.
        assert_relative_eq!(availability(432.0, 480.0), 0.9, epsilon = 1e-9);
        // T_cycle=1 min, 456 pièces sur 480 min de marche → P = 0,95.
        assert_relative_eq!(performance(1.0, 456.0, 480.0), 0.95, epsilon = 1e-9);
        // 451 bonnes sur 456 → Q ≈ 0,989.
        assert_relative_eq!(quality(451.0, 456.0), 451.0 / 456.0, epsilon = 1e-9);
    }

    #[test]
    fn oee_is_product_of_rates() {
        assert_relative_eq!(oee(0.9, 0.95, 0.99), 0.9 * 0.95 * 0.99, epsilon = 1e-12);
    }

    #[test]
    fn perfect_process_reaches_unity() {
        assert_relative_eq!(oee(1.0, 1.0, 1.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn oee_never_exceeds_its_factors() {
        // Le TRS est toujours ≤ à chacun de ses facteurs (tous dans [0,1]).
        let (a, p, q) = (0.9, 0.95, 0.99);
        let trs = oee(a, p, q);
        assert!(trs <= a);
        assert!(trs <= p);
        assert!(trs <= q);
    }

    #[test]
    #[should_panic(expected = "temps requis")]
    fn zero_required_time_panics() {
        availability(432.0, 0.0);
    }
}
