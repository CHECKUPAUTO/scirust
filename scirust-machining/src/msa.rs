//! MSA — analyse du système de mesure (**Gage R&R**) : répétabilité,
//! reproductibilité, variation totale, %R&R et nombre de catégories distinctes.
//!
//! ```text
//! R&R              GRR = √(EV² + AV²)         (répétabilité ⊕ reproductibilité)
//! variation totale TV  = √(GRR² + PV²)         (PV = variation des pièces)
//! pourcentage R&R  %R&R = 100·GRR/TV
//! catégories distinctes  ndc = 1,41·(PV/GRR)   (tronqué)
//! ```
//!
//! `EV` répétabilité (equipment variation, écart-type du même opérateur/pièce),
//! `AV` reproductibilité (appraiser variation, écart entre opérateurs), `GRR`
//! variation du système de mesure, `PV` variation réelle des pièces, `TV`
//! variation totale. Un `%R&R < 10 %` est acceptable, `> 30 %` inacceptable ;
//! `ndc ≥ 5` est recommandé.
//!
//! **Convention** : `EV`, `AV`, `PV` sont des **écarts-types** (ou étendues déjà
//! converties) dans l'unité de mesure. **Limite honnête** : combinaison
//! **quadratique** des composantes de variance (hypothèse d'indépendance) ; le
//! dépouillement d'un plan R&R (moyennes/étendues ou ANOVA) à partir des données
//! brutes n'est pas fait ici — l'appelant fournit les composantes.

/// Variation du système de mesure `GRR = √(EV² + AV²)`.
pub fn gage_rr(repeatability_ev: f64, reproducibility_av: f64) -> f64 {
    (repeatability_ev * repeatability_ev + reproducibility_av * reproducibility_av).sqrt()
}

/// Variation totale `TV = √(GRR² + PV²)`.
pub fn total_variation(gage_rr: f64, part_variation_pv: f64) -> f64 {
    (gage_rr * gage_rr + part_variation_pv * part_variation_pv).sqrt()
}

/// Pourcentage R&R `%R&R = 100·GRR/TV`.
///
/// Panique si `total_variation <= 0`.
pub fn percent_rr(gage_rr: f64, total_variation: f64) -> f64 {
    assert!(
        total_variation > 0.0,
        "la variation totale doit être strictement positive"
    );
    100.0 * gage_rr / total_variation
}

/// Nombre de catégories distinctes `ndc = 1,41·(PV/GRR)` (valeur non tronquée).
///
/// Panique si `gage_rr <= 0`.
pub fn number_distinct_categories(part_variation_pv: f64, gage_rr: f64) -> f64 {
    assert!(
        gage_rr > 0.0,
        "la variation R&R doit être strictement positive"
    );
    1.41 * part_variation_pv / gage_rr
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn grr_combines_repeatability_and_reproducibility() {
        // EV=3, AV=4 → GRR = 5.
        assert_relative_eq!(gage_rr(3.0, 4.0), 5.0, epsilon = 1e-9);
    }

    #[test]
    fn total_variation_includes_parts() {
        // GRR=5, PV=12 → TV = 13.
        assert_relative_eq!(total_variation(5.0, 12.0), 13.0, epsilon = 1e-9);
    }

    #[test]
    fn percent_rr_and_acceptance() {
        // GRR=5, TV=13 → %R&R ≈ 38,5 % (inacceptable, > 30 %).
        let pct = percent_rr(5.0, 13.0);
        assert_relative_eq!(pct, 100.0 * 5.0 / 13.0, epsilon = 1e-9);
        assert!(pct > 30.0);
    }

    #[test]
    fn ndc_from_part_and_rr() {
        // PV=12, GRR=5 → ndc = 1,41·12/5 = 3,384 (< 5, insuffisant).
        let ndc = number_distinct_categories(12.0, 5.0);
        assert_relative_eq!(ndc, 1.41 * 12.0 / 5.0, epsilon = 1e-9);
        assert!(ndc < 5.0);
    }

    #[test]
    fn better_gage_gives_more_categories() {
        // Réduire le GRR augmente ndc (meilleure discrimination).
        assert!(number_distinct_categories(12.0, 2.0) > number_distinct_categories(12.0, 5.0));
    }

    #[test]
    #[should_panic(expected = "variation totale")]
    fn zero_total_variation_panics() {
        percent_rr(5.0, 0.0);
    }
}
