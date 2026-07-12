//! Chaînes de cotes — cumul **arithmétique** (pire cas) et **statistique** (RSS)
//! des tolérances le long d'une chaîne, cote de fermeture et jeux extrêmes.
//!
//! ```text
//! cote de fermeture (nominal)  N = Σ Ni      (Ni signés : + maillon récepteur, − donneur)
//! pire cas (arithmétique)      T_wc = Σ |ti|
//! statistique (RSS)            T_rss = √(Σ ti²)
//! extrêmes                     N ± T
//! ```
//!
//! `Ni` cotes nominales **signées** des maillons (contribution positive ou
//! négative à la cote de fermeture), `ti` demi-intervalles de tolérance
//! (symétriques `±ti`, tous positifs). Le cumul **pire cas** garantit
//! l'interchangeabilité totale ; le cumul **RSS** (hypothèse de tolérances
//! indépendantes centrées) est moins pénalisant mais admet un faible taux de
//! rebut.
//!
//! **Convention** : unités cohérentes de l'appelant. **Limite honnête** : chaîne
//! **linéaire** (coefficients de sensibilité unitaires ±1) ; le RSS suppose des
//! contributeurs **indépendants** de tolérance symétrique. Le tolérancement
//! inertiel/statistique avancé et les capabilités Cp/Cpk relèvent de la crate
//! `scirust-tolerance`.

/// Cote de fermeture nominale `N = Σ Ni` (cotes signées).
pub fn closing_nominal(signed_nominals: &[f64]) -> f64 {
    signed_nominals.iter().sum()
}

/// Tolérance cumulée **pire cas** `T_wc = Σ |ti|`.
pub fn worst_case_tolerance(half_tolerances: &[f64]) -> f64 {
    half_tolerances.iter().map(|t| t.abs()).sum()
}

/// Tolérance cumulée **statistique** (RSS) `T_rss = √(Σ ti²)`.
pub fn rss_tolerance(half_tolerances: &[f64]) -> f64 {
    half_tolerances.iter().map(|t| t * t).sum::<f64>().sqrt()
}

/// Valeur maximale de la cote de fermeture `N + T`.
pub fn closing_max(nominal: f64, half_tolerance: f64) -> f64 {
    nominal + half_tolerance
}

/// Valeur minimale de la cote de fermeture `N − T`.
pub fn closing_min(nominal: f64, half_tolerance: f64) -> f64 {
    nominal - half_tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn closing_nominal_sums_signed_links() {
        // Arbre 50 (+), deux entretoises −20 et −20, alésage... jeu = 50 − 20 − 20 = 10.
        assert_relative_eq!(closing_nominal(&[50.0, -20.0, -20.0]), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn worst_case_sums_absolute_tolerances() {
        // ±0,1, ±0,05, ±0,05 → T_wc = 0,2.
        assert_relative_eq!(
            worst_case_tolerance(&[0.1, 0.05, 0.05]),
            0.2,
            epsilon = 1e-9
        );
    }

    #[test]
    fn rss_is_below_worst_case() {
        // RSS = √(0,01+0,0025+0,0025) = √0,015 ≈ 0,1225 < 0,2.
        let tols = [0.1, 0.05, 0.05];
        let rss = rss_tolerance(&tols);
        assert_relative_eq!(rss, 0.015f64.sqrt(), epsilon = 1e-12);
        assert!(rss < worst_case_tolerance(&tols));
    }

    #[test]
    fn extremes_bracket_the_nominal() {
        // Jeu 10 ±0,2 → [9,8 ; 10,2].
        let n = closing_nominal(&[50.0, -20.0, -20.0]);
        let t = worst_case_tolerance(&[0.1, 0.05, 0.05]);
        assert_relative_eq!(closing_min(n, t), 9.8, epsilon = 1e-9);
        assert_relative_eq!(closing_max(n, t), 10.2, epsilon = 1e-9);
    }

    #[test]
    fn identical_links_scale_rss_by_sqrt_n() {
        // n maillons identiques ±t → RSS = t·√n.
        let rss = rss_tolerance(&[0.1, 0.1, 0.1, 0.1]);
        assert_relative_eq!(rss, 0.1 * 4.0f64.sqrt(), epsilon = 1e-12);
    }
}
