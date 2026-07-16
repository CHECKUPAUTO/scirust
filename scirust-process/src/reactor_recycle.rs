//! Réacteur avec recyclage — relations de bilan pour une boucle
//! réacteur-séparateur-recyclage en **régime permanent**, avec purge.
//!
//! ```text
//! taux de recyclage       R      = Ṙ / Ṗ_prod                          [sans dimension]
//! conversion globale      X_ov   = X_sp / (1 − f·(1 − X_sp))           [sans dimension]
//! débit entrée réacteur   Ḟ_réac = Ḟ_frais + Ṙ                         [mol/s ou kg/s]
//! fraction de purge       y_purge = Ṗ / (Ṙ + Ṗ)                        [sans dimension]
//! ```
//!
//! `R` taux de recyclage [sans dimension], `Ṙ` débit recyclé [mol/s ou kg/s],
//! `Ṗ_prod` débit de produit sortant de la boucle [mol/s ou kg/s], `X_ov`
//! conversion globale (rapportée à l'alimentation fraîche) [sans dimension],
//! `X_sp` conversion par passe dans le réacteur [sans dimension], `f` fraction
//! des non-convertis effectivement recyclée après séparation [sans dimension],
//! `Ḟ_réac` débit à l'entrée du réacteur [mol/s ou kg/s], `Ḟ_frais` débit
//! d'alimentation fraîche [mol/s ou kg/s], `y_purge` fraction du courant
//! recyclage-plus-purge qui part en purge [sans dimension], `Ṗ` débit de purge
//! [mol/s ou kg/s].
//!
//! **Limite honnête** : ces relations décrivent une **boucle
//! réacteur-séparateur-recyclage en régime permanent** par simple **comptage
//! molaire**. La **conversion par passe** `X_sp` et la **fraction de
//! non-convertis recyclée** `f` (efficacité de séparation) sont **fournies par
//! l'appelant** — elles ne sont ni calculées à partir d'une cinétique ni d'un
//! équilibre, qui relèvent d'autres modules. Un **recyclage total sans purge**
//! (`f = 1`) accumule les inertes indéfiniment : une **purge** est alors
//! indispensable, et sa **fraction est également fournie**. Aucune enthalpie,
//! constante cinétique, volatilité ou isotherme n'est inventée ici ; ce module
//! ne **résout pas** le système d'équations couplé du schéma complet.

/// Taux de recyclage `R = Ṙ / Ṗ_prod` [sans dimension].
///
/// `recycle_flow` `Ṙ` débit du courant recyclé vers le réacteur
/// [mol/s ou kg/s], `product_flow` `Ṗ_prod` débit du produit quittant la boucle
/// [mol/s ou kg/s]. `R = 4` signifie qu'on recycle quatre fois le débit de
/// produit.
///
/// Panique si `recycle_flow` est négatif ou non fini, ou si `product_flow`
/// n'est pas strictement positif (division).
pub fn recy_recycle_ratio(recycle_flow: f64, product_flow: f64) -> f64 {
    assert!(
        recycle_flow.is_finite() && recycle_flow >= 0.0,
        "le débit de recyclage doit être fini et positif ou nul (mol/s ou kg/s)"
    );
    assert!(
        product_flow > 0.0,
        "le débit de produit doit être strictement positif (mol/s ou kg/s)"
    );
    recycle_flow / product_flow
}

/// Conversion globale d'une boucle avec séparation et recyclage des
/// non-convertis `X_ov = X_sp / (1 − f·(1 − X_sp))` [sans dimension].
///
/// `single_pass_conversion` `X_sp` conversion par passe dans le réacteur
/// [sans dimension, dans `[0, 1]`], `fraction_unreacted_recycled` `f` fraction
/// des non-convertis renvoyée au réacteur par le séparateur [sans dimension,
/// dans `[0, 1]`]. La conversion globale est rapportée à l'alimentation
/// fraîche : `f = 0` (aucun recyclage) redonne `X_ov = X_sp`, tandis que
/// `f = 1` (recyclage total des non-convertis) donne `X_ov = 1`.
///
/// Panique si `single_pass_conversion` ou `fraction_unreacted_recycled` sort de
/// `[0, 1]`, ou si le dénominateur `1 − f·(1 − X_sp)` n'est pas strictement
/// positif (cas dégénéré `X_sp = 0` et `f = 1`).
pub fn recy_overall_conversion_with_separation(
    single_pass_conversion: f64,
    fraction_unreacted_recycled: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&single_pass_conversion),
        "la conversion par passe doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        (0.0..=1.0).contains(&fraction_unreacted_recycled),
        "la fraction recyclée doit être comprise dans [0, 1] (sans dimension)"
    );
    let denominator = 1.0 - fraction_unreacted_recycled * (1.0 - single_pass_conversion);
    assert!(
        denominator > 0.0,
        "le dénominateur 1 − f·(1 − X_sp) doit être strictement positif (cas dégénéré X_sp = 0 et f = 1)"
    );
    single_pass_conversion / denominator
}

/// Débit total à l'entrée du réacteur `Ḟ_réac = Ḟ_frais + Ṙ`
/// [mol/s ou kg/s], somme de l'alimentation fraîche et du recyclage au nœud de
/// mélange.
///
/// `fresh_feed` `Ḟ_frais` débit d'alimentation fraîche [mol/s ou kg/s],
/// `recycle_flow` `Ṙ` débit recyclé rejoignant l'alimentation [mol/s ou kg/s].
/// Le réacteur voit ce débit combiné, d'où une conversion par passe rapportée à
/// `Ḟ_réac` et non à `Ḟ_frais`.
///
/// Panique si l'un des débits est négatif ou non fini.
pub fn recy_reactor_feed(fresh_feed: f64, recycle_flow: f64) -> f64 {
    assert!(
        fresh_feed.is_finite() && fresh_feed >= 0.0,
        "le débit d'alimentation fraîche doit être fini et positif ou nul (mol/s ou kg/s)"
    );
    assert!(
        recycle_flow.is_finite() && recycle_flow >= 0.0,
        "le débit de recyclage doit être fini et positif ou nul (mol/s ou kg/s)"
    );
    fresh_feed + recycle_flow
}

/// Fraction du courant recyclage-plus-purge dérivée en purge
/// `y_purge = Ṗ / (Ṙ + Ṗ)` [sans dimension].
///
/// `purge_flow` `Ṗ` débit de purge [mol/s ou kg/s],
/// `recycle_plus_purge_flow` `Ṙ + Ṗ` débit total avant séparation
/// purge/recyclage [mol/s ou kg/s]. La purge évacue les inertes accumulés ;
/// la fraction complémentaire `1 − y_purge` est recyclée.
///
/// Panique si `recycle_plus_purge_flow` n'est pas strictement positif, si
/// `purge_flow` est négatif ou non fini, ou si la purge dépasse le débit total
/// (`y_purge > 1`).
pub fn recy_purge_fraction(purge_flow: f64, recycle_plus_purge_flow: f64) -> f64 {
    assert!(
        recycle_plus_purge_flow > 0.0,
        "le débit recyclage-plus-purge doit être strictement positif (mol/s ou kg/s)"
    );
    assert!(
        purge_flow.is_finite() && purge_flow >= 0.0,
        "le débit de purge doit être fini et positif ou nul (mol/s ou kg/s)"
    );
    assert!(
        purge_flow <= recycle_plus_purge_flow,
        "le débit de purge ne peut pas dépasser le débit recyclage-plus-purge"
    );
    purge_flow / recycle_plus_purge_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn recycle_ratio_known_case() {
        // Recyclage 400, produit 100 → R = 4.
        assert_relative_eq!(recy_recycle_ratio(400.0, 100.0), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn overall_conversion_no_recycle_equals_single_pass() {
        // f = 0 : aucun recyclage → la conversion globale vaut la conversion par passe.
        let x_sp = 0.27;
        assert_relative_eq!(
            recy_overall_conversion_with_separation(x_sp, 0.0),
            x_sp,
            epsilon = 1e-12
        );
    }

    #[test]
    fn overall_conversion_full_recycle_reaches_unity() {
        // f = 1 : tous les non-convertis recyclés → conversion globale = 1
        // (limite idéale sans purge). Dénominateur = 1 − 1·(1 − 0,2) = 0,2.
        assert_relative_eq!(
            recy_overall_conversion_with_separation(0.2, 1.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn overall_conversion_realistic_value() {
        // X_sp = 0,25 ; f = 0,90.
        // Dénominateur = 1 − 0,90·(1 − 0,25) = 1 − 0,90·0,75 = 1 − 0,675 = 0,325.
        // X_ov = 0,25 / 0,325 = 0,769230769230769…
        assert_relative_eq!(
            recy_overall_conversion_with_separation(0.25, 0.90),
            0.769_230_769_230_769,
            epsilon = 1e-3
        );
    }

    #[test]
    fn reactor_feed_sums_fresh_and_recycle() {
        // Alimentation fraîche 100 + recyclage 300 → 400 à l'entrée réacteur.
        assert_relative_eq!(recy_reactor_feed(100.0, 300.0), 400.0, epsilon = 1e-12);
        // Cohérence avec le taux de recyclage : R = 300/100 = 3 sur ce débit produit.
        assert_relative_eq!(recy_recycle_ratio(300.0, 100.0), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn purge_fraction_complements_recycled_fraction() {
        // Courant recyclage-plus-purge 500, purge 25 → y_purge = 0,05.
        let y_purge = recy_purge_fraction(25.0, 500.0);
        assert_relative_eq!(y_purge, 0.05, epsilon = 1e-12);
        // La fraction recyclée est le complément à 1 : 0,95.
        assert_relative_eq!(1.0 - y_purge, 0.95, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positif")]
    fn zero_product_flow_panics() {
        // Division par un débit de produit nul : rejet.
        recy_recycle_ratio(400.0, 0.0);
    }
}
