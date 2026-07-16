//! Réactions **multiples** (parallèles ou en série) — **rendement** et
//! **sélectivité** : rendement global, sélectivité désiré/indésiré, sélectivité
//! instantanée (rapport des vitesses) et rendement obtenu à partir de la
//! conversion et de la sélectivité fractionnelle.
//!
//! ```text
//! rendement global        Y  = n_D / n_A,cons                       [sans dimension]
//! sélectivité (globale)   S  = n_D / n_U                            [sans dimension]
//! rendement = conv × sél  Y  = X · S_frac                           [sans dimension]
//! sélectivité instantanée s  = r_D / r_U                            [sans dimension]
//! ```
//!
//! `Y` rendement (moles de produit désiré formé par mole de réactif limitant
//! consommé, ou par mole chargée selon la définition) [sans dimension],
//! `n_D` quantité de **produit désiré formé** [mol], `n_A,cons` quantité de
//! **réactif limitant consommé** [mol], `S` sélectivité **globale** désiré/indésiré
//! [sans dimension], `n_U` quantité de **produit indésiré formé** [mol],
//! `X` taux de **conversion** du réactif limitant [sans dimension, `[0, 1]`],
//! `S_frac` sélectivité **fractionnelle** (fraction du réactif consommé orientée
//! vers le produit désiré) [sans dimension, `[0, 1]`], `s` sélectivité
//! **instantanée** (rapport local des vitesses) [sans dimension], `r_D`/`r_U`
//! vitesses de formation des produits désiré/indésiré [mol·m⁻³·s⁻¹, même unité].
//!
//! **Limite honnête** : ces relations ne sont que les **définitions molaires** du
//! rendement et de la sélectivité pour des réactions **parallèles ou en série** ;
//! aucun mécanisme, aucune stœchiométrie ni aucune loi cinétique n'est déduit. Les
//! **quantités molaires** (`n_D`, `n_U`, `n_A,cons`), la **conversion** `X`, la
//! **sélectivité fractionnelle** `S_frac` et les **vitesses** (`r_D`, `r_U`) sont
//! **fournies par l'appelant** (bilans matière, mesures, cinétique intégrée) ;
//! aucune valeur n'est inventée. La **sélectivité instantanée** dépend des
//! **concentrations locales**, donc du **type de réacteur** et du profil de
//! mélange : c'est à l'appelant de fournir des vitesses cohérentes avec son point
//! de fonctionnement. Les **définitions molaires** (base « consommé » ou base
//! « chargé », prise en compte de la stœchiométrie) doivent être **cohérentes** et
//! sont à la **charge de l'appelant** ; le module se contente d'évaluer le rapport
//! ou le produit demandé.

/// Rendement **global** en produit désiré : `Y = n_D / n_A,cons`
/// (moles de produit désiré formé par mole de réactif limitant consommé)
/// [sans dimension].
///
/// `desired_product_formed` `n_D` quantité de produit désiré **formé** [mol],
/// `limiting_reactant_consumed` `n_A,cons` quantité de réactif limitant
/// **consommé** [mol]. La base molaire (avec ou sans facteur stœchiométrique) est à
/// la charge de l'appelant.
///
/// Panique si `desired_product_formed` est négatif ou non fini, ou si
/// `limiting_reactant_consumed` n'est pas strictement positif ou n'est pas fini
/// (division).
pub fn yieldsel_overall_yield(desired_product_formed: f64, limiting_reactant_consumed: f64) -> f64 {
    assert!(
        desired_product_formed.is_finite() && desired_product_formed >= 0.0,
        "la quantité de produit désiré formé doit être finie et positive ou nulle (mol)"
    );
    assert!(
        limiting_reactant_consumed.is_finite() && limiting_reactant_consumed > 0.0,
        "la quantité de réactif limitant consommé doit être finie et strictement positive (mol)"
    );
    desired_product_formed / limiting_reactant_consumed
}

/// Sélectivité **globale** désiré/indésiré : `S = n_D / n_U`
/// (moles de produit désiré formé par mole de produit indésiré formé)
/// [sans dimension].
///
/// `desired_product_formed` `n_D` quantité de produit désiré **formé** [mol],
/// `undesired_product_formed` `n_U` quantité de produit indésiré **formé** [mol].
/// Une valeur `S > 1` indique que le produit désiré domine.
///
/// Panique si `desired_product_formed` est négatif ou non fini, ou si
/// `undesired_product_formed` n'est pas strictement positif ou n'est pas fini
/// (division).
pub fn yieldsel_selectivity(desired_product_formed: f64, undesired_product_formed: f64) -> f64 {
    assert!(
        desired_product_formed.is_finite() && desired_product_formed >= 0.0,
        "la quantité de produit désiré formé doit être finie et positive ou nulle (mol)"
    );
    assert!(
        undesired_product_formed.is_finite() && undesired_product_formed > 0.0,
        "la quantité de produit indésiré formé doit être finie et strictement positive (mol)"
    );
    desired_product_formed / undesired_product_formed
}

/// Rendement obtenu comme **produit** de la conversion et de la sélectivité
/// fractionnelle : `Y = X · S_frac` [sans dimension].
///
/// `conversion` `X` taux de conversion du réactif limitant [sans dimension,
/// `[0, 1]`], `instantaneous_selectivity` `S_frac` sélectivité **fractionnelle**
/// (fraction du réactif consommé orientée vers le produit désiré) [sans dimension,
/// `[0, 1]`]. Comme les deux facteurs sont dans `[0, 1]`, le rendement l'est aussi.
///
/// Panique si `conversion` ou `instantaneous_selectivity` n'est pas fini ou sort de
/// l'intervalle `[0, 1]`.
pub fn yieldsel_yield_from_conversion_selectivity(
    conversion: f64,
    instantaneous_selectivity: f64,
) -> f64 {
    assert!(
        conversion.is_finite() && (0.0..=1.0).contains(&conversion),
        "la conversion doit être finie et comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        instantaneous_selectivity.is_finite() && (0.0..=1.0).contains(&instantaneous_selectivity),
        "la sélectivité fractionnelle doit être finie et comprise dans [0, 1] (sans dimension)"
    );
    conversion * instantaneous_selectivity
}

/// Sélectivité **instantanée** comme rapport des vitesses locales :
/// `s = r_D / r_U` [sans dimension].
///
/// `desired_rate` `r_D` vitesse de formation du produit désiré [mol·m⁻³·s⁻¹],
/// `undesired_rate` `r_U` vitesse de formation du produit indésiré [même unité].
/// Ce rapport dépend des **concentrations locales**, donc du type de réacteur.
///
/// Panique si `desired_rate` est négatif ou non fini, ou si `undesired_rate` n'est
/// pas strictement positif ou n'est pas fini (division).
pub fn yieldsel_instantaneous_selectivity(desired_rate: f64, undesired_rate: f64) -> f64 {
    assert!(
        desired_rate.is_finite() && desired_rate >= 0.0,
        "la vitesse de formation du produit désiré doit être finie et positive ou nulle (mol·m⁻³·s⁻¹)"
    );
    assert!(
        undesired_rate.is_finite() && undesired_rate > 0.0,
        "la vitesse de formation du produit indésiré doit être finie et strictement positive (mol·m⁻³·s⁻¹)"
    );
    desired_rate / undesired_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn overall_yield_known_case_and_proportionality() {
        // 45 mol de produit désiré pour 60 mol de réactif consommé : Y = 45/60 = 0,75.
        assert_relative_eq!(yieldsel_overall_yield(45.0, 60.0), 0.75, epsilon = 1e-12);
        // Doubler le produit formé double le rendement (dénominateur fixe).
        assert_relative_eq!(yieldsel_overall_yield(90.0, 60.0), 1.5, epsilon = 1e-12);
        // Aucun produit formé → rendement nul.
        assert_relative_eq!(yieldsel_overall_yield(0.0, 60.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn selectivity_is_reciprocal_when_roles_swap() {
        // S = n_D/n_U = 45/15 = 3 ; en échangeant désiré/indésiré on obtient 1/S.
        let s = yieldsel_selectivity(45.0, 15.0);
        assert_relative_eq!(s, 3.0, epsilon = 1e-12);
        let s_swapped = yieldsel_selectivity(15.0, 45.0);
        assert_relative_eq!(s * s_swapped, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn yield_equals_conversion_times_selectivity_edges() {
        // À conversion totale (X = 1), le rendement égale la sélectivité fractionnelle.
        assert_relative_eq!(
            yieldsel_yield_from_conversion_selectivity(1.0, 0.75),
            0.75,
            epsilon = 1e-12
        );
        // À sélectivité fractionnelle unité (S_frac = 1), le rendement égale la conversion.
        assert_relative_eq!(
            yieldsel_yield_from_conversion_selectivity(0.6, 1.0),
            0.6,
            epsilon = 1e-12
        );
    }

    #[test]
    fn yield_on_feed_basis_matches_direct_ratio() {
        // Réactions parallèles A→D (désiré), A→U : 100 mol A chargées, X = 0,60 →
        // 60 mol consommées, dont 45 → D et 15 → U.
        // Sélectivité fractionnelle = Y_global = n_D/n_cons = 45/60 = 0,75.
        let s_frac = yieldsel_overall_yield(45.0, 60.0);
        // Rendement sur base charge = X · S_frac = 0,60 · 0,75 = 0,45.
        let y_feed = yieldsel_yield_from_conversion_selectivity(0.6, s_frac);
        assert_relative_eq!(y_feed, 0.45, epsilon = 1e-12);
        // Doit coïncider avec n_D / n_chargé = 45/100 = 0,45.
        assert_relative_eq!(y_feed, 45.0 / 100.0, epsilon = 1e-12);
    }

    #[test]
    fn instantaneous_selectivity_ratio_of_rates() {
        // s = r_D/r_U = 4,5/1,5 = 3.
        assert_relative_eq!(
            yieldsel_instantaneous_selectivity(4.5, 1.5),
            3.0,
            epsilon = 1e-12
        );
        // Vitesses égales → sélectivité unité.
        assert_relative_eq!(
            yieldsel_instantaneous_selectivity(2.0, 2.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn overall_yield_zero_consumed_panics() {
        // Y = n_D / n_cons avec n_cons = 0 : division rejetée.
        yieldsel_overall_yield(45.0, 0.0);
    }
}
