//! Vaporisation partielle par détente à l'équilibre — **flash isotherme** d'une
//! alimentation multiconstituant en une phase vapeur et une phase liquide en
//! équilibre thermodynamique.
//!
//! ```text
//! équilibre L-V     yᵢ = Kᵢ·xᵢ                                    [sans dimension]
//! terme R-R         gᵢ = zᵢ·(Kᵢ − 1) / (1 + V·(Kᵢ − 1))          [sans dimension]
//! équation R-R      Σᵢ gᵢ = 0            (résolue en V)           [sans dimension]
//! partage des débits (V̇, L̇) = (F·V, F·(1 − V))                  [mol/s ou kg/s]
//! composition liq.  xᵢ = zᵢ / (1 + V·(Kᵢ − 1))                   [sans dimension]
//! ```
//!
//! `zᵢ` fraction molaire du constituant dans l'alimentation [sans dimension],
//! `xᵢ` fraction molaire dans le liquide sortant [sans dimension], `yᵢ` fraction
//! molaire dans la vapeur sortante [sans dimension], `Kᵢ = yᵢ/xᵢ` coefficient de
//! partage (constante d'équilibre) du constituant [sans dimension], `V` fraction
//! molaire vaporisée `V̇/F` [sans dimension, dans `[0, 1]`], `gᵢ` terme d'un
//! constituant dans l'équation de Rachford-Rice [sans dimension], `F` débit
//! d'alimentation [mol/s ou kg/s], `V̇` débit de vapeur [mol/s ou kg/s], `L̇`
//! débit de liquide [mol/s ou kg/s].
//!
//! **Limite honnête** : il s'agit d'un flash **à l'équilibre** (une seule détente,
//! un étage théorique), pas d'une colonne. Les **coefficients de partage** `Kᵢ`
//! sont **fournis par l'appelant** : ils dépendent de la température, de la
//! pression et de la composition et doivent provenir d'un modèle
//! thermodynamique (loi de Raoult `Kᵢ = P_sat,ᵢ/P` en mélange idéal, corrélation,
//! équation d'état…) ou de données ; aucune valeur « par défaut » n'est inventée
//! ici. La **fraction vaporisée** `V` est celle qui **annule** la somme des termes
//! de Rachford-Rice (`Σᵢ gᵢ = 0`) : la **résolution numérique** de cette équation
//! scalaire en `V` (bissection, Newton…) est **à la charge de l'appelant**. Ces
//! fonctions fournissent les **relations élémentaires** évaluées pour un `V` donné,
//! elles ne résolvent ni la boucle d'équilibre ni le bilan enthalpique du flash.

/// Fraction molaire d'un constituant dans la vapeur à l'équilibre `yᵢ = Kᵢ·xᵢ`
/// [sans dimension].
///
/// `liquid_fraction` `xᵢ` fraction molaire dans le liquide [sans dimension, dans
/// `[0, 1]`], `equilibrium_ratio` `Kᵢ` coefficient de partage **fourni**
/// [sans dimension, strictement positif].
///
/// Panique si `liquid_fraction` sort de `[0, 1]` ou si `equilibrium_ratio` n'est
/// pas strictement positif (ou est non fini).
pub fn flash_equilibrium_vapor(liquid_fraction: f64, equilibrium_ratio: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&liquid_fraction),
        "la fraction molaire liquide doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        equilibrium_ratio.is_finite() && equilibrium_ratio > 0.0,
        "le coefficient de partage K doit être fini et strictement positif (sans dimension)"
    );
    equilibrium_ratio * liquid_fraction
}

/// Terme d'un constituant dans l'équation de Rachford-Rice
/// `gᵢ = zᵢ·(Kᵢ − 1) / (1 + V·(Kᵢ − 1))` [sans dimension].
///
/// La **somme** de ces termes sur tous les constituants s'annule à la solution du
/// flash (`Σᵢ gᵢ = 0`), ce qui détermine la fraction vaporisée `V`.
///
/// `feed_fraction` `zᵢ` fraction molaire dans l'alimentation [sans dimension, dans
/// `[0, 1]`], `equilibrium_ratio` `Kᵢ` coefficient de partage **fourni**
/// [sans dimension, strictement positif], `vapor_fraction` `V` fraction molaire
/// vaporisée [sans dimension, dans `[0, 1]`]. Le dénominateur `1 + V·(Kᵢ − 1)`
/// reste strictement positif pour `V ∈ [0, 1]` et `Kᵢ > 0`.
///
/// Panique si `feed_fraction` sort de `[0, 1]`, si `equilibrium_ratio` n'est pas
/// fini et strictement positif, ou si `vapor_fraction` sort de `[0, 1]`.
pub fn flash_rachford_rice_term(
    feed_fraction: f64,
    equilibrium_ratio: f64,
    vapor_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&feed_fraction),
        "la fraction molaire d'alimentation doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        equilibrium_ratio.is_finite() && equilibrium_ratio > 0.0,
        "le coefficient de partage K doit être fini et strictement positif (sans dimension)"
    );
    assert!(
        (0.0..=1.0).contains(&vapor_fraction),
        "la fraction vaporisée V doit être comprise dans [0, 1] (sans dimension)"
    );
    let km1 = equilibrium_ratio - 1.0;
    feed_fraction * km1 / (1.0 + vapor_fraction * km1)
}

/// Partage des débits entre phases `(V̇, L̇) = (F·V, F·(1 − V))` [mol/s ou kg/s].
///
/// Renvoie le couple `(débit vapeur, débit liquide)`, dont la somme reproduit le
/// débit d'alimentation `F`.
///
/// `feed_flow` `F` débit d'alimentation [mol/s ou kg/s, positif ou nul],
/// `vapor_fraction` `V` fraction molaire vaporisée [sans dimension, dans `[0, 1]`].
///
/// Panique si `feed_flow` est négatif ou non fini, ou si `vapor_fraction` sort de
/// `[0, 1]`.
pub fn flash_vapor_liquid_split(feed_flow: f64, vapor_fraction: f64) -> (f64, f64) {
    assert!(
        feed_flow.is_finite() && feed_flow >= 0.0,
        "le débit d'alimentation doit être fini et positif ou nul (mol/s ou kg/s)"
    );
    assert!(
        (0.0..=1.0).contains(&vapor_fraction),
        "la fraction vaporisée V doit être comprise dans [0, 1] (sans dimension)"
    );
    (
        feed_flow * vapor_fraction,
        feed_flow * (1.0 - vapor_fraction),
    )
}

/// Fraction molaire d'un constituant dans le liquide sortant
/// `xᵢ = zᵢ / (1 + V·(Kᵢ − 1))` [sans dimension].
///
/// `feed_fraction` `zᵢ` fraction molaire dans l'alimentation [sans dimension, dans
/// `[0, 1]`], `equilibrium_ratio` `Kᵢ` coefficient de partage **fourni**
/// [sans dimension, strictement positif], `vapor_fraction` `V` fraction molaire
/// vaporisée [sans dimension, dans `[0, 1]`]. La fraction vapeur correspondante
/// s'obtient par `yᵢ = Kᵢ·xᵢ` ([`flash_equilibrium_vapor`]).
///
/// Panique si `feed_fraction` sort de `[0, 1]`, si `equilibrium_ratio` n'est pas
/// fini et strictement positif, ou si `vapor_fraction` sort de `[0, 1]`.
pub fn flash_liquid_composition(
    feed_fraction: f64,
    equilibrium_ratio: f64,
    vapor_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&feed_fraction),
        "la fraction molaire d'alimentation doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        equilibrium_ratio.is_finite() && equilibrium_ratio > 0.0,
        "le coefficient de partage K doit être fini et strictement positif (sans dimension)"
    );
    assert!(
        (0.0..=1.0).contains(&vapor_fraction),
        "la fraction vaporisée V doit être comprise dans [0, 1] (sans dimension)"
    );
    feed_fraction / (1.0 + vapor_fraction * (equilibrium_ratio - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas binaire de référence, résolu à la main.
    // Alimentation z1 = z2 = 0,5 ; K1 = 2,0 ; K2 = 0,5.
    // Rachford-Rice : 0,5·(1)/(1+V) + 0,5·(−0,5)/(1−0,5V) = 0 ⇒ V = 0,5.
    // Vérif : 0,5/1,5 − 0,25/0,75 = 1/3 − 1/3 = 0.
    const Z1: f64 = 0.5;
    const Z2: f64 = 0.5;
    const K1: f64 = 2.0;
    const K2: f64 = 0.5;
    const V_SOL: f64 = 0.5;

    #[test]
    fn rachford_rice_sum_vanishes_at_solution() {
        // La somme des termes s'annule pour la fraction vaporisée solution.
        let g1 = flash_rachford_rice_term(Z1, K1, V_SOL);
        let g2 = flash_rachford_rice_term(Z2, K2, V_SOL);
        // g1 = 0,5·1/1,5 = 1/3 ; g2 = 0,5·(−0,5)/0,75 = −1/3.
        assert_relative_eq!(g1, 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(g1 + g2, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn compositions_normalize_and_close_overall_balance() {
        // x1 = 0,5/(1+0,5) = 1/3 ; x2 = 0,5/(1−0,25) = 2/3 ⇒ Σx = 1.
        let x1 = flash_liquid_composition(Z1, K1, V_SOL);
        let x2 = flash_liquid_composition(Z2, K2, V_SOL);
        assert_relative_eq!(x1, 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(x1 + x2, 1.0, epsilon = 1e-12);

        // y1 = K1·x1 = 2/3 ; y2 = K2·x2 = 1/3 ⇒ Σy = 1.
        let y1 = flash_equilibrium_vapor(x1, K1);
        let y2 = flash_equilibrium_vapor(x2, K2);
        assert_relative_eq!(y1, 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(y1 + y2, 1.0, epsilon = 1e-12);

        // Bilan par constituant : V·yᵢ + (1−V)·xᵢ = zᵢ.
        assert_relative_eq!(V_SOL * y1 + (1.0 - V_SOL) * x1, Z1, epsilon = 1e-12);
        assert_relative_eq!(V_SOL * y2 + (1.0 - V_SOL) * x2, Z2, epsilon = 1e-12);
    }

    #[test]
    fn rachford_rice_term_equals_vapor_minus_liquid() {
        // Identité : gᵢ = zᵢ(Kᵢ−1)/(…) = (Kᵢ−1)·xᵢ = yᵢ − xᵢ.
        let x1 = flash_liquid_composition(Z1, K1, V_SOL);
        let y1 = flash_equilibrium_vapor(x1, K1);
        let g1 = flash_rachford_rice_term(Z1, K1, V_SOL);
        assert_relative_eq!(g1, y1 - x1, epsilon = 1e-12);
    }

    #[test]
    fn split_conserves_feed_flow() {
        // F = 100 mol/s, V = 0,5 ⇒ vapeur 50, liquide 50, somme = 100.
        let feed = 100.0;
        let (vap, liq) = flash_vapor_liquid_split(feed, V_SOL);
        assert_relative_eq!(vap, 50.0, epsilon = 1e-12);
        assert_relative_eq!(liq, 50.0, epsilon = 1e-12);
        assert_relative_eq!(vap + liq, feed, epsilon = 1e-12);
    }

    #[test]
    fn zero_vapor_fraction_is_pure_liquid_limit() {
        // À V = 0 (point de bulle), tout part en liquide : xᵢ = zᵢ, débit vapeur nul.
        let x1 = flash_liquid_composition(Z1, K1, 0.0);
        assert_relative_eq!(x1, Z1, epsilon = 1e-12);
        let (vap, liq) = flash_vapor_liquid_split(80.0, 0.0);
        assert_relative_eq!(vap, 0.0, epsilon = 1e-12);
        assert_relative_eq!(liq, 80.0, epsilon = 1e-12);
        // Le terme R-R vaut alors simplement zᵢ(Kᵢ−1) : 0,5·1 = 0,5.
        assert_relative_eq!(flash_rachford_rice_term(Z1, K1, 0.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn equilibrium_vapor_is_linear_in_liquid_fraction() {
        // y = K·x : doubler x double y (proportionnalité), K fixé.
        let k = 1.8;
        let y_small = flash_equilibrium_vapor(0.2, k);
        let y_double = flash_equilibrium_vapor(0.4, k);
        assert_relative_eq!(y_double, 2.0 * y_small, epsilon = 1e-12);
        assert_relative_eq!(y_small, 0.36, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positif")]
    fn nonpositive_equilibrium_ratio_panics() {
        // Un coefficient de partage nul n'a pas de sens physique : rejet.
        flash_equilibrium_vapor(0.3, 0.0);
    }
}
