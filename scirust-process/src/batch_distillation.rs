//! Distillation discontinue **simple** (équation de Rayleigh) d'un mélange
//! binaire à volatilité relative **constante**, sans reflux.
//!
//! ```text
//! Rayleigh (intégrale)   ln(L₂/L₁) = ∫_{x₁}^{x₂} dx / (y*(x) − x)        [—]
//! équilibre (α const.)   y*(x) = α·x / (1 + (α−1)·x)                     [—]
//! forme intégrée         ln(L₂/L₁) = 1/(α−1)·[ln(x₂/x₁) − α·ln((1−x₂)/(1−x₁))]
//! moles restantes        L₂ = L₁·(x₂/x₁)^{1/(α−1)}·((1−x₁)/(1−x₂))^{α/(α−1)}  [mol]
//! moles distillées       D = L₁ − L₂                                     [mol]
//! composition moyenne    x_D = (L₁·x₁ − L₂·x₂) / (L₁ − L₂)               [—]
//! taux de récupération    R = D·x_D / (L₁·x₁)                             [—]
//! ```
//!
//! `L₁` charge initiale du bouilleur [mol], `L₂` charge restante en fin
//! d'opération [mol], `x₁` fraction molaire initiale du constituant léger dans le
//! liquide [sans dimension, dans `]0, 1[`], `x₂` fraction molaire finale dans le
//! liquide [sans dimension, dans `]0, 1[`], `y*(x)` fraction molaire du léger dans
//! la vapeur à l'équilibre [sans dimension], `α` volatilité relative léger/lourd
//! [sans dimension, `> 0` et `≠ 1`], `D` quantité totale distillée [mol], `x_D`
//! composition molaire moyenne du distillat [sans dimension], `R` taux de
//! récupération d'un constituant [sans dimension].
//!
//! **Limite honnête** : il s'agit d'une distillation **simple** (différentielle,
//! de Rayleigh) d'un **binaire** : à chaque instant une **seule vaporisation
//! d'équilibre** enlève la vapeur formée (**pas de reflux**, pas de colonne, pas
//! d'étages). La **volatilité relative** `α` et les **compositions** `x₁`, `x₂`
//! sont **fournies par l'appelant** : la forme intégrée ci-dessus n'est exacte que
//! si `α` est **réellement constant** le long de la trajectoire. La composition
//! instantanée de la vapeur suit l'équilibre `y*(x) = α·x/(1+(α−1)·x)` ;
//! l'**intégration exacte** de Rayleigh requiert la **courbe d'équilibre** réelle
//! `y*(x)`, qui doit provenir de données ou d'un modèle thermodynamique. Ce module
//! **n'invente aucune volatilité ni aucune enthalpie de vaporisation** ; il ne
//! résout pas non plus le bilan enthalpique du bouilleur ni la dynamique
//! temporelle de l'opération.

/// Logarithme du rapport des charges `ln(L₂/L₁)` par l'intégrale de Rayleigh à
/// volatilité relative constante
/// `ln(L₂/L₁) = 1/(α−1)·[ln(x₂/x₁) − α·ln((1−x₂)/(1−x₁))]` [sans dimension].
///
/// `initial_fraction` `x₁` et `final_fraction` `x₂` fractions molaires du léger
/// [sans dimension, dans `]0, 1[`], `relative_volatility` `α` volatilité relative
/// **fournie** [sans dimension, strictement positive et différente de 1].
///
/// Panique si `initial_fraction` ou `final_fraction` sort de `]0, 1[`, ou si
/// `relative_volatility` n'est pas fini, n'est pas strictement positif, ou vaut 1.
pub fn bdist_rayleigh_constant_volatility(
    initial_fraction: f64,
    final_fraction: f64,
    relative_volatility: f64,
) -> f64 {
    assert!(
        initial_fraction.is_finite() && initial_fraction > 0.0 && initial_fraction < 1.0,
        "la fraction molaire initiale x₁ doit être dans ]0, 1[ (sans dimension)"
    );
    assert!(
        final_fraction.is_finite() && final_fraction > 0.0 && final_fraction < 1.0,
        "la fraction molaire finale x₂ doit être dans ]0, 1[ (sans dimension)"
    );
    assert!(
        relative_volatility.is_finite() && relative_volatility > 0.0,
        "la volatilité relative α doit être finie et strictement positive (sans dimension)"
    );
    assert!(
        (relative_volatility - 1.0).abs() > 0.0,
        "la volatilité relative α doit être différente de 1 (division par α−1)"
    );
    let ratio_light = (final_fraction / initial_fraction).ln();
    let ratio_heavy = ((1.0 - final_fraction) / (1.0 - initial_fraction)).ln();
    (ratio_light - relative_volatility * ratio_heavy) / (relative_volatility - 1.0)
}

/// Charge restante au bouilleur en fin d'opération
/// `L₂ = L₁·(x₂/x₁)^{1/(α−1)}·((1−x₁)/(1−x₂))^{α/(α−1)}` [mol].
///
/// Forme exponentielle cohérente de l'intégrale de Rayleigh :
/// `L₂ = L₁·exp(ln(L₂/L₁))` avec le logarithme donné par
/// [`bdist_rayleigh_constant_volatility`].
///
/// `initial_moles` `L₁` charge initiale [mol, strictement positive],
/// `initial_fraction` `x₁` et `final_fraction` `x₂` fractions molaires du léger
/// [sans dimension, dans `]0, 1[`], `relative_volatility` `α` volatilité relative
/// **fournie** [sans dimension, strictement positive et différente de 1].
///
/// Panique si `initial_moles` n'est pas fini ou n'est pas strictement positif, si
/// une fraction sort de `]0, 1[`, ou si `relative_volatility` n'est pas fini, pas
/// strictement positif, ou vaut 1.
pub fn bdist_remaining_moles(
    initial_moles: f64,
    initial_fraction: f64,
    final_fraction: f64,
    relative_volatility: f64,
) -> f64 {
    assert!(
        initial_moles.is_finite() && initial_moles > 0.0,
        "la charge initiale L₁ doit être finie et strictement positive (mol)"
    );
    assert!(
        initial_fraction.is_finite() && initial_fraction > 0.0 && initial_fraction < 1.0,
        "la fraction molaire initiale x₁ doit être dans ]0, 1[ (sans dimension)"
    );
    assert!(
        final_fraction.is_finite() && final_fraction > 0.0 && final_fraction < 1.0,
        "la fraction molaire finale x₂ doit être dans ]0, 1[ (sans dimension)"
    );
    assert!(
        relative_volatility.is_finite() && relative_volatility > 0.0,
        "la volatilité relative α doit être finie et strictement positive (sans dimension)"
    );
    assert!(
        (relative_volatility - 1.0).abs() > 0.0,
        "la volatilité relative α doit être différente de 1 (division par α−1)"
    );
    let exponent_light = 1.0 / (relative_volatility - 1.0);
    let exponent_heavy = relative_volatility / (relative_volatility - 1.0);
    initial_moles
        * (final_fraction / initial_fraction).powf(exponent_light)
        * ((1.0 - initial_fraction) / (1.0 - final_fraction)).powf(exponent_heavy)
}

/// Quantité totale distillée par bilan matière global `D = L₁ − L₂` [mol].
///
/// `initial_moles` `L₁` charge initiale [mol, strictement positive],
/// `remaining_moles` `L₂` charge restante [mol, positive et au plus égale à `L₁`].
///
/// Panique si l'une des charges n'est pas finie ou est négative, ou si
/// `remaining_moles` dépasse `initial_moles` (distillat négatif impossible).
pub fn bdist_distillate_moles(initial_moles: f64, remaining_moles: f64) -> f64 {
    assert!(
        initial_moles.is_finite() && initial_moles > 0.0,
        "la charge initiale L₁ doit être finie et strictement positive (mol)"
    );
    assert!(
        remaining_moles.is_finite() && remaining_moles >= 0.0,
        "la charge restante L₂ doit être finie et positive ou nulle (mol)"
    );
    assert!(
        remaining_moles <= initial_moles,
        "la charge restante L₂ ne peut pas dépasser la charge initiale L₁ (mol)"
    );
    initial_moles - remaining_moles
}

/// Composition molaire moyenne du distillat par bilan matière
/// `x_D = (L₁·x₁ − L₂·x₂) / (L₁ − L₂)` [sans dimension].
///
/// `initial_moles` `L₁` charge initiale [mol], `initial_fraction` `x₁` fraction
/// molaire initiale du léger [sans dimension, dans `[0, 1]`], `remaining_moles`
/// `L₂` charge restante [mol], `remaining_fraction` `x₂` fraction molaire finale du
/// léger [sans dimension, dans `[0, 1]`]. Le dénominateur `L₁ − L₂` est le distillat
/// total `D`, qui doit être strictement positif.
///
/// Panique si une charge n'est pas finie ou est négative, si une fraction sort de
/// `[0, 1]`, ou si `remaining_moles ≥ initial_moles` (aucun distillat produit).
pub fn bdist_average_distillate_fraction(
    initial_moles: f64,
    initial_fraction: f64,
    remaining_moles: f64,
    remaining_fraction: f64,
) -> f64 {
    assert!(
        initial_moles.is_finite() && initial_moles > 0.0,
        "la charge initiale L₁ doit être finie et strictement positive (mol)"
    );
    assert!(
        remaining_moles.is_finite() && remaining_moles >= 0.0,
        "la charge restante L₂ doit être finie et positive ou nulle (mol)"
    );
    assert!(
        (0.0..=1.0).contains(&initial_fraction),
        "la fraction molaire initiale x₁ doit être dans [0, 1] (sans dimension)"
    );
    assert!(
        (0.0..=1.0).contains(&remaining_fraction),
        "la fraction molaire finale x₂ doit être dans [0, 1] (sans dimension)"
    );
    assert!(
        remaining_moles < initial_moles,
        "un distillat doit être produit : L₂ doit être strictement inférieur à L₁ (mol)"
    );
    (initial_moles * initial_fraction - remaining_moles * remaining_fraction)
        / (initial_moles - remaining_moles)
}

/// Taux de récupération d'un constituant dans le distillat
/// `R = n_{D} / n_{0}` [sans dimension].
///
/// `distillate_moles_component` `n_D` moles du constituant passées au distillat
/// [mol, positives], `initial_moles_component` `n₀` moles initiales de ce
/// constituant dans la charge [mol, strictement positives]. Pour le constituant
/// léger, `n_D = L₁·x₁ − L₂·x₂` et `n₀ = L₁·x₁`.
///
/// Panique si `initial_moles_component` n'est pas fini ou n'est pas strictement
/// positif, ou si `distillate_moles_component` n'est pas fini ou est négatif.
pub fn bdist_recovery(distillate_moles_component: f64, initial_moles_component: f64) -> f64 {
    assert!(
        distillate_moles_component.is_finite() && distillate_moles_component >= 0.0,
        "les moles distillées du constituant doivent être finies et positives ou nulles (mol)"
    );
    assert!(
        initial_moles_component.is_finite() && initial_moles_component > 0.0,
        "les moles initiales du constituant doivent être finies et strictement positives (mol)"
    );
    distillate_moles_component / initial_moles_component
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas chiffré de référence, calculé à la main puis recalculé.
    // L₁ = 100 mol, x₁ = 0,5, x₂ = 0,2, α = 2.
    // L₂ = 100·(0,2/0,5)^{1/1}·((1−0,5)/(1−0,2))^{2/1}
    //    = 100·0,4·(0,5/0,8)² = 100·0,4·0,625² = 100·0,4·0,390625 = 15,625 mol.
    // D  = 100 − 15,625 = 84,375 mol.
    // x_D = (100·0,5 − 15,625·0,2)/84,375 = (50 − 3,125)/84,375 = 46,875/84,375 = 0,555…
    // R_léger = (50 − 3,125)/50 = 46,875/50 = 0,9375.
    const L1: f64 = 100.0;
    const X1: f64 = 0.5;
    const X2: f64 = 0.2;
    const ALPHA: f64 = 2.0;
    const L2_REF: f64 = 15.625;

    #[test]
    fn remaining_moles_matches_reference_case() {
        // Cas chiffré vérifié à la main : L₂ = 15,625 mol.
        let l2 = bdist_remaining_moles(L1, X1, X2, ALPHA);
        assert_relative_eq!(l2, L2_REF, epsilon = 1e-9);
    }

    #[test]
    fn log_and_exponential_forms_are_consistent() {
        // Identité : L₂ = L₁·exp(ln(L₂/L₁)).
        let ln_ratio = bdist_rayleigh_constant_volatility(X1, X2, ALPHA);
        let l2_from_log = L1 * ln_ratio.exp();
        let l2_direct = bdist_remaining_moles(L1, X1, X2, ALPHA);
        assert_relative_eq!(l2_from_log, l2_direct, epsilon = 1e-9);
    }

    #[test]
    fn distillate_closes_global_balance() {
        // Bilan global : L₂ + D = L₁.
        let l2 = bdist_remaining_moles(L1, X1, X2, ALPHA);
        let d = bdist_distillate_moles(L1, l2);
        assert_relative_eq!(d, 84.375, epsilon = 1e-9);
        assert_relative_eq!(l2 + d, L1, epsilon = 1e-9);
    }

    #[test]
    fn component_balance_closes_with_average_distillate() {
        // Bilan par constituant : L₁·x₁ = L₂·x₂ + D·x_D.
        let l2 = bdist_remaining_moles(L1, X1, X2, ALPHA);
        let d = bdist_distillate_moles(L1, l2);
        let x_d = bdist_average_distillate_fraction(L1, X1, l2, X2);
        assert_relative_eq!(x_d, 46.875 / 84.375, epsilon = 1e-9);
        assert_relative_eq!(L2_REF * X2 + d * x_d, L1 * X1, epsilon = 1e-9);
    }

    #[test]
    fn recovery_matches_component_balance() {
        // R_léger = D·x_D / (L₁·x₁) = (L₁·x₁ − L₂·x₂)/(L₁·x₁) = 0,9375.
        let l2 = bdist_remaining_moles(L1, X1, X2, ALPHA);
        let n0_light = L1 * X1;
        let nd_light = n0_light - l2 * X2;
        let r = bdist_recovery(nd_light, n0_light);
        assert_relative_eq!(r, 0.9375, epsilon = 1e-9);
    }

    #[test]
    fn no_change_gives_no_distillation() {
        // Cas limite x₂ = x₁ : rien n'a été distillé, L₂ = L₁ (ln(L₂/L₁) = 0).
        let ln_ratio = bdist_rayleigh_constant_volatility(X1, X1, ALPHA);
        assert_relative_eq!(ln_ratio, 0.0, epsilon = 1e-12);
        let l2 = bdist_remaining_moles(L1, X1, X1, ALPHA);
        assert_relative_eq!(l2, L1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "différente de 1")]
    fn unit_relative_volatility_panics() {
        // α = 1 annule le dénominateur (α−1) : rejet.
        bdist_rayleigh_constant_volatility(X1, X2, 1.0);
    }
}
