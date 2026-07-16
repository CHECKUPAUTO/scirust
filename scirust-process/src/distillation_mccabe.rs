//! Distillation binaire — méthode **McCabe-Thiele / Fenske** : volatilité
//! relative, relation d'équilibre y-x, étages minimaux au reflux total, pente
//! opératoire de rectification et reflux minimal (point de pincement).
//!
//! ```text
//! volatilité relative   α  = K_L / K_H                                       [-]
//! équilibre y-x         y  = α·x / (1 + (α − 1)·x)                           [-]
//! Fenske (reflux total) Nₘ = ln[(d_L/d_H)·(b_H/b_L)] / ln(α)                [étages]
//! pente rectification   s  = R / (R + 1)                                     [-]
//! reflux minimal        Rₘ = (x_D − y*) / (y* − z_F)                         [-]
//! ```
//!
//! `α` volatilité relative du constituant léger sur le lourd [sans dimension],
//! `K_L`, `K_H` constantes d'équilibre (coefficients de partage vapeur/liquide)
//! des constituants léger et lourd [sans dimension], `x` fraction molaire du léger
//! en phase liquide [sans dimension, 0 ≤ x ≤ 1], `y` fraction molaire du léger en
//! phase vapeur à l'équilibre [sans dimension], `Nₘ` nombre minimal d'étages
//! théoriques au reflux total [étages], `d_L`, `d_H` débits molaires du léger et
//! du lourd dans le distillat [mol·s⁻¹], `b_L`, `b_H` débits molaires du léger et
//! du lourd dans le résidu de pied [mol·s⁻¹], `s` pente de la droite opératoire de
//! rectification [sans dimension], `R` taux de reflux L/D [sans dimension],
//! `Rₘ` taux de reflux minimal [sans dimension], `x_D` fraction molaire du léger
//! au distillat [sans dimension], `y*` fraction vapeur d'équilibre à la composition
//! d'alimentation [sans dimension], `z_F` fraction molaire du léger à
//! l'alimentation [sans dimension].
//!
//! **Limite honnête** : ces relations valent pour une **distillation binaire** à
//! **volatilité relative constante FOURNIE par l'appelant** (jamais inventée ;
//! issue de mesures, de tables ou du rapport de constantes d'équilibre), avec
//! l'hypothèse de **débit molaire constant** (débordement équimolaire, McCabe-
//! Thiele) et des **étages théoriques**. La méthode de **Fenske** donne les étages
//! au **reflux total** ; le **rendement d'étage réel** (efficacité de Murphree ou
//! globale) est **fourni par l'appelant** pour convertir en étages réels. Le
//! reflux minimal ci-dessous suppose une **alimentation liquide bouillante** (droite
//! q verticale), la composition vapeur d'équilibre `y*` à l'alimentation étant
//! **fournie**. Les enthalpies, volatilités et coefficients de partage ne sont
//! **pas** calculés ici.

/// Volatilité relative `α = K_L / K_H` à partir des constantes d'équilibre
/// (coefficients de partage vapeur/liquide) [sans dimension].
///
/// `k_light` `K_L` constante d'équilibre du constituant léger [sans dimension],
/// `k_heavy` `K_H` constante d'équilibre du constituant lourd [sans dimension].
///
/// Panique si `k_light` ou `k_heavy` n'est pas fini ou n'est pas strictement
/// positif.
pub fn dist_relative_volatility(k_light: f64, k_heavy: f64) -> f64 {
    assert!(
        k_light.is_finite() && k_light > 0.0,
        "la constante d'équilibre du léger doit être finie et strictement positive"
    );
    assert!(
        k_heavy.is_finite() && k_heavy > 0.0,
        "la constante d'équilibre du lourd doit être finie et strictement positive"
    );
    k_light / k_heavy
}

/// Fraction molaire vapeur à l'équilibre pour un binaire à volatilité relative
/// constante : `y = α·x / (1 + (α − 1)·x)` [sans dimension].
///
/// `liquid_fraction` `x` fraction molaire du léger en phase liquide
/// [sans dimension, 0 ≤ x ≤ 1], `relative_volatility` `α` volatilité relative
/// du léger sur le lourd [sans dimension].
///
/// Panique si `liquid_fraction` n'est pas dans `[0, 1]`, ou si
/// `relative_volatility` n'est pas fini ou n'est pas strictement positif.
pub fn dist_equilibrium_vapor(liquid_fraction: f64, relative_volatility: f64) -> f64 {
    assert!(
        liquid_fraction.is_finite() && (0.0..=1.0).contains(&liquid_fraction),
        "la fraction molaire liquide doit être dans l'intervalle [0, 1]"
    );
    assert!(
        relative_volatility.is_finite() && relative_volatility > 0.0,
        "la volatilité relative doit être finie et strictement positive"
    );
    relative_volatility * liquid_fraction / (1.0 + (relative_volatility - 1.0) * liquid_fraction)
}

/// Nombre minimal d'étages théoriques au **reflux total** (équation de Fenske) :
/// `Nₘ = ln[(d_L/d_H)·(b_H/b_L)] / ln(α)` [étages].
///
/// `distillate_light` `d_L` et `distillate_heavy` `d_H` débits molaires des
/// constituants léger et lourd dans le distillat [mol·s⁻¹], `bottoms_light` `b_L`
/// et `bottoms_heavy` `b_H` débits molaires des constituants léger et lourd dans
/// le résidu de pied [mol·s⁻¹], `relative_volatility` `α` volatilité relative
/// [sans dimension].
///
/// Panique si un des quatre débits n'est pas fini ou n'est pas strictement
/// positif, ou si `relative_volatility` n'est pas fini, pas strictement positif,
/// ou égal à 1 (le logarithme au dénominateur s'annulerait).
pub fn dist_fenske_minimum_stages(
    distillate_light: f64,
    distillate_heavy: f64,
    bottoms_light: f64,
    bottoms_heavy: f64,
    relative_volatility: f64,
) -> f64 {
    assert!(
        distillate_light.is_finite() && distillate_light > 0.0,
        "le débit de léger au distillat doit être fini et strictement positif (mol·s⁻¹)"
    );
    assert!(
        distillate_heavy.is_finite() && distillate_heavy > 0.0,
        "le débit de lourd au distillat doit être fini et strictement positif (mol·s⁻¹)"
    );
    assert!(
        bottoms_light.is_finite() && bottoms_light > 0.0,
        "le débit de léger au résidu doit être fini et strictement positif (mol·s⁻¹)"
    );
    assert!(
        bottoms_heavy.is_finite() && bottoms_heavy > 0.0,
        "le débit de lourd au résidu doit être fini et strictement positif (mol·s⁻¹)"
    );
    assert!(
        relative_volatility.is_finite() && relative_volatility > 0.0 && relative_volatility != 1.0,
        "la volatilité relative doit être finie, strictement positive et différente de 1"
    );
    ((distillate_light / distillate_heavy) * (bottoms_heavy / bottoms_light)).ln()
        / relative_volatility.ln()
}

/// Pente de la droite opératoire de **rectification** : `s = R / (R + 1)`
/// [sans dimension], où `R` est le taux de reflux `L/D`.
///
/// `reflux_ratio` `R` taux de reflux (rapport du reflux liquide au débit de
/// distillat) [sans dimension].
///
/// Panique si `reflux_ratio` n'est pas fini ou est négatif.
pub fn dist_rectifying_operating_slope(reflux_ratio: f64) -> f64 {
    assert!(
        reflux_ratio.is_finite() && reflux_ratio >= 0.0,
        "le taux de reflux doit être fini et positif ou nul"
    );
    reflux_ratio / (reflux_ratio + 1.0)
}

/// Taux de reflux **minimal** par la méthode du point de pincement, alimentation
/// liquide bouillante : `Rₘ = (x_D − y*) / (y* − z_F)` [sans dimension].
///
/// `distillate_fraction` `x_D` fraction molaire du léger au distillat
/// [sans dimension, 0 ≤ x_D ≤ 1], `feed_vapor_fraction_equilibrium` `y*` fraction
/// molaire vapeur d'équilibre à la composition d'alimentation [sans dimension,
/// 0 ≤ y* ≤ 1], `feed_fraction` `z_F` fraction molaire du léger à l'alimentation
/// [sans dimension, 0 ≤ z_F ≤ 1].
///
/// Panique si l'une des trois fractions n'est pas dans `[0, 1]`, ou si
/// `feed_vapor_fraction_equilibrium` n'est pas strictement supérieure à
/// `feed_fraction` (dénominateur nul ou négatif : pincement non physique).
pub fn dist_minimum_reflux_ratio(
    distillate_fraction: f64,
    feed_vapor_fraction_equilibrium: f64,
    feed_fraction: f64,
) -> f64 {
    assert!(
        distillate_fraction.is_finite() && (0.0..=1.0).contains(&distillate_fraction),
        "la fraction molaire au distillat doit être dans l'intervalle [0, 1]"
    );
    assert!(
        feed_vapor_fraction_equilibrium.is_finite()
            && (0.0..=1.0).contains(&feed_vapor_fraction_equilibrium),
        "la fraction vapeur d'équilibre à l'alimentation doit être dans l'intervalle [0, 1]"
    );
    assert!(
        feed_fraction.is_finite() && (0.0..=1.0).contains(&feed_fraction),
        "la fraction molaire à l'alimentation doit être dans l'intervalle [0, 1]"
    );
    assert!(
        feed_vapor_fraction_equilibrium > feed_fraction,
        "la fraction vapeur d'équilibre doit être strictement supérieure à la fraction d'alimentation"
    );
    (distillate_fraction - feed_vapor_fraction_equilibrium)
        / (feed_vapor_fraction_equilibrium - feed_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn relative_volatility_is_ratio_of_partition_constants() {
        // α = K_L/K_H ; avec K_L = 3, K_H = 1,2 → α = 2,5.
        let alpha = dist_relative_volatility(3.0, 1.2);
        assert_relative_eq!(alpha, 2.5, epsilon = 1e-12);
        // Constituants d'égale volatilité (K_L = K_H) → α = 1 (séparation impossible).
        assert_relative_eq!(dist_relative_volatility(1.4, 1.4), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn equilibrium_endpoints_and_realistic_case() {
        // Les points fixes de la courbe d'équilibre : y(0) = 0 et y(1) = 1.
        assert_relative_eq!(dist_equilibrium_vapor(0.0, 2.5), 0.0, epsilon = 1e-12);
        assert_relative_eq!(dist_equilibrium_vapor(1.0, 2.5), 1.0, epsilon = 1e-12);
        // Cas chiffré : α = 2,5, x = 0,5.
        // y = 2,5·0,5 / (1 + 1,5·0,5) = 1,25 / 1,75 = 5/7 ≈ 0,714285714.
        let y = dist_equilibrium_vapor(0.5, 2.5);
        assert_relative_eq!(y, 5.0 / 7.0, epsilon = 1e-9);
    }

    #[test]
    fn equilibrium_satisfies_alpha_odds_ratio_identity() {
        // Identité exacte de la relation à α constant : y/(1−y) = α·x/(1−x).
        let alpha = 3.2_f64;
        let x = 0.35_f64;
        let y = dist_equilibrium_vapor(x, alpha);
        assert_relative_eq!(y / (1.0 - y), alpha * x / (1.0 - x), epsilon = 1e-12);
    }

    #[test]
    fn fenske_recovers_exact_power_case() {
        // Si le facteur de séparation vaut α^N, Fenske doit rendre N.
        // (d_L/d_H)·(b_H/b_L) = (8/2)·(8/2) = 4·4 = 16 = 2^4, avec α = 2 → Nₘ = 4.
        let n = dist_fenske_minimum_stages(8.0, 2.0, 2.0, 8.0, 2.0);
        assert_relative_eq!(n, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn rectifying_slope_bounds_and_complement() {
        // s = R/(R+1) : à R = 0 la pente est nulle, à R = 3 elle vaut 0,75.
        assert_relative_eq!(dist_rectifying_operating_slope(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(dist_rectifying_operating_slope(3.0), 0.75, epsilon = 1e-12);
        // Complément : 1 − s = 1/(R+1) (ordonnée à l'origine réduite).
        let r = 4.0_f64;
        let s = dist_rectifying_operating_slope(r);
        assert_relative_eq!(1.0 - s, 1.0 / (r + 1.0), epsilon = 1e-12);
    }

    #[test]
    fn minimum_reflux_pinch_realistic_case() {
        // Rₘ = (x_D − y*)/(y* − z_F) avec x_D = 0,95, y* = 0,70, z_F = 0,40.
        // Rₘ = (0,95 − 0,70)/(0,70 − 0,40) = 0,25/0,30 = 5/6 ≈ 0,833333.
        let r_min = dist_minimum_reflux_ratio(0.95, 0.70, 0.40);
        assert_relative_eq!(r_min, 5.0 / 6.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(
        expected = "volatilité relative doit être finie, strictement positive et différente de 1"
    )]
    fn fenske_panics_when_alpha_is_one() {
        // α = 1 : ln(α) = 0, division par zéro (séparation impossible au reflux total).
        dist_fenske_minimum_stages(8.0, 2.0, 2.0, 8.0, 1.0);
    }
}
