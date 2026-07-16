//! Équilibre liquide-vapeur **idéal** décrit par la **loi de Raoult** : pressions
//! partielles, constantes d'équilibre `K`, pression de bulle et composition de la
//! vapeur d'un binaire, volatilité relative.
//!
//! ```text
//! loi de Raoult      pᵢ = xᵢ·P_sat,ᵢ                              [Pa]
//! constante d'équil. Kᵢ = P_sat,ᵢ / P                             [sans dimension]
//! pression de bulle  P_bulle = x₁·P_sat,1 + (1 − x₁)·P_sat,2      [Pa]  (binaire)
//! composition vapeur y₁ = x₁·P_sat,1 / P_bulle                    [sans dimension]
//! volatilité relative α = P_sat,1 / P_sat,2                       [sans dimension]
//! ```
//!
//! `xᵢ` fraction molaire du constituant dans le liquide [sans dimension, dans
//! `[0, 1]`], `yᵢ` fraction molaire dans la vapeur [sans dimension, dans `[0, 1]`],
//! `pᵢ` pression partielle du constituant [Pa], `P_sat,ᵢ` pression de vapeur
//! saturante du corps pur à la température considérée [Pa, strictement positive],
//! `P` pression totale du système [Pa, strictement positive], `P_bulle` pression de
//! bulle du mélange (pression totale au point de bulle) [Pa], `Kᵢ = yᵢ/xᵢ`
//! coefficient de partage (constante d'équilibre) du constituant [sans dimension],
//! `α` volatilité relative du constituant 1 par rapport au constituant 2
//! [sans dimension].
//!
//! **Limite honnête** : ces relations supposent un **mélange idéal** (loi de Raoult,
//! phase vapeur assimilée à un gaz parfait, absence d'interactions spécifiques). Les
//! **pressions de vapeur saturantes** `P_sat,ᵢ` des **corps purs**, à la
//! température considérée, sont **fournies par l'appelant** (corrélation d'Antoine,
//! table, mesure…) : aucune valeur n'est inventée ici. Les constantes d'équilibre
//! `Kᵢ` et la volatilité relative `α` en **dérivent** directement. Pour un mélange
//! **non idéal**, `pᵢ = γᵢ·xᵢ·P_sat,ᵢ` : les **coefficients d'activité** `γᵢ`
//! (modèles de Wilson, NRTL, UNIQUAC…) sont **à la charge de l'appelant**. Les gaz
//! **dilués** relèvent plutôt de la **loi de Henry** ; ce module ne couvre ni la
//! résolution de la boucle bulle/rosée ni les bilans enthalpiques.

/// Pression partielle d'un constituant selon la loi de Raoult `pᵢ = xᵢ·P_sat,ᵢ`
/// [Pa].
///
/// `liquid_fraction` `xᵢ` fraction molaire dans le liquide [sans dimension, dans
/// `[0, 1]`], `vapor_pressure_pure` `P_sat,ᵢ` pression de vapeur saturante du corps
/// pur **fournie** [Pa, strictement positive].
///
/// Panique si `liquid_fraction` sort de `[0, 1]` ou si `vapor_pressure_pure` n'est
/// pas fini et strictement positif.
pub fn vle_partial_pressure_raoult(liquid_fraction: f64, vapor_pressure_pure: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&liquid_fraction),
        "la fraction molaire liquide doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        vapor_pressure_pure.is_finite() && vapor_pressure_pure > 0.0,
        "la pression de vapeur saturante doit être finie et strictement positive (Pa)"
    );
    liquid_fraction * vapor_pressure_pure
}

/// Constante d'équilibre (coefficient de partage) d'un constituant en mélange idéal
/// `Kᵢ = P_sat,ᵢ / P` [sans dimension].
///
/// `vapor_pressure_pure` `P_sat,ᵢ` pression de vapeur saturante du corps pur
/// **fournie** [Pa, strictement positive], `total_pressure` `P` pression totale du
/// système [Pa, strictement positive].
///
/// Panique si `vapor_pressure_pure` ou `total_pressure` n'est pas fini et
/// strictement positif.
pub fn vle_equilibrium_ratio(vapor_pressure_pure: f64, total_pressure: f64) -> f64 {
    assert!(
        vapor_pressure_pure.is_finite() && vapor_pressure_pure > 0.0,
        "la pression de vapeur saturante doit être finie et strictement positive (Pa)"
    );
    assert!(
        total_pressure.is_finite() && total_pressure > 0.0,
        "la pression totale doit être finie et strictement positive (Pa)"
    );
    vapor_pressure_pure / total_pressure
}

/// Pression de bulle d'un mélange binaire idéal
/// `P_bulle = x₁·P_sat,1 + (1 − x₁)·P_sat,2` [Pa].
///
/// C'est la somme des pressions partielles des deux constituants ; elle donne la
/// pression totale à laquelle le liquide de composition `x₁` commence à bouillir.
///
/// `liquid_fraction_1` `x₁` fraction molaire du constituant 1 dans le liquide
/// [sans dimension, dans `[0, 1]`], `vapor_pressure_1` `P_sat,1` et
/// `vapor_pressure_2` `P_sat,2` pressions de vapeur saturantes des corps purs
/// **fournies** [Pa, strictement positives].
///
/// Panique si `liquid_fraction_1` sort de `[0, 1]` ou si l'une des pressions de
/// vapeur n'est pas finie et strictement positive.
pub fn vle_bubble_pressure_binary(
    liquid_fraction_1: f64,
    vapor_pressure_1: f64,
    vapor_pressure_2: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&liquid_fraction_1),
        "la fraction molaire liquide du constituant 1 doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        vapor_pressure_1.is_finite() && vapor_pressure_1 > 0.0,
        "la pression de vapeur saturante du constituant 1 doit être finie et strictement positive (Pa)"
    );
    assert!(
        vapor_pressure_2.is_finite() && vapor_pressure_2 > 0.0,
        "la pression de vapeur saturante du constituant 2 doit être finie et strictement positive (Pa)"
    );
    liquid_fraction_1 * vapor_pressure_1 + (1.0 - liquid_fraction_1) * vapor_pressure_2
}

/// Fraction molaire du constituant 1 dans la vapeur d'un binaire idéal
/// `y₁ = x₁·P_sat,1 / P_bulle` [sans dimension].
///
/// Au point de bulle, la pression totale vaut `P_bulle` ; cette relation équivaut
/// donc à `y₁ = K₁·x₁` avec `K₁ = P_sat,1/P_bulle`. Le résultat appartient à
/// `[0, 1]` dès lors que `P_bulle ≥ x₁·P_sat,1`, ce qui est garanti pour un binaire
/// idéal cohérent (`P_bulle` issu de [`vle_bubble_pressure_binary`]).
///
/// `liquid_fraction_1` `x₁` fraction molaire du constituant 1 dans le liquide
/// [sans dimension, dans `[0, 1]`], `vapor_pressure_1` `P_sat,1` pression de vapeur
/// saturante du corps pur 1 **fournie** [Pa, strictement positive],
/// `bubble_pressure` `P_bulle` pression de bulle du mélange [Pa, strictement
/// positive].
///
/// Panique si `liquid_fraction_1` sort de `[0, 1]`, ou si `vapor_pressure_1` ou
/// `bubble_pressure` n'est pas fini et strictement positif.
pub fn vle_vapor_fraction_binary(
    liquid_fraction_1: f64,
    vapor_pressure_1: f64,
    bubble_pressure: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&liquid_fraction_1),
        "la fraction molaire liquide du constituant 1 doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        vapor_pressure_1.is_finite() && vapor_pressure_1 > 0.0,
        "la pression de vapeur saturante du constituant 1 doit être finie et strictement positive (Pa)"
    );
    assert!(
        bubble_pressure.is_finite() && bubble_pressure > 0.0,
        "la pression de bulle doit être finie et strictement positive (Pa)"
    );
    liquid_fraction_1 * vapor_pressure_1 / bubble_pressure
}

/// Volatilité relative du constituant 1 par rapport au constituant 2 en mélange
/// idéal `α = P_sat,1 / P_sat,2` [sans dimension].
///
/// `vapor_pressure_1` `P_sat,1` et `vapor_pressure_2` `P_sat,2` pressions de vapeur
/// saturantes des corps purs **fournies** [Pa, strictement positives]. Une valeur
/// `α > 1` indique que le constituant 1 est le plus volatil.
///
/// Panique si `vapor_pressure_1` ou `vapor_pressure_2` n'est pas fini et strictement
/// positif.
pub fn vle_relative_volatility_raoult(vapor_pressure_1: f64, vapor_pressure_2: f64) -> f64 {
    assert!(
        vapor_pressure_1.is_finite() && vapor_pressure_1 > 0.0,
        "la pression de vapeur saturante du constituant 1 doit être finie et strictement positive (Pa)"
    );
    assert!(
        vapor_pressure_2.is_finite() && vapor_pressure_2 > 0.0,
        "la pression de vapeur saturante du constituant 2 doit être finie et strictement positive (Pa)"
    );
    vapor_pressure_1 / vapor_pressure_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas binaire de référence, type benzène(1)/toluène(2) à une température donnée,
    // pressions de vapeur saturantes fournies (en pascals) :
    //   P_sat,1 = 135 000 Pa, P_sat,2 = 54 000 Pa, P = 101 325 Pa, x1 = 0,5.
    // Pression partielle p1 = 0,5·135 000 = 67 500 Pa.
    // Pression de bulle  = 0,5·135 000 + 0,5·54 000 = 67 500 + 27 000 = 94 500 Pa.
    // Fraction vapeur y1 = 67 500 / 94 500 = 5/7 ≈ 0,714 285 714.
    // Volatilité relative α = 135 000 / 54 000 = 2,5.
    const PSAT1: f64 = 135_000.0;
    const PSAT2: f64 = 54_000.0;
    const P_TOT: f64 = 101_325.0;
    const X1: f64 = 0.5;

    #[test]
    fn realistic_binary_reference_case() {
        // Valeurs chiffrées recalculées à la main (voir commentaire ci-dessus).
        let p1 = vle_partial_pressure_raoult(X1, PSAT1);
        assert_relative_eq!(p1, 67_500.0, epsilon = 1e-6);

        let p_bubble = vle_bubble_pressure_binary(X1, PSAT1, PSAT2);
        assert_relative_eq!(p_bubble, 94_500.0, epsilon = 1e-6);

        let y1 = vle_vapor_fraction_binary(X1, PSAT1, p_bubble);
        assert_relative_eq!(y1, 5.0 / 7.0, epsilon = 1e-3);

        let alpha = vle_relative_volatility_raoult(PSAT1, PSAT2);
        assert_relative_eq!(alpha, 2.5, epsilon = 1e-12);
    }

    #[test]
    fn bubble_pressure_is_sum_of_partial_pressures() {
        // Identité : P_bulle = p1 + p2 (somme des pressions partielles de Raoult).
        let p1 = vle_partial_pressure_raoult(X1, PSAT1);
        let p2 = vle_partial_pressure_raoult(1.0 - X1, PSAT2);
        let p_bubble = vle_bubble_pressure_binary(X1, PSAT1, PSAT2);
        assert_relative_eq!(p1 + p2, p_bubble, epsilon = 1e-9);
    }

    #[test]
    fn vapor_fraction_equals_k_times_liquid_at_bubble_point() {
        // Au point de bulle, P = P_bulle donc y1 = K1·x1 avec K1 = P_sat,1/P_bulle.
        let p_bubble = vle_bubble_pressure_binary(X1, PSAT1, PSAT2);
        let y1 = vle_vapor_fraction_binary(X1, PSAT1, p_bubble);
        let k1 = vle_equilibrium_ratio(PSAT1, p_bubble);
        assert_relative_eq!(y1, k1 * X1, epsilon = 1e-12);
        // Les fractions vapeur des deux constituants somment à 1.
        let p_bubble_2 = vle_bubble_pressure_binary(X1, PSAT1, PSAT2);
        let y2 = (1.0 - X1) * PSAT2 / p_bubble_2;
        assert_relative_eq!(y1 + y2, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn relative_volatility_is_ratio_of_k_values() {
        // α = P_sat,1/P_sat,2 = K1/K2, indépendant de la pression totale de référence.
        let alpha = vle_relative_volatility_raoult(PSAT1, PSAT2);
        let k1 = vle_equilibrium_ratio(PSAT1, P_TOT);
        let k2 = vle_equilibrium_ratio(PSAT2, P_TOT);
        assert_relative_eq!(alpha, k1 / k2, epsilon = 1e-12);
        // Réciprocité : la volatilité inverse est bien 1/α.
        let alpha_inv = vle_relative_volatility_raoult(PSAT2, PSAT1);
        assert_relative_eq!(alpha * alpha_inv, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_component_limit() {
        // Constituant pur (x1 = 1) : la pression de bulle vaut P_sat,1 et y1 = 1.
        let p_bubble = vle_bubble_pressure_binary(1.0, PSAT1, PSAT2);
        assert_relative_eq!(p_bubble, PSAT1, epsilon = 1e-12);
        let y1 = vle_vapor_fraction_binary(1.0, PSAT1, p_bubble);
        assert_relative_eq!(y1, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn partial_pressure_is_proportional_to_liquid_fraction() {
        // p = x·P_sat : doubler x double la pression partielle (P_sat fixée).
        let p_small = vle_partial_pressure_raoult(0.2, PSAT1);
        let p_double = vle_partial_pressure_raoult(0.4, PSAT1);
        assert_relative_eq!(p_double, 2.0 * p_small, epsilon = 1e-9);
        assert_relative_eq!(p_small, 27_000.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn nonpositive_vapor_pressure_panics() {
        // Une pression de vapeur saturante nulle n'a pas de sens physique : rejet.
        vle_equilibrium_ratio(0.0, P_TOT);
    }
}
