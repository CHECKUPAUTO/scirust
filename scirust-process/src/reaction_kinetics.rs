//! Cinétique chimique **homogène** — loi de vitesse d'ordre `n`, dépendance en
//! température (Arrhenius), conversion en réacteur batch d'ordre 1 et
//! identification de l'énergie d'activation sur deux températures.
//!
//! ```text
//! vitesse d'ordre n   r  = k · Cⁿ                                 [mol·m⁻³·s⁻¹]
//! Arrhenius           k  = A · exp(−Eₐ / (R·T))                   [unité de k]
//! conversion batch    X  = 1 − exp(−k·t)   (ordre 1)              [sans dimension]
//! demi-vie (ordre 1)  t½ = ln 2 / k                               [s]
//! énergie d'activation Eₐ = R · ln(k₂/k₁) / (1/T₁ − 1/T₂)         [J·mol⁻¹]
//! ```
//!
//! `r` vitesse de réaction [mol·m⁻³·s⁻¹, cohérente avec `k` et `C`], `k` constante
//! de vitesse [unité dépendant de l'ordre], `C` concentration du réactif
//! [mol·m⁻³ ou unité cohérente avec `k`], `n` ordre de réaction [sans dimension,
//! fourni], `A` facteur pré-exponentiel [même unité que `k`], `Eₐ` énergie
//! d'activation [J·mol⁻¹], `R` constante des gaz parfaits [J·mol⁻¹·K⁻¹], `T`
//! température **absolue** [K], `X` taux de conversion [sans dimension], `t` temps
//! de séjour/réaction [s], `t½` demi-vie [s], `k₁`/`k₂` constantes de vitesse aux
//! températures `T₁`/`T₂` [même unité].
//!
//! **Limite honnête** : ces relations décrivent une **cinétique homogène** avec un
//! **ordre de réaction fourni** (aucun mécanisme n'est déduit). La constante de
//! vitesse `k`, le couple `(A, Eₐ)`, les concentrations et la constante des gaz
//! `R` sont **fournis par l'appelant** (mesures, tables, régression) ; aucune
//! valeur « par défaut » n'est inventée. Les **températures sont en kelvin**
//! (absolues) et les **unités de `C` doivent être cohérentes avec `k`** pour que
//! `r` ait un sens. La loi d'Arrhenius suppose `Eₐ` et `A` **indépendants de la
//! température** sur l'intervalle considéré ; la conversion `X` suppose un
//! **réacteur batch isotherme à volume constant** et une **réaction d'ordre 1**.

/// Vitesse d'une réaction d'ordre `n` : `r = k · Cⁿ` [mol·m⁻³·s⁻¹].
///
/// `rate_constant` `k` constante de vitesse [unité cohérente avec l'ordre],
/// `concentration` `C` concentration du réactif [mol·m⁻³ ou unité cohérente],
/// `order` `n` ordre de réaction [sans dimension, fourni]. L'ordre peut être non
/// entier (cinétique apparente).
///
/// Panique si `rate_constant` ou `concentration` est négatif ou non fini, ou si
/// `order` n'est pas fini.
pub fn kinetics_rate(rate_constant: f64, concentration: f64, order: f64) -> f64 {
    assert!(
        rate_constant.is_finite() && rate_constant >= 0.0,
        "la constante de vitesse doit être finie et positive ou nulle"
    );
    assert!(
        concentration.is_finite() && concentration >= 0.0,
        "la concentration doit être finie et positive ou nulle (mol·m⁻³)"
    );
    assert!(order.is_finite(), "l'ordre de réaction doit être fini");
    rate_constant * concentration.powf(order)
}

/// Constante de vitesse selon la loi d'Arrhenius :
/// `k = A · exp(−Eₐ / (R·T))` [même unité que `A`].
///
/// `pre_exponential` `A` facteur pré-exponentiel [même unité que `k`],
/// `activation_energy` `Eₐ` énergie d'activation [J·mol⁻¹], `gas_constant` `R`
/// constante des gaz [J·mol⁻¹·K⁻¹], `temperature` `T` température **absolue** [K].
///
/// Panique si `pre_exponential` ou `activation_energy` est négatif ou non fini, ou
/// si `gas_constant` ou `temperature` n'est pas strictement positif.
pub fn kinetics_arrhenius(
    pre_exponential: f64,
    activation_energy: f64,
    gas_constant: f64,
    temperature: f64,
) -> f64 {
    assert!(
        pre_exponential.is_finite() && pre_exponential >= 0.0,
        "le facteur pré-exponentiel doit être fini et positif ou nul"
    );
    assert!(
        activation_energy.is_finite() && activation_energy >= 0.0,
        "l'énergie d'activation doit être finie et positive ou nulle (J·mol⁻¹)"
    );
    assert!(
        gas_constant > 0.0,
        "la constante des gaz doit être strictement positive (J·mol⁻¹·K⁻¹)"
    );
    assert!(
        temperature > 0.0,
        "la température absolue doit être strictement positive (K)"
    );
    pre_exponential * (-activation_energy / (gas_constant * temperature)).exp()
}

/// Taux de conversion d'une réaction d'**ordre 1** en réacteur batch isotherme :
/// `X = 1 − exp(−k·t)` [sans dimension, dans `[0, 1[`].
///
/// `rate_constant` `k` constante de vitesse d'ordre 1 [s⁻¹], `time` `t` durée de
/// réaction [s]. `X → 1` quand `t → ∞` ; `X = 0` à `t = 0`.
///
/// Panique si `rate_constant` ou `time` est négatif ou non fini.
pub fn kinetics_first_order_conversion(rate_constant: f64, time: f64) -> f64 {
    assert!(
        rate_constant.is_finite() && rate_constant >= 0.0,
        "la constante de vitesse doit être finie et positive ou nulle (s⁻¹)"
    );
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    1.0 - (-rate_constant * time).exp()
}

/// Demi-vie d'une réaction d'**ordre 1** : `t½ = ln 2 / k` [s],
/// indépendante de la concentration initiale.
///
/// `rate_constant` `k` constante de vitesse d'ordre 1 [s⁻¹].
///
/// Panique si `rate_constant` n'est pas strictement positif (division) ou n'est
/// pas fini.
pub fn kinetics_half_life_first_order(rate_constant: f64) -> f64 {
    assert!(
        rate_constant.is_finite() && rate_constant > 0.0,
        "la constante de vitesse doit être finie et strictement positive (s⁻¹)"
    );
    core::f64::consts::LN_2 / rate_constant
}

/// Énergie d'activation identifiée sur **deux températures** :
/// `Eₐ = R · ln(k₂/k₁) / (1/T₁ − 1/T₂)` [J·mol⁻¹].
///
/// `rate1` `k₁` et `rate2` `k₂` constantes de vitesse aux températures `T₁` et
/// `T₂` [même unité], `temperature1` `T₁` et `temperature2` `T₂` températures
/// **absolues** [K], `gas_constant` `R` constante des gaz [J·mol⁻¹·K⁻¹]. Forme
/// intégrée de l'équation d'Arrhenius entre deux points.
///
/// Panique si `rate1`, `rate2`, `temperature1`, `temperature2` ou `gas_constant`
/// n'est pas strictement positif ou n'est pas fini, ou si les deux températures
/// sont égales (dénominateur nul).
pub fn kinetics_activation_energy_from_two_rates(
    rate1: f64,
    rate2: f64,
    temperature1: f64,
    temperature2: f64,
    gas_constant: f64,
) -> f64 {
    assert!(
        rate1.is_finite() && rate1 > 0.0,
        "la constante de vitesse k₁ doit être finie et strictement positive"
    );
    assert!(
        rate2.is_finite() && rate2 > 0.0,
        "la constante de vitesse k₂ doit être finie et strictement positive"
    );
    assert!(
        temperature1.is_finite() && temperature1 > 0.0,
        "la température T₁ doit être finie et strictement positive (K)"
    );
    assert!(
        temperature2.is_finite() && temperature2 > 0.0,
        "la température T₂ doit être finie et strictement positive (K)"
    );
    assert!(
        gas_constant > 0.0,
        "la constante des gaz doit être strictement positive (J·mol⁻¹·K⁻¹)"
    );
    assert!(
        temperature1 != temperature2,
        "les deux températures doivent être distinctes (dénominateur nul)"
    );
    gas_constant * (rate2 / rate1).ln() / (1.0 / temperature1 - 1.0 / temperature2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_second_order_known_case() {
        // Ordre 2 : r = k·C² = 2·3² = 2·9 = 18.
        assert_relative_eq!(kinetics_rate(2.0, 3.0, 2.0), 18.0, epsilon = 1e-12);
    }

    #[test]
    fn rate_order_zero_is_constant_order_one_is_linear() {
        // Ordre 0 : r = k·C⁰ = k, indépendant de la concentration.
        assert_relative_eq!(kinetics_rate(0.5, 4.0, 0.0), 0.5, epsilon = 1e-12);
        // Ordre 1 : r = k·C = 0,5·4 = 2, et doubler C double r.
        assert_relative_eq!(kinetics_rate(0.5, 4.0, 1.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(kinetics_rate(0.5, 8.0, 1.0), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn arrhenius_reduces_to_pre_exponential_when_ea_zero() {
        // Eₐ = 0 → exp(0) = 1 → k = A, quelle que soit T.
        assert_relative_eq!(
            kinetics_arrhenius(1.0e10, 0.0, 8.314, 298.15),
            1.0e10,
            epsilon = 1e-3
        );
    }

    #[test]
    fn first_order_conversion_known_case_and_origin() {
        // k·t = 0,1·10 = 1 → X = 1 − e⁻¹ = 1 − 0,367879441… = 0,632120558…
        let expected = 1.0 - (-1.0_f64).exp();
        assert_relative_eq!(
            kinetics_first_order_conversion(0.1, 10.0),
            expected,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            kinetics_first_order_conversion(0.1, 10.0),
            0.632_120_558_8,
            epsilon = 1e-9
        );
        // À t = 0 : aucune conversion.
        assert_relative_eq!(
            kinetics_first_order_conversion(0.1, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn conversion_reaches_one_half_at_half_life() {
        // Par construction t½ = ln2/k, la conversion vaut exactement 1/2 en t½.
        let k = 0.023_f64;
        let t_half = kinetics_half_life_first_order(k);
        // Vérifie aussi le lien k·t½ = ln 2.
        assert_relative_eq!(k * t_half, core::f64::consts::LN_2, epsilon = 1e-12);
        assert_relative_eq!(
            kinetics_first_order_conversion(k, t_half),
            0.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn activation_energy_round_trips_through_arrhenius() {
        // On génère k₁, k₂ avec un Eₐ connu, puis on le ré-identifie :
        // ln(k₂/k₁) = (Eₐ/R)(1/T₁ − 1/T₂) donc Eₐ_id = Eₐ exactement.
        let a = 1.0e13_f64;
        let ea = 75_000.0_f64;
        let r = 8.314_f64;
        let t1 = 300.0_f64;
        let t2 = 320.0_f64;
        let k1 = kinetics_arrhenius(a, ea, r, t1);
        let k2 = kinetics_arrhenius(a, ea, r, t2);
        assert_relative_eq!(
            kinetics_activation_energy_from_two_rates(k1, k2, t1, t2, r),
            ea,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_rate_constant_half_life_panics() {
        // t½ = ln2/k avec k = 0 : division rejetée.
        kinetics_half_life_first_order(0.0);
    }
}
