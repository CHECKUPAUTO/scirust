//! Équilibre chimique — constante d'équilibre en température (loi de van't Hoff),
//! conversion à l'équilibre d'une réaction réversible équimolaire et lien avec
//! l'enthalpie libre standard `ΔG°`.
//!
//! ```text
//! van't Hoff       K(T) = K_ref · exp[ (−ΔHr/R) · (1/T − 1/T_ref) ]   [sans dim.]
//! conversion A⇌B   X_eq = K / (1 + K)                                  [sans dim.]
//! ΔG° depuis K     ΔG°  = −R·T·ln K                                    [J·mol⁻¹]
//! K depuis ΔG°     K    = exp[ −ΔG° / (R·T) ]                          [sans dim.]
//! ```
//!
//! `K`/`K_ref` constante d'équilibre [sans dimension], `ΔHr` enthalpie standard de
//! réaction [J·mol⁻¹, positive endothermique, négative exothermique], `R` constante
//! des gaz parfaits [J·mol⁻¹·K⁻¹], `T`/`T_ref` températures **absolues** [K],
//! `X_eq` conversion à l'équilibre d'une réaction `A⇌B` équimolaire [sans dimension,
//! dans `[0, 1[`], `ΔG°` enthalpie libre standard de réaction [J·mol⁻¹].
//!
//! **Limite honnête** : ces relations décrivent un **équilibre thermodynamique** en
//! **gaz parfaits / solutions idéales**. La constante d'équilibre de référence
//! `K_ref`, l'**enthalpie de réaction** `ΔHr` et la constante des gaz `R` sont
//! **fournies par l'appelant** (tables, mesures, régression) ; aucune propriété
//! n'est inventée. La loi de van't Hoff suppose `ΔHr` **constant** sur l'intervalle
//! de température considéré. Les **températures sont en kelvin** (absolues). La
//! conversion à l'équilibre `X_eq` ne concerne que la stœchiométrie `A⇌B`
//! équimolaire et **borne** (majore) la conversion atteignable par un réacteur réel.

/// Constante d'équilibre à la température `T` par la loi de van't Hoff intégrée :
/// `K(T) = K_ref · exp[ (−ΔHr/R) · (1/T − 1/T_ref) ]` [sans dimension].
///
/// `equilibrium_constant_ref` `K_ref` constante d'équilibre à la température de
/// référence [sans dimension], `enthalpy_of_reaction` `ΔHr` enthalpie standard de
/// réaction [J·mol⁻¹], `gas_constant` `R` constante des gaz [J·mol⁻¹·K⁻¹],
/// `temperature_ref` `T_ref` température de référence **absolue** [K],
/// `temperature` `T` température cible **absolue** [K]. Pour `ΔHr < 0`
/// (exothermique) `K` **décroît** avec `T` ; pour `ΔHr > 0` (endothermique) `K`
/// **croît** avec `T`. À `T = T_ref`, `K = K_ref`.
///
/// Panique si `equilibrium_constant_ref` n'est pas strictement positif, si
/// `enthalpy_of_reaction` n'est pas fini, si `gas_constant`, `temperature_ref` ou
/// `temperature` n'est pas strictement positif ou n'est pas fini.
pub fn equil_vant_hoff(
    equilibrium_constant_ref: f64,
    enthalpy_of_reaction: f64,
    gas_constant: f64,
    temperature_ref: f64,
    temperature: f64,
) -> f64 {
    assert!(
        equilibrium_constant_ref.is_finite() && equilibrium_constant_ref > 0.0,
        "la constante d'équilibre de référence doit être finie et strictement positive"
    );
    assert!(
        enthalpy_of_reaction.is_finite(),
        "l'enthalpie de réaction doit être finie (J·mol⁻¹)"
    );
    assert!(
        gas_constant.is_finite() && gas_constant > 0.0,
        "la constante des gaz doit être finie et strictement positive (J·mol⁻¹·K⁻¹)"
    );
    assert!(
        temperature_ref.is_finite() && temperature_ref > 0.0,
        "la température de référence doit être finie et strictement positive (K)"
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "la température cible doit être finie et strictement positive (K)"
    );
    equilibrium_constant_ref
        * ((-enthalpy_of_reaction / gas_constant) * (1.0 / temperature - 1.0 / temperature_ref))
            .exp()
}

/// Conversion à l'équilibre d'une réaction réversible équimolaire `A⇌B` :
/// `X_eq = K / (1 + K)` [sans dimension, dans `[0, 1[`].
///
/// `equilibrium_constant` `K` constante d'équilibre [sans dimension]. `X_eq → 0`
/// quand `K → 0`, `X_eq = 1/2` pour `K = 1`, `X_eq → 1` quand `K → ∞`. Cette
/// conversion majore (borne) la conversion d'un réacteur réel.
///
/// Panique si `equilibrium_constant` est négatif ou n'est pas fini.
pub fn equil_conversion_first_order_reversible(equilibrium_constant: f64) -> f64 {
    assert!(
        equilibrium_constant.is_finite() && equilibrium_constant >= 0.0,
        "la constante d'équilibre doit être finie et positive ou nulle"
    );
    equilibrium_constant / (1.0 + equilibrium_constant)
}

/// Enthalpie libre standard de réaction à partir de la constante d'équilibre :
/// `ΔG° = −R·T·ln K` [J·mol⁻¹].
///
/// `gas_constant` `R` constante des gaz [J·mol⁻¹·K⁻¹], `temperature` `T`
/// température **absolue** [K], `equilibrium_constant` `K` constante d'équilibre
/// [sans dimension]. `ΔG° < 0` pour `K > 1` (réaction favorisée), `ΔG° = 0` pour
/// `K = 1`, `ΔG° > 0` pour `K < 1`.
///
/// Panique si `gas_constant` ou `temperature` n'est pas strictement positif, ou si
/// `equilibrium_constant` n'est pas strictement positif (logarithme), ou si l'une
/// des valeurs n'est pas finie.
pub fn equil_gibbs_from_k(gas_constant: f64, temperature: f64, equilibrium_constant: f64) -> f64 {
    assert!(
        gas_constant.is_finite() && gas_constant > 0.0,
        "la constante des gaz doit être finie et strictement positive (J·mol⁻¹·K⁻¹)"
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "la température absolue doit être finie et strictement positive (K)"
    );
    assert!(
        equilibrium_constant.is_finite() && equilibrium_constant > 0.0,
        "la constante d'équilibre doit être finie et strictement positive (logarithme)"
    );
    -gas_constant * temperature * equilibrium_constant.ln()
}

/// Constante d'équilibre à partir de l'enthalpie libre standard de réaction :
/// `K = exp[ −ΔG° / (R·T) ]` [sans dimension].
///
/// `gibbs_energy` `ΔG°` enthalpie libre standard de réaction [J·mol⁻¹],
/// `gas_constant` `R` constante des gaz [J·mol⁻¹·K⁻¹], `temperature` `T`
/// température **absolue** [K]. Réciproque de [`equil_gibbs_from_k`].
///
/// Panique si `gibbs_energy` n'est pas fini, ou si `gas_constant` ou `temperature`
/// n'est pas strictement positif ou n'est pas fini.
pub fn equil_k_from_gibbs(gibbs_energy: f64, gas_constant: f64, temperature: f64) -> f64 {
    assert!(
        gibbs_energy.is_finite(),
        "l'enthalpie libre standard doit être finie (J·mol⁻¹)"
    );
    assert!(
        gas_constant.is_finite() && gas_constant > 0.0,
        "la constante des gaz doit être finie et strictement positive (J·mol⁻¹·K⁻¹)"
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "la température absolue doit être finie et strictement positive (K)"
    );
    (-gibbs_energy / (gas_constant * temperature)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn vant_hoff_returns_reference_at_reference_temperature() {
        // À T = T_ref, l'exposant s'annule → K = K_ref exactement.
        let k = equil_vant_hoff(4.0, -52_000.0, 8.314, 298.15, 298.15);
        assert_relative_eq!(k, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn vant_hoff_round_trip_is_reciprocal() {
        // Passer de T_ref à T puis de T à T_ref restitue K_ref.
        let k_ref = 2.5_f64;
        let dh = -48_000.0_f64;
        let r = 8.314_f64;
        let t_ref = 300.0_f64;
        let t = 360.0_f64;
        let k_t = equil_vant_hoff(k_ref, dh, r, t_ref, t);
        let back = equil_vant_hoff(k_t, dh, r, t, t_ref);
        assert_relative_eq!(back, k_ref, epsilon = 1e-9);
        // Exothermique (ΔHr < 0) : chauffer diminue K.
        assert!(k_t < k_ref);
    }

    #[test]
    fn conversion_limits_and_symmetry() {
        // K = 1 → X = 1/2 ; K = 3 → X = 3/4 ; K = 0 → X = 0.
        assert_relative_eq!(
            equil_conversion_first_order_reversible(1.0),
            0.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            equil_conversion_first_order_reversible(3.0),
            0.75,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            equil_conversion_first_order_reversible(0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn gibbs_and_k_are_inverse() {
        // K → ΔG° → K restitue la valeur de départ.
        let r = 8.314_f64;
        let t = 298.15_f64;
        let k = 12.0_f64;
        let g = equil_gibbs_from_k(r, t, k);
        assert_relative_eq!(equil_k_from_gibbs(g, r, t), k, epsilon = 1e-9);
        // K > 1 → ΔG° < 0 (réaction favorisée).
        assert!(g < 0.0);
    }

    #[test]
    fn gibbs_from_k_known_case() {
        // K = e → ln K = 1 → ΔG° = −R·T = −8,314·298,15 = −2478,8191 J·mol⁻¹.
        let g = equil_gibbs_from_k(8.314, 298.15, core::f64::consts::E);
        assert_relative_eq!(g, -2478.8191, epsilon = 1e-3);
        // K = 1 → ΔG° = 0 exactement.
        assert_relative_eq!(equil_gibbs_from_k(8.314, 298.15, 1.0), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn k_from_gibbs_zero_gives_unity() {
        // ΔG° = 0 → exp(0) = 1, quelle que soit T.
        assert_relative_eq!(equil_k_from_gibbs(0.0, 8.314, 298.15), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn gibbs_from_zero_k_panics() {
        // ln K avec K = 0 : rejeté.
        equil_gibbs_from_k(8.314, 298.15, 0.0);
    }
}
