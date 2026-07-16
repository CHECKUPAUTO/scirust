//! Réacteur **adiabatique** — élévation de température liée à l'avancement d'une
//! réaction par le bilan enthalpique (aucun échange de chaleur avec l'extérieur).
//!
//! ```text
//! élévation            ΔT      = (−ΔHr)·C_A0·X / (ρ·cp)     [K]
//! température de sortie T       = T_in + ΔT                  [K]
//! élévation maximale    ΔT_max  = (−ΔHr)·C_A0 / (ρ·cp)       [K]  (X = 1)
//! conversion(T)         X       = (T − T_in) / ΔT_max        [sans dimension]
//! ```
//!
//! `ΔT` élévation de température adiabatique [K], `ΔHr` chaleur (enthalpie) de
//! réaction **par mole de réactif limitant** [J/mol] (`ΔHr < 0` exothermique),
//! `C_A0` concentration d'entrée du réactif limitant [mol/m³], `X` taux de
//! conversion [sans dimension, dans `[0, 1]`], `ρ` masse volumique du mélange
//! [kg/m³], `cp` capacité thermique massique du mélange [J/(kg·K)], `T_in`
//! température d'entrée [K], `T` température [K], `ΔT_max` élévation à conversion
//! totale [K]. La relation `X ↔ T` est **linéaire** (droite adiabatique du bilan
//! enthalpique).
//!
//! **Limite honnête** : le réacteur est supposé **strictement adiabatique**
//! (aucun échange thermique). La chaleur de réaction `ΔHr`, la masse volumique
//! `ρ` et la capacité thermique `cp` du mélange sont **fournies par l'appelant**
//! (table, corrélation, mesure) et supposées **constantes** sur l'intervalle de
//! température : aucune enthalpie, constante cinétique, volatilité ou isotherme
//! n'est inventée. La linéarité `X ↔ T` découle directement de ces hypothèses de
//! propriétés constantes ; hors de leur domaine de validité (changement de
//! phase, cp fortement variable) la droite adiabatique n'est qu'approchée.

/// Élévation de température adiabatique
/// `ΔT = (−ΔHr)·C_A0·X / (ρ·cp)` [K].
///
/// `heat_of_reaction` `ΔHr` chaleur de réaction par mole de réactif limitant
/// [J/mol] (négative pour une réaction exothermique, d'où `ΔT > 0`),
/// `inlet_concentration` `C_A0` concentration d'entrée du réactif limitant
/// [mol/m³], `conversion` `X` taux de conversion [sans dimension, dans `[0, 1]`],
/// `density` `ρ` masse volumique du mélange [kg/m³], `specific_heat` `cp`
/// capacité thermique massique du mélange [J/(kg·K)].
///
/// Panique si `heat_of_reaction` n'est pas fini, si `inlet_concentration` est
/// négatif ou non fini, si `conversion` sort de `[0, 1]`, ou si `density` ou
/// `specific_heat` n'est pas strictement positif (division).
pub fn adiab_temperature_rise(
    heat_of_reaction: f64,
    inlet_concentration: f64,
    conversion: f64,
    density: f64,
    specific_heat: f64,
) -> f64 {
    assert!(
        heat_of_reaction.is_finite(),
        "la chaleur de réaction doit être finie (J/mol)"
    );
    assert!(
        inlet_concentration.is_finite() && inlet_concentration >= 0.0,
        "la concentration d'entrée doit être finie et positive ou nulle (mol/m³)"
    );
    assert!(
        (0.0..=1.0).contains(&conversion),
        "la conversion doit être comprise dans [0, 1] (sans dimension)"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive (kg/m³)"
    );
    assert!(
        specific_heat > 0.0,
        "la capacité thermique massique doit être strictement positive (J/(kg·K))"
    );
    (-heat_of_reaction) * inlet_concentration * conversion / (density * specific_heat)
}

/// Température de sortie adiabatique `T = T_in + ΔT` [K].
///
/// `inlet_temperature` `T_in` température d'entrée [K], `temperature_rise` `ΔT`
/// élévation de température adiabatique [K] (positive si exothermique, négative
/// si endothermique).
///
/// Panique si `inlet_temperature` n'est pas fini ou est négatif (température
/// absolue en kelvin), ou si `temperature_rise` n'est pas fini.
pub fn adiab_adiabatic_temperature(inlet_temperature: f64, temperature_rise: f64) -> f64 {
    assert!(
        inlet_temperature.is_finite() && inlet_temperature >= 0.0,
        "la température d'entrée doit être finie et positive ou nulle (K)"
    );
    assert!(
        temperature_rise.is_finite(),
        "l'élévation de température doit être finie (K)"
    );
    inlet_temperature + temperature_rise
}

/// Élévation de température adiabatique **maximale**, atteinte à conversion
/// totale, `ΔT_max = (−ΔHr)·C_A0 / (ρ·cp)` [K].
///
/// `heat_of_reaction` `ΔHr` chaleur de réaction par mole de réactif limitant
/// [J/mol], `inlet_concentration` `C_A0` concentration d'entrée du réactif
/// limitant [mol/m³], `density` `ρ` masse volumique du mélange [kg/m³],
/// `specific_heat` `cp` capacité thermique massique du mélange [J/(kg·K)].
/// C'est [`adiab_temperature_rise`] évaluée à `X = 1`.
///
/// Panique si `heat_of_reaction` n'est pas fini, si `inlet_concentration` est
/// négatif ou non fini, ou si `density` ou `specific_heat` n'est pas strictement
/// positif (division).
pub fn adiab_maximum_temperature_rise(
    heat_of_reaction: f64,
    inlet_concentration: f64,
    density: f64,
    specific_heat: f64,
) -> f64 {
    assert!(
        heat_of_reaction.is_finite(),
        "la chaleur de réaction doit être finie (J/mol)"
    );
    assert!(
        inlet_concentration.is_finite() && inlet_concentration >= 0.0,
        "la concentration d'entrée doit être finie et positive ou nulle (mol/m³)"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive (kg/m³)"
    );
    assert!(
        specific_heat > 0.0,
        "la capacité thermique massique doit être strictement positive (J/(kg·K))"
    );
    (-heat_of_reaction) * inlet_concentration / (density * specific_heat)
}

/// Conversion déduite de la température par la droite adiabatique
/// `X = (T − T_in) / ΔT_max` [sans dimension] — réciproque linéaire de
/// [`adiab_temperature_rise`].
///
/// `temperature` `T` température [K], `inlet_temperature` `T_in` température
/// d'entrée [K], `maximum_temperature_rise` `ΔT_max` élévation à conversion
/// totale [K]. Pour une exothermique, `ΔT_max > 0` et `X` croît avec `T`.
///
/// Panique si `temperature` ou `inlet_temperature` n'est pas fini, ou si
/// `maximum_temperature_rise` n'est pas fini ou est nul (division).
pub fn adiab_conversion_from_temperature(
    temperature: f64,
    inlet_temperature: f64,
    maximum_temperature_rise: f64,
) -> f64 {
    assert!(
        temperature.is_finite(),
        "la température doit être finie (K)"
    );
    assert!(
        inlet_temperature.is_finite(),
        "la température d'entrée doit être finie (K)"
    );
    assert!(
        maximum_temperature_rise.is_finite() && maximum_temperature_rise != 0.0,
        "l'élévation maximale de température doit être finie et non nulle (K)"
    );
    (temperature - inlet_temperature) / maximum_temperature_rise
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Jeu de propriétés exothermique de référence (fourni par « l'appelant ») :
    // ΔHr = −50 000 J/mol, C_A0 = 1000 mol/m³, ρ = 1000 kg/m³, cp = 4000 J/(kg·K).
    // ΔT_max = 50 000·1000 / (1000·4000) = 5.0e7 / 4.0e6 = 12,5 K.

    #[test]
    fn maximum_temperature_rise_known_case() {
        assert_relative_eq!(
            adiab_maximum_temperature_rise(-50_000.0, 1000.0, 1000.0, 4000.0),
            12.5,
            epsilon = 1e-9
        );
    }

    #[test]
    fn temperature_rise_known_case() {
        // À X = 0,8 : ΔT = 12,5·0,8 = 10 K.
        assert_relative_eq!(
            adiab_temperature_rise(-50_000.0, 1000.0, 0.8, 1000.0, 4000.0),
            10.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn rise_is_fraction_of_maximum() {
        // ΔT(X) = X·ΔT_max : identité entre les deux fonctions à densité/cp fixés.
        let dtmax = adiab_maximum_temperature_rise(-50_000.0, 1000.0, 1000.0, 4000.0);
        let dt = adiab_temperature_rise(-50_000.0, 1000.0, 0.8, 1000.0, 4000.0);
        assert_relative_eq!(dt, 0.8 * dtmax, epsilon = 1e-9);
    }

    #[test]
    fn temperature_and_conversion_are_reciprocal() {
        // X → ΔT → T → X doit boucler exactement (droite adiabatique linéaire).
        let dtmax = adiab_maximum_temperature_rise(-50_000.0, 1000.0, 1000.0, 4000.0);
        let x = 0.8_f64;
        let dt = adiab_temperature_rise(-50_000.0, 1000.0, x, 1000.0, 4000.0);
        let t = adiab_adiabatic_temperature(300.0, dt);
        // T = 300 + 10 = 310 K.
        assert_relative_eq!(t, 310.0, epsilon = 1e-9);
        let x_back = adiab_conversion_from_temperature(t, 300.0, dtmax);
        assert_relative_eq!(x_back, x, epsilon = 1e-9);
    }

    #[test]
    fn zero_conversion_gives_no_rise() {
        // Cas limite : X = 0 → ΔT = 0, donc T = T_in.
        let dt = adiab_temperature_rise(-50_000.0, 1000.0, 0.0, 1000.0, 4000.0);
        assert_relative_eq!(dt, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            adiab_adiabatic_temperature(300.0, dt),
            300.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn endothermic_reaction_cools_the_stream() {
        // ΔHr > 0 (endothermique) → ΔT < 0 : la conversion refroidit le mélange.
        let dt = adiab_temperature_rise(80_000.0, 500.0, 0.5, 1000.0, 4000.0);
        // ΔT = −80 000·500·0,5 / (1000·4000) = −2.0e7 / 4.0e6 = −5,0 K.
        assert_relative_eq!(dt, -5.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "non nulle")]
    fn zero_maximum_rise_panics() {
        // ΔT_max = 0 fait diverger X = (T − T_in)/ΔT_max : entrée rejetée.
        adiab_conversion_from_temperature(310.0, 300.0, 0.0);
    }
}
