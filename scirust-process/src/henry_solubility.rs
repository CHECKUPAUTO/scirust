//! Solubilité des gaz par la **loi de Henry** — pression partielle du gaz
//! dissous, fraction molaire dissoute réciproque, concentration à partir d'une
//! constante de volatilité et variation de la constante de Henry avec la
//! température (forme de van't Hoff).
//!
//! ```text
//! pression partielle (p = H·x)   p = H · x
//! fraction dissoute (réciproque) x = p / H
//! concentration (p = Hc·c)       c = p / Hc
//! van't Hoff (température)        H(T) = H_ref · exp[ (−ΔH_sol / R) · (1/T − 1/T_ref) ]
//! ```
//!
//! `p` pression partielle du gaz au-dessus de la solution [Pa], `H` constante de
//! Henry en convention **pression–fraction molaire** `p = H·x` [Pa, car `x` est
//! sans dimension], `x` fraction molaire du gaz dissous dans le liquide [sans
//! dimension], `Hc` constante de Henry en convention **pression–concentration**
//! `p = Hc·c` [Pa·m³·mol⁻¹], `c` concentration molaire du gaz dissous
//! [mol·m⁻³], `H_ref` constante de Henry à la température de référence [même
//! unité que `H`], `ΔH_sol` enthalpie de dissolution du gaz [J·mol⁻¹], `R`
//! constante des gaz parfaits [J·mol⁻¹·K⁻¹], `T`/`T_ref` température et
//! température de référence [**K**].
//!
//! **Limite honnête** : la loi de Henry n'est valable que pour des **solutions
//! DILUÉES**, **loin de la saturation** ; au-delà (fractions élevées, proximité
//! du point de bulle), il faut un **modèle VLE complet** (activités, fugacités).
//! La **constante de Henry** est **FOURNIE** par l'appelant — issue de tables ou
//! d'essais — et **jamais inventée** ; l'appelant doit veiller à la
//! **CONVENTION** employée (`p = H·x`, `p = Hc·c`, ou encore une forme en
//! molalité) et à ses **unités**, car elles ne sont pas interchangeables. Pour la
//! dépendance en température, l'**enthalpie de dissolution `ΔH_sol` est FOURNIE**
//! et supposée **constante** sur l'intervalle ; les **températures sont en
//! KELVIN**. Aucune propriété physique (enthalpie, constante d'équilibre,
//! coefficient d'activité…) n'est estimée ici.

/// Pression partielle du gaz dissous par la loi de Henry `p = H · x`
/// (convention pression–fraction molaire), en Pa.
///
/// `henry_constant` (H) constante de Henry `p = H·x` [Pa] ;
/// `liquid_mole_fraction` (x) fraction molaire du gaz dissous [sans dimension].
///
/// Panique si `henry_constant <= 0` ou si `liquid_mole_fraction` n'est pas dans
/// `[0, 1]`.
pub fn henry_partial_pressure(henry_constant: f64, liquid_mole_fraction: f64) -> f64 {
    assert!(henry_constant > 0.0, "H > 0 requis");
    assert!(
        (0.0..=1.0).contains(&liquid_mole_fraction),
        "fraction molaire x dans [0, 1] requise"
    );
    henry_constant * liquid_mole_fraction
}

/// Fraction molaire du gaz dissous, réciproque de la loi de Henry `x = p / H`
/// (convention pression–fraction molaire), sans dimension.
///
/// `partial_pressure` (p) pression partielle du gaz [Pa] ; `henry_constant` (H)
/// constante de Henry `p = H·x` [Pa].
///
/// Panique si `partial_pressure < 0` ou si `henry_constant <= 0`.
pub fn henry_dissolved_fraction(partial_pressure: f64, henry_constant: f64) -> f64 {
    assert!(partial_pressure >= 0.0, "p ≥ 0 requis");
    assert!(henry_constant > 0.0, "H > 0 requis");
    partial_pressure / henry_constant
}

/// Concentration molaire du gaz dissous à partir d'une constante de Henry en
/// convention **volatilité** `p = Hc·c`, soit `c = p / Hc`, en mol·m⁻³.
///
/// `partial_pressure` (p) pression partielle du gaz [Pa] ;
/// `henry_volatility_constant` (Hc) constante de Henry `p = Hc·c`
/// [Pa·m³·mol⁻¹].
///
/// Panique si `partial_pressure < 0` ou si `henry_volatility_constant <= 0`.
pub fn henry_concentration_from_pressure(
    partial_pressure: f64,
    henry_volatility_constant: f64,
) -> f64 {
    assert!(partial_pressure >= 0.0, "p ≥ 0 requis");
    assert!(henry_volatility_constant > 0.0, "Hc > 0 requis");
    partial_pressure / henry_volatility_constant
}

/// Variation de la constante de Henry avec la température (forme de **van't
/// Hoff**) `H(T) = H_ref · exp[ (−ΔH_sol / R) · (1/T − 1/T_ref) ]`.
///
/// `henry_ref` (H_ref) constante de Henry à `T_ref` [même unité que le résultat] ;
/// `enthalpy_of_dissolution` (ΔH_sol) enthalpie de dissolution [J·mol⁻¹, signe
/// quelconque] ; `gas_constant` (R) constante des gaz parfaits [J·mol⁻¹·K⁻¹] ;
/// `temperature_ref` (T_ref) et `temperature` (T) températures [**K**].
///
/// Panique si `henry_ref <= 0`, `gas_constant <= 0`, `temperature_ref <= 0` ou
/// `temperature <= 0`.
pub fn henry_temperature_dependence(
    henry_ref: f64,
    enthalpy_of_dissolution: f64,
    gas_constant: f64,
    temperature_ref: f64,
    temperature: f64,
) -> f64 {
    assert!(henry_ref > 0.0, "H_ref > 0 requis");
    assert!(gas_constant > 0.0, "R > 0 requis");
    assert!(
        temperature_ref > 0.0 && temperature > 0.0,
        "températures T et T_ref > 0 K requises"
    );
    let exponent =
        (-enthalpy_of_dissolution / gas_constant) * (1.0 / temperature - 1.0 / temperature_ref);
    henry_ref * exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn partial_pressure_and_reciprocity() {
        // H = 2000 Pa, x = 0.01 ⇒ p = 2000·0.01 = 20 Pa.
        let p = henry_partial_pressure(2000.0_f64, 0.01_f64);
        assert_relative_eq!(p, 20.0, max_relative = 1e-12);
        // Réciprocité stricte : x = p/H redonne la fraction de départ.
        assert_relative_eq!(
            henry_dissolved_fraction(p, 2000.0_f64),
            0.01,
            max_relative = 1e-12
        );
    }

    #[test]
    fn zero_fraction_gives_zero_pressure() {
        // Aucune espèce dissoute (x = 0) ⇒ pression partielle nulle.
        assert_relative_eq!(
            henry_partial_pressure(1.5e5_f64, 0.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn concentration_is_proportional_to_pressure() {
        // c = p/Hc : doubler p double c pour Hc constant.
        let hc = 3.0_f64; // Pa·m³·mol⁻¹
        let c1 = henry_concentration_from_pressure(30.0_f64, hc);
        let c2 = henry_concentration_from_pressure(60.0_f64, hc);
        assert_relative_eq!(c1, 10.0, max_relative = 1e-12);
        assert_relative_eq!(c2, 2.0 * c1, max_relative = 1e-12);
    }

    #[test]
    fn temperature_dependence_identity_at_reference() {
        // À T = T_ref l'exposant est nul : H(T_ref) = H_ref quel que soit ΔH_sol.
        assert_relative_eq!(
            henry_temperature_dependence(
                5000.0_f64,
                -25000.0_f64,
                8.314_f64,
                298.15_f64,
                298.15_f64
            ),
            5000.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn temperature_dependence_realistic_case() {
        // H_ref = 3000 Pa à T_ref = 298.15 K, dissolution exothermique
        // ΔH_sol = −20000 J·mol⁻¹, R = 8.314 J·mol⁻¹·K⁻¹, T = 308.15 K.
        // exposant = (20000/8.314)·(1/308.15 − 1/298.15)
        //          = 2405.581·(−1.088436e−4) = −0.261834
        // H(T) = 3000·exp(−0.261834) = 3000·0.769643 ≈ 2308.9 Pa.
        assert_relative_eq!(
            henry_temperature_dependence(
                3000.0_f64,
                -20000.0_f64,
                8.314_f64,
                298.15_f64,
                308.15_f64
            ),
            2308.9,
            max_relative = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "H > 0 requis")]
    fn partial_pressure_panics_on_nonpositive_constant() {
        // Une constante de Henry nulle ou négative n'a pas de sens physique.
        let _ = henry_partial_pressure(0.0_f64, 0.01_f64);
    }
}
