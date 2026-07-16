//! Pression de vapeur d'un corps pur — équation d'Antoine (avec sa réciproque en
//! température de saturation) et relation de Clausius-Clapeyron intégrée (avec
//! l'estimation de la chaleur latente à partir de deux points de la courbe).
//!
//! ```text
//! Antoine                     log₁₀ P = A − B / (C + T)
//!   pression                  P    = 10^(A − B / (C + T))
//!   température de saturation T_sat = B / (A − log₁₀ P) − C
//! Clausius-Clapeyron intégrée P    = P_ref · exp[ (L / R) · (1/T_ref − 1/T) ]
//!   chaleur latente (2 points) L   = R · ln(P₂/P₁) / (1/T₁ − 1/T₂)
//! ```
//!
//! `A`, `B`, `C` coefficients d'Antoine [unités **FOURNIES** par la table :
//! l'échelle de `P` et l'unité de `T` (souvent °C, parfois K) sont celles des
//! coefficients], `T` température [dans l'unité des coefficients d'Antoine],
//! `P` pression de vapeur saturante [dans l'échelle des coefficients], `T_sat`
//! température de saturation réciproque [même unité que `T`] ; pour
//! Clausius-Clapeyron `P_ref` pression de référence [Pa], `T_ref` température de
//! référence [K], `T` température [K], `L` chaleur (enthalpie) latente de
//! vaporisation [J·mol⁻¹], `R` constante des gaz parfaits [J·mol⁻¹·K⁻¹],
//! `P₁`/`P₂` pressions aux températures `T₁`/`T₂` [Pa et K]. Les grandeurs sans
//! dimension le restent ; toutes les grandeurs de Clausius-Clapeyron sont en
//! unités SI cohérentes.
//!
//! **Limite honnête** : le corps est **pur** (une seule espèce ; aucune
//! correction de mélange, d'activité ou de fugacité). Les **coefficients
//! d'Antoine** — avec leurs **UNITÉS spécifiques** — et la **chaleur latente**
//! sont **FOURNIS** par les tables (Antoine, DIPPR, tables thermodynamiques) et
//! ne sont **jamais** inventés ni ajustés ici. La forme intégrée de
//! Clausius-Clapeyron suppose une **chaleur latente constante** sur l'intervalle
//! et une **vapeur assimilée à un gaz parfait** de volume molaire très supérieur
//! à celui du liquide ; elle n'est qu'une approximation hors d'un domaine étroit.
//! Aucune propriété physique (enthalpies, volumes molaires, facteurs de
//! compressibilité, coefficients d'activité…) n'est calculée : elles restent à la
//! charge de l'appelant.

/// Pression de vapeur saturante par l'équation d'Antoine
/// `P = 10^(A − B / (C + T))` (équivalent à `log₁₀ P = A − B / (C + T)`).
///
/// `coefficient_a` (A), `coefficient_b` (B), `coefficient_c` (C) coefficients
/// d'Antoine **FOURNIS** par la table, `temperature` (T) température **dans
/// l'unité de ces coefficients** (souvent °C). Le résultat est dans l'échelle de
/// pression propre aux coefficients (mmHg, bar, kPa… selon la table).
///
/// Panique si `coefficient_b < 0` ou si `C + T <= 0` (dénominateur non
/// strictement positif, hors du domaine physique de l'équation).
pub fn vp_antoine(
    coefficient_a: f64,
    coefficient_b: f64,
    coefficient_c: f64,
    temperature: f64,
) -> f64 {
    assert!(coefficient_b >= 0.0, "B ≥ 0 requis (coefficient d'Antoine)");
    assert!(
        coefficient_c + temperature > 0.0,
        "C + T > 0 requis (dénominateur d'Antoine strictement positif)"
    );
    10.0_f64.powf(coefficient_a - coefficient_b / (coefficient_c + temperature))
}

/// Température de saturation par la réciproque de l'équation d'Antoine
/// `T_sat = B / (A − log₁₀ P) − C`.
///
/// `coefficient_a` (A), `coefficient_b` (B), `coefficient_c` (C) coefficients
/// d'Antoine **FOURNIS**, `pressure` (P) pression de vapeur imposée **dans
/// l'échelle de pression de ces coefficients**. La température rendue est dans
/// l'unité des coefficients (même convention que [`vp_antoine`]).
///
/// Panique si `pressure <= 0` (logarithme indéfini) ou si `A − log₁₀ P <= 0`
/// (dénominateur non strictement positif, pression hors du domaine d'Antoine).
pub fn vp_antoine_temperature(
    coefficient_a: f64,
    coefficient_b: f64,
    coefficient_c: f64,
    pressure: f64,
) -> f64 {
    assert!(pressure > 0.0, "P > 0 requis (logarithme de la pression)");
    let denominator = coefficient_a - pressure.log10();
    assert!(
        denominator > 0.0,
        "A − log₁₀ P > 0 requis (pression dans le domaine d'Antoine)"
    );
    coefficient_b / denominator - coefficient_c
}

/// Pression de vapeur par la relation de Clausius-Clapeyron intégrée
/// `P = P_ref · exp[ (L / R) · (1/T_ref − 1/T) ]` (chaleur latente constante,
/// vapeur assimilée à un gaz parfait).
///
/// `reference_pressure` (P_ref) pression de référence [Pa], `latent_heat` (L)
/// chaleur latente de vaporisation **FOURNIE** [J·mol⁻¹], `gas_constant` (R)
/// constante des gaz parfaits [J·mol⁻¹·K⁻¹], `reference_temperature` (T_ref) et
/// `temperature` (T) températures [K]. Résultat en pascals.
///
/// Panique si `reference_pressure <= 0`, si `latent_heat < 0`, si
/// `gas_constant <= 0`, si `reference_temperature <= 0` ou si `temperature <= 0`.
pub fn vp_clausius_clapeyron(
    reference_pressure: f64,
    latent_heat: f64,
    gas_constant: f64,
    reference_temperature: f64,
    temperature: f64,
) -> f64 {
    assert!(
        reference_pressure > 0.0,
        "P_ref > 0 requis (pression de référence)"
    );
    assert!(latent_heat >= 0.0, "L ≥ 0 requis (chaleur latente)");
    assert!(
        gas_constant > 0.0,
        "R > 0 requis (constante des gaz parfaits)"
    );
    assert!(
        reference_temperature > 0.0,
        "T_ref > 0 K requis (température de référence)"
    );
    assert!(temperature > 0.0, "T > 0 K requis (température)");
    reference_pressure
        * ((latent_heat / gas_constant) * (1.0 / reference_temperature - 1.0 / temperature)).exp()
}

/// Chaleur latente de vaporisation estimée à partir de deux points de la courbe
/// de vapeur `L = R · ln(P₂/P₁) / (1/T₁ − 1/T₂)` (Clausius-Clapeyron intégrée
/// inversée, chaleur latente supposée constante entre les deux points).
///
/// `pressure1` (P₁) et `pressure2` (P₂) pressions de vapeur [Pa] aux températures
/// `temperature1` (T₁) et `temperature2` (T₂) [K], `gas_constant` (R) constante
/// des gaz parfaits [J·mol⁻¹·K⁻¹]. Résultat en J·mol⁻¹.
///
/// Panique si `pressure1 <= 0`, si `pressure2 <= 0`, si `temperature1 <= 0`, si
/// `temperature2 <= 0`, si `gas_constant <= 0` ou si `temperature1 == temperature2`
/// (dénominateur nul, les deux points doivent différer).
pub fn vp_clausius_latent_heat(
    pressure1: f64,
    pressure2: f64,
    temperature1: f64,
    temperature2: f64,
    gas_constant: f64,
) -> f64 {
    assert!(pressure1 > 0.0, "P₁ > 0 requis (pression au point 1)");
    assert!(pressure2 > 0.0, "P₂ > 0 requis (pression au point 2)");
    assert!(
        temperature1 > 0.0,
        "T₁ > 0 K requis (température au point 1)"
    );
    assert!(
        temperature2 > 0.0,
        "T₂ > 0 K requis (température au point 2)"
    );
    assert!(
        gas_constant > 0.0,
        "R > 0 requis (constante des gaz parfaits)"
    );
    assert!(
        temperature1 != temperature2,
        "T₁ ≠ T₂ requis (les deux points doivent différer)"
    );
    gas_constant * (pressure2 / pressure1).ln() / (1.0 / temperature1 - 1.0 / temperature2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn antoine_water_at_normal_boiling_point() {
        // Coefficients d'Antoine de l'eau (P en mmHg, T en °C) :
        // A = 8.07131, B = 1730.63, C = 233.426. À T = 100 °C :
        //   log₁₀ P = 8.07131 − 1730.63/(233.426 + 100)
        //           = 8.07131 − 1730.63/333.426
        //           = 8.07131 − 5.190517… = 2.880793…
        //   P = 10^2.880793… ≈ 759.9 mmHg  (≈ 1 atm, attendu à l'ébullition).
        let p = vp_antoine(8.07131_f64, 1730.63_f64, 233.426_f64, 100.0_f64);
        assert_relative_eq!(p, 759.9, max_relative = 1e-3);
    }

    #[test]
    fn antoine_temperature_is_inverse_of_pressure() {
        // Réciprocité : T → P (Antoine) → T (Antoine inverse) doit boucler.
        let (a, b, c) = (8.07131_f64, 1730.63_f64, 233.426_f64);
        let t = 60.0_f64;
        let p = vp_antoine(a, b, c, t);
        let t_back = vp_antoine_temperature(a, b, c, p);
        assert_relative_eq!(t_back, t, max_relative = 1e-9);
    }

    #[test]
    fn clausius_returns_reference_at_reference_temperature() {
        // À T = T_ref l'exponentielle vaut exp(0) = 1 ⇒ P = P_ref exactement.
        let p = vp_clausius_clapeyron(
            101_325.0_f64,
            40_660.0_f64,
            8.314_f64,
            373.15_f64,
            373.15_f64,
        );
        assert_relative_eq!(p, 101_325.0, max_relative = 1e-12);
    }

    #[test]
    fn clausius_water_pressure_drops_below_reference() {
        // Eau : P_ref = 101325 Pa à T_ref = 373.15 K, L = 40660 J/mol, R = 8.314.
        // À T = 353.15 K (80 °C) :
        //   L/R = 40660/8.314 = 4890.546…
        //   1/373.15 − 1/353.15 = 0.00267989… − 0.00283166… = −1.51770e-4
        //   exposant = 4890.546 · (−1.51770e-4) = −0.742339…
        //   P = 101325 · exp(−0.742339) = 101325 · 0.476022… ≈ 48233 Pa.
        let p = vp_clausius_clapeyron(
            101_325.0_f64,
            40_660.0_f64,
            8.314_f64,
            373.15_f64,
            353.15_f64,
        );
        assert_relative_eq!(p, 48_233.0, max_relative = 1e-3);
        // La pression saturante décroît quand la température baisse.
        assert!(p < 101_325.0);
    }

    #[test]
    fn latent_heat_recovers_clausius_input() {
        // Réciprocité : construire P₂ par Clausius-Clapeyron avec une L connue,
        // puis retrouver exactement cette L à partir des deux points (P₁,T₁),
        // (P₂,T₂).
        let (p1, l, r, t1, t2) = (
            101_325.0_f64,
            40_660.0_f64,
            8.314_f64,
            373.15_f64,
            353.15_f64,
        );
        let p2 = vp_clausius_clapeyron(p1, l, r, t1, t2);
        let l_back = vp_clausius_latent_heat(p1, p2, t1, t2, r);
        assert_relative_eq!(l_back, l, max_relative = 1e-9);
    }

    #[test]
    fn latent_heat_positive_when_pressure_rises_with_temperature() {
        // P monte avec T (P₂ > P₁ pour T₂ > T₁) ⇒ chaleur latente positive.
        let l = vp_clausius_latent_heat(
            70_000.0_f64,
            101_325.0_f64,
            363.15_f64,
            373.15_f64,
            8.314_f64,
        );
        assert!(l > 0.0);
    }

    #[test]
    #[should_panic(expected = "T₁ ≠ T₂ requis")]
    fn latent_heat_panics_on_equal_temperatures() {
        // Deux points à la même température ⇒ dénominateur nul ⇒ panique.
        let _ = vp_clausius_latent_heat(
            101_325.0_f64,
            90_000.0_f64,
            373.15_f64,
            373.15_f64,
            8.314_f64,
        );
    }
}
