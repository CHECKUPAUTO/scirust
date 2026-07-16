//! **Thermique de dissipateur** — module de dissipation thermique d'un
//! composant semi-connecteur (diode, transistor, IGBT…) en **régime permanent**,
//! par analogie électrique : les résistances thermiques en série s'ajoutent comme
//! des résistances électriques, la puissance dissipée joue le rôle du courant et
//! l'écart de température celui de la tension.
//!
//! ```text
//! température de jonction     Tj = Ta + P · Rth,ja
//! résistance thermique série  Rth,ja = Rth,jc + Rth,cs + Rth,sa
//! puissance max dissipable    P_max = (Tj,max − Ta) / Rth,ja
//! résistance dissipateur req. Rth,sa = (Tj,max − Ta) / P − Rth,jc − Rth,cs
//! ```
//!
//! `Tj` température de jonction (°C), `Ta` température ambiante (°C), `P`
//! puissance dissipée (W), `Rth,ja` résistance thermique jonction-ambiant
//! (K/W ≡ °C/W), `Rth,jc` résistance thermique jonction-boîtier (K/W),
//! `Rth,cs` résistance thermique boîtier-dissipateur (K/W, interface), `Rth,sa`
//! résistance thermique dissipateur-ambiant (K/W), `Tj,max` température de
//! jonction maximale admissible (°C).
//!
//! **Convention** : SI ; températures en °C (les écarts sont identiques en K),
//! résistances thermiques en K/W, puissances en W. **Limite honnête** : régime
//! **permanent** thermique uniquement ; les résistances thermiques
//! (jonction-boîtier, boîtier-dissipateur, dissipateur-ambiant) sont **fournies
//! par l'appelant** (fiches techniques du composant, de la graisse/pad
//! d'interface et du dissipateur) — aucune valeur « typique » n'est inventée.
//! Le modèle est l'**analogie électrique série** des résistances thermiques ; la
//! température de jonction ne doit pas dépasser `Tj,max`. Ce module ne modélise
//! **pas le transitoire** (impédance thermique `Zth(t)`, capacités thermiques,
//! pointes de puissance) ni le couplage entre composants voisins.

/// Température de jonction en régime permanent
/// `Tj = ambient_temperature + power_dissipation · total_thermal_resistance`
/// (°C), montée thermique au-dessus de l'ambiant proportionnelle à la puissance
/// dissipée et à la résistance thermique jonction-ambiant.
///
/// Panique si `power_dissipation < 0` (puissance dissipée non physique) ou si
/// `total_thermal_resistance < 0` (résistance thermique non physique).
pub fn heatsink_junction_temperature(
    ambient_temperature: f64,
    power_dissipation: f64,
    total_thermal_resistance: f64,
) -> f64 {
    assert!(
        power_dissipation >= 0.0,
        "la puissance dissipée power_dissipation doit être ≥ 0"
    );
    assert!(
        total_thermal_resistance >= 0.0,
        "la résistance thermique total_thermal_resistance doit être ≥ 0"
    );
    ambient_temperature + power_dissipation * total_thermal_resistance
}

/// Résistance thermique jonction-ambiant totale
/// `Rth,ja = junction_case + case_sink + sink_ambient` (K/W), somme des trois
/// résistances thermiques en série (analogie électrique).
///
/// Panique si l'une des résistances `junction_case`, `case_sink` ou
/// `sink_ambient` est `< 0` (résistance thermique non physique).
pub fn heatsink_total_thermal_resistance(
    junction_case: f64,
    case_sink: f64,
    sink_ambient: f64,
) -> f64 {
    assert!(
        junction_case >= 0.0,
        "la résistance thermique jonction-boîtier junction_case doit être ≥ 0"
    );
    assert!(
        case_sink >= 0.0,
        "la résistance thermique boîtier-dissipateur case_sink doit être ≥ 0"
    );
    assert!(
        sink_ambient >= 0.0,
        "la résistance thermique dissipateur-ambiant sink_ambient doit être ≥ 0"
    );
    junction_case + case_sink + sink_ambient
}

/// Puissance maximale dissipable en régime permanent
/// `P_max = (max_junction_temperature − ambient_temperature) /
/// total_thermal_resistance` (W), déduite du budget thermique disponible entre
/// la jonction maximale admissible et l'ambiant.
///
/// Panique si `total_thermal_resistance <= 0` (division par zéro ou résistance
/// non physique) ou si `max_junction_temperature <= ambient_temperature`
/// (aucun budget thermique : la jonction est déjà à l'ambiant ou au-dessus).
pub fn heatsink_max_power_dissipation(
    max_junction_temperature: f64,
    ambient_temperature: f64,
    total_thermal_resistance: f64,
) -> f64 {
    assert!(
        total_thermal_resistance > 0.0,
        "la résistance thermique total_thermal_resistance doit être strictement positive"
    );
    assert!(
        max_junction_temperature > ambient_temperature,
        "la jonction maximale max_junction_temperature doit dépasser l'ambiant ambient_temperature"
    );
    (max_junction_temperature - ambient_temperature) / total_thermal_resistance
}

/// Résistance thermique de dissipateur requise
/// `Rth,sa = (max_junction_temperature − ambient_temperature) /
/// power_dissipation − junction_case − case_sink` (K/W), valeur maximale du
/// dissipateur-ambiant pour tenir la jonction sous `Tj,max` à la puissance
/// dissipée donnée.
///
/// Panique si `power_dissipation <= 0` (division par zéro ou puissance non
/// physique), si `max_junction_temperature <= ambient_temperature` (aucun budget
/// thermique) ou si `junction_case < 0` ou `case_sink < 0` (résistances
/// thermiques non physiques). Une valeur de retour négative signale que le budget
/// thermique est déjà épuisé par les résistances amont : aucun dissipateur ne
/// suffit à la puissance demandée.
pub fn heatsink_required_sink_resistance(
    max_junction_temperature: f64,
    ambient_temperature: f64,
    power_dissipation: f64,
    junction_case: f64,
    case_sink: f64,
) -> f64 {
    assert!(
        power_dissipation > 0.0,
        "la puissance dissipée power_dissipation doit être strictement positive"
    );
    assert!(
        max_junction_temperature > ambient_temperature,
        "la jonction maximale max_junction_temperature doit dépasser l'ambiant ambient_temperature"
    );
    assert!(
        junction_case >= 0.0,
        "la résistance thermique jonction-boîtier junction_case doit être ≥ 0"
    );
    assert!(
        case_sink >= 0.0,
        "la résistance thermique boîtier-dissipateur case_sink doit être ≥ 0"
    );
    (max_junction_temperature - ambient_temperature) / power_dissipation - junction_case - case_sink
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn junction_temperature_numeric() {
        // Cas chiffré : Ta = 25 °C, P = 40 W, Rth,ja = 1,5 K/W.
        //   Tj = 25 + 40 · 1,5 = 25 + 60 = 85 °C.
        assert_relative_eq!(
            heatsink_junction_temperature(25.0, 40.0, 1.5),
            85.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn total_resistance_is_series_sum() {
        // Somme série des trois résistances : 0,5 + 0,2 + 1,8 = 2,5 K/W.
        let rth = heatsink_total_thermal_resistance(0.5, 0.2, 1.8);
        assert_relative_eq!(rth, 2.5, epsilon = 1e-12);
        // Une résistance nulle (contact parfait) ne change rien à l'ordre de la
        // somme : additivité vérifiée.
        assert_relative_eq!(
            heatsink_total_thermal_resistance(0.5, 0.0, 1.8),
            0.5 + 1.8,
            epsilon = 1e-12
        );
    }

    #[test]
    fn max_power_and_junction_temperature_are_reciprocal() {
        // Réciprocité : à P_max, la jonction atteint exactement Tj,max.
        //   Tj,max = 150 °C, Ta = 25 °C, Rth,ja = 2,5 K/W.
        //   P_max = (150 − 25) / 2,5 = 125 / 2,5 = 50 W.
        let p_max = heatsink_max_power_dissipation(150.0, 25.0, 2.5);
        assert_relative_eq!(p_max, 50.0, epsilon = 1e-9);
        // Réinjectée, cette puissance ramène bien la jonction à 150 °C.
        let tj = heatsink_junction_temperature(25.0, p_max, 2.5);
        assert_relative_eq!(tj, 150.0, epsilon = 1e-9);
    }

    #[test]
    fn required_sink_resistance_matches_series_budget() {
        // Cas chiffré : Tj,max = 150 °C, Ta = 25 °C, P = 50 W, Rth,jc = 0,5,
        //   Rth,cs = 0,2 K/W.
        //   Rth,sa = (150 − 25) / 50 − 0,5 − 0,2 = 2,5 − 0,7 = 1,8 K/W.
        let rsa = heatsink_required_sink_resistance(150.0, 25.0, 50.0, 0.5, 0.2);
        assert_relative_eq!(rsa, 1.8, epsilon = 1e-9);
        // Cohérence : le total série reconstitué (jc + cs + sa) redonne la
        // résistance jonction-ambiant qui, à 50 W, atteint tout juste Tj,max.
        let rth_ja = heatsink_total_thermal_resistance(0.5, 0.2, rsa);
        assert_relative_eq!(rth_ja, 2.5, epsilon = 1e-9);
    }

    #[test]
    fn max_power_scales_inversely_with_resistance() {
        // Proportionnalité inverse : à budget thermique fixé, halver la
        // résistance thermique double la puissance dissipable.
        let p1 = heatsink_max_power_dissipation(150.0, 30.0, 2.4);
        let p2 = heatsink_max_power_dissipation(150.0, 30.0, 1.2);
        assert_relative_eq!(p1, 50.0, epsilon = 1e-9);
        assert_relative_eq!(p2, 2.0 * p1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(
        expected = "la résistance thermique total_thermal_resistance doit être strictement positive"
    )]
    fn zero_resistance_max_power_panics() {
        heatsink_max_power_dissipation(150.0, 25.0, 0.0);
    }
}
