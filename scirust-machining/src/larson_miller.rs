//! Fluage — **paramètre de Larson-Miller** pour l'extrapolation de la durée de
//! vie en rupture (relation temps–température des données de fluage).
//!
//! ```text
//! paramètre        P  = T·(C + log10 t_r)          (T en K, t_r en h)
//! temps à rupture  t_r = 10^{P/T − C}              (réciproque, en h)
//! température      T  = P/(C + log10 t_r)          (K, pour une vie visée)
//! ```
//!
//! `T` température **absolue** (K), `t_r` temps à rupture (h), `C` constante du
//! matériau (sans dimension, ≈ 20 pour de nombreux aciers), `P` paramètre de
//! Larson-Miller (K, car `C + log10 t_r` est sans dimension).
//!
//! **Convention** : température en **kelvin**, temps en **heures**, logarithme
//! décimal. **Limite honnête** : la constante matériau `C` et le paramètre `P`
//! (issu d'une courbe maîtresse) sont des **données fournies par l'appelant** ;
//! aucune valeur « par défaut » n'est inventée. Le modèle est une **corrélation
//! empirique** d'extrapolation des essais de fluage : toute prédiction en dehors
//! de la plage mesurée doit être **validée expérimentalement**.

/// Paramètre de Larson-Miller `P = T·(C + log10 t_r)` (en K).
///
/// `temperature_kelvin` en K, `rupture_time_hours` en h, `material_constant`
/// (`C`) sans dimension et fourni par l'appelant.
///
/// Panique si `temperature_kelvin <= 0` ou `rupture_time_hours <= 0`.
pub fn larson_miller_parameter_value(
    temperature_kelvin: f64,
    rupture_time_hours: f64,
    material_constant: f64,
) -> f64 {
    assert!(
        temperature_kelvin > 0.0,
        "la température (K) doit être strictement positive"
    );
    assert!(
        rupture_time_hours > 0.0,
        "le temps à rupture (h) doit être strictement positif"
    );
    temperature_kelvin * (material_constant + rupture_time_hours.log10())
}

/// Temps à rupture déduit du paramètre `t_r = 10^{P/T − C}` (réciproque, en h).
///
/// `parameter` (`P`) en K, `temperature_kelvin` en K, `material_constant` (`C`)
/// sans dimension.
///
/// Panique si `temperature_kelvin <= 0`.
pub fn larson_miller_rupture_time(
    parameter: f64,
    temperature_kelvin: f64,
    material_constant: f64,
) -> f64 {
    assert!(
        temperature_kelvin > 0.0,
        "la température (K) doit être strictement positive"
    );
    10.0_f64.powf(parameter / temperature_kelvin - material_constant)
}

/// Température permettant une vie visée `T = P/(C + log10 t_r)` (en K).
///
/// `parameter` (`P`) en K, `rupture_time_hours` en h, `material_constant` (`C`)
/// sans dimension.
///
/// Panique si `rupture_time_hours <= 0` ou si `C + log10 t_r == 0` (température
/// non définie).
pub fn larson_miller_temperature_for_life(
    parameter: f64,
    rupture_time_hours: f64,
    material_constant: f64,
) -> f64 {
    assert!(
        rupture_time_hours > 0.0,
        "le temps à rupture (h) doit être strictement positif"
    );
    let denominator = material_constant + rupture_time_hours.log10();
    assert!(
        denominator != 0.0,
        "C + log10 t_r doit être non nul pour définir la température"
    );
    parameter / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn parameter_and_rupture_time_round_trip() {
        // T=800 K, t_r=1000 h, C=20 → P = 800·(20+3) = 18400 K.
        let p = larson_miller_parameter_value(800.0, 1000.0, 20.0);
        assert_relative_eq!(p, 800.0 * 23.0, epsilon = 1e-9);
        // L'inverse doit redonner exactement 1000 h.
        assert_relative_eq!(
            larson_miller_rupture_time(p, 800.0, 20.0),
            1000.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn temperature_for_life_is_inverse_of_parameter() {
        // Avec P=18400, t_r=1000 h, C=20 : T = 18400/(20+3) = 800 K.
        let p = larson_miller_parameter_value(800.0, 1000.0, 20.0);
        assert_relative_eq!(
            larson_miller_temperature_for_life(p, 1000.0, 20.0),
            800.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn parameter_at_one_hour_reduces_to_temperature_times_c() {
        // t_r = 1 h → log10(1) = 0 → P = T·C (cas limite).
        assert_relative_eq!(
            larson_miller_parameter_value(750.0, 1.0, 20.0),
            750.0 * 20.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn parameter_is_linear_in_temperature_at_fixed_time() {
        // P = T·(C + log10 t_r) : à t_r et C fixes, P est proportionnel à T.
        let p1 = larson_miller_parameter_value(800.0, 1000.0, 20.0);
        let p2 = larson_miller_parameter_value(1600.0, 1000.0, 20.0);
        assert_relative_eq!(p2 / p1, 2.0, max_relative = 1e-9);
    }

    #[test]
    fn realistic_creep_extrapolation() {
        // Cas chiffré : T=811 K (~1000 °F), t_r=100000 h, C=20.
        // log10(100000) = 5 → P = 811·(20+5) = 811·25 = 20275 K.
        let p = larson_miller_parameter_value(811.0, 100000.0, 20.0);
        assert_relative_eq!(p, 20275.0, epsilon = 1e-6);
        // Réciproque : à P et C donnés, une vie de 100000 h correspond bien à 811 K.
        assert_relative_eq!(
            larson_miller_temperature_for_life(p, 100000.0, 20.0),
            811.0,
            max_relative = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "temps à rupture")]
    fn zero_rupture_time_panics() {
        larson_miller_parameter_value(800.0, 0.0, 20.0);
    }
}
