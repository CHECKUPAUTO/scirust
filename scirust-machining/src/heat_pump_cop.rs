//! Coefficient de performance (COP) d'une pompe à chaleur — bilan énergétique
//! en régime permanent pour les modes chauffage et froid, borne de Carnot et
//! rendement de 2e principe.
//!
//! ```text
//! COP chauffage      COP_c = Q_chaud / W                 (sans dimension)
//! COP froid (EER)    COP_f = Q_froid / W                 (sans dimension)
//! borne de Carnot    COP_carnot = T_chaud / (T_chaud − T_froid)
//! rendement 2e princ η = COP_reel / COP_carnot           (sans dimension)
//! relation machine   COP_chaud = COP_froid + 1
//! ```
//!
//! `Q_chaud` chaleur délivrée au puits chaud (J ou W) ; `Q_froid` chaleur
//! absorbée à la source froide (J ou W) ; `W` travail (ou puissance) fourni au
//! compresseur, même unité que les chaleurs (J ou W) ; `T_chaud`, `T_froid`
//! températures absolues du puits et de la source (K) ; `η` rendement de 2e
//! principe (sans dimension).
//!
//! **Limite honnête** : bilan énergétique d'une machine en **régime
//! permanent** (`Q_chaud = Q_froid + W`). Les **chaleurs et le travail**, ou
//! les **températures de source/puits en KELVIN** pour la borne de Carnot,
//! sont **fournis par l'appelant** ; aucune valeur de fluide, de procédé ou de
//! matériau n'est supposée par défaut. Le COP réel est toujours **inférieur**
//! au COP de Carnot (le rendement de 2e principe fourni/calculé quantifie
//! l'écart). Distinct de [`crate::refrigeration_cycle`], qui raisonne sur les
//! enthalpies des points du cycle à compression de vapeur.

/// COP en mode chauffage `COP_c = Q_chaud / W` (sans dimension).
///
/// `heat_delivered` = Q_chaud chaleur délivrée au puits chaud (J ou W),
/// `work_input` = W travail fourni au compresseur (même unité).
///
/// Panique si `heat_delivered < 0` ou `work_input <= 0`.
pub fn heatpump_cop_heating(heat_delivered: f64, work_input: f64) -> f64 {
    assert!(
        heat_delivered >= 0.0,
        "la chaleur délivrée Q_chaud doit être positive ou nulle"
    );
    assert!(
        work_input > 0.0,
        "le travail fourni W doit être strictement positif"
    );
    heat_delivered / work_input
}

/// COP en mode froid (EER) `COP_f = Q_froid / W` (sans dimension).
///
/// `heat_absorbed` = Q_froid chaleur absorbée à la source froide (J ou W),
/// `work_input` = W travail fourni au compresseur (même unité).
///
/// Panique si `heat_absorbed < 0` ou `work_input <= 0`.
pub fn heatpump_cop_cooling(heat_absorbed: f64, work_input: f64) -> f64 {
    assert!(
        heat_absorbed >= 0.0,
        "la chaleur absorbée Q_froid doit être positive ou nulle"
    );
    assert!(
        work_input > 0.0,
        "le travail fourni W doit être strictement positif"
    );
    heat_absorbed / work_input
}

/// Borne de Carnot en mode chauffage `COP_carnot = T_chaud / (T_chaud − T_froid)`.
///
/// `hot_temperature_kelvin` = T_chaud température du puits chaud (K),
/// `cold_temperature_kelvin` = T_froid température de la source froide (K).
///
/// Panique si l'une des températures est `<= 0` (Kelvin) ou si
/// `hot_temperature_kelvin <= cold_temperature_kelvin`.
pub fn heatpump_carnot_cop_heating(
    hot_temperature_kelvin: f64,
    cold_temperature_kelvin: f64,
) -> f64 {
    assert!(
        cold_temperature_kelvin > 0.0,
        "la température froide doit être strictement positive (Kelvin)"
    );
    assert!(
        hot_temperature_kelvin > cold_temperature_kelvin,
        "la température chaude doit dépasser la température froide"
    );
    hot_temperature_kelvin / (hot_temperature_kelvin - cold_temperature_kelvin)
}

/// Rendement de 2e principe `η = COP_reel / COP_carnot` (sans dimension).
///
/// `actual_cop` = COP réel mesuré/estimé, `carnot_cop` = borne de Carnot
/// correspondante (même mode). Vaut au plus 1 pour une machine physique.
///
/// Panique si `actual_cop < 0` ou `carnot_cop <= 0`.
pub fn heatpump_second_law_efficiency(actual_cop: f64, carnot_cop: f64) -> f64 {
    assert!(actual_cop >= 0.0, "le COP réel doit être positif ou nul");
    assert!(
        carnot_cop > 0.0,
        "le COP de Carnot doit être strictement positif"
    );
    actual_cop / carnot_cop
}

/// Identité `COP_chaud = COP_froid + 1` pour une même machine (sans dimension).
///
/// `cop_cooling` = COP froid de la machine ; le résultat est le COP chauffage
/// correspondant, conséquence de `Q_chaud = Q_froid + W`.
///
/// Panique si `cop_cooling < 0`.
pub fn heatpump_cop_relation(cop_cooling: f64) -> f64 {
    assert!(cop_cooling >= 0.0, "le COP froid doit être positif ou nul");
    cop_cooling + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Machine en régime permanent : Q_chaud = Q_froid + W.
    // 5000 = 3750 + 1250 (W). COP_chaud = 5000/1250 = 4, COP_froid = 3750/1250 = 3.
    #[test]
    fn heating_and_cooling_cop_on_steady_balance() {
        let (q_hot, q_cold, w) = (5000.0_f64, 3750.0_f64, 1250.0_f64);
        assert_relative_eq!(heatpump_cop_heating(q_hot, w), 4.0, epsilon = 1e-12);
        assert_relative_eq!(heatpump_cop_cooling(q_cold, w), 3.0, epsilon = 1e-12);
    }

    // L'identité COP_chaud = COP_froid + 1 doit tomber sur le COP chauffage direct.
    #[test]
    fn cop_relation_matches_heating_cop() {
        let (q_hot, q_cold, w) = (5000.0_f64, 3750.0_f64, 1250.0_f64);
        let cop_cold = heatpump_cop_cooling(q_cold, w);
        assert_relative_eq!(
            heatpump_cop_relation(cop_cold),
            heatpump_cop_heating(q_hot, w),
            epsilon = 1e-12
        );
    }

    // Carnot chauffage : T_chaud = 300 K, T_froid = 270 K → 300/30 = 10.
    #[test]
    fn carnot_heating_bound_reference_case() {
        assert_relative_eq!(
            heatpump_carnot_cop_heating(300.0, 270.0),
            10.0,
            epsilon = 1e-12
        );
    }

    // Rendement de 2e principe : COP réel 4 face à Carnot 10 → 0,4.
    #[test]
    fn second_law_efficiency_ratio() {
        let carnot = heatpump_carnot_cop_heating(300.0, 270.0);
        let actual = heatpump_cop_heating(5000.0, 1250.0);
        assert_relative_eq!(
            heatpump_second_law_efficiency(actual, carnot),
            0.4,
            epsilon = 1e-12
        );
    }

    // Borne de Carnot : COP_chaud,carnot = COP_froid,carnot + 1.
    // COP_froid,carnot = T_froid/(T_chaud − T_froid) = 270/30 = 9 ; +1 = 10.
    #[test]
    fn carnot_bounds_satisfy_machine_relation() {
        let (t_hot, t_cold) = (300.0_f64, 270.0_f64);
        let carnot_cold = t_cold / (t_hot - t_cold);
        assert_relative_eq!(
            heatpump_carnot_cop_heating(t_hot, t_cold),
            heatpump_cop_relation(carnot_cold),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "la température chaude doit dépasser la température froide")]
    fn carnot_panics_when_hot_not_above_cold() {
        heatpump_carnot_cop_heating(280.0, 280.0);
    }
}
