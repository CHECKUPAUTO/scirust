//! Bilan macroscopique d'une **tour de refroidissement** à circulation d'eau —
//! plage de refroidissement (*range*), approche à la température humide
//! (*approach*), efficacité, pertes par évaporation par bilan enthalpique, eau
//! d'appoint (*make-up*) et cycles de concentration.
//!
//! ```text
//! plage de refroidissement   range = T_hot − T_cold                        [K]
//! approche (temp. humide)    approach = T_cold − T_wb                       [K]
//! efficacité                 ε = range / (range + approach)                 [-]
//! pertes par évaporation     E = ṁ_w·c_p·range / λ                          [kg·s⁻¹]
//! eau d'appoint totale       M = E + D + B                                  [kg·s⁻¹]
//! cycles de concentration    C = M / B                                      [-]
//! ```
//!
//! `T_hot` température de l'eau chaude en entrée de tour [K], `T_cold` température
//! de l'eau froide en sortie (bassin) [K], `T_wb` température humide de l'air
//! ambiant [K], `range` plage de refroidissement [K], `approach` approche par
//! rapport à la température humide [K], `ε` efficacité de la tour
//! [sans dimension, 0 ≤ ε ≤ 1], `ṁ_w` débit massique d'eau en circulation
//! [kg·s⁻¹], `c_p` capacité thermique massique de l'eau [J·kg⁻¹·K⁻¹], `λ` chaleur
//! latente de vaporisation de l'eau [J·kg⁻¹], `E` pertes par évaporation
//! [kg·s⁻¹], `D` pertes par entraînement (*drift*) [kg·s⁻¹], `B` purge de
//! déconcentration (*blowdown*) [kg·s⁻¹], `M` débit d'eau d'appoint [kg·s⁻¹],
//! `C` cycles de concentration [sans dimension].
//!
//! **Limite honnête** : la **température humide** `T_wb`, la **chaleur latente**
//! `λ` et la **capacité thermique** `c_p` de l'eau sont **FOURNIES par
//! l'appelant** (tables psychrométriques, propriétés d'état, essais) — jamais
//! inventées ici. L'**approche est bornée par la température humide** : l'eau
//! froide ne peut, à l'équilibre, descendre en dessous de `T_wb`, d'où
//! `approach ≥ 0`. Ces relations sont des **bilans macroscopiques en régime
//! permanent** ; elles **ne modélisent pas le transfert dans le garnissage**
//! (nombre d'unités de transfert, méthode de Merkel/`KaV/L`) et ne calculent ni
//! équilibres air-eau, ni profils de température le long de la tour. Ce module se
//! tient au niveau des **opérations unitaires** et ne duplique ni
//! `scirust-thermo` (propriétés d'état, cycles) ni `scirust-fluids` (mécanique
//! des fluides fondamentale).

/// Plage de refroidissement `range = T_hot − T_cold` [K] : chute de température
/// de l'eau à travers la tour.
///
/// `hot_water_temperature` `T_hot` température de l'eau chaude en entrée [K, > 0],
/// `cold_water_temperature` `T_cold` température de l'eau froide en sortie
/// [K, > 0].
///
/// Panique si l'une des températures n'est pas finie ou n'est pas strictement
/// positive (échelle absolue), ou si `hot_water_temperature` est inférieure à
/// `cold_water_temperature` (une tour refroidit l'eau : `range ≥ 0`).
pub fn ctwr_range(hot_water_temperature: f64, cold_water_temperature: f64) -> f64 {
    assert!(
        hot_water_temperature.is_finite() && hot_water_temperature > 0.0,
        "la température de l'eau chaude doit être finie et strictement positive (K)"
    );
    assert!(
        cold_water_temperature.is_finite() && cold_water_temperature > 0.0,
        "la température de l'eau froide doit être finie et strictement positive (K)"
    );
    assert!(
        hot_water_temperature >= cold_water_temperature,
        "l'eau chaude doit être au moins aussi chaude que l'eau froide (range ≥ 0)"
    );
    hot_water_temperature - cold_water_temperature
}

/// Approche `approach = T_cold − T_wb` [K] : écart entre l'eau froide et la
/// température humide de l'air, mesure de la finesse d'approche de la tour.
///
/// `cold_water_temperature` `T_cold` température de l'eau froide en sortie
/// [K, > 0], `wet_bulb_temperature` `T_wb` température humide de l'air ambiant
/// [K, > 0 ; FOURNIE par l'appelant].
///
/// Panique si l'une des températures n'est pas finie ou n'est pas strictement
/// positive (échelle absolue), ou si `cold_water_temperature` est inférieure à
/// `wet_bulb_temperature` : l'eau ne peut descendre en dessous de la température
/// humide (`approach ≥ 0`).
pub fn ctwr_approach(cold_water_temperature: f64, wet_bulb_temperature: f64) -> f64 {
    assert!(
        cold_water_temperature.is_finite() && cold_water_temperature > 0.0,
        "la température de l'eau froide doit être finie et strictement positive (K)"
    );
    assert!(
        wet_bulb_temperature.is_finite() && wet_bulb_temperature > 0.0,
        "la température humide doit être finie et strictement positive (K)"
    );
    assert!(
        cold_water_temperature >= wet_bulb_temperature,
        "l'eau ne peut descendre en dessous de la température humide (approach ≥ 0)"
    );
    cold_water_temperature - wet_bulb_temperature
}

/// Efficacité de la tour `ε = range / (range + approach)` [sans dimension,
/// 0 ≤ ε ≤ 1] : fraction de l'écart maximal théorique `T_hot − T_wb`
/// effectivement récupérée en refroidissement.
///
/// `range` plage de refroidissement [K, ≥ 0], `approach` approche à la
/// température humide [K, ≥ 0].
///
/// Panique si `range` ou `approach` n'est pas fini ou est négatif, ou si leur
/// somme est nulle (efficacité indéterminée : ni plage ni approche).
pub fn ctwr_effectiveness(range: f64, approach: f64) -> f64 {
    assert!(
        range.is_finite() && range >= 0.0,
        "la plage de refroidissement doit être finie et positive ou nulle"
    );
    assert!(
        approach.is_finite() && approach >= 0.0,
        "l'approche doit être finie et positive ou nulle"
    );
    assert!(
        range + approach > 0.0,
        "la somme plage + approche doit être strictement positive"
    );
    range / (range + approach)
}

/// Pertes par évaporation `E = ṁ_w·c_p·range / λ` [kg·s⁻¹] par bilan
/// enthalpique : la chaleur sensible cédée par l'eau est évacuée par
/// vaporisation d'une fraction du débit en circulation.
///
/// `water_flow` `ṁ_w` débit massique d'eau en circulation [kg·s⁻¹, ≥ 0],
/// `range` plage de refroidissement [K, ≥ 0], `latent_heat` `λ` chaleur latente
/// de vaporisation de l'eau [J·kg⁻¹, > 0 ; FOURNIE par l'appelant],
/// `water_heat_capacity` `c_p` capacité thermique massique de l'eau
/// [J·kg⁻¹·K⁻¹, > 0 ; FOURNIE par l'appelant].
///
/// Panique si `water_flow` ou `range` n'est pas fini ou est négatif, ou si
/// `latent_heat` ou `water_heat_capacity` n'est pas fini ou n'est pas strictement
/// positif.
pub fn ctwr_evaporation_loss(
    water_flow: f64,
    range: f64,
    latent_heat: f64,
    water_heat_capacity: f64,
) -> f64 {
    assert!(
        water_flow.is_finite() && water_flow >= 0.0,
        "le débit d'eau doit être fini et positif ou nul"
    );
    assert!(
        range.is_finite() && range >= 0.0,
        "la plage de refroidissement doit être finie et positive ou nulle"
    );
    assert!(
        latent_heat.is_finite() && latent_heat > 0.0,
        "la chaleur latente doit être finie et strictement positive"
    );
    assert!(
        water_heat_capacity.is_finite() && water_heat_capacity > 0.0,
        "la capacité thermique doit être finie et strictement positive"
    );
    water_flow * water_heat_capacity * range / latent_heat
}

/// Eau d'appoint totale `M = E + D + B` [kg·s⁻¹] : elle compense l'ensemble des
/// pertes en eau de la tour (évaporation, entraînement, purge).
///
/// `evaporation_loss` `E` pertes par évaporation [kg·s⁻¹, ≥ 0], `drift_loss` `D`
/// pertes par entraînement de gouttelettes [kg·s⁻¹, ≥ 0], `blowdown` `B` purge
/// de déconcentration [kg·s⁻¹, ≥ 0].
///
/// Panique si l'un des trois débits n'est pas fini ou est négatif.
pub fn ctwr_makeup_water(evaporation_loss: f64, drift_loss: f64, blowdown: f64) -> f64 {
    assert!(
        evaporation_loss.is_finite() && evaporation_loss >= 0.0,
        "les pertes par évaporation doivent être finies et positives ou nulles"
    );
    assert!(
        drift_loss.is_finite() && drift_loss >= 0.0,
        "les pertes par entraînement doivent être finies et positives ou nulles"
    );
    assert!(
        blowdown.is_finite() && blowdown >= 0.0,
        "la purge doit être finie et positive ou nulle"
    );
    evaporation_loss + drift_loss + blowdown
}

/// Cycles de concentration `C = M / B` [sans dimension] : rapport entre le débit
/// d'appoint et la purge, mesurant l'enrichissement en sels de l'eau en
/// circulation par rapport à l'eau d'appoint.
///
/// `makeup` `M` débit d'eau d'appoint [kg·s⁻¹, ≥ 0], `blowdown` `B` purge de
/// déconcentration [kg·s⁻¹, > 0].
///
/// Panique si `makeup` n'est pas fini ou est négatif, ou si `blowdown` n'est pas
/// fini ou n'est pas strictement positif (division par la purge).
pub fn ctwr_cycles_of_concentration(makeup: f64, blowdown: f64) -> f64 {
    assert!(
        makeup.is_finite() && makeup >= 0.0,
        "le débit d'appoint doit être fini et positif ou nul"
    );
    assert!(
        blowdown.is_finite() && blowdown > 0.0,
        "la purge doit être finie et strictement positive"
    );
    makeup / blowdown
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn range_chiffre_et_additivite() {
        // T_hot = 310 K, T_cold = 300 K → range = 10 K.
        let r = ctwr_range(310.0, 300.0);
        assert_relative_eq!(r, 10.0, epsilon = 1.0e-3);
        // range est une différence : décaler les deux bornes le laisse inchangé.
        assert_relative_eq!(ctwr_range(315.0, 305.0), r, epsilon = 1.0e-3);
    }

    #[test]
    fn approach_chiffre_et_lien_effectivite() {
        // T_cold = 300 K, T_wb = 294 K → approach = 6 K.
        let a = ctwr_approach(300.0, 294.0);
        assert_relative_eq!(a, 6.0, epsilon = 1.0e-3);
        // Avec range = 10 K : range + approach = 16 K = T_hot − T_wb (310 − 294).
        let r = ctwr_range(310.0, 300.0);
        assert_relative_eq!(r + a, 310.0 - 294.0, epsilon = 1.0e-3);
    }

    #[test]
    fn effectivite_limites_et_cas_chiffre() {
        // approach = 0 → ε = 1 (eau refroidie jusqu'à la température humide).
        assert_relative_eq!(ctwr_effectiveness(10.0, 0.0), 1.0, epsilon = 1.0e-3);
        // range = approach → ε = 1/2.
        assert_relative_eq!(ctwr_effectiveness(6.0, 6.0), 0.5, epsilon = 1.0e-3);
        // range = 10, approach = 6 → ε = 10/16 = 0,625.
        assert_relative_eq!(ctwr_effectiveness(10.0, 6.0), 0.625, epsilon = 1.0e-3);
    }

    #[test]
    fn evaporation_cas_chiffre_et_proportionnalite() {
        // ṁ_w = 1000 kg/s, c_p = 4200 J/(kg·K), range = 10 K, λ = 2,4e6 J/kg.
        // E = 1000·4200·10 / 2,4e6 = 42 000 000 / 2 400 000 = 17,5 kg/s.
        let e = ctwr_evaporation_loss(1000.0, 10.0, 2.4e6, 4200.0);
        assert_relative_eq!(e, 17.5, epsilon = 1.0e-3);
        // E est linéaire en débit d'eau : doubler ṁ_w double E.
        assert_relative_eq!(
            ctwr_evaporation_loss(2000.0, 10.0, 2.4e6, 4200.0),
            2.0 * e,
            epsilon = 1.0e-3
        );
    }

    #[test]
    fn appoint_somme_et_cycles_chiffres() {
        // E = 17,5 ; D = 0,5 ; B = 2,0 → M = 20,0 kg/s.
        let m = ctwr_makeup_water(17.5, 0.5, 2.0);
        assert_relative_eq!(m, 20.0, epsilon = 1.0e-3);
        // C = M / B = 20,0 / 2,0 = 10.
        assert_relative_eq!(ctwr_cycles_of_concentration(m, 2.0), 10.0, epsilon = 1.0e-3);
    }

    #[test]
    #[should_panic(expected = "en dessous de la température humide")]
    fn approach_refuse_eau_sous_temperature_humide() {
        // T_cold = 290 K < T_wb = 294 K : physiquement impossible en régime permanent.
        let _ = ctwr_approach(290.0, 294.0);
    }
}
