//! **Rendement de chaudière** — méthode directe (entrées/sorties), méthode
//! indirecte (bilan des pertes), perte par fumées sèches et taux de purge.
//!
//! ```text
//! méthode directe       η = m_v · (h_v − h_e) / (m_f · CV)
//! méthode indirecte     η = 1 − Σ L_i
//! perte fumées sèches    L_fg = m_fg · c_p · (T_fg − T_amb) / CV
//! taux de purge          b = TDS_e / (TDS_c − TDS_e)
//! ```
//!
//! `m_v` débit de vapeur (kg/s), `h_v`/`h_e` enthalpies vapeur/eau d'alimentation
//! (J/kg), `m_f` débit de combustible (kg/s), `CV` pouvoir calorifique (J/kg),
//! `η` rendement (sans dimension), `L_i`/`Σ L_i` pertes fractionnaires
//! individuelles/cumulées (sans dimension), `m_fg` masse de fumées par unité de
//! combustible (kg/kg), `c_p` chaleur massique des fumées (J/(kg·K)),
//! `T_fg`/`T_amb` températures fumées/ambiante (K ; seule la différence importe),
//! `TDS_e`/`TDS_c` teneurs en solides dissous eau d'alimentation/chaudière (même
//! unité, p. ex. ppm), `b` taux de purge (sans dimension).
//!
//! **Convention** : SI ; régime permanent. **Limite honnête** : les enthalpies
//! (tables de vapeur), le pouvoir calorifique `CV`, la chaleur massique `c_p` et
//! tous les débits sont **fournis par l'appelant** (mesures et tables réelles) ;
//! la méthode indirecte exige que **TOUTES** les pertes (fumées sèches,
//! imbrûlés, rayonnement, purge, humidité…) soient déjà calculées et **fournies**
//! par l'appelant — aucune valeur « par défaut » n'est inventée.

/// Rendement par la méthode directe `η = m_v · (h_v − h_e) / (m_f · CV)`
/// (rapport de la chaleur utile transmise à la vapeur sur l'énergie du
/// combustible).
///
/// Panique si un débit ou `CV` n'est pas strictement positif, ou si
/// `h_steam <= h_feed` (chaleur nette non positive).
pub fn boiler_efficiency_direct(
    steam_mass: f64,
    steam_enthalpy: f64,
    feedwater_enthalpy: f64,
    fuel_mass: f64,
    calorific_value: f64,
) -> f64 {
    assert!(steam_mass > 0.0, "débit de vapeur m_v > 0 requis");
    assert!(fuel_mass > 0.0, "débit de combustible m_f > 0 requis");
    assert!(calorific_value > 0.0, "pouvoir calorifique CV > 0 requis");
    assert!(
        steam_enthalpy > feedwater_enthalpy,
        "h_v > h_e requis (chaleur nette transmise positive)"
    );
    steam_mass * (steam_enthalpy - feedwater_enthalpy) / (fuel_mass * calorific_value)
}

/// Rendement par la méthode indirecte `η = 1 − Σ L_i` (complément à l'unité de
/// la somme de **toutes** les pertes fractionnaires fournies).
///
/// Panique si `losses_fraction_sum` n'est pas dans `[0, 1]`.
pub fn boiler_efficiency_indirect(losses_fraction_sum: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&losses_fraction_sum),
        "somme des pertes Σ L_i ∈ [0, 1] requise"
    );
    1.0 - losses_fraction_sum
}

/// Perte fractionnaire par fumées sèches
/// `L_fg = m_fg · c_p · (T_fg − T_amb) / CV`.
///
/// Panique si `flue_gas_mass`, `specific_heat` ou `fuel_calorific_value` n'est
/// pas strictement positif, ou si `flue_temperature < ambient_temperature`.
pub fn boiler_dry_flue_gas_loss(
    flue_gas_mass: f64,
    specific_heat: f64,
    flue_temperature: f64,
    ambient_temperature: f64,
    fuel_calorific_value: f64,
) -> f64 {
    assert!(flue_gas_mass > 0.0, "masse de fumées m_fg > 0 requise");
    assert!(specific_heat > 0.0, "chaleur massique c_p > 0 requise");
    assert!(
        fuel_calorific_value > 0.0,
        "pouvoir calorifique CV > 0 requis"
    );
    assert!(
        flue_temperature >= ambient_temperature,
        "T_fg ≥ T_amb requis (fumées plus chaudes que l'ambiance)"
    );
    flue_gas_mass * specific_heat * (flue_temperature - ambient_temperature) / fuel_calorific_value
}

/// Taux de purge continue `b = TDS_e / (TDS_c − TDS_e)`, fraction du débit
/// d'alimentation à purger pour maintenir la teneur en solides de la chaudière.
///
/// Panique si `feedwater_tds < 0` ou si `boiler_tds <= feedwater_tds`
/// (dénominateur non positif).
pub fn boiler_blowdown_rate(feedwater_tds: f64, boiler_tds: f64) -> f64 {
    assert!(feedwater_tds >= 0.0, "TDS_e ≥ 0 requis");
    assert!(
        boiler_tds > feedwater_tds,
        "TDS_c > TDS_e requis (concentration effective de la chaudière)"
    );
    feedwater_tds / (boiler_tds - feedwater_tds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn direct_realistic_case() {
        // Cas chiffré : m_v = 10 kg/s, h_v = 2800 kJ/kg, h_e = 400 kJ/kg,
        // m_f = 0.8 kg/s, CV = 40 MJ/kg.
        // η = 10·(2.8e6 − 4.0e5) / (0.8·4.0e7) = 2.4e7 / 3.2e7 = 0.75.
        let (m_v, h_v, h_e, m_f, cv) = (10.0_f64, 2_800.0e3_f64, 400.0e3_f64, 0.8_f64, 40.0e6_f64);
        assert_relative_eq!(
            boiler_efficiency_direct(m_v, h_v, h_e, m_f, cv),
            0.75,
            epsilon = 1e-12
        );
    }

    #[test]
    fn direct_scales_with_steam_mass() {
        // Proportionnalité : doubler le débit de vapeur double le rendement direct.
        let (h_v, h_e, m_f, cv) = (2_700.0e3_f64, 300.0e3_f64, 0.5_f64, 42.0e6_f64);
        let base = boiler_efficiency_direct(4.0, h_v, h_e, m_f, cv);
        let doubled = boiler_efficiency_direct(8.0, h_v, h_e, m_f, cv);
        assert_relative_eq!(doubled, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    fn indirect_sums_the_losses() {
        // Identité méthode indirecte : η = 1 − Σ L_i pour des pertes fournies.
        let losses = [0.08_f64, 0.05_f64, 0.02_f64, 0.01_f64];
        let total: f64 = losses.iter().sum();
        assert_relative_eq!(total, 0.16, epsilon = 1e-12);
        assert_relative_eq!(boiler_efficiency_indirect(total), 0.84, epsilon = 1e-12);
    }

    #[test]
    fn indirect_no_loss_is_unity() {
        // Cas limite : aucune perte → rendement unitaire.
        assert_relative_eq!(boiler_efficiency_indirect(0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn dry_flue_gas_loss_realistic_case() {
        // Cas chiffré : m_fg = 20 kg/kg, c_p = 1000 J/(kg·K),
        // T_fg = 480 K, T_amb = 300 K (ΔT = 180 K), CV = 45 MJ/kg.
        // L_fg = 20·1000·180 / 4.5e7 = 3.6e6 / 4.5e7 = 0.08.
        let (m_fg, cp, t_fg, t_amb, cv) = (20.0_f64, 1_000.0_f64, 480.0_f64, 300.0_f64, 45.0e6_f64);
        assert_relative_eq!(
            boiler_dry_flue_gas_loss(m_fg, cp, t_fg, t_amb, cv),
            0.08,
            epsilon = 1e-12
        );
    }

    #[test]
    fn blowdown_doubles_concentration_identity() {
        // Identité : si TDS_c = 2·TDS_e, alors b = TDS_e/(2·TDS_e − TDS_e) = 1.
        let feed = 60.0_f64;
        assert_relative_eq!(boiler_blowdown_rate(feed, 2.0 * feed), 1.0, epsilon = 1e-15);
        // Cas chiffré : TDS_e = 100 ppm, TDS_c = 2100 ppm → b = 100/2000 = 0.05.
        assert_relative_eq!(boiler_blowdown_rate(100.0, 2_100.0), 0.05, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "TDS_c > TDS_e requis")]
    fn blowdown_non_concentrating_panics() {
        boiler_blowdown_rate(500.0, 500.0);
    }
}
