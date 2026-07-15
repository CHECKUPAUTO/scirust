//! **Tour de refroidissement** — performance thermique d'une tour humide :
//! plage de refroidissement, approche du bulbe humide, efficacité, perte par
//! évaporation et puissance thermique rejetée par le débit d'eau en circulation.
//!
//! ```text
//! plage             range     = T_chaud − T_froid
//! approche          approach  = T_froid − T_humide
//! efficacité        eff       = range / (range + approach)
//! perte évaporation Q_evap    = water_flow · range · evaporation_factor
//! puissance rejetée Q_rejet   = mdot · cp · range
//! ```
//!
//! `T_chaud` température de l'eau chaude en entrée de tour (K), `T_froid`
//! température de l'eau froide en sortie de bassin (K), `T_humide` température
//! du **bulbe humide** de l'air ambiant (K), `range` plage de refroidissement
//! (K), `approach` approche du bulbe humide (K), `eff` efficacité (sans
//! dimension, 0 à 1), `water_flow` débit d'eau en circulation (même unité que
//! `Q_evap`, p.ex. kg·s⁻¹ ou m³·h⁻¹), `evaporation_factor` facteur d'évaporation
//! (K⁻¹, ≈ 0,00085 K⁻¹), `Q_evap` perte d'eau par évaporation (même unité que
//! `water_flow`), `mdot` débit-masse d'eau (kg·s⁻¹), `cp` chaleur massique de
//! l'eau (J·kg⁻¹·K⁻¹), `Q_rejet` puissance thermique rejetée (W).
//!
//! **Convention** : températures en kelvin (les différences `range` et
//! `approach` sont numériquement identiques en °C) ; unités SI cohérentes par
//! ailleurs. L'efficacité est le rapport de la plage effective à la plage
//! maximale théorique `range + approach` (eau refroidie jusqu'au bulbe humide).
//!
//! **Limite honnête** : simple bilan thermique global à débit d'eau constant,
//! sans modèle de garnissage ni de transfert couplé masse/chaleur (Merkel,
//! e-NTU). Les températures — **dont la température de bulbe humide** — sont des
//! **données fournies par l'appelant** (psychrométrie ou mesure), de même que le
//! **facteur d'évaporation** (il dépend du climat et de la charge) et la chaleur
//! massique de l'eau : aucune constante physique, de matériau ou de procédé
//! n'est inventée « par défaut » ici. L'approche est bornée par le bulbe humide,
//! qui n'est **jamais atteint** en pratique (`approach > 0`). Complète
//! [`crate::psychrometrics`].

/// Plage de refroidissement `range = T_chaud − T_froid` (K).
///
/// `hot_water_temp` température de l'eau chaude en entrée `T_chaud` (K),
/// `cold_water_temp` température de l'eau froide en sortie `T_froid` (K) ;
/// renvoie la plage de refroidissement en K.
///
/// Panique si `cold_water_temp > hot_water_temp` (plage négative interdite).
pub fn coolingtower_range(hot_water_temp: f64, cold_water_temp: f64) -> f64 {
    assert!(
        hot_water_temp >= cold_water_temp,
        "plage négative interdite : l'eau chaude doit être au moins aussi chaude que l'eau froide"
    );
    hot_water_temp - cold_water_temp
}

/// Approche du bulbe humide `approach = T_froid − T_humide` (K).
///
/// `cold_water_temp` température de l'eau froide en sortie `T_froid` (K),
/// `wet_bulb_temp` température de bulbe humide de l'air `T_humide` (K) ; renvoie
/// l'approche en K. Le bulbe humide est la limite basse jamais atteinte par
/// l'eau refroidie, d'où `approach >= 0`.
///
/// Panique si `wet_bulb_temp > cold_water_temp` (approche négative interdite).
pub fn coolingtower_approach(cold_water_temp: f64, wet_bulb_temp: f64) -> f64 {
    assert!(
        cold_water_temp >= wet_bulb_temp,
        "approche négative interdite : l'eau froide ne peut être plus froide que le bulbe humide"
    );
    cold_water_temp - wet_bulb_temp
}

/// Efficacité thermique `eff = range / (range + approach)` (sans dimension).
///
/// `range` plage de refroidissement (K), `approach` approche du bulbe humide
/// (K) ; renvoie l'efficacité comprise entre 0 et 1 (1 = eau refroidie jusqu'au
/// bulbe humide, approche nulle).
///
/// Panique si `range < 0`, si `approach < 0` ou si `range + approach <= 0`
/// (plage totale nulle, efficacité indéfinie).
pub fn coolingtower_effectiveness(range: f64, approach: f64) -> f64 {
    assert!(range >= 0.0, "plage de refroidissement négative interdite");
    assert!(approach >= 0.0, "approche négative interdite");
    assert!(
        range + approach > 0.0,
        "plage totale strictement positive requise (efficacité indéfinie sinon)"
    );
    range / (range + approach)
}

/// Perte d'eau par évaporation `Q_evap = water_flow · range · evaporation_factor`.
///
/// `water_flow` débit d'eau en circulation (même unité que le résultat, p.ex.
/// kg·s⁻¹ ou m³·h⁻¹), `range` plage de refroidissement (K), `evaporation_factor`
/// facteur d'évaporation **fourni** (K⁻¹, ≈ 0,00085 K⁻¹) ; renvoie la perte par
/// évaporation dans la même unité que `water_flow`.
///
/// Panique si `water_flow < 0`, si `range < 0` ou si `evaporation_factor < 0`.
pub fn coolingtower_evaporation_loss(water_flow: f64, range: f64, evaporation_factor: f64) -> f64 {
    assert!(water_flow >= 0.0, "débit d'eau négatif interdit");
    assert!(range >= 0.0, "plage de refroidissement négative interdite");
    assert!(
        evaporation_factor >= 0.0,
        "facteur d'évaporation négatif interdit"
    );
    water_flow * range * evaporation_factor
}

/// Puissance thermique rejetée `Q_rejet = mdot · cp · range` (W).
///
/// `water_flow` débit-masse d'eau `mdot` (kg·s⁻¹), `specific_heat` chaleur
/// massique de l'eau `cp` **fournie** (J·kg⁻¹·K⁻¹), `range` plage de
/// refroidissement (K) ; renvoie la puissance thermique rejetée en W.
///
/// Panique si `water_flow < 0`, si `specific_heat <= 0` ou si `range < 0`.
pub fn coolingtower_heat_rejected(water_flow: f64, specific_heat: f64, range: f64) -> f64 {
    assert!(water_flow >= 0.0, "débit-masse d'eau négatif interdit");
    assert!(
        specific_heat > 0.0,
        "chaleur massique strictement positive requise"
    );
    assert!(range >= 0.0, "plage de refroidissement négative interdite");
    water_flow * specific_heat * range
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn range_matches_hand_calc() {
        // Eau chaude 313,15 K (40 °C), eau froide 305,15 K (32 °C) :
        // range = 313,15 − 305,15 = 8 K.
        let range = coolingtower_range(313.15, 305.15);
        assert_relative_eq!(range, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn approach_matches_hand_calc() {
        // Eau froide 305,15 K (32 °C), bulbe humide 301,15 K (28 °C) :
        // approach = 305,15 − 301,15 = 4 K.
        let approach = coolingtower_approach(305.15, 301.15);
        assert_relative_eq!(approach, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn effectiveness_composes_with_range_and_approach() {
        // range = 8 K, approach = 4 K : eff = 8 / (8 + 4) = 2/3.
        // Vérifie aussi la cohérence de bout en bout à partir des températures.
        let range = coolingtower_range(313.15, 305.15);
        let approach = coolingtower_approach(305.15, 301.15);
        let eff = coolingtower_effectiveness(range, approach);
        assert_relative_eq!(eff, 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn effectiveness_is_unity_when_approach_is_zero() {
        // Approche nulle : eau refroidie jusqu'au bulbe humide, eff = 1.
        let eff = coolingtower_effectiveness(8.0, 0.0);
        assert_relative_eq!(eff, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn evaporation_loss_matches_hand_calc() {
        // water_flow = 1000 kg/s, range = 8 K, facteur = 0,00085 K⁻¹ :
        // Q_evap = 1000 · 8 · 0,00085 = 6,8 kg/s.
        let q_evap = coolingtower_evaporation_loss(1000.0, 8.0, 0.000_85);
        assert_relative_eq!(q_evap, 6.8, epsilon = 1e-12);
    }

    #[test]
    fn heat_rejected_matches_hand_calc_and_scales_with_range() {
        // mdot = 50 kg/s, cp = 4186 J/(kg·K), range = 8 K :
        // Q = 50 · 4186 · 8 = 1 674 400 W.
        let q = coolingtower_heat_rejected(50.0, 4186.0, 8.0);
        assert_relative_eq!(q, 1_674_400.0, epsilon = 1e-6);
        // La puissance est proportionnelle à la plage : doubler range double Q.
        let q_double = coolingtower_heat_rejected(50.0, 4186.0, 16.0);
        assert_relative_eq!(q_double, 2.0 * q, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "plage négative interdite")]
    fn range_rejects_cold_hotter_than_hot() {
        let _ = coolingtower_range(305.15, 313.15);
    }
}
