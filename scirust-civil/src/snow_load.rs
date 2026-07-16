//! Actions de la neige sur les toitures selon l'**Eurocode 1** (EN 1991-1-3) :
//! charge de neige sur toiture, coefficient de forme d'une toiture à un versant,
//! majoration d'altitude de la charge au sol et charge d'accumulation (congère).
//!
//! ```text
//! charge sur toiture     s   = µi·Ce·Ct·sk
//! forme un versant       µ1  = 0,8                       si α ≤ 30°
//!                        µ1  = 0,8·(60 − α)/30           si 30° < α < 60°
//!                        µ1  = 0                          si α ≥ 60°
//! majoration d'altitude  sk  = sk0·(1 + A·cA)
//! charge d'accumulation  sd  = µd·sk
//! ```
//!
//! `µi` coefficient de forme de la toiture (–), `Ce` coefficient d'exposition
//! (–), `Ct` coefficient thermique (–), `sk` charge de neige au sol (Pa),
//! `s` charge de neige sur la toiture (Pa), `α` angle d'inclinaison du versant
//! (degrés), `µ1` coefficient de forme d'une toiture à un versant (–), `sk0`
//! charge de neige au sol au niveau de la mer (Pa), `A` altitude du site (m),
//! `cA` coefficient de majoration d'altitude (m⁻¹, Annexe Nationale), `µd`
//! coefficient de forme d'accumulation (–), `sd` charge d'accumulation (Pa).
//!
//! **Convention** : SI strict et cohérent — pascals (Pa, soit N/m²) pour les
//! charges surfaciques, mètres (m) pour l'altitude, degrés pour l'angle
//! d'inclinaison. Types `f64`.
//!
//! **Limite honnête** : l'action de la neige suit l'Eurocode 1-3. La charge de
//! neige au sol `sk` (carte régionale de la zone climatique), le coefficient
//! d'exposition `Ce` et le coefficient thermique `Ct` sont **fournis par
//! l'appelant** d'après l'Eurocode et son Annexe Nationale — jamais inventés.
//! Le coefficient de forme `µ1` est donné ici pour une **toiture à un versant**
//! (les autres géométries — deux versants, noue, toiture cylindrique — sont à
//! la charge de l'appelant). La majoration d'altitude suit l'Annexe Nationale
//! avec un coefficient `cA` **fourni**, la charge au niveau de la mer `sk0`
//! étant elle aussi **fournie**. La charge d'accumulation utilise un coefficient
//! de forme `µd` **fourni** (dépendant de l'obstacle et de la configuration).
//! Ce module ne fournit **aucune** valeur de carte ni de table par défaut.

/// Charge de neige sur la toiture `s = µi·Ce·Ct·sk` (Pa).
///
/// Panique si l'un des coefficients `shape_coefficient`, `exposure_coefficient`,
/// `thermal_coefficient` est négatif, ou si `ground_snow_load < 0`.
pub fn snow_load_on_roof(
    shape_coefficient: f64,
    exposure_coefficient: f64,
    thermal_coefficient: f64,
    ground_snow_load: f64,
) -> f64 {
    assert!(
        shape_coefficient >= 0.0,
        "le coefficient de forme µi doit être positif ou nul"
    );
    assert!(
        exposure_coefficient >= 0.0,
        "le coefficient d'exposition Ce doit être positif ou nul"
    );
    assert!(
        thermal_coefficient >= 0.0,
        "le coefficient thermique Ct doit être positif ou nul"
    );
    assert!(
        ground_snow_load >= 0.0,
        "la charge de neige au sol sk doit être positive ou nulle"
    );
    shape_coefficient * exposure_coefficient * thermal_coefficient * ground_snow_load
}

/// Coefficient de forme `µ1` d'une toiture à un versant en fonction de l'angle
/// d'inclinaison `α` (degrés) : `0,8` si `α ≤ 30°`, `0,8·(60 − α)/30` si
/// `30° < α < 60°`, et `0` si `α ≥ 60°`.
///
/// Panique si `pitch_angle_deg < 0` ou `pitch_angle_deg > 90`.
pub fn snow_shape_coefficient_monopitch(pitch_angle_deg: f64) -> f64 {
    assert!(
        pitch_angle_deg >= 0.0,
        "l'angle d'inclinaison α doit être positif ou nul"
    );
    assert!(
        pitch_angle_deg <= 90.0,
        "l'angle d'inclinaison α doit être inférieur ou égal à 90 degrés"
    );
    if pitch_angle_deg <= 30.0
    {
        0.8
    }
    else if pitch_angle_deg < 60.0
    {
        0.8 * (60.0 - pitch_angle_deg) / 30.0
    }
    else
    {
        0.0
    }
}

/// Majoration d'altitude de la charge de neige au sol
/// `sk = sk0·(1 + A·cA)` (Pa), où `sk0` est la charge au niveau de la mer,
/// `A` l'altitude (m) et `cA` le coefficient de majoration (m⁻¹, Annexe
/// Nationale).
///
/// Panique si `sea_level_snow_load < 0`, `altitude < 0` ou
/// `altitude_coefficient < 0`.
pub fn snow_altitude_adjustment(
    sea_level_snow_load: f64,
    altitude: f64,
    altitude_coefficient: f64,
) -> f64 {
    assert!(
        sea_level_snow_load >= 0.0,
        "la charge au niveau de la mer sk0 doit être positive ou nulle"
    );
    assert!(altitude >= 0.0, "l'altitude A doit être positive ou nulle");
    assert!(
        altitude_coefficient >= 0.0,
        "le coefficient de majoration d'altitude cA doit être positif ou nul"
    );
    sea_level_snow_load * (1.0 + altitude * altitude_coefficient)
}

/// Charge d'accumulation (congère) `sd = µd·sk` (Pa), produit du coefficient de
/// forme d'accumulation par la charge de neige au sol.
///
/// Panique si `drift_shape_coefficient < 0` ou `ground_snow_load < 0`.
pub fn snow_drift_load(drift_shape_coefficient: f64, ground_snow_load: f64) -> f64 {
    assert!(
        drift_shape_coefficient >= 0.0,
        "le coefficient de forme d'accumulation µd doit être positif ou nul"
    );
    assert!(
        ground_snow_load >= 0.0,
        "la charge de neige au sol sk doit être positive ou nulle"
    );
    drift_shape_coefficient * ground_snow_load
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn roof_load_is_product_of_coefficients() {
        // s = µi·Ce·Ct·sk : identité produit avec Ce = Ct = 1.
        // s = 0,8·1·1·900 = 720 Pa.
        let s = snow_load_on_roof(0.8, 1.0, 1.0, 900.0);
        assert_relative_eq!(s, 720.0, max_relative = 1e-12);
        // Proportionnalité : doubler sk double la charge sur toiture.
        assert_relative_eq!(
            snow_load_on_roof(0.8, 1.0, 1.0, 1800.0),
            2.0 * s,
            max_relative = 1e-12
        );
    }

    #[test]
    fn shape_coefficient_plateau_and_zero() {
        // α ≤ 30° : plateau à 0,8 (toits faiblement inclinés).
        assert_relative_eq!(
            snow_shape_coefficient_monopitch(0.0),
            0.8,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            snow_shape_coefficient_monopitch(30.0),
            0.8,
            max_relative = 1e-12
        );
        // α ≥ 60° : la neige glisse, µ1 = 0.
        assert_relative_eq!(snow_shape_coefficient_monopitch(60.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(snow_shape_coefficient_monopitch(75.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn shape_coefficient_linear_transition() {
        // 30° < α < 60° : décroissance linéaire.
        // α = 45° : µ1 = 0,8·(60 − 45)/30 = 0,8·0,5 = 0,4.
        assert_relative_eq!(
            snow_shape_coefficient_monopitch(45.0),
            0.4,
            max_relative = 1e-12
        );
        // Milieu de l'intervalle : la valeur est la moyenne des bornes 0,8 et 0.
        let mid = snow_shape_coefficient_monopitch(45.0);
        assert_relative_eq!(
            mid,
            0.5 * (snow_shape_coefficient_monopitch(30.0) + 0.0),
            max_relative = 1e-12
        );
    }

    #[test]
    fn altitude_adjustment_reduces_to_sea_level() {
        // À l'altitude nulle, sk = sk0 (pas de majoration).
        assert_relative_eq!(
            snow_altitude_adjustment(550.0, 0.0, 0.001),
            550.0,
            max_relative = 1e-12
        );
        // Cas chiffré : sk0 = 550 Pa, A = 1000 m, cA = 0,001 m⁻¹.
        // sk = 550·(1 + 1000·0,001) = 550·2 = 1100 Pa.
        assert_relative_eq!(
            snow_altitude_adjustment(550.0, 1000.0, 0.001),
            1100.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn worked_case_alpine_roof() {
        // Toiture à un versant, α = 40°, site à 1000 m d'altitude.
        // Coefficient de forme : µ1 = 0,8·(60 − 40)/30 = 0,8·(20/30)
        //   = 0,8·0,666667 = 0,533333.
        let mu1 = snow_shape_coefficient_monopitch(40.0);
        assert_relative_eq!(mu1, 0.533333, max_relative = 1e-3);
        // Charge au sol majorée : sk0 = 650 Pa, cA = 0,001 m⁻¹.
        // sk = 650·(1 + 1000·0,001) = 650·2 = 1300 Pa.
        let sk = snow_altitude_adjustment(650.0, 1000.0, 0.001);
        assert_relative_eq!(sk, 1300.0, max_relative = 1e-12);
        // Charge sur toiture : Ce = 1,0, Ct = 1,0.
        // s = 0,533333·1·1·1300 = 693,333 Pa.
        let s = snow_load_on_roof(mu1, 1.0, 1.0, sk);
        assert_relative_eq!(s, 693.333, max_relative = 1e-3);
        // Charge d'accumulation contre un acrotère : µd = 2,0.
        // sd = 2,0·1300 = 2600 Pa.
        let sd = snow_drift_load(2.0, sk);
        assert_relative_eq!(sd, 2600.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la charge de neige au sol sk doit être positive ou nulle")]
    fn negative_ground_load_panics() {
        let _ = snow_load_on_roof(0.8, 1.0, 1.0, -10.0);
    }
}
