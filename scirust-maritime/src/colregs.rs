//! Classification de situation de rencontre COLREG (*International
//! Regulations for Preventing Collisions at Sea*, règles 13-15) à partir
//! du relèvement relatif — la simplification géométrique à deux seuils
//! utilisée dans la littérature d'évitement de collision autonome (par
//! ex. Kuwata et al., "Safe Maritime Autonomous Navigation With COLREGS,
//! Using Velocity Obstacles", IEEE J. Oceanic Eng. 39(1), 2014 ; Campbell,
//! Naeem & Irwin, "A review on improving the autonomy of unmanned surface
//! vehicles through intelligent collision avoidance manoeuvres", Annual
//! Reviews in Control 36(2), 2012).
//!
//! ## Règle de classification
//! Relèvement relatif de la cible vue depuis le navire propre, mesuré
//! depuis l'avant (`0°` = droit devant, `+90°` = tribord travers, `-90°` =
//! bâbord travers, `±180°` = droit arrière) :
//! - `|relèvement| <= 15°` : **rencontre de face** (Règle 14).
//! - `|relèvement| >= 112.5°` : **rattrapage** (Règle 13) — le navire qui
//!   voit l'autre dans ce secteur arrière est celui qui doit manœuvrer,
//!   qu'il soit le rattrapant ou le rattrapé.
//! - sinon : **croisement** (Règle 15) — cible sur tribord (`> 0`) =
//!   navire propre non privilégié (doit manœuvrer) ; cible sur bâbord
//!   (`< 0`) = navire propre privilégié.
//!
//! **Limite honnête** : ceci classe la *géométrie* de rencontre, pas la
//! situation réglementaire complète (qui dépend aussi du statut des deux
//! navires — à voile, à propulsion mécanique, capacité de manœuvre
//! restreinte, etc., Règles 11-18) : c'est la brique géométrique sur
//! laquelle une logique COLREG complète se construirait, pas cette
//! logique elle-même.

/// Normalise un angle en degrés dans `(-180, 180]`.
fn wrap_pm180(angle_deg: f64) -> f64 {
    let mut a = angle_deg % 360.0;
    if a <= -180.0
    {
        a += 360.0;
    }
    else if a > 180.0
    {
        a -= 360.0;
    }
    a
}

/// Relèvement relatif (degrés, convention ci-dessus) de la cible à
/// `target_pos` vue depuis `own_pos` (coordonnées est/nord, même unité),
/// compte tenu du cap propre `own_heading_deg` (convention compas :
/// `0°` = nord, `90°` = est, mesuré horaire).
pub fn relative_bearing_deg(
    own_pos: (f64, f64),
    own_heading_deg: f64,
    target_pos: (f64, f64),
) -> f64 {
    let (dx, dy) = (target_pos.0 - own_pos.0, target_pos.1 - own_pos.1);
    let bearing_to_target = dx.atan2(dy).to_degrees();
    wrap_pm180(bearing_to_target - own_heading_deg)
}

/// Type de situation de rencontre COLREG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncounterType {
    HeadOn,
    Overtaking,
    /// Croisement où le navire propre doit manœuvrer (cible sur tribord, Règle 15).
    CrossingGiveWay,
    /// Croisement où le navire propre est privilégié (cible sur bâbord, Règle 15).
    CrossingStandOn,
}

/// Classifie la situation de rencontre à partir du relèvement relatif de
/// la cible (degrés, convention de [`relative_bearing_deg`]).
pub fn classify_encounter(bearing_deg: f64) -> EncounterType {
    let b = bearing_deg.abs();
    if b <= 15.0
    {
        EncounterType::HeadOn
    }
    else if b >= 112.5
    {
        EncounterType::Overtaking
    }
    else if bearing_deg > 0.0
    {
        EncounterType::CrossingGiveWay
    }
    else
    {
        EncounterType::CrossingStandOn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn target_dead_ahead_on_own_heading_zero() {
        let b = relative_bearing_deg((0.0, 0.0), 0.0, (0.0, 10.0));
        assert_relative_eq!(b, 0.0, epsilon = 1e-9);
        assert_eq!(classify_encounter(b), EncounterType::HeadOn);
    }

    #[test]
    fn target_dead_ahead_on_a_non_zero_heading() {
        // Own heading 045°, target along that same bearing from own ship.
        let b = relative_bearing_deg((0.0, 0.0), 45.0, (10.0, 10.0));
        assert_relative_eq!(b, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn target_on_starboard_beam_is_crossing_give_way() {
        let b = relative_bearing_deg((0.0, 0.0), 0.0, (10.0, 0.0));
        assert_relative_eq!(b, 90.0, epsilon = 1e-9);
        assert_eq!(classify_encounter(b), EncounterType::CrossingGiveWay);
    }

    #[test]
    fn target_on_port_beam_is_crossing_stand_on() {
        let b = relative_bearing_deg((0.0, 0.0), 0.0, (-10.0, 0.0));
        assert_relative_eq!(b, -90.0, epsilon = 1e-9);
        assert_eq!(classify_encounter(b), EncounterType::CrossingStandOn);
    }

    #[test]
    fn target_dead_astern_is_overtaking() {
        let b = relative_bearing_deg((0.0, 0.0), 0.0, (0.0, -10.0));
        assert_relative_eq!(b.abs(), 180.0, epsilon = 1e-9);
        assert_eq!(classify_encounter(b), EncounterType::Overtaking);
    }

    #[test]
    fn threshold_boundaries_are_classified_consistently() {
        assert_eq!(classify_encounter(15.0), EncounterType::HeadOn);
        assert_eq!(classify_encounter(15.001), EncounterType::CrossingGiveWay);
        assert_eq!(classify_encounter(112.5), EncounterType::Overtaking);
        assert_eq!(classify_encounter(112.499), EncounterType::CrossingGiveWay);
        assert_eq!(classify_encounter(-112.5), EncounterType::Overtaking);
    }
}
