//! Point d'approche le plus proche (CPA) et temps jusqu'au CPA (TCPA) —
//! le calcul standard d'évaluation du risque de collision en trajectoires
//! rectilignes à vitesse constante (par ex. Lenart, "Collision Threat
//! Parameters for a New Radar Display and Plot Technique", Journal of
//! Navigation 36(3), 1983 ; méthode ARPA classique).
//!
//! `r` = position relative (cible − propre), `v` = vitesse relative
//! (cible − propre) ; trajectoires supposées rectilignes à vitesse
//! constante, `d(t) = r + v·t`, minimisée par
//! `TCPA = -(r·v)/(v·v)`, `CPA = |r + v·TCPA|`.
//!
//! Vérifié contre un exemple travaillé indépendant (calcul numérique
//! direct, numpy) : navire A en (1,1) nm cap 045°T à 6 nd, navire B en
//! (9,8) nm cap 270°T à 6 nd → TCPA≈54.5 min, CPA≈3.41 nm.

/// Résultat de l'évaluation du risque de collision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpaTcpa {
    /// Temps jusqu'au point d'approche le plus proche, mêmes unités de
    /// temps que celles utilisées pour les vitesses fournies (par ex.
    /// heures si les vitesses sont en nd et les positions en nm).
    /// Négatif si le CPA est déjà passé.
    pub tcpa: f64,
    /// Distance au point d'approche le plus proche (même unité que les positions).
    pub cpa: f64,
}

/// Vecteur vitesse (est, nord) depuis un cap compas (degrés, `0°` = nord,
/// horaire) et une vitesse scalaire.
pub fn velocity_from_heading(heading_deg: f64, speed: f64) -> (f64, f64) {
    let h = heading_deg.to_radians();
    (speed * h.sin(), speed * h.cos())
}

/// Calcule CPA/TCPA entre deux mobiles à trajectoire rectiligne uniforme.
/// Positions et vitesses en (est, nord).
///
/// Si les deux mobiles ont une vitesse relative nulle (`v·v` négligeable
/// — mêmes route et vitesse), la distance ne varie jamais : `tcpa = 0.0`
/// et `cpa` est la distance actuelle.
pub fn cpa_tcpa(
    own_pos: (f64, f64),
    own_vel: (f64, f64),
    target_pos: (f64, f64),
    target_vel: (f64, f64),
) -> CpaTcpa {
    let r = (target_pos.0 - own_pos.0, target_pos.1 - own_pos.1);
    let v = (target_vel.0 - own_vel.0, target_vel.1 - own_vel.1);
    let vv = v.0 * v.0 + v.1 * v.1;
    if vv < 1e-12
    {
        return CpaTcpa {
            tcpa: 0.0,
            cpa: (r.0 * r.0 + r.1 * r.1).sqrt(),
        };
    }
    let rv = r.0 * v.0 + r.1 * v.1;
    let tcpa = -rv / vv;
    let (cx, cy) = (r.0 + v.0 * tcpa, r.1 + v.1 * tcpa);
    CpaTcpa {
        tcpa,
        cpa: (cx * cx + cy * cy).sqrt(),
    }
}

/// Classification simple du risque à partir de seuils DCPA/TCPA fournis
/// par l'appelant (la pratique ARPA usuelle — pas des constantes
/// réglementaires universelles) : risque si le CPA est à venir
/// (`0 < tcpa <= tcpa_max`) et suffisamment proche (`cpa <=
/// cpa_threshold`).
pub fn is_collision_risk(result: CpaTcpa, cpa_threshold: f64, tcpa_max: f64) -> bool {
    result.tcpa > 0.0 && result.tcpa <= tcpa_max && result.cpa <= cpa_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn matches_the_two_ship_worked_example() {
        let own_pos = (1.0, 1.0);
        let own_vel = velocity_from_heading(45.0, 6.0);
        let target_pos = (9.0, 8.0);
        let target_vel = velocity_from_heading(270.0, 6.0);

        let result = cpa_tcpa(own_pos, own_vel, target_pos, target_vel);
        assert_relative_eq!(result.tcpa * 60.0, 54.497_474_68, epsilon = 1e-4);
        assert_relative_eq!(result.cpa, 3.405_689_269, epsilon = 1e-6);
    }

    #[test]
    fn identical_course_and_speed_never_closes() {
        let vel = velocity_from_heading(90.0, 10.0);
        let result = cpa_tcpa((0.0, 0.0), vel, (5.0, 5.0), vel);
        assert_relative_eq!(result.tcpa, 0.0, epsilon = 1e-12);
        assert_relative_eq!(result.cpa, (50.0_f64).sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn head_on_closing_ships_meet_at_the_midpoint() {
        // Two ships 10 nm apart on the same line, closing head-on at equal speed.
        let own_pos = (0.0, 0.0);
        let own_vel = velocity_from_heading(0.0, 5.0); // heading north
        let target_pos = (0.0, 10.0);
        let target_vel = velocity_from_heading(180.0, 5.0); // heading south
        let result = cpa_tcpa(own_pos, own_vel, target_pos, target_vel);
        assert_relative_eq!(result.tcpa, 1.0, epsilon = 1e-9); // 10nm gap closing at 10kn combined
        assert_relative_eq!(result.cpa, 0.0, epsilon = 1e-9); // collision course
    }

    #[test]
    fn is_collision_risk_respects_both_thresholds() {
        let risky = CpaTcpa {
            tcpa: 0.5,
            cpa: 0.3,
        };
        let too_far = CpaTcpa {
            tcpa: 0.5,
            cpa: 5.0,
        };
        let too_late = CpaTcpa {
            tcpa: -0.1,
            cpa: 0.1,
        };
        assert!(is_collision_risk(risky, 1.0, 1.0));
        assert!(!is_collision_risk(too_far, 1.0, 1.0));
        assert!(!is_collision_risk(too_late, 1.0, 1.0));
    }
}
