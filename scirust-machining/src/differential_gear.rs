//! Différentiel automobile **ouvert idéal** — relation cinématique entre les
//! vitesses des deux roues, la vitesse du boîtier (couronne) et le couple.
//!
//! ```text
//! somme des vitesses de roue     Σω = ω_L + ω_R = 2·ω_case
//! vitesse du boîtier (couronne)  ω_case = (ω_L + ω_R)/2
//! vitesse de l'autre roue        ω_other = 2·ω_case − ω_one
//! couple par roue (diff. ouvert) T_wheel = T_case/2
//! ```
//!
//! `ω_L`, `ω_R` vitesses angulaires des roues gauche et droite (rad/s),
//! `ω_case` vitesse du boîtier / couronne du différentiel (rad/s),
//! `ω_one`, `ω_other` vitesses de l'une et de l'autre roue (rad/s),
//! `T_case` couple d'entrée sur la couronne (N·m), `T_wheel` couple
//! transmis à une roue (N·m).
//!
//! **Convention** : unités SI (rad/s, N·m), vitesses algébriques de même
//! convention de signe. **Limite honnête** : différentiel **OUVERT IDÉAL SANS
//! FROTTEMENT** — le couple est réparti **également** entre les deux roues
//! quelles que soient leurs vitesses, pertes (engrènement, paliers, glissement)
//! **négligées**. En virage la somme des vitesses de roue reste égale à 2× la
//! vitesse de la couronne (une roue accélère, l'autre ralentit d'autant). Les
//! constantes physiques, propriétés matériaux et paramètres procédé sont
//! **fournies par l'appelant** ; aucune valeur « par défaut » n'est inventée.
//! Complète [`crate::gear_planetary_torque`] (répartition des couples).

/// Somme des vitesses des deux roues `Σω = ω_L + ω_R` (rad/s).
///
/// Identité du différentiel : cette somme vaut `2·ω_case`, donc elle est
/// **indépendante** de la répartition entre les roues (virage).
///
/// `left_wheel_speed`, `right_wheel_speed` en rad/s.
///
/// Panique si l'une des vitesses n'est pas finie.
pub fn diffgear_wheel_speed_sum(left_wheel_speed: f64, right_wheel_speed: f64) -> f64 {
    assert!(
        left_wheel_speed.is_finite() && right_wheel_speed.is_finite(),
        "les vitesses de roue doivent être finies"
    );
    left_wheel_speed + right_wheel_speed
}

/// Vitesse du boîtier (couronne) `ω_case = (ω_L + ω_R)/2` (rad/s).
///
/// Moyenne des vitesses des deux roues, invariante en virage.
///
/// `left_wheel_speed`, `right_wheel_speed` en rad/s.
///
/// Panique si l'une des vitesses n'est pas finie.
pub fn diffgear_case_speed(left_wheel_speed: f64, right_wheel_speed: f64) -> f64 {
    assert!(
        left_wheel_speed.is_finite() && right_wheel_speed.is_finite(),
        "les vitesses de roue doivent être finies"
    );
    (left_wheel_speed + right_wheel_speed) / 2.0
}

/// Vitesse de l'**autre** roue `ω_other = 2·ω_case − ω_one` (rad/s).
///
/// Cas limite : une roue immobilisée (`ω_one = 0`) ⇒ l'autre tourne à
/// `2·ω_case`, soit deux fois la vitesse du boîtier.
///
/// `case_speed`, `one_wheel_speed` en rad/s.
///
/// Panique si l'une des vitesses n'est pas finie.
pub fn diffgear_other_wheel_speed(case_speed: f64, one_wheel_speed: f64) -> f64 {
    assert!(
        case_speed.is_finite() && one_wheel_speed.is_finite(),
        "les vitesses doivent être finies"
    );
    2.0 * case_speed - one_wheel_speed
}

/// Couple transmis à **une** roue `T_wheel = T_case/2` (N·m).
///
/// Répartition **égale** du couple d'entrée (différentiel ouvert idéal),
/// indépendante des vitesses de roue.
///
/// `case_torque` en N·m.
///
/// Panique si `case_torque` n'est pas fini.
pub fn diffgear_wheel_torque(case_torque: f64) -> f64 {
    assert!(case_torque.is_finite(), "le couple d'entrée doit être fini");
    case_torque / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sum_equals_twice_case_speed() {
        // Identité fondamentale : ω_L + ω_R = 2·ω_case.
        let (wl, wr) = (30.0, 50.0);
        let sum = diffgear_wheel_speed_sum(wl, wr);
        let case = diffgear_case_speed(wl, wr);
        assert_relative_eq!(sum, 2.0 * case, epsilon = 1e-12);
        // Cas chiffré : (30+50)/2 = 40 rad/s.
        assert_relative_eq!(case, 40.0, epsilon = 1e-12);
        assert_relative_eq!(sum, 80.0, epsilon = 1e-12);
    }

    #[test]
    fn sum_invariant_in_turn() {
        // En virage la somme reste égale à 2·ω_case : on redistribue ±10 rad/s
        // autour de la moyenne, la couronne ne « voit » aucun changement.
        let straight = diffgear_wheel_speed_sum(40.0, 40.0);
        let cornering = diffgear_wheel_speed_sum(30.0, 50.0);
        assert_relative_eq!(straight, cornering, epsilon = 1e-12);
        // Les deux configurations partagent la même vitesse de boîtier.
        assert_relative_eq!(
            diffgear_case_speed(40.0, 40.0),
            diffgear_case_speed(30.0, 50.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn stopped_wheel_doubles_the_other() {
        // Roue immobilisée (ω_one = 0) ⇒ l'autre tourne à 2·ω_case.
        let case = 40.0;
        let other = diffgear_other_wheel_speed(case, 0.0);
        assert_relative_eq!(other, 2.0 * case, epsilon = 1e-12);
        assert_relative_eq!(other, 80.0, epsilon = 1e-12);
    }

    #[test]
    fn other_wheel_reciprocity() {
        // Réciprocité : reconstruire une roue à partir du boîtier et de l'autre
        // redonne exactement la vitesse d'origine.
        let (wl, wr) = (30.0, 50.0);
        let case = diffgear_case_speed(wl, wr);
        let recovered_left = diffgear_other_wheel_speed(case, wr);
        let recovered_right = diffgear_other_wheel_speed(case, wl);
        assert_relative_eq!(recovered_left, wl, epsilon = 1e-12);
        assert_relative_eq!(recovered_right, wr, epsilon = 1e-12);
    }

    #[test]
    fn wheel_torque_is_half_and_conserves_total() {
        // Répartition égale : chaque roue reçoit la moitié du couple d'entrée.
        let case_torque = 240.0;
        let wheel = diffgear_wheel_torque(case_torque);
        assert_relative_eq!(wheel, 120.0, epsilon = 1e-12);
        // Conservation : la somme des deux couples de roue redonne l'entrée.
        assert_relative_eq!(2.0 * wheel, case_torque, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "doit être fini")]
    fn non_finite_case_torque_panics() {
        diffgear_wheel_torque(f64::NAN);
    }
}
