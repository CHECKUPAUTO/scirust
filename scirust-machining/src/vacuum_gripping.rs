//! Préhension par le **vide** — ventouses : effort théorique, charge admissible
//! avec coefficient de sécurité, diamètre requis et nombre de ventouses.
//!
//! ```text
//! effort théorique   F = Δp·A                  (A = (π/4)·D²)
//! charge admissible  F_adm = Δp·A/S            (S coefficient de sécurité)
//! diamètre requis    D = √(4·F·S/(π·Δp))       (une ventouse, F charge à tenir)
//! nombre de ventouses n = ⌈F_tot / F_adm⌉
//! ```
//!
//! `Δp` **dépression** (différence pression atmosphérique − pression de vide, Pa,
//! positive), `A` surface utile d'aspiration (m²), `D` diamètre de ventouse (m),
//! `S` coefficient de sécurité (≈ 1,5 charge horizontale ; 2 verticale ; ≥ 4 avec
//! mouvement/accélération), `F` charge à maintenir (N).
//!
//! **Convention** : SI ; `Δp` en Pa (une dépression de 60 % ≈ 0,6·101 325 ≈
//! 60 800 Pa). **Limite honnête** : effort **statique** normal à la surface, joint
//! **parfaitement étanche** sur une surface plane, lisse et rigide ; ne prend en
//! compte ni le cisaillement tangentiel (glissement), ni les fuites, ni les
//! efforts d'accélération — ceux-ci sont couverts par le coefficient `S` fourni
//! par l'appelant.

use core::f64::consts::PI;

/// Effort théorique de préhension `F = Δp·A`.
///
/// Panique si `vacuum_pressure < 0` ou `area <= 0`.
pub fn theoretical_holding_force(vacuum_pressure: f64, area: f64) -> f64 {
    assert!(
        vacuum_pressure >= 0.0 && area > 0.0,
        "Δp ≥ 0 et A > 0 requis"
    );
    vacuum_pressure * area
}

/// Charge **admissible** d'une ventouse `F_adm = Δp·A/S`.
///
/// Panique si `vacuum_pressure < 0`, `area <= 0` ou `safety_factor < 1`.
pub fn working_load(vacuum_pressure: f64, area: f64, safety_factor: f64) -> f64 {
    assert!(
        safety_factor >= 1.0,
        "le coefficient de sécurité doit être ≥ 1"
    );
    theoretical_holding_force(vacuum_pressure, area) / safety_factor
}

/// Diamètre de ventouse requis pour tenir `force` `D = √(4·F·S/(π·Δp))`.
///
/// Panique si `force < 0`, `vacuum_pressure <= 0` ou `safety_factor < 1`.
pub fn required_diameter(force: f64, vacuum_pressure: f64, safety_factor: f64) -> f64 {
    assert!(
        force >= 0.0 && vacuum_pressure > 0.0 && safety_factor >= 1.0,
        "F ≥ 0, Δp > 0 et S ≥ 1 requis"
    );
    (4.0 * force * safety_factor / (PI * vacuum_pressure)).sqrt()
}

/// Nombre de ventouses `n = ⌈F_tot / F_adm⌉` pour une charge admissible unitaire.
///
/// Panique si `per_cup_working_load <= 0` ou `total_force < 0`.
pub fn number_of_cups(total_force: f64, per_cup_working_load: f64) -> u32 {
    assert!(
        per_cup_working_load > 0.0 && total_force >= 0.0,
        "charge unitaire > 0 et charge totale ≥ 0 requises"
    );
    (total_force / per_cup_working_load).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn holding_force_is_pressure_times_area() {
        // Ventouse Ø40 mm, dépression 60 kPa → F = 60e3·π/4·0,04² ≈ 75,4 N.
        let a = PI / 4.0 * 0.040 * 0.040;
        let f = theoretical_holding_force(60e3, a);
        assert_relative_eq!(f, 60e3 * a, epsilon = 1e-9);
        assert!(f > 75.0 && f < 76.0);
    }

    #[test]
    fn safety_factor_reduces_working_load() {
        let a = PI / 4.0 * 0.040 * 0.040;
        assert_relative_eq!(
            working_load(60e3, a, 2.0),
            theoretical_holding_force(60e3, a) / 2.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn required_diameter_holds_the_load() {
        // Le diamètre calculé doit fournir exactement F_adm = force à S donné.
        let d = required_diameter(50.0, 60e3, 2.0);
        let a = PI / 4.0 * d * d;
        assert_relative_eq!(working_load(60e3, a, 2.0), 50.0, max_relative = 1e-9);
    }

    #[test]
    fn number_of_cups_rounds_up() {
        // 100 N à tenir, 30 N par ventouse → ⌈3,33⌉ = 4 ventouses.
        assert_eq!(number_of_cups(100.0, 30.0), 4);
        // Charge nulle → 0 ventouse.
        assert_eq!(number_of_cups(0.0, 30.0), 0);
    }

    #[test]
    #[should_panic(expected = "coefficient de sécurité")]
    fn safety_factor_below_one_panics() {
        working_load(60e3, 1e-3, 0.5);
    }
}
