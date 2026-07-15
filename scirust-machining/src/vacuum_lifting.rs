//! **Levage par ventouse** (préhension sous vide) : force de préhension
//! théorique d'une ventouse étanche, force effective avec coefficient de
//! sécurité, aire de ventouse requise et aire d'un disque à partir de son
//! diamètre.
//!
//! ```text
//! force théorique     F = Δp · A
//! force effective     F_eff = Δp · A / s
//! aire requise        A = m·g_load · s / Δp        (charge = poids en N)
//! aire du disque      A = π · d² / 4
//! ```
//!
//! `Δp` dépression = **différence** entre la pression atmosphérique et la
//! pression sous la ventouse (Pa, strictement positive), `A` aire de contact
//! utile de la ventouse (m²), `F` force de préhension verticale (N), `s`
//! coefficient de sécurité adimensionnel (`s ≥ 1`), `load` charge à lever
//! exprimée en **force** (N, c.-à-d. `m·g`), `d` diamètre du disque de contact
//! (m).
//!
//! **Convention** : unités SI (Pa, m², N), dépression `Δp` **relative** à
//! l'atmosphère (pas une pression absolue).
//! **Limite honnête** : ventouse supposée **parfaitement étanche** et levage
//! **vertical statique**. La dépression `Δp` (fonction de la pompe/du venturi et
//! des fuites) et le coefficient de sécurité `s` sont des données **fournies par
//! l'appelant** ; aucune valeur « par défaut » n'est supposée. En cisaillement ou
//! frottement latéral, les efforts tangentiels imposent des coefficients de
//! sécurité **plus élevés**, à la charge de l'appelant.

use core::f64::consts::PI;

/// Force de préhension **théorique** d'une ventouse `F = Δp · A`
/// (dépression relative à l'atmosphère fois aire utile).
///
/// Panique si `vacuum_pressure <= 0` ou `cup_area <= 0`.
pub fn vacuum_lift_force(vacuum_pressure: f64, cup_area: f64) -> f64 {
    assert!(
        vacuum_pressure > 0.0 && cup_area > 0.0,
        "la dépression Δp et l'aire A doivent être strictement positives"
    );
    vacuum_pressure * cup_area
}

/// Force de préhension **effective** avec coefficient de sécurité
/// `F_eff = Δp · A / s` (réduit la force théorique de [`vacuum_lift_force`]).
///
/// Panique si `vacuum_pressure <= 0`, `cup_area <= 0` ou `safety_factor < 1`.
pub fn vacuum_effective_lift(vacuum_pressure: f64, cup_area: f64, safety_factor: f64) -> f64 {
    assert!(
        vacuum_pressure > 0.0 && cup_area > 0.0,
        "la dépression Δp et l'aire A doivent être strictement positives"
    );
    assert!(
        safety_factor >= 1.0,
        "le coefficient de sécurité s doit être supérieur ou égal à 1"
    );
    vacuum_pressure * cup_area / safety_factor
}

/// Aire de ventouse **requise** pour lever une charge donnée
/// `A = load · s / Δp` (réciproque de [`vacuum_effective_lift`] ;
/// `load` est un **poids en newtons**).
///
/// Panique si `load <= 0`, `vacuum_pressure <= 0` ou `safety_factor < 1`.
pub fn vacuum_required_area(load: f64, vacuum_pressure: f64, safety_factor: f64) -> f64 {
    assert!(
        load > 0.0 && vacuum_pressure > 0.0,
        "la charge load (N) et la dépression Δp doivent être strictement positives"
    );
    assert!(
        safety_factor >= 1.0,
        "le coefficient de sécurité s doit être supérieur ou égal à 1"
    );
    load * safety_factor / vacuum_pressure
}

/// Aire de contact d'une ventouse circulaire à partir de son diamètre
/// `A = π · d² / 4`.
///
/// Panique si `diameter <= 0`.
pub fn vacuum_cup_area_from_diameter(diameter: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre d doit être strictement positif"
    );
    PI * diameter * diameter / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn theoretical_force_matches_hand_calc() {
        // Δp = 50 000 Pa (0,5 bar de dépression), A = 0,01 m² :
        // F = 50 000 · 0,01 = 500 N.
        let f = vacuum_lift_force(50_000.0, 0.01);
        assert_relative_eq!(f, 500.0, epsilon = 1e-9);
    }

    #[test]
    fn force_proportional_to_area() {
        // F ∝ A : tripler l'aire triple la force.
        let base = vacuum_lift_force(50_000.0, 0.01);
        let triple = vacuum_lift_force(50_000.0, 0.03);
        assert_relative_eq!(triple, 3.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn effective_lift_is_theoretical_over_safety() {
        // F_eff = F / s : avec s = 2, la force effective vaut la moitié.
        let theo = vacuum_lift_force(50_000.0, 0.01);
        let eff = vacuum_effective_lift(50_000.0, 0.01, 2.0);
        assert_relative_eq!(eff, theo / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn required_area_and_effective_lift_are_reciprocal() {
        // L'aire dimensionnée pour lever une charge donnée doit,
        // réinjectée, restituer exactement cette charge en force effective.
        let (load, dp, s) = (250.0_f64, 50_000.0, 2.0);
        let area = vacuum_required_area(load, dp, s);
        let eff = vacuum_effective_lift(dp, area, s);
        assert_relative_eq!(eff, load, epsilon = 1e-9);
    }

    #[test]
    fn required_area_matches_hand_calc() {
        // load = 250 N, Δp = 50 000 Pa, s = 2 :
        // A = 250 · 2 / 50 000 = 0,01 m².
        let area = vacuum_required_area(250.0, 50_000.0, 2.0);
        assert_relative_eq!(area, 0.01, epsilon = 1e-12);
    }

    #[test]
    fn disk_area_matches_pi_d2_over_4() {
        // d = 0,1 m : A = π·0,01/4 = 0,007853981633974483 m².
        let area = vacuum_cup_area_from_diameter(0.1);
        assert_relative_eq!(area, PI * 0.01 / 4.0, epsilon = 1e-15);
        assert_relative_eq!(area, 0.007_853_981_633_974_483, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "coefficient de sécurité")]
    fn effective_lift_rejects_safety_below_one() {
        let _ = vacuum_effective_lift(50_000.0, 0.01, 0.5);
    }
}
