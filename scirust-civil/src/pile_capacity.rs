//! **Géotechnique — capacité portante d'un pieu isolé** : résistance de pointe
//! `Qp` (sol pulvérulent), frottement latéral `Qs`, capacité ultime `Qu`
//! (déduction faite du poids propre du pieu) et capacité admissible `Qa`
//! obtenue par un coefficient de sécurité global.
//!
//! ```text
//! résistance de pointe     Qp = Nq · σ'v · Ap
//! frottement latéral       Qs = fs · P · L
//! capacité ultime          Qu = Qp + Qs − W
//! capacité admissible      Qa = Qu / F
//! ```
//!
//! `Qp` résistance de pointe (N), `Nq` = `bearing_capacity_factor` facteur de
//! portance de pointe (sans dimension), `σ'v` = `effective_stress_at_tip`
//! contrainte verticale effective au niveau de la pointe (Pa), `Ap` =
//! `pile_tip_area` aire de la section de pointe (m²) ; `Qs` frottement latéral
//! (N), `fs` = `average_skin_friction` frottement latéral unitaire moyen (Pa),
//! `P` = `shaft_perimeter` périmètre du fût (m), `L` = `embedded_length`
//! longueur du fût en contact avec le sol (m) ; `Qu` capacité ultime (N), `W` =
//! `pile_weight` poids propre du pieu (N), `Qa` capacité admissible (N), `F` =
//! `safety_factor` coefficient de sécurité global (sans dimension, `> 0`).
//!
//! **Convention** : SI strict — **N, m, Pa** (avec `1 Pa = 1 N/m²`). Les
//! contraintes (`σ'v`, `fs`) sont en **pascals**, les longueurs et périmètres
//! en **mètres**, les aires en **mètres carrés**, les efforts (`Qp`, `Qs`,
//! `Qu`, `W`, `Qa`) en **newtons** ; le facteur `Nq` et le coefficient `F` sont
//! **sans dimension**.
//!
//! **Limite honnête** : pieu **isolé** ; la résistance de pointe suppose un sol
//! **pulvérulent** avec un facteur `Nq` **fourni par l'appelant** (selon l'angle
//! de frottement `φ` et le type de pieu — foré, battu…), le frottement latéral
//! unitaire moyen `fs` est **fourni par l'appelant** (méthode `α` en contrainte
//! totale ou `β` en contrainte effective, à sa charge), de même que la
//! contrainte effective `σ'v` au niveau de la pointe et le poids propre `W`. Ce
//! module **ne traite ni l'effet de groupe ni le tassement**. Les résistances
//! caractéristiques du sol **et** les coefficients partiels (ou le coefficient
//! de sécurité global `F`) sont **fournis par l'appelant** d'après l'Eurocode 7
//! et son Annexe Nationale ; aucune valeur « par défaut » n'est inventée.

/// Résistance de pointe d'un pieu en sol pulvérulent
/// `Qp = Nq · σ'v · Ap` (N), avec `σ'v` en Pa et `Ap` en m².
///
/// Panique si `bearing_capacity_factor <= 0`, si `effective_stress_at_tip < 0`
/// ou si `pile_tip_area <= 0`.
pub fn pile_end_bearing(
    bearing_capacity_factor: f64,
    effective_stress_at_tip: f64,
    pile_tip_area: f64,
) -> f64 {
    assert!(
        bearing_capacity_factor > 0.0,
        "le facteur de portance de pointe Nq doit être strictement positif"
    );
    assert!(
        effective_stress_at_tip >= 0.0,
        "la contrainte effective à la pointe σ'v doit être ≥ 0"
    );
    assert!(
        pile_tip_area > 0.0,
        "l'aire de pointe Ap doit être strictement positive"
    );
    bearing_capacity_factor * effective_stress_at_tip * pile_tip_area
}

/// Frottement latéral mobilisé le long du fût
/// `Qs = fs · P · L` (N), avec `fs` en Pa, `P` en m et `L` en m.
///
/// Panique si `average_skin_friction < 0`, si `shaft_perimeter <= 0` ou si
/// `embedded_length <= 0`.
pub fn pile_shaft_friction(
    average_skin_friction: f64,
    shaft_perimeter: f64,
    embedded_length: f64,
) -> f64 {
    assert!(
        average_skin_friction >= 0.0,
        "le frottement latéral unitaire moyen fs doit être ≥ 0"
    );
    assert!(
        shaft_perimeter > 0.0,
        "le périmètre du fût P doit être strictement positif"
    );
    assert!(
        embedded_length > 0.0,
        "la longueur d'ancrage L doit être strictement positive"
    );
    average_skin_friction * shaft_perimeter * embedded_length
}

/// Capacité portante ultime d'un pieu isolé
/// `Qu = Qp + Qs − W` (N) : la résistance de pointe et le frottement latéral
/// mobilisables, déduction faite du poids propre du pieu.
///
/// Panique si `end_bearing < 0`, si `shaft_friction < 0` ou si
/// `pile_weight < 0`.
pub fn pile_ultimate_capacity(end_bearing: f64, shaft_friction: f64, pile_weight: f64) -> f64 {
    assert!(
        end_bearing >= 0.0,
        "la résistance de pointe Qp doit être ≥ 0"
    );
    assert!(
        shaft_friction >= 0.0,
        "le frottement latéral Qs doit être ≥ 0"
    );
    assert!(
        pile_weight >= 0.0,
        "le poids propre du pieu W doit être ≥ 0"
    );
    end_bearing + shaft_friction - pile_weight
}

/// Capacité portante admissible `Qa = Qu / F` (N), avec `Qu` en N et `F`
/// coefficient de sécurité global.
///
/// Panique si `ultimate_capacity < 0` ou si `safety_factor <= 0` (division par
/// zéro).
pub fn pile_allowable_capacity(ultimate_capacity: f64, safety_factor: f64) -> f64 {
    assert!(
        ultimate_capacity >= 0.0,
        "la capacité ultime Qu doit être ≥ 0"
    );
    assert!(
        safety_factor > 0.0,
        "le coefficient de sécurité F doit être strictement positif"
    );
    ultimate_capacity / safety_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn end_bearing_is_product_of_three_factors() {
        // Qp est un simple produit : doubler σ'v double Qp (proportionnalité),
        // et le résultat vaut Nq·σ'v·Ap.
        let (nq, sigma, ap) = (40.0, 200_000.0, 0.20);
        let qp = pile_end_bearing(nq, sigma, ap);
        assert_relative_eq!(qp, nq * sigma * ap, epsilon = 1e-9);
        let qp_double = pile_end_bearing(nq, 2.0 * sigma, ap);
        assert_relative_eq!(qp_double, 2.0 * qp, epsilon = 1e-9);
    }

    #[test]
    fn shaft_friction_is_product_and_scales_with_length() {
        // Qs = fs·P·L : proportionnel à la longueur d'ancrage.
        let (fs, p) = (50_000.0, 1.57);
        let qs_10 = pile_shaft_friction(fs, p, 10.0);
        let qs_20 = pile_shaft_friction(fs, p, 20.0);
        assert_relative_eq!(qs_10, fs * p * 10.0, epsilon = 1e-9);
        assert_relative_eq!(qs_20, 2.0 * qs_10, epsilon = 1e-9);
    }

    #[test]
    fn ultimate_adds_bearing_and_friction_minus_weight() {
        // Qu = Qp + Qs − W : additivité, avec déduction stricte du poids propre.
        let (qp, qs, w) = (1_500_000.0, 1_100_000.0, 70_000.0);
        let qu = pile_ultimate_capacity(qp, qs, w);
        assert_relative_eq!(qu, qp + qs - w, epsilon = 1e-6);
        // Un poids propre nul redonne la somme pointe + frottement.
        assert_relative_eq!(pile_ultimate_capacity(qp, qs, 0.0), qp + qs, epsilon = 1e-6);
    }

    #[test]
    fn allowable_is_reciprocal_of_safety_factor() {
        // Réciprocité : Qa · F = Qu, et Qa(Qu, 1) = Qu.
        let qu = 2_600_000.0_f64;
        let f = 2.5_f64;
        let qa = pile_allowable_capacity(qu, f);
        assert_relative_eq!(qa * f, qu, epsilon = 1e-6);
        assert_relative_eq!(pile_allowable_capacity(qu, 1.0), qu, epsilon = 1e-9);
    }

    #[test]
    fn realistic_bored_pile_chain() {
        // Pieu foré : diamètre d = 0,5 m, ancrage L = 15 m.
        //   Ap = π·d²/4 = π·0,0625 = 0,196349540849 m²
        //   P  = π·d    = π·0,5    = 1,570796326795 m
        // Sol : Nq = 40 (fourni), σ'v = 200 kPa à la pointe,
        //       fs = 50 kPa (méthode β, fournie), W = 70 kN (fourni), F = 2,5.
        //   Qp = 40·200 000·0,196349540849 = 1 570 796,32679 N
        //   Qs = 50 000·1,570796326795·15  = 1 178 097,24510 N
        //   Qu = Qp + Qs − 70 000          = 2 678 893,57189 N
        //   Qa = Qu / 2,5                  = 1 071 557,42876 N
        let d = 0.5_f64;
        let ap = PI / 4.0 * d * d;
        let perimeter = PI * d;
        let qp = pile_end_bearing(40.0, 200_000.0, ap);
        let qs = pile_shaft_friction(50_000.0, perimeter, 15.0);
        let qu = pile_ultimate_capacity(qp, qs, 70_000.0);
        let qa = pile_allowable_capacity(qu, 2.5);
        assert_relative_eq!(qp, 1_570_796.326_79, max_relative = 1e-3);
        assert_relative_eq!(qs, 1_178_097.245_10, max_relative = 1e-3);
        assert_relative_eq!(qu, 2_678_893.571_89, max_relative = 1e-3);
        assert_relative_eq!(qa, 1_071_557.428_76, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient de sécurité F doit être strictement positif")]
    fn allowable_rejects_zero_safety_factor() {
        // F = 0 interdit : division par zéro.
        pile_allowable_capacity(1_000_000.0, 0.0);
    }
}
