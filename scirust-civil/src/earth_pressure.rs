//! **Géotechnique — poussée des terres** sur un mur de soutènement selon la
//! théorie de **Rankine** : coefficients de poussée active `Ka` et de butée
//! passive `Kp`, contrainte horizontale à une profondeur donnée, poussée
//! résultante par mètre de mur et terme de réduction dû à la cohésion.
//!
//! ```text
//! coefficient actif   Ka = tan²(π/4 − φ/2)
//! coefficient passif  Kp = tan²(π/4 + φ/2)
//! contrainte à z      σh = K · γ · z
//! poussée résultante  P  = ½ · K · γ · H²        (appliquée à H/3)
//! réduction cohésion  Δσ = 2 · c · √K            (terme −2c√Ka)
//! ```
//!
//! `φ` angle de frottement interne du sol (rad), `Ka` coefficient de poussée
//! active (sans dimension), `Kp` coefficient de butée passive (sans dimension),
//! `γ` poids volumique du sol (N/m³), `z` profondeur mesurée depuis la surface
//! du terre-plein (m), `σh` contrainte horizontale à la profondeur `z` (Pa),
//! `H` hauteur du mur (m), `P` poussée résultante par mètre linéaire de mur
//! (N/m, point d'application à `H/3` au-dessus de la base pour un diagramme
//! triangulaire), `c` cohésion du sol (Pa), `Δσ` diminution de contrainte due
//! à la cohésion (Pa).
//!
//! **Convention** : **SI strict** — longueurs en **m**, forces en **N**,
//! contraintes/pressions en **Pa** (1 Pa = 1 N/m²), poids volumique en **N/m³**,
//! angles en **radians**. Types `f64`.
//!
//! **Limite honnête** : **théorie de Rankine** — **mur vertical à parement
//! lisse**, **terre-plein horizontal**, **sol homogène**. Ne sont **pas**
//! traités : le **frottement mur-sol** (théorie de Coulomb, poussée inclinée),
//! la **surcharge** en tête (inclinée ou non) et la **nappe phréatique** (la
//! **poussée hydrostatique** de l'eau doit être **ajoutée par l'appelant**).
//! L'**angle de frottement interne** `φ`, le **poids volumique** `γ` et la
//! **cohésion** `c` sont des **paramètres géotechniques fournis par l'appelant**
//! (essais, Eurocode 7 / EN 1997 et son Annexe Nationale) — aucune valeur
//! « par défaut » n'est inventée. Les **coefficients partiels de sécurité**
//! (γφ, γγ, γc… de l'Eurocode) sont eux aussi à appliquer par l'appelant, sur
//! les paramètres ou sur les résultats, selon l'approche de calcul retenue.

use core::f64::consts::FRAC_PI_4;

/// Coefficient de poussée active de Rankine `Ka = tan²(π/4 − φ/2)` (sans
/// dimension), pour un sol pulvérulent, un terre-plein horizontal et un mur
/// vertical lisse.
///
/// `friction_angle_rad` = `φ` angle de frottement interne du sol (rad) ; renvoie
/// `Ka` (sans dimension). Décroît de 1 (à `φ = 0`) vers 0 quand `φ` augmente.
///
/// Panique si `friction_angle_rad < 0` ou si `friction_angle_rad >= π/2`
/// (tangente non définie et hors du domaine physique du frottement).
pub fn earthp_rankine_active_coefficient(friction_angle_rad: f64) -> f64 {
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < core::f64::consts::FRAC_PI_2,
        "l'angle de frottement φ doit être < π/2"
    );
    (FRAC_PI_4 - friction_angle_rad / 2.0).tan().powi(2)
}

/// Coefficient de butée passive de Rankine `Kp = tan²(π/4 + φ/2)` (sans
/// dimension), pour un sol pulvérulent, un terre-plein horizontal et un mur
/// vertical lisse.
///
/// `friction_angle_rad` = `φ` angle de frottement interne du sol (rad) ; renvoie
/// `Kp` (sans dimension). Croît de 1 (à `φ = 0`) vers l'infini quand `φ` tend
/// vers `π/2`. On a l'identité `Kp = 1 / Ka`.
///
/// Panique si `friction_angle_rad < 0` ou si `friction_angle_rad >= π/2`
/// (tangente non définie et hors du domaine physique du frottement).
pub fn earthp_rankine_passive_coefficient(friction_angle_rad: f64) -> f64 {
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < core::f64::consts::FRAC_PI_2,
        "l'angle de frottement φ doit être < π/2"
    );
    (FRAC_PI_4 + friction_angle_rad / 2.0).tan().powi(2)
}

/// Contrainte horizontale de poussée à la profondeur `z` : `σh = K · γ · z`
/// (Pa), variation linéaire (diagramme triangulaire) depuis la surface du
/// terre-plein.
///
/// `coefficient` = `K` coefficient de poussée (`Ka` actif ou `Kp` passif, sans
/// dimension), `unit_weight` = `γ` poids volumique du sol (N/m³), `depth` = `z`
/// profondeur depuis la surface (m) ; renvoie la contrainte horizontale (Pa, car
/// N/m³ · m = N/m² = Pa).
///
/// Panique si `coefficient < 0`, si `unit_weight < 0` ou si `depth < 0`
/// (grandeurs physiquement non négatives).
pub fn earthp_active_pressure_at_depth(coefficient: f64, unit_weight: f64, depth: f64) -> f64 {
    assert!(coefficient >= 0.0, "le coefficient K doit être ≥ 0");
    assert!(unit_weight >= 0.0, "le poids volumique γ doit être ≥ 0");
    assert!(depth >= 0.0, "la profondeur z doit être ≥ 0");
    coefficient * unit_weight * depth
}

/// Poussée résultante par mètre linéaire de mur `P = ½ · K · γ · H²` (N/m),
/// aire du diagramme triangulaire de contrainte, appliquée à `H/3` au-dessus de
/// la base.
///
/// `coefficient` = `K` coefficient de poussée (sans dimension), `unit_weight` =
/// `γ` poids volumique du sol (N/m³), `height` = `H` hauteur du mur (m) ;
/// renvoie la poussée résultante par mètre de mur (N/m, car N/m³ · m² = N/m).
///
/// Panique si `coefficient < 0`, si `unit_weight < 0` ou si `height < 0`
/// (grandeurs physiquement non négatives).
pub fn earthp_active_thrust(coefficient: f64, unit_weight: f64, height: f64) -> f64 {
    assert!(coefficient >= 0.0, "le coefficient K doit être ≥ 0");
    assert!(unit_weight >= 0.0, "le poids volumique γ doit être ≥ 0");
    assert!(height >= 0.0, "la hauteur H doit être ≥ 0");
    0.5 * coefficient * unit_weight * height * height
}

/// Réduction de contrainte de poussée due à la cohésion `Δσ = 2 · c · √K` (Pa),
/// terme `−2c√Ka` à soustraire de la contrainte de poussée active d'un sol
/// cohérent.
///
/// `coefficient` = `K` coefficient de poussée (sans dimension), `cohesion` = `c`
/// cohésion du sol (Pa) ; renvoie la diminution de contrainte (Pa). L'appelant
/// retranche cette valeur de `σh = Ka·γ·z` et, la poussée active ne pouvant être
/// négative, écrête à zéro au-dessus de la profondeur de fissuration de tension.
///
/// Panique si `coefficient < 0` (racine carrée non définie) ou si `cohesion < 0`
/// (cohésion physiquement non négative).
pub fn earthp_cohesion_reduction(coefficient: f64, cohesion: f64) -> f64 {
    assert!(
        coefficient >= 0.0,
        "le coefficient K doit être ≥ 0 (racine carrée)"
    );
    assert!(cohesion >= 0.0, "la cohésion c doit être ≥ 0");
    2.0 * cohesion * coefficient.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn active_passive_are_reciprocal() {
        // Identité de Rankine : Ka · Kp = 1, car tan(π/4−φ/2)·tan(π/4+φ/2) = 1
        // pour tout φ. Vérifié pour plusieurs angles.
        for &phi in &[0.1_f64, 0.3, 0.5, 0.7]
        {
            let ka = earthp_rankine_active_coefficient(phi);
            let kp = earthp_rankine_passive_coefficient(phi);
            assert_relative_eq!(ka * kp, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn coefficients_unity_at_zero_friction() {
        // À φ = 0 : Ka = tan²(π/4) = 1 et Kp = tan²(π/4) = 1 (état au repos
        // dégénéré du modèle de Rankine, cas limite d'un sol sans frottement).
        assert_relative_eq!(earthp_rankine_active_coefficient(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(
            earthp_rankine_passive_coefficient(0.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn active_coefficient_known_value_at_thirty_degrees() {
        // φ = 30° = π/6 : Ka = tan²(π/4−π/12) = tan²(π/6) = (1/√3)² = 1/3.
        let phi = core::f64::consts::PI / 6.0;
        let ka = earthp_rankine_active_coefficient(phi);
        assert_relative_eq!(ka, 1.0 / 3.0, epsilon = 1e-3);
        // Butée correspondante : Kp = tan²(π/3) = (√3)² = 3.
        let kp = earthp_rankine_passive_coefficient(phi);
        assert_relative_eq!(kp, 3.0, epsilon = 1e-3);
    }

    #[test]
    fn thrust_matches_triangular_area_and_scales() {
        // La poussée résultante est l'aire du triangle de contrainte :
        //   P = ½ · σh(H) · H = ½ · (K·γ·H) · H = ½·K·γ·H².
        let (k, gamma, h) = (1.0 / 3.0, 18_000.0_f64, 5.0);
        let sigma_base = earthp_active_pressure_at_depth(k, gamma, h);
        let thrust = earthp_active_thrust(k, gamma, h);
        assert_relative_eq!(thrust, 0.5 * sigma_base * h, epsilon = 1e-6);
        // La poussée croît comme le carré de la hauteur : ×2 sur H ⇒ ×4 sur P.
        let thrust_double = earthp_active_thrust(k, gamma, 2.0 * h);
        assert_relative_eq!(thrust_double, 4.0 * thrust, epsilon = 1e-6);
    }

    #[test]
    fn pressure_is_linear_in_depth() {
        // σh = K·γ·z est linéaire en z : nul en surface, doublé à profondeur
        // double.
        let (k, gamma) = (1.0 / 3.0, 18_000.0_f64);
        assert_relative_eq!(
            earthp_active_pressure_at_depth(k, gamma, 0.0),
            0.0,
            epsilon = 1e-12
        );
        let sigma3 = earthp_active_pressure_at_depth(k, gamma, 3.0);
        let sigma6 = earthp_active_pressure_at_depth(k, gamma, 6.0);
        assert_relative_eq!(sigma6, 2.0 * sigma3, epsilon = 1e-6);
    }

    #[test]
    fn realistic_granular_backfill_case() {
        // Mur vertical H = 5 m retenant un remblai pulvérulent :
        //   φ = 30° (→ Ka = 1/3), γ = 18 kN/m³ = 18 000 N/m³.
        //   σh(5 m) = Ka·γ·H = (1/3)·18 000·5      = 30 000 Pa  = 30 kPa
        //   P       = ½·Ka·γ·H² = ½·(1/3)·18 000·25 = 75 000 N/m = 75 kN/m
        // Terme de cohésion pour un sol c = 10 kPa = 10 000 Pa :
        //   Δσ = 2·c·√Ka = 2·10 000·√(1/3) = 20 000·0,5773503 = 11 547,005 Pa
        let phi = core::f64::consts::PI / 6.0;
        let ka = earthp_rankine_active_coefficient(phi);
        let gamma = 18_000.0_f64;

        let sigma_base = earthp_active_pressure_at_depth(ka, gamma, 5.0);
        assert_relative_eq!(sigma_base, 30_000.0, epsilon = 1e-1);

        let thrust = earthp_active_thrust(ka, gamma, 5.0);
        assert_relative_eq!(thrust, 75_000.0, epsilon = 1e-1);

        let delta = earthp_cohesion_reduction(ka, 10_000.0);
        assert_relative_eq!(delta, 11_547.005, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "l'angle de frottement φ doit être < π/2")]
    fn friction_angle_at_ninety_degrees_panics() {
        earthp_rankine_active_coefficient(core::f64::consts::FRAC_PI_2);
    }
}
