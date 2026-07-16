//! **Géotechnique — portance d'une fondation superficielle (Terzaghi)** :
//! capacité portante ultime `qult` d'une semelle filante par superposition des
//! contributions de cohésion, de surcharge et de poids volumique, facteurs de
//! portance `Nq` (Prandtl-Reissner) et `Nc`, puis contrainte admissible obtenue
//! par un coefficient de sécurité global.
//!
//! ```text
//! portance ultime   qult = c·Nc + q·Nq + 0,5·γ·B·Nγ
//! facteur Nq        Nq   = exp(π·tan φ) · tan²(π/4 + φ/2)
//! facteur Nc        Nc   = (Nq − 1) / tan φ
//! portance admissible  qadm = qult / F
//! ```
//!
//! `qult` capacité portante ultime (Pa), `c` = `cohesion` cohésion du sol (Pa),
//! `Nc` = `nc` facteur de portance de cohésion (sans dimension), `q` =
//! `surcharge` surcharge effective au niveau de la base (Pa), `Nq` = `nq`
//! facteur de portance de surcharge (sans dimension), `γ` = `unit_weight` poids
//! volumique du sol (N/m³), `B` = `foundation_width` largeur de la semelle (m),
//! `Nγ` = `ngamma` facteur de portance de poids volumique (sans dimension), `φ`
//! = `friction_angle_rad` angle de frottement interne (rad), `F` =
//! `safety_factor` coefficient de sécurité global (sans dimension, `> 1`),
//! `qadm` capacité portante admissible (Pa).
//!
//! **Convention** : SI strict — **N, m, Pa** (avec `1 Pa = 1 N/m²`). Les
//! contraintes (`c`, `q`, `qult`, `qadm`) ressortent en **pascals**, le poids
//! volumique en **newtons par mètre cube** et la largeur en **mètres** ; les
//! facteurs de portance `Nc`, `Nq`, `Nγ` sont **sans dimension**.
//!
//! **Limite honnête** : équation de Terzaghi pour une semelle **filante**, sol
//! **homogène**, charge **verticale centrée** ; les facteurs `Nc` et `Nq`
//! dérivent du **seul** angle de frottement, tandis que `Nγ` est **fourni par
//! l'appelant** (sa formule n'est pas unique selon les auteurs), de même que la
//! cohésion, le poids volumique et la surcharge. Les éventuelles résistances
//! caractéristiques du sol **et** les coefficients partiels de sécurité (`γc`,
//! `γs`, `γM`…, ou le coefficient global `F`) sont **fournis par l'appelant**
//! d'après l'Eurocode 7 et son Annexe Nationale ; aucune valeur « par défaut »
//! n'est inventée. Ce module **ne traite pas** les facteurs de forme, de
//! profondeur ou d'inclinaison de charge, ni le cas purement cohérent (`φ = 0`,
//! pour lequel `Nc` n'est pas défini par cette formule).

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Capacité portante ultime d'une semelle filante (Terzaghi)
/// `qult = c·Nc + q·Nq + 0,5·γ·B·Nγ` (Pa), avec `c`, `q` en Pa, `γ` en N/m³ et
/// `B` en m.
///
/// Panique si `cohesion < 0`, si `nc <= 0`, si `surcharge < 0`, si `nq <= 0`,
/// si `unit_weight <= 0`, si `foundation_width <= 0` ou si `ngamma < 0`.
pub fn geobear_terzaghi_ultimate(
    cohesion: f64,
    nc: f64,
    surcharge: f64,
    nq: f64,
    unit_weight: f64,
    foundation_width: f64,
    ngamma: f64,
) -> f64 {
    assert!(cohesion >= 0.0, "la cohésion c doit être ≥ 0");
    assert!(nc > 0.0, "le facteur Nc doit être strictement positif");
    assert!(surcharge >= 0.0, "la surcharge q doit être ≥ 0");
    assert!(nq > 0.0, "le facteur Nq doit être strictement positif");
    assert!(
        unit_weight > 0.0,
        "le poids volumique γ doit être strictement positif"
    );
    assert!(
        foundation_width > 0.0,
        "la largeur de semelle B doit être strictement positive"
    );
    assert!(ngamma >= 0.0, "le facteur Nγ doit être ≥ 0");
    cohesion * nc + surcharge * nq + 0.5 * unit_weight * foundation_width * ngamma
}

/// Facteur de portance de surcharge de Prandtl-Reissner
/// `Nq = exp(π·tan φ) · tan²(π/4 + φ/2)` (sans dimension), avec `φ` en rad.
///
/// Panique si `friction_angle_rad < 0` ou si `friction_angle_rad >= π/2` (la
/// tangente diverge).
pub fn geobear_bearing_factor_nq(friction_angle_rad: f64) -> f64 {
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < FRAC_PI_2,
        "l'angle de frottement φ doit être strictement inférieur à π/2"
    );
    (PI * friction_angle_rad.tan()).exp() * (FRAC_PI_4 + friction_angle_rad / 2.0).tan().powi(2)
}

/// Facteur de portance de cohésion `Nc = (Nq − 1) / tan φ` (sans dimension),
/// avec `φ` en rad ; le cas `φ → 0` n'est pas couvert (division par zéro).
///
/// Panique si `nq < 1`, si `friction_angle_rad <= 0` ou si
/// `friction_angle_rad >= π/2`.
pub fn geobear_bearing_factor_nc(nq: f64, friction_angle_rad: f64) -> f64 {
    assert!(nq >= 1.0, "le facteur Nq doit être ≥ 1");
    assert!(
        friction_angle_rad > 0.0,
        "l'angle de frottement φ doit être strictement positif (cas φ = 0 non couvert)"
    );
    assert!(
        friction_angle_rad < FRAC_PI_2,
        "l'angle de frottement φ doit être strictement inférieur à π/2"
    );
    (nq - 1.0) / friction_angle_rad.tan()
}

/// Capacité portante admissible `qadm = qult / F` (Pa), avec `qult` en Pa et `F`
/// coefficient de sécurité global.
///
/// Panique si `ultimate_bearing < 0` ou si `safety_factor <= 0` (division par
/// zéro).
pub fn geobear_allowable_bearing(ultimate_bearing: f64, safety_factor: f64) -> f64 {
    assert!(
        ultimate_bearing >= 0.0,
        "la portance ultime qult doit être ≥ 0"
    );
    assert!(
        safety_factor > 0.0,
        "le coefficient de sécurité F doit être strictement positif"
    );
    ultimate_bearing / safety_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_6;

    #[test]
    fn ultimate_is_superposition_of_three_terms() {
        // La portance ultime est une somme de trois contributions indépendantes.
        // On isole chacune en annulant les deux autres (cohésion, surcharge et
        // Nγ acceptent la valeur 0), puis on vérifie l'égalité de superposition.
        let (c, nc) = (12_000.0, 30.0);
        let (q, nq) = (25_000.0, 18.0);
        let (gamma, b, ng) = (19_000.0, 2.5, 20.0);
        let full = geobear_terzaghi_ultimate(c, nc, q, nq, gamma, b, ng);
        let only_c = geobear_terzaghi_ultimate(c, nc, 0.0, nq, gamma, b, 0.0);
        let only_q = geobear_terzaghi_ultimate(0.0, nc, q, nq, gamma, b, 0.0);
        let only_g = geobear_terzaghi_ultimate(0.0, nc, 0.0, nq, gamma, b, ng);
        assert_relative_eq!(only_c, c * nc, epsilon = 1e-9);
        assert_relative_eq!(only_q, q * nq, epsilon = 1e-9);
        assert_relative_eq!(only_g, 0.5 * gamma * b * ng, epsilon = 1e-9);
        assert_relative_eq!(full, only_c + only_q + only_g, max_relative = 1e-12);
    }

    #[test]
    fn nq_equals_one_for_frictionless_soil() {
        // Cas limite φ = 0 : tan φ = 0 ⇒ exp(0) = 1 et tan²(π/4) = 1, donc Nq = 1.
        let nq0 = geobear_bearing_factor_nq(0.0);
        assert_relative_eq!(nq0, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn bearing_factors_match_reference_at_30_degrees() {
        // Valeurs de référence pour φ = 30° (π/6) :
        //   Nq = exp(π·tan30°)·tan²(60°) = exp(1,813799…)·3 ≈ 18,40112
        //   Nc = (Nq − 1)/tan30° = 17,40112/0,577350… ≈ 30,13963
        let phi = FRAC_PI_6;
        let nq = geobear_bearing_factor_nq(phi);
        let nc = geobear_bearing_factor_nc(nq, phi);
        assert_relative_eq!(nq, 18.401_12, epsilon = 1e-3);
        assert_relative_eq!(nc, 30.139_63, epsilon = 1e-3);
    }

    #[test]
    fn allowable_bearing_is_reciprocal_of_safety_factor() {
        // Réciprocité : qadm · F = qult, et qadm(qult, 1) = qult.
        let qult = 1_200_000.0_f64;
        let f = 3.0_f64;
        let qadm = geobear_allowable_bearing(qult, f);
        assert_relative_eq!(qadm * f, qult, epsilon = 1e-6);
        assert_relative_eq!(geobear_allowable_bearing(qult, 1.0), qult, epsilon = 1e-9);
    }

    #[test]
    fn realistic_strip_footing_chain() {
        // Semelle filante réaliste : c = 10 kPa, φ = 30°, surcharge q = 27 kPa
        // (γ·Df = 18 kN/m³ × 1,5 m), γ = 18 kN/m³, B = 2 m, Nγ = 22,40 (fourni),
        // coefficient de sécurité F = 3.
        //   Nq   ≈ 18,40112 ; Nc ≈ 30,13963
        //   qult = 10 000·30,13963 + 27 000·18,40112 + 0,5·18 000·2·22,40
        //        = 301 396,3 + 496 830,3 + 403 200 ≈ 1 201 426,6 Pa
        //   qadm = qult / 3 ≈ 400 475,5 Pa
        let phi = FRAC_PI_6;
        let nq = geobear_bearing_factor_nq(phi);
        let nc = geobear_bearing_factor_nc(nq, phi);
        let qult = geobear_terzaghi_ultimate(10_000.0, nc, 27_000.0, nq, 18_000.0, 2.0, 22.40);
        let qadm = geobear_allowable_bearing(qult, 3.0);
        assert_relative_eq!(qult, 1_201_426.6, max_relative = 1e-3);
        assert_relative_eq!(qadm, 400_475.5, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "l'angle de frottement φ doit être strictement positif")]
    fn nc_rejects_zero_friction_angle() {
        // Cas φ = 0 non couvert : division par tan(0) = 0 interdite.
        geobear_bearing_factor_nc(1.0, 0.0);
    }
}
