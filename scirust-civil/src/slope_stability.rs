//! **Géotechnique — stabilité d'une pente infinie** : coefficient de sécurité
//! `F` d'un talus vis-à-vis d'une rupture plane parallèle à la surface, pour un
//! sol sableux sec, un sol cohérent sans nappe, ou un sol saturé siège d'un
//! écoulement parallèle à la pente.
//!
//! ```text
//! sable sec         F = tan φ / tan β
//! avec cohésion     F = (c + γ·z·cos²β·tan φ) / (γ·z·sin β·cos β)
//! avec écoulement   F = (c + (γsat − γw)·z·cos²β·tan φ) / (γsat·z·sin β·cos β)
//! stabilité         stable  ⇔  F > 1
//! ```
//!
//! `F` coefficient de sécurité (sans dimension), `φ` = `friction_angle_rad`
//! angle de frottement interne (rad), `β` = `slope_angle_rad` angle de la pente
//! sur l'horizontale (rad), `c` = `cohesion` cohésion effective du sol (Pa),
//! `γ` = `unit_weight` poids volumique du sol (N/m³), `γsat` =
//! `unit_weight_sat` poids volumique saturé (N/m³), `γw` = `unit_weight_water`
//! poids volumique de l'eau (N/m³), `z` = `depth` profondeur de la surface de
//! rupture mesurée verticalement (m).
//!
//! **Convention** : SI strict — **N, m, Pa** (avec `1 Pa = 1 N/m²`). La
//! cohésion est en **pascals**, les poids volumiques en **newtons par mètre
//! cube**, la profondeur en **mètres** et les angles en **radians** ; le
//! coefficient de sécurité est **sans dimension**.
//!
//! **Limite honnête** : modèle de **pente infinie** (rupture plane parallèle à
//! la surface, épaisseur de la couche instable faible devant la longueur du
//! talus, effets de bord négligés) ; la cohésion, l'angle de frottement et les
//! poids volumiques sont **fournis par l'appelant**. Le cas avec écoulement
//! suppose une **nappe parallèle à la pente** affleurant la surface. Ce module
//! **ne couvre pas** les ruptures circulaires ou non planes (méthode des
//! tranches, Bishop, Fellenius… à la charge de l'appelant). Les éventuelles
//! résistances caractéristiques du sol **et** les coefficients partiels de
//! sécurité (`γc`, `γs`, `γM`…, ou le seuil `F` retenu) relèvent de l'Eurocode 7
//! et de son Annexe Nationale, **fournis par l'appelant** ; aucune valeur « par
//! défaut » n'est inventée.

use core::f64::consts::FRAC_PI_2;

/// Coefficient de sécurité d'une pente infinie sableuse sèche (sans cohésion)
/// `F = tan φ / tan β` (sans dimension), avec `φ`, `β` en rad.
///
/// Panique si `friction_angle_rad < 0`, si `friction_angle_rad >= π/2`, si
/// `slope_angle_rad <= 0` (division par `tan β = 0`) ou si
/// `slope_angle_rad >= π/2`.
pub fn slope_infinite_dry_cohesionless(friction_angle_rad: f64, slope_angle_rad: f64) -> f64 {
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < FRAC_PI_2,
        "l'angle de frottement φ doit être strictement inférieur à π/2"
    );
    assert!(
        slope_angle_rad > 0.0,
        "l'angle de pente β doit être strictement positif (cas β = 0 non couvert)"
    );
    assert!(
        slope_angle_rad < FRAC_PI_2,
        "l'angle de pente β doit être strictement inférieur à π/2"
    );
    friction_angle_rad.tan() / slope_angle_rad.tan()
}

/// Coefficient de sécurité d'une pente infinie cohérente, sans nappe
/// `F = (c + γ·z·cos²β·tan φ) / (γ·z·sin β·cos β)` (sans dimension), avec `c`
/// en Pa, `γ` en N/m³, `z` en m et `φ`, `β` en rad.
///
/// Panique si `cohesion < 0`, si `unit_weight <= 0`, si `depth <= 0`, si
/// `slope_angle_rad <= 0`, si `slope_angle_rad >= π/2`, si
/// `friction_angle_rad < 0` ou si `friction_angle_rad >= π/2`.
pub fn slope_infinite_with_cohesion(
    cohesion: f64,
    unit_weight: f64,
    depth: f64,
    slope_angle_rad: f64,
    friction_angle_rad: f64,
) -> f64 {
    assert!(cohesion >= 0.0, "la cohésion c doit être ≥ 0");
    assert!(
        unit_weight > 0.0,
        "le poids volumique γ doit être strictement positif"
    );
    assert!(
        depth > 0.0,
        "la profondeur z doit être strictement positive"
    );
    assert!(
        slope_angle_rad > 0.0,
        "l'angle de pente β doit être strictement positif (cas β = 0 non couvert)"
    );
    assert!(
        slope_angle_rad < FRAC_PI_2,
        "l'angle de pente β doit être strictement inférieur à π/2"
    );
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < FRAC_PI_2,
        "l'angle de frottement φ doit être strictement inférieur à π/2"
    );
    let numerator =
        cohesion + unit_weight * depth * slope_angle_rad.cos().powi(2) * friction_angle_rad.tan();
    let denominator = unit_weight * depth * slope_angle_rad.sin() * slope_angle_rad.cos();
    numerator / denominator
}

/// Coefficient de sécurité d'une pente infinie saturée avec écoulement parallèle
/// à la pente
/// `F = (c + (γsat − γw)·z·cos²β·tan φ) / (γsat·z·sin β·cos β)` (sans
/// dimension), avec `c` en Pa, `γsat`, `γw` en N/m³, `z` en m et `φ`, `β` en
/// rad.
///
/// Panique si `cohesion < 0`, si `unit_weight_water <= 0`, si
/// `unit_weight_sat <= unit_weight_water` (poids volumique déjaugé négatif ou
/// nul), si `depth <= 0`, si `slope_angle_rad <= 0`, si
/// `slope_angle_rad >= π/2`, si `friction_angle_rad < 0` ou si
/// `friction_angle_rad >= π/2`.
pub fn slope_infinite_with_seepage(
    cohesion: f64,
    unit_weight_sat: f64,
    unit_weight_water: f64,
    depth: f64,
    slope_angle_rad: f64,
    friction_angle_rad: f64,
) -> f64 {
    assert!(cohesion >= 0.0, "la cohésion c doit être ≥ 0");
    assert!(
        unit_weight_water > 0.0,
        "le poids volumique de l'eau γw doit être strictement positif"
    );
    assert!(
        unit_weight_sat > unit_weight_water,
        "le poids volumique saturé γsat doit être strictement supérieur à γw"
    );
    assert!(
        depth > 0.0,
        "la profondeur z doit être strictement positive"
    );
    assert!(
        slope_angle_rad > 0.0,
        "l'angle de pente β doit être strictement positif (cas β = 0 non couvert)"
    );
    assert!(
        slope_angle_rad < FRAC_PI_2,
        "l'angle de pente β doit être strictement inférieur à π/2"
    );
    assert!(
        friction_angle_rad >= 0.0,
        "l'angle de frottement φ doit être ≥ 0"
    );
    assert!(
        friction_angle_rad < FRAC_PI_2,
        "l'angle de frottement φ doit être strictement inférieur à π/2"
    );
    let submerged_weight = unit_weight_sat - unit_weight_water;
    let numerator = cohesion
        + submerged_weight * depth * slope_angle_rad.cos().powi(2) * friction_angle_rad.tan();
    let denominator = unit_weight_sat * depth * slope_angle_rad.sin() * slope_angle_rad.cos();
    numerator / denominator
}

/// Critère de stabilité d'une pente : `stable ⇔ F > 1` (le talus est jugé
/// stable lorsque le coefficient de sécurité dépasse strictement l'unité).
///
/// Panique si `factor_of_safety < 0` (un coefficient de sécurité négatif n'a
/// pas de sens physique).
pub fn slope_is_stable(factor_of_safety: f64) -> bool {
    assert!(
        factor_of_safety >= 0.0,
        "le coefficient de sécurité F doit être ≥ 0"
    );
    factor_of_safety > 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_6;

    #[test]
    fn dry_cohesionless_satisfies_tangent_identity() {
        // Identité fondamentale : F·tan β = tan φ. On la vérifie sans nombre
        // magique, pour φ = 35° et β = 30°.
        let phi = 35.0_f64.to_radians();
        let beta = FRAC_PI_6;
        let f = slope_infinite_dry_cohesionless(phi, beta);
        assert_relative_eq!(f * beta.tan(), phi.tan(), epsilon = 1e-12);
    }

    #[test]
    fn dry_slope_at_friction_angle_is_limit_equilibrium() {
        // Cas limite β = φ : la pente est à l'équilibre limite, F = 1 exactement.
        let phi = 32.0_f64.to_radians();
        let f = slope_infinite_dry_cohesionless(phi, phi);
        assert_relative_eq!(f, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cohesion_reduces_to_dry_case_when_c_zero() {
        // Réciprocité entre modèles : sans cohésion, la formule cohérente se
        // réduit exactement au sable sec F = tan φ / tan β.
        let (gamma, z) = (18_000.0, 3.0);
        let phi = 28.0_f64.to_radians();
        let beta = 22.0_f64.to_radians();
        let with_c = slope_infinite_with_cohesion(0.0, gamma, z, beta, phi);
        let dry = slope_infinite_dry_cohesionless(phi, beta);
        assert_relative_eq!(with_c, dry, epsilon = 1e-12);
    }

    #[test]
    fn cohesion_realistic_case() {
        // Talus cohérent : c = 10 kPa, γ = 18 kN/m³, z = 3 m, β = φ = 30°.
        //   num = 10 000 + 18 000·3·cos²30°·tan30°
        //       = 10 000 + 54 000·0,75·0,5773503 ≈ 33 382,686
        //   den = 54 000·sin30°·cos30° = 54 000·0,5·0,8660254 ≈ 23 382,686
        //   F   = 33 382,686 / 23 382,686 ≈ 1,4276669
        let phi = FRAC_PI_6;
        let beta = FRAC_PI_6;
        let f = slope_infinite_with_cohesion(10_000.0, 18_000.0, 3.0, beta, phi);
        assert_relative_eq!(f, 1.427_666_9, max_relative = 1e-3);
        assert!(slope_is_stable(f));
    }

    #[test]
    fn seepage_reduces_effective_weight_ratio_when_c_zero() {
        // Sans cohésion, l'écoulement multiplie le cas sec par (γsat − γw)/γsat.
        // Cas chiffré : γsat = 20 kN/m³, γw = 9,81 kN/m³, z = 5 m, β = φ = 30°.
        //   F = ((20 000 − 9 810)/20 000)·tan30°/tan30° = 10 190/20 000 = 0,5095
        let (gsat, gw, z) = (20_000.0, 9_810.0, 5.0);
        let phi = FRAC_PI_6;
        let beta = FRAC_PI_6;
        let f = slope_infinite_with_seepage(0.0, gsat, gw, z, beta, phi);
        let ratio = (gsat - gw) / gsat;
        assert_relative_eq!(
            f,
            ratio * slope_infinite_dry_cohesionless(phi, beta),
            epsilon = 1e-12
        );
        assert_relative_eq!(f, 0.5095, epsilon = 1e-3);
    }

    #[test]
    fn seepage_realistic_case_is_unstable() {
        // Talus saturé : c = 6 kPa, γsat = 20 kN/m³, γw = 9,81 kN/m³, z = 5 m,
        // β = φ = 30°.
        //   num = 6 000 + 10 190·5·0,75·0,5773503 ≈ 28 061,997
        //   den = 20 000·5·sin30°·cos30° ≈ 43 301,270
        //   F   ≈ 0,6480641  →  F < 1  →  instable
        let phi = FRAC_PI_6;
        let beta = FRAC_PI_6;
        let f = slope_infinite_with_seepage(6_000.0, 20_000.0, 9_810.0, 5.0, beta, phi);
        assert_relative_eq!(f, 0.648_064_1, max_relative = 1e-3);
        assert!(!slope_is_stable(f));
    }

    #[test]
    #[should_panic(expected = "l'angle de pente β doit être strictement positif")]
    fn dry_rejects_zero_slope_angle() {
        // Cas β = 0 non couvert : division par tan(0) = 0 interdite.
        slope_infinite_dry_cohesionless(FRAC_PI_6, 0.0);
    }
}
