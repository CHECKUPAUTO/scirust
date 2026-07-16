//! **Béton armé — torsion (Eurocode 2, ELU, modèle du treillis spatial)** :
//! épaisseur équivalente de paroi mince, flux de cisaillement de torsion, aire
//! d'armatures longitudinales et densité d'étriers d'après le modèle du treillis
//! spatial à paroi mince fermée.
//!
//! ```text
//! épaisseur paroi mince     tef     = Ak / uk               (souvent tef = A/u)
//! flux de cisaillement      τ·tef   = TEd / (2 · Ak)
//! armatures longitudinales  ΣAsl    = TEd · uk · cot θ / (2 · Ak · fyd)
//! étriers (par espacement)  Asw/s   = TEd · tan θ / (2 · Ak · fywd)
//! ```
//!
//! `tef` épaisseur équivalente de la paroi mince (mm), `Ak` aire enveloppée par
//! la ligne moyenne de la paroi mince (mm²), `uk` périmètre de cette ligne
//! moyenne (mm), `A` aire de la section pleine et `u` son périmètre extérieur
//! (mm² et mm), `TEd` moment de torsion de calcul (N·mm), `τ·tef` flux de
//! cisaillement de torsion, c'est-à-dire une force par unité de longueur (N/mm),
//! `ΣAsl` aire totale des armatures longitudinales de torsion (mm²), `Asw/s`
//! section d'un cours d'étriers fermés par unité de longueur (mm²/mm), `fyd` et
//! `fywd` limites d'élasticité de calcul des armatures longitudinales et des
//! étriers (MPa), `θ` angle des bielles de béton (rad, `cot θ = 1/tan θ`).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`), donc les moments de
//! torsion sont en **N·mm** et les résistances en MPa ; angles en **radians**
//! pour la trigonométrie.
//! **Limite honnête** : modèle du **treillis spatial à paroi mince fermée** de
//! l'Eurocode 2. L'aire enveloppe `Ak` et le périmètre `uk` (ou l'aire `A` et le
//! périmètre `u` de la section pleine) sont **fournis par l'appelant** d'après la
//! géométrie ; l'angle des bielles `θ` est **fourni** et doit respecter
//! `1 ≤ cot θ ≤ 2,5` (soit `θ ∈ [21,8° ; 45°]`). Les résistances caractéristiques
//! (`fyk`…) **et** les coefficients partiels de sécurité (`γs`, `γc`…) — donc
//! `fyd`, `fywd` — sont **fournis par l'appelant** d'après l'Eurocode 2 et son
//! Annexe Nationale ; aucune valeur « par défaut » n'est inventée. L'**interaction
//! torsion–effort tranchant** (`TEd/TRd,max + VEd/VRd,max ≤ 1`), l'écrasement des
//! bielles et les dispositions constructives restent à vérifier **séparément** par
//! l'appelant.

use core::f64::consts::FRAC_PI_2;

/// Épaisseur équivalente de paroi mince `tef = A / u` (mm), rapport de l'aire de
/// la section pleine `A` (mm²) à son périmètre extérieur `u` (mm).
///
/// Panique si `area <= 0` ou si `perimeter <= 0` (division par zéro).
pub fn rctor_thin_wall_thickness(area: f64, perimeter: f64) -> f64 {
    assert!(
        area > 0.0,
        "l'aire de la section A doit être strictement positive"
    );
    assert!(
        perimeter > 0.0,
        "le périmètre u doit être strictement positif"
    );
    area / perimeter
}

/// Flux de cisaillement de torsion `τ·tef = TEd / (2 · Ak)` (N/mm), avec le
/// moment de torsion `TEd` en N·mm et l'aire enveloppe `Ak` en mm².
///
/// Panique si `design_torque < 0` ou si `enclosed_area <= 0` (division par zéro).
pub fn rctor_shear_flow(design_torque: f64, enclosed_area: f64) -> f64 {
    assert!(
        design_torque >= 0.0,
        "le moment de torsion TEd doit être ≥ 0"
    );
    assert!(
        enclosed_area > 0.0,
        "l'aire enveloppe Ak doit être strictement positive"
    );
    design_torque / (2.0 * enclosed_area)
}

/// Aire totale des armatures longitudinales de torsion
/// `ΣAsl = TEd · uk · cot θ / (2 · Ak · fyd)` (mm²), modèle du treillis spatial à
/// angle de bielle `θ` (rad), avec `TEd` en N·mm, `uk` en mm, `Ak` en mm² et
/// `fyd` en MPa.
///
/// Panique si `design_torque < 0`, si `enclosed_area <= 0`, si `perimeter < 0`,
/// si `strut_angle_rad` n'est pas dans `]0, π/2[` (bornes où `tan θ` s'annule ou
/// diverge) ou si `yield_strength <= 0`.
pub fn rctor_longitudinal_reinforcement(
    design_torque: f64,
    enclosed_area: f64,
    perimeter: f64,
    strut_angle_rad: f64,
    yield_strength: f64,
) -> f64 {
    assert!(
        design_torque >= 0.0,
        "le moment de torsion TEd doit être ≥ 0"
    );
    assert!(
        enclosed_area > 0.0,
        "l'aire enveloppe Ak doit être strictement positive"
    );
    assert!(perimeter >= 0.0, "le périmètre uk doit être ≥ 0");
    assert!(
        strut_angle_rad > 0.0 && strut_angle_rad < FRAC_PI_2,
        "l'angle de bielle θ doit être dans ]0, π/2["
    );
    assert!(
        yield_strength > 0.0,
        "la limite d'élasticité fyd doit être strictement positive"
    );
    let cot_theta = 1.0 / strut_angle_rad.tan();
    design_torque * perimeter * cot_theta / (2.0 * enclosed_area * yield_strength)
}

/// Densité d'étriers fermés de torsion `Asw/s = TEd · tan θ / (2 · Ak · fywd)`
/// (mm²/mm), modèle du treillis spatial à angle de bielle `θ` (rad), avec `TEd`
/// en N·mm, `Ak` en mm² et `fywd` en MPa.
///
/// Panique si `design_torque < 0`, si `enclosed_area <= 0`, si `yield_strength <= 0`
/// ou si `strut_angle_rad` n'est pas dans `]0, π/2[` (bornes où `tan θ` s'annule ou
/// diverge).
pub fn rctor_stirrup_area_per_spacing(
    design_torque: f64,
    enclosed_area: f64,
    yield_strength: f64,
    strut_angle_rad: f64,
) -> f64 {
    assert!(
        design_torque >= 0.0,
        "le moment de torsion TEd doit être ≥ 0"
    );
    assert!(
        enclosed_area > 0.0,
        "l'aire enveloppe Ak doit être strictement positive"
    );
    assert!(
        yield_strength > 0.0,
        "la limite d'élasticité fywd doit être strictement positive"
    );
    assert!(
        strut_angle_rad > 0.0 && strut_angle_rad < FRAC_PI_2,
        "l'angle de bielle θ doit être dans ]0, π/2["
    );
    design_torque * strut_angle_rad.tan() / (2.0 * enclosed_area * yield_strength)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_4;

    #[test]
    fn thin_wall_thickness_clean_case_and_reciprocity() {
        // Cas chiffré : A = 100 000 mm², u = 2 000 mm → tef = 50 mm.
        let tef = rctor_thin_wall_thickness(100_000.0, 2_000.0);
        assert_relative_eq!(tef, 50.0, epsilon = 1e-9);
        // Réciprocité géométrique : A = tef · u, on retrouve bien l'aire.
        assert_relative_eq!(tef * 2_000.0, 100_000.0, epsilon = 1e-6);
    }

    #[test]
    fn shear_flow_clean_case_and_proportionality() {
        // Cas chiffré : TEd = 87·10^6 N·mm, Ak = 100 000 mm² →
        //   τ·tef = 87e6 / (2 · 100 000) = 87e6 / 2e5 = 435 N/mm.
        let q = rctor_shear_flow(87.0e6, 100_000.0);
        assert_relative_eq!(q, 435.0, epsilon = 1e-6);
        // Proportionnalité : doubler TEd double le flux de cisaillement.
        let q2 = rctor_shear_flow(174.0e6, 100_000.0);
        assert_relative_eq!(q2 / q, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn longitudinal_reinforcement_clean_case() {
        // Cas chiffré à θ = 45° (cot θ = 1) :
        //   ΣAsl = TEd · uk · cot θ / (2 · Ak · fyd)
        //        = 87e6 · 2000 · 1 / (2 · 100 000 · 435)
        //        = 1,74e11 / 8,7e7 = 2000 mm².
        let asl = rctor_longitudinal_reinforcement(87.0e6, 100_000.0, 2_000.0, FRAC_PI_4, 435.0);
        assert_relative_eq!(asl, 2_000.0, epsilon = 1e-3);
    }

    #[test]
    fn stirrup_area_clean_case_and_proportionality() {
        // Cas chiffré à θ = 45° (tan θ = 1) :
        //   Asw/s = TEd · tan θ / (2 · Ak · fywd)
        //         = 87e6 · 1 / (2 · 100 000 · 435) = 87e6 / 8,7e7 = 1,0 mm²/mm.
        let asws = rctor_stirrup_area_per_spacing(87.0e6, 100_000.0, 435.0, FRAC_PI_4);
        assert_relative_eq!(asws, 1.0, epsilon = 1e-6);
        // Proportionnalité : doubler TEd double la densité d'étriers.
        let asws2 = rctor_stirrup_area_per_spacing(174.0e6, 100_000.0, 435.0, FRAC_PI_4);
        assert_relative_eq!(asws2 / asws, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn reinforcement_product_is_angle_independent() {
        // Identité du treillis spatial : ΣAsl utilise cot θ et Asw/s utilise
        // tan θ, donc leur produit est indépendant de l'angle des bielles :
        //   ΣAsl · (Asw/s) = TEd² · uk / (4 · Ak² · fyd · fywd).
        // On compare θ = 45° (cot θ = 1) et θ = atan(0,4) (cot θ = 2,5).
        let t = 87.0e6;
        let ak = 100_000.0;
        let uk = 2_000.0;
        let fyd = 435.0;
        let theta_a = FRAC_PI_4;
        let theta_b = (0.4_f64).atan();
        let prod_a = rctor_longitudinal_reinforcement(t, ak, uk, theta_a, fyd)
            * rctor_stirrup_area_per_spacing(t, ak, fyd, theta_a);
        let prod_b = rctor_longitudinal_reinforcement(t, ak, uk, theta_b, fyd)
            * rctor_stirrup_area_per_spacing(t, ak, fyd, theta_b);
        assert_relative_eq!(prod_a, prod_b, epsilon = 1e-3);
        // Valeur théorique : 87e6² · 2000 / (4 · 100000² · 435 · 435).
        let expected = t * t * uk / (4.0 * ak * ak * fyd * fyd);
        assert_relative_eq!(prod_a, expected, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "l'angle de bielle θ doit être dans ]0, π/2[")]
    fn stirrup_area_rejects_null_angle() {
        // tan(0) = 0 : le modèle du treillis exige θ ∈ ]0, π/2[.
        rctor_stirrup_area_per_spacing(87.0e6, 100_000.0, 435.0, 0.0);
    }
}
