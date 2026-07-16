//! **Béton armé — flexion simple** d'une section rectangulaire à l'**ELU**
//! (Eurocode 2) : résistances de calcul du béton et de l'acier, moment réduit,
//! bras de levier des forces internes et aire d'acier tendu d'une section
//! simplement armée.
//!
//! ```text
//! résistance béton calcul  fcd = α_cc · fck / γ_c
//! résistance acier calcul  fyd = fyk / γ_s
//! moment réduit            µ   = M_Ed / (b · d² · fcd)
//! bras de levier           z   = d · (1 + √(1 − 2·µ)) / 2
//! aire d'acier tendu       A_s = M_Ed / (z · fyd)
//! ```
//!
//! `fck` résistance caractéristique en compression du béton (MPa), `α_cc`
//! coefficient tenant compte des effets de longue durée sur la résistance en
//! compression (sans dimension), `γ_c` coefficient partiel de sécurité du béton
//! (sans dimension), `fcd` résistance de calcul du béton (MPa) ; `fyk` limite
//! d'élasticité caractéristique de l'acier (MPa), `γ_s` coefficient partiel de
//! sécurité de l'acier (sans dimension), `fyd` résistance de calcul de l'acier
//! (MPa) ; `M_Ed` moment fléchissant de calcul (N·mm), `b` largeur de la section
//! (mm), `d` hauteur utile — distance de la fibre comprimée au centre de gravité
//! des aciers tendus (mm), `µ` moment réduit (sans dimension, `∈ [0, 0,5]`), `z`
//! bras de levier des forces internes (mm), `A_s` aire d'acier tendu (mm²).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les moments s'expriment donc en **N·mm** (1 kN·m = 10⁶
//! N·mm) et les aires en **mm²**. Types `f64`.
//!
//! **Limite honnête** : section **rectangulaire** en **flexion simple** à
//! l'**ELU**, diagramme rectangulaire simplifié du béton comprimé, acier à
//! palier plastique parfait ; section **simplement armée** — ni la vérification
//! du **pivot** (compatibilité des déformations), ni l'ajout d'**acier
//! comprimé** ne sont traités. Les **résistances caractéristiques** (`fck`,
//! `fyk`) et **tous les coefficients partiels de sécurité** (`γ_c`, `γ_s`,
//! `α_cc`) sont **fournis par l'appelant** d'après l'**Eurocode 2 (EN 1992-1-1)**
//! et son **Annexe Nationale** — aucune valeur « par défaut » n'est inventée.

/// Résistance de calcul en compression du béton `fcd = α_cc · fck / γ_c` (MPa).
///
/// `α_cc` et `γ_c` sont fournis par l'Eurocode 2 et son Annexe Nationale.
///
/// Panique si `fck < 0`, si `alpha_cc < 0` ou si `gamma_c <= 0` (division par
/// zéro).
pub fn rcbeam_design_concrete_strength(fck: f64, alpha_cc: f64, gamma_c: f64) -> f64 {
    assert!(
        fck >= 0.0,
        "la résistance caractéristique fck doit être ≥ 0"
    );
    assert!(alpha_cc >= 0.0, "le coefficient α_cc doit être ≥ 0");
    assert!(
        gamma_c > 0.0,
        "le coefficient partiel γ_c doit être strictement positif"
    );
    alpha_cc * fck / gamma_c
}

/// Résistance de calcul de l'acier `fyd = fyk / γ_s` (MPa).
///
/// `γ_s` est fourni par l'Eurocode 2 et son Annexe Nationale.
///
/// Panique si `fyk < 0` ou si `gamma_s <= 0` (division par zéro).
pub fn rcbeam_design_steel_strength(fyk: f64, gamma_s: f64) -> f64 {
    assert!(fyk >= 0.0, "la limite d'élasticité fyk doit être ≥ 0");
    assert!(
        gamma_s > 0.0,
        "le coefficient partiel γ_s doit être strictement positif"
    );
    fyk / gamma_s
}

/// Moment réduit `µ = M_Ed / (b · d² · fcd)` (sans dimension), rapport du moment
/// de calcul au moment de référence du béton comprimé.
///
/// `M_Ed` en N·mm, `b` et `d` en mm, `fcd` en MPa ; le résultat est adimensionnel.
///
/// Panique si `design_moment < 0`, si `width <= 0`, si `effective_depth <= 0`
/// ou si `fcd <= 0` (division par zéro).
pub fn rcbeam_reduced_moment(
    design_moment: f64,
    width: f64,
    effective_depth: f64,
    fcd: f64,
) -> f64 {
    assert!(
        design_moment >= 0.0,
        "le moment de calcul M_Ed doit être ≥ 0"
    );
    assert!(width > 0.0, "la largeur b doit être strictement positive");
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    assert!(
        fcd > 0.0,
        "la résistance de calcul fcd doit être strictement positive"
    );
    design_moment / (width * effective_depth * effective_depth * fcd)
}

/// Bras de levier des forces internes `z = d · (1 + √(1 − 2·µ)) / 2` (mm),
/// distance entre la résultante de compression du béton et les aciers tendus.
///
/// `d` en mm, `µ` le moment réduit adimensionnel. Le cas `µ = 0` donne `z = d`.
///
/// Panique si `effective_depth <= 0` ou si `reduced_moment` n'est pas dans
/// `[0, 0,5]` (au-delà de `0,5`, `1 − 2·µ < 0` : section non simplement armée).
pub fn rcbeam_lever_arm(effective_depth: f64, reduced_moment: f64) -> f64 {
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    assert!(
        (0.0..=0.5).contains(&reduced_moment),
        "le moment réduit µ doit être dans [0, 0,5] (section simplement armée)"
    );
    effective_depth * (1.0 + (1.0 - 2.0 * reduced_moment).sqrt()) / 2.0
}

/// Aire d'acier tendu `A_s = M_Ed / (z · fyd)` (mm²) d'une section simplement
/// armée.
///
/// `M_Ed` en N·mm, `z` en mm, `fyd` en MPa ; le résultat est en mm².
///
/// Panique si `design_moment < 0`, si `lever_arm <= 0` ou si `fyd <= 0`
/// (division par zéro).
pub fn rcbeam_required_steel_area(design_moment: f64, lever_arm: f64, fyd: f64) -> f64 {
    assert!(
        design_moment >= 0.0,
        "le moment de calcul M_Ed doit être ≥ 0"
    );
    assert!(
        lever_arm > 0.0,
        "le bras de levier z doit être strictement positif"
    );
    assert!(
        fyd > 0.0,
        "la résistance de calcul fyd doit être strictement positive"
    );
    design_moment / (lever_arm * fyd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn concrete_strength_scales_linearly_with_fck() {
        // Proportionnalité : à α_cc et γ_c fixés, doubler fck double fcd.
        let f1 = rcbeam_design_concrete_strength(25.0, 1.0, 1.5);
        let f2 = rcbeam_design_concrete_strength(50.0, 1.0, 1.5);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn steel_strength_reciprocity() {
        // Réciprocité : fyd · γ_s restitue la limite caractéristique fyk.
        let fyk = 500.0_f64;
        let gamma_s = 1.15_f64;
        let fyd = rcbeam_design_steel_strength(fyk, gamma_s);
        assert_relative_eq!(fyd * gamma_s, fyk, epsilon = 1e-9);
    }

    #[test]
    fn reduced_moment_proportional_to_design_moment() {
        // Linéarité : à section et fcd fixés, doubler M_Ed double µ.
        let m1 = rcbeam_reduced_moment(100.0e6, 300.0, 550.0, 16.666_666_666_666_67);
        let m2 = rcbeam_reduced_moment(200.0e6, 300.0, 550.0, 16.666_666_666_666_67);
        assert_relative_eq!(m2 / m1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn lever_arm_equals_depth_at_zero_moment() {
        // Cas limite : µ = 0 ⇒ √1 = 1 ⇒ z = d·(1+1)/2 = d.
        let d = 550.0_f64;
        assert_relative_eq!(rcbeam_lever_arm(d, 0.0), d, epsilon = 1e-9);
    }

    #[test]
    fn steel_area_reciprocity() {
        // Réciprocité : A_s · z · fyd restitue le moment de calcul M_Ed.
        let m = 200.0e6_f64;
        let z = 510.0_f64;
        let fyd = 434.782_608_695_652_2_f64;
        let a_s = rcbeam_required_steel_area(m, z, fyd);
        assert_relative_eq!(a_s * z * fyd, m, epsilon = 1e-3);
    }

    #[test]
    fn realistic_rc_beam_design_case() {
        // Poutre rectangulaire, béton C25, acier B500, à l'ELU :
        //   fck = 25, α_cc = 1, γ_c = 1,5 ⇒ fcd = 25/1,5 = 16,6667 MPa
        //   fyk = 500, γ_s = 1,15        ⇒ fyd = 500/1,15 = 434,7826 MPa
        //   b = 300 mm, d = 550 mm, M_Ed = 200 kN·m = 200e6 N·mm
        //   µ  = 200e6 / (300·550²·16,6667)
        //      = 600e6 / 4 537 500 000            = 0,132231
        //   z  = 550·(1 + √(1 − 2·0,132231))/2
        //      = 550·(1 + √0,735537)/2
        //      = 550·(1 + 0,857635)/2             = 510,8495 mm
        //   A_s = 200e6 / (510,8495·434,7826)     = 900,461 mm²
        let fcd = rcbeam_design_concrete_strength(25.0, 1.0, 1.5);
        assert_relative_eq!(fcd, 16.666_666_666_666_67, epsilon = 1e-3);
        let fyd = rcbeam_design_steel_strength(500.0, 1.15);
        assert_relative_eq!(fyd, 434.782_608_695_652_2, epsilon = 1e-3);
        let mu = rcbeam_reduced_moment(200.0e6, 300.0, 550.0, fcd);
        assert_relative_eq!(mu, 0.132_231_404_958_677_7, epsilon = 1e-6);
        let z = rcbeam_lever_arm(550.0, mu);
        assert_relative_eq!(z, 510.849_528_301_415_1, epsilon = 1e-3);
        let a_s = rcbeam_required_steel_area(200.0e6, z, fyd);
        assert_relative_eq!(a_s, 900.460_849_067_452_8, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(
        expected = "le moment réduit µ doit être dans [0, 0,5] (section simplement armée)"
    )]
    fn lever_arm_rejects_reduced_moment_above_half() {
        rcbeam_lever_arm(550.0, 0.6);
    }
}
