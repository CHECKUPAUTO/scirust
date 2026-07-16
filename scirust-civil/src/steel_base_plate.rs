//! **Charpente métallique — platine de pied de poteau sous charge de compression
//! centrée** (Eurocode 3, EN 1993-1-8 §6.2.5) : résistance d'appui du béton `fjd`,
//! aire d'appui requise sous l'effort normal centré, débord `c` de la surface
//! d'appui en T équivalent, et épaisseur de platine requise par la flexion du
//! débord.
//!
//! ```text
//! résistance d'appui       fjd     = βj · kj · fcd
//! aire d'appui requise     Areq    = NEd / fjd
//! débord d'appui           c       = t · √( fy / (3 · fjd · γM0) )
//! épaisseur requise        treq    = c · √( 3 · σ · γM0 / fy )
//! ```
//!
//! `fcd` = `concrete_design_strength` résistance de calcul du béton en compression
//! (MPa = N/mm²), `βj` = `joint_coefficient` coefficient du matériau de scellement
//! (sans dimension, `2/3` en règle générale), `kj` = `concentration_factor`
//! coefficient de concentration de contrainte de l'appui (sans dimension, `≥ 1`),
//! `fjd` = `bearing_strength` résistance d'appui de calcul (MPa) ; `NEd` =
//! `axial_load` effort normal de compression centré (N), `Areq` = aire d'appui
//! requise (mm²) ; `t` = `plate_thickness` épaisseur de la platine (mm), `fy` =
//! `yield_strength` limite d'élasticité de l'acier de la platine (MPa), `γM0` =
//! `gamma_m0` coefficient partiel de la section (sans dimension), `c` =
//! `cantilever` largeur de débord de la surface d'appui en T équivalent (mm) ;
//! `σ` = `bearing_pressure` contrainte d'appui sollicitant le débord (MPa), `treq`
//! = épaisseur de platine requise (mm).
//!
//! **Convention** : unités **N, mm, MPa** (avec `1 MPa = 1 N/mm²`, donc `fjd·A` est
//! en N), cohérentes entre elles (Eurocode). Types `f64`.
//!
//! **Limite honnête** : ce module traite la seule **platine sous charge de
//! compression centrée**, modélisée par la **surface d'appui en T équivalent**
//! (débord `c` autour du contour du profil). La **résistance d'appui** `fjd` est
//! bâtie à partir de `βj`, `kj` et `fcd` **fournis par l'appelant** d'après
//! l'**Eurocode 2/3 (EN 1992-1-1, EN 1993-1-8)** et son **Annexe Nationale** ; la
//! **limite d'élasticité** `fy` et le **coefficient partiel** `γM0` sont **fournis**
//! de la même manière — aucune valeur « par défaut » n'est inventée. L'**épaisseur
//! de platine** résulte de la **flexion du débord** en console. Ne sont **pas**
//! traités ici : le **moment de flexion** appliqué au pied (répartition d'appui
//! non uniforme), ni les **tiges d'ancrage tendues** — leur vérification reste à la
//! charge de l'appelant.

/// Résistance d'appui de calcul du béton `fjd = βj · kj · fcd` (MPa) sous la
/// platine (Eurocode 3, EN 1993-1-8 §6.2.5).
///
/// `concrete_design_strength` = `fcd` résistance de calcul du béton (MPa),
/// `joint_coefficient` = `βj` coefficient du matériau de scellement (sans
/// dimension, `2/3` usuel), `concentration_factor` = `kj` coefficient de
/// concentration (sans dimension, `≥ 1`), tous fournis par l'Eurocode et son
/// Annexe Nationale ; renvoie une contrainte (MPa).
///
/// Panique si `concrete_design_strength < 0`, si `joint_coefficient < 0` ou si
/// `concentration_factor < 0`.
pub fn steelbase_bearing_strength(
    concrete_design_strength: f64,
    joint_coefficient: f64,
    concentration_factor: f64,
) -> f64 {
    assert!(
        concrete_design_strength >= 0.0,
        "la résistance de calcul du béton fcd doit être ≥ 0 (MPa)"
    );
    assert!(
        joint_coefficient >= 0.0,
        "le coefficient de scellement βj doit être ≥ 0"
    );
    assert!(
        concentration_factor >= 0.0,
        "le coefficient de concentration kj doit être ≥ 0"
    );
    joint_coefficient * concentration_factor * concrete_design_strength
}

/// Aire d'appui requise sous charge centrée `Areq = NEd / fjd` (mm²) : surface
/// nécessaire pour ne pas dépasser la résistance d'appui du béton.
///
/// `axial_load` = `NEd` effort normal de compression centré (N),
/// `bearing_strength` = `fjd` résistance d'appui de calcul (MPa) ; renvoie une aire
/// (mm²).
///
/// Panique si `axial_load < 0` ou si `bearing_strength <= 0` (division par zéro).
pub fn steelbase_required_area(axial_load: f64, bearing_strength: f64) -> f64 {
    assert!(
        axial_load >= 0.0,
        "l'effort normal NEd doit être ≥ 0 (N, compression)"
    );
    assert!(
        bearing_strength > 0.0,
        "la résistance d'appui fjd doit être strictement positive (MPa)"
    );
    axial_load / bearing_strength
}

/// Largeur de débord de la surface d'appui en T équivalent
/// `c = t · √( fy / (3 · fjd · γM0) )` (mm) (Eurocode 3, EN 1993-1-8 §6.2.5).
///
/// `plate_thickness` = `t` épaisseur de la platine (mm), `yield_strength` = `fy`
/// limite d'élasticité de l'acier (MPa), `bearing_strength` = `fjd` résistance
/// d'appui de calcul (MPa), `gamma_m0` = `γM0` coefficient partiel (sans dimension)
/// fourni par l'Eurocode et son Annexe Nationale ; renvoie un débord (mm).
///
/// Panique si `plate_thickness < 0`, si `yield_strength < 0`, si
/// `bearing_strength <= 0` ou si `gamma_m0 <= 0` (division par zéro).
pub fn steelbase_additional_bearing_width(
    plate_thickness: f64,
    yield_strength: f64,
    bearing_strength: f64,
    gamma_m0: f64,
) -> f64 {
    assert!(
        plate_thickness >= 0.0,
        "l'épaisseur de la platine t doit être ≥ 0 (mm)"
    );
    assert!(
        yield_strength >= 0.0,
        "la limite d'élasticité fy doit être ≥ 0 (MPa)"
    );
    assert!(
        bearing_strength > 0.0,
        "la résistance d'appui fjd doit être strictement positive (MPa)"
    );
    assert!(
        gamma_m0 > 0.0,
        "le coefficient partiel γM0 doit être strictement positif"
    );
    plate_thickness * (yield_strength / (3.0 * bearing_strength * gamma_m0)).sqrt()
}

/// Épaisseur de platine requise par la flexion du débord en console
/// `treq = c · √( 3 · σ · γM0 / fy )` (mm) (Eurocode 3, EN 1993-1-8 §6.2.5).
///
/// `cantilever` = `c` porte-à-faux du débord d'appui (mm), `bearing_pressure` =
/// `σ` contrainte d'appui sollicitant le débord (MPa, au plus `fjd`),
/// `yield_strength` = `fy` limite d'élasticité de l'acier (MPa), `gamma_m0` = `γM0`
/// coefficient partiel (sans dimension) fourni par l'Eurocode et son Annexe
/// Nationale ; renvoie une épaisseur requise (mm).
///
/// Panique si `cantilever < 0`, si `bearing_pressure < 0`, si `yield_strength <= 0`
/// (division par zéro) ou si `gamma_m0 < 0`.
pub fn steelbase_plate_thickness_required(
    cantilever: f64,
    bearing_pressure: f64,
    yield_strength: f64,
    gamma_m0: f64,
) -> f64 {
    assert!(cantilever >= 0.0, "le débord c doit être ≥ 0 (mm)");
    assert!(
        bearing_pressure >= 0.0,
        "la contrainte d'appui σ doit être ≥ 0 (MPa)"
    );
    assert!(
        yield_strength > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    assert!(gamma_m0 >= 0.0, "le coefficient partiel γM0 doit être ≥ 0");
    cantilever * (3.0 * bearing_pressure * gamma_m0 / yield_strength).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bearing_strength_beta_two_thirds_kj_one_point_five() {
        // βj = 2/3, kj = 1,5, fcd = 20 MPa : (2/3)·1,5 = 1,0 donc fjd = fcd = 20 MPa.
        let fjd = steelbase_bearing_strength(20.0, 2.0 / 3.0, 1.5);
        assert_relative_eq!(fjd, 20.0, epsilon = 1e-9);
        // Proportionnalité au coefficient de concentration kj.
        let fjd2 = steelbase_bearing_strength(20.0, 2.0 / 3.0, 3.0);
        assert_relative_eq!(fjd2, 2.0 * fjd, epsilon = 1e-9);
    }

    #[test]
    fn required_area_inverts_bearing() {
        // Areq · fjd = NEd : l'aire requise multipliée par la résistance d'appui
        // restitue l'effort normal centré.
        let (ned, fjd) = (1_000_000.0_f64, 20.0);
        let area = steelbase_required_area(ned, fjd);
        assert_relative_eq!(area, 50_000.0, epsilon = 1e-6);
        assert_relative_eq!(area * fjd, ned, epsilon = 1e-6);
    }

    #[test]
    fn width_and_thickness_are_reciprocal() {
        // c = t·√(fy/(3·fjd·γM0)) et treq = c·√(3·σ·γM0/fy) sont réciproques
        // lorsque σ = fjd : partant de t, on doit retrouver t.
        let (t, fy, fjd, gamma) = (20.0_f64, 235.0, 20.0, 1.0);
        let c = steelbase_additional_bearing_width(t, fy, fjd, gamma);
        let t_back = steelbase_plate_thickness_required(c, fjd, fy, gamma);
        assert_relative_eq!(t_back, t, epsilon = 1e-9);
    }

    #[test]
    fn additional_width_proportional_to_thickness() {
        // c ∝ t à fy, fjd et γM0 fixés : doubler l'épaisseur double le débord.
        let c1 = steelbase_additional_bearing_width(15.0, 355.0, 25.0, 1.0);
        let c2 = steelbase_additional_bearing_width(30.0, 355.0, 25.0, 1.0);
        assert_relative_eq!(c2, 2.0 * c1, epsilon = 1e-9);
    }

    #[test]
    fn additional_width_worked_case() {
        // t = 20 mm, fy = 235 MPa, fjd = 20 MPa, γM0 = 1,0 :
        //   c = 20·√(235/(3·20·1)) = 20·√(235/60) = 20·√3,916667
        //     = 20·1,9790572 = 39,581144 mm.
        let c = steelbase_additional_bearing_width(20.0, 235.0, 20.0, 1.0);
        assert_relative_eq!(c, 39.581_144, epsilon = 1e-3);
    }

    #[test]
    fn realistic_centred_column_base_c25_s235() {
        // Poteau HEB centré, béton C25/30, acier S235, γM0 = 1,0.
        //   fcd = 16,667 MPa, βj = 2/3, kj = 1,5.
        //   fjd = (2/3)·1,5·16,667 = 1,0·16,667 = 16,667 MPa.
        //   NEd = 1 500 000 N (1500 kN).
        //   Areq = 1 500 000 / 16,667 = 90 000 mm² (soit ≈ 300×300 mm).
        //   Platine t = 25 mm : c = 25·√(235/(3·16,667·1)) = 25·√(235/50)
        //     = 25·√4,7 = 25·2,1679483 = 54,198708 mm.
        //   Épaisseur requise sous σ = fjd = 16,667 MPa, cantilever c :
        //     treq = c·√(3·16,667·1/235) = 54,198708·√(50/235)
        //          = 54,198708·√0,2127660 = 54,198708·0,4612657 = 25,0 mm.
        let fjd = steelbase_bearing_strength(16.667, 2.0 / 3.0, 1.5);
        assert_relative_eq!(fjd, 16.667, epsilon = 1e-3);
        let area = steelbase_required_area(1_500_000.0, fjd);
        assert_relative_eq!(area, 90_000.0, epsilon = 5.0);
        let c = steelbase_additional_bearing_width(25.0, 235.0, fjd, 1.0);
        assert_relative_eq!(c, 54.198_71, epsilon = 1e-2);
        let treq = steelbase_plate_thickness_required(c, fjd, 235.0, 1.0);
        assert_relative_eq!(treq, 25.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la résistance d'appui fjd doit être strictement positive")]
    fn required_area_rejects_zero_bearing() {
        // fjd = 0 : division par zéro, entrée refusée.
        steelbase_required_area(1_000_000.0, 0.0);
    }
}
