//! **Béton armé — poteau en compression** (Eurocode 2) : résistance de la
//! section d'un poteau en **compression centrée**, élancement mécanique, rayon
//! de giration d'une section rectangulaire, excentricité minimale et élancement
//! limite au-delà duquel les effets du **second ordre** ne sont plus
//! négligeables.
//!
//! ```text
//! résistance axiale centrée  N_Rd  = A_c · f_cd + A_s · f_yd
//! élancement mécanique       λ     = l_0 / i
//! rayon de giration (rect.)  i     = h / √12
//! excentricité minimale      e_0   = max(h / 30, 20)          (mm)
//! élancement limite          λ_lim = 20 · A · B · C / √n
//! ```
//!
//! `A_c` aire de béton comprimé (mm²), `f_cd` résistance de calcul en
//! compression du béton (MPa), `A_s` aire totale d'acier longitudinal (mm²),
//! `f_yd` résistance de calcul de l'acier (MPa), `N_Rd` effort normal résistant
//! de calcul (N) ; `l_0` longueur efficace (de flambement) du poteau (mm), `i`
//! rayon de giration de la section dans le plan de flambement (mm), `λ`
//! élancement mécanique (sans dimension) ; `h` hauteur (dimension) de la section
//! dans le plan de flambement (mm), `e_0` excentricité minimale (mm) ; `A`, `B`,
//! `C` coefficients de l'élancement limite (sans dimension), `n` effort normal
//! relatif `N_Ed / (A_c · f_cd)` (sans dimension), `λ_lim` élancement limite
//! (sans dimension).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les efforts s'expriment donc en **N** (1 kN = 10³ N) et
//! les aires en **mm²**. Types `f64`.
//!
//! **Limite honnête** : la résistance de section `N_Rd` vaut pour un poteau en
//! **compression centrée** (effort normal appliqué au centre de gravité, sans
//! moment). L'**élancement limite** ne sert qu'à **comparer** `λ` à `λ_lim` pour
//! décider si le **second ordre** est négligeable ; les effets du second ordre
//! (méthode de la rigidité nominale ou de la courbure nominale) ne sont **pas**
//! calculés ici et restent à la charge de l'appelant. Les **coefficients** `A`,
//! `B`, `C` et l'**effort normal relatif** `n`, ainsi que les **résistances de
//! calcul** `f_cd` et `f_yd` (issues des résistances caractéristiques `fck`,
//! `fyk` et des **coefficients partiels de sécurité** `γ_c`, `γ_s`), sont
//! **fournis par l'appelant** d'après l'**Eurocode 2 (EN 1992-1-1)** et son
//! **Annexe Nationale** — aucune valeur « par défaut » n'est inventée.

/// Effort normal résistant de calcul en compression centrée
/// `N_Rd = A_c · f_cd + A_s · f_yd` (N).
///
/// `A_c` et `A_s` en mm², `f_cd` et `f_yd` en MPa ; le résultat est en N.
///
/// Panique si l'une des quatre grandeurs (`concrete_area`, `fcd`, `steel_area`,
/// `fyd`) est négative.
pub fn rccol_axial_resistance(concrete_area: f64, fcd: f64, steel_area: f64, fyd: f64) -> f64 {
    assert!(concrete_area >= 0.0, "l'aire de béton A_c doit être ≥ 0");
    assert!(fcd >= 0.0, "la résistance de calcul f_cd doit être ≥ 0");
    assert!(steel_area >= 0.0, "l'aire d'acier A_s doit être ≥ 0");
    assert!(fyd >= 0.0, "la résistance de calcul f_yd doit être ≥ 0");
    concrete_area * fcd + steel_area * fyd
}

/// Élancement mécanique `λ = l_0 / i` (sans dimension), rapport de la longueur
/// efficace au rayon de giration.
///
/// `l_0` et `i` en mm ; le résultat est adimensionnel.
///
/// Panique si `effective_length < 0` ou si `radius_of_gyration <= 0` (division
/// par zéro).
pub fn rccol_slenderness_ratio(effective_length: f64, radius_of_gyration: f64) -> f64 {
    assert!(
        effective_length >= 0.0,
        "la longueur efficace l_0 doit être ≥ 0"
    );
    assert!(
        radius_of_gyration > 0.0,
        "le rayon de giration i doit être strictement positif"
    );
    effective_length / radius_of_gyration
}

/// Rayon de giration d'une section rectangulaire dans le plan de flambement
/// `i = h / √12` (mm), déduit de `i = √(I / A)` avec `I = b · h³ / 12` et
/// `A = b · h`.
///
/// `h` en mm ; le résultat est en mm et ne dépend pas de la largeur `b`.
///
/// Panique si `depth <= 0`.
pub fn rccol_radius_of_gyration_rectangular(depth: f64) -> f64 {
    assert!(
        depth > 0.0,
        "la hauteur de section h doit être strictement positive"
    );
    depth / 12.0_f64.sqrt()
}

/// Excentricité minimale `e_0 = max(h / 30, 20)` (mm) d'un poteau, plancher de
/// `20 mm` imposé par l'Eurocode 2.
///
/// `h` en mm ; le résultat est en mm.
///
/// Panique si `depth <= 0`.
pub fn rccol_minimum_eccentricity(depth: f64) -> f64 {
    assert!(
        depth > 0.0,
        "la hauteur de section h doit être strictement positive"
    );
    (depth / 30.0).max(20.0)
}

/// Élancement limite `λ_lim = 20 · A · B · C / √n` (sans dimension) : en deçà,
/// les effets du second ordre peuvent être négligés.
///
/// `A`, `B`, `C` et `n` sont fournis par l'Eurocode 2 (`n = N_Ed / (A_c · f_cd)`
/// est l'effort normal relatif) ; le résultat est adimensionnel.
///
/// Panique si `coefficient_a <= 0`, `coefficient_b <= 0`, `coefficient_c <= 0`
/// ou `relative_normal_force <= 0` (racine et division par zéro).
pub fn rccol_limiting_slenderness(
    coefficient_a: f64,
    coefficient_b: f64,
    coefficient_c: f64,
    relative_normal_force: f64,
) -> f64 {
    assert!(
        coefficient_a > 0.0,
        "le coefficient A doit être strictement positif"
    );
    assert!(
        coefficient_b > 0.0,
        "le coefficient B doit être strictement positif"
    );
    assert!(
        coefficient_c > 0.0,
        "le coefficient C doit être strictement positif"
    );
    assert!(
        relative_normal_force > 0.0,
        "l'effort normal relatif n doit être strictement positif"
    );
    20.0 * coefficient_a * coefficient_b * coefficient_c / relative_normal_force.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn axial_resistance_decomposes_into_concrete_and_steel() {
        // Additivité : N_Rd(A_c, A_s) = part béton (A_s = 0) + part acier (A_c = 0).
        let ac = 90_000.0_f64;
        let fcd = 16.0_f64;
        let a_s = 1_200.0_f64;
        let fyd = 435.0_f64;
        let concrete_only = rccol_axial_resistance(ac, fcd, 0.0, fyd);
        let steel_only = rccol_axial_resistance(0.0, fcd, a_s, fyd);
        assert_relative_eq!(
            rccol_axial_resistance(ac, fcd, a_s, fyd),
            concrete_only + steel_only,
            epsilon = 1e-9
        );
    }

    #[test]
    fn slenderness_is_reciprocal_of_radius() {
        // Réciprocité : λ · i restitue la longueur efficace l_0.
        let l0 = 4_200.0_f64;
        let i = 86.6_f64;
        let lambda = rccol_slenderness_ratio(l0, i);
        assert_relative_eq!(lambda * i, l0, epsilon = 1e-9);
    }

    #[test]
    fn radius_of_gyration_satisfies_area_moment_identity() {
        // Identité : i = √(I/A) ⇒ i² · 12 = h² pour une section rectangulaire.
        let h = 300.0_f64;
        let i = rccol_radius_of_gyration_rectangular(h);
        assert_relative_eq!(i * i * 12.0, h * h, epsilon = 1e-6);
    }

    #[test]
    fn minimum_eccentricity_floors_at_twenty() {
        // Plancher : h/30 < 20 ⇒ e_0 = 20 ; h/30 > 20 ⇒ e_0 = h/30.
        assert_relative_eq!(rccol_minimum_eccentricity(300.0), 20.0, epsilon = 1e-12);
        assert_relative_eq!(rccol_minimum_eccentricity(900.0), 30.0, epsilon = 1e-12);
        // Point de bascule h = 600 mm : h/30 = 20 = plancher.
        assert_relative_eq!(rccol_minimum_eccentricity(600.0), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn limiting_slenderness_scales_inversely_with_sqrt_of_n() {
        // Proportionnalité : n → 4·n divise λ_lim par √4 = 2.
        let base = rccol_limiting_slenderness(0.7, 1.1, 0.7, 0.3);
        let quadrupled = rccol_limiting_slenderness(0.7, 1.1, 0.7, 4.0 * 0.3);
        assert_relative_eq!(base / quadrupled, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_rc_column_case() {
        // Poteau 300×300 mm, béton C25 (f_cd = 25/1,5 = 16,6667 MPa), 4 HA20
        // (A_s = 400·π = 1256,637 mm²), acier B500 (f_yd = 500/1,15 = 434,7826 MPa) :
        //   N_Rd = 90000·16,6667 + 1256,637·434,7826
        //        = 1 500 000 + 546 363,94            = 2 046 363,94 N ≈ 2046 kN
        //   i    = 300/√12                            = 86,60254 mm
        //   λ    = 4200/86,60254                      = 48,49742
        //   λ_lim (A=0,7 ; B=1,1 ; C=0,7 ; n=0,3)
        //        = 20·0,7·1,1·0,7/√0,3                = 19,68150
        //   λ = 48,50 > λ_lim = 19,68 ⇒ second ordre à ne PAS négliger (non traité).
        let fcd = 16.666_666_666_666_67_f64;
        let fyd = 434.782_608_695_652_2_f64;
        let steel_area = 400.0 * core::f64::consts::PI;
        let nrd = rccol_axial_resistance(90_000.0, fcd, steel_area, fyd);
        assert_relative_eq!(nrd, 2_046_363.939_754_75, max_relative = 1e-6);

        let i = rccol_radius_of_gyration_rectangular(300.0);
        assert_relative_eq!(i, 86.602_540_378_443_88, epsilon = 1e-3);

        let lambda = rccol_slenderness_ratio(4_200.0, i);
        assert_relative_eq!(lambda, 48.497_422_611_928_56, epsilon = 1e-3);

        let lambda_lim = rccol_limiting_slenderness(0.7, 1.1, 0.7, 0.3);
        assert_relative_eq!(lambda_lim, 19.681_497_233_018_97, epsilon = 1e-3);

        assert!(lambda > lambda_lim);
    }

    #[test]
    #[should_panic(expected = "le rayon de giration i doit être strictement positif")]
    fn slenderness_rejects_zero_radius() {
        rccol_slenderness_ratio(4_200.0, 0.0);
    }
}
