//! Poutre fortement courbe (théorie de Winkler-Bach) — décalage de l'axe
//! neutre vers le centre de courbure et contrainte de flexion hyperbolique.
//!
//! ```text
//! rayon axe neutre       Rn = A / ∫(dA/r)
//! section rectangulaire   Rn = h / ln(ro/ri)
//! intégrale rectangle    ∫(dA/r) = b·ln(ro/ri)
//! décalage axe neutre    e  = Rc − Rn
//! contrainte de flexion  σ  = M·(Rn − r) / (A·e·r)
//! ```
//!
//! `A` aire de section (m² ou mm²), `∫(dA/r)` intégrale de section (m ou mm),
//! `Rn` rayon de l'axe neutre, `Rc` rayon du centre de gravité (centroïde),
//! `e = Rc − Rn` décalage de l'axe neutre vers le centre de courbure (positif),
//! `r` rayon de la fibre considérée, `b` largeur et `h` hauteur du rectangle,
//! `ri`/`ro` rayons intérieur/extérieur, `M` moment fléchissant (N·m ou N·mm),
//! `σ` contrainte normale. Un `M` qui tend à diminuer la courbure met en
//! tension la fibre intérieure (`σ > 0` pour `r < Rn`).
//!
//! **Convention** : unités cohérentes de l'appelant (SI : N, m, Pa, ou N, mm,
//! MPa). Tous les rayons sont mesurés depuis le centre de courbure.
//! **Limite honnête** : théorie de Winkler-Bach, valable pour une poutre
//! **fortement courbe** (Rc/h faible) où l'axe neutre ne passe plus par le
//! centroïde ; la formule fermée `Rn = h/ln(ro/ri)` suppose une **section
//! rectangulaire**. Aucune constante physique ou géométrique n'est fournie par
//! défaut : l'aire, les rayons et le moment sont TOUS passés par l'appelant.
//! Complète [`crate::beams`] (poutres droites, axe neutre au centroïde).

/// Rayon de l'axe neutre `Rn = A / ∫(dA/r)` d'une section quelconque, à partir
/// de l'aire `area` et de l'intégrale de section `cross_section_integral`.
///
/// Panique si `area <= 0` ou `cross_section_integral <= 0`.
pub fn curvedbeam_neutral_radius(area: f64, cross_section_integral: f64) -> f64 {
    assert!(
        area > 0.0,
        "l'aire de section doit être strictement positive"
    );
    assert!(
        cross_section_integral > 0.0,
        "l'intégrale de section ∫(dA/r) doit être strictement positive"
    );
    area / cross_section_integral
}

/// Intégrale de section `∫(dA/r) = b·ln(ro/ri)` d'un rectangle de largeur
/// `width`, entre les rayons intérieur `inner_radius` et extérieur
/// `outer_radius`.
///
/// Panique si `width <= 0`, `inner_radius <= 0` ou `outer_radius <= inner_radius`.
pub fn curvedbeam_rectangular_section_integral(
    width: f64,
    inner_radius: f64,
    outer_radius: f64,
) -> f64 {
    assert!(width > 0.0, "la largeur doit être strictement positive");
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur doit être supérieur au rayon intérieur"
    );
    width * (outer_radius / inner_radius).ln()
}

/// Rayon de l'axe neutre d'une section **rectangulaire**
/// `Rn = h / ln(ro/ri)`, hauteur radiale `height = ro − ri`, rayons intérieur
/// `inner_radius` et extérieur `outer_radius`.
///
/// Panique si `height <= 0`, `inner_radius <= 0` ou
/// `outer_radius <= inner_radius`.
pub fn curvedbeam_rectangular_neutral_radius(
    height: f64,
    inner_radius: f64,
    outer_radius: f64,
) -> f64 {
    assert!(
        height > 0.0,
        "la hauteur radiale doit être strictement positive"
    );
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur doit être supérieur au rayon intérieur"
    );
    height / (outer_radius / inner_radius).ln()
}

/// Décalage de l'axe neutre `e = Rc − Rn` (centroïde `centroid_radius` moins
/// axe neutre `neutral_radius`). Pour une poutre courbe valide `e > 0` : l'axe
/// neutre est décalé vers le centre de courbure.
///
/// Panique si `centroid_radius <= 0`, `neutral_radius <= 0` ou si le décalage
/// n'est pas strictement positif (`Rc <= Rn`).
pub fn curvedbeam_neutral_axis_shift(centroid_radius: f64, neutral_radius: f64) -> f64 {
    assert!(
        centroid_radius > 0.0,
        "le rayon du centroïde doit être strictement positif"
    );
    assert!(
        neutral_radius > 0.0,
        "le rayon de l'axe neutre doit être strictement positif"
    );
    let e = centroid_radius - neutral_radius;
    assert!(
        e > 0.0,
        "décalage non physique : le centroïde doit être plus éloigné que l'axe neutre (Rc > Rn)"
    );
    e
}

/// Contrainte de flexion `σ = M·(Rn − r) / (A·e·r)` dans une poutre courbe,
/// avec `e = Rc − Rn`. `moment` moment fléchissant, `area` aire de section,
/// `neutral_radius` rayon de l'axe neutre `Rn`, `centroid_radius` rayon du
/// centroïde `Rc`, `fiber_radius` rayon `r` de la fibre étudiée.
///
/// La contrainte s'annule sur l'axe neutre (`r = Rn`) et varie de façon
/// hyperbolique ; elle est maximale en valeur absolue sur la fibre intérieure.
///
/// Panique si `area <= 0`, `fiber_radius <= 0` ou si le décalage
/// `e = Rc − Rn` n'est pas strictement positif.
pub fn curvedbeam_stress(
    moment: f64,
    area: f64,
    neutral_radius: f64,
    centroid_radius: f64,
    fiber_radius: f64,
) -> f64 {
    assert!(
        area > 0.0,
        "l'aire de section doit être strictement positive"
    );
    assert!(
        fiber_radius > 0.0,
        "le rayon de la fibre doit être strictement positif"
    );
    let e = centroid_radius - neutral_radius;
    assert!(
        e > 0.0,
        "décalage non physique : le centroïde doit être plus éloigné que l'axe neutre (Rc > Rn)"
    );
    moment * (neutral_radius - fiber_radius) / (area * e * fiber_radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Section rectangulaire de référence : b=20, h=40, ri=50, ro=90 (mm).
    // A = 800 mm² ; Rc = 70 mm ; Rn = 40/ln(1,8) = 68,051901… mm.
    const B: f64 = 20.0;
    const H: f64 = 40.0;
    const RI: f64 = 50.0;
    const RO: f64 = 90.0;

    #[test]
    fn rectangular_neutral_radius_matches_general_formula() {
        // Réciprocité : Rn = A/∫(dA/r) doit coïncider avec Rn = h/ln(ro/ri),
        // car pour un rectangle ∫(dA/r) = b·ln(ro/ri) et A = b·h.
        let area = B * H;
        let integral = curvedbeam_rectangular_section_integral(B, RI, RO);
        let rn_general = curvedbeam_neutral_radius(area, integral);
        let rn_rect = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        assert_relative_eq!(rn_general, rn_rect, epsilon = 1e-12);
    }

    #[test]
    fn neutral_radius_realistic_value() {
        // Cas chiffré indépendant : 40/ln(90/50) = 40/ln(1,8) ≈ 68,051901 mm.
        let rn = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        assert_relative_eq!(rn, 68.051_901_120_725_46, epsilon = 1e-9);
    }

    #[test]
    fn neutral_axis_shift_is_positive_and_small() {
        // e = Rc − Rn = 70 − 68,051901 = 1,948099 mm (l'axe neutre est décalé
        // vers le centre de courbure).
        let rc = (RI + RO) / 2.0;
        let rn = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        let e = curvedbeam_neutral_axis_shift(rc, rn);
        assert_relative_eq!(e, 1.948_098_879_274_539_3, epsilon = 1e-9);
    }

    #[test]
    fn stress_vanishes_on_neutral_axis() {
        // Identité : σ(r = Rn) = 0 quel que soit le moment.
        let rc = (RI + RO) / 2.0;
        let rn = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        let s = curvedbeam_stress(1.0e6, B * H, rn, rc, rn);
        assert_relative_eq!(s, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn stress_is_proportional_to_moment() {
        // Linéarité : doubler le moment double la contrainte.
        let rc = (RI + RO) / 2.0;
        let rn = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        let s1 = curvedbeam_stress(1.0e6, B * H, rn, rc, RI);
        let s2 = curvedbeam_stress(2.0e6, B * H, rn, rc, RI);
        assert_relative_eq!(s2, 2.0 * s1, epsilon = 1e-9);
    }

    #[test]
    fn inner_fiber_stress_realistic_value() {
        // M = 1,0·10⁶ N·mm sur la section de référence.
        // σ_i = M·(Rn − ri)/(A·e·ri)
        //     = 1e6·(68,051901 − 50)/(800·1,948099·50) ≈ 231,66 MPa (tension).
        let rc = (RI + RO) / 2.0;
        let rn = curvedbeam_rectangular_neutral_radius(H, RI, RO);
        let s = curvedbeam_stress(1.0e6, B * H, rn, rc, RI);
        assert_relative_eq!(s, 231.660_483_366_325_37, epsilon = 1e-6);
        assert!(s > 0.0, "la fibre intérieure doit être en tension");
    }

    #[test]
    #[should_panic(expected = "décalage non physique")]
    fn stress_panics_when_centroid_not_beyond_neutral_axis() {
        // Rc <= Rn est impossible pour une poutre courbe : doit paniquer.
        curvedbeam_stress(1.0e6, 800.0, 70.0, 68.0, 50.0);
    }
}
