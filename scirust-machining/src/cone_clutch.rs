//! **Embrayage à cône** — couple transmissible, effort d'engagement axial et
//! largeur de portée d'un accouplement à surfaces coniques de frottement.
//!
//! ```text
//! usure uniforme       C = µ·F·rm/sin(α)
//! pression uniforme    C = (2/3)·µ·F·(ro³ − ri³)/[(ro² − ri²)·sin(α)]
//! effort d'engagement  Fa = N·(sin(α) + µ·cos(α))
//! largeur de portée    b = (ro − ri)/sin(α)
//! ```
//!
//! `µ` coefficient de frottement (sans dimension), `F` effort presseur axial (N),
//! `rm` rayon moyen de la portée conique (m), `α` demi-angle au sommet du cône
//! (rad, mesuré entre l'axe et la génératrice), `ro`/`ri` rayons extérieur/
//! intérieur de la portée (m), `N` effort normal appliqué sur la surface conique
//! (N), `Fa` effort axial d'engagement (N), `b` largeur de portée mesurée le long
//! de la génératrice (m), `C` couple transmissible (N·m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : surfaces **coniques** et
//! frottement de Coulomb idéalisé. L'hypothèse d'**usure uniforme** (p·r = cte,
//! garniture rodée) est la plus prudente et **sous-estime** le couple par rapport
//! à la **pression uniforme** (p = cte, garniture neuve) ; le comportement réel
//! est intermédiaire entre ces deux **bornes**. Le coefficient `µ` et le demi-angle
//! au sommet `α` sont **fournis par l'appelant** (aucune valeur « par défaut »
//! n'est inventée). Le facteur `1/sin(α)` traduit l'amplification par coincement
//! propre au cône. Distinct de [`crate::disc_clutch`] (surfaces planes).

use core::f64::consts::FRAC_PI_2;

/// Couple transmissible, hypothèse d'**usure uniforme**
/// `C = µ·F·rm/sin(α)` (N·m).
///
/// Sous usure uniforme, le produit pression·rayon est constant et le couple
/// s'exprime au **rayon moyen** `rm`. Le facteur `1/sin(α)` majore le couple par
/// rapport au disque plan (effet de coincement du cône). Borne prudente.
///
/// Panique si `friction_coefficient < 0`, `axial_force < 0`, `mean_radius <= 0`,
/// ou si `semi_cone_angle_rad` n'est pas dans `]0, π/2[`.
pub fn cone_clutch_torque_uniform_wear(
    friction_coefficient: f64,
    axial_force: f64,
    mean_radius: f64,
    semi_cone_angle_rad: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement µ doit être positif"
    );
    assert!(
        axial_force >= 0.0,
        "l'effort presseur axial F doit être positif"
    );
    assert!(
        mean_radius > 0.0,
        "le rayon moyen rm doit être strictement positif"
    );
    assert!(
        semi_cone_angle_rad > 0.0 && semi_cone_angle_rad < FRAC_PI_2,
        "le demi-angle au sommet α doit être dans ]0, π/2["
    );
    friction_coefficient * axial_force * mean_radius / semi_cone_angle_rad.sin()
}

/// Couple transmissible, hypothèse de **pression uniforme**
/// `C = (2/3)·µ·F·(ro³ − ri³)/[(ro² − ri²)·sin(α)]` (N·m).
///
/// Sous pression uniforme (garniture neuve), le rayon effectif est supérieur au
/// rayon moyen : cette borne **majore** le couple par rapport à l'usure uniforme.
///
/// Panique si `friction_coefficient < 0`, `axial_force < 0`,
/// `outer_radius > inner_radius > 0` est violé, ou si `semi_cone_angle_rad`
/// n'est pas dans `]0, π/2[`.
pub fn cone_clutch_torque_uniform_pressure(
    friction_coefficient: f64,
    axial_force: f64,
    outer_radius: f64,
    inner_radius: f64,
    semi_cone_angle_rad: f64,
) -> f64 {
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement µ doit être positif"
    );
    assert!(
        axial_force >= 0.0,
        "l'effort presseur axial F doit être positif"
    );
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    assert!(
        semi_cone_angle_rad > 0.0 && semi_cone_angle_rad < FRAC_PI_2,
        "le demi-angle au sommet α doit être dans ]0, π/2["
    );
    let two_thirds = 2.0_f64 / 3.0;
    let radii_ratio = (outer_radius.powi(3) - inner_radius.powi(3))
        / (outer_radius.powi(2) - inner_radius.powi(2));
    two_thirds * friction_coefficient * axial_force * radii_ratio / semi_cone_angle_rad.sin()
}

/// Effort axial d'**engagement** `Fa = N·(sin(α) + µ·cos(α))` (N).
///
/// Effort axial à appliquer pour engager le cône : il doit vaincre la composante
/// axiale de la réaction normale `N·sin(α)` et le frottement de glissement
/// `µ·N·cos(α)` le long de la génératrice.
///
/// Panique si `normal_force < 0`, `friction_coefficient < 0`, ou si
/// `semi_cone_angle_rad` n'est pas dans `]0, π/2[`.
pub fn cone_clutch_engagement_force(
    normal_force: f64,
    semi_cone_angle_rad: f64,
    friction_coefficient: f64,
) -> f64 {
    assert!(normal_force >= 0.0, "l'effort normal N doit être positif");
    assert!(
        friction_coefficient >= 0.0,
        "le coefficient de frottement µ doit être positif"
    );
    assert!(
        semi_cone_angle_rad > 0.0 && semi_cone_angle_rad < FRAC_PI_2,
        "le demi-angle au sommet α doit être dans ]0, π/2["
    );
    normal_force * (semi_cone_angle_rad.sin() + friction_coefficient * semi_cone_angle_rad.cos())
}

/// Largeur de portée conique `b = (ro − ri)/sin(α)` (m).
///
/// Largeur de la surface de friction mesurée le long de la génératrice du cône ;
/// elle est supérieure à la différence de rayons `(ro − ri)` d'un facteur
/// `1/sin(α)`.
///
/// Panique si `outer_radius > inner_radius > 0` est violé, ou si
/// `semi_cone_angle_rad` n'est pas dans `]0, π/2[`.
pub fn cone_clutch_face_width(
    outer_radius: f64,
    inner_radius: f64,
    semi_cone_angle_rad: f64,
) -> f64 {
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    assert!(
        semi_cone_angle_rad > 0.0 && semi_cone_angle_rad < FRAC_PI_2,
        "le demi-angle au sommet α doit être dans ]0, π/2["
    );
    (outer_radius - inner_radius) / semi_cone_angle_rad.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_6;

    #[test]
    fn realistic_case_wear_and_pressure() {
        // α = 30° (sin α = 1/2), µ = 0,3, F = 5000 N, ro = 0,15 m, ri = 0,10 m.
        // rm = (ro + ri)/2 = 0,125 m.
        // Usure : C = 0,3·5000·0,125/0,5 = 187,5/0,5 = 375 N·m.
        // Pression : (ro³ − ri³)/(ro² − ri²) = 0,002375/0,0125 = 0,19 ;
        //            C = (2/3)·0,3·5000·0,19/0,5 = 1000·0,19/0,5 = 380 N·m.
        let (mu, force, ro, ri, alpha) = (0.3, 5000.0, 0.15, 0.10, FRAC_PI_6);
        let rm = (ro + ri) / 2.0;
        let cw = cone_clutch_torque_uniform_wear(mu, force, rm, alpha);
        let cp = cone_clutch_torque_uniform_pressure(mu, force, ro, ri, alpha);
        assert_relative_eq!(cw, 375.0, epsilon = 1e-9);
        assert_relative_eq!(cp, 380.0, epsilon = 1e-9);
    }

    #[test]
    fn pressure_torque_bounds_wear_torque_above() {
        // La pression uniforme majore l'usure uniforme (rayon effectif > rayon
        // moyen) tant que ro > ri, pour un même effort et un même angle.
        let (mu, force, ro, ri, alpha) = (0.4, 6000.0, 0.20, 0.05, FRAC_PI_6);
        let rm = (ro + ri) / 2.0;
        let cw = cone_clutch_torque_uniform_wear(mu, force, rm, alpha);
        let cp = cone_clutch_torque_uniform_pressure(mu, force, ro, ri, alpha);
        assert!(cp > cw);
    }

    #[test]
    fn wear_torque_scales_inversely_with_sin_alpha() {
        // À géométrie de rayon fixée, C·sin(α) est indépendant de α : c'est
        // le couple « disque plan » µ·F·rm. Réciprocité du facteur 1/sin(α).
        let (mu, force, rm) = (0.28, 3200.0, 0.12);
        let flat = mu * force * rm;
        for &alpha in &[FRAC_PI_6, 0.20_f64, 1.0_f64]
        {
            let c = cone_clutch_torque_uniform_wear(mu, force, rm, alpha);
            assert_relative_eq!(c * alpha.sin(), flat, epsilon = 1e-9);
        }
    }

    #[test]
    fn face_width_reciprocity() {
        // b·sin(α) = ro − ri : identité géométrique de la génératrice.
        let (ro, ri, alpha) = (0.14, 0.09, 0.15_f64);
        let b = cone_clutch_face_width(ro, ri, alpha);
        assert_relative_eq!(b * alpha.sin(), ro - ri, epsilon = 1e-12);
    }

    #[test]
    fn engagement_force_frictionless_limit() {
        // µ = 0 : l'effort d'engagement se réduit à la composante axiale
        // de la réaction normale Fa = N·sin(α).
        let (normal, alpha) = (1500.0, FRAC_PI_6);
        let fa = cone_clutch_engagement_force(normal, alpha, 0.0);
        assert_relative_eq!(fa, normal * alpha.sin(), epsilon = 1e-9);
    }

    #[test]
    fn engagement_force_realistic_case() {
        // N = 1000 N, α = 30°, µ = 0,3 :
        // Fa = 1000·(0,5 + 0,3·cos30°) = 1000·(0,5 + 0,3·0,866025…) = 759,8076… N.
        let fa = cone_clutch_engagement_force(1000.0, FRAC_PI_6, 0.3);
        let expected = 1000.0 * (0.5 + 0.3 * FRAC_PI_6.cos());
        assert_relative_eq!(fa, expected, epsilon = 1e-9);
        assert_relative_eq!(fa, 759.807_621_135_331_6, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "le rayon extérieur ro doit être supérieur au rayon intérieur ri")]
    fn inverted_radii_panics() {
        // ro < ri : géométrie invalide.
        cone_clutch_face_width(0.08, 0.12, FRAC_PI_6);
    }
}
