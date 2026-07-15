//! **Embrayage/frein à disques** — couple transmissible et effort presseur
//! axial d'un accouplement à disques plans, selon les deux hypothèses classiques
//! d'usure uniforme (garniture rodée) et de pression uniforme (garniture neuve).
//!
//! ```text
//! usure uniforme      C = µ·F·n·(ro + ri)/2
//! pression uniforme   C = µ·F·n·(2/3)·(ro³ − ri³)/(ro² − ri²)
//! effort axial (usure) F = pmax·2·π·ri·(ro − ri)
//! rayon moyen          rm = (ro + ri)/2
//! ```
//!
//! `µ` coefficient de frottement (sans dimension), `F` effort presseur axial (N),
//! `n` nombre de surfaces de frottement (sans dimension), `ro`/`ri` rayons
//! extérieur/intérieur de la couronne de friction (m), `pmax` pression de contact
//! maximale (Pa, atteinte au rayon intérieur en usure uniforme), `C` couple
//! transmissible (N·m), `rm` rayon moyen (m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : disques **plans** et
//! frottement de Coulomb idéalisé. L'hypothèse d'**usure uniforme** (p·r = cte,
//! garniture rodée) est la plus prudente et **sous-estime** le couple par rapport
//! à la **pression uniforme** (p = cte, garniture neuve) ; le comportement réel
//! est intermédiaire entre ces deux **bornes**. Le coefficient `µ`, le nombre de
//! surfaces de frottement `n` et la pression admissible `pmax` sont **fournis par
//! l'appelant** (aucune valeur « par défaut » n'est inventée). Distinct de
//! [`crate::clutch_engagement`] (énergie de glissement et échauffement).

use core::f64::consts::PI;

/// Couple transmissible, hypothèse d'**usure uniforme**
/// `C = µ·F·n·(ro + ri)/2` (N·m).
///
/// Sous usure uniforme, le produit pression·rayon est constant et le couple
/// s'exprime au **rayon moyen** `(ro + ri)/2`. Borne prudente (garniture rodée).
///
/// Panique si `friction_coefficient < 0`, `axial_force < 0`,
/// `number_of_surfaces == 0`, ou si `outer_radius > inner_radius > 0` est violé.
pub fn disc_clutch_transmissible_torque_uniform_wear(
    friction_coefficient: f64,
    axial_force: f64,
    outer_radius: f64,
    inner_radius: f64,
    number_of_surfaces: u32,
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
        number_of_surfaces >= 1,
        "le nombre de surfaces de frottement n doit être au moins 1"
    );
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    friction_coefficient
        * axial_force
        * f64::from(number_of_surfaces)
        * (outer_radius + inner_radius)
        / 2.0
}

/// Couple transmissible, hypothèse de **pression uniforme**
/// `C = µ·F·n·(2/3)·(ro³ − ri³)/(ro² − ri²)` (N·m).
///
/// Sous pression uniforme (garniture neuve), le rayon effectif est supérieur au
/// rayon moyen : cette borne **majore** le couple par rapport à l'usure uniforme.
///
/// Panique si `friction_coefficient < 0`, `axial_force < 0`,
/// `number_of_surfaces == 0`, ou si `outer_radius > inner_radius > 0` est violé.
pub fn disc_clutch_transmissible_torque_uniform_pressure(
    friction_coefficient: f64,
    axial_force: f64,
    outer_radius: f64,
    inner_radius: f64,
    number_of_surfaces: u32,
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
        number_of_surfaces >= 1,
        "le nombre de surfaces de frottement n doit être au moins 1"
    );
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    let two_thirds = 2.0_f64 / 3.0;
    let radii_ratio = (outer_radius.powi(3) - inner_radius.powi(3))
        / (outer_radius.powi(2) - inner_radius.powi(2));
    friction_coefficient * axial_force * f64::from(number_of_surfaces) * two_thirds * radii_ratio
}

/// Effort presseur axial en **usure uniforme** `F = pmax·2·π·ri·(ro − ri)` (N).
///
/// Intègre la répartition `p·r = pmax·ri` sur la couronne de friction : c'est
/// l'effort axial associé à une pression de contact maximale `pmax` au rayon
/// intérieur.
///
/// Panique si `max_pressure < 0` ou si `outer_radius > inner_radius > 0` est violé.
pub fn disc_clutch_axial_force_uniform_wear(
    max_pressure: f64,
    inner_radius: f64,
    outer_radius: f64,
) -> f64 {
    assert!(
        max_pressure >= 0.0,
        "la pression maximale pmax doit être positive"
    );
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    max_pressure * 2.0 * PI * inner_radius * (outer_radius - inner_radius)
}

/// Rayon moyen de la couronne de friction `rm = (ro + ri)/2` (m).
///
/// Rayon effectif du couple sous l'hypothèse d'usure uniforme.
///
/// Panique si `outer_radius > inner_radius > 0` est violé.
pub fn disc_clutch_mean_radius(outer_radius: f64, inner_radius: f64) -> f64 {
    assert!(
        inner_radius > 0.0,
        "le rayon intérieur ri doit être strictement positif"
    );
    assert!(
        outer_radius > inner_radius,
        "le rayon extérieur ro doit être supérieur au rayon intérieur ri"
    );
    (outer_radius + inner_radius) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn wear_torque_equals_force_times_mean_radius() {
        // Identité inter-fonctions : C_usure = µ·F·n·rm.
        let mu = 0.35;
        let force = 4200.0;
        let (ro, ri) = (0.16, 0.11);
        let n = 3;
        let rm = disc_clutch_mean_radius(ro, ri);
        let c = disc_clutch_transmissible_torque_uniform_wear(mu, force, ro, ri, n);
        assert_relative_eq!(c, mu * force * f64::from(n) * rm, epsilon = 1e-9);
    }

    #[test]
    fn realistic_case_wear_and_pressure() {
        // ro=0,15 m, ri=0,10 m, µ=0,3, F=5000 N, n=2.
        // Usure : C = 0,3·5000·2·(0,25/2) = 375 N·m.
        // Pression : (ro³−ri³)/(ro²−ri²) = 0,002375/0,0125 = 0,19 ;
        //            C = 0,3·5000·2·(2/3)·0,19 = 3000·0,126666… = 380 N·m.
        let (mu, force, ro, ri, n) = (0.3, 5000.0, 0.15, 0.10, 2);
        let cw = disc_clutch_transmissible_torque_uniform_wear(mu, force, ro, ri, n);
        let cp = disc_clutch_transmissible_torque_uniform_pressure(mu, force, ro, ri, n);
        assert_relative_eq!(cw, 375.0, epsilon = 1e-9);
        assert_relative_eq!(cp, 380.0, epsilon = 1e-9);
    }

    #[test]
    fn pressure_torque_bounds_wear_torque_above() {
        // La pression uniforme majore toujours l'usure uniforme (rayon effectif
        // supérieur au rayon moyen) tant que ro > ri.
        let (mu, force, ro, ri, n) = (0.4, 6000.0, 0.20, 0.05, 4);
        let cw = disc_clutch_transmissible_torque_uniform_wear(mu, force, ro, ri, n);
        let cp = disc_clutch_transmissible_torque_uniform_pressure(mu, force, ro, ri, n);
        assert!(cp > cw);
    }

    #[test]
    fn torque_scales_linearly_with_number_of_surfaces() {
        // Doubler le nombre de surfaces double le couple (n en facteur).
        let (mu, force, ro, ri) = (0.25, 3000.0, 0.14, 0.09);
        let c2 = disc_clutch_transmissible_torque_uniform_wear(mu, force, ro, ri, 2);
        let c4 = disc_clutch_transmissible_torque_uniform_wear(mu, force, ro, ri, 4);
        assert_relative_eq!(c4, 2.0 * c2, epsilon = 1e-9);
    }

    #[test]
    fn axial_force_realistic_case() {
        // pmax=1 MPa, ri=0,10 m, ro=0,15 m :
        // F = 1e6·2π·0,10·0,05 = 1e6·0,01·π = 1e4·π ≈ 31415,93 N.
        let f = disc_clutch_axial_force_uniform_wear(1.0e6, 0.10, 0.15);
        assert_relative_eq!(f, 1.0e4 * PI, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "le rayon extérieur ro doit être supérieur au rayon intérieur ri")]
    fn inverted_radii_panics() {
        // ro < ri : géométrie invalide.
        disc_clutch_mean_radius(0.08, 0.12);
    }
}
