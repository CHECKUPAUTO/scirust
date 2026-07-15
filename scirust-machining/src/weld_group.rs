//! Groupe de cordons de soudure d'angle chargé **excentriquement** — méthode
//! élastique du moment polaire d'inertie autour du centroïde du groupe.
//!
//! ```text
//! cisaillement direct     tau_d = P / A                       (aire de gorge totale)
//! module polaire          Z_p   = J_unit / c
//! cisaillement de torsion tau_t = P·e·r / J
//! contrainte résultante   tau_r = √(tau_d² + tau_t² + 2·tau_d·tau_t·cos θ)
//! ```
//!
//! `P` = effort appliqué au groupe (N), `A` = aire de gorge totale des cordons
//! (m²), `e` = excentricité de l'effort par rapport au centroïde (m), `r` =
//! distance du centroïde au point de cordon étudié (m), `J` = moment polaire
//! d'inertie du groupe traité comme une ligne d'épaisseur = gorge (m⁴),
//! `J_unit` = moment polaire d'inertie unitaire (par unité de gorge, m³), `c` =
//! distance maximale au centroïde (m), `Z_p` = module polaire (m²·gorge, m²·… selon
//! la convention d'unité retenue par l'appelant), `tau_d` = contrainte de
//! cisaillement direct (Pa), `tau_t` = contrainte de cisaillement de torsion (Pa),
//! `theta` = angle entre les vecteurs `tau_d` et `tau_t` au point étudié (rad),
//! `tau_r` = contrainte de cisaillement résultante (Pa).
//!
//! **Convention** : SI cohérent — dimensions en m, aires en m², efforts en N,
//! contraintes en Pa, angles en radians.
//!
//! **Limite honnête** : modèle des **cordons d'angle** analysés par la **méthode
//! élastique**, la soudure étant traitée comme une **ligne d'épaisseur égale à la
//! gorge** et le groupe tournant autour de son **centroïde**. La géométrie du
//! groupe — aire de gorge totale `A`, moment polaire d'inertie `J` (ou `J_unit` et
//! `c`) — est **fournie par l'appelant** ; ce module ne la déduit pas et n'invente
//! aucune constante de procédé, de matériau ou de facteur de forme. La méthode
//! ignore la plasticité, la concentration de contrainte aux extrémités et la
//! flexion hors plan.

/// Contrainte de cisaillement direct `tau_d = P / A`.
///
/// `load` = `P` effort appliqué (N), `throat_area` = `A` aire de gorge totale du
/// groupe (m²) ; renvoie une contrainte (Pa).
///
/// Panique si `load < 0` ou `throat_area <= 0`.
pub fn weldgroup_direct_shear_stress(load: f64, throat_area: f64) -> f64 {
    assert!(load >= 0.0 && throat_area > 0.0, "P ≥ 0 et A > 0 requis");
    load / throat_area
}

/// Module polaire du groupe `Z_p = J_unit / c`.
///
/// `polar_inertia_unit` = `J_unit` moment polaire d'inertie unitaire (m³),
/// `max_distance` = `c` distance maximale au centroïde (m) ; renvoie le module
/// polaire (m²).
///
/// Panique si `polar_inertia_unit < 0` ou `max_distance <= 0`.
pub fn weldgroup_polar_modulus(polar_inertia_unit: f64, max_distance: f64) -> f64 {
    assert!(
        polar_inertia_unit >= 0.0 && max_distance > 0.0,
        "J_unit ≥ 0 et c > 0 requis"
    );
    polar_inertia_unit / max_distance
}

/// Contrainte de cisaillement de torsion `tau_t = P·e·r / J`.
///
/// `load` = `P` effort appliqué (N), `eccentricity` = `e` excentricité par rapport
/// au centroïde (m), `distance` = `r` distance du centroïde au point étudié (m),
/// `polar_inertia` = `J` moment polaire d'inertie du groupe (m⁴) ; renvoie une
/// contrainte (Pa).
///
/// Panique si `load < 0`, `eccentricity < 0`, `distance < 0` ou `polar_inertia <= 0`.
pub fn weldgroup_torsional_shear_stress(
    load: f64,
    eccentricity: f64,
    distance: f64,
    polar_inertia: f64,
) -> f64 {
    assert!(
        load >= 0.0 && eccentricity >= 0.0 && distance >= 0.0 && polar_inertia > 0.0,
        "P ≥ 0, e ≥ 0, r ≥ 0 et J > 0 requis"
    );
    load * eccentricity * distance / polar_inertia
}

/// Contrainte de cisaillement résultante par la loi des cosinus
/// `tau_r = √(tau_d² + tau_t² + 2·tau_d·tau_t·cos θ)`.
///
/// `direct_stress` = `tau_d` cisaillement direct (Pa), `torsional_stress` = `tau_t`
/// cisaillement de torsion (Pa), `angle_between_rad` = `theta` angle entre les deux
/// vecteurs de contrainte au point étudié (rad) ; renvoie la contrainte résultante
/// (Pa).
///
/// Panique si `direct_stress < 0` ou `torsional_stress < 0`.
pub fn weldgroup_resultant_stress(
    direct_stress: f64,
    torsional_stress: f64,
    angle_between_rad: f64,
) -> f64 {
    assert!(
        direct_stress >= 0.0 && torsional_stress >= 0.0,
        "tau_d ≥ 0 et tau_t ≥ 0 requis"
    );
    (direct_stress * direct_stress
        + torsional_stress * torsional_stress
        + 2.0 * direct_stress * torsional_stress * angle_between_rad.cos())
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{PI, SQRT_2};

    #[test]
    fn direct_shear_proportional_to_load() {
        // tau_d ∝ P à aire constante : doubler l'effort double la contrainte.
        let single = weldgroup_direct_shear_stress(5_000.0, 0.0018);
        let double = weldgroup_direct_shear_stress(10_000.0, 0.0018);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-6);
    }

    #[test]
    fn polar_modulus_reciprocity() {
        // Z_p·c redonne J_unit (définition du module polaire).
        let (j_unit, c) = (2.4e-4_f64, 0.060);
        let zp = weldgroup_polar_modulus(j_unit, c);
        assert_relative_eq!(zp * c, j_unit, epsilon = 1e-15);
    }

    #[test]
    fn torsional_stress_identity() {
        // tau_t·J = P·e·r (réarrangement exact de la formule).
        let (p, e, r, j) = (12_000.0_f64, 0.120, 0.060, 1.2e-5);
        let tau_t = weldgroup_torsional_shear_stress(p, e, r, j);
        assert_relative_eq!(tau_t * j, p * e * r, epsilon = 1e-6);
    }

    #[test]
    fn resultant_limit_angles() {
        // θ = 0 → somme ; θ = π → écart absolu ; θ = π/2 → hypoténuse.
        let (d, t) = (6_000_000.0_f64, 4_000_000.0);
        assert_relative_eq!(weldgroup_resultant_stress(d, t, 0.0), d + t, epsilon = 1e-3);
        assert_relative_eq!(
            weldgroup_resultant_stress(d, t, PI),
            (d - t).abs(),
            epsilon = 1e-3
        );
        assert_relative_eq!(
            weldgroup_resultant_stress(d, t, PI / 2.0),
            (d * d + t * t).sqrt(),
            epsilon = 1e-3
        );
    }

    #[test]
    fn resultant_equal_components_orthogonal() {
        // Composantes égales à 90° : tau_r = tau·√2.
        let tau = 5_000_000.0_f64;
        assert_relative_eq!(
            weldgroup_resultant_stress(tau, tau, PI / 2.0),
            tau * SQRT_2,
            epsilon = 1e-3
        );
    }

    #[test]
    fn realistic_eccentric_weld_group() {
        // Groupe : P = 12 kN, aire de gorge A = 1800 mm² = 0,0018 m².
        // tau_d = 12000/0,0018 = 6,6667 MPa.
        let tau_d = weldgroup_direct_shear_stress(12_000.0, 0.0018);
        assert_relative_eq!(tau_d, 6_666_666.666_666_7_f64, epsilon = 1e-1);
        // Torsion : e = 120 mm, r = 60 mm, J = 1,2e-5 m⁴.
        // tau_t = 12000·0,120·0,060 / 1,2e-5 = 7,200 MPa.
        let tau_t = weldgroup_torsional_shear_stress(12_000.0, 0.120, 0.060, 1.2e-5);
        assert_relative_eq!(tau_t, 7_200_000.0_f64, epsilon = 1e-1);
        // Résultante à 90° : √(6,6667² + 7,2²) MPa = 9,8125 MPa.
        let tau_r = weldgroup_resultant_stress(tau_d, tau_t, PI / 2.0);
        assert_relative_eq!(tau_r, 9_812_463.729_586_2_f64, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "A > 0")]
    fn zero_throat_area_panics() {
        weldgroup_direct_shear_stress(1_000.0, 0.0);
    }
}
