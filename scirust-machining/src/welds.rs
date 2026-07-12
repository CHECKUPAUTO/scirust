//! Soudures — cordons d'angle (**fillet**) et bout à bout : gorge, contrainte de
//! cisaillement directe, et cisaillement d'un groupe de cordons sous moment.
//!
//! ```text
//! gorge d'un cordon d'angle  a = 0,707·z          (z = côté du cordon)
//! aire de gorge              A = a·L
//! cisaillement direct        τ = F/(a·L)
//! contrainte bout à bout     σ = F/(a·L)          (a = épaisseur pénétrée)
//! cisaillement de groupe     τ = T·r/J            (torsion dans le plan)
//! ```
//!
//! `z` côté du cordon d'angle (m), `a` gorge (m), `L` longueur (m), `F` effort
//! (N), `T` moment dans le plan du groupe (N·m), `r` distance au barycentre, `J`
//! moment polaire du groupe de cordons.
//!
//! **Convention** : SI cohérent. **Limite honnête** : cordon d'angle à gorge
//! **isocèle** chargé en cisaillement moyen (méthode classique de résistance des
//! matériaux) ; ne calcule pas les coefficients de qualité/soudure normalisés
//! (Eurocode 3, méthode directionnelle) que l'appelant applique ensuite.

/// Gorge d'un cordon d'angle isocèle `a = 0,707·z` (m).
pub fn throat_thickness(leg_size: f64) -> f64 {
    core::f64::consts::FRAC_1_SQRT_2 * leg_size
}

/// Aire de gorge d'un cordon d'angle `A = a·L = 0,707·z·L` (m²).
pub fn fillet_throat_area(leg_size: f64, length: f64) -> f64 {
    throat_thickness(leg_size) * length
}

/// Cisaillement moyen dans un cordon d'angle `τ = F/(0,707·z·L)` (Pa).
///
/// Panique si `leg_size*length <= 0`.
pub fn fillet_direct_shear_stress(force: f64, leg_size: f64, length: f64) -> f64 {
    let area = fillet_throat_area(leg_size, length);
    assert!(area > 0.0, "l'aire de gorge doit être strictement positive");
    force / area
}

/// Contrainte normale d'un cordon **bout à bout** à pleine pénétration
/// `σ = F/(a·L)` (Pa), `a` épaisseur pénétrée.
///
/// Panique si `throat*length <= 0`.
pub fn butt_weld_stress(force: f64, throat: f64, length: f64) -> f64 {
    assert!(throat * length > 0.0, "a·L doit être strictement positif");
    force / (throat * length)
}

/// Cisaillement d'une soudure d'un **groupe de cordons** sous moment de torsion
/// dans le plan `τ = T·r/J` (Pa), `J` moment polaire du groupe.
///
/// Panique si `polar_moment <= 0`.
pub fn weld_group_torsional_shear(moment: f64, radius: f64, polar_moment: f64) -> f64 {
    assert!(
        polar_moment > 0.0,
        "le moment polaire doit être strictement positif"
    );
    moment * radius / polar_moment
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn throat_is_root_two_over_two_of_leg() {
        // z=6 mm → a = 0,707·6 ≈ 4,243 mm.
        assert_relative_eq!(throat_thickness(6.0), 6.0 / 2.0f64.sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn fillet_shear_of_a_side_weld() {
        // F=20 kN, z=6 mm, L=100 mm : A = 0,707·6·100 = 424,3 mm² → τ ≈ 47,1 MPa.
        let tau = fillet_direct_shear_stress(20_000.0, 6.0, 100.0);
        assert_relative_eq!(
            tau,
            20_000.0 / (6.0 / 2.0f64.sqrt() * 100.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn butt_weld_direct_stress() {
        // F=50 kN, a=10 mm, L=100 mm → σ = 50000/1000 = 50 MPa (unités N, mm).
        assert_relative_eq!(
            butt_weld_stress(50_000.0, 10.0, 100.0),
            50.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn group_torsional_shear_scales_with_radius() {
        // τ = T·r/J : doubler r double la contrainte.
        let t1 = weld_group_torsional_shear(500.0, 0.05, 1e-5);
        let t2 = weld_group_torsional_shear(500.0, 0.10, 1e-5);
        assert_relative_eq!(t2 / t1, 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "aire de gorge")]
    fn zero_length_weld_panics() {
        fillet_direct_shear_stress(20_000.0, 6.0, 0.0);
    }
}
