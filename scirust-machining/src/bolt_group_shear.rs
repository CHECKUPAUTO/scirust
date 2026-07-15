//! Groupe de boulons chargé excentriquement en cisaillement — méthode élastique
//! (superposition du cisaillement direct et du cisaillement secondaire de torsion).
//!
//! ```text
//! cisaillement direct par boulon    V_d = P / n
//! moment quadratique polaire        J = Σ r_i²
//! cisaillement secondaire (torsion) V_t = P·e·r / J
//! cisaillement résultant            V_r = √(V_d² + V_t² + 2·V_d·V_t·cos φ)
//! ```
//!
//! `P` charge appliquée au groupe (N), `n` nombre de boulons (`n ≥ 1`), `V_d`
//! cisaillement direct par boulon (N, charge supposée répartie également), `r_i`
//! distance du boulon `i` au centroïde du groupe (m), `J` moment quadratique polaire
//! du groupe (m², les aires des boulons étant supposées égales et prises unitaires),
//! `e` excentricité de la charge par rapport au centroïde (m), `r` distance du
//! boulon considéré au centroïde (m), `V_t` cisaillement secondaire dû au moment
//! `P·e` (N), `φ` angle entre les vecteurs de cisaillement direct et secondaire au
//! boulon (rad), `V_r` cisaillement résultant sur le boulon (N).
//!
//! **Convention** : unités SI cohérentes (N, m, rad). `J` est exprimé en m² car les
//! sections des boulons sont supposées identiques et normalisées à l'unité ; toute
//! aire commune se simplifie dans le rapport `r/J`.
//!
//! **Limite honnête** : boulons de même section, plaques rigides tournant autour du
//! centroïde du groupe, comportement purement élastique (analyse vectorielle
//! classique). Les positions et rayons des boulons, la charge, l'excentricité et
//! l'angle géométrique sont FOURNIS par l'appelant à partir de la géométrie réelle
//! du groupe ; ce module n'invente aucune valeur « par défaut » de disposition, de
//! section ni de matériau.

/// Cisaillement direct par boulon : `V_d = P / n` (N).
///
/// La charge appliquée est supposée répartie également sur les `n` boulons.
///
/// Panique si `bolt_count == 0` ou si `load < 0`.
pub fn boltgroup_direct_shear(load: f64, bolt_count: u32) -> f64 {
    assert!(bolt_count >= 1, "le groupe doit compter au moins 1 boulon");
    assert!(
        load >= 0.0,
        "la charge appliquée doit être positive ou nulle"
    );
    load / bolt_count as f64
}

/// Moment quadratique polaire du groupe : `J = Σ r_i²` (m², aires unitaires égales).
///
/// Somme des carrés des distances des boulons au centroïde du groupe.
///
/// Panique si `radii` est vide ou si l'un des rayons est négatif.
pub fn boltgroup_polar_inertia(radii: &[f64]) -> f64 {
    assert!(
        !radii.is_empty(),
        "le groupe doit contenir au moins un boulon (liste de rayons non vide)"
    );
    assert!(
        radii.iter().all(|&r| r >= 0.0),
        "chaque rayon au centroïde doit être positif ou nul"
    );
    radii.iter().map(|&r| r * r).sum()
}

/// Cisaillement secondaire dû au moment : `V_t = P·e·r / J` (N).
///
/// Contribution du moment `P·e` (torsion du groupe) au cisaillement du boulon
/// situé à la distance `bolt_radius` du centroïde.
///
/// Panique si `load < 0`, `eccentricity < 0`, `bolt_radius < 0` ou `polar_inertia <= 0`.
pub fn boltgroup_torsional_shear(
    load: f64,
    eccentricity: f64,
    bolt_radius: f64,
    polar_inertia: f64,
) -> f64 {
    assert!(
        load >= 0.0,
        "la charge appliquée doit être positive ou nulle"
    );
    assert!(
        eccentricity >= 0.0,
        "l'excentricité doit être positive ou nulle"
    );
    assert!(
        bolt_radius >= 0.0,
        "la distance du boulon au centroïde doit être positive ou nulle"
    );
    assert!(
        polar_inertia > 0.0,
        "le moment quadratique polaire doit être strictement positif"
    );
    load * eccentricity * bolt_radius / polar_inertia
}

/// Cisaillement résultant sur un boulon (composition vectorielle) :
/// `V_r = √(V_d² + V_t² + 2·V_d·V_t·cos φ)` (N).
///
/// Combine le cisaillement direct `V_d` et le cisaillement secondaire `V_t` séparés
/// de l'angle `φ` (loi des cosinus appliquée aux deux vecteurs de cisaillement).
///
/// Panique si `direct_shear < 0` ou si `torsional_shear < 0`.
pub fn boltgroup_resultant_shear(
    direct_shear: f64,
    torsional_shear: f64,
    angle_between_rad: f64,
) -> f64 {
    assert!(
        direct_shear >= 0.0,
        "le cisaillement direct doit être positif ou nul"
    );
    assert!(
        torsional_shear >= 0.0,
        "le cisaillement secondaire doit être positif ou nul"
    );
    (direct_shear * direct_shear
        + torsional_shear * torsional_shear
        + 2.0 * direct_shear * torsional_shear * angle_between_rad.cos())
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn direct_shear_distributes_the_load_over_all_bolts() {
        // Réciprocité : V_d · n doit restituer la charge totale P.
        let load = 12_000.0_f64;
        for &n in &[1u32, 2, 4, 6, 8]
        {
            let vd = boltgroup_direct_shear(load, n);
            assert_relative_eq!(vd * n as f64, load, epsilon = 1e-9);
        }
    }

    #[test]
    fn polar_inertia_of_bolts_on_a_circle_is_n_times_r_squared() {
        // n boulons tous à la même distance r du centroïde : J = n·r².
        let r = 0.070_710_678_f64;
        let radii = [r, r, r, r];
        assert_relative_eq!(
            boltgroup_polar_inertia(&radii),
            radii.len() as f64 * r * r,
            epsilon = 1e-12
        );
    }

    #[test]
    fn torsional_shear_reciprocity_recovers_the_load() {
        // Identité inverse : V_t · J / (e · r) = P.
        let load = 20_000.0_f64;
        let e = 0.3_f64;
        let r = 0.070_710_678_f64;
        let j = 0.02_f64;
        let vt = boltgroup_torsional_shear(load, e, r, j);
        assert_relative_eq!(vt * j / (e * r), load, epsilon = 1e-6);
    }

    #[test]
    fn resultant_reduces_to_the_law_of_cosines_special_cases() {
        // φ = 0 → somme ; φ = π/2 → hypoténuse ; φ = π → différence absolue.
        let vd = 3_000.0_f64;
        let vt = 4_000.0_f64;
        assert_relative_eq!(
            boltgroup_resultant_shear(vd, vt, 0.0),
            vd + vt,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            boltgroup_resultant_shear(vd, vt, PI / 2.0),
            (vd * vd + vt * vt).sqrt(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            boltgroup_resultant_shear(vd, vt, PI),
            (vt - vd).abs(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn realistic_eccentric_bracket_chain() {
        // Console : 4 boulons aux coins d'un carré de côté 100 mm (a = 50 mm),
        // donc r = √(0,05² + 0,05²) = 0,05·√2 m et J = 4·(0,05² + 0,05²) = 0,02 m².
        let a = 0.05_f64;
        let r = (a * a + a * a).sqrt();
        let radii = [r, r, r, r];
        let j = boltgroup_polar_inertia(&radii);
        assert_relative_eq!(j, 0.02, epsilon = 1e-12);

        // Charge P = 20 kN, excentricité e = 0,3 m.
        let load = 20_000.0_f64;
        let e = 0.3_f64;
        let vd = boltgroup_direct_shear(load, radii.len() as u32);
        assert_relative_eq!(vd, 5_000.0, epsilon = 1e-9);

        // V_t = P·e·r / J = 6000 · 2,5·√2 = 21 213,20344 N (car r/J = 2,5·√2).
        let vt = boltgroup_torsional_shear(load, e, r, j);
        assert_relative_eq!(vt, 6_000.0 * 2.5 * 2.0_f64.sqrt(), epsilon = 1e-6);

        // À φ = 90°, V_r = √(V_d² + V_t²) = √(25e6 + 450e6) = √(475e6) N.
        let vr = boltgroup_resultant_shear(vd, vt, PI / 2.0);
        assert_relative_eq!(vr, 475_000_000.0_f64.sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn torsional_shear_scales_linearly_with_eccentricity() {
        // Proportionnalité : doubler l'excentricité double le cisaillement secondaire.
        let load = 15_000.0_f64;
        let r = 0.08_f64;
        let j = 0.03_f64;
        let vt1 = boltgroup_torsional_shear(load, 0.2, r, j);
        let vt2 = boltgroup_torsional_shear(load, 0.4, r, j);
        assert_relative_eq!(vt2, 2.0 * vt1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins 1 boulon")]
    fn zero_bolts_has_no_direct_shear() {
        boltgroup_direct_shear(10_000.0, 0);
    }
}
