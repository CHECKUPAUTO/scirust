//! Engrenage intérieur (couronne à denture intérieure + pignon) — géométrie
//! d'un couple couronne/pignon à denture droite en développante et module commun.
//!
//! ```text
//! entraxe intérieur       a  = m · (z_ring − z_pinion) / 2
//! rapport de transmission i  = z_ring / z_pinion
//! diamètre intérieur      d_i = m · (z_ring − 2·h_a*)   (cercle de tête de la couronne)
//! ```
//!
//! `m` = `module_mm` module réel commun aux deux roues (mm), `z_ring` =
//! `ring_teeth` nombre de dents de la couronne (denture intérieure), `z_pinion`
//! = `pinion_teeth` nombre de dents du pignon (denture extérieure), `h_a*` =
//! `addendum_factor` coefficient de saillie (sans dimension), `a` entraxe
//! intérieur (mm), `i` rapport de transmission (sans dimension), `d_i` diamètre
//! du cercle de tête intérieur de la couronne (mm). Sur une couronne intérieure
//! la tête des dents pointe vers l'axe : le cercle de tête a donc un diamètre
//! **inférieur** au diamètre primitif `m·z_ring`.
//!
//! **Convention** : longueurs en mm (module et diamètres partagent la même unité
//! de longueur), rapport et coefficient de saillie sans dimension ; denture
//! droite à profil en développante, module commun aux deux roues, déport nul.
//! **Limite honnête** : géométrie exacte d'un couple non déporté ; l'absence
//! d'interférence de taillage ou de fonctionnement (rognage, butée) N'EST PAS
//! vérifiée et relève de l'appelant. Aucun module, coefficient de saillie,
//! angle de pression ni matériau « par défaut » n'est supposé : le coefficient
//! de saillie et le module sont **fournis par l'appelant**.

/// Entraxe intérieur `a = m · (z_ring − z_pinion) / 2` (mm) d'un couple
/// couronne/pignon à module commun `module_mm`.
///
/// Panique si `module_mm <= 0`, si `ring_teeth == 0`, si `pinion_teeth == 0`
/// ou si `ring_teeth <= pinion_teeth` (la couronne doit avoir strictement plus
/// de dents que le pignon pour former un engrenage intérieur).
pub fn internal_center_distance(module_mm: f64, ring_teeth: u32, pinion_teeth: u32) -> f64 {
    assert!(module_mm > 0.0, "le module m doit être strictement positif");
    assert!(
        ring_teeth >= 1,
        "le nombre de dents de la couronne doit être au moins 1"
    );
    assert!(
        pinion_teeth >= 1,
        "le nombre de dents du pignon doit être au moins 1"
    );
    assert!(
        ring_teeth > pinion_teeth,
        "la couronne doit avoir strictement plus de dents que le pignon"
    );
    module_mm * (f64::from(ring_teeth) - f64::from(pinion_teeth)) / 2.0
}

/// Rapport de transmission `i = z_ring / z_pinion` (sans dimension) d'un
/// engrenage intérieur couronne/pignon.
///
/// Panique si `ring_teeth == 0` ou si `pinion_teeth == 0`.
pub fn internal_gear_ratio(ring_teeth: u32, pinion_teeth: u32) -> f64 {
    assert!(
        ring_teeth >= 1,
        "le nombre de dents de la couronne doit être au moins 1"
    );
    assert!(
        pinion_teeth >= 1,
        "le nombre de dents du pignon doit être au moins 1"
    );
    f64::from(ring_teeth) / f64::from(pinion_teeth)
}

/// Diamètre du cercle de tête intérieur de la couronne
/// `d_i = m · (z_ring − 2·h_a*)` (mm) : sur une denture intérieure la saillie
/// est retranchée au diamètre primitif car les têtes pointent vers l'axe.
///
/// Panique si `module_mm <= 0`, si `ring_teeth == 0`, si `addendum_factor < 0`
/// ou si `2·addendum_factor >= ring_teeth` (diamètre de tête non positif,
/// géométrie de couronne impossible).
pub fn internal_ring_tip_diameter(module_mm: f64, ring_teeth: u32, addendum_factor: f64) -> f64 {
    assert!(module_mm > 0.0, "le module m doit être strictement positif");
    assert!(
        ring_teeth >= 1,
        "le nombre de dents de la couronne doit être au moins 1"
    );
    assert!(
        addendum_factor >= 0.0,
        "le coefficient de saillie doit être positif ou nul"
    );
    assert!(
        2.0 * addendum_factor < f64::from(ring_teeth),
        "2·h_a* doit rester inférieur au nombre de dents de la couronne"
    );
    module_mm * (f64::from(ring_teeth) - 2.0 * addendum_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn center_distance_is_proportional_to_module() {
        // a est linéaire en m : doubler le module double l'entraxe (z fixés).
        let a1 = internal_center_distance(1.0, 60, 20);
        let a2 = internal_center_distance(2.0, 60, 20);
        assert_relative_eq!(a2, 2.0 * a1, max_relative = 1e-12);
    }

    #[test]
    fn center_distance_is_half_difference_of_pitch_radii() {
        // Identité géométrique : a = (d_ring − d_pinion)/2 = m·(z_ring − z_pinion)/2.
        let m = 3.0_f64;
        let d_ring = m * 60.0;
        let d_pinion = m * 20.0;
        let a = internal_center_distance(m, 60, 20);
        assert_relative_eq!(a, (d_ring - d_pinion) / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn gear_ratio_is_reciprocal_when_roles_swap() {
        // i(z_ring, z_pinion) = 1 / i(z_pinion, z_ring) — réciprocité du rapport.
        let i = internal_gear_ratio(72, 24);
        let i_swapped = internal_gear_ratio(24, 72);
        assert_relative_eq!(i * i_swapped, 1.0, max_relative = 1e-12);
        assert_relative_eq!(i, 3.0, max_relative = 1e-12);
    }

    #[test]
    fn tip_diameter_is_smaller_than_pitch_diameter() {
        // Sur une couronne intérieure d_i < d = m·z (têtes vers l'axe).
        let m = 2.0_f64;
        let z = 80_u32;
        let pitch = m * f64::from(z);
        let tip = internal_ring_tip_diameter(m, z, 1.0);
        assert!(tip < pitch);
        // Écart = 2·m·h_a* exactement.
        assert_relative_eq!(pitch - tip, 2.0 * m * 1.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_internal_gear_case() {
        // Couple réaliste : m = 2,5 mm, couronne z = 100, pignon z = 30, h_a* = 1.
        // a = 2,5·(100 − 30)/2 = 87,5 mm ; i = 100/30 ≈ 3,3333 ;
        // d_i = 2,5·(100 − 2) = 245 mm.
        let m = 2.5_f64;
        assert_relative_eq!(
            internal_center_distance(m, 100, 30),
            87.5,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            internal_gear_ratio(100, 30),
            100.0 / 30.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            internal_ring_tip_diameter(m, 100, 1.0),
            245.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "la couronne doit avoir strictement plus de dents")]
    fn center_distance_panics_when_ring_not_larger() {
        internal_center_distance(2.0, 20, 20);
    }
}
