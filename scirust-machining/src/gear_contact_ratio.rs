//! **Rapport de conduite d'un engrenage cylindrique droit** — longueur du segment
//! d'action, pas de base et rapport de conduite d'une denture à développante.
//!
//! ```text
//! longueur d'action   Z  = √(ra_p² − rb_p²) + √(ra_g² − rb_g²) − C·sin α
//! pas de base         pb = π·m·cos α
//! rapport de conduite ε  = Z / pb
//! ```
//!
//! `ra_p`/`ra_g` rayons de tête (addendum) du pignon et de la roue (m), `rb_p`/`rb_g`
//! rayons de base (m), `C` entraxe de fonctionnement (m), `α` angle de pression de
//! fonctionnement (rad), `m` module (m), `Z` longueur du segment d'action (m), `pb`
//! pas de base mesuré sur la ligne d'action (m), `ε` rapport de conduite (sans
//! dimension).
//!
//! **Convention** : SI ; toutes les longueurs dans la même unité cohérente (m ou mm,
//! au choix de l'appelant du moment qu'elle est unique) ; angle de pression en
//! radians. **Limite honnête** : engrenage cylindrique **droit** à denture
//! **développante** ; la géométrie (rayons de tête et de base, entraxe, module,
//! angle de pression) est **fournie par l'appelant** (issue de la taille réelle et
//! du profil de référence) — aucune valeur « par défaut » n'est inventée. Un rapport
//! de conduite `ε > 1` est nécessaire pour une transmission continue (une paire de
//! dents prend le relais avant que la précédente ne quitte l'engrènement),
//! idéalement `ε > 1,4` pour une marche régulière. Distinct de
//! [`crate::gear_efficiency`] (pertes) et de [`crate::gears`] (géométrie générale).

use core::f64::consts::PI;

/// Longueur du segment d'action `Z = √(ra_p² − rb_p²) + √(ra_g² − rb_g²) − C·sin α`
/// (portion de la ligne d'action réellement parcourue par le point de contact).
///
/// Toutes les longueurs sont exprimées dans une même unité cohérente.
///
/// Panique si un rayon est `<= 0`, si `outer_radius <= base_radius` pour le pignon ou
/// la roue (tête sous le cercle de base), si `center_distance <= 0` ou si
/// `pressure_angle_rad` n'est pas dans `]0, π/2[`.
pub fn gear_contact_length_of_action(
    outer_radius_pinion: f64,
    base_radius_pinion: f64,
    outer_radius_gear: f64,
    base_radius_gear: f64,
    center_distance: f64,
    pressure_angle_rad: f64,
) -> f64 {
    assert!(
        base_radius_pinion > 0.0 && base_radius_gear > 0.0,
        "rayons de base rb_p > 0 et rb_g > 0 requis"
    );
    assert!(
        outer_radius_pinion > base_radius_pinion,
        "ra_p > rb_p requis (tête au-dessus du cercle de base, pignon)"
    );
    assert!(
        outer_radius_gear > base_radius_gear,
        "ra_g > rb_g requis (tête au-dessus du cercle de base, roue)"
    );
    assert!(center_distance > 0.0, "entraxe C > 0 requis");
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "0 < α < π/2 requis"
    );
    (outer_radius_pinion * outer_radius_pinion - base_radius_pinion * base_radius_pinion).sqrt()
        + (outer_radius_gear * outer_radius_gear - base_radius_gear * base_radius_gear).sqrt()
        - center_distance * pressure_angle_rad.sin()
}

/// Pas de base sur la ligne d'action `pb = π·m·cos α`.
///
/// Le module et le pas de base partagent la même unité de longueur.
///
/// Panique si `module_metric <= 0` ou si `pressure_angle_rad` n'est pas dans
/// `]0, π/2[`.
pub fn gear_contact_base_pitch(module_metric: f64, pressure_angle_rad: f64) -> f64 {
    assert!(module_metric > 0.0, "module m > 0 requis");
    assert!(
        pressure_angle_rad > 0.0 && pressure_angle_rad < PI / 2.0,
        "0 < α < π/2 requis"
    );
    PI * module_metric * pressure_angle_rad.cos()
}

/// Rapport de conduite `ε = Z / pb` (nombre moyen de paires de dents en prise).
///
/// `length_of_action` et `base_pitch` doivent être exprimés dans la même unité de
/// longueur ; le résultat est sans dimension. Une transmission continue exige
/// `ε > 1` (idéalement `> 1,4`).
///
/// Panique si `length_of_action < 0` ou si `base_pitch <= 0`.
pub fn gear_contact_ratio(length_of_action: f64, base_pitch: f64) -> f64 {
    assert!(length_of_action >= 0.0, "longueur d'action Z ≥ 0 requise");
    assert!(base_pitch > 0.0, "pas de base pb > 0 requis");
    length_of_action / base_pitch
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn base_pitch_proportional_to_module() {
        // pb = π·m·cos α : doubler le module double le pas de base.
        let a = gear_contact_base_pitch(5.0, 20.0_f64.to_radians());
        let b = gear_contact_base_pitch(10.0, 20.0_f64.to_radians());
        assert_relative_eq!(b, 2.0 * a, epsilon = 1e-12);
    }

    #[test]
    fn base_pitch_at_zero_alpha_limit() {
        // cos α → 1 quand α → 0 : pb → π·m. On teste avec un très petit angle.
        let tiny = 1e-6;
        assert_relative_eq!(
            gear_contact_base_pitch(3.0, tiny),
            PI * 3.0 * tiny.cos(),
            epsilon = 1e-15
        );
        // Et la borne théorique π·m est approchée.
        assert_relative_eq!(gear_contact_base_pitch(3.0, tiny), PI * 3.0, epsilon = 1e-9);
    }

    #[test]
    fn length_of_action_symmetric_in_pinion_gear() {
        // Z est la somme des deux termes de tête + un terme d'entraxe symétrique :
        // échanger pignon et roue laisse Z inchangé.
        let alpha = 20.0_f64.to_radians();
        let z1 = gear_contact_length_of_action(55.0, 46.984_63, 105.0, 93.969_26, 150.0, alpha);
        let z2 = gear_contact_length_of_action(105.0, 93.969_26, 55.0, 46.984_63, 150.0, alpha);
        assert_relative_eq!(z1, z2, epsilon = 1e-12);
    }

    #[test]
    fn contact_ratio_proportional_to_length() {
        // ε = Z/pb : à pas de base fixé, ε est proportionnel à Z.
        let pb = 14.76;
        let a = gear_contact_ratio(24.0, pb);
        let b = gear_contact_ratio(48.0, pb);
        assert_relative_eq!(b, 2.0 * a, epsilon = 1e-12);
    }

    #[test]
    fn realistic_spur_pair() {
        // Paire droite standard : m=5, α=20°, z1=20, z2=40.
        // rp=50, rg=100, C=rp+rg=150 ; addendum = m → ra=r+m ; rb=r·cos α.
        let m = 5.0_f64;
        let alpha = 20.0_f64.to_radians();
        let rp = m * 20.0 / 2.0; // 50
        let rg = m * 40.0 / 2.0; // 100
        let c = rp + rg; // 150
        let z = gear_contact_length_of_action(
            rp + m,
            rp * alpha.cos(),
            rg + m,
            rg * alpha.cos(),
            c,
            alpha,
        );
        let pb = gear_contact_base_pitch(m, alpha);
        let ratio = gear_contact_ratio(z, pb);
        // Valeur calculée à la main : ε ≈ 1,635186.
        assert_relative_eq!(ratio, 1.635_186, epsilon = 1e-5);
        // Rapport de conduite réaliste, supérieur à 1,4 (marche régulière).
        assert!(ratio > 1.4 && ratio < 2.0);
    }

    #[test]
    fn ratio_matches_definition() {
        // Cohérence de composition : ε·pb = Z.
        let z = 24.136_419;
        let pb = 14.760_657;
        assert_relative_eq!(gear_contact_ratio(z, pb) * pb, z, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ra_p > rb_p")]
    fn tip_below_base_circle_panics() {
        // Rayon de tête inférieur au rayon de base : géométrie impossible.
        gear_contact_length_of_action(40.0, 46.98, 105.0, 93.97, 150.0, 20.0_f64.to_radians());
    }
}
