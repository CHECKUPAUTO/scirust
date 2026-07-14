//! Angle de pression d'une came à disque avec suiveur translatant radial : angle
//! entre la normale de contact (direction de l'effort transmis) et la direction
//! de déplacement du suiveur.
//!
//! ```text
//! Forme générale (excentricité e) : tan(φ) = (ds/dθ − e) / (r0 + s)
//! Suiveur radial (e = 0)          : tan(φ) = (ds/dθ) / (r0 + s)
//!                                   φ = atan( v_θ / (r0 + s) )
//! Rayon primitif instantané       : r_p = r0 + s
//! ```
//!
//! `φ` angle de pression (rad) ; `ds/dθ = v_θ` gradient de déplacement du suiveur
//! par rapport à l'angle de came (longueur/rad) ; `r0` rayon du cercle de base
//! (longueur) ; `s` déplacement instantané du suiveur (longueur) ; `e`
//! excentricité (longueur) ; `r_p` rayon du cercle primitif instantané
//! (longueur). Lien à la cinématique : `v_θ = v / ω` où `v` est la vitesse
//! linéaire du suiveur (longueur/s) et `ω` la vitesse de rotation (rad/s).
//!
//! **Convention** : longueurs dans une unité cohérente commune (`r0`, `s`, `e`,
//! `v_θ`) ; angles en rad ; `v` en (longueur/s), `ω` en rad/s.
//!
//! **Limite honnête** : modèle géométrique du suiveur translatant **radial**
//! (excentricité nulle dans les fonctions dédiées, ou explicitement fournie dans
//! la forme générale). Ne couvre ni le suiveur oscillant, ni le rayon de galet,
//! ni le rayon de courbure du profil, ni la dynamique de contact. L'angle de
//! pression conditionne l'effort transversal et le risque de coincement : en
//! pratique on surveille un maximum de l'ordre de 30° pour un suiveur
//! translatant, mais ce seuil admissible est une décision de conception
//! **fournie par l'appelant** — aucune valeur limite n'est codée en dur ici.

use core::f64::consts::PI;

/// Angle de pression d'un suiveur translatant **radial** (excentricité nulle) :
/// `φ = atan( v_θ / (r0 + s) )`.
///
/// `velocity_per_rad` est `ds/dθ = v_θ` (longueur/rad), `base_radius` est `r0` et
/// `displacement` est `s` ; toutes les longueurs dans la même unité. Le résultat
/// est en radians et peut être signé selon le signe de `v_θ` (montée positive,
/// descente négative).
///
/// Panique si `base_radius <= 0`, si `displacement < 0`, ou si le rayon primitif
/// `r0 + s` n'est pas strictement positif.
pub fn cam_pressure_angle_rad(velocity_per_rad: f64, base_radius: f64, displacement: f64) -> f64 {
    assert!(
        base_radius > 0.0,
        "le rayon du cercle de base doit être strictement positif"
    );
    assert!(
        displacement >= 0.0,
        "le déplacement du suiveur ne peut pas être négatif"
    );
    let pitch_radius = base_radius + displacement;
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif (r0 + s) doit être strictement positif"
    );
    (velocity_per_rad / pitch_radius).atan()
}

/// Angle de pression d'un suiveur translatant **excentré** (forme générale) :
/// `φ = atan( (v_θ − e) / (r0 + s) )`.
///
/// `velocity_per_rad` est `ds/dθ = v_θ`, `base_radius` est `r0`, `displacement`
/// est `s` et `offset` est l'excentricité `e` (peut être de signe quelconque
/// selon le côté de l'axe). Toutes les longueurs dans la même unité ; résultat en
/// radians.
///
/// Panique si `base_radius <= 0`, si `displacement < 0`, ou si le rayon primitif
/// `r0 + s` n'est pas strictement positif.
pub fn cam_pressure_angle_offset_rad(
    velocity_per_rad: f64,
    base_radius: f64,
    displacement: f64,
    offset: f64,
) -> f64 {
    assert!(
        base_radius > 0.0,
        "le rayon du cercle de base doit être strictement positif"
    );
    assert!(
        displacement >= 0.0,
        "le déplacement du suiveur ne peut pas être négatif"
    );
    let pitch_radius = base_radius + displacement;
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif (r0 + s) doit être strictement positif"
    );
    ((velocity_per_rad - offset) / pitch_radius).atan()
}

/// Gradient de déplacement `v_θ = ds/dθ = v / ω` à partir de la vitesse linéaire
/// du suiveur et de la vitesse de rotation de la came.
///
/// `follower_velocity` est `v` (longueur/s), `angular_velocity` est `ω` (rad/s).
/// Résultat en (longueur/rad), directement utilisable comme `velocity_per_rad`.
///
/// Panique si `angular_velocity == 0` (division par zéro).
pub fn cam_velocity_per_rad(follower_velocity: f64, angular_velocity: f64) -> f64 {
    assert!(
        angular_velocity != 0.0,
        "la vitesse de rotation ne peut pas être nulle (division par zéro)"
    );
    follower_velocity / angular_velocity
}

/// Rayon du cercle primitif instantané `r_p = r0 + s`.
///
/// `base_radius` est `r0`, `displacement` est `s`, même unité. C'est le
/// dénominateur du calcul d'angle de pression.
///
/// Panique si `base_radius <= 0` ou si `displacement < 0`.
pub fn cam_pitch_radius(base_radius: f64, displacement: f64) -> f64 {
    assert!(
        base_radius > 0.0,
        "le rayon du cercle de base doit être strictement positif"
    );
    assert!(
        displacement >= 0.0,
        "le déplacement du suiveur ne peut pas être négatif"
    );
    base_radius + displacement
}

/// Rayon de base minimal `r0` requis pour ne pas dépasser un angle de pression
/// admissible `φ_max`, à un point donné (suiveur radial) :
/// `r0 ≥ v_θ / tan(φ_max) − s`.
///
/// `velocity_per_rad` est `v_θ`, `displacement` est `s`, `max_angle_rad` est le
/// seuil `φ_max` **fourni par l'appelant** (rad). Le résultat peut être négatif
/// si la condition est déjà satisfaite sans marge de rayon de base ; l'appelant
/// interprète alors qu'aucune contrainte de rayon n'est active à ce point.
///
/// Panique si `max_angle_rad <= 0` ou `max_angle_rad >= π/2` (tangente nulle,
/// négative ou infinie hors du domaine physique d'un angle de pression), ou si
/// `displacement < 0`.
pub fn cam_min_base_radius(velocity_per_rad: f64, displacement: f64, max_angle_rad: f64) -> f64 {
    assert!(
        max_angle_rad > 0.0,
        "l'angle de pression admissible doit être strictement positif"
    );
    assert!(
        max_angle_rad < PI / 2.0,
        "l'angle de pression admissible doit être strictement inférieur à π/2"
    );
    assert!(
        displacement >= 0.0,
        "le déplacement du suiveur ne peut pas être négatif"
    );
    velocity_per_rad / max_angle_rad.tan() - displacement
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zero_velocity_gives_zero_angle() {
        // Sans gradient de déplacement, la normale est radiale : φ = 0.
        assert_relative_eq!(cam_pressure_angle_rad(0.0, 40.0, 5.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn reciprocity_tan_matches_definition() {
        // Identité : tan(φ) = v_θ / (r0 + s).
        let v_theta = 12.0_f64;
        let r0 = 30.0_f64;
        let s = 6.0_f64;
        let phi = cam_pressure_angle_rad(v_theta, r0, s);
        assert_relative_eq!(phi.tan(), v_theta / (r0 + s), epsilon = 1e-12);
    }

    #[test]
    fn forty_five_degrees_when_velocity_equals_pitch_radius() {
        // Cas chiffré : si v_θ = r0 + s alors tan(φ) = 1 donc φ = 45° = π/4.
        let r0 = 25.0_f64;
        let s = 15.0_f64; // r0 + s = 40
        let v_theta = 40.0_f64;
        assert_relative_eq!(
            cam_pressure_angle_rad(v_theta, r0, s),
            PI / 4.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn larger_base_radius_reduces_pressure_angle() {
        // Proportionnalité inverse : augmenter r0 réduit φ (monotone).
        let v_theta = 20.0_f64;
        let s = 4.0_f64;
        let small = cam_pressure_angle_rad(v_theta, 20.0, s);
        let large = cam_pressure_angle_rad(v_theta, 80.0, s);
        assert!(large < small);
        assert!(large > 0.0);
    }

    #[test]
    fn offset_reduces_to_radial_when_zero() {
        // La forme excentrée avec e = 0 coïncide avec la forme radiale.
        let v_theta = 18.0_f64;
        let r0 = 50.0_f64;
        let s = 8.0_f64;
        assert_relative_eq!(
            cam_pressure_angle_offset_rad(v_theta, r0, s, 0.0),
            cam_pressure_angle_rad(v_theta, r0, s),
            epsilon = 1e-12
        );
    }

    #[test]
    fn velocity_per_rad_inverts_and_min_base_radius_meets_threshold() {
        // v_θ = v / ω, et le rayon de base minimal atteint exactement φ_max.
        let v = 300.0_f64;
        let omega = 10.0_f64;
        let v_theta = cam_velocity_per_rad(v, omega);
        assert_relative_eq!(v_theta, 30.0, epsilon = 1e-12);

        let s = 5.0_f64;
        let phi_max = 0.5_f64; // rad, seuil fourni
        let r0 = cam_min_base_radius(v_theta, s, phi_max);
        // Avec ce r0 minimal, l'angle de pression égale exactement φ_max.
        let phi = cam_pressure_angle_rad(v_theta, r0, s);
        assert_relative_eq!(phi, phi_max, epsilon = 1e-12);
        // Le rayon primitif est cohérent.
        assert_relative_eq!(cam_pitch_radius(r0, s), r0 + s, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "vitesse de rotation ne peut pas être nulle")]
    fn zero_angular_velocity_panics() {
        cam_velocity_per_rad(5.0, 0.0);
    }
}
