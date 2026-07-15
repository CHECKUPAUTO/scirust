//! Rochet à cliquet (**ratchet & pawl**) — indexation angulaire par pas entiers de
//! dents et couple de maintien statique repris par le cliquet au rayon primitif.
//!
//! ```text
//! pas angulaire        θ_p = 2π / z
//! couple de maintien   T   = F · r
//! effort tangentiel    F   = T / r
//! rayon primitif       r   = T / F
//! dents minimales      z_min = ⌈ 2π / θ_i ⌉
//! ```
//!
//! `z` nombre de dents du rochet (entier) ; `θ_p` pas angulaire entre deux dents
//! consécutives (rad) ; `θ_i` pas d'indexation maximal admissible (rad) ; `z_min`
//! nombre minimal de dents pour ne pas dépasser `θ_i` ; `F` effort tangentiel
//! appliqué au rayon primitif (N) ; `r` rayon primitif du rochet (m) ; `T` couple
//! de maintien statique repris par le cliquet (N·m).
//!
//! **Convention** : SI cohérent (rayon en m, effort en N, couple en N·m, angles en
//! radians).
//!
//! **Limite honnête** : indexation supposée par **pas entiers de dents** réguliers
//! (rochet à denture uniforme), effort tangentiel **statique** appliqué au rayon
//! primitif **fourni par l'appelant**. Ce modèle géométrique et d'équilibre
//! statique ne couvre ni la résistance de la dent ou du cliquet, ni le frottement
//! au contact, ni la dynamique d'engagement (choc, rebond), ni l'angle de pression
//! de la denture. Les propriétés matériaux, contraintes admissibles et seuils de
//! conception sont **fournis par l'appelant** — aucune valeur n'est inventée ici.

use core::f64::consts::TAU;

/// Pas angulaire entre deux dents consécutives d'un rochet : `θ_p = 2π / z` (rad).
///
/// `teeth` est le nombre de dents `z` (entier). Résultat en radians.
///
/// Panique si `teeth == 0` (rochet sans dent, division par zéro).
pub fn ratchet_tooth_pitch_angle(teeth: u32) -> f64 {
    assert!(
        teeth >= 1,
        "le rochet doit comporter au moins une dent (z >= 1)"
    );
    TAU / f64::from(teeth)
}

/// Couple de maintien statique repris par le cliquet : `T = F · r` (N·m).
///
/// `tangential_force` est l'effort tangentiel `F` (N) appliqué au rayon primitif,
/// `pitch_radius` est le rayon primitif `r` (m). Résultat en N·m.
///
/// Panique si `pitch_radius <= 0` ou si `tangential_force < 0`.
pub fn ratchet_holding_torque(tangential_force: f64, pitch_radius: f64) -> f64 {
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif doit être strictement positif"
    );
    assert!(
        tangential_force >= 0.0,
        "l'effort tangentiel ne peut pas être négatif"
    );
    tangential_force * pitch_radius
}

/// Effort tangentiel au rayon primitif à partir du couple de maintien :
/// `F = T / r` (N). Réciproque de [`ratchet_holding_torque`].
///
/// `holding_torque` est le couple `T` (N·m), `pitch_radius` est le rayon primitif
/// `r` (m). Résultat en N.
///
/// Panique si `pitch_radius <= 0` ou si `holding_torque < 0`.
pub fn ratchet_tangential_force(holding_torque: f64, pitch_radius: f64) -> f64 {
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif doit être strictement positif"
    );
    assert!(
        holding_torque >= 0.0,
        "le couple de maintien ne peut pas être négatif"
    );
    holding_torque / pitch_radius
}

/// Rayon primitif à partir du couple de maintien et de l'effort tangentiel :
/// `r = T / F` (m). Réciproque de [`ratchet_holding_torque`].
///
/// `holding_torque` est le couple `T` (N·m), `tangential_force` est l'effort
/// tangentiel `F` (N). Résultat en m.
///
/// Panique si `tangential_force <= 0` ou si `holding_torque < 0`.
pub fn ratchet_pitch_radius(holding_torque: f64, tangential_force: f64) -> f64 {
    assert!(
        tangential_force > 0.0,
        "l'effort tangentiel doit être strictement positif"
    );
    assert!(
        holding_torque >= 0.0,
        "le couple de maintien ne peut pas être négatif"
    );
    holding_torque / tangential_force
}

/// Nombre minimal de dents pour que le pas d'indexation n'excède pas `θ_i` :
/// `z_min = ⌈ 2π / θ_i ⌉`.
///
/// `index_angle_rad` est le pas d'indexation maximal admissible `θ_i` (rad),
/// **fourni par l'appelant**. Le résultat garantit `2π / z_min <= θ_i` (le pas
/// réel du rochet à `z_min` dents ne dépasse pas la valeur demandée). Une petite
/// tolérance amortit le bruit de virgule flottante au voisinage d'un entier exact.
///
/// Panique si `index_angle_rad <= 0`.
pub fn ratchet_min_teeth_for_angle(index_angle_rad: f64) -> u32 {
    assert!(
        index_angle_rad > 0.0,
        "le pas d'indexation admissible doit être strictement positif"
    );
    let raw = TAU / index_angle_rad;
    // Retrait d'une tolérance avant l'arrondi supérieur : évite d'ajouter une dent
    // superflue lorsque 2π/θ_i tombe sur un entier à l'erreur d'arrondi près.
    let teeth = (raw - 1e-9_f64).ceil().max(1.0_f64);
    teeth as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn pitch_angle_full_turn_identity() {
        // Identité : z pas de θ_p couvrent exactement un tour complet (2π).
        for z in [1_u32, 3, 8, 24, 100]
        {
            let theta = ratchet_tooth_pitch_angle(z);
            assert_relative_eq!(theta * f64::from(z), TAU, epsilon = 1e-12);
        }
    }

    #[test]
    fn pitch_angle_eight_teeth_is_45_degrees() {
        // Cas chiffré : 8 dents => θ_p = 2π/8 = π/4 = 45°.
        assert_relative_eq!(ratchet_tooth_pitch_angle(8), PI / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn holding_torque_reciprocity() {
        // Réciprocité T = F·r, F = T/r, r = T/F sur un cas chiffré réaliste :
        // F = 250 N au rayon r = 0.06 m => T = 15 N·m.
        let force = 250.0_f64;
        let radius = 0.06_f64;
        let torque = ratchet_holding_torque(force, radius);
        assert_relative_eq!(torque, 15.0, epsilon = 1e-12);
        assert_relative_eq!(
            ratchet_tangential_force(torque, radius),
            force,
            epsilon = 1e-12
        );
        assert_relative_eq!(ratchet_pitch_radius(torque, force), radius, epsilon = 1e-12);
    }

    #[test]
    fn holding_torque_proportional_to_radius() {
        // Proportionnalité : doubler le rayon double le couple à effort constant.
        let force = 120.0_f64;
        let t1 = ratchet_holding_torque(force, 0.03);
        let t2 = ratchet_holding_torque(force, 0.06);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-12);
    }

    #[test]
    fn min_teeth_meets_angle_constraint() {
        // Cas chiffré : θ_i = 0.5 rad => z_min = ⌈2π/0.5⌉ = ⌈12.566⌉ = 13.
        let index = 0.5_f64;
        let z_min = ratchet_min_teeth_for_angle(index);
        assert_eq!(z_min, 13);
        // Le pas réel à z_min dents ne dépasse pas θ_i...
        assert!(ratchet_tooth_pitch_angle(z_min) <= index);
        // ...mais une dent de moins le dépasserait (minimalité).
        assert!(ratchet_tooth_pitch_angle(z_min - 1) > index);
    }

    #[test]
    fn min_teeth_exact_integer_no_extra_tooth() {
        // Bord exact : θ_i = π/4 = 2π/8 => z_min doit valoir 8, pas 9.
        assert_eq!(ratchet_min_teeth_for_angle(PI / 4.0), 8);
    }

    #[test]
    #[should_panic(expected = "au moins une dent")]
    fn zero_teeth_panics() {
        ratchet_tooth_pitch_angle(0);
    }
}
