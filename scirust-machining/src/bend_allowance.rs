//! Développé de tôle pliée (pli à l'air/en Vé) — **longueur développée**,
//! **facteur K** (position de la fibre neutre), **retrait** et **longueur à
//! plat** de la pièce.
//!
//! ```text
//! rayon de fibre neutre  Rn = Ri + K·t
//! longueur développée    BA = θ·(Ri + K·t) = θ·Rn
//! retrait extérieur      OSSB = tan(θ/2)·(Ri + t)
//! déduction de pli       BD = 2·OSSB − BA
//! longueur à plat        L = L1 + L2 − BD
//! ```
//!
//! `θ` angle de pli (rad, angle balayé par la matière dans le pli, ∈ ]0, π[),
//! `Ri` rayon intérieur de pliage (m), `t` épaisseur de la tôle (m),
//! `K` facteur K (adimensionnel, ∈ [0, 0,5] : position relative de la fibre
//! neutre depuis la face intérieure, 0 = face intérieure, 0,5 = mi-épaisseur),
//! `Rn` rayon de la fibre neutre (m), `BA` longueur de fibre neutre dans le pli
//! (m), `OSSB` retrait extérieur (m), `BD` déduction de pli (m), `L1`/`L2`
//! longueurs des ailes mesurées entre arêtes extérieures (m), `L` longueur de
//! flan à plat (m).
//!
//! **Convention** : SI cohérent (m, rad, adimensionnel) ; angle de pli en
//! **radians**. **Limite honnête** : modèle géométrique du **pli à l'air/en
//! Vé** avec fibre neutre à rayon constant ; le **facteur K** (position de la
//! fibre neutre) est **fourni par l'appelant** selon le couple procédé/matériau
//! — aucune valeur « par défaut » n'est inventée. Le **retour élastique**
//! (springback) n'est **pas** modélisé ici (voir `roll_bending`).

use core::f64::consts::PI;

/// Rayon de la fibre neutre `Rn = Ri + K·t` (m).
///
/// Position radiale de la fibre neutre, à la fraction `K` de l'épaisseur
/// mesurée depuis la face intérieure de rayon `Ri`.
///
/// Panique si `inner_radius < 0`, `thickness <= 0` ou `k_factor ∉ [0, 0,5]`.
pub fn bend_neutral_radius(inner_radius: f64, thickness: f64, k_factor: f64) -> f64 {
    assert!(
        inner_radius >= 0.0,
        "le rayon intérieur doit être positif ou nul"
    );
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    assert!(
        (0.0..=0.5).contains(&k_factor),
        "le facteur K doit être dans [0, 0,5]"
    );
    inner_radius + k_factor * thickness
}

/// Longueur développée dans le pli `BA = θ·(Ri + K·t)` (m).
///
/// Longueur de la fibre neutre balayée par l'angle de pli `θ` ; équivaut à
/// `θ·Rn` où `Rn` est le rayon de la fibre neutre.
///
/// Panique si `bend_angle_rad ∉ ]0, π[`, `inner_radius < 0`, `thickness <= 0`
/// ou `k_factor ∉ [0, 0,5]`.
pub fn bend_allowance_length(
    bend_angle_rad: f64,
    inner_radius: f64,
    thickness: f64,
    k_factor: f64,
) -> f64 {
    assert!(
        bend_angle_rad > 0.0 && bend_angle_rad < PI,
        "l'angle de pli doit être dans ]0, π[ radians"
    );
    bend_angle_rad * bend_neutral_radius(inner_radius, thickness, k_factor)
}

/// Retrait extérieur `OSSB = tan(θ/2)·(Ri + t)` (m).
///
/// Distance de l'arête extérieure théorique (intersection des faces
/// extérieures prolongées) à la ligne tangente au rayon extérieur `Ri + t`.
///
/// Panique si `bend_angle_rad ∉ ]0, π[`, `inner_radius < 0` ou `thickness <= 0`.
pub fn bend_outside_setback(bend_angle_rad: f64, inner_radius: f64, thickness: f64) -> f64 {
    assert!(
        bend_angle_rad > 0.0 && bend_angle_rad < PI,
        "l'angle de pli doit être dans ]0, π[ radians"
    );
    assert!(
        inner_radius >= 0.0,
        "le rayon intérieur doit être positif ou nul"
    );
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    (bend_angle_rad / 2.0).tan() * (inner_radius + thickness)
}

/// Déduction de pli `BD = 2·OSSB − BA` (m).
///
/// Correction à retrancher à la somme des ailes mesurées entre arêtes
/// extérieures pour obtenir la longueur à plat ; combine le double retrait
/// extérieur et la longueur développée.
///
/// Panique si `bend_angle_rad ∉ ]0, π[`, `inner_radius < 0`, `thickness <= 0`
/// ou `k_factor ∉ [0, 0,5]`.
pub fn bend_deduction_length(
    bend_angle_rad: f64,
    inner_radius: f64,
    thickness: f64,
    k_factor: f64,
) -> f64 {
    let setback = bend_outside_setback(bend_angle_rad, inner_radius, thickness);
    let allowance = bend_allowance_length(bend_angle_rad, inner_radius, thickness, k_factor);
    2.0 * setback - allowance
}

/// Longueur de flan à plat `L = L1 + L2 − BD` (m).
///
/// Développé d'une pièce à un seul pli : somme des ailes (mesurées entre arêtes
/// extérieures) moins la déduction de pli.
///
/// Panique si `leg1 <= 0`, `leg2 <= 0` ou si `deduction >= leg1 + leg2`
/// (longueur à plat non strictement positive).
pub fn bend_flat_length(leg1: f64, leg2: f64, deduction: f64) -> f64 {
    assert!(
        leg1 > 0.0,
        "la première aile doit être strictement positive"
    );
    assert!(leg2 > 0.0, "la seconde aile doit être strictement positive");
    assert!(
        deduction < leg1 + leg2,
        "la déduction de pli doit être inférieure à la somme des ailes"
    );
    leg1 + leg2 - deduction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn allowance_equals_angle_times_neutral_radius() {
        // Identité BA = θ·Rn : la longueur développée est exactement l'angle
        // multiplié par le rayon de la fibre neutre.
        let angle = PI / 2.0;
        let (ri, t, k) = (0.003_f64, 0.002_f64, 0.4_f64);
        let ba = bend_allowance_length(angle, ri, t, k);
        let rn = bend_neutral_radius(ri, t, k);
        assert_relative_eq!(ba, angle * rn, epsilon = 1e-15);
    }

    #[test]
    fn allowance_realistic_value() {
        // Pli 90° : Ri = 3 mm, t = 2 mm, K = 0,4.
        // BA = (π/2)·(0,003 + 0,4·0,002) = (π/2)·0,0038 = 0,005969026… m.
        let ba = bend_allowance_length(PI / 2.0, 0.003, 0.002, 0.4);
        assert_relative_eq!(ba, 0.005_969_026_041_820_6, epsilon = 1e-15);
    }

    #[test]
    fn allowance_is_proportional_to_angle() {
        // BA ∝ θ : doubler l'angle double la longueur développée.
        let (ri, t, k) = (0.005_f64, 0.001_f64, 0.33_f64);
        let ba1 = bend_allowance_length(0.5, ri, t, k);
        let ba2 = bend_allowance_length(1.0, ri, t, k);
        assert_relative_eq!(ba2 / ba1, 2.0, epsilon = 1e-15);
    }

    #[test]
    fn deduction_realistic_value() {
        // Même pli 90° : OSSB = tan(45°)·(0,003+0,002) = 0,005 m,
        // BD = 2·0,005 − 0,005969026… = 0,004030974… m.
        let bd = bend_deduction_length(PI / 2.0, 0.003, 0.002, 0.4);
        assert_relative_eq!(bd, 0.004_030_973_958_179_4, epsilon = 1e-14);
    }

    #[test]
    fn flat_length_reciprocity_with_deduction() {
        // Réciprocité : L1 + L2 = L + BD. On reconstitue la somme des ailes.
        let (l1, l2) = (0.05_f64, 0.03_f64);
        let bd = bend_deduction_length(PI / 2.0, 0.003, 0.002, 0.4);
        let flat = bend_flat_length(l1, l2, bd);
        assert_relative_eq!(flat + bd, l1 + l2, epsilon = 1e-15);
    }

    #[test]
    fn neutral_radius_limits() {
        // K = 0 → fibre neutre sur la face intérieure (Rn = Ri).
        assert_relative_eq!(
            bend_neutral_radius(0.004, 0.002, 0.0),
            0.004,
            epsilon = 1e-15
        );
        // K = 0,5 → fibre neutre à mi-épaisseur (Rn = Ri + t/2).
        assert_relative_eq!(
            bend_neutral_radius(0.004, 0.002, 0.5),
            0.004 + 0.002 / 2.0,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic(expected = "facteur K")]
    fn out_of_range_k_factor_panics() {
        bend_neutral_radius(0.003, 0.002, 0.7);
    }
}
