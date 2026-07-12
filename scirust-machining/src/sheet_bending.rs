//! Pliage de tôle — **développé** : allongement au pli (bend allowance), retrait
//! (bend deduction) et longueur à plat, d'après le facteur `K` de fibre neutre.
//!
//! ```text
//! allongement au pli   BA = θ·(R + K·t)
//! retrait extérieur    OSSB = tan(θ/2)·(R + t)
//! retrait au pli       BD = 2·OSSB − BA
//! longueur à plat      L = Σ segments + Σ BA   (= Σ segments_ext − Σ BD)
//! ```
//!
//! `θ` angle **de pliage** (rotation, rad), `R` rayon intérieur, `t` épaisseur,
//! `K` facteur de position de la fibre neutre (`0,3`–`0,5` selon `R/t`), `BA`
//! allongement, `OSSB` retrait extérieur, `BD` retrait. La fibre neutre se
//! déplace vers l'intérieur au pliage : la longueur développée diffère de la
//! somme des cotes extérieures.
//!
//! **Convention** : longueurs cohérentes, angles en rad. **Limite honnête** :
//! modèle géométrique du **développé** (fibre neutre à `K·t`) ; ne calcule ni le
//! retour élastique (springback), ni l'effort de pliage, ni l'amincissement. Le
//! facteur `K` est fourni par l'appelant d'après le couple matière/procédé.

/// Allongement au pli `BA = θ·(R + K·t)` (longueur d'arc de la fibre neutre).
///
/// Panique si `bend_angle_rad < 0`.
pub fn bend_allowance(
    bend_angle_rad: f64,
    inner_radius: f64,
    thickness: f64,
    k_factor: f64,
) -> f64 {
    assert!(bend_angle_rad >= 0.0, "l'angle de pliage doit être positif");
    bend_angle_rad * (inner_radius + k_factor * thickness)
}

/// Retrait extérieur (outside setback) `OSSB = tan(θ/2)·(R + t)`.
pub fn outside_setback(bend_angle_rad: f64, inner_radius: f64, thickness: f64) -> f64 {
    (bend_angle_rad / 2.0).tan() * (inner_radius + thickness)
}

/// Retrait au pli `BD = 2·OSSB − BA`.
pub fn bend_deduction(
    bend_angle_rad: f64,
    inner_radius: f64,
    thickness: f64,
    k_factor: f64,
) -> f64 {
    2.0 * outside_setback(bend_angle_rad, inner_radius, thickness)
        - bend_allowance(bend_angle_rad, inner_radius, thickness, k_factor)
}

/// Longueur développée à plat `L = Σ segments (fibre neutre) + Σ BA`.
pub fn developed_length(flat_segments_sum: f64, total_bend_allowance: f64) -> f64 {
    flat_segments_sum + total_bend_allowance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_2;

    #[test]
    fn bend_allowance_is_neutral_fibre_arc() {
        // Pli 90°, R=3, t=2, K=0,44 → BA = (π/2)·(3+0,88) = (π/2)·3,88.
        let ba = bend_allowance(FRAC_PI_2, 3.0, 2.0, 0.44);
        assert_relative_eq!(ba, FRAC_PI_2 * (3.0 + 0.44 * 2.0), epsilon = 1e-12);
    }

    #[test]
    fn setback_at_ninety_degrees() {
        // θ=90° : tan45°=1 → OSSB = R+t = 5.
        assert_relative_eq!(outside_setback(FRAC_PI_2, 3.0, 2.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn bend_deduction_relation() {
        // BD = 2·OSSB − BA.
        let bd = bend_deduction(FRAC_PI_2, 3.0, 2.0, 0.44);
        let expected = 2.0 * 5.0 - FRAC_PI_2 * (3.0 + 0.88);
        assert_relative_eq!(bd, expected, epsilon = 1e-12);
    }

    #[test]
    fn developed_length_adds_allowance() {
        // Deux ailes de 20 (à la fibre neutre) + un pli BA → L = 40 + BA.
        let ba = bend_allowance(FRAC_PI_2, 3.0, 2.0, 0.44);
        assert_relative_eq!(developed_length(40.0, ba), 40.0 + ba, epsilon = 1e-12);
    }

    #[test]
    fn flatter_bend_needs_more_material() {
        // À rayon/épaisseur égaux, un plus grand angle allonge davantage.
        let small = bend_allowance(1.0, 3.0, 2.0, 0.44);
        let large = bend_allowance(2.0, 3.0, 2.0, 0.44);
        assert!(large > small);
    }

    #[test]
    #[should_panic(expected = "angle de pliage")]
    fn negative_angle_panics() {
        bend_allowance(-0.1, 3.0, 2.0, 0.44);
    }
}
