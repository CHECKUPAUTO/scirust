//! Groupes de fixations (boulons/rivets) sous charge **excentrée** dans leur
//! plan — cisaillement primaire (charge directe) et secondaire (moment), puis
//! effort résultant sur la fixation critique.
//!
//! ```text
//! cisaillement primaire   F_p = F/n                (réparti également)
//! moment                  M = F·e                  (e = excentricité)
//! moment polaire du groupe J = Σ r_i²              (aires unitaires)
//! cisaillement secondaire F_s = M·r/J              (⊥ au rayon)
//! résultante              F_R = √(F_p² + F_s² + 2·F_p·F_s·cosθ)
//! ```
//!
//! `F` effort appliqué (N), `n` nombre de fixations, `e` excentricité au
//! barycentre du groupe, `r` distance d'une fixation au barycentre, `θ` angle
//! entre les cisaillements primaire et secondaire à la fixation considérée. Les
//! fixations étant supposées de même section, on travaille en « aires unitaires »
//! (le moment polaire se réduit à `Σ r_i²`).
//!
//! **Convention** : SI cohérent, effort et distances dans le plan du groupe.
//! **Limite honnête** : hypothèse classique de **rotation rigide** autour du
//! barycentre (fixations élastiques identiques) ; ne traite ni le frottement des
//! pièces, ni la précharge, ni le basculement hors plan.

/// Cisaillement primaire (charge directe) `F_p = F/n` (N).
///
/// Panique si `count == 0`.
pub fn primary_shear(force: f64, count: u32) -> f64 {
    assert!(count > 0, "le groupe doit compter au moins une fixation");
    force / count as f64
}

/// Moment polaire du groupe `J = Σ r_i²` (aires unitaires), `r_i` distances des
/// fixations au barycentre.
///
/// Panique si la liste est vide.
pub fn group_polar_moment(radii: &[f64]) -> f64 {
    assert!(!radii.is_empty(), "au moins une fixation est requise");
    radii.iter().map(|&r| r * r).sum()
}

/// Cisaillement secondaire dû au moment `F_s = M·r/J` (N).
///
/// Panique si `polar_moment <= 0`.
pub fn secondary_shear(moment: f64, radius: f64, polar_moment: f64) -> f64 {
    assert!(
        polar_moment > 0.0,
        "le moment polaire doit être strictement positif"
    );
    moment * radius / polar_moment
}

/// Effort résultant sur une fixation `F_R = √(F_p² + F_s² + 2·F_p·F_s·cosθ)` (N),
/// `θ` angle entre cisaillements primaire et secondaire.
pub fn resultant_shear(primary: f64, secondary: f64, angle_rad: f64) -> f64 {
    (primary * primary + secondary * secondary + 2.0 * primary * secondary * angle_rad.cos()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn primary_shear_splits_evenly() {
        // 12 kN sur 4 boulons → 3 kN chacun.
        assert_relative_eq!(primary_shear(12_000.0, 4), 3000.0, epsilon = 1e-9);
    }

    #[test]
    fn polar_moment_of_a_square_pattern() {
        // 4 boulons à r=0,1 m du barycentre → J = 4·0,01 = 0,04.
        assert_relative_eq!(
            group_polar_moment(&[0.1, 0.1, 0.1, 0.1]),
            0.04,
            epsilon = 1e-12
        );
    }

    #[test]
    fn secondary_shear_from_moment() {
        // M=1000 N·m, r=0,1, J=0,04 → F_s = 1000·0,1/0,04 = 2500 N.
        assert_relative_eq!(secondary_shear(1000.0, 0.1, 0.04), 2500.0, epsilon = 1e-9);
    }

    #[test]
    fn resultant_is_sum_when_aligned() {
        // θ=0 : F_R = F_p + F_s. θ=π : |F_p − F_s|. θ=π/2 : √(F_p²+F_s²).
        assert_relative_eq!(resultant_shear(3000.0, 2500.0, 0.0), 5500.0, epsilon = 1e-6);
        assert_relative_eq!(resultant_shear(3000.0, 2500.0, PI), 500.0, epsilon = 1e-6);
        assert_relative_eq!(
            resultant_shear(3000.0, 2500.0, FRAC_PI_2),
            (3000.0f64 * 3000.0 + 2500.0 * 2500.0).sqrt(),
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "au moins une fixation")]
    fn empty_group_panics() {
        group_polar_moment(&[]);
    }
}
