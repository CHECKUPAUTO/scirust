//! Usinage — **taraudage** : couple de coupe (modèle mécaniste et forme
//! empirique) et puissance de broche associée.
//!
//! ```text
//! section de filet     A  = p·(D − d1)/2            (mm²)
//! rayon moyen          rm = (D + d1)/4              (mm)
//! couple mécaniste     M  = kc·A·rm                 (N·mm)
//! couple empirique     M  = C·D²·p                  (N·mm)
//! puissance            P  = M·ω                     (W, M en N·m, ω en rad/s)
//! ```
//!
//! `kc` effort spécifique de coupe (N/mm²), `p` pas du filet (mm), `d1` diamètre
//! sur fond (mineur, mm), `D` diamètre nominal/majeur (mm), `A` section radiale
//! de filet coupée (mm²), `rm` rayon moyen du filet (mm), `C` coefficient
//! empirique de couple (N/mm³, calé sur essai/table), `M` couple (N·mm dans les
//! deux modèles ci-dessus), `ω` vitesse angulaire de la broche (rad/s), `P`
//! puissance (W).
//!
//! **Convention** : unités de fiche outil (mm, N/mm²) pour les couples ; la
//! puissance suit le SI strict (couple en **N·m**, `ω` en **rad/s** → **W**).
//! **Limite honnête** : modèles d'ingénieur pour un filet triangulaire coupé en
//! une passe ; le couple mécaniste suppose toute la section de filet enlevée par
//! l'arête et un bras de levier au rayon moyen. `kc` et surtout le coefficient
//! empirique `C` sont des données de couple outil/matière FOURNIES par l'appelant
//! (essai, catalogue) : aucune valeur « par défaut » n'est inventée. Ne modélise
//! pas le frottement des flancs, la reprise, le taraud à refouler ni l'usure.

/// Couple de taraudage mécaniste `M = kc·A·rm` avec `A = p·(D − d1)/2` et
/// `rm = (D + d1)/4` (N·mm si `kc` en N/mm² et les dimensions en mm).
///
/// Panique si un argument est négatif, si `pitch <= 0`, ou si
/// `major_diameter <= minor_diameter`.
pub fn tapping_torque_cutting(
    specific_cutting_force: f64,
    pitch: f64,
    minor_diameter: f64,
    major_diameter: f64,
) -> f64 {
    assert!(
        specific_cutting_force >= 0.0,
        "l'effort spécifique de coupe doit être positif ou nul"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    assert!(
        minor_diameter >= 0.0,
        "le diamètre mineur doit être positif ou nul"
    );
    assert!(
        major_diameter > minor_diameter,
        "le diamètre majeur doit être supérieur au diamètre mineur"
    );
    let cross_section = pitch * (major_diameter - minor_diameter) / 2.0;
    let mean_radius = (major_diameter + minor_diameter) / 4.0;
    specific_cutting_force * cross_section * mean_radius
}

/// Couple de taraudage empirique `M = C·D²·p` (N·mm si `C` en N/mm³, `D`,`p` en
/// mm), forme calée sur essai/table via le coefficient `C`.
///
/// Panique si un argument est négatif ou si `pitch <= 0`.
pub fn tapping_torque_empirical(
    cutting_force_coefficient: f64,
    nominal_diameter: f64,
    pitch: f64,
) -> f64 {
    assert!(
        cutting_force_coefficient >= 0.0,
        "le coefficient empirique de couple doit être positif ou nul"
    );
    assert!(
        nominal_diameter >= 0.0,
        "le diamètre nominal doit être positif ou nul"
    );
    assert!(pitch > 0.0, "le pas doit être strictement positif");
    cutting_force_coefficient * nominal_diameter * nominal_diameter * pitch
}

/// Puissance de taraudage `P = M·ω` (W avec `torque` en **N·m** et
/// `angular_speed` en **rad/s**).
///
/// Panique si un argument est négatif.
pub fn tapping_power(torque: f64, angular_speed: f64) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif ou nul");
    assert!(
        angular_speed >= 0.0,
        "la vitesse angulaire doit être positive ou nulle"
    );
    torque * angular_speed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cutting_torque_linear_in_specific_force() {
        // M ∝ kc : doubler l'effort spécifique double le couple.
        let m1 = tapping_torque_cutting(2000.0, 1.25, 6.647, 8.0);
        let m2 = tapping_torque_cutting(4000.0, 1.25, 6.647, 8.0);
        assert_relative_eq!(m2 / m1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn cutting_torque_matches_section_times_radius() {
        // Cas chiffré M8×1,25 : A = 1,25·(8−6,647)/2, rm = (8+6,647)/4.
        let kc = 2500.0_f64;
        let (p, d1, d) = (1.25_f64, 6.647_f64, 8.0_f64);
        let expected = kc * (p * (d - d1) / 2.0) * ((d + d1) / 4.0);
        assert_relative_eq!(
            tapping_torque_cutting(kc, p, d1, d),
            expected,
            epsilon = 1e-9
        );
    }

    #[test]
    fn cutting_torque_vanishes_for_zero_thread_depth() {
        // Cas limite : section de filet quasi nulle → couple → 0.
        let m = tapping_torque_cutting(3000.0, 1.0, 7.999_999, 8.0);
        assert!(m > 0.0 && m < 1e-2);
    }

    #[test]
    fn empirical_torque_scales_as_diameter_squared_and_pitch() {
        // M ∝ D² : diamètre ×2 → couple ×4 ; M ∝ p : pas ×2 → couple ×2.
        let base = tapping_torque_empirical(0.5, 10.0, 1.5);
        let bigger_d = tapping_torque_empirical(0.5, 20.0, 1.5);
        let bigger_p = tapping_torque_empirical(0.5, 10.0, 3.0);
        assert_relative_eq!(bigger_d / base, 4.0, epsilon = 1e-12);
        assert_relative_eq!(bigger_p / base, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn power_is_torque_times_angular_speed() {
        // M=8 N·m, ω=100 rad/s → P = 800 W ; réciprocité M = P/ω.
        let m = 8.0_f64;
        let w = 100.0_f64;
        let p = tapping_power(m, w);
        assert_relative_eq!(p, 800.0, epsilon = 1e-12);
        assert_relative_eq!(p / w, m, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "diamètre majeur doit être supérieur")]
    fn cutting_torque_rejects_inverted_diameters() {
        tapping_torque_cutting(2000.0, 1.25, 8.0, 6.647);
    }
}
