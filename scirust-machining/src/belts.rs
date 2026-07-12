//! Transmissions par courroie — relation de **Euler-Eytelwein** (équation du
//! cabestan) entre les tensions des deux brins, angle d'enroulement et
//! puissance transmissible, pour courroies plates et trapézoïdales.
//!
//! À la limite du glissement, le rapport des tensions brin tendu `T1` / brin mou
//! `T2` sur une poulie vaut :
//!
//! ```text
//! courroie plate :        T1/T2 = e^(μ·β)
//! courroie trapézoïdale : T1/T2 = e^(μ·β / sin γ)
//! ```
//!
//! `μ` coefficient de frottement, `β` angle d'enroulement (rad), `γ` demi-angle
//! de gorge de la courroie trapézoïdale. Le coin de la gorge démultiplie
//! l'adhérence (`sin γ < 1`), d'où la bien meilleure capacité des courroies
//! trapézoïdales.
//!
//! La puissance transmise est l'effort utile (différence de tensions) multiplié
//! par la vitesse linéaire :
//!
//! ```text
//! P = (T1 − T2) · v
//! ```
//!
//! **Convention d'unités** : tensions en N, vitesse `v` en m/s, puissance en W,
//! diamètres en mm, vitesse de rotation en tr/min, angles en radians (sauf le
//! demi-angle de gorge, donné en degrés).
//!
//! **Limite honnête** : modèle statique à la limite du glissement (adhérence
//! pleinement mobilisée). Il ignore la force centrifuge (`T_c = m'·v²`) qui
//! réduit la capacité à grande vitesse, le fluage/glissement fonctionnel, et la
//! fatigue de flexion de la courroie — à prendre en compte séparément selon le
//! type et le fabricant.

use core::f64::consts::PI;

/// Vitesse linéaire de la courroie `v = π·D·n / 60000` (m/s), diamètre de
/// poulie `pulley_diameter` (mm) et vitesse `n` (tr/min).
pub fn belt_speed_m_s(pulley_diameter_mm: f64, n_rpm: f64) -> f64 {
    PI * pulley_diameter_mm * n_rpm / 60_000.0
}

/// Angle d'enroulement sur la **petite poulie** `β = π − 2·asin((D−d)/(2·C))`
/// (rad), pour une transmission ouverte : petit diamètre `small`, grand
/// diamètre `large`, entraxe `center_distance` (mm).
///
/// Panique si `center_distance <= 0` ou si la géométrie est impossible
/// (`(D−d)/(2C) > 1`).
pub fn wrap_angle_small_pulley_rad(small_mm: f64, large_mm: f64, center_distance_mm: f64) -> f64 {
    assert!(
        center_distance_mm > 0.0,
        "l'entraxe doit être strictement positif"
    );
    let ratio = (large_mm - small_mm) / (2.0 * center_distance_mm);
    assert!(ratio.abs() <= 1.0, "géométrie de transmission impossible");
    PI - 2.0 * ratio.asin()
}

/// Rapport des tensions `T1/T2 = e^(μ·β)` d'une **courroie plate**, frottement
/// `mu` et angle d'enroulement `wrap` (rad).
pub fn tension_ratio_flat(mu: f64, wrap_rad: f64) -> f64 {
    (mu * wrap_rad).exp()
}

/// Rapport des tensions `T1/T2 = e^(μ·β / sin γ)` d'une **courroie
/// trapézoïdale**, frottement `mu`, angle d'enroulement `wrap` (rad) et
/// demi-angle de gorge `groove_half_angle` (degrés).
///
/// Panique si `groove_half_angle` n'est pas dans `]0°, 90°]`.
pub fn tension_ratio_vbelt(mu: f64, wrap_rad: f64, groove_half_angle_deg: f64) -> f64 {
    assert!(
        groove_half_angle_deg > 0.0 && groove_half_angle_deg <= 90.0,
        "le demi-angle de gorge doit être dans ]0°, 90°]"
    );
    (mu * wrap_rad / groove_half_angle_deg.to_radians().sin()).exp()
}

/// Puissance transmissible `P = (T1 − T2)·v` (W), tensions `t1`, `t2` (N) et
/// vitesse de courroie `speed` (m/s).
pub fn transmissible_power_w(t1_n: f64, t2_n: f64, speed_m_s: f64) -> f64 {
    (t1_n - t2_n) * speed_m_s
}

/// Tension du brin mou `T2 = T1 / (T1/T2)` (N) déduite du brin tendu `t1` et du
/// rapport de tensions `ratio`.
///
/// Panique si `ratio <= 0`.
pub fn slack_tension(t1_n: f64, ratio: f64) -> f64 {
    assert!(
        ratio > 0.0,
        "le rapport de tensions doit être strictement positif"
    );
    t1_n / ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn belt_speed_of_a_200mm_pulley() {
        // Ø200 à 1450 tr/min → v = π·200·1450/60000 ≈ 15,18 m/s.
        assert_relative_eq!(belt_speed_m_s(200.0, 1450.0), 15.184, epsilon = 1e-3);
    }

    #[test]
    fn wrap_angle_is_below_pi_for_unequal_pulleys() {
        // d=100, D=200, C=500 → β = π − 2·asin(0,1) ≈ 2,941 rad.
        let beta = wrap_angle_small_pulley_rad(100.0, 200.0, 500.0);
        assert_relative_eq!(beta, PI - 2.0 * 0.1f64.asin(), epsilon = 1e-9);
        assert!(beta < PI);
    }

    #[test]
    fn flat_belt_ratio_matches_euler_eytelwein() {
        // μ=0,3, β=π → e^(0,3π) ≈ 2,566.
        assert_relative_eq!(
            tension_ratio_flat(0.3, PI),
            (0.3 * PI).exp(),
            epsilon = 1e-9
        );
        assert_relative_eq!(tension_ratio_flat(0.3, PI), 2.566, epsilon = 1e-3);
    }

    #[test]
    fn vbelt_grips_much_better_than_flat_belt() {
        // À γ=17°, le rapport est bien supérieur à la courroie plate équivalente.
        let flat = tension_ratio_flat(0.3, PI);
        let vee = tension_ratio_vbelt(0.3, PI, 17.0);
        assert!(vee > flat);
        assert_relative_eq!(
            vee,
            (0.3 * PI / 17f64.to_radians().sin()).exp(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn power_is_effective_pull_times_speed() {
        // T1=1000, T2=390, v=15,18 → P = 610·15,18 ≈ 9260 W.
        assert_relative_eq!(
            transmissible_power_w(1000.0, 390.0, 15.184),
            610.0 * 15.184,
            epsilon = 1e-6
        );
    }

    #[test]
    fn slack_tension_inverts_the_ratio() {
        // T1=1000, ratio=2,566 → T2 ≈ 389,7 N ; le rapport se reconstruit.
        let ratio = tension_ratio_flat(0.3, PI);
        let t2 = slack_tension(1000.0, ratio);
        assert_relative_eq!(1000.0 / t2, ratio, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "demi-angle de gorge")]
    fn invalid_groove_angle_panics() {
        tension_ratio_vbelt(0.3, PI, 0.0);
    }
}
