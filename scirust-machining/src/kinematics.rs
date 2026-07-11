//! Cinématique de coupe — conversions vitesse de coupe ↔ fréquence de
//! rotation, vitesse d'avance et débit de copeaux (MRR), pour le tournage,
//! le fraisage et le perçage.
//!
//! Toutes les grandeurs sont en unités du système technique de l'usinage,
//! celles employées sur les fiches outil des fabricants (Sandvik, Kennametal,
//! …) :
//!
//! - vitesse de coupe `Vc` en **m/min**,
//! - diamètre `D` en **mm** (diamètre de la pièce en tournage, de l'outil en
//!   fraisage/perçage),
//! - fréquence de rotation `N` en **tr/min**,
//! - avance `f` par tour (`fn`) ou par dent (`fz`) en **mm**,
//! - vitesse d'avance `Vf` en **mm/min**,
//! - débit de copeaux `Q` en **cm³/min**.
//!
//! Relation fondamentale entre vitesse de coupe et rotation :
//!
//! ```text
//! Vc = π · D · N / 1000       (D en mm, Vc en m/min, N en tr/min)
//! ```
//!
//! **Limite honnête** : ces relations sont la cinématique pure de l'enlèvement
//! de matière — elles ne présument d'aucune valeur de `Vc` ou d'avance
//! « recommandée ». Le choix des paramètres de coupe dépend du couple
//! outil/matière et relève des données du fabricant ou d'essais ; ce module
//! calcule les conséquences géométriques d'un jeu de paramètres donné, pas leur
//! valeur admissible.

use core::f64::consts::PI;

/// Fréquence de rotation `N` (tr/min) pour une vitesse de coupe `vc` (m/min)
/// et un diamètre `diameter` (mm) : `N = 1000·Vc / (π·D)`.
///
/// Panique si `diameter <= 0`.
pub fn spindle_speed_rpm(vc_m_min: f64, diameter_mm: f64) -> f64 {
    assert!(
        diameter_mm > 0.0,
        "le diamètre doit être strictement positif"
    );
    1000.0 * vc_m_min / (PI * diameter_mm)
}

/// Vitesse de coupe `Vc` (m/min) pour une rotation `n` (tr/min) et un diamètre
/// `diameter` (mm) : `Vc = π·D·N / 1000`. Réciproque de [`spindle_speed_rpm`].
pub fn cutting_speed_m_min(n_rpm: f64, diameter_mm: f64) -> f64 {
    PI * diameter_mm * n_rpm / 1000.0
}

/// Vitesse d'avance `Vf` (mm/min) pour une avance par tour `feed_per_rev` (mm)
/// et une rotation `n` (tr/min) : `Vf = f·N`.
///
/// C'est la formule du tournage et du perçage, où l'avance est exprimée par
/// tour. En fraisage, obtenez d'abord l'avance par tour avec
/// [`feed_per_rev_milling`].
pub fn feed_velocity_mm_min(feed_per_rev_mm: f64, n_rpm: f64) -> f64 {
    feed_per_rev_mm * n_rpm
}

/// Avance par tour `fn` (mm) d'une fraise à `teeth` dents tournant à une avance
/// par dent `feed_per_tooth` (mm/dent) : `fn = fz·z`.
///
/// Panique si `teeth == 0`.
pub fn feed_per_rev_milling(feed_per_tooth_mm: f64, teeth: u32) -> f64 {
    assert!(teeth > 0, "une fraise a au moins une dent");
    feed_per_tooth_mm * teeth as f64
}

/// Débit de copeaux en **tournage**, `Q` (cm³/min) : `Q = Vc·ap·fn`.
///
/// Avec `Vc` en m/min, la profondeur de passe `ap` en mm et l'avance par tour
/// `fn` en mm, le produit est directement en cm³/min (la conversion mm³→cm³
/// compense le facteur 1000 des m/min → mm/min).
pub fn mrr_turning_cm3_min(vc_m_min: f64, depth_of_cut_mm: f64, feed_per_rev_mm: f64) -> f64 {
    vc_m_min * depth_of_cut_mm * feed_per_rev_mm
}

/// Débit de copeaux en **fraisage**, `Q` (mm³/min) :
/// `Q = ap · ae · Vf`, avec la profondeur axiale `ap` (mm), l'engagement
/// radial `ae` (mm) et la vitesse d'avance `Vf` (mm/min).
pub fn mrr_milling_mm3_min(
    axial_depth_mm: f64,
    radial_width_mm: f64,
    feed_velocity_mm_min: f64,
) -> f64 {
    axial_depth_mm * radial_width_mm * feed_velocity_mm_min
}

/// Débit de copeaux en **perçage** d'un trou plein, `Q` (mm³/min) :
/// `Q = (π·D²/4) · Vf`, section du trou multipliée par la vitesse d'avance
/// `Vf` (mm/min).
pub fn mrr_drilling_mm3_min(diameter_mm: f64, feed_velocity_mm_min: f64) -> f64 {
    PI * diameter_mm * diameter_mm / 4.0 * feed_velocity_mm_min
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spindle_speed_and_cutting_speed_are_reciprocal() {
        // Vc = 150 m/min sur un Ø100 mm → N ≈ 477,46 tr/min.
        let n = spindle_speed_rpm(150.0, 100.0);
        assert_relative_eq!(n, 477.4648, epsilon = 1e-3);
        // et la réciproque redonne 150 m/min.
        assert_relative_eq!(cutting_speed_m_min(n, 100.0), 150.0, epsilon = 1e-9);
    }

    #[test]
    fn feed_velocity_multiplies_feed_by_speed() {
        // f = 0,2 mm/tr à 500 tr/min → 100 mm/min.
        assert_relative_eq!(feed_velocity_mm_min(0.2, 500.0), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn milling_feed_per_rev_sums_the_teeth() {
        // fz = 0,1 mm/dent, 4 dents → fn = 0,4 mm/tr.
        assert_relative_eq!(feed_per_rev_milling(0.1, 4), 0.4, epsilon = 1e-9);
    }

    #[test]
    fn turning_mrr_is_speed_times_section() {
        // Vc=200 m/min, ap=3 mm, fn=0,3 mm/tr → 180 cm³/min.
        assert_relative_eq!(mrr_turning_cm3_min(200.0, 3.0, 0.3), 180.0, epsilon = 1e-9);
    }

    #[test]
    fn milling_mrr_is_engagement_times_feed() {
        // ap=5, ae=10, Vf=300 mm/min → 15000 mm³/min.
        assert_relative_eq!(
            mrr_milling_mm3_min(5.0, 10.0, 300.0),
            15000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn drilling_mrr_is_hole_section_times_feed() {
        // Ø10 mm, Vf=100 mm/min → (π·100/4)·100 ≈ 7853,98 mm³/min.
        assert_relative_eq!(mrr_drilling_mm3_min(10.0, 100.0), 7853.9816, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "diamètre")]
    fn zero_diameter_panics() {
        spindle_speed_rpm(100.0, 0.0);
    }
}
