//! Pompe hydraulique **à palettes** (déplacement positif) — cylindrée approchée
//! par la géométrie de came, débit théorique et débit réel selon le rendement
//! volumétrique.
//!
//! ```text
//! cylindrée         Vd = 2·π·D·e·b         (m³/tr)
//! débit théorique   Qth = Vd·N/60          (m³/s)
//! débit réel        Qa  = Qth·ηv           (m³/s)
//! rendement vol.    ηv  = Qa/Qth           (sans dimension)
//! ```
//!
//! `Vd` cylindrée (volume refoulé par tour, m³/tr), `D` diamètre de l'anneau de
//! came (m), `e` excentricité rotor/anneau (m), `b` largeur de rotor (m), `N`
//! fréquence de rotation (tr/min), `Qth` débit théorique (m³/s), `Qa` débit réel
//! (m³/s), `ηv` rendement volumétrique (sans dimension, 0 < ηv ≤ 1).
//!
//! **Convention** : SI cohérent (mètres, secondes) ; les longueurs sont exprimées
//! en **mètres** (D = 50 mm → 0,050) et la vitesse en **tours par minute**. La
//! cylindrée `Vd = 2·π·D·e·b` est la formule usuelle **approchée** (couronne de
//! fluide comprise entre rotor et anneau de came assimilée à `2·π·D·e·b`,
//! épaisseur de palettes et jeux négligés) : elle surestime légèrement la
//! cylindrée réelle mesurée. **Limite honnête** : pompe à palettes équilibrée ou
//! déséquilibrée à excentricité **fournie** par l'appelant ; le rendement
//! volumétrique traduit globalement les fuites internes et n'est pas modélisé, il
//! est **fourni par l'appelant**, tout comme la géométrie de came — aucune valeur
//! « par défaut » n'est inventée. Ce module est **distinct** de
//! [`crate::gear_pump`] (engrenage externe) ; régime permanent.

use core::f64::consts::PI;

/// Cylindrée géométrique approchée d'une pompe à palettes
/// `Vd = 2·π·D·e·b` (m³/tr).
///
/// `cam_ring_diameter` diamètre de l'anneau de came `D` (m), `eccentricity`
/// excentricité rotor/anneau `e` (m), `rotor_width` largeur de rotor `b` (m).
///
/// Panique si `cam_ring_diameter < 0`, `eccentricity < 0` ou `rotor_width < 0`.
pub fn vanepump_displacement(cam_ring_diameter: f64, eccentricity: f64, rotor_width: f64) -> f64 {
    assert!(
        cam_ring_diameter >= 0.0,
        "le diamètre de l'anneau de came ne peut pas être négatif"
    );
    assert!(
        eccentricity >= 0.0,
        "l'excentricité ne peut pas être négative"
    );
    assert!(
        rotor_width >= 0.0,
        "la largeur de rotor ne peut pas être négative"
    );
    2.0 * PI * cam_ring_diameter * eccentricity * rotor_width
}

/// Débit théorique refoulé par la pompe `Qth = Vd·N/60` (m³/s).
///
/// `displacement` cylindrée `Vd` (m³/tr), `rotational_speed_rpm` fréquence de
/// rotation `N` (tr/min).
///
/// Panique si `displacement < 0` ou `rotational_speed_rpm < 0`.
pub fn vanepump_theoretical_flow(displacement: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(
        displacement >= 0.0,
        "la cylindrée ne peut pas être négative"
    );
    assert!(
        rotational_speed_rpm >= 0.0,
        "la fréquence de rotation ne peut pas être négative"
    );
    displacement * rotational_speed_rpm / 60.0
}

/// Débit réel en sortie de pompe `Qa = Qth·ηv` (m³/s).
///
/// `theoretical_flow` débit théorique `Qth` (m³/s), `volumetric_efficiency`
/// rendement volumétrique `ηv` (sans dimension, 0 < ηv ≤ 1).
///
/// Panique si `theoretical_flow < 0`, `volumetric_efficiency <= 0` ou
/// `volumetric_efficiency > 1`.
pub fn vanepump_actual_flow(theoretical_flow: f64, volumetric_efficiency: f64) -> f64 {
    assert!(
        theoretical_flow >= 0.0,
        "le débit théorique ne peut pas être négatif"
    );
    assert!(
        volumetric_efficiency > 0.0,
        "le rendement volumétrique doit être strictement positif"
    );
    assert!(
        volumetric_efficiency <= 1.0,
        "le rendement volumétrique ne peut pas dépasser 1"
    );
    theoretical_flow * volumetric_efficiency
}

/// Rendement volumétrique déduit des débits `ηv = Qa/Qth` (sans dimension).
///
/// `actual_flow` débit réel `Qa` (m³/s), `theoretical_flow` débit théorique
/// `Qth` (m³/s).
///
/// Panique si `actual_flow < 0` ou `theoretical_flow <= 0`.
pub fn vanepump_volumetric_efficiency(actual_flow: f64, theoretical_flow: f64) -> f64 {
    assert!(actual_flow >= 0.0, "le débit réel ne peut pas être négatif");
    assert!(
        theoretical_flow > 0.0,
        "le débit théorique doit être strictement positif"
    );
    actual_flow / theoretical_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn displacement_scales_linearly_with_each_factor() {
        // Vd = 2π·D·e·b est linéaire en chacun de ses facteurs : doubler D, e
        // OU b double la cylindrée.
        let base = vanepump_displacement(0.050, 0.004, 0.030);
        assert_relative_eq!(
            vanepump_displacement(0.100, 0.004, 0.030),
            2.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            vanepump_displacement(0.050, 0.008, 0.030),
            2.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            vanepump_displacement(0.050, 0.004, 0.060),
            2.0 * base,
            epsilon = 1e-18
        );
    }

    #[test]
    fn realistic_displacement_value() {
        // Cas chiffré : D = 50 mm, e = 4 mm, b = 30 mm.
        // Vd = 2π·0,050·0,004·0,030 = 2π·6×10⁻⁶ ≈ 3,769911×10⁻⁵ m³/tr
        // (≈ 37,7 cm³/tr).
        let vd = vanepump_displacement(0.050, 0.004, 0.030);
        assert_relative_eq!(vd, 2.0 * PI * 0.050 * 0.004 * 0.030, epsilon = 1e-18);
        assert_relative_eq!(vd, 3.769911_f64 * 1e-5, epsilon = 1e-9);
    }

    #[test]
    fn theoretical_flow_matches_manual_case() {
        // Qth = Vd·N/60. Avec Vd ≈ 3,769911×10⁻⁵ m³/tr et N = 1500 tr/min :
        // Qth = 3,769911×10⁻⁵ · 25 ≈ 9,424778×10⁻⁴ m³/s (≈ 56,5 L/min).
        let vd = vanepump_displacement(0.050, 0.004, 0.030);
        let qth = vanepump_theoretical_flow(vd, 1500.0);
        assert_relative_eq!(qth, vd * 25.0, epsilon = 1e-18);
        assert_relative_eq!(qth, 9.424778_f64 * 1e-4, epsilon = 1e-8);
    }

    #[test]
    fn actual_flow_and_efficiency_are_reciprocal() {
        // ηv = Qa/Qth et Qa = Qth·ηv sont réciproques : partir d'un ηv, calculer
        // le débit réel, puis en redéduire ηv doit redonner la valeur initiale.
        let qth = 9.424778e-4_f64;
        let eta = 0.9_f64;
        let qa = vanepump_actual_flow(qth, eta);
        assert_relative_eq!(qa, eta * qth, epsilon = 1e-18);
        assert_relative_eq!(
            vanepump_volumetric_efficiency(qa, qth),
            eta,
            epsilon = 1e-15
        );
    }

    #[test]
    fn perfect_pump_actual_equals_theoretical() {
        // Rendement volumétrique unitaire (ηv = 1) : le débit réel égale le débit
        // théorique, et ηv redéduit vaut alors exactement 1.
        let qth = 9.4e-4_f64;
        let qa = vanepump_actual_flow(qth, 1.0);
        assert_relative_eq!(qa, qth, epsilon = 1e-18);
        assert_relative_eq!(
            vanepump_volumetric_efficiency(qa, qth),
            1.0,
            epsilon = 1e-18
        );
    }

    #[test]
    fn actual_flow_never_exceeds_theoretical() {
        // Comme 0 < ηv ≤ 1, le débit réel est toujours inférieur ou égal au débit
        // théorique : les fuites internes ne peuvent qu'abaisser le débit.
        let qth = 5.0e-4_f64;
        let qa = vanepump_actual_flow(qth, 0.88);
        assert!(qa <= qth);
    }

    #[test]
    #[should_panic(expected = "rendement volumétrique ne peut pas dépasser 1")]
    fn efficiency_above_one_panics() {
        vanepump_actual_flow(9.4e-4, 1.05);
    }
}
