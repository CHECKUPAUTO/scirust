//! Pompe hydraulique **à engrenages** — cylindrée géométrique approchée, débit
//! théorique et débit réel en fonction du rendement volumétrique.
//!
//! ```text
//! cylindrée         Vd = 2·π·m²·z·b        (m³/tr)
//! débit théorique   Qth = Vd·n             (m³/s)
//! débit réel        Qa  = Qth·ηv           (m³/s)
//! ```
//!
//! `Vd` cylindrée (volume refoulé par tour, m³/tr), `m` module de la denture (m),
//! `z` nombre de dents d'un pignon, `b` largeur de denture (m), `n` fréquence de
//! rotation (tr/s), `Qth` débit théorique (m³/s), `Qa` débit réel (m³/s), `ηv`
//! rendement volumétrique (sans dimension, 0 < ηv ≤ 1).
//!
//! **Convention** : SI cohérent (mètres, secondes) ; le module et la largeur de
//! denture sont exprimés en **mètres** (m = 3 mm → 0,003). **Limite honnête** :
//! la cylindrée `Vd = 2·π·m²·z·b` est la formule usuelle **approchée** (volume des
//! creux assimilé au volume des dents, engrènement idéal, jeu radial négligé) ;
//! elle surestime légèrement la cylindrée réelle mesurée. Le rendement volumétrique
//! traduit globalement les fuites internes et n'est pas modélisé : il est
//! **fourni par l'appelant**, tout comme le module, la denture et la largeur —
//! aucune valeur « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Cylindrée géométrique approchée d'une pompe à engrenages
/// `Vd = 2·π·m²·z·b` (m³/tr).
///
/// `gear_module` module de la denture (m), `teeth` nombre de dents d'un pignon,
/// `face_width` largeur de denture (m).
///
/// Panique si `gear_module < 0`, `face_width < 0` ou `teeth == 0`.
pub fn gearpump_displacement(gear_module: f64, teeth: u32, face_width: f64) -> f64 {
    assert!(
        gear_module >= 0.0,
        "le module de denture ne peut pas être négatif"
    );
    assert!(
        face_width >= 0.0,
        "la largeur de denture ne peut pas être négative"
    );
    assert!(
        teeth > 0,
        "le nombre de dents doit être strictement positif"
    );
    2.0 * PI * gear_module * gear_module * teeth as f64 * face_width
}

/// Débit théorique refoulé par la pompe `Qth = Vd·n` (m³/s).
///
/// `displacement` cylindrée (m³/tr), `speed_rev_per_s` fréquence de rotation (tr/s).
///
/// Panique si `displacement < 0` ou `speed_rev_per_s < 0`.
pub fn gearpump_theoretical_flow(displacement: f64, speed_rev_per_s: f64) -> f64 {
    assert!(
        displacement >= 0.0,
        "la cylindrée ne peut pas être négative"
    );
    assert!(
        speed_rev_per_s >= 0.0,
        "la fréquence de rotation ne peut pas être négative"
    );
    displacement * speed_rev_per_s
}

/// Débit réel en sortie de pompe `Qa = Qth·ηv` (m³/s).
///
/// `theoretical_flow` débit théorique (m³/s), `volumetric_efficiency` rendement
/// volumétrique `ηv` (sans dimension, 0 < ηv ≤ 1).
///
/// Panique si `theoretical_flow < 0`, `volumetric_efficiency <= 0` ou
/// `volumetric_efficiency > 1`.
pub fn gearpump_actual_flow(theoretical_flow: f64, volumetric_efficiency: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn displacement_scales_with_teeth_and_width() {
        // À module fixe, la cylindrée est proportionnelle au produit z·b :
        // doubler le nombre de dents OU la largeur double la cylindrée.
        let m = 0.003_f64;
        let base = gearpump_displacement(m, 12, 0.020);
        assert_relative_eq!(
            gearpump_displacement(m, 24, 0.020),
            2.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            gearpump_displacement(m, 12, 0.040),
            2.0 * base,
            epsilon = 1e-18
        );
    }

    #[test]
    fn displacement_scales_with_module_squared() {
        // La cylindrée varie comme le carré du module : m×2 → cylindrée ×4.
        let base = gearpump_displacement(0.002, 15, 0.025);
        let doubled = gearpump_displacement(0.004, 15, 0.025);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-18);
    }

    #[test]
    fn realistic_displacement_value() {
        // Cas chiffré : m = 3 mm, z = 12, b = 20 mm.
        // Vd = 2π·(0,003)²·12·0,020 ≈ 1,3572×10⁻⁵ m³/tr (≈ 13,6 cm³/tr).
        let vd = gearpump_displacement(0.003, 12, 0.020);
        assert_relative_eq!(vd, 2.0 * PI * 0.003 * 0.003 * 12.0 * 0.020, epsilon = 1e-18);
        assert_relative_eq!(vd, 1.357168_f64 * 1e-5, epsilon = 1e-10);
    }

    #[test]
    fn theoretical_flow_is_displacement_times_speed() {
        // Qth = Vd·n : à 25 tr/s, une cylindrée de 1,3572×10⁻⁵ m³/tr donne
        // Qth ≈ 3,393×10⁻⁴ m³/s. Réciproquement Qth/n = Vd.
        let vd = gearpump_displacement(0.003, 12, 0.020);
        let n = 25.0_f64;
        let qth = gearpump_theoretical_flow(vd, n);
        assert_relative_eq!(qth, vd * n, epsilon = 1e-18);
        assert_relative_eq!(qth / n, vd, epsilon = 1e-18);
    }

    #[test]
    fn perfect_pump_actual_equals_theoretical() {
        // Rendement volumétrique unitaire (ηv = 1) : le débit réel égale le
        // débit théorique. Avec ηv = 0,92, le débit réel vaut 92 % du théorique.
        let qth = 3.4e-4_f64;
        assert_relative_eq!(gearpump_actual_flow(qth, 1.0), qth, epsilon = 1e-18);
        assert_relative_eq!(gearpump_actual_flow(qth, 0.92), 0.92 * qth, epsilon = 1e-18);
    }

    #[test]
    fn actual_flow_never_exceeds_theoretical() {
        // Comme 0 < ηv ≤ 1, le débit réel est toujours inférieur ou égal au
        // débit théorique (les fuites internes ne peuvent qu'abaisser le débit).
        let qth = 5.0e-4_f64;
        let qa = gearpump_actual_flow(qth, 0.88);
        assert!(qa <= qth);
    }

    #[test]
    #[should_panic(expected = "rendement volumétrique ne peut pas dépasser 1")]
    fn efficiency_above_one_panics() {
        gearpump_actual_flow(3.4e-4, 1.05);
    }
}
