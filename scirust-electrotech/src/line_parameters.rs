//! **Paramètres linéiques d'une ligne aérienne** — inductance et capacité par
//! unité de longueur, réactance linéique et distance moyenne géométrique d'une
//! ligne triphasée, à partir de la géométrie des conducteurs.
//!
//! ```text
//! inductance linéique       L' = (µ0 / 2π)·ln(DMG / RMG)          [H/m]
//! capacité linéique         C' = 2π·ε0 / ln(DMG / r)             [F/m]
//! réactance linéique        X' = 2π·f·L'                          [Ω/m]
//! distance moyenne géom.    DMG = (d_ab·d_bc·d_ca)^(1/3)          [m]
//! ```
//!
//! `L'` inductance par unité de longueur (H/m), `C'` capacité par unité de
//! longueur (F/m), `X'` réactance par unité de longueur (Ω/m), `µ0`
//! perméabilité du vide (H/m), `ε0` permittivité du vide (F/m), `DMG` distance
//! moyenne géométrique entre conducteurs (m), `RMG` rayon moyen géométrique du
//! conducteur — rayon effectif intégrant le flux interne (m), `r` rayon
//! physique du conducteur (m), `f` fréquence du réseau (Hz), `d_ab`, `d_bc`,
//! `d_ca` distances entre les trois phases (m).
//!
//! **Convention** : SI ; longueurs en m, inductance linéique en H/m, capacité
//! linéique en F/m, réactance linéique en Ω/m, fréquence en Hz, perméabilité en
//! H/m, permittivité en F/m ; les logarithmes sont naturels.
//!
//! **Limite honnête** : ligne **aérienne** à conducteurs ronds, effet de sol et
//! effet de proximité **négligés**. La distance moyenne géométrique `DMG` et le
//! rayon moyen géométrique `RMG` (ce dernier intégrant la **correction du flux
//! interne** du conducteur pour l'inductance) sont **fournis par l'appelant**
//! ou calculés à partir de la géométrie réelle de la ligne — aucune valeur de
//! ligne ou de conducteur n'est inventée. La perméabilité et la permittivité du
//! vide sont exposées comme constantes nommées et peuvent aussi être passées
//! explicitement aux fonctions concernées.

/// Perméabilité magnétique du vide `µ0 = 4π·10⁻⁷` (H/m).
pub const VACUUM_PERMEABILITY: f64 = 4.0 * core::f64::consts::PI * 1e-7;

/// Permittivité diélectrique du vide `ε0` (F/m).
pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

/// Inductance par unité de longueur d'une ligne aérienne
/// `L' = (µ0 / 2π)·ln(DMG / RMG)` (H/m), où `RMG` intègre la correction du flux
/// interne du conducteur.
///
/// Panique si `geometric_mean_distance <= 0`, si `geometric_mean_radius <= 0` ou
/// si `vacuum_permeability <= 0`.
pub fn linep_inductance_per_length(
    geometric_mean_distance: f64,
    geometric_mean_radius: f64,
    vacuum_permeability: f64,
) -> f64 {
    assert!(
        geometric_mean_distance > 0.0,
        "la distance moyenne géométrique DMG doit être > 0"
    );
    assert!(
        geometric_mean_radius > 0.0,
        "le rayon moyen géométrique RMG doit être > 0"
    );
    assert!(
        vacuum_permeability > 0.0,
        "la perméabilité du vide µ0 doit être > 0"
    );
    (vacuum_permeability / (2.0 * core::f64::consts::PI))
        * (geometric_mean_distance / geometric_mean_radius).ln()
}

/// Capacité par unité de longueur d'une ligne aérienne
/// `C' = 2π·ε0 / ln(DMG / r)` (F/m), avec `r` le rayon physique du conducteur.
///
/// Panique si `geometric_mean_distance <= 0`, si `conductor_radius <= 0`, si
/// `vacuum_permittivity <= 0` ou si `geometric_mean_distance == conductor_radius`
/// (logarithme nul).
pub fn linep_capacitance_per_length(
    geometric_mean_distance: f64,
    conductor_radius: f64,
    vacuum_permittivity: f64,
) -> f64 {
    assert!(
        geometric_mean_distance > 0.0,
        "la distance moyenne géométrique DMG doit être > 0"
    );
    assert!(
        conductor_radius > 0.0,
        "le rayon du conducteur r doit être > 0"
    );
    assert!(
        vacuum_permittivity > 0.0,
        "la permittivité du vide ε0 doit être > 0"
    );
    assert!(
        (geometric_mean_distance - conductor_radius).abs() > 0.0,
        "DMG et r doivent différer (ln(DMG / r) ne doit pas être nul)"
    );
    2.0 * core::f64::consts::PI * vacuum_permittivity
        / (geometric_mean_distance / conductor_radius).ln()
}

/// Réactance par unité de longueur d'une ligne `X' = 2π·f·L'` (Ω/m).
///
/// Panique si `inductance_per_length < 0` ou si `frequency < 0`.
pub fn linep_reactance_per_length(inductance_per_length: f64, frequency: f64) -> f64 {
    assert!(
        inductance_per_length >= 0.0,
        "l'inductance linéique L' doit être ≥ 0"
    );
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0");
    2.0 * core::f64::consts::PI * frequency * inductance_per_length
}

/// Distance moyenne géométrique d'une ligne **triphasée**
/// `DMG = (d_ab·d_bc·d_ca)^(1/3)` (m).
///
/// Panique si `distance_ab <= 0`, si `distance_bc <= 0` ou si `distance_ca <= 0`.
pub fn linep_geometric_mean_distance_three_phase(
    distance_ab: f64,
    distance_bc: f64,
    distance_ca: f64,
) -> f64 {
    assert!(distance_ab > 0.0, "la distance d_ab doit être > 0");
    assert!(distance_bc > 0.0, "la distance d_bc doit être > 0");
    assert!(distance_ca > 0.0, "la distance d_ca doit être > 0");
    (distance_ab * distance_bc * distance_ca).cbrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Identité : la DMG de trois distances égales à d vaut d.
    #[test]
    fn dmg_of_equal_distances_is_the_distance() {
        assert_relative_eq!(
            linep_geometric_mean_distance_three_phase(3.5, 3.5, 3.5),
            3.5,
            epsilon = 1e-12
        );
    }

    // Cas chiffré CALCULÉ : DMG(1, 2, 4) = (1·2·4)^(1/3) = 8^(1/3) = 2.
    // Recalcul : produit = 8, racine cubique de 8 = 2.
    #[test]
    fn dmg_three_phase_numeric() {
        assert_relative_eq!(
            linep_geometric_mean_distance_three_phase(1.0, 2.0, 4.0),
            2.0,
            epsilon = 1e-12
        );
    }

    // Identité : la DMG est symétrique par permutation des distances.
    #[test]
    fn dmg_is_symmetric() {
        let a = linep_geometric_mean_distance_three_phase(1.2, 2.4, 3.6);
        let b = linep_geometric_mean_distance_three_phase(3.6, 1.2, 2.4);
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }

    // Proportionnalité : X' = 2π·f·L' est proportionnelle à la fréquence.
    #[test]
    fn reactance_is_proportional_to_frequency() {
        let l = 1.0e-6;
        let x50 = linep_reactance_per_length(l, 50.0);
        let x100 = linep_reactance_per_length(l, 100.0);
        assert_relative_eq!(x100, 2.0 * x50, epsilon = 1e-12);
    }

    // Cas chiffré CALCULÉ : L' = (µ0/2π)·ln(DMG/RMG) avec DMG = 2, RMG = 0.01.
    // µ0/2π = (4π·10⁻⁷)/(2π) = 2·10⁻⁷.  ln(2/0.01) = ln(200) = 5.298317366548036.
    // L' = 2·10⁻⁷ · 5.298317366548036 = 1.0596634733096072e-6 H/m.
    // Recalcul : 2e-7 × 5.298317366548036 = 1.0596634733096072e-6.
    #[test]
    fn inductance_per_length_numeric() {
        let l = linep_inductance_per_length(2.0, 0.01, VACUUM_PERMEABILITY);
        assert_relative_eq!(l, 1.0596634733096072e-6, epsilon = 1e-9);
    }

    // Identité de définition : C'·ln(DMG/r) = 2π·ε0 (retour à la constante).
    #[test]
    fn capacitance_recovers_two_pi_epsilon() {
        let dmg = 2.0;
        let r = 0.01;
        let c = linep_capacitance_per_length(dmg, r, VACUUM_PERMITTIVITY);
        let recovered = c * (dmg / r).ln();
        assert_relative_eq!(
            recovered,
            2.0 * core::f64::consts::PI * VACUUM_PERMITTIVITY,
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "le rayon moyen géométrique RMG doit être > 0")]
    fn inductance_rejects_zero_radius() {
        let _ = linep_inductance_per_length(2.0, 0.0, VACUUM_PERMEABILITY);
    }
}
