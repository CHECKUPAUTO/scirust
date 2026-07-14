//! Vis sans fin transporteuse **(vis d'Archimède)** — débit volumétrique
//! théorique et débit massique d'un produit granulaire acheminé par une vis
//! hélicoïdale.
//!
//! ```text
//! débit volumétrique   Qv = (π/4)·D²·p·n·λ     (m³/s)
//! débit massique       Qm = Qv·ρ               (kg/s)
//! ```
//!
//! `Qv` débit volumétrique (m³/s), `D` diamètre extérieur de la vis (m),
//! `p` pas de l'hélice (m, avance par tour), `n` fréquence de rotation (tr/s),
//! `λ` coefficient de remplissage (sans dimension, 0 < λ ≤ 1), `Qm` débit
//! massique (kg/s), `ρ` masse volumique apparente du produit en vrac (kg/m³).
//!
//! **Convention** : SI cohérent (mètres, secondes). Le débit volumétrique
//! assimile le volume acheminé par tour au volume d'un cylindre de diamètre `D`
//! et de longueur égale au pas `p`, pondéré par le taux de remplissage.
//! **Limite honnête** : modèle valable pour un **produit granulaire** en régime
//! établi ; le glissement du produit sur la vis (recirculation, angle de repos,
//! effet de l'inclinaison) est **négligé** — le rendement de transport réel est
//! donc inférieur. Le volume de l'arbre central et l'épaisseur du filet sont
//! ignorés (vis pleine idéale). Le coefficient de remplissage `λ` et la masse
//! volumique apparente `ρ` sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » n'est inventée.

use core::f64::consts::PI;

/// Débit volumétrique théorique d'une vis transporteuse
/// `Qv = (π/4)·D²·p·n·λ` (m³/s).
///
/// `diameter` diamètre extérieur de la vis `D` (m), `pitch` pas de l'hélice `p`
/// (m, avance par tour), `speed_rev_per_s` fréquence de rotation `n` (tr/s),
/// `fill_factor` coefficient de remplissage `λ` (sans dimension, 0 < λ ≤ 1).
///
/// Panique si `diameter < 0`, `pitch < 0`, `speed_rev_per_s < 0`,
/// `fill_factor <= 0` ou `fill_factor > 1`.
pub fn screw_conveyor_volumetric_flow(
    diameter: f64,
    pitch: f64,
    speed_rev_per_s: f64,
    fill_factor: f64,
) -> f64 {
    assert!(
        diameter >= 0.0,
        "le diamètre de la vis ne peut pas être négatif"
    );
    assert!(pitch >= 0.0, "le pas de l'hélice ne peut pas être négatif");
    assert!(
        speed_rev_per_s >= 0.0,
        "la fréquence de rotation ne peut pas être négative"
    );
    assert!(
        fill_factor > 0.0,
        "le coefficient de remplissage doit être strictement positif"
    );
    assert!(
        fill_factor <= 1.0,
        "le coefficient de remplissage ne peut pas dépasser 1"
    );
    (PI / 4.0) * diameter * diameter * pitch * speed_rev_per_s * fill_factor
}

/// Débit massique acheminé par la vis `Qm = Qv·ρ` (kg/s).
///
/// `volumetric_flow` débit volumétrique `Qv` (m³/s), `bulk_density` masse
/// volumique apparente du produit en vrac `ρ` (kg/m³).
///
/// Panique si `volumetric_flow < 0` ou `bulk_density < 0`.
pub fn screw_conveyor_mass_flow(volumetric_flow: f64, bulk_density: f64) -> f64 {
    assert!(
        volumetric_flow >= 0.0,
        "le débit volumétrique ne peut pas être négatif"
    );
    assert!(
        bulk_density >= 0.0,
        "la masse volumique apparente ne peut pas être négative"
    );
    volumetric_flow * bulk_density
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn volumetric_flow_scales_with_diameter_squared() {
        // À pas, vitesse et remplissage fixes, le débit varie comme D² :
        // doubler le diamètre quadruple le débit.
        let base = screw_conveyor_volumetric_flow(0.20, 0.20, 1.0, 0.30);
        let doubled = screw_conveyor_volumetric_flow(0.40, 0.20, 1.0, 0.30);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-18);
    }

    #[test]
    fn volumetric_flow_proportional_to_pitch_speed_and_fill() {
        // Le débit est linéaire vis-à-vis du pas, de la vitesse et du
        // coefficient de remplissage : doubler l'un quelconque double le débit.
        let base = screw_conveyor_volumetric_flow(0.25, 0.25, 1.5, 0.40);
        assert_relative_eq!(
            screw_conveyor_volumetric_flow(0.25, 0.50, 1.5, 0.40),
            2.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            screw_conveyor_volumetric_flow(0.25, 0.25, 3.0, 0.40),
            2.0 * base,
            epsilon = 1e-18
        );
        assert_relative_eq!(
            screw_conveyor_volumetric_flow(0.25, 0.25, 1.5, 0.80),
            2.0 * base,
            epsilon = 1e-18
        );
    }

    #[test]
    fn realistic_volumetric_flow_value() {
        // Cas chiffré : D = 200 mm, p = 200 mm, n = 1 tr/s, λ = 0,30.
        // Qv = (π/4)·0,2²·0,2·1·0,30 ≈ 1,88496×10⁻³ m³/s (≈ 6,79 m³/h).
        let qv = screw_conveyor_volumetric_flow(0.20, 0.20, 1.0, 0.30);
        assert_relative_eq!(
            qv,
            (PI / 4.0) * 0.20 * 0.20 * 0.20 * 1.0 * 0.30,
            epsilon = 1e-18
        );
        assert_relative_eq!(qv, 1.884956_f64 * 1e-3, epsilon = 1e-9);
    }

    #[test]
    fn mass_flow_is_volumetric_times_density() {
        // Qm = Qv·ρ : pour un blé (ρ ≈ 780 kg/m³) et Qv ≈ 1,885×10⁻³ m³/s,
        // Qm ≈ 1,470 kg/s. Réciproquement Qm/ρ redonne le débit volumétrique.
        let qv = screw_conveyor_volumetric_flow(0.20, 0.20, 1.0, 0.30);
        let rho = 780.0_f64;
        let qm = screw_conveyor_mass_flow(qv, rho);
        assert_relative_eq!(qm, qv * rho, epsilon = 1e-18);
        assert_relative_eq!(qm / rho, qv, epsilon = 1e-18);
    }

    #[test]
    fn zero_fill_limit_via_small_factor() {
        // Le remplissage ne peut pas être nul (assert), mais quand λ → 0 le
        // débit tend vers 0 : à λ très faible, le débit reste proportionnel.
        let tiny = screw_conveyor_volumetric_flow(0.30, 0.30, 2.0, 1e-6);
        let full = screw_conveyor_volumetric_flow(0.30, 0.30, 2.0, 1.0);
        assert_relative_eq!(tiny, 1e-6 * full, epsilon = 1e-24);
    }

    #[test]
    fn mass_flow_zero_when_density_zero() {
        // Masse volumique nulle (produit fictif) → débit massique nul,
        // quel que soit le débit volumétrique.
        assert_relative_eq!(screw_conveyor_mass_flow(2.5e-3, 0.0), 0.0, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "coefficient de remplissage ne peut pas dépasser 1")]
    fn fill_factor_above_one_panics() {
        screw_conveyor_volumetric_flow(0.20, 0.20, 1.0, 1.2);
    }
}
