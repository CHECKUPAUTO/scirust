//! Fluidisation d'un lit de particules — vitesse minimale de fluidisation en
//! régime laminaire (corrélation d'Ergun), perte de charge du lit, vitesse
//! terminale de chute libre (Stokes) et plage de fluidisation.
//!
//! ```text
//! vitesse minimale (Ergun laminaire)
//!   U_mf = [ d_p² · (ρ_p − ρ_f) · g / (150 · μ) ]
//!          · [ ε_mf³ · φ² / (1 − ε_mf) ]                       [m·s⁻¹]
//! perte de charge du lit fluidisé
//!   ΔP   = H · (1 − ε) · (ρ_p − ρ_f) · g                       [Pa]
//! vitesse terminale (Stokes, Re ≪ 1)
//!   U_t  = (ρ_p − ρ_f) · g · d_p² / (18 · μ)                   [m·s⁻¹]
//! plage de fluidisation
//!   R    = U_t / U_mf                                          [-]
//! ```
//!
//! `d_p` diamètre de particule [m], `ρ_p` masse volumique de la particule
//! [kg·m⁻³], `ρ_f` masse volumique du fluide [kg·m⁻³], `g` accélération de la
//! pesanteur [m·s⁻²], `μ` viscosité dynamique du fluide [Pa·s], `ε_mf` porosité
//! (fraction de vide) au minimum de fluidisation [sans dimension, 0 < ε_mf < 1],
//! `φ` sphéricité de la particule [sans dimension, 0 < φ ≤ 1] ; `U_mf` vitesse
//! superficielle minimale de fluidisation [m·s⁻¹] ; `H` hauteur du lit [m], `ε`
//! porosité du lit [sans dimension, 0 ≤ ε < 1], `ΔP` perte de charge à travers le
//! lit [Pa] ; `U_t` vitesse terminale de chute de la particule [m·s⁻¹] ; `R`
//! rapport de fluidisation [sans dimension], marge entre l'apparition de la
//! fluidisation et l'entraînement des particules.
//!
//! **Limite honnête** : modèle de lit de particules à l'échelle des **opérations
//! unitaires**. La vitesse minimale de fluidisation suppose le **régime
//! laminaire** (fines particules, `Re_mf ≪ 1`), c'est-à-dire la seule branche
//! visqueuse de l'équation d'Ergun ; le terme d'inertie n'est **pas** ajouté ici.
//! La porosité au minimum de fluidisation `ε_mf` et la **sphéricité** `φ` sont
//! **FOURNIES** par l'appelant (elles dépendent du matériau et de la forme des
//! grains, jamais supposées « par défaut »). La vitesse terminale par la loi de
//! **Stokes** borne la fluidisation avant **entraînement** et n'est valable que
//! pour `Re ≪ 1`. Toutes les **propriétés du fluide** (masse volumique,
//! viscosité) et de la particule sont **FOURNIES** : aucune valeur physique
//! n'est calculée ni inventée par ce module.

/// Vitesse superficielle minimale de fluidisation en **régime laminaire**
/// (branche visqueuse de l'équation d'Ergun)
/// `U_mf = [d_p²·(ρ_p − ρ_f)·g / (150·μ)] · [ε_mf³·φ² / (1 − ε_mf)]` (m·s⁻¹).
///
/// `particle_diameter` (d_p) [m], `particle_density` (ρ_p) [kg·m⁻³],
/// `fluid_density` (ρ_f) [kg·m⁻³], `fluid_viscosity` (μ) [Pa·s], `gravity` (g)
/// [m·s⁻²], `voidage_mf` (ε_mf) porosité au minimum de fluidisation
/// [sans dimension], `sphericity` (φ) sphéricité [sans dimension].
///
/// Panique si `d_p ≤ 0`, si `μ ≤ 0`, si `g ≤ 0`, si `ρ_f < 0`, si
/// `ρ_p ≤ ρ_f` (aucune force motrice ascendante), si `ε_mf` hors de `]0, 1[`, ou
/// si `φ` hors de `]0, 1]`.
pub fn fluidize_minimum_velocity_fine(
    particle_diameter: f64,
    particle_density: f64,
    fluid_density: f64,
    fluid_viscosity: f64,
    gravity: f64,
    voidage_mf: f64,
    sphericity: f64,
) -> f64 {
    assert!(
        particle_diameter > 0.0,
        "d_p > 0 requis (diamètre de particule)"
    );
    assert!(
        fluid_density >= 0.0,
        "ρ_f ≥ 0 requis (masse volumique du fluide)"
    );
    assert!(
        particle_density > fluid_density,
        "ρ_p > ρ_f requis (particule plus dense que le fluide)"
    );
    assert!(fluid_viscosity > 0.0, "μ > 0 requis (viscosité du fluide)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    assert!(
        voidage_mf > 0.0 && voidage_mf < 1.0,
        "0 < ε_mf < 1 requis (porosité au minimum de fluidisation)"
    );
    assert!(
        sphericity > 0.0 && sphericity <= 1.0,
        "0 < φ ≤ 1 requis (sphéricité)"
    );
    (particle_diameter * particle_diameter * (particle_density - fluid_density) * gravity
        / (150.0 * fluid_viscosity))
        * (voidage_mf.powi(3) * sphericity * sphericity / (1.0 - voidage_mf))
}

/// Perte de charge à travers un lit fluidisé
/// `ΔP = H · (1 − ε) · (ρ_p − ρ_f) · g` (Pa), égale au **poids apparent** des
/// solides par unité de section (le lit est soutenu par le fluide).
///
/// `bed_height` (H) [m], `voidage` (ε) porosité du lit [sans dimension],
/// `particle_density` (ρ_p) [kg·m⁻³], `fluid_density` (ρ_f) [kg·m⁻³],
/// `gravity` (g) [m·s⁻²].
///
/// Panique si `H < 0`, si `ε` hors de `[0, 1[`, si `ρ_f < 0`, si `ρ_p < ρ_f`
/// (perte de charge négative non physique), ou si `g ≤ 0`.
pub fn fluidize_pressure_drop(
    bed_height: f64,
    voidage: f64,
    particle_density: f64,
    fluid_density: f64,
    gravity: f64,
) -> f64 {
    assert!(bed_height >= 0.0, "H ≥ 0 requis (hauteur du lit)");
    assert!(
        (0.0..1.0).contains(&voidage),
        "0 ≤ ε < 1 requis (porosité du lit)"
    );
    assert!(
        fluid_density >= 0.0,
        "ρ_f ≥ 0 requis (masse volumique du fluide)"
    );
    assert!(
        particle_density >= fluid_density,
        "ρ_p ≥ ρ_f requis (poids apparent non négatif)"
    );
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    bed_height * (1.0 - voidage) * (particle_density - fluid_density) * gravity
}

/// Vitesse terminale de chute libre d'une particule sphérique en régime de
/// **Stokes** (`Re ≪ 1`)
/// `U_t = (ρ_p − ρ_f) · g · d_p² / (18 · μ)` (m·s⁻¹).
///
/// `particle_diameter` (d_p) [m], `particle_density` (ρ_p) [kg·m⁻³],
/// `fluid_density` (ρ_f) [kg·m⁻³], `fluid_viscosity` (μ) [Pa·s], `gravity` (g)
/// [m·s⁻²].
///
/// Panique si `d_p ≤ 0`, si `μ ≤ 0`, si `g ≤ 0`, si `ρ_f < 0`, ou si
/// `ρ_p ≤ ρ_f` (la particule ne sédimente pas).
pub fn fluidize_terminal_velocity_stokes(
    particle_diameter: f64,
    particle_density: f64,
    fluid_density: f64,
    fluid_viscosity: f64,
    gravity: f64,
) -> f64 {
    assert!(
        particle_diameter > 0.0,
        "d_p > 0 requis (diamètre de particule)"
    );
    assert!(
        fluid_density >= 0.0,
        "ρ_f ≥ 0 requis (masse volumique du fluide)"
    );
    assert!(
        particle_density > fluid_density,
        "ρ_p > ρ_f requis (particule plus dense que le fluide)"
    );
    assert!(fluid_viscosity > 0.0, "μ > 0 requis (viscosité du fluide)");
    assert!(gravity > 0.0, "g > 0 requis (pesanteur)");
    (particle_density - fluid_density) * gravity * particle_diameter * particle_diameter
        / (18.0 * fluid_viscosity)
}

/// Rapport de fluidisation `R = U_t / U_mf` (sans dimension), largeur de la plage
/// de vitesses superficielles entre l'apparition de la fluidisation (`U_mf`) et
/// l'entraînement des particules (`U_t`).
///
/// `terminal_velocity` (U_t) vitesse terminale [m·s⁻¹],
/// `minimum_fluidization_velocity` (U_mf) vitesse minimale de fluidisation
/// [m·s⁻¹].
///
/// Panique si `U_t < 0` ou si `U_mf ≤ 0`.
pub fn fluidize_ratio(terminal_velocity: f64, minimum_fluidization_velocity: f64) -> f64 {
    assert!(
        terminal_velocity >= 0.0,
        "U_t ≥ 0 requis (vitesse terminale)"
    );
    assert!(
        minimum_fluidization_velocity > 0.0,
        "U_mf > 0 requis (vitesse minimale de fluidisation)"
    );
    terminal_velocity / minimum_fluidization_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pressure_drop_is_apparent_weight_per_area() {
        // H = 1 m, ε = 0.5, ρ_p − ρ_f = 2000 kg/m³, g = 9.81 m/s² ⇒
        //   ΔP = 1·0.5·2000·9.81 = 9810 Pa.
        let dp = fluidize_pressure_drop(1.0_f64, 0.5_f64, 2001.2_f64, 1.2_f64, 9.81_f64);
        assert_relative_eq!(dp, 9810.0, max_relative = 1e-12);
    }

    #[test]
    fn pressure_drop_scales_linearly_with_height() {
        // ΔP ∝ H : doubler la hauteur double la perte de charge.
        let single = fluidize_pressure_drop(0.5_f64, 0.4_f64, 2500.0_f64, 1.0_f64, 9.81_f64);
        let double = fluidize_pressure_drop(1.0_f64, 0.4_f64, 2500.0_f64, 1.0_f64, 9.81_f64);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn terminal_velocity_stokes_realistic_case() {
        // d_p = 50 µm, ρ_p = 2600, ρ_f = 1.2, μ = 1.8e-5, g = 9.81 ⇒
        //   U_t = (2598.8)·9.81·(5e-5)² / (18·1.8e-5)
        //       = 25494.228·2.5e-9 / 3.24e-4
        //       = 6.373557e-5 / 3.24e-4 ≈ 0.196715 m/s.
        let u_t = fluidize_terminal_velocity_stokes(
            5.0e-5_f64, 2600.0_f64, 1.2_f64, 1.8e-5_f64, 9.81_f64,
        );
        assert_relative_eq!(u_t, 0.196715, max_relative = 1e-3);
    }

    #[test]
    fn minimum_velocity_fine_realistic_case() {
        // Mêmes propriétés, ε_mf = 0.4, φ = 1 ⇒
        //   A = 2.5e-9·25494.228 / (150·1.8e-5) = 6.373557e-5 / 2.7e-3 = 0.0236058
        //   B = 0.4³·1² / (1−0.4) = 0.064 / 0.6 = 0.1066667
        //   U_mf = 0.0236058·0.1066667 ≈ 0.00251795 m/s.
        let u_mf = fluidize_minimum_velocity_fine(
            5.0e-5_f64, 2600.0_f64, 1.2_f64, 1.8e-5_f64, 9.81_f64, 0.4_f64, 1.0_f64,
        );
        assert_relative_eq!(u_mf, 0.00251795, max_relative = 1e-3);
    }

    #[test]
    fn ratio_matches_analytic_and_reciprocity() {
        // Pour φ = 1, ε_mf = 0.4 le rapport analytique est indépendant des
        // propriétés : R = 150/(18·B) = 150/(18·0.064/0.6) = 150/1.92 = 78.125.
        let u_t = fluidize_terminal_velocity_stokes(
            5.0e-5_f64, 2600.0_f64, 1.2_f64, 1.8e-5_f64, 9.81_f64,
        );
        let u_mf = fluidize_minimum_velocity_fine(
            5.0e-5_f64, 2600.0_f64, 1.2_f64, 1.8e-5_f64, 9.81_f64, 0.4_f64, 1.0_f64,
        );
        let r = fluidize_ratio(u_t, u_mf);
        assert_relative_eq!(r, 78.125, max_relative = 1e-3);
        // Réciprocité : R · U_mf = U_t.
        assert_relative_eq!(r * u_mf, u_t, max_relative = 1e-12);
    }

    #[test]
    fn ratio_simple_value() {
        // Cas simple : U_t = 10, U_mf = 2 ⇒ R = 5.
        assert_relative_eq!(fluidize_ratio(10.0_f64, 2.0_f64), 5.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < ε_mf < 1 requis")]
    fn minimum_velocity_panics_on_invalid_voidage() {
        // ε_mf = 1 ⇒ division par (1 − ε_mf) = 0 ⇒ entrée rejetée.
        let _ = fluidize_minimum_velocity_fine(
            5.0e-5_f64, 2600.0_f64, 1.2_f64, 1.8e-5_f64, 9.81_f64, 1.0_f64, 1.0_f64,
        );
    }
}
