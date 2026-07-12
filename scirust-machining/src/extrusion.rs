//! Mise en forme — **extrusion** (filage) : rapport d'extrusion, déformation
//! vraie, pression (corrélation de **Johnson**) et effort.
//!
//! ```text
//! rapport d'extrusion  R = A0/Af
//! déformation vraie    ε = ln(R)
//! pression (Johnson)   p = Ȳ·(a + b·ln R)
//! effort               F = p·A0
//! ```
//!
//! `A0`/`Af` aires du conteneur/produit (m²), `Ȳ` contrainte d'écoulement
//! **moyenne** (Pa), `a`/`b` constantes empiriques de Johnson (`a ≈ 0,8`,
//! `b ≈ 1,2`–`1,5`, incluant frottement et déformation redondante), `p` pression
//! sur le poinçon (Pa), `F` effort (N).
//!
//! **Convention** : SI cohérent. **Limite honnête** : corrélation empirique de
//! **Johnson** pour l'extrusion directe ; `a`, `b` et `Ȳ` sont des données
//! (essais/abaques) fournies par l'appelant. Ne distingue pas extrusion
//! directe/indirecte ni la géométrie de filière en détail.

/// Rapport d'extrusion `R = A0/Af`.
///
/// Panique si `final_area <= 0`.
pub fn extrusion_ratio(initial_area: f64, final_area: f64) -> f64 {
    assert!(
        final_area > 0.0,
        "l'aire de sortie doit être strictement positive"
    );
    initial_area / final_area
}

/// Déformation vraie `ε = ln(R)`.
///
/// Panique si `ratio <= 0`.
pub fn extrusion_true_strain(ratio: f64) -> f64 {
    assert!(
        ratio > 0.0,
        "le rapport d'extrusion doit être strictement positif"
    );
    ratio.ln()
}

/// Pression d'extrusion (Johnson) `p = Ȳ·(a + b·ln R)` (Pa).
///
/// Panique si `ratio <= 0`.
pub fn extrusion_pressure(avg_flow_stress: f64, a: f64, b: f64, ratio: f64) -> f64 {
    assert!(
        ratio > 0.0,
        "le rapport d'extrusion doit être strictement positif"
    );
    avg_flow_stress * (a + b * ratio.ln())
}

/// Effort d'extrusion `F = p·A0` (N).
pub fn extrusion_force(pressure: f64, initial_area: f64) -> f64 {
    pressure * initial_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ratio_and_strain() {
        // A0=2000 mm², Af=500 mm² → R=4, ε = ln4 ≈ 1,386.
        let r = extrusion_ratio(2000e-6, 500e-6);
        assert_relative_eq!(r, 4.0, epsilon = 1e-9);
        assert_relative_eq!(extrusion_true_strain(r), 4.0f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn johnson_pressure_grows_with_ratio() {
        // Plus le rapport est grand, plus la pression est élevée.
        let p4 = extrusion_pressure(200e6, 0.8, 1.5, 4.0);
        let p8 = extrusion_pressure(200e6, 0.8, 1.5, 8.0);
        assert!(p8 > p4);
        assert_relative_eq!(p4, 200e6 * (0.8 + 1.5 * 4.0f64.ln()), epsilon = 1e-3);
    }

    #[test]
    fn force_is_pressure_times_container_area() {
        let p = extrusion_pressure(200e6, 0.8, 1.5, 4.0);
        assert_relative_eq!(extrusion_force(p, 2000e-6), p * 2000e-6, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "aire de sortie")]
    fn zero_exit_area_panics() {
        extrusion_ratio(2000e-6, 0.0);
    }
}
