//! Mise en forme — **refoulement** (forgeage à matrice ouverte) : déformation
//! vraie, effort avec frottement et travail de déformation.
//!
//! ```text
//! déformation vraie  ε = ln(h0/hf)
//! effort (frottement) F = Y·A·(1 + µ·d/(3·h))
//! travail idéal      W = Ȳ·V·ε
//! ```
//!
//! `h0`/`hf` hauteurs initiale/finale (m), `Y` contrainte d'écoulement à la
//! déformation courante (Pa), `A` aire instantanée (m²), `µ` frottement à
//! l'interface, `d`/`h` diamètre/hauteur instantanés (m), `Ȳ` contrainte
//! d'écoulement **moyenne** (Pa), `V` volume (m³). Le terme `µ·d/(3h)` traduit le
//! frottement matrice-pièce (effet de tonneau).
//!
//! **Convention** : SI cohérent, volume **constant** (plasticité). **Limite
//! honnête** : refoulement d'un lopin cylindrique, frottement de Coulomb faible ;
//! `Y`/`Ȳ` (courbe d'écrouissage) sont fournis par l'appelant — voir
//! [`crate::true_stress_strain`] pour Hollomon. Pas d'échauffement ni d'anisotropie.

/// Déformation vraie en hauteur `ε = ln(h0/hf)`.
///
/// Panique si `h0 <= 0` ou `hf <= 0`.
pub fn upsetting_true_strain(initial_height: f64, final_height: f64) -> f64 {
    assert!(
        initial_height > 0.0 && final_height > 0.0,
        "h0 > 0 et hf > 0 requis"
    );
    (initial_height / final_height).ln()
}

/// Effort de refoulement avec frottement `F = Y·A·(1 + µ·d/(3·h))` (N).
///
/// Panique si `height <= 0`.
pub fn upsetting_force(
    flow_stress: f64,
    area: f64,
    friction: f64,
    diameter: f64,
    height: f64,
) -> f64 {
    assert!(height > 0.0, "la hauteur doit être strictement positive");
    flow_stress * area * (1.0 + friction * diameter / (3.0 * height))
}

/// Travail de déformation idéal `W = Ȳ·V·ε` (J).
pub fn forming_work(avg_flow_stress: f64, volume: f64, true_strain: f64) -> f64 {
    avg_flow_stress * volume * true_strain
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn true_strain_of_a_halving() {
        // Hauteur divisée par 2 → ε = ln 2 ≈ 0,693.
        assert_relative_eq!(
            upsetting_true_strain(20.0, 10.0),
            2.0f64.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn friction_raises_the_force() {
        // Le terme de frottement augmente l'effort au-dessus de Y·A.
        let frictionless = upsetting_force(300e6, 1e-3, 0.0, 0.05, 0.02);
        let with_friction = upsetting_force(300e6, 1e-3, 0.2, 0.05, 0.02);
        assert_relative_eq!(frictionless, 300e6 * 1e-3, epsilon = 1e-3);
        assert!(with_friction > frictionless);
    }

    #[test]
    fn flatter_billet_has_more_friction_penalty() {
        // Plus la pièce est plate (d/h grand), plus le surcoût de frottement croît.
        let tall = upsetting_force(300e6, 1e-3, 0.2, 0.05, 0.04);
        let flat = upsetting_force(300e6, 1e-3, 0.2, 0.05, 0.01);
        assert!(flat > tall);
    }

    #[test]
    fn forming_work_definition() {
        // Ȳ=250 MPa, V=1e-4 m³, ε=0,7 → W = 17,5 kJ.
        assert_relative_eq!(
            forming_work(250e6, 1e-4, 0.7),
            250e6 * 1e-4 * 0.7,
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "h0 > 0")]
    fn zero_height_strain_panics() {
        upsetting_true_strain(0.0, 10.0);
    }
}
