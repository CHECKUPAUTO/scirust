//! Mise en forme — **formage par étirage** (stretch forming) : déformation vraie,
//! effort d'étirage et épaisseur après étirage à volume constant.
//!
//! ```text
//! déformation vraie    ε = ln(Lf/L0)
//! effort d'étirage     F = σ·A
//! épaisseur finale     tf = t0·exp(-ε)   (conservation du volume, uniaxial)
//! ```
//!
//! `L0`/`Lf` longueurs initiale/finale (m), `ε` déformation vraie logarithmique
//! (sans dimension), `σ` contrainte d'écoulement à la déformation courante (Pa),
//! `A` aire de la section instantanée (m²), `t0`/`tf` épaisseurs initiale/finale
//! (m). En étirage uniaxial à volume constant, l'allongement `exp(ε)` s'accompagne
//! d'une réduction d'épaisseur `exp(-ε)` (la contraction transverse se partage
//! entre largeur et épaisseur ; ici on la reporte entièrement sur l'épaisseur).
//!
//! **Convention** : SI cohérent, volume **constant** (plasticité). **Limite
//! honnête** : étirage uniaxial, déformation supposée homogène, striction (localisation)
//! **non** modélisée ; la contrainte d'écoulement `σ` (courbe d'écrouissage) est
//! fournie par l'appelant — voir [`crate::true_stress_strain`] pour Hollomon. Aucune
//! constante matériau/procédé n'est inventée ici.

/// Déformation vraie en étirage `ε = ln(Lf/L0)`.
///
/// Positive en allongement (`Lf > L0`).
///
/// Panique si `initial_length <= 0` ou `final_length <= 0`.
pub fn stretch_true_strain(initial_length: f64, final_length: f64) -> f64 {
    assert!(
        initial_length > 0.0 && final_length > 0.0,
        "L0 > 0 et Lf > 0 requis"
    );
    (final_length / initial_length).ln()
}

/// Effort d'étirage `F = σ·A` (N).
///
/// Panique si `cross_section_area < 0`.
pub fn stretch_force(flow_stress: f64, cross_section_area: f64) -> f64 {
    assert!(
        cross_section_area >= 0.0,
        "l'aire de section doit être positive ou nulle"
    );
    flow_stress * cross_section_area
}

/// Épaisseur après étirage `tf = t0·exp(-ε)` (m), à volume constant (uniaxial).
///
/// Une déformation positive (allongement) amincit la tôle.
///
/// Panique si `initial_thickness < 0`.
pub fn stretch_thickness_after(initial_thickness: f64, true_strain: f64) -> f64 {
    assert!(
        initial_thickness >= 0.0,
        "l'épaisseur initiale doit être positive ou nulle"
    );
    initial_thickness * (-true_strain).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn true_strain_of_a_doubling() {
        // Longueur doublée → ε = ln 2 ≈ 0,693.
        assert_relative_eq!(stretch_true_strain(1.0, 2.0), 2.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn no_stretch_gives_zero_strain() {
        // Lf = L0 → aucune déformation.
        assert_relative_eq!(stretch_true_strain(0.8, 0.8), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn force_is_linear_in_stress_and_area() {
        // F = σ·A : proportionnel séparément à σ et à A.
        assert_relative_eq!(stretch_force(400e6, 2e-4), 400e6 * 2e-4, epsilon = 1e-3);
        assert_relative_eq!(
            stretch_force(400e6, 4e-4),
            2.0 * stretch_force(400e6, 2e-4),
            epsilon = 1e-3
        );
    }

    #[test]
    fn thickness_and_strain_are_reciprocal() {
        // Réciprocité : amincir puis retrouver ε via t0/tf = exp(ε).
        let t0 = 1.5e-3;
        let strain = 0.3;
        let tf = stretch_thickness_after(t0, strain);
        assert!(tf < t0);
        assert_relative_eq!((t0 / tf).ln(), strain, epsilon = 1e-12);
    }

    #[test]
    fn volume_is_conserved_uniaxial() {
        // Volume constant : L0·A0·t0 étiré à Lf conserve L·(section) via l'épaisseur.
        // Ici on vérifie L0·t0 = Lf·tf (largeur supposée constante, uniaxial pur).
        let l0 = 0.5;
        let t0 = 2.0e-3;
        let lf = 0.65;
        let strain = stretch_true_strain(l0, lf);
        let tf = stretch_thickness_after(t0, strain);
        assert_relative_eq!(l0 * t0, lf * tf, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "L0 > 0")]
    fn zero_length_strain_panics() {
        stretch_true_strain(0.0, 2.0);
    }
}
