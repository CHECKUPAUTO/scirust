//! Mise en forme — **pliage à la presse** (presse plieuse, matrice en V) :
//! effort de pliage et **retour élastique** (springback).
//!
//! ```text
//! effort de pliage    F = k·(σu·w·t²)/V
//! retour élastique    Ri/Rf = 4·x³ − 3·x + 1,   x = Ri·σy/(E·t)
//! ```
//!
//! `σu` résistance à la traction (Pa), `w` longueur pliée (m), `t` épaisseur (m),
//! `V` ouverture de la matrice en V (m), `k` coefficient de matrice (≈ 1,33 pour
//! un V standard `V ≈ 8·t`), `Ri`/`Rf` rayons avant/après relâchement, `σy` limite
//! élastique, `E` module de Young. Le retour élastique **augmente** le rayon
//! (`Rf > Ri`) et ouvre l'angle.
//!
//! **Convention** : SI cohérent. **Limite honnête** : pliage en l'air en matrice
//! V (effort empirique) ; le facteur `k` et `σu` sont fournis par l'appelant. La
//! formule de springback (Kalpakjian) suppose un pliage élastoplastique pur, sans
//! amincissement ni frottement de matrice.

/// Effort de pliage en l'air `F = k·(σu·w·t²)/V` (N).
///
/// Panique si `die_opening <= 0`.
pub fn bending_force(
    ultimate_strength: f64,
    width: f64,
    thickness: f64,
    die_opening: f64,
    die_factor: f64,
) -> f64 {
    assert!(
        die_opening > 0.0,
        "l'ouverture de matrice doit être strictement positive"
    );
    die_factor * ultimate_strength * width * thickness * thickness / die_opening
}

/// Rapport de retour élastique `Ri/Rf = 4·x³ − 3·x + 1` avec `x = Ri·σy/(E·t)`.
///
/// Renvoie `Ri/Rf ∈ ]0, 1]` : plus il est petit, plus le retour est important.
/// Panique si `youngs_modulus·thickness <= 0`.
pub fn springback_ratio(
    initial_radius: f64,
    yield_stress: f64,
    youngs_modulus: f64,
    thickness: f64,
) -> f64 {
    assert!(
        youngs_modulus * thickness > 0.0,
        "E·t doit être strictement positif"
    );
    let x = initial_radius * yield_stress / (youngs_modulus * thickness);
    4.0 * x.powi(3) - 3.0 * x + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bending_force_scales_with_thickness_squared() {
        // F ∝ t² : doubler l'épaisseur quadruple l'effort.
        let f1 = bending_force(400e6, 1.0, 0.002, 0.016, 1.33);
        let f2 = bending_force(400e6, 1.0, 0.004, 0.016, 1.33);
        assert_relative_eq!(f2 / f1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn bending_force_value() {
        // σu=400 MPa, w=1 m, t=2 mm, V=16 mm, k=1,33.
        let f = bending_force(400e6, 1.0, 0.002, 0.016, 1.33);
        assert_relative_eq!(f, 1.33 * 400e6 * 1.0 * 0.002 * 0.002 / 0.016, epsilon = 1.0);
    }

    #[test]
    fn stiff_thick_sheet_barely_springs_back() {
        // x→0 (rayon petit, tôle épaisse/raide) → Ri/Rf → 1 (retour négligeable).
        let ks = springback_ratio(0.001, 250e6, 210e9, 0.01);
        assert!(ks > 0.99 && ks <= 1.0);
    }

    #[test]
    fn more_springback_for_high_strength_thin_sheet() {
        // Une tôle plus résistante / plus fine revient davantage (Ri/Rf plus petit).
        let stiff = springback_ratio(0.05, 250e6, 210e9, 0.003);
        let springy = springback_ratio(0.05, 1000e6, 210e9, 0.003);
        assert!(springy < stiff);
    }

    #[test]
    #[should_panic(expected = "ouverture de matrice")]
    fn zero_die_opening_panics() {
        bending_force(400e6, 1.0, 0.002, 0.0, 1.33);
    }
}
