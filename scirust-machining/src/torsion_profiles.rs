//! Torsion des profils **non circulaires** (Saint-Venant) — tubes minces fermés
//! (formule de **Bredt**) et sections minces ouvertes (bande rectangulaire).
//!
//! ```text
//! tube mince fermé (Bredt)   τ = T/(2·Am·t)
//!   angle de torsion/longueur θ' = T·s/(4·Am²·G·t)   (épaisseur t constante)
//! bande mince ouverte        J = b·t³/3   τmax = 3·T/(b·t²) = T·t/J
//! rectangle plein            τmax = T/(α·a·b²)   J = β·a·b³   (α,β tabulés)
//! ```
//!
//! `T` couple de torsion (N·m), `Am` aire **enclose par la ligne moyenne** de la
//! paroi (m²), `t` épaisseur de paroi (m), `s` périmètre moyen (m), `G` module de
//! cisaillement (Pa), `b` grand côté et `t`/`a`,`b` dimensions de section pleine.
//! Pour le rectangle plein, `α` et `β` dépendent du rapport `a/b` (tables de
//! Roark) et sont fournis par l'appelant.
//!
//! **Convention** : SI cohérent. **Limite honnête** : torsion de Saint-Venant
//! (gauchissement libre, section constante, élastique linéaire) ; Bredt suppose
//! une paroi **mince** ; la bande ouverte suppose `b ≫ t`. Pas de gauchissement
//! empêché ni de concentrations d'angle.

/// Cisaillement dans un **tube mince fermé** (Bredt) `τ = T/(2·Am·t)` (Pa).
///
/// Panique si `enclosed_area*thickness <= 0`.
pub fn bredt_shear_stress(torque: f64, enclosed_area: f64, thickness: f64) -> f64 {
    assert!(
        enclosed_area * thickness > 0.0,
        "aire enclose et épaisseur doivent être strictement positives"
    );
    torque / (2.0 * enclosed_area * thickness)
}

/// Angle de torsion **par unité de longueur** d'un tube mince fermé à épaisseur
/// constante `θ' = T·s/(4·Am²·G·t)` (rad/m).
///
/// Panique si `enclosed_area`, `g` ou `thickness` rendent le dénominateur nul.
pub fn bredt_twist_rate(
    torque: f64,
    enclosed_area: f64,
    perimeter: f64,
    g: f64,
    thickness: f64,
) -> f64 {
    let denom = 4.0 * enclosed_area * enclosed_area * g * thickness;
    assert!(
        denom > 0.0,
        "aire enclose, module G et épaisseur doivent être strictement positifs"
    );
    torque * perimeter / denom
}

/// Constante de torsion d'une **bande mince ouverte** `J = b·t³/3` (m⁴).
pub fn thin_strip_torsion_constant(b: f64, t: f64) -> f64 {
    b * t.powi(3) / 3.0
}

/// Cisaillement maximal d'une bande mince ouverte `τmax = 3·T/(b·t²)` (Pa).
///
/// Panique si `b*t² <= 0`.
pub fn thin_strip_max_shear(torque: f64, b: f64, t: f64) -> f64 {
    assert!(b * t * t > 0.0, "b et t doivent être strictement positifs");
    3.0 * torque / (b * t * t)
}

/// Cisaillement maximal d'un **rectangle plein** `τmax = T/(α·a·b²)` (Pa),
/// `a` grand côté, `b` petit côté, `α` coefficient tabulé (fonction de `a/b`).
///
/// Panique si `alpha*a*b² <= 0`.
pub fn rectangular_max_shear(torque: f64, a: f64, b: f64, alpha: f64) -> f64 {
    assert!(
        alpha * a * b * b > 0.0,
        "α, a et b doivent être strictement positifs"
    );
    torque / (alpha * a * b * b)
}

/// Constante de torsion d'un rectangle plein `J = β·a·b³` (m⁴), `β` tabulé.
pub fn rectangular_torsion_constant(a: f64, b: f64, beta: f64) -> f64 {
    beta * a * b.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bredt_shear_of_a_square_tube() {
        // Tube carré ligne moyenne 100 mm de côté → Am=0,01 m², t=5 mm, T=1000 N·m.
        // τ = 1000/(2·0,01·0,005) = 10 MPa.
        assert_relative_eq!(bredt_shear_stress(1000.0, 0.01, 0.005), 10e6, epsilon = 1.0);
    }

    #[test]
    fn bredt_twist_rate_is_positive() {
        // Périmètre 0,4 m, G=80 GPa, t=5 mm.
        let tp = bredt_twist_rate(1000.0, 0.01, 0.4, 80e9, 0.005);
        assert_relative_eq!(
            tp,
            1000.0 * 0.4 / (4.0 * 0.01 * 0.01 * 80e9 * 0.005),
            epsilon = 1e-18
        );
        assert!(tp > 0.0);
    }

    #[test]
    fn thin_strip_constant_and_shear() {
        // Bande b=50 mm, t=3 mm : J = 0,05·(0,003)³/3.
        let j = thin_strip_torsion_constant(0.05, 0.003);
        assert_relative_eq!(j, 0.05 * 0.003f64.powi(3) / 3.0, epsilon = 1e-20);
        // τmax = T·t/J doit égaler 3T/(b·t²).
        let t_torque = 20.0;
        let via_j = t_torque * 0.003 / j;
        assert_relative_eq!(
            thin_strip_max_shear(t_torque, 0.05, 0.003),
            via_j,
            epsilon = 1e-3
        );
    }

    #[test]
    fn square_section_roark_coefficients() {
        // Rectangle carré a=b : α=β≈0,208 (Roark). Cohérence dimensionnelle.
        let (a, b, alpha, beta) = (0.02, 0.02, 0.208, 0.1406);
        let tau = rectangular_max_shear(50.0, a, b, alpha);
        assert_relative_eq!(tau, 50.0 / (alpha * a * b * b), epsilon = 1e-6);
        let j = rectangular_torsion_constant(a, b, beta);
        assert_relative_eq!(j, beta * a * b.powi(3), epsilon = 1e-20);
    }

    #[test]
    #[should_panic(expected = "épaisseur")]
    fn bredt_zero_thickness_panics() {
        bredt_shear_stress(1000.0, 0.01, 0.0);
    }
}
