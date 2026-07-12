//! Assemblages **frettés** (ajustements serrés, frettage/emmanchement) — pression
//! de contact d'un arbre plein dans un moyeu (Lamé), effort/couple transmissible
//! par frottement et échauffement de montage.
//!
//! ```text
//! pression de contact (même matériau, arbre plein)
//!   p = δ·E·(Do² − d²) / (2·d·Do²)        (δ = serrage diamétral)
//! effort axial transmissible   Fa = µ·p·π·d·L
//! couple transmissible         C = µ·p·π·d²·L/2 = Fa·d/2
//! contrainte tangentielle moyeu (au diamètre d)  σθ = p·(Do² + d²)/(Do² − d²)
//! échauffement de montage       ΔT = δ/(α·d)
//! ```
//!
//! `δ` serrage **diamétral** (m), `E` module de Young (Pa), `d` diamètre nominal
//! de contact (m), `Do` diamètre extérieur du moyeu (m), `L` longueur emmanchée
//! (m), `µ` coefficient de frottement, `α` dilatation (1/K). La formule de
//! pression suppose arbre et moyeu de **même matériau** et arbre **plein**.
//!
//! **Convention** : SI cohérent. **Limite honnête** : cylindres épais de Lamé,
//! même matériau, arbre plein, régime élastique ; ne traite pas les matériaux
//! différents, l'arbre creux, ni la plastification locale. `µ`, `α`, `E` sont
//! fournis par l'appelant.

use core::f64::consts::PI;

/// Pression de contact d'un arbre plein fretté dans un moyeu de même matériau
/// `p = δ·E·(Do² − d²)/(2·d·Do²)` (Pa), `δ` serrage diamétral.
///
/// Panique si `d <= 0`, `outer_diameter <= d`.
pub fn contact_pressure_same_material(
    diametral_interference: f64,
    d: f64,
    outer_diameter: f64,
    youngs_modulus: f64,
) -> f64 {
    assert!(d > 0.0 && outer_diameter > d, "0 < d < Do requis");
    diametral_interference * youngs_modulus * (outer_diameter * outer_diameter - d * d)
        / (2.0 * d * outer_diameter * outer_diameter)
}

/// Effort axial transmissible par frottement `Fa = µ·p·π·d·L` (N).
pub fn holding_axial_force(mu: f64, pressure: f64, d: f64, length: f64) -> f64 {
    mu * pressure * PI * d * length
}

/// Couple transmissible par frottement `C = µ·p·π·d²·L/2` (N·m).
pub fn holding_torque(mu: f64, pressure: f64, d: f64, length: f64) -> f64 {
    mu * pressure * PI * d * d * length / 2.0
}

/// Contrainte tangentielle (circonférentielle) dans le moyeu au diamètre de
/// contact `σθ = p·(Do² + d²)/(Do² − d²)` (Pa), maximale à l'alésage.
///
/// Panique si `outer_diameter <= d`.
pub fn hub_hoop_stress(pressure: f64, d: f64, outer_diameter: f64) -> f64 {
    assert!(outer_diameter > d, "Do doit être supérieur à d");
    pressure * (outer_diameter * outer_diameter + d * d) / (outer_diameter * outer_diameter - d * d)
}

/// Échauffement du moyeu nécessaire au montage `ΔT = δ/(α·d)` (K).
///
/// Panique si `alpha*d <= 0`.
pub fn assembly_temperature_rise(diametral_interference: f64, alpha: f64, d: f64) -> f64 {
    assert!(alpha * d > 0.0, "α·d doit être strictement positif");
    diametral_interference / (alpha * d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn contact_pressure_grows_with_interference() {
        // δ=30 µm, d=50 mm, Do=100 mm, E=210 GPa.
        let (d, dout, e) = (0.050, 0.100, 210e9);
        let p = contact_pressure_same_material(30e-6, d, dout, e);
        assert_relative_eq!(
            p,
            30e-6 * e * (dout * dout - d * d) / (2.0 * d * dout * dout),
            epsilon = 1e-3
        );
        // Doubler le serrage double la pression (linéaire).
        let p2 = contact_pressure_same_material(60e-6, d, dout, e);
        assert_relative_eq!(p2 / p, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn torque_is_force_times_radius() {
        // C = Fa·d/2.
        let (mu, p, d, l) = (0.12, 50e6, 0.050, 0.060);
        let fa = holding_axial_force(mu, p, d, l);
        assert_relative_eq!(holding_torque(mu, p, d, l), fa * d / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn hoop_stress_exceeds_contact_pressure() {
        // σθ = p·(Do²+d²)/(Do²−d²) > p toujours (facteur > 1).
        let sigma = hub_hoop_stress(50e6, 0.050, 0.100);
        // Do=2d → facteur = (4+1)/(4−1) = 5/3.
        assert_relative_eq!(sigma, 50e6 * 5.0 / 3.0, epsilon = 1.0);
        assert!(sigma > 50e6);
    }

    #[test]
    fn assembly_heating() {
        // δ=30 µm, α=12e-6/K, d=50 mm → ΔT = 30e-6/(12e-6·0,05) = 50 K.
        assert_relative_eq!(
            assembly_temperature_rise(30e-6, 12e-6, 0.050),
            50.0,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "0 < d < Do")]
    fn hub_smaller_than_shaft_panics() {
        contact_pressure_same_material(30e-6, 0.100, 0.050, 210e9);
    }
}
