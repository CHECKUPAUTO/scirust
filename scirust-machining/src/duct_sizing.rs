//! Dimensionnement de **gaine** de ventilation (HVAC) — diamètre hydraulique,
//! diamètre circulaire équivalent (Huebscher), vitesse, pression dynamique et
//! perte de charge par frottement (Darcy).
//!
//! ```text
//! diamètre hydraulique     Dh = 4·A / P
//! diamètre équivalent      De = 1,30·(w·h)^0,625 / (w+h)^0,25      (Huebscher)
//! vitesse d'air            v  = Q / A
//! pression dynamique       pv = ½·ρ·v²
//! perte par frottement     Δp = f·(L/Dh)·½·ρ·v²                    (Darcy)
//! ```
//!
//! `A` section (m²), `P` périmètre mouillé (m), `w`,`h` largeur et hauteur du
//! rectangle (m), `Q` débit volumique (m³/s), `v` vitesse (m/s), `ρ` masse
//! volumique de l'air (kg/m³), `f` facteur de frottement de Darcy (sans dim.),
//! `L` longueur de gaine (m), `Dh` diamètre hydraulique (m), `pv`,`Δp` en Pa.
//!
//! **Convention** : unités SI cohérentes.
//! **Limite honnête** : écoulement d'air **incompressible établi**. La masse
//! volumique `ρ` et le facteur de frottement `f` sont **fournis par l'appelant**
//! (aucune valeur par défaut inventée) ; le diamètre équivalent rectangulaire
//! suit la **corrélation empirique de Huebscher** (gaines de forme usuelle).

/// Diamètre hydraulique `Dh = 4·A / P`.
///
/// Panique si `area <= 0` ou `perimeter <= 0`.
pub fn duct_hydraulic_diameter(area: f64, perimeter: f64) -> f64 {
    assert!(
        area > 0.0 && perimeter > 0.0,
        "section et périmètre doivent être strictement positifs"
    );
    4.0 * area / perimeter
}

/// Diamètre circulaire **équivalent** (Huebscher)
/// `De = 1,30·(w·h)^0,625 / (w+h)^0,25`.
///
/// Panique si `width <= 0` ou `height <= 0`.
pub fn duct_rectangular_equivalent_diameter(width: f64, height: f64) -> f64 {
    assert!(
        width > 0.0 && height > 0.0,
        "largeur et hauteur doivent être strictement positives"
    );
    1.30 * (width * height).powf(0.625) / (width + height).powf(0.25)
}

/// Vitesse moyenne de l'air `v = Q / A`.
///
/// Panique si `area <= 0` ou `volumetric_flow < 0`.
pub fn duct_velocity(volumetric_flow: f64, area: f64) -> f64 {
    assert!(area > 0.0, "la section doit être strictement positive");
    assert!(
        volumetric_flow >= 0.0,
        "le débit volumique doit être positif ou nul"
    );
    volumetric_flow / area
}

/// Pression dynamique (vélocité) `pv = ½·ρ·v²`.
///
/// Panique si `air_density <= 0`.
pub fn duct_velocity_pressure(air_density: f64, velocity: f64) -> f64 {
    assert!(
        air_density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    0.5 * air_density * velocity * velocity
}

/// Perte de charge par frottement (Darcy)
/// `Δp = f·(L/Dh)·½·ρ·v²`.
///
/// Panique si `hydraulic_diameter <= 0`, `air_density <= 0`,
/// `friction_factor < 0` ou `length < 0`.
pub fn duct_friction_loss(
    friction_factor: f64,
    length: f64,
    hydraulic_diameter: f64,
    air_density: f64,
    velocity: f64,
) -> f64 {
    assert!(
        hydraulic_diameter > 0.0,
        "le diamètre hydraulique doit être strictement positif"
    );
    assert!(
        air_density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    assert!(
        friction_factor >= 0.0 && length >= 0.0,
        "facteur de frottement et longueur doivent être positifs ou nuls"
    );
    friction_factor * (length / hydraulic_diameter) * 0.5 * air_density * velocity * velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn hydraulic_diameter_of_circle_equals_diameter() {
        // Pour un cercle de diamètre D : A = π·D²/4, P = π·D → Dh = D.
        let d = 0.315_f64;
        let area = PI * d * d / 4.0;
        let perimeter = PI * d;
        assert_relative_eq!(duct_hydraulic_diameter(area, perimeter), d, epsilon = 1e-12);
    }

    #[test]
    fn hydraulic_diameter_of_square_equals_side() {
        // Carré de côté a : A = a², P = 4a → Dh = a.
        let a = 0.25_f64;
        assert_relative_eq!(duct_hydraulic_diameter(a * a, 4.0 * a), a, epsilon = 1e-12);
    }

    #[test]
    fn huebscher_equivalent_diameter_numeric() {
        // Gaine 400 × 200 mm : De = 1,30·(0,08)^0,625/(0,6)^0,25 ≈ 0,304675 m.
        let de = duct_rectangular_equivalent_diameter(0.4, 0.2);
        assert_relative_eq!(de, 0.304_674_973, max_relative = 1e-6);
    }

    #[test]
    fn velocity_is_flow_over_area() {
        // Q = 1 m³/s dans A = 0,25 m² → v = 4 m/s.
        assert_relative_eq!(duct_velocity(1.0, 0.25), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn velocity_pressure_scales_with_square_of_velocity() {
        // pv ∝ v² : doubler v quadruple la pression dynamique.
        let pv1 = duct_velocity_pressure(1.2, 5.0);
        let pv2 = duct_velocity_pressure(1.2, 10.0);
        assert_relative_eq!(pv2 / pv1, 4.0, epsilon = 1e-12);
        // Cas chiffré : ½·1,2·10² = 60 Pa.
        assert_relative_eq!(pv2, 60.0, epsilon = 1e-9);
    }

    #[test]
    fn friction_loss_numeric() {
        // f=0,02 ; L=10 m ; Dh=0,3 m ; ρ=1,2 ; v=8 m/s.
        // Δp = 0,02·(10/0,3)·½·1,2·8² = 0,666…·38,4 = 25,6 Pa.
        let dp = duct_friction_loss(0.02, 10.0, 0.3, 1.2, 8.0);
        assert_relative_eq!(dp, 25.6, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "diamètre hydraulique")]
    fn friction_loss_zero_diameter_panics() {
        duct_friction_loss(0.02, 10.0, 0.0, 1.2, 8.0);
    }
}
