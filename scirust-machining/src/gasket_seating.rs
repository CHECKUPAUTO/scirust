//! Joints plats sous brides boulonnées (**ASME VIII**, facteurs `m` et `y`) —
//! charge d'assise minimale, charge en service et charge de boulonnerie requise.
//!
//! ```text
//! charge d'assise      W_seat = y·A_c                 (serrage à froid)
//! charge en service    W_op   = m·P·A_g               (maintien de l'étanchéité)
//! poussée hydrostatique H_end  = P·A_e                 (effort de fond)
//! charge de boulons     W_bolt = max(W_seat, W_op + H_end)
//! ```
//!
//! `y` **contrainte d'assise** du joint (Pa), `m` **facteur de maintien** du joint
//! (sans dimension), `A_c` aire de contact du joint (m²), `A_g` aire effective du
//! joint reprenant la pression (m²), `A_e` aire effective soumise à la pression
//! interne / poussée de fond (m², ≈ (π/4)·G² avec `G` diamètre effectif), `P`
//! pression interne (Pa), `W` charges (N), `H_end` poussée hydrostatique (N).
//!
//! **Convention** : SI cohérent — contraintes et pressions en Pa, aires en m²,
//! charges en N.
//!
//! **Limite honnête** : modèle **statique** à deux conditions de calcul (assise à
//! froid et service à chaud), sans fluage/relaxation du joint, sans flexion de
//! bride ni répartition entre boulons. Les facteurs `m` et `y` sont **tabulés**
//! (ASME VIII Div. 1, Tableau 2-5.1) et **dépendent du matériau et de la forme du
//! joint** : ils sont **fournis par l'appelant**, de même que les aires effectives
//! `A_c`, `A_g`, `A_e` — aucune valeur « par défaut » n'est inventée ici.

/// Charge minimale d'assise du joint `W_seat = y·A_c` (serrage à froid).
///
/// `seating_stress` = `y` (Pa), `contact_area` = `A_c` (m²) ; renvoie une charge (N).
///
/// Panique si `seating_stress < 0` ou `contact_area <= 0`.
pub fn gasket_min_seating_load(seating_stress: f64, contact_area: f64) -> f64 {
    assert!(
        seating_stress >= 0.0 && contact_area > 0.0,
        "y ≥ 0 et A_c > 0 requis"
    );
    seating_stress * contact_area
}

/// Charge de maintien en service `W_op = m·P·A_g`.
///
/// `maintenance_factor` = `m` (sans dimension), `internal_pressure` = `P` (Pa),
/// `gasket_area` = `A_g` (m²) ; renvoie une charge (N).
///
/// Panique si `maintenance_factor < 0`, `internal_pressure < 0` ou `gasket_area <= 0`.
pub fn gasket_operating_load(
    maintenance_factor: f64,
    internal_pressure: f64,
    gasket_area: f64,
) -> f64 {
    assert!(
        maintenance_factor >= 0.0 && internal_pressure >= 0.0 && gasket_area > 0.0,
        "m ≥ 0, P ≥ 0 et A_g > 0 requis"
    );
    maintenance_factor * internal_pressure * gasket_area
}

/// Poussée hydrostatique de fond `H_end = P·A_e`.
///
/// `internal_pressure` = `P` (Pa), `effective_area` = `A_e` (m²) ; renvoie une
/// force (N).
///
/// Panique si `internal_pressure < 0` ou `effective_area <= 0`.
pub fn hydrostatic_end_force(internal_pressure: f64, effective_area: f64) -> f64 {
    assert!(
        internal_pressure >= 0.0 && effective_area > 0.0,
        "P ≥ 0 et A_e > 0 requis"
    );
    internal_pressure * effective_area
}

/// Charge de boulonnerie requise `W_bolt = max(W_seat, W_op + H_end)`.
///
/// Retient le plus contraignant des deux cas de calcul : assise à froid ou
/// service (maintien + poussée de fond). Toutes les charges sont en N.
///
/// Panique si l'une des charges est négative.
pub fn required_bolt_load(seating_load: f64, operating_load: f64, end_force: f64) -> f64 {
    assert!(
        seating_load >= 0.0 && operating_load >= 0.0 && end_force >= 0.0,
        "les charges doivent être ≥ 0"
    );
    seating_load.max(operating_load + end_force)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn seating_load_is_stress_times_area() {
        // Joint fibre y = 11 MPa, aire de contact 20 cm² → 22 kN.
        let a = 20e-4_f64;
        let w = gasket_min_seating_load(11e6, a);
        assert_relative_eq!(w, 11e6 * a, epsilon = 1e-6);
        assert_relative_eq!(w, 22e3, epsilon = 1.0);
    }

    #[test]
    fn seating_load_is_linear_in_area() {
        // Doubler l'aire de contact double la charge d'assise.
        let single = gasket_min_seating_load(11e6, 15e-4);
        let double = gasket_min_seating_load(11e6, 30e-4);
        assert_relative_eq!(double, 2.0 * single, epsilon = 1e-6);
    }

    #[test]
    fn operating_load_scales_with_pressure() {
        // W_op = m·P·A_g proportionnel à la pression interne.
        let low = gasket_operating_load(2.0, 1e6, 20e-4);
        let high = gasket_operating_load(2.0, 3e6, 20e-4);
        assert_relative_eq!(high, 3.0 * low, epsilon = 1e-6);
        assert_relative_eq!(low, 2.0 * 1e6 * 20e-4, epsilon = 1e-6);
    }

    #[test]
    fn hydrostatic_end_force_is_pressure_times_area() {
        // Poussée de fond H = P·A_e, cas limite pression nulle → force nulle.
        assert_relative_eq!(hydrostatic_end_force(2e6, 8e-3), 2e6 * 8e-3, epsilon = 1e-6);
        assert_relative_eq!(hydrostatic_end_force(0.0, 8e-3), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn bolt_load_selects_governing_case() {
        // Assise dominante : W_seat > W_op + H_end → retient l'assise.
        assert_relative_eq!(required_bolt_load(50e3, 10e3, 20e3), 50e3, epsilon = 1e-6);
        // Service dominant : W_op + H_end > W_seat → retient le service.
        assert_relative_eq!(required_bolt_load(20e3, 30e3, 25e3), 55e3, epsilon = 1e-6);
    }

    #[test]
    fn realistic_flange_case() {
        // Bride Ø effectif G = 0,1 m, P = 2 MPa, m = 2, y = 11 MPa,
        // aire de contact du joint ≈ 3,3e-3 m², A_g ≈ A_e ≈ (π/4)·G².
        use core::f64::consts::PI;
        let g = 0.10_f64;
        let a_e = PI / 4.0 * g * g;
        let seat = gasket_min_seating_load(11e6, 3.3e-3);
        let op = gasket_operating_load(2.0, 2e6, a_e);
        let end = hydrostatic_end_force(2e6, a_e);
        let bolt = required_bolt_load(seat, op, end);
        // Le service (op + end) doit gouverner ce cas chargé.
        assert_relative_eq!(bolt, op + end, epsilon = 1e-6);
        assert!(bolt > seat);
    }

    #[test]
    #[should_panic(expected = "y ≥ 0")]
    fn negative_seating_stress_panics() {
        gasket_min_seating_load(-1.0, 20e-4);
    }
}
