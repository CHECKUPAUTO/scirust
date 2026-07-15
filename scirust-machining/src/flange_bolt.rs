//! Boulonnage de bride d'appareil à pression (**méthode ASME VIII simplifiée**) —
//! effort de fond, charge d'écrasement du joint, charges de boulonnerie et aire
//! de boulons requise.
//!
//! ```text
//! effort de fond hydrostatique   H     = P·π·G²/4
//! charge d'écrasement en service Hp    = 2·π·G·b·m·P
//! charge boulon en service       Wm1   = H + Hp
//! charge de préserrage du joint  Wm2   = π·G·b·y
//! charge de calcul               W     = max(Wm1, Wm2)
//! aire de boulons requise        Ab    = W / S_bolt
//! ```
//!
//! `P` pression de calcul (Pa), `G` diamètre de réaction du joint (m), `b` largeur
//! de contact effective du joint (m), `m` facteur de joint (sans dimension), `y`
//! contrainte d'assise du joint (Pa), `H`/`Hp`/`Wm1`/`Wm2`/`W` charges (N),
//! `S_bolt` contrainte admissible des boulons (Pa), `Ab` aire de section des
//! boulons (m²).
//!
//! **Convention** : SI cohérent — pressions et contraintes en Pa, longueurs en m,
//! charges en N, aires en m².
//!
//! **Limite honnête** : méthode **ASME VIII Div. 1 simplifiée**, statique, sans
//! flexion de bride, fluage ni relaxation du joint. Le facteur de joint `m` et la
//! contrainte d'assise `y` sont **tabulés** (table du joint, fonction du matériau
//! et de la forme) et **fournis par l'appelant**, de même que le diamètre de
//! réaction `G`, la largeur de contact `b` et la contrainte admissible des boulons
//! `S_bolt` — aucune valeur « par défaut » n'est inventée ici. Distinct de
//! [`crate::bolted_joints`] et de [`crate::gasket_seating`].

use core::f64::consts::PI;

/// Effort de fond hydrostatique `H = P·π·G²/4`.
///
/// `design_pressure` = `P` (Pa), `gasket_reaction_diameter` = `G` (m) ; renvoie
/// une force (N).
///
/// Panique si `design_pressure < 0` ou `gasket_reaction_diameter <= 0`.
pub fn flangebolt_hydrostatic_end_force(
    design_pressure: f64,
    gasket_reaction_diameter: f64,
) -> f64 {
    assert!(
        design_pressure >= 0.0 && gasket_reaction_diameter > 0.0,
        "P ≥ 0 et G > 0 requis"
    );
    design_pressure * PI * gasket_reaction_diameter * gasket_reaction_diameter / 4.0
}

/// Charge d'écrasement du joint en service `Hp = 2·π·G·b·m·P`.
///
/// `gasket_reaction_diameter` = `G` (m), `gasket_contact_width` = `b` (m),
/// `gasket_factor_m` = `m` (sans dimension), `design_pressure` = `P` (Pa) ;
/// renvoie une charge (N).
///
/// Panique si `gasket_reaction_diameter <= 0`, `gasket_contact_width <= 0`,
/// `gasket_factor_m < 0` ou `design_pressure < 0`.
pub fn flangebolt_gasket_load_operating(
    design_pressure: f64,
    gasket_reaction_diameter: f64,
    gasket_contact_width: f64,
    gasket_factor_m: f64,
) -> f64 {
    assert!(
        gasket_reaction_diameter > 0.0
            && gasket_contact_width > 0.0
            && gasket_factor_m >= 0.0
            && design_pressure >= 0.0,
        "G > 0, b > 0, m ≥ 0 et P ≥ 0 requis"
    );
    2.0 * PI * gasket_reaction_diameter * gasket_contact_width * gasket_factor_m * design_pressure
}

/// Charge de boulon en service `Wm1 = H + Hp`.
///
/// `hydrostatic_end_force` = `H` (N), `gasket_operating_load` = `Hp` (N) ; renvoie
/// une charge (N).
///
/// Panique si `hydrostatic_end_force < 0` ou `gasket_operating_load < 0`.
pub fn flangebolt_operating_bolt_load(
    hydrostatic_end_force: f64,
    gasket_operating_load: f64,
) -> f64 {
    assert!(
        hydrostatic_end_force >= 0.0 && gasket_operating_load >= 0.0,
        "H ≥ 0 et Hp ≥ 0 requis"
    );
    hydrostatic_end_force + gasket_operating_load
}

/// Charge de préserrage du joint `Wm2 = π·G·b·y`.
///
/// `gasket_reaction_diameter` = `G` (m), `gasket_contact_width` = `b` (m),
/// `gasket_seating_stress_y` = `y` (Pa) ; renvoie une charge (N).
///
/// Panique si `gasket_reaction_diameter <= 0`, `gasket_contact_width <= 0` ou
/// `gasket_seating_stress_y < 0`.
pub fn flangebolt_seating_bolt_load(
    gasket_reaction_diameter: f64,
    gasket_contact_width: f64,
    gasket_seating_stress_y: f64,
) -> f64 {
    assert!(
        gasket_reaction_diameter > 0.0
            && gasket_contact_width > 0.0
            && gasket_seating_stress_y >= 0.0,
        "G > 0, b > 0 et y ≥ 0 requis"
    );
    PI * gasket_reaction_diameter * gasket_contact_width * gasket_seating_stress_y
}

/// Aire de section des boulons requise `Ab = W / S_bolt`.
///
/// `required_load` = `W` = max(Wm1, Wm2) (N), `allowable_bolt_stress` = `S_bolt`
/// (Pa) ; renvoie une aire (m²). Il incombe à l'appelant de passer la plus
/// contraignante des deux charges de calcul.
///
/// Panique si `required_load < 0` ou `allowable_bolt_stress <= 0`.
pub fn flangebolt_required_bolt_area(required_load: f64, allowable_bolt_stress: f64) -> f64 {
    assert!(
        required_load >= 0.0 && allowable_bolt_stress > 0.0,
        "W ≥ 0 et S_bolt > 0 requis"
    );
    required_load / allowable_bolt_stress
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn end_force_matches_pressure_times_area() {
        // H = P·(π/4)·G² : l'effort de fond est la pression multipliée par
        // l'aire du disque de diamètre G.
        let (pressure, diameter) = (1.0e6_f64, 0.30_f64);
        let area = PI * diameter * diameter / 4.0;
        let h = flangebolt_hydrostatic_end_force(pressure, diameter);
        assert_relative_eq!(h, pressure * area, max_relative = 1e-12);
    }

    #[test]
    fn operating_load_linear_in_width() {
        // Hp ∝ b : doubler la largeur de contact double la charge d'écrasement.
        let (pressure, diameter, width, m) = (0.8e6_f64, 0.25_f64, 0.012_f64, 2.5_f64);
        let base = flangebolt_gasket_load_operating(pressure, diameter, width, m);
        let doubled = flangebolt_gasket_load_operating(pressure, diameter, 2.0 * width, m);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn operating_bolt_load_is_sum() {
        // Wm1 = H + Hp : additivité exacte des deux contributions.
        let (pressure, diameter, width, m) = (1.2e6_f64, 0.40_f64, 0.015_f64, 3.0_f64);
        let h = flangebolt_hydrostatic_end_force(pressure, diameter);
        let hp = flangebolt_gasket_load_operating(pressure, diameter, width, m);
        let wm1 = flangebolt_operating_bolt_load(h, hp);
        assert_relative_eq!(wm1, h + hp, max_relative = 1e-12);
    }

    #[test]
    fn seating_load_linear_in_stress() {
        // Wm2 ∝ y : proportionnalité à la contrainte d'assise du joint.
        let (diameter, width) = (0.30_f64, 0.010_f64);
        let low = flangebolt_seating_bolt_load(diameter, width, 20.0e6_f64);
        let high = flangebolt_seating_bolt_load(diameter, width, 60.0e6_f64);
        assert_relative_eq!(high, 3.0 * low, max_relative = 1e-12);
    }

    #[test]
    fn required_area_recovers_load() {
        // Réciprocité Ab = W/S_bolt ⇒ Ab·S_bolt = W.
        let (load, stress) = (188_495.559_215_f64, 138.0e6_f64);
        let area = flangebolt_required_bolt_area(load, stress);
        assert_relative_eq!(area * stress, load, max_relative = 1e-12);
    }

    #[test]
    fn worked_pressure_vessel_case() {
        // Cas chiffré : P = 1 MPa, G = 0,30 m, b = 0,010 m, m = 2, y = 20 MPa.
        // H  = 1e6·π·0,09/4      = 22500·π   ≈ 70 685,834705 N
        // Hp = 2·π·0,30·0,010·2·1e6 = 12000·π ≈ 37 699,111843 N
        // Wm1 = H + Hp                        ≈ 108 384,946548 N
        // Wm2 = π·0,30·0,010·20e6 = 60000·π   ≈ 188 495,559215 N  (dimensionnant)
        let (pressure, diameter, width, m, y) =
            (1.0e6_f64, 0.30_f64, 0.010_f64, 2.0_f64, 20.0e6_f64);
        let h = flangebolt_hydrostatic_end_force(pressure, diameter);
        let hp = flangebolt_gasket_load_operating(pressure, diameter, width, m);
        let wm1 = flangebolt_operating_bolt_load(h, hp);
        let wm2 = flangebolt_seating_bolt_load(diameter, width, y);
        assert_relative_eq!(h, 70_685.834_705, max_relative = 1e-9);
        assert_relative_eq!(hp, 37_699.111_843, max_relative = 1e-9);
        assert_relative_eq!(wm1, 108_384.946_548, max_relative = 1e-9);
        assert_relative_eq!(wm2, 188_495.559_215, max_relative = 1e-9);
        // La charge de calcul est la plus contraignante des deux.
        let w = wm1.max(wm2);
        assert_relative_eq!(w, wm2, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "S_bolt > 0")]
    fn required_area_rejects_zero_stress() {
        let _ = flangebolt_required_bolt_area(1.0e5_f64, 0.0_f64);
    }
}
