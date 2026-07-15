//! **Débit sur déversoir** — mesure d'un débit à surface libre à partir de la
//! charge amont sur un déversoir mince en paroi (rectangulaire ou triangulaire).
//!
//! ```text
//! déversoir rectangulaire   Q = (2/3)·Cd·b·√(2·g)·H^(3/2)
//! déversoir triangulaire    Q = (8/15)·Cd·tan(θ/2)·√(2·g)·H^(5/2)
//! charge (réciproque rect.) H = ( Q / ((2/3)·Cd·b·√(2·g)) )^(2/3)
//! ```
//!
//! `Q` débit volumique (m³/s), `Cd` coefficient de décharge (sans dimension),
//! `b` largeur de la crête du déversoir rectangulaire (m), `θ` angle d'ouverture
//! total de l'échancrure en V (rad), `H` charge amont mesurée au-dessus de la
//! crête (m), `g` accélération de la pesanteur (m/s²).
//!
//! **Convention** : SI ; angle en radians. **Limite honnête** : déversoir
//! **mince en paroi**, écoulement **dénoyé** (nappe libre et ventilée, aval sous
//! la crête), charge `H` mesurée **en amont hors de la zone d'abaissement** de la
//! surface libre. Le **coefficient de décharge `Cd` est fourni par l'appelant**
//! (il dépend de la géométrie du seuil, de la charge relative et de la viscosité)
//! et n'est jamais supposé ; de même l'accélération de la pesanteur `g` est une
//! **donnée de l'appelant**. Aucune correction de vitesse d'approche, de
//! contraction latérale ou de tension superficielle n'est appliquée.

use core::f64::consts::PI;

/// Débit sur **déversoir rectangulaire** mince `Q = (2/3)·Cd·b·√(2·g)·H^(3/2)` (m³/s).
///
/// Formule de Poleni : le débit croît comme la puissance 3/2 de la charge amont.
///
/// Panique si `discharge_coefficient <= 0`, `crest_width <= 0`, `head < 0`
/// ou `gravity <= 0`.
pub fn weir_rectangular_flow(
    discharge_coefficient: f64,
    crest_width: f64,
    head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de décharge Cd doit être > 0"
    );
    assert!(crest_width > 0.0, "la largeur de crête b doit être > 0");
    assert!(head >= 0.0, "la charge amont H doit être ≥ 0");
    assert!(gravity > 0.0, "la pesanteur g doit être > 0");
    (2.0 / 3.0) * discharge_coefficient * crest_width * (2.0 * gravity).sqrt() * head.powf(1.5)
}

/// Débit sur **déversoir triangulaire** (échancrure en V)
/// `Q = (8/15)·Cd·tan(θ/2)·√(2·g)·H^(5/2)` (m³/s).
///
/// Formule de Thomson : la sensibilité à la charge (puissance 5/2) rend ce
/// déversoir précis pour les faibles débits.
///
/// Panique si `discharge_coefficient <= 0`, si `notch_angle_rad` n'est pas dans
/// `]0, π[`, si `head < 0` ou si `gravity <= 0`.
pub fn weir_vnotch_flow(
    discharge_coefficient: f64,
    notch_angle_rad: f64,
    head: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de décharge Cd doit être > 0"
    );
    assert!(
        notch_angle_rad > 0.0 && notch_angle_rad < PI,
        "l'angle d'ouverture θ doit être dans ]0, π["
    );
    assert!(head >= 0.0, "la charge amont H doit être ≥ 0");
    assert!(gravity > 0.0, "la pesanteur g doit être > 0");
    (8.0 / 15.0)
        * discharge_coefficient
        * (notch_angle_rad / 2.0).tan()
        * (2.0 * gravity).sqrt()
        * head.powf(2.5)
}

/// Charge amont d'un **déversoir rectangulaire** à partir de son débit
/// `H = ( Q / ((2/3)·Cd·b·√(2·g)) )^(2/3)` (m) — réciproque de [`weir_rectangular_flow`].
///
/// Panique si `discharge_coefficient <= 0`, `crest_width <= 0`, `discharge < 0`
/// ou `gravity <= 0`.
pub fn weir_head_from_rectangular_flow(
    discharge_coefficient: f64,
    crest_width: f64,
    discharge: f64,
    gravity: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de décharge Cd doit être > 0"
    );
    assert!(crest_width > 0.0, "la largeur de crête b doit être > 0");
    assert!(discharge >= 0.0, "le débit Q doit être ≥ 0");
    assert!(gravity > 0.0, "la pesanteur g doit être > 0");
    let coefficient = (2.0 / 3.0) * discharge_coefficient * crest_width * (2.0 * gravity).sqrt();
    (discharge / coefficient).powf(2.0 / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rectangular_flow_and_head_are_reciprocal() {
        // Aller-retour : H → Q → H doit redonner la charge initiale.
        let cd = 0.62_f64;
        let b = 0.5_f64;
        let h = 0.15_f64;
        let g = 9.81_f64;
        let q = weir_rectangular_flow(cd, b, h, g);
        let h_back = weir_head_from_rectangular_flow(cd, b, q, g);
        assert_relative_eq!(h_back, h, epsilon = 1e-12);
    }

    #[test]
    fn rectangular_flow_is_linear_in_crest_width() {
        // Q ∝ b à charge et coefficient fixés : doubler la largeur double le débit.
        let cd = 0.6_f64;
        let h = 0.2_f64;
        let g = 9.81_f64;
        let q1 = weir_rectangular_flow(cd, 0.4, h, g);
        let q2 = weir_rectangular_flow(cd, 0.8, h, g);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn rectangular_flow_realistic_case() {
        // Cd = 0,62, b = 0,5 m, H = 0,15 m, g = 9,81 m/s².
        // √(2·9,81) = 4,4294469 ; H^1,5 = 0,058094745.
        // Q = (2/3)·0,62·0,5·4,4294469·0,058094745 = 0,0531810 m³/s.
        let q = weir_rectangular_flow(0.62, 0.5, 0.15, 9.81);
        assert_relative_eq!(q, 0.053_181_04, epsilon = 1e-6);
    }

    #[test]
    fn vnotch_flow_scales_as_head_power_five_halves() {
        // Q ∝ H^(5/2) : doubler la charge multiplie le débit par 2^2,5.
        let cd = 0.58_f64;
        let angle = PI / 2.0;
        let g = 9.81_f64;
        let q1 = weir_vnotch_flow(cd, angle, 0.1, g);
        let q2 = weir_vnotch_flow(cd, angle, 0.2, g);
        assert_relative_eq!(q2 / q1, 2.0_f64.powf(2.5), epsilon = 1e-12);
    }

    #[test]
    fn vnotch_ninety_degree_realistic_case() {
        // Échancrure à 90° (θ = π/2, tan(θ/2) = 1), Cd = 0,58, H = 0,2 m, g = 9,81.
        // √(2·9,81) = 4,4294469 ; H^2,5 = 0,017888544.
        // Q = (8/15)·0,58·1·4,4294469·0,017888544 = 0,0245104 m³/s.
        let q = weir_vnotch_flow(0.58, PI / 2.0, 0.2, 9.81);
        assert_relative_eq!(q, 0.024_510_44, epsilon = 1e-6);
    }

    #[test]
    fn vnotch_flow_is_linear_in_discharge_coefficient() {
        // Q ∝ Cd à géométrie et charge fixées.
        let angle = PI / 3.0;
        let g = 9.81_f64;
        let q1 = weir_vnotch_flow(0.55, angle, 0.12, g);
        let q2 = weir_vnotch_flow(0.60, angle, 0.12, g);
        assert_relative_eq!(q2 / q1, 0.60 / 0.55, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient de décharge Cd doit être > 0")]
    fn non_positive_discharge_coefficient_panics() {
        weir_rectangular_flow(0.0, 0.5, 0.15, 9.81);
    }
}
