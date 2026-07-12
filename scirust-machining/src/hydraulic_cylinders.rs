//! Vérins hydrauliques — efforts en sortie/rentrée, vitesse de tige, débit et
//! puissance d'un vérin double effet.
//!
//! ```text
//! section de piston (fût)  A = π·D²/4
//! section côté tige        A' = π·(D² − d²)/4
//! effort en sortie         Fs = p·A
//! effort en rentrée        Fr = p·A'
//! vitesse de tige          v = Q/A          (Q débit)
//! puissance fluide         P = p·Q
//! ```
//!
//! `D` alésage (m), `d` diamètre de tige (m), `p` pression (Pa), `Q` débit
//! (m³/s), `A` section du fût, `A'` section annulaire côté tige. La rentrée
//! (côté tige) développe un effort et une vitesse différents à débit/pression
//! donnés, car la tige réduit la section active.
//!
//! **Convention** : SI cohérent. **Limite honnête** : bilan **statique/
//! cinématique** idéal (pas de pertes de charge internes, de frottement de
//! joints, ni de compressibilité) ; `p` et `Q` effectifs sont fournis par
//! l'appelant.

use core::f64::consts::PI;

/// Section du piston côté fût `A = π·D²/4` (m²).
pub fn bore_area(bore_diameter: f64) -> f64 {
    PI * bore_diameter * bore_diameter / 4.0
}

/// Section annulaire côté tige `A' = π·(D² − d²)/4` (m²).
///
/// Panique si `rod_diameter >= bore_diameter`.
pub fn rod_side_area(bore_diameter: f64, rod_diameter: f64) -> f64 {
    assert!(
        rod_diameter < bore_diameter,
        "la tige doit être plus fine que l'alésage"
    );
    PI * (bore_diameter * bore_diameter - rod_diameter * rod_diameter) / 4.0
}

/// Effort en **sortie de tige** `Fs = p·A` (N).
pub fn extend_force(pressure: f64, bore_diameter: f64) -> f64 {
    pressure * bore_area(bore_diameter)
}

/// Effort en **rentrée de tige** `Fr = p·A'` (N).
pub fn retract_force(pressure: f64, bore_diameter: f64, rod_diameter: f64) -> f64 {
    pressure * rod_side_area(bore_diameter, rod_diameter)
}

/// Vitesse de la tige `v = Q/A` (m/s), `area` la section active considérée.
///
/// Panique si `area <= 0`.
pub fn piston_speed(flow_m3_s: f64, area_m2: f64) -> f64 {
    assert!(
        area_m2 > 0.0,
        "la section active doit être strictement positive"
    );
    flow_m3_s / area_m2
}

/// Puissance fluide `P = p·Q` (W).
pub fn fluid_power(pressure: f64, flow_m3_s: f64) -> f64 {
    pressure * flow_m3_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn extend_force_from_bore() {
        // D=63 mm, p=160 bar=16 MPa → A=π·0,063²/4 ; Fs = p·A ≈ 49,9 kN.
        let a = bore_area(0.063);
        assert_relative_eq!(extend_force(16e6, 0.063), 16e6 * a, epsilon = 1e-3);
        assert!(extend_force(16e6, 0.063) > 49_000.0 && extend_force(16e6, 0.063) < 51_000.0);
    }

    #[test]
    fn retract_force_is_lower_than_extend() {
        // La section annulaire est plus petite → effort de rentrée plus faible.
        let fe = extend_force(16e6, 0.063);
        let fr = retract_force(16e6, 0.063, 0.036);
        assert!(fr < fe);
        assert_relative_eq!(fr, 16e6 * rod_side_area(0.063, 0.036), epsilon = 1e-3);
    }

    #[test]
    fn retract_is_faster_at_same_flow() {
        // À débit égal, la vitesse de rentrée (section annulaire) dépasse la sortie.
        let q = 20e-3 / 60.0; // 20 L/min
        let v_out = piston_speed(q, bore_area(0.063));
        let v_in = piston_speed(q, rod_side_area(0.063, 0.036));
        assert!(v_in > v_out);
    }

    #[test]
    fn fluid_power_is_pressure_times_flow() {
        // p=16 MPa, Q=20 L/min = 3,33e-4 m³/s → P ≈ 5,33 kW.
        let q = 20e-3 / 60.0;
        assert_relative_eq!(fluid_power(16e6, q), 16e6 * q, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "tige doit être plus fine")]
    fn oversized_rod_panics() {
        rod_side_area(0.063, 0.063);
    }
}
