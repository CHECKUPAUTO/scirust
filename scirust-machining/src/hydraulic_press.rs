//! Presse et vérin hydrauliques — multiplication d'effort par le principe de
//! Pascal, avec conservation du volume (course) et de l'énergie.
//!
//! ```text
//! effort en sortie        Fout = Fin·(Aout/Ain)   (multiplication d'effort)
//! course en sortie        sout = sin·(Ain/Aout)   (conservation du volume)
//! effort depuis pression  F    = p·A
//! pression amplifiée      pout = pin·(Alarge/Asmall)   (multiplicateur)
//! énergie conservée       Fout·sout = Fin·sin
//! ```
//!
//! `Fin`/`Fout` efforts d'entrée/sortie (N), `Ain`/`Aout` sections des pistons
//! d'entrée/sortie (m²), `sin`/`sout` courses (m), `p` pression (Pa), `A`
//! section de piston (m²). Le grand piston multiplie l'effort dans le rapport
//! des sections mais réduit la course dans le rapport inverse : le produit
//! effort × course (l'énergie) reste constant.
//!
//! **Convention** : SI cohérent. **Limite honnête** : fluide **incompressible**,
//! pertes de charge et frottements **négligés** (rendement idéal = 1) ; les
//! sections de pistons et les pressions effectives sont **fournies par
//! l'appelant** (aucune valeur « par défaut » inventée). Distinct de
//! [`crate::hydraulic_cylinders`], qui traite la géométrie fût/tige d'un vérin.

/// Effort en sortie `Fout = Fin·(Aout/Ain)` (N), par le principe de Pascal.
///
/// Panique si `input_area <= 0` ou `output_area <= 0`.
pub fn hydraulic_press_output_force(input_force: f64, input_area: f64, output_area: f64) -> f64 {
    assert!(
        input_area > 0.0,
        "la section du piston d'entrée doit être strictement positive"
    );
    assert!(
        output_area > 0.0,
        "la section du piston de sortie doit être strictement positive"
    );
    input_force * (output_area / input_area)
}

/// Course en sortie `sout = sin·(Ain/Aout)` (m), par conservation du volume.
///
/// Panique si `input_area <= 0` ou `output_area <= 0`.
pub fn hydraulic_press_output_stroke(input_stroke: f64, input_area: f64, output_area: f64) -> f64 {
    assert!(
        input_area > 0.0,
        "la section du piston d'entrée doit être strictement positive"
    );
    assert!(
        output_area > 0.0,
        "la section du piston de sortie doit être strictement positive"
    );
    input_stroke * (input_area / output_area)
}

/// Effort développé par un piston `F = p·A` (N).
///
/// Panique si `piston_area < 0`.
pub fn hydraulic_press_force_from_pressure(pressure: f64, piston_area: f64) -> f64 {
    assert!(
        piston_area >= 0.0,
        "la section du piston doit être positive ou nulle"
    );
    pressure * piston_area
}

/// Pression amplifiée par un multiplicateur `pout = pin·(Alarge/Asmall)` (Pa).
///
/// Le grand piston reçoit `pin`, le petit piston (couplé mécaniquement)
/// restitue une pression plus élevée dans le rapport des sections.
///
/// Panique si `large_area <= 0` ou `small_area <= 0`.
pub fn hydraulic_press_intensifier_pressure(
    input_pressure: f64,
    large_area: f64,
    small_area: f64,
) -> f64 {
    assert!(
        large_area > 0.0,
        "la section du grand piston doit être strictement positive"
    );
    assert!(
        small_area > 0.0,
        "la section du petit piston doit être strictement positive"
    );
    input_pressure * (large_area / small_area)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn energy_is_conserved() {
        // Fout·sout doit égaler Fin·sin quel que soit le rapport des sections.
        let (fin, sin, ain, aout) = (200.0, 0.10, 1.0e-4, 1.0e-2);
        let fout = hydraulic_press_output_force(fin, ain, aout);
        let sout = hydraulic_press_output_stroke(sin, ain, aout);
        assert_relative_eq!(fout * sout, fin * sin, epsilon = 1e-12);
    }

    #[test]
    fn realistic_multiplication() {
        // Ain=1 cm²=1e-4 m², Aout=100 cm²=1e-2 m², rapport = 100.
        // Fin=200 N → Fout=20 000 N ; sin=0,1 m → sout=0,001 m.
        let fout = hydraulic_press_output_force(200.0, 1.0e-4, 1.0e-2);
        let sout = hydraulic_press_output_stroke(0.10, 1.0e-4, 1.0e-2);
        assert_relative_eq!(fout, 20_000.0, epsilon = 1e-9);
        assert_relative_eq!(sout, 0.001, epsilon = 1e-12);
    }

    #[test]
    fn equal_areas_are_neutral() {
        // Sections égales : ni multiplication d'effort ni réduction de course.
        assert_relative_eq!(
            hydraulic_press_output_force(750.0, 2.5e-3, 2.5e-3),
            750.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            hydraulic_press_output_stroke(0.42, 2.5e-3, 2.5e-3),
            0.42,
            epsilon = 1e-12
        );
    }

    #[test]
    fn force_from_pressure_matches_pascal() {
        // p·A appliqué au piston de sortie reconstitue l'effort de sortie :
        // p = Fin/Ain (pression du circuit), F = p·Aout = Fin·(Aout/Ain).
        let (fin, ain, aout) = (300.0, 5.0e-4, 4.0e-3);
        let pressure = fin / ain;
        assert_relative_eq!(
            hydraulic_press_force_from_pressure(pressure, aout),
            hydraulic_press_output_force(fin, ain, aout),
            epsilon = 1e-9
        );
    }

    #[test]
    fn intensifier_scales_pressure() {
        // pin=10 MPa, grand piston 20 cm², petit 4 cm² → rapport 5 → 50 MPa.
        assert_relative_eq!(
            hydraulic_press_intensifier_pressure(10.0e6, 20.0e-4, 4.0e-4),
            50.0e6,
            epsilon = 1e-3
        );
        // Proportionnalité à la pression d'entrée.
        let p1 = hydraulic_press_intensifier_pressure(1.0e6, 20.0e-4, 4.0e-4);
        let p2 = hydraulic_press_intensifier_pressure(3.0e6, 20.0e-4, 4.0e-4);
        assert_relative_eq!(p2, 3.0 * p1, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la section du piston d'entrée doit être strictement positive")]
    fn zero_input_area_panics() {
        let _ = hydraulic_press_output_force(100.0, 0.0, 1.0e-3);
    }
}
