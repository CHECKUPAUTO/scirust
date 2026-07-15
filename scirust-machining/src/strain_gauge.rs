//! Jauge de déformation (extensométrie) et pont de Wheatstone — variation de
//! résistance et tension de sortie en montages quart de pont et pont complet.
//!
//! ```text
//! variation résistance      ΔR = GF · R · ε
//! déformation (réciproque)  ε  = ΔR / (GF · R)
//! quart de pont (1 jauge)   Vo = (GF · ε / 4) · Vex
//! pont complet (4 jauges)   Vo = GF · ε · Vex
//! ```
//!
//! `GF` facteur de jauge (sans dimension), `R` résistance nominale (Ω), `ε`
//! déformation (m/m, sans dimension), `ΔR` variation de résistance (Ω), `Vex`
//! tension d'excitation (V), `Vo` tension de sortie du pont (V). Le pont
//! complet (4 jauges actives) délivre une sortie quatre fois supérieure à celle
//! du quart de pont pour une même déformation.
//!
//! **Convention** : SI cohérent. **Limite honnête** : le facteur de jauge `GF`,
//! la résistance nominale `R` et la tension d'excitation `Vex` sont **fournis
//! par l'appelant** (constructeur du capteur, alimentation) — aucune valeur « par
//! défaut » n'est inventée. Comportement **élastique linéaire** ; les effets de
//! température sont supposés **compensés par le montage** (non modélisés ici) et
//! la résistance des fils de liaison est négligée.

/// Variation de résistance de la jauge `ΔR = GF · R · ε` (Ω).
///
/// Panique si `gauge_factor <= 0` ou `nominal_resistance <= 0`.
pub fn straingauge_resistance_change(
    gauge_factor: f64,
    nominal_resistance: f64,
    strain: f64,
) -> f64 {
    assert!(
        gauge_factor > 0.0,
        "le facteur de jauge doit être strictement positif"
    );
    assert!(
        nominal_resistance > 0.0,
        "la résistance nominale doit être strictement positive"
    );
    gauge_factor * nominal_resistance * strain
}

/// Déformation déduite d'une variation de résistance `ε = ΔR / (GF · R)`
/// (m/m) — réciproque de [`straingauge_resistance_change`].
///
/// Panique si `gauge_factor <= 0` ou `nominal_resistance <= 0`.
pub fn straingauge_strain_from_resistance(
    gauge_factor: f64,
    nominal_resistance: f64,
    resistance_change: f64,
) -> f64 {
    assert!(
        gauge_factor > 0.0,
        "le facteur de jauge doit être strictement positif"
    );
    assert!(
        nominal_resistance > 0.0,
        "la résistance nominale doit être strictement positive"
    );
    resistance_change / (gauge_factor * nominal_resistance)
}

/// Tension de sortie d'un pont de Wheatstone en quart de pont (1 jauge active)
/// `Vo = (GF · ε / 4) · Vex` (V).
///
/// Panique si `gauge_factor <= 0`.
pub fn straingauge_quarter_bridge_output(
    gauge_factor: f64,
    strain: f64,
    excitation_voltage: f64,
) -> f64 {
    assert!(
        gauge_factor > 0.0,
        "le facteur de jauge doit être strictement positif"
    );
    gauge_factor * strain / 4.0 * excitation_voltage
}

/// Tension de sortie d'un pont de Wheatstone complet (4 jauges actives)
/// `Vo = GF · ε · Vex` (V).
///
/// Panique si `gauge_factor <= 0`.
pub fn straingauge_full_bridge_output(
    gauge_factor: f64,
    strain: f64,
    excitation_voltage: f64,
) -> f64 {
    assert!(
        gauge_factor > 0.0,
        "le facteur de jauge doit être strictement positif"
    );
    gauge_factor * strain * excitation_voltage
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reciprocity_resistance_strain() {
        // Aller-retour ΔR → ε → ΔR : identité exacte.
        let gf = 2.1;
        let r = 350.0;
        let strain = 8.5e-4;
        let dr = straingauge_resistance_change(gf, r, strain);
        let back = straingauge_strain_from_resistance(gf, r, dr);
        assert_relative_eq!(back, strain, epsilon = 1e-15);
    }

    #[test]
    fn resistance_change_reference_case() {
        // GF=2,0, R=350 Ω, ε=1000 µm/m → ΔR = 2,0·350·0,001 = 0,7 Ω.
        let dr = straingauge_resistance_change(2.0, 350.0, 1000e-6);
        assert_relative_eq!(dr, 0.7, epsilon = 1e-12);
    }

    #[test]
    fn quarter_bridge_reference_case() {
        // GF=2,0, ε=1000 µm/m, Vex=10 V → Vo = (2,0·0,001/4)·10 = 5 mV.
        let vo = straingauge_quarter_bridge_output(2.0, 1000e-6, 10.0);
        assert_relative_eq!(vo, 5e-3, epsilon = 1e-12);
    }

    #[test]
    fn full_bridge_is_four_times_quarter() {
        // À déformation et excitation égales, le pont complet sort 4× le quart.
        let gf = 2.05;
        let strain = 3.2e-4;
        let vex = 5.0;
        let quarter = straingauge_quarter_bridge_output(gf, strain, vex);
        let full = straingauge_full_bridge_output(gf, strain, vex);
        assert_relative_eq!(full, 4.0 * quarter, epsilon = 1e-15);
    }

    #[test]
    fn output_scales_linearly_with_strain() {
        // Sortie proportionnelle à la déformation : doubler ε double Vo.
        let v1 = straingauge_full_bridge_output(2.0, 5e-4, 10.0);
        let v2 = straingauge_full_bridge_output(2.0, 1e-3, 10.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "facteur de jauge")]
    fn zero_gauge_factor_panics() {
        straingauge_resistance_change(0.0, 350.0, 1e-3);
    }
}
