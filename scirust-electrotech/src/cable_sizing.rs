//! **Dimensionnement de câble** — résistance du conducteur, chute de tension en
//! régime monophasé et triphasé, densité de courant et pertes Joule par phase à
//! partir des grandeurs géométriques et linéiques du câble.
//!
//! ```text
//! résistance conducteur       R  = ρ · L / A
//! chute de tension monophasé  ΔU = 2 · I · L · (R'·cos φ + X'·√(1 − cos²φ))
//! chute de tension triphasé   ΔU = √3 · I · L · (R'·cos φ + X'·√(1 − cos²φ))
//! densité de courant          J  = I / A
//! pertes Joule par phase      P  = I² · R
//! ```
//!
//! `ρ` résistivité du matériau conducteur (Ω·m), `L` longueur du câble (m), `A`
//! section du conducteur (m²), `R` résistance du conducteur (Ω), `I` courant
//! de service (A), `R'` résistance linéique (Ω/m), `X'` réactance linéique
//! (Ω/m), `cos φ` facteur de puissance de la charge (sans dimension,
//! `∈ [0, 1]`), `√(1 − cos²φ) = sin φ` la composante réactive, `ΔU` chute de
//! tension (V), `J` densité de courant (A/m²), `P` pertes Joule par phase (W).
//!
//! **Convention** : SI ; tensions et chutes de tension en V, courants en A,
//! sections en m², longueurs en m, résistivités en Ω·m, résistances et
//! réactances (totales ou linéiques) en Ω resp. Ω/m, pertes en W. Le facteur
//! de puissance sert de `cos φ` ; sa composante réactive `sin φ` est prise
//! positive. **Limite honnête** : régime **permanent** ; la résistivité `ρ`, la
//! section `A`, les grandeurs **linéiques** `R'` et `X'` (par unité de longueur)
//! et le facteur de puissance sont **fournis par l'appelant** (données de câble
//! et de réseau) — aucune valeur « par défaut » n'est inventée. La chute de
//! tension utilise l'**approximation classique** (projection de l'impédance sur
//! l'axe de la tension) valable pour les faibles chutes. Le **déclassement
//! thermique** (température ambiante, groupement de circuits, mode de pose) est
//! **fourni par l'appelant** conformément aux normes **CEI 60364**.

/// Résistance du conducteur `R = ρ · L / A` (Ω), à partir de la résistivité du
/// matériau, de la longueur et de la section.
///
/// Panique si `resistivity < 0`, si `length < 0` ou si `cross_section_area <= 0`
/// (division par zéro).
pub fn cable_resistance(resistivity: f64, length: f64, cross_section_area: f64) -> f64 {
    assert!(resistivity >= 0.0, "la résistivité ρ doit être ≥ 0");
    assert!(length >= 0.0, "la longueur L doit être ≥ 0");
    assert!(
        cross_section_area > 0.0,
        "la section A doit être strictement positive"
    );
    resistivity * length / cross_section_area
}

/// Chute de tension aller-retour en **monophasé**
/// `ΔU = 2 · I · L · (R'·cos φ + X'·√(1 − cos²φ))` (V).
///
/// Le facteur `2` tient compte des conducteurs aller et retour. `R'` et `X'`
/// sont les grandeurs **linéiques** (Ω/m), `L` la longueur simple du câble (m).
///
/// Panique si `current < 0`, si `length < 0`, si `resistance_per_length < 0`,
/// si `reactance_per_length < 0` ou si `power_factor` n'est pas dans `[0, 1]`.
pub fn cable_voltage_drop_single_phase(
    current: f64,
    length: f64,
    resistance_per_length: f64,
    reactance_per_length: f64,
    power_factor: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant I doit être ≥ 0");
    assert!(length >= 0.0, "la longueur L doit être ≥ 0");
    assert!(
        resistance_per_length >= 0.0,
        "la résistance linéique R' doit être ≥ 0"
    );
    assert!(
        reactance_per_length >= 0.0,
        "la réactance linéique X' doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    let sin_phi = (1.0 - power_factor * power_factor).sqrt();
    2.0 * current * length * (resistance_per_length * power_factor + reactance_per_length * sin_phi)
}

/// Chute de tension composée en **triphasé équilibré**
/// `ΔU = √3 · I · L · (R'·cos φ + X'·√(1 − cos²φ))` (V).
///
/// `R'` et `X'` sont les grandeurs **linéiques** (Ω/m), `L` la longueur du câble
/// (m). Le rapport avec la formule monophasée vaut `2 / √3`.
///
/// Panique si `current < 0`, si `length < 0`, si `resistance_per_length < 0`,
/// si `reactance_per_length < 0` ou si `power_factor` n'est pas dans `[0, 1]`.
pub fn cable_voltage_drop_three_phase(
    current: f64,
    length: f64,
    resistance_per_length: f64,
    reactance_per_length: f64,
    power_factor: f64,
) -> f64 {
    assert!(current >= 0.0, "le courant I doit être ≥ 0");
    assert!(length >= 0.0, "la longueur L doit être ≥ 0");
    assert!(
        resistance_per_length >= 0.0,
        "la résistance linéique R' doit être ≥ 0"
    );
    assert!(
        reactance_per_length >= 0.0,
        "la réactance linéique X' doit être ≥ 0"
    );
    assert!(
        (0.0..=1.0).contains(&power_factor),
        "le facteur de puissance cos φ doit être dans [0, 1]"
    );
    let sin_phi = (1.0 - power_factor * power_factor).sqrt();
    3.0_f64.sqrt()
        * current
        * length
        * (resistance_per_length * power_factor + reactance_per_length * sin_phi)
}

/// Densité de courant `J = I / A` (A/m²), courant rapporté à la section du
/// conducteur.
///
/// Panique si `current < 0` ou si `cross_section_area <= 0` (division par zéro).
pub fn cable_current_density(current: f64, cross_section_area: f64) -> f64 {
    assert!(current >= 0.0, "le courant I doit être ≥ 0");
    assert!(
        cross_section_area > 0.0,
        "la section A doit être strictement positive"
    );
    current / cross_section_area
}

/// Pertes Joule par phase `P = I² · R` (W), dissipées dans la résistance du
/// conducteur.
///
/// Panique si `resistance < 0` (une résistance négative est non physique ; le
/// courant peut être négatif, seul son carré intervient).
pub fn cable_power_loss_per_phase(current: f64, resistance: f64) -> f64 {
    assert!(resistance >= 0.0, "la résistance R doit être ≥ 0");
    current * current * resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn resistance_scales_with_length() {
        // Proportionnalité : à section et résistivité fixées, doubler la
        // longueur double la résistance.
        let r1 = cable_resistance(1.68e-8, 100.0, 1.0e-6);
        let r2 = cable_resistance(1.68e-8, 200.0, 1.0e-6);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn single_to_three_phase_ratio() {
        // Identité : à mêmes paramètres, ΔU_mono / ΔU_tri = 2 / √3, car seuls
        // les préfacteurs 2 et √3 diffèrent.
        let one = cable_voltage_drop_single_phase(100.0, 50.0, 0.001, 0.0008, 0.8);
        let three = cable_voltage_drop_three_phase(100.0, 50.0, 0.001, 0.0008, 0.8);
        assert_relative_eq!(one / three, 2.0 / 3.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn power_loss_is_quadratic_in_current() {
        // Loi quadratique : à résistance fixée, doubler le courant quadruple
        // les pertes Joule.
        let p1 = cable_power_loss_per_phase(50.0, 1.68);
        let p2 = cable_power_loss_per_phase(100.0, 1.68);
        assert_relative_eq!(p2 / p1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn current_density_times_area_recovers_current() {
        // Réciprocité : J · A restitue le courant injecté.
        let i = 100.0_f64;
        let a = 1.0e-5_f64;
        assert_relative_eq!(cable_current_density(i, a) * a, i, epsilon = 1e-9);
    }

    #[test]
    fn realistic_copper_feeder_case() {
        // Cas chiffré, conducteur cuivre :
        //   R  = ρ·L/A = 1,68e-8·100 / 1e-6            = 1,68 Ω
        //   ΔU monophasé (I=100 A, L=50 m, cos φ=0,8, sin φ=0,6,
        //                 R'=0,001 Ω/m, X'=0,0008 Ω/m) :
        //        2·100·50·(0,001·0,8 + 0,0008·0,6)
        //      = 10000·(0,0008 + 0,00048) = 10000·0,00128 = 12,8 V
        //   J  = I/A = 100 / 1e-5                       = 1e7 A/m²
        //   P  = I²·R = 100²·1,68                       = 16 800 W
        let r = cable_resistance(1.68e-8, 100.0, 1.0e-6);
        assert_relative_eq!(r, 1.68, epsilon = 1e-3);
        let du = cable_voltage_drop_single_phase(100.0, 50.0, 0.001, 0.0008, 0.8);
        assert_relative_eq!(du, 12.8, epsilon = 1e-3);
        assert_relative_eq!(cable_current_density(100.0, 1.0e-5), 1.0e7, epsilon = 1e-3);
        assert_relative_eq!(
            cable_power_loss_per_phase(100.0, 1.68),
            16_800.0,
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "le facteur de puissance cos φ doit être dans [0, 1]")]
    fn voltage_drop_rejects_power_factor_above_one() {
        cable_voltage_drop_three_phase(100.0, 50.0, 0.001, 0.0008, 1.5);
    }
}
