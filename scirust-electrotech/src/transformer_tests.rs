//! **Essais du transformateur (schéma équivalent)** — exploitation des essais
//! à vide et en court-circuit d'un transformateur monophasé pour identifier les
//! éléments du schéma équivalent : résistance de pertes fer, réactance
//! magnétisante, résistance et réactance équivalentes de fuite, puis rendement
//! à une charge partielle donnée.
//!
//! ```text
//! résistance de pertes fer     R_c   = V_0² / P_0
//! réactance magnétisante       X_m   = V_0 / √(I_0² − (P_0 / V_0)²)
//! résistance équivalente       R_eq  = P_cc / I_cc²
//! réactance équivalente        X_eq  = √((V_cc / I_cc)² − R_eq²)
//! rendement à charge partielle η(x)  = x·S·cosφ
//!                                      ────────────────────────────
//!                                      x·S·cosφ + P_fe + x²·P_cu,fl
//! ```
//!
//! `V_0`, `I_0`, `P_0` tension (V), courant (A) et puissance (W) lus à l'essai
//! **à vide** (secondaire ouvert) ; `R_c` résistance modélisant les pertes fer
//! (Ω) et `X_m` réactance magnétisante de la branche shunt (Ω). `V_cc`, `I_cc`,
//! `P_cc` tension (V), courant (A) et puissance (W) lus à l'essai en
//! **court-circuit** (secondaire fermé, tension réduite) ; `R_eq` résistance
//! équivalente (pertes cuivre ramenées, Ω) et `X_eq` réactance équivalente de
//! fuite (Ω). `S` puissance apparente nominale (VA), `cosφ` facteur de
//! puissance de la charge (sans dimension, `∈ [0, 1]`), `x` fraction de charge
//! (sans dimension, `> 0`, `x = 1` en pleine charge), `P_fe` pertes fer
//! (constantes, W), `P_cu,fl` pertes cuivre en pleine charge (W), `η` rendement
//! (sans dimension, `∈ [0, 1]`).
//!
//! **Convention** : SI ; tensions en V, courants en A, puissances actives en W,
//! apparentes en VA, impédances en Ω ; grandeurs efficaces en **régime
//! sinusoïdal permanent**. **Limite honnête** : les relevés des essais **à
//! vide** (pertes fer, courant magnétisant) et en **court-circuit** (pertes
//! cuivre) sont **fournis par l'appelant** ; les pertes fer sont supposées
//! **constantes** et les pertes cuivre varient en **charge²** (`x²`). Toutes les
//! grandeurs doivent être **ramenées à un même côté** (celui de la mesure) par
//! l'appelant ; ce module traite des modules réels (Ω) et ne modélise pas le
//! déphasage complexe ni la saturation magnétique.

/// Résistance de pertes fer `R_c = V_0² / P_0` (branche shunt, essai à vide).
///
/// La puissance à vide `P_0` est dissipée dans `R_c` sous la tension `V_0`.
///
/// Panique si `open_circuit_voltage <= 0` ou si `open_circuit_power <= 0`.
pub fn xfmrtest_iron_loss_resistance(open_circuit_voltage: f64, open_circuit_power: f64) -> f64 {
    assert!(open_circuit_voltage > 0.0, "V_0 > 0 requis");
    assert!(open_circuit_power > 0.0, "P_0 > 0 requis");
    open_circuit_voltage * open_circuit_voltage / open_circuit_power
}

/// Réactance magnétisante `X_m = V_0 / √(I_0² − (P_0 / V_0)²)` (essai à vide).
///
/// Le courant à vide `I_0` se décompose en une composante active `P_0 / V_0`
/// (dans `R_c`) et une composante magnétisante `√(I_0² − (P_0 / V_0)²)` (dans
/// `X_m`), d'où `X_m = V_0 / I_m`.
///
/// Panique si `open_circuit_voltage <= 0`, si `open_circuit_current <= 0`, si
/// `open_circuit_power < 0`, ou si la composante magnétisante n'est pas
/// strictement positive (`I_0² <= (P_0 / V_0)²`).
pub fn xfmrtest_magnetizing_reactance(
    open_circuit_voltage: f64,
    open_circuit_current: f64,
    open_circuit_power: f64,
) -> f64 {
    assert!(open_circuit_voltage > 0.0, "V_0 > 0 requis");
    assert!(open_circuit_current > 0.0, "I_0 > 0 requis");
    assert!(open_circuit_power >= 0.0, "P_0 ≥ 0 requis");
    let active_current = open_circuit_power / open_circuit_voltage;
    let magnetizing_current_squared =
        open_circuit_current * open_circuit_current - active_current * active_current;
    assert!(
        magnetizing_current_squared > 0.0,
        "I_0² > (P_0 / V_0)² requis (composante magnétisante > 0)"
    );
    open_circuit_voltage / magnetizing_current_squared.sqrt()
}

/// Résistance équivalente `R_eq = P_cc / I_cc²` (pertes cuivre, essai en
/// court-circuit).
///
/// En court-circuit la puissance mesurée est essentiellement dissipée par effet
/// Joule dans les enroulements, d'où `R_eq = P_cc / I_cc²`.
///
/// Panique si `short_circuit_power < 0` ou si `short_circuit_current <= 0`.
pub fn xfmrtest_equivalent_resistance(short_circuit_power: f64, short_circuit_current: f64) -> f64 {
    assert!(short_circuit_power >= 0.0, "P_cc ≥ 0 requis");
    assert!(short_circuit_current > 0.0, "I_cc > 0 requis");
    short_circuit_power / (short_circuit_current * short_circuit_current)
}

/// Réactance équivalente `X_eq = √((V_cc / I_cc)² − R_eq²)` (fuites, essai en
/// court-circuit).
///
/// L'impédance équivalente vaut `Z_eq = V_cc / I_cc` ; la réactance s'en déduit
/// par `X_eq = √(Z_eq² − R_eq²)`.
///
/// Panique si `short_circuit_voltage < 0`, si `short_circuit_current <= 0`, si
/// `equivalent_resistance < 0`, ou si `R_eq > Z_eq` (radicande négatif).
pub fn xfmrtest_equivalent_reactance(
    short_circuit_voltage: f64,
    short_circuit_current: f64,
    equivalent_resistance: f64,
) -> f64 {
    assert!(short_circuit_voltage >= 0.0, "V_cc ≥ 0 requis");
    assert!(short_circuit_current > 0.0, "I_cc > 0 requis");
    assert!(equivalent_resistance >= 0.0, "R_eq ≥ 0 requis");
    let equivalent_impedance = short_circuit_voltage / short_circuit_current;
    let reactance_squared =
        equivalent_impedance * equivalent_impedance - equivalent_resistance * equivalent_resistance;
    assert!(
        reactance_squared >= 0.0,
        "R_eq ≤ Z_eq = V_cc / I_cc requis (radicande ≥ 0)"
    );
    reactance_squared.sqrt()
}

/// Rendement à charge partielle
/// `η(x) = x·S·cosφ / (x·S·cosφ + P_fe + x²·P_cu,fl)`.
///
/// La puissance utile est `x·S·cosφ`, les pertes fer `P_fe` sont constantes et
/// les pertes cuivre valent `x²·P_cu,fl` (proportionnelles au carré de la
/// charge).
///
/// Panique si `rated_power <= 0`, si `power_factor` n'est pas dans `[0, 1]`, si
/// `load_fraction <= 0`, si `iron_loss < 0`, si `full_load_copper_loss < 0`, ou
/// si le dénominateur (puissance absorbée) n'est pas strictement positif.
pub fn xfmrtest_efficiency_at_load(
    rated_power: f64,
    power_factor: f64,
    load_fraction: f64,
    iron_loss: f64,
    full_load_copper_loss: f64,
) -> f64 {
    assert!(rated_power > 0.0, "S > 0 requis");
    assert!((0.0..=1.0).contains(&power_factor), "cosφ ∈ [0, 1] requis");
    assert!(load_fraction > 0.0, "x > 0 requis");
    assert!(iron_loss >= 0.0, "P_fe ≥ 0 requis");
    assert!(full_load_copper_loss >= 0.0, "P_cu,fl ≥ 0 requis");
    let output_power = load_fraction * rated_power * power_factor;
    let copper_loss = load_fraction * load_fraction * full_load_copper_loss;
    let input_power = output_power + iron_loss + copper_loss;
    assert!(input_power > 0.0, "puissance absorbée > 0 requise");
    output_power / input_power
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn no_load_test_recovers_shunt_branch() {
        // Essai à vide : triangle des courants 0,6 / 0,8 / 1,0 (3-4-5).
        // V_0 = 200 V, I_0 = 1,0 A, P_0 = 120 W → composante active
        // P_0 / V_0 = 0,6 A, composante magnétisante √(1 − 0,36) = 0,8 A.
        let r_c = xfmrtest_iron_loss_resistance(200.0, 120.0);
        // R_c = 200² / 120 = 40000 / 120 = 333,333... Ω.
        assert_relative_eq!(r_c, 40000.0 / 120.0, epsilon = 1e-9);
        let x_m = xfmrtest_magnetizing_reactance(200.0, 1.0, 120.0);
        // X_m = 200 / 0,8 = 250 Ω.
        assert_relative_eq!(x_m, 250.0, epsilon = 1e-9);
    }

    #[test]
    fn short_circuit_impedance_triangle_is_consistent() {
        // Essai en court-circuit : Z_eq² = R_eq² + X_eq² (Pythagore).
        // I_cc = 10 A, P_cc = 300 W → R_eq = 300 / 100 = 3 Ω.
        let r_eq = xfmrtest_equivalent_resistance(300.0, 10.0);
        assert_relative_eq!(r_eq, 3.0, epsilon = 1e-12);
        // V_cc = 50 V → Z_eq = 5 Ω, X_eq = √(25 − 9) = 4 Ω.
        let x_eq = xfmrtest_equivalent_reactance(50.0, 10.0, r_eq);
        assert_relative_eq!(x_eq, 4.0, epsilon = 1e-9);
        // Réciprocité : Z_eq² = R_eq² + X_eq² = 9 + 16 = 25.
        let z_eq = 50.0_f64 / 10.0;
        assert_relative_eq!(r_eq * r_eq + x_eq * x_eq, z_eq * z_eq, epsilon = 1e-9);
    }

    #[test]
    fn purely_reactive_no_load_gives_current_as_magnetizing() {
        // Cas limite : P_0 = 0 → toute la composante est magnétisante, I_m = I_0,
        // donc X_m = V_0 / I_0.
        let x_m = xfmrtest_magnetizing_reactance(230.0, 2.0, 0.0);
        assert_relative_eq!(x_m, 230.0 / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn copper_loss_scales_with_load_squared() {
        // Proportionnalité : à cosφ = 1 et sans pertes fer, l'inverse du
        // rendement moins 1 vaut x·P_cu,fl / S, donc croît linéairement en x.
        let s = 10_000.0_f64;
        let p_cu = 400.0_f64;
        let eta_half = xfmrtest_efficiency_at_load(s, 1.0, 0.5, 0.0, p_cu);
        let eta_full = xfmrtest_efficiency_at_load(s, 1.0, 1.0, 0.0, p_cu);
        // (1/η − 1) = x·P_cu,fl / S ; le rapport plein/demi vaut 2.
        let excess_half = 1.0 / eta_half - 1.0;
        let excess_full = 1.0 / eta_full - 1.0;
        assert_relative_eq!(excess_full / excess_half, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn full_load_efficiency_numeric_case() {
        // Cas chiffré : S = 10 kVA, cosφ = 1, pleine charge (x = 1),
        // P_fe = 100 W, P_cu,fl = 300 W.
        // P_out = 1·10000·1 = 10000 W ; P_abs = 10000 + 100 + 300 = 10400 W ;
        // η = 10000 / 10400 = 0,9615384615...
        let eta = xfmrtest_efficiency_at_load(10_000.0, 1.0, 1.0, 100.0, 300.0);
        assert_relative_eq!(eta, 10_000.0 / 10_400.0, epsilon = 1e-12);
        assert_relative_eq!(eta, 0.961_538_461_538, epsilon = 1e-9);
    }

    #[test]
    fn lossless_transformer_reaches_unit_efficiency() {
        // Cas limite : sans pertes fer ni cuivre → η = 1 quelle que soit la
        // charge.
        let eta = xfmrtest_efficiency_at_load(5_000.0, 0.9, 0.75, 0.0, 0.0);
        assert_relative_eq!(eta, 1.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "I_cc > 0 requis")]
    fn zero_short_circuit_current_panics() {
        xfmrtest_equivalent_resistance(300.0, 0.0);
    }
}
