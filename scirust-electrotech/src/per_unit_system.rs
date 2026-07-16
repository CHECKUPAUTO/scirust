//! **Système per-unit (grandeurs réduites)** — impédance et courant de base
//! triphasés, réduction d'une grandeur en valeur réduite (p.u.) et changement
//! de base d'une impédance en régime permanent équilibré.
//!
//! ```text
//! impédance de base (triphasée)   Z_base = V_base² / S_base
//! courant de base                 I_base = S_base / (√3 · V_base)
//! valeur réduite                  x_pu   = X_actual / X_base
//! changement de base (impédance)
//!     Z_pu,new = Z_pu,old · (S_base,new / S_base,old) · (V_base,old / V_base,new)²
//! ```
//!
//! `V_base` tension **composée** de base (V), `S_base` puissance apparente
//! triphasée de base commune (VA), `Z_base` impédance de base (Ω), `I_base`
//! courant de ligne de base (A), `X_actual` grandeur physique dans son unité SI
//! (V, A, Ω, VA…), `X_base` grandeur de base **de même nature** (même unité),
//! `x_pu` valeur réduite (per-unit, sans dimension), `Z_pu,old` / `Z_pu,new`
//! impédance réduite avant / après changement de base (p.u.), `S_base,old` /
//! `S_base,new` puissances apparentes de base ancienne / nouvelle (VA),
//! `V_base,old` / `V_base,new` tensions composées de base ancienne / nouvelle (V).
//!
//! **Convention** : SI ; tensions **composées** en V, puissances apparentes en
//! VA, impédances en Ω, courants de ligne en A ; le facteur `√3` traduit le
//! passage tension composée ↔ tension simple d'un système triphasé équilibré.
//! **Limite honnête** : système per-unit **triphasé équilibré** avec une
//! **puissance apparente de base commune** et une **tension de base composée** ;
//! le changement de base suppose des grandeurs d'**impédance** (loi en carré de
//! la tension). Les bases (`V_base`, `S_base`) et les grandeurs réelles sont
//! **fournies par l'appelant** d'après une plaque signalétique, un plan de
//! réseau ou une mesure — aucune valeur de base « par défaut » n'est inventée.

// √3 dérivé du facteur exact du système triphasé (pas une constante physique
// « inventée » ni un littéral trigonométrique).
fn sqrt_three() -> f64 {
    3.0_f64.sqrt()
}

/// Impédance de base triphasée `Z_base = V_base² / S_base` (Ω), avec `V_base`
/// tension **composée** de base.
///
/// Panique si `base_voltage <= 0` ou si `base_power <= 0`.
pub fn pu_base_impedance(base_voltage: f64, base_power: f64) -> f64 {
    assert!(
        base_voltage > 0.0,
        "la tension de base composée V_base doit être > 0"
    );
    assert!(
        base_power > 0.0,
        "la puissance apparente de base S_base doit être > 0"
    );
    base_voltage * base_voltage / base_power
}

/// Courant de ligne de base `I_base = S_base / (√3 · V_base)` (A), avec
/// `V_base` tension **composée** de base.
///
/// Panique si `base_power < 0` ou si `base_voltage <= 0`.
pub fn pu_base_current(base_power: f64, base_voltage: f64) -> f64 {
    assert!(
        base_power >= 0.0,
        "la puissance apparente de base S_base doit être ≥ 0"
    );
    assert!(
        base_voltage > 0.0,
        "la tension de base composée V_base doit être > 0"
    );
    base_power / (sqrt_three() * base_voltage)
}

/// Valeur réduite (per-unit) d'une grandeur `x_pu = X_actual / X_base` (sans
/// dimension), `X_actual` et `X_base` étant de **même nature** (même unité).
///
/// Panique si `base_value == 0` (base de réduction nulle).
pub fn pu_per_unit_value(actual_value: f64, base_value: f64) -> f64 {
    assert!(
        base_value != 0.0,
        "la grandeur de base X_base doit être ≠ 0"
    );
    actual_value / base_value
}

/// Changement de base d'une **impédance** réduite
/// `Z_pu,new = Z_pu,old · (S_base,new / S_base,old) · (V_base,old / V_base,new)²`
/// (p.u.).
///
/// Panique si `base_power_old <= 0`, si `base_power_new < 0`, si
/// `base_voltage_old < 0` ou si `base_voltage_new <= 0`.
pub fn pu_change_base_impedance(
    pu_old: f64,
    base_power_old: f64,
    base_power_new: f64,
    base_voltage_old: f64,
    base_voltage_new: f64,
) -> f64 {
    assert!(
        base_power_old > 0.0,
        "l'ancienne puissance de base S_base,old doit être > 0"
    );
    assert!(
        base_power_new >= 0.0,
        "la nouvelle puissance de base S_base,new doit être ≥ 0"
    );
    assert!(
        base_voltage_old >= 0.0,
        "l'ancienne tension de base V_base,old doit être ≥ 0"
    );
    assert!(
        base_voltage_new > 0.0,
        "la nouvelle tension de base V_base,new doit être > 0"
    );
    let voltage_ratio = base_voltage_old / base_voltage_new;
    pu_old * (base_power_new / base_power_old) * voltage_ratio.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn base_impedance_realistic_case() {
        // V_base = 20 kV (composée), S_base = 100 MVA :
        //   Z_base = 20000² / 100e6 = 4e8 / 1e8 = 4 Ω
        let z = pu_base_impedance(20_000.0, 100.0e6);
        assert_relative_eq!(z, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn base_current_realistic_case() {
        // V_base = 20 kV, S_base = 100 MVA :
        //   I_base = 100e6 / (√3 · 20000) = 100e6 / 34641.016... ≈ 2886.7513 A
        let i = pu_base_current(100.0e6, 20_000.0);
        let expected = 100.0e6 / (3.0_f64.sqrt() * 20_000.0);
        assert_relative_eq!(i, expected, epsilon = 1e-6);
        assert_relative_eq!(i, 2_886.751_346, epsilon = 1e-3);
    }

    #[test]
    fn base_impedance_relates_to_base_current() {
        // Identité triphasée : Z_base = V_base / (√3 · I_base).
        // En effet V/(√3·I_base) = V / (√3 · S/(√3·V)) = V²/S = Z_base.
        let v = 20_000.0_f64;
        let s = 100.0e6_f64;
        let z = pu_base_impedance(v, s);
        let i = pu_base_current(s, v);
        assert_relative_eq!(z, v / (3.0_f64.sqrt() * i), epsilon = 1e-6);
    }

    #[test]
    fn per_unit_value_is_proportional_and_normalizes_base() {
        // x_pu = X_actual / X_base : proportionnalité et x_pu = 1 quand
        // X_actual = X_base.
        let base = 4.0_f64;
        assert_relative_eq!(pu_per_unit_value(2.0, base), 0.5, epsilon = 1e-12);
        assert_relative_eq!(pu_per_unit_value(base, base), 1.0, epsilon = 1e-12);
        assert_relative_eq!(pu_per_unit_value(8.0, base), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn change_base_power_only_scales_linearly() {
        // Tension de base inchangée : Z_pu,new = Z_pu,old · S_new/S_old.
        // 0.1 · (200e6/100e6) · 1 = 0.2
        let z_new = pu_change_base_impedance(0.1, 100.0e6, 200.0e6, 20_000.0, 20_000.0);
        assert_relative_eq!(z_new, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn change_base_is_reversible() {
        // Réciprocité : passer de (S_old,V_old) à (S_new,V_new) puis revenir
        // restitue la valeur réduite d'origine.
        let z0 = 0.15_f64;
        let z1 = pu_change_base_impedance(z0, 100.0e6, 250.0e6, 20_000.0, 22_000.0);
        let z2 = pu_change_base_impedance(z1, 250.0e6, 100.0e6, 22_000.0, 20_000.0);
        assert_relative_eq!(z2, z0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la grandeur de base X_base doit être ≠ 0")]
    fn per_unit_value_rejects_zero_base() {
        pu_per_unit_value(230.0, 0.0);
    }
}
