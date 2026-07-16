//! **Circuit d'amortissement RC (snubber)** — dimensionnement d'un réseau RC
//! de protection placé aux bornes d'un interrupteur (thyristor, IGBT, contact) :
//! capacité limitant la vitesse de montée de tension `dv/dt`, résistance
//! limitant le courant de décharge au réamorçage, puissance dissipée en
//! commutation et résistance d'amortissement critique de la maille parasite.
//!
//! ```text
//! capacité snubber      C = I / (dV/dt)
//! résistance de décharge R = V / I_peak
//! puissance dissipée     P = C · V² · f
//! amortissement critique R_crit = 2 · √(L / C)
//! ```
//!
//! `I` courant de charge admissible (A), `dV/dt` vitesse de montée de tension
//! admissible par l'interrupteur (V/s), `C` capacité du snubber (F), `V` tension
//! aux bornes du condensateur au moment du réamorçage (V), `I_peak` courant de
//! décharge crête admissible (A), `R` résistance série du snubber (Ω), `f`
//! fréquence de commutation (Hz), `P` puissance active dissipée (W), `L`
//! inductance parasite de la maille (H), `R_crit` résistance d'amortissement
//! critique (Ω).
//!
//! **Convention** : SI ; tensions en V, courants en A, capacités en F,
//! inductances en H, résistances en Ω, fréquences en Hz, puissances en W.
//! Types `f64`, arithmétique réelle.
//!
//! **Limite honnête** : snubber RC de protection d'un **interrupteur**. Le
//! courant de charge `I` et la vitesse de montée de tension admissible `dV/dt`
//! sont **fournis par l'appelant** (d'après la fiche du composant ou une
//! mesure) : la capacité limite le `dv/dt`, la résistance limite le courant de
//! décharge au réamorçage et la puissance dissipée croît avec la fréquence de
//! commutation. L'amortissement critique `R_crit = 2·√(L/C)` évite les
//! oscillations de la maille parasite (`L`, `C` **fournis** par l'appelant, non
//! inventés). Ces relations sont des dimensionnements de première approche : ni
//! l'effet des résistances/inductances réparties, ni les pertes fer/diélectrique
//! ne sont modélisés.

/// Capacité du snubber limitant la vitesse de montée de tension aux bornes de
/// l'interrupteur : `C = I / (dV/dt)` (F). Plus le `dv/dt` admissible est faible,
/// plus la capacité requise est grande.
///
/// `load_current` en A, `voltage_rise_rate` en V/s, résultat en F.
///
/// Panique si `voltage_rise_rate <= 0` (division par zéro) ou si
/// `load_current < 0` (courant physiquement ≥ 0).
pub fn snub_capacitance(load_current: f64, voltage_rise_rate: f64) -> f64 {
    assert!(load_current >= 0.0, "le courant de charge I doit être ≥ 0");
    assert!(
        voltage_rise_rate > 0.0,
        "la vitesse de montée de tension dV/dt doit être strictement positive"
    );
    load_current / voltage_rise_rate
}

/// Résistance série du snubber limitant le courant de décharge du condensateur
/// au réamorçage de l'interrupteur : `R = V / I_peak` (Ω).
///
/// `voltage` en V, `peak_discharge_current` en A, résultat en Ω.
///
/// Panique si `peak_discharge_current <= 0` (division par zéro) ou si
/// `voltage < 0` (tension en module ≥ 0).
pub fn snub_resistance(voltage: f64, peak_discharge_current: f64) -> f64 {
    assert!(voltage >= 0.0, "la tension V doit être ≥ 0");
    assert!(
        peak_discharge_current > 0.0,
        "le courant de décharge crête I_peak doit être strictement positif"
    );
    voltage / peak_discharge_current
}

/// Puissance active dissipée dans un snubber RC : à chaque commutation l'énergie
/// `½·C·V²` est chargée puis dissipée, soit `P = C · V² · f` sur un cycle
/// charge/décharge (W). Elle croît linéairement avec la fréquence de commutation.
///
/// `capacitance` en F, `voltage` en V, `switching_frequency` en Hz, résultat en W.
///
/// Panique si `capacitance < 0` ou si `switching_frequency < 0` (grandeurs
/// physiquement ≥ 0).
pub fn snub_power_dissipation(capacitance: f64, voltage: f64, switching_frequency: f64) -> f64 {
    assert!(capacitance >= 0.0, "la capacité C doit être ≥ 0");
    assert!(
        switching_frequency >= 0.0,
        "la fréquence de commutation f doit être ≥ 0"
    );
    capacitance * voltage * voltage * switching_frequency
}

/// Résistance d'amortissement critique d'une maille RLC parasite (inductance de
/// câblage `L` et capacité du snubber `C`) : `R_crit = 2 · √(L / C)` (Ω). Une
/// résistance série égale à `R_crit` amène la maille au régime critique et évite
/// les oscillations (dépassement) au réamorçage.
///
/// `inductance` en H, `capacitance` en F, résultat en Ω.
///
/// Panique si `inductance < 0` (inductance physiquement ≥ 0) ou si
/// `capacitance <= 0` (division par zéro sous la racine).
pub fn snub_critical_damping_resistance(inductance: f64, capacitance: f64) -> f64 {
    assert!(inductance >= 0.0, "l'inductance parasite L doit être ≥ 0");
    assert!(
        capacitance > 0.0,
        "la capacité C doit être strictement positive"
    );
    2.0 * (inductance / capacitance).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn capacitance_scales_inversely_with_rise_rate() {
        // Proportionnalité inverse : à courant fixé, doubler le dv/dt admissible
        // divise par deux la capacité requise.
        let i = 10.0_f64;
        let c1 = snub_capacitance(i, 500.0e6);
        let c2 = snub_capacitance(i, 1000.0e6);
        assert_relative_eq!(c1 / c2, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn capacitance_worked_case() {
        // Cas chiffré : I = 20 A, dV/dt = 200 V/µs = 200e6 V/s.
        //   C = 20 / 200e6 = 1e-7 F = 100 nF.
        // Recalcul indépendant : 20 / 2.0e8 = 1.0e-7. (littéral vérifié deux fois)
        let c = snub_capacitance(20.0, 200.0e6);
        assert_relative_eq!(c, 1.0e-7, epsilon = 1e-12);
    }

    #[test]
    fn resistance_and_ohms_law_reciprocity() {
        // Réciprocité loi d'Ohm : R = V/I_peak, donc V = R·I_peak restitue la
        // tension de départ.
        let v = 400.0_f64;
        let i_peak = 50.0_f64;
        let r = snub_resistance(v, i_peak);
        assert_relative_eq!(r * i_peak, v, epsilon = 1e-9);
    }

    #[test]
    fn power_scales_with_voltage_squared_and_frequency() {
        // Proportionnalités : P ∝ V² et P ∝ f. Doubler V quadruple P ;
        // doubler f double P.
        let c = 100.0e-9_f64;
        let p1 = snub_power_dissipation(c, 300.0, 10.0e3);
        let p_v2 = snub_power_dissipation(c, 600.0, 10.0e3);
        let p_f2 = snub_power_dissipation(c, 300.0, 20.0e3);
        assert_relative_eq!(p_v2 / p1, 4.0, epsilon = 1e-9);
        assert_relative_eq!(p_f2 / p1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn power_worked_case() {
        // Cas chiffré : C = 100 nF, V = 500 V, f = 20 kHz.
        //   P = 100e-9 · 500² · 20e3 = 1e-7 · 250000 · 20000 = 500 W.
        // Recalcul : 1e-7 · 2.5e5 = 2.5e-2 ; 2.5e-2 · 2.0e4 = 500. (vérifié deux fois)
        let p = snub_power_dissipation(100.0e-9, 500.0, 20.0e3);
        assert_relative_eq!(p, 500.0, epsilon = 1e-3);
    }

    #[test]
    fn critical_damping_worked_case() {
        // Cas chiffré : L = 1 µH, C = 100 nF.
        //   L/C = 1e-6 / 1e-7 = 10 ; √10 ≈ 3.16227766 ; R_crit = 2·√10 ≈ 6.32455532.
        // Recalcul : 2 · 3.16227766 = 6.32455532. (vérifié deux fois)
        let r_crit = snub_critical_damping_resistance(1.0e-6, 100.0e-9);
        assert_relative_eq!(r_crit, 6.324_555_320, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(
        expected = "la vitesse de montée de tension dV/dt doit être strictement positive"
    )]
    fn zero_rise_rate_panics() {
        snub_capacitance(10.0, 0.0);
    }
}
