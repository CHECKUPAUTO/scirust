//! Électrotechnique — **alternateur synchrone à pôles lisses** (rotor
//! cylindrique) : f.é.m. induite par phase (formule de Kapp), vitesse de
//! synchronisme, puissance active selon l'angle de charge et régulation de
//! tension.
//!
//! ```text
//! f.é.m. par phase (Kapp) E   = 4,44 · f · Φ · N · k_w          [V]
//! vitesse de synchronisme N_s = 60 · f / p                       [tr/min]
//! puissance active        P   = E · V · sin δ / X_s              [W]
//! régulation de tension   ΔU  = (E0 − V) / V                     [sans dim.]
//! (cas limite)            P_max = E · V / X_s   pour δ = π/2      [W]
//! ```
//!
//! `f` fréquence électrique [Hz], `Φ` flux utile par pôle [Wb], `N`
//! nombre de spires **en série par phase** [sans dimension], `k_w` facteur
//! de bobinage [sans dimension, 0…1], `E` f.é.m. efficace par phase [V],
//! `N_s` vitesse de synchronisme [tr/min], `p` nombre de **paires** de pôles
//! [sans dimension], `V` tension aux bornes (par phase) [V], `X_s` réactance
//! synchrone [Ω], `δ` angle de charge (angle interne) [rad], `E0` f.é.m. à
//! vide [V], `ΔU` régulation de tension [sans dimension, ×100 pour des %].
//! Le coefficient `4,44 = √2 · π` (valeur efficace d'une f.é.m. sinusoïdale).
//!
//! **Limite honnête** : modèle de machine synchrone à **pôles lisses**
//! (rotor cylindrique, réactance synchrone `X_s` **unique** — pas de
//! saillance `X_d`/`X_q`), en **régime permanent équilibré** et sinusoïdal,
//! **résistance d'induit négligée**. Le flux `Φ`, le facteur de bobinage
//! `k_w`, la réactance `X_s` et les tensions sont **fournis par l'appelant**
//! (essais à vide/en court-circuit, catalogue ou norme) ; aucune valeur
//! « par défaut » n'est inventée. La saturation magnétique, la réaction
//! d'induit non linéaire et les régimes transitoires sont **négligés**.

/// Coefficient de Kapp `4,44 = √2 · π` pour la f.é.m. efficace d'une onde
/// sinusoïdale (sans dimension).
pub const SYNCGEN_KAPP_COEFFICIENT: f64 = 4.44;

/// F.é.m. efficace induite par phase `E = 4,44 · f · Φ · N · k_w` [V]
/// (formule de Kapp).
///
/// `frequency` fréquence électrique en hertz (Hz), `flux_per_pole` flux
/// utile par pôle en webers (Wb), `turns_per_phase` nombre de spires en
/// série par phase (sans dimension), `winding_factor` facteur de bobinage
/// `k_w` (sans dimension, `]0, 1]`) ; le résultat est la f.é.m. efficace par
/// phase en volts (V).
///
/// Panique si `frequency < 0`, `flux_per_pole < 0`, `turns_per_phase < 0`,
/// ou si `winding_factor` n'est pas dans l'intervalle `]0, 1]`.
pub fn syncgen_generated_emf_rms(
    frequency: f64,
    flux_per_pole: f64,
    turns_per_phase: f64,
    winding_factor: f64,
) -> f64 {
    assert!(
        frequency >= 0.0,
        "la fréquence doit être positive ou nulle (Hz)"
    );
    assert!(
        flux_per_pole >= 0.0,
        "le flux par pôle doit être positif ou nul (Wb)"
    );
    assert!(
        turns_per_phase >= 0.0,
        "le nombre de spires par phase doit être positif ou nul"
    );
    assert!(
        winding_factor > 0.0 && winding_factor <= 1.0,
        "le facteur de bobinage doit être dans ]0, 1]"
    );
    SYNCGEN_KAPP_COEFFICIENT * frequency * flux_per_pole * turns_per_phase * winding_factor
}

/// Vitesse de synchronisme `N_s = 60 · f / p` [tr/min].
///
/// `frequency` fréquence électrique en hertz (Hz), `pole_pairs` nombre de
/// **paires** de pôles `p` (sans dimension, strictement positif) ; le
/// résultat est en tours par minute (tr/min).
///
/// Panique si `frequency < 0` ou si `pole_pairs <= 0` (division).
pub fn syncgen_synchronous_speed_rpm(frequency: f64, pole_pairs: f64) -> f64 {
    assert!(
        frequency >= 0.0,
        "la fréquence doit être positive ou nulle (Hz)"
    );
    assert!(
        pole_pairs > 0.0,
        "le nombre de paires de pôles doit être strictement positif"
    );
    60.0 * frequency / pole_pairs
}

/// Puissance active développée `P = E · V · sin δ / X_s` [W] (machine à
/// rotor cylindrique, résistance d'induit négligée).
///
/// `emf` f.é.m. efficace par phase en volts (V), `terminal_voltage` tension
/// aux bornes par phase en volts (V), `synchronous_reactance` réactance
/// synchrone `X_s` en ohms (Ω, strictement positive), `load_angle_rad`
/// angle de charge `δ` en **radians** ; le résultat est en watts (W). La
/// puissance est maximale (limite de décrochage) pour `δ = π/2` :
/// `P_max = E · V / X_s`.
///
/// Panique si `emf < 0`, `terminal_voltage < 0`, ou si
/// `synchronous_reactance <= 0` (division).
pub fn syncgen_power_angle_power(
    emf: f64,
    terminal_voltage: f64,
    synchronous_reactance: f64,
    load_angle_rad: f64,
) -> f64 {
    assert!(emf >= 0.0, "la f.é.m. doit être positive ou nulle (V)");
    assert!(
        terminal_voltage >= 0.0,
        "la tension aux bornes doit être positive ou nulle (V)"
    );
    assert!(
        synchronous_reactance > 0.0,
        "la réactance synchrone doit être strictement positive (Ω)"
    );
    emf * terminal_voltage * load_angle_rad.sin() / synchronous_reactance
}

/// Régulation de tension `ΔU = (E0 − V) / V` [sans dimension] (multiplier
/// par 100 pour l'exprimer en pourcentage).
///
/// `no_load_emf` f.é.m. à vide (à excitation constante) en volts (V),
/// `terminal_voltage` tension aux bornes en charge en volts (V, strictement
/// positive) ; le résultat est sans dimension. Une valeur positive traduit
/// une chute de tension en charge (charge inductive), négative une
/// surtension (charge capacitive, effet Ferranti).
///
/// Panique si `no_load_emf < 0` ou si `terminal_voltage <= 0` (division).
pub fn syncgen_voltage_regulation(no_load_emf: f64, terminal_voltage: f64) -> f64 {
    assert!(
        no_load_emf >= 0.0,
        "la f.é.m. à vide doit être positive ou nulle (V)"
    );
    assert!(
        terminal_voltage > 0.0,
        "la tension aux bornes doit être strictement positive (V)"
    );
    (no_load_emf - terminal_voltage) / terminal_voltage
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_6};

    #[test]
    fn realistic_emf_case() {
        // Alternateur 50 Hz : Φ = 0,02 Wb/pôle, N = 200 spires/phase,
        // k_w = 0,96.
        // E = 4,44 · 50 · 0,02 · 200 · 0,96
        //   = 222 · 0,02 · 200 · 0,96 = 4,44 · 200 · 0,96
        //   = 888 · 0,96 = 852,48 V.
        let emf = syncgen_generated_emf_rms(50.0, 0.02, 200.0, 0.96);
        assert_relative_eq!(emf, 852.48, epsilon = 1e-6);
    }

    #[test]
    fn synchronous_speed_50hz_two_pole_pairs() {
        // 50 Hz, p = 2 paires (4 pôles) ⇒ N_s = 60·50/2 = 1500 tr/min.
        assert_relative_eq!(
            syncgen_synchronous_speed_rpm(50.0, 2.0),
            1500.0,
            epsilon = 1e-9
        );
        // Proportionnalité : N_s ∝ f à p fixé (doubler f double N_s).
        let n1 = syncgen_synchronous_speed_rpm(50.0, 3.0);
        let n2 = syncgen_synchronous_speed_rpm(100.0, 3.0);
        assert_relative_eq!(n2, 2.0 * n1, epsilon = 1e-9);
    }

    #[test]
    fn emf_scales_linearly_with_frequency_and_flux() {
        // E ∝ f et E ∝ Φ à autres paramètres fixés.
        let base = syncgen_generated_emf_rms(50.0, 0.02, 200.0, 0.95);
        let double_f = syncgen_generated_emf_rms(100.0, 0.02, 200.0, 0.95);
        let double_phi = syncgen_generated_emf_rms(50.0, 0.04, 200.0, 0.95);
        assert_relative_eq!(double_f, 2.0 * base, epsilon = 1e-9);
        assert_relative_eq!(double_phi, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn power_angle_realistic_and_limits() {
        // E = 1000 V, V = 800 V, X_s = 5 Ω.
        let (emf, v, xs) = (1000.0_f64, 800.0_f64, 5.0_f64);
        // δ = π/6 ⇒ sin δ = 0,5 ⇒ P = 1000·800·0,5/5 = 80 000 W.
        let p = syncgen_power_angle_power(emf, v, xs, FRAC_PI_6);
        assert_relative_eq!(p, 80_000.0, epsilon = 1e-6);
        // δ = 0 ⇒ P = 0 (pas de transfert de puissance active).
        assert_relative_eq!(
            syncgen_power_angle_power(emf, v, xs, 0.0),
            0.0,
            epsilon = 1e-9
        );
        // δ = π/2 ⇒ décrochage : P_max = E·V/X_s = 160 000 W.
        assert_relative_eq!(
            syncgen_power_angle_power(emf, v, xs, FRAC_PI_2),
            emf * v / xs,
            epsilon = 1e-6
        );
    }

    #[test]
    fn voltage_regulation_sign_and_zero() {
        // E0 = V ⇒ régulation nulle.
        assert_relative_eq!(
            syncgen_voltage_regulation(400.0, 400.0),
            0.0,
            epsilon = 1e-12
        );
        // E0 = 852,48 V, V = 800 V ⇒ ΔU = 52,48/800 = 0,0656 (soit 6,56 %).
        assert_relative_eq!(
            syncgen_voltage_regulation(852.48, 800.0),
            0.0656,
            epsilon = 1e-9
        );
        // E0 < V (charge capacitive) ⇒ régulation négative.
        assert!(syncgen_voltage_regulation(780.0, 800.0) < 0.0);
    }

    #[test]
    fn power_scales_linearly_with_emf() {
        // À V, X_s, δ fixés : P ∝ E (doubler E double P).
        let (v, xs, delta) = (800.0, 5.0, FRAC_PI_6);
        let p1 = syncgen_power_angle_power(1000.0, v, xs, delta);
        let p2 = syncgen_power_angle_power(2000.0, v, xs, delta);
        assert_relative_eq!(p2, 2.0 * p1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "réactance synchrone doit être strictement positive")]
    fn zero_reactance_panics() {
        syncgen_power_angle_power(1000.0, 800.0, 0.0, FRAC_PI_6);
    }
}
