//! Amortissement de **Coulomb** (frottement sec) d'un oscillateur à 1 ddl :
//! décroissance **linéaire** de l'amplitude et arrêt dans la bande morte.
//!
//! ```text
//! perte par cycle       ΔA = 4·F_f/k        (décroissance linéaire)
//! demi-bande d'arrêt    δ  = F_f/k          (bande morte)
//! amplitude à n cycles  A_n = A_0 − n·4·F_f/k   (bornée à δ)
//! cycles jusqu'à l'arrêt N = (A_0 − F_f/k)/(4·F_f/k)
//! ```
//!
//! `F_f` force de frottement de Coulomb (N), `k` raideur de l'oscillateur (N·m⁻¹),
//! `A_0` amplitude initiale (m), `A_n` amplitude après `n` cycles (m), `n` nombre
//! de cycles (sans dimension), `ΔA` perte d'amplitude par cycle (m), `δ` demi-bande
//! morte où le mouvement cesse (m), `N` nombre de cycles jusqu'à l'arrêt.
//!
//! **Convention** : SI. **Limite honnête** : frottement sec **constant** — la force
//! de Coulomb `F_f` (issue du coefficient de frottement et de la force normale) est
//! **fournie par l'appelant**, jamais inventée ; oscillateur à 1 ddl dont l'amplitude
//! décroît **linéairement** (≠ décroissance exponentielle du visqueux) et dont le
//! mouvement s'arrête dans la bande morte. Distinct de [`crate::vibrations`]
//! (amortissement **visqueux**).

/// Perte d'amplitude par cycle `ΔA = 4·F_f/k`.
///
/// Panique si `friction_force < 0` ou `stiffness <= 0`.
pub fn coulomb_amplitude_loss_per_cycle(friction_force: f64, stiffness: f64) -> f64 {
    assert!(
        friction_force >= 0.0 && stiffness > 0.0,
        "F_f ≥ 0 et k > 0 requis"
    );
    4.0 * friction_force / stiffness
}

/// Demi-bande morte `δ = F_f/k` (amplitude en deçà de laquelle le mouvement cesse).
///
/// Panique si `friction_force < 0` ou `stiffness <= 0`.
pub fn coulomb_dead_band(friction_force: f64, stiffness: f64) -> f64 {
    assert!(
        friction_force >= 0.0 && stiffness > 0.0,
        "F_f ≥ 0 et k > 0 requis"
    );
    friction_force / stiffness
}

/// Amplitude après `cycles` cycles `A_n = A_0 − n·4·F_f/k`, bornée à la demi-bande
/// morte `δ = F_f/k` (le mouvement ne peut descendre sous la bande d'arrêt).
///
/// Panique si `initial_amplitude < 0`, `friction_force < 0`, `stiffness <= 0`
/// ou `cycles < 0`.
pub fn coulomb_amplitude_after_cycles(
    initial_amplitude: f64,
    friction_force: f64,
    stiffness: f64,
    cycles: f64,
) -> f64 {
    assert!(
        initial_amplitude >= 0.0 && friction_force >= 0.0 && stiffness > 0.0 && cycles >= 0.0,
        "A_0 ≥ 0, F_f ≥ 0, k > 0 et n ≥ 0 requis"
    );
    let dead_band = friction_force / stiffness;
    let amplitude = initial_amplitude - cycles * 4.0 * friction_force / stiffness;
    amplitude.max(dead_band)
}

/// Nombre de cycles jusqu'à l'arrêt `N = (A_0 − F_f/k)/(4·F_f/k)`.
///
/// Panique si `initial_amplitude < 0`, `friction_force <= 0` ou `stiffness <= 0`.
pub fn coulomb_cycles_to_stop(initial_amplitude: f64, friction_force: f64, stiffness: f64) -> f64 {
    assert!(
        initial_amplitude >= 0.0 && friction_force > 0.0 && stiffness > 0.0,
        "A_0 ≥ 0, F_f > 0 et k > 0 requis"
    );
    (initial_amplitude - friction_force / stiffness) / (4.0 * friction_force / stiffness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn loss_is_four_dead_bands() {
        // Identité structurelle : ΔA = 4·δ.
        let (friction_force, stiffness) = (10.0_f64, 1000.0_f64);
        let loss = coulomb_amplitude_loss_per_cycle(friction_force, stiffness);
        let dead_band = coulomb_dead_band(friction_force, stiffness);
        assert_relative_eq!(loss, 4.0 * dead_band, max_relative = 1e-12);
    }

    #[test]
    fn loss_proportional_to_friction() {
        // ΔA ∝ F_f : doubler la force de frottement double la perte par cycle.
        let l1 = coulomb_amplitude_loss_per_cycle(10.0, 1000.0);
        let l2 = coulomb_amplitude_loss_per_cycle(20.0, 1000.0);
        assert_relative_eq!(l2 / l1, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_case_values() {
        // F_f = 10 N, k = 1000 N/m, A_0 = 0,1 m :
        // ΔA = 4·10/1000 = 0,04 m ; δ = 10/1000 = 0,01 m.
        let (friction_force, stiffness, initial_amplitude) = (10.0_f64, 1000.0_f64, 0.1_f64);
        assert_relative_eq!(
            coulomb_amplitude_loss_per_cycle(friction_force, stiffness),
            0.04,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            coulomb_dead_band(friction_force, stiffness),
            0.01,
            max_relative = 1e-12
        );
        // A_2 = 0,1 − 2·0,04 = 0,02 m (au-dessus de la bande morte).
        assert_relative_eq!(
            coulomb_amplitude_after_cycles(initial_amplitude, friction_force, stiffness, 2.0),
            0.02,
            max_relative = 1e-12
        );
        // N = (0,1 − 0,01)/0,04 = 2,25 cycles.
        assert_relative_eq!(
            coulomb_cycles_to_stop(initial_amplitude, friction_force, stiffness),
            2.25,
            max_relative = 1e-12
        );
    }

    #[test]
    fn amplitude_at_stop_equals_dead_band() {
        // À N cycles, l'amplitude atteint exactement la demi-bande morte.
        let (friction_force, stiffness, initial_amplitude) = (10.0_f64, 1000.0_f64, 0.1_f64);
        let n_stop = coulomb_cycles_to_stop(initial_amplitude, friction_force, stiffness);
        let dead_band = coulomb_dead_band(friction_force, stiffness);
        assert_relative_eq!(
            coulomb_amplitude_after_cycles(initial_amplitude, friction_force, stiffness, n_stop),
            dead_band,
            max_relative = 1e-12
        );
    }

    #[test]
    fn amplitude_clamped_to_dead_band() {
        // Au-delà de l'arrêt, l'amplitude reste bornée à la bande morte.
        let (friction_force, stiffness, initial_amplitude) = (10.0_f64, 1000.0_f64, 0.1_f64);
        let dead_band = coulomb_dead_band(friction_force, stiffness);
        assert_relative_eq!(
            coulomb_amplitude_after_cycles(initial_amplitude, friction_force, stiffness, 100.0),
            dead_band,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "k > 0")]
    fn zero_stiffness_panics() {
        coulomb_amplitude_loss_per_cycle(10.0, 0.0);
    }
}
