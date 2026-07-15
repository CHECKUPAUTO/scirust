//! **Couple efficace (RMS) sur un cycle de service** pour le dimensionnement
//! thermique d'un moteur (couple efficace, facteur de marche, puissance moyenne,
//! rapport crête/RMS).
//!
//! ```text
//! couple efficace   C_rms = √( Σ Cᵢ²·tᵢ / Σ tᵢ )
//! facteur de marche  D = t_on / t_cycle
//! puissance moyenne  P_moy = Σ Pᵢ·tᵢ / Σ tᵢ
//! rapport crête/RMS  k = C_crête / C_rms
//! ```
//!
//! `Cᵢ` couples des segments du profil de mission (N·m), `tᵢ` durées associées
//! (s), `C_rms` couple efficace (N·m), `t_on` temps sous charge et `t_cycle`
//! période du cycle (s), `D` facteur de marche (sans dimension, 0..1), `Pᵢ`
//! puissances des segments (W), `P_moy` puissance moyenne (W), `C_crête` couple
//! de pointe du cycle (N·m), `k` rapport crête/RMS (sans dimension).
//!
//! **Convention** : SI (N·m, s, W). **Limite honnête** : le couple efficace
//! (RMS) reflète l'échauffement (pertes ∝ couple²) et doit rester **≤ couple
//! nominal continu** du moteur ; le couple **crête** du cycle doit rester
//! **≤ couple de pointe admissible** du moteur ; ces deux limites moteur, ainsi
//! que les segments `(couple/puissance, durée)` du profil de mission, sont des
//! **données fournies** par l'appelant — aucune valeur « par défaut » n'est
//! inventée ici.

/// Couple **efficace (RMS)** d'un cycle `C_rms = √(Σ Cᵢ²·tᵢ / Σ tᵢ)`.
///
/// `torques` couples des segments (N·m), `durations` durées associées (s),
/// même longueur. Base du dimensionnement thermique du moteur.
/// Panique si les tranches sont vides ou de longueurs différentes, si une
/// durée est négative ou si la durée totale du cycle est nulle.
pub fn dutycycle_rms_torque(torques: &[f64], durations: &[f64]) -> f64 {
    assert!(!torques.is_empty(), "au moins un segment est requis");
    assert_eq!(
        torques.len(),
        durations.len(),
        "couples et durées doivent avoir la même longueur"
    );
    let mut sum_sq_t = 0.0_f64;
    let mut sum_t = 0.0_f64;
    for (&torque, &dt) in torques.iter().zip(durations.iter())
    {
        assert!(dt >= 0.0, "les durées doivent être positives");
        sum_sq_t += torque * torque * dt;
        sum_t += dt;
    }
    assert!(
        sum_t > 0.0,
        "la durée totale du cycle doit être strictement positive"
    );
    (sum_sq_t / sum_t).sqrt()
}

/// **Facteur de marche** `D = t_on / t_cycle`.
///
/// `on_time` temps sous charge (s), `cycle_time` période du cycle (s).
/// Panique si `cycle_time <= 0`, si `on_time < 0` ou si `on_time > cycle_time`.
pub fn dutycycle_duty_factor(on_time: f64, cycle_time: f64) -> f64 {
    assert!(
        cycle_time > 0.0,
        "la période du cycle doit être strictement positive"
    );
    assert!(on_time >= 0.0, "le temps sous charge doit être positif");
    assert!(
        on_time <= cycle_time,
        "le temps sous charge ne peut pas dépasser la période du cycle"
    );
    on_time / cycle_time
}

/// **Puissance moyenne** sur un cycle `P_moy = Σ Pᵢ·tᵢ / Σ tᵢ`.
///
/// `powers` puissances des segments (W), `durations` durées associées (s),
/// même longueur.
/// Panique si les tranches sont vides ou de longueurs différentes, si une
/// durée est négative ou si la durée totale du cycle est nulle.
pub fn dutycycle_average_power(powers: &[f64], durations: &[f64]) -> f64 {
    assert!(!powers.is_empty(), "au moins un segment est requis");
    assert_eq!(
        powers.len(),
        durations.len(),
        "puissances et durées doivent avoir la même longueur"
    );
    let mut sum_p_t = 0.0_f64;
    let mut sum_t = 0.0_f64;
    for (&power, &dt) in powers.iter().zip(durations.iter())
    {
        assert!(dt >= 0.0, "les durées doivent être positives");
        sum_p_t += power * dt;
        sum_t += dt;
    }
    assert!(
        sum_t > 0.0,
        "la durée totale du cycle doit être strictement positive"
    );
    sum_p_t / sum_t
}

/// **Rapport crête/RMS** `k = C_crête / C_rms`.
///
/// `peak_torque` couple de pointe du cycle (N·m), `rms_torque` couple efficace
/// (N·m). Indicateur du sur-dimensionnement en pointe.
/// Panique si `rms_torque <= 0` ou si `peak_torque < 0`.
pub fn dutycycle_peak_to_rms_ratio(peak_torque: f64, rms_torque: f64) -> f64 {
    assert!(
        rms_torque > 0.0,
        "le couple efficace doit être strictement positif"
    );
    assert!(peak_torque >= 0.0, "le couple de pointe doit être positif");
    peak_torque / rms_torque
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rms_of_constant_is_that_value() {
        // Couple constant sur tout le cycle → RMS = ce couple (identité).
        let c = dutycycle_rms_torque(&[7.0, 7.0, 7.0], &[1.0, 2.0, 5.0]);
        assert_relative_eq!(c, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn rms_between_min_and_max() {
        // Cycle accél/repos/accél : RMS entre le couple mini et maxi du cycle.
        // √((900·0,2 + 100·0,6 + 900·0,2)/1,0) = √420 ≈ 20,4939.
        let torques = [30.0, 10.0, 30.0];
        let durations = [0.2, 0.6, 0.2];
        let c_rms = dutycycle_rms_torque(&torques, &durations);
        assert_relative_eq!(c_rms, (420.0_f64).sqrt(), epsilon = 1e-9);
        assert!(c_rms > 10.0 && c_rms < 30.0);
    }

    #[test]
    fn duty_factor_bounds_and_value() {
        // Marche 3 s sur une période de 12 s → D = 0,25.
        assert_relative_eq!(dutycycle_duty_factor(3.0, 12.0), 0.25, epsilon = 1e-12);
        // Cas limites : marche nulle → 0, marche continue → 1.
        assert_relative_eq!(dutycycle_duty_factor(0.0, 12.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(dutycycle_duty_factor(12.0, 12.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn average_power_is_time_weighted_mean() {
        // Deux segments : (400 W, 2 s) et (100 W, 6 s).
        // (400·2 + 100·6)/8 = (800 + 600)/8 = 175 W.
        let powers = [400.0, 100.0];
        let durations = [2.0, 6.0];
        assert_relative_eq!(
            dutycycle_average_power(&powers, &durations),
            175.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn peak_to_rms_reciprocity() {
        // k = crête/RMS ; crête = k·RMS reconstitue la crête (réciprocité).
        let peak = 45.0;
        let rms = (420.0_f64).sqrt();
        let k = dutycycle_peak_to_rms_ratio(peak, rms);
        assert_relative_eq!(k * rms, peak, epsilon = 1e-9);
        // Un couple constant a un rapport crête/RMS de 1.
        assert_relative_eq!(
            dutycycle_peak_to_rms_ratio(12.0, 12.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "au moins un segment")]
    fn empty_cycle_panics() {
        dutycycle_rms_torque(&[], &[]);
    }
}
