//! **Exposition au bruit** — dose de bruit, durée d'exposition admissible et
//! niveau continu équivalent (ISO 1999 / OSHA), pour évaluer le risque auditif
//! d'un poste de travail (atelier, machine-outil).
//!
//! ```text
//! dose de bruit          D = 100·T / T_perm           (% de la dose journalière)
//! durée admissible       T_perm = 8 / 2^((L−L_c)/q)   (heures)
//! niveau équivalent      L_eq = 10·log10( Σ tᵢ·10^(Lᵢ/10) / Σ tᵢ )
//! ```
//!
//! `T` durée d'exposition réelle (h), `T_perm` durée d'exposition admissible (h),
//! `D` dose de bruit (% ; 100 % = pleine dose journalière autorisée), `L` niveau
//! de pression acoustique pondéré A (dB(A)), `L_c` niveau critère réglementaire
//! (dB(A)), `q` taux d'échange (dB ; +q dB divise par deux la durée admissible),
//! `Lᵢ`/`tᵢ` niveaux et durées des segments d'exposition, `L_eq` niveau continu
//! équivalent sur la durée totale (dB(A)).
//!
//! **Convention** : durées en heures, niveaux en décibels pondérés A ; le
//! logarithme est en base 10. **Limite honnête** : le niveau critère `L_c`
//! (typiquement 85 dB(A)) et le taux d'échange `q` (3 dB selon ISO 1999,
//! 5 dB selon OSHA) sont **fixés par la réglementation applicable** et
//! **fournis par l'appelant** ; aucune valeur « par défaut » n'est inventée.
//! Ces formules ne remplacent pas une **dosimétrie** normalisée réalisée avec un
//! appareil étalonné.

/// Dose de bruit `D = 100·T / T_perm` (en % de la dose journalière).
///
/// Panique si `exposure_time < 0` ou `permitted_time <= 0`.
pub fn noise_dose_percent(exposure_time: f64, permitted_time: f64) -> f64 {
    assert!(
        exposure_time >= 0.0,
        "la durée d'exposition doit être positive"
    );
    assert!(
        permitted_time > 0.0,
        "T_perm > 0 requis (dénominateur non nul)"
    );
    100.0 * exposure_time / permitted_time
}

/// Durée d'exposition admissible `T_perm = 8 / 2^((L−L_c)/q)` (en heures).
///
/// Au niveau critère (`sound_level == criterion_level`) elle vaut 8 h ; chaque
/// tranche de `q` dB au-dessus du critère la divise par deux.
///
/// Panique si `exchange_rate <= 0`.
pub fn noise_permitted_time(sound_level: f64, criterion_level: f64, exchange_rate: f64) -> f64 {
    assert!(
        exchange_rate > 0.0,
        "le taux d'échange q doit être strictement positif"
    );
    8.0 / 2.0_f64.powf((sound_level - criterion_level) / exchange_rate)
}

/// Niveau continu équivalent `L_eq = 10·log10( Σ tᵢ·10^(Lᵢ/10) / Σ tᵢ )` (dB(A)).
///
/// Moyenne énergétique des niveaux `levels`, pondérée par les durées `times`.
///
/// Panique si `levels` est vide, si `levels.len() != times.len()`, si une durée
/// est négative ou si la durée totale `Σ tᵢ` est nulle.
pub fn noise_equivalent_continuous_level(levels: &[f64], times: &[f64]) -> f64 {
    assert!(
        !levels.is_empty(),
        "au moins un segment d'exposition requis"
    );
    assert!(
        levels.len() == times.len(),
        "levels et times doivent avoir la même longueur"
    );
    let mut total_time = 0.0_f64;
    let mut weighted_energy = 0.0_f64;
    for (&level, &time) in levels.iter().zip(times.iter())
    {
        assert!(time >= 0.0, "chaque durée doit être positive");
        total_time += time;
        weighted_energy += time * 10.0_f64.powf(level / 10.0);
    }
    assert!(
        total_time > 0.0,
        "la durée totale doit être strictement positive"
    );
    10.0 * (weighted_energy / total_time).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dose_is_proportional_to_exposure_time() {
        // D ∝ T : doubler la durée d'exposition double la dose.
        let d1 = noise_dose_percent(4.0, 8.0);
        let d2 = noise_dose_percent(8.0, 8.0);
        assert_relative_eq!(d2, 2.0 * d1, epsilon = 1e-9);
        // Exposition = durée admissible → 100 % pile.
        assert_relative_eq!(noise_dose_percent(8.0, 8.0), 100.0, epsilon = 1e-9);
    }

    #[test]
    fn dose_realistic_case() {
        // 8 h d'exposition pour 4 h admissibles → dose de 200 %.
        assert_relative_eq!(noise_dose_percent(8.0, 4.0), 200.0, epsilon = 1e-9);
    }

    #[test]
    fn permitted_time_equals_eight_hours_at_criterion() {
        // Au niveau critère l'exposant est nul → T_perm = 8 h (cas limite).
        assert_relative_eq!(noise_permitted_time(85.0, 85.0, 3.0), 8.0, epsilon = 1e-12);
    }

    #[test]
    fn permitted_time_halves_per_exchange_rate() {
        // +q dB divise par deux la durée admissible (identité de définition).
        let base = noise_permitted_time(85.0, 85.0, 3.0);
        let plus_q = noise_permitted_time(88.0, 85.0, 3.0);
        assert_relative_eq!(plus_q, 0.5 * base, epsilon = 1e-12);
        // 88 dB, critère 85, q=3 → 8/2^1 = 4 h.
        assert_relative_eq!(plus_q, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn equivalent_level_of_constant_exposure_is_that_level() {
        // Niveau constant sur tous les segments → L_eq = ce niveau (réciprocité).
        let leq = noise_equivalent_continuous_level(&[90.0, 90.0], &[4.0, 4.0]);
        assert_relative_eq!(leq, 90.0, epsilon = 1e-9);
    }

    #[test]
    fn equivalent_level_realistic_case() {
        // 1 h à 80 dB puis 1 h à 90 dB → L_eq = 10·log10((1e8+1e9)/2) ≈ 87,4036 dB.
        let leq = noise_equivalent_continuous_level(&[80.0, 90.0], &[1.0, 1.0]);
        assert_relative_eq!(leq, 87.403_626_894_942_44, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "T_perm > 0 requis")]
    fn zero_permitted_time_panics() {
        noise_dose_percent(8.0, 0.0);
    }
}
