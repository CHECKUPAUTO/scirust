//! Réducteur à **onde de déformation** (harmonic drive / strain wave) —
//! rapport de réduction, vitesse et couple de sortie en montage usuel.
//!
//! ```text
//! rapport de réduction  R = z_f/(z_c − z_f)      (circular fixe, entrée wave generator)
//! différence de dents   Δz = z_c − z_f           (typiquement 2)
//! vitesse de sortie     ω_out = ω_in/R
//! couple de sortie      T_out = T_in·R·η
//! ```
//!
//! `z_f` nombre de dents du flexspline, `z_c` nombre de dents du circular
//! spline (`z_c > z_f`), `R` rapport de réduction (sans unité), `Δz`
//! différence de dents (sans unité), `ω_in`, `ω_out` vitesses angulaires
//! d'entrée (wave generator) et de sortie (flexspline) dans la **même**
//! unité (rad/s ou tr/min), `T_in`, `T_out` couples d'entrée et de sortie
//! (N·m), `η` rendement (sans unité, 0 < η ≤ 1).
//!
//! **Convention** : montage usuel — le **circular spline est FIXE**, le wave
//! generator est l'entrée, le flexspline est la sortie ; unités SI cohérentes
//! (N·m pour les couples, unité commune pour les vitesses). **Limite honnête** :
//! rapport **idéal** ; la différence de dents (2 typiquement) et le rendement
//! `η` sont **FOURNIS par l'appelant** ; les non-linéarités (raideur torsionnelle,
//! hystérésis et fluage du flexspline, jeu, erreur cinématique) ne sont **pas**
//! modélisées. Les constantes physiques, propriétés matériaux et paramètres
//! procédé sont **fournies par l'appelant** ; aucune valeur « par défaut » n'est
//! inventée. Complète [`crate::gear_trains`] (rapports d'engrenages classiques).

/// Différence de dents `Δz = z_c − z_f` (sans unité, typiquement 2).
///
/// Nombre de dents dont le circular spline dépasse le flexspline ; c'est cette
/// petite différence qui produit la forte réduction.
///
/// `flexspline_teeth` = `z_f`, `circular_spline_teeth` = `z_c`, en nombre de dents.
///
/// Panique si un nombre de dents n'est pas fini ou n'est pas strictement positif,
/// ou si `circular_spline_teeth ≤ flexspline_teeth`.
pub fn harmonic_drive_tooth_difference(flexspline_teeth: f64, circular_spline_teeth: f64) -> f64 {
    assert!(
        flexspline_teeth.is_finite() && circular_spline_teeth.is_finite(),
        "les nombres de dents doivent être finis"
    );
    assert!(
        flexspline_teeth > 0.0 && circular_spline_teeth > 0.0,
        "les nombres de dents doivent être strictement positifs"
    );
    assert!(
        circular_spline_teeth > flexspline_teeth,
        "le circular spline doit avoir plus de dents que le flexspline"
    );
    circular_spline_teeth - flexspline_teeth
}

/// Rapport de réduction `R = z_f/(z_c − z_f)` (sans unité).
///
/// Montage usuel : circular spline **fixe**, entrée sur le wave generator,
/// sortie sur le flexspline. Le rapport est d'autant plus élevé que la
/// différence de dents `Δz = z_c − z_f` est faible.
///
/// `flexspline_teeth` = `z_f`, `circular_spline_teeth` = `z_c`, en nombre de dents.
///
/// Panique si un nombre de dents n'est pas fini ou n'est pas strictement positif,
/// ou si `circular_spline_teeth ≤ flexspline_teeth`.
pub fn harmonic_drive_reduction_ratio(flexspline_teeth: f64, circular_spline_teeth: f64) -> f64 {
    let tooth_difference = harmonic_drive_tooth_difference(flexspline_teeth, circular_spline_teeth);
    flexspline_teeth / tooth_difference
}

/// Vitesse angulaire de sortie `ω_out = ω_in/R` (même unité que `ω_in`).
///
/// La sortie (flexspline) tourne `R` fois plus lentement que l'entrée (wave
/// generator). Unité de sortie identique à celle de `input_speed`.
///
/// `input_speed` = `ω_in` (rad/s ou tr/min), `reduction_ratio` = `R` (sans unité).
///
/// Panique si `input_speed` n'est pas fini, ou si `reduction_ratio` n'est pas
/// fini ou est nul.
pub fn harmonic_drive_output_speed(input_speed: f64, reduction_ratio: f64) -> f64 {
    assert!(
        input_speed.is_finite(),
        "la vitesse d'entrée doit être finie"
    );
    assert!(
        reduction_ratio.is_finite() && reduction_ratio != 0.0,
        "le rapport de réduction doit être fini et non nul"
    );
    input_speed / reduction_ratio
}

/// Couple de sortie `T_out = T_in·R·η` (N·m).
///
/// Le couple est multiplié par le rapport de réduction puis atténué par le
/// rendement `η` (pertes d'engrènement et de déformation).
///
/// `input_torque` = `T_in` (N·m), `reduction_ratio` = `R` (sans unité),
/// `efficiency` = `η` (sans unité, 0 < η ≤ 1).
///
/// Panique si `input_torque` ou `reduction_ratio` n'est pas fini, si
/// `reduction_ratio` est négatif, ou si `efficiency` n'est pas dans `]0, 1]`.
pub fn harmonic_drive_output_torque(
    input_torque: f64,
    reduction_ratio: f64,
    efficiency: f64,
) -> f64 {
    assert!(
        input_torque.is_finite(),
        "le couple d'entrée doit être fini"
    );
    assert!(
        reduction_ratio.is_finite() && reduction_ratio >= 0.0,
        "le rapport de réduction doit être fini et positif"
    );
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement doit être dans l'intervalle ]0, 1]"
    );
    input_torque * reduction_ratio * efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tooth_difference_definition() {
        // Cas usuel : Δz = 202 − 200 = 2.
        let dz = harmonic_drive_tooth_difference(200.0, 202.0);
        assert_relative_eq!(dz, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn reduction_ratio_worked_case() {
        // z_f = 200, z_c = 202 ⇒ R = 200/(202−200) = 200/2 = 100.
        let ratio = harmonic_drive_reduction_ratio(200.0, 202.0);
        assert_relative_eq!(ratio, 100.0, epsilon = 1e-12);
        // Cohérence avec la différence de dents : R = z_f/Δz.
        let dz = harmonic_drive_tooth_difference(200.0, 202.0);
        assert_relative_eq!(ratio, 200.0 / dz, epsilon = 1e-12);
    }

    #[test]
    fn speed_and_torque_reciprocity() {
        // Réciprocité vitesse : ω_in = ω_out·R (la réduction est exactement
        // compensée en remontant la chaîne).
        let ratio = harmonic_drive_reduction_ratio(200.0, 202.0);
        let input_speed = 3000.0; // tr/min
        let output_speed = harmonic_drive_output_speed(input_speed, ratio);
        assert_relative_eq!(output_speed, 30.0, epsilon = 1e-12);
        assert_relative_eq!(output_speed * ratio, input_speed, epsilon = 1e-9);
    }

    #[test]
    fn output_torque_worked_case_and_ideal_limit() {
        // T_in = 0.5 N·m, R = 100, η = 0.8 ⇒ T_out = 0.5·100·0.8 = 40 N·m.
        let ratio = harmonic_drive_reduction_ratio(200.0, 202.0);
        let torque = harmonic_drive_output_torque(0.5, ratio, 0.8);
        assert_relative_eq!(torque, 40.0, epsilon = 1e-12);
        // Limite idéale η = 1 : T_out = T_in·R = 50 N·m, et le rendement
        // n'agit qu'en facteur multiplicatif linéaire (40 = 0.8·50).
        let ideal = harmonic_drive_output_torque(0.5, ratio, 1.0);
        assert_relative_eq!(ideal, 50.0, epsilon = 1e-12);
        assert_relative_eq!(torque, 0.8 * ideal, epsilon = 1e-12);
    }

    #[test]
    fn power_balance_with_efficiency() {
        // Bilan de puissance : P_out/P_in = η. Avec P = T·ω et ω_out = ω_in/R,
        // on a T_out·ω_out = (T_in·R·η)·(ω_in/R) = η·T_in·ω_in.
        let ratio = harmonic_drive_reduction_ratio(160.0, 162.0); // R = 80
        assert_relative_eq!(ratio, 80.0, epsilon = 1e-12);
        let (input_torque, input_speed, efficiency) = (2.0, 4000.0, 0.75);
        let output_torque = harmonic_drive_output_torque(input_torque, ratio, efficiency);
        let output_speed = harmonic_drive_output_speed(input_speed, ratio);
        let power_in = input_torque * input_speed;
        let power_out = output_torque * output_speed;
        assert_relative_eq!(power_out, efficiency * power_in, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "plus de dents que le flexspline")]
    fn non_positive_tooth_difference_panics() {
        // z_c ≤ z_f est physiquement impossible pour ce montage.
        harmonic_drive_reduction_ratio(200.0, 200.0);
    }
}
