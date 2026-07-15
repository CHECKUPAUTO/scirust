//! **Transmission à variation continue (CVT à poulies-courroie)** — rapport de
//! vitesse fixé par les rayons d'enroulement effectifs, vitesse et couple de
//! sortie par conservation de la puissance, et plage (spread) de variation.
//!
//! ```text
//! rapport de vitesse   r = R_menante / R_menée        (sortie/entrée)
//! vitesse de sortie    ω_out = ω_in · r
//! couple de sortie     T_out = (T_in / r) · η         (couple ∝ 1/r)
//! plage de variation   spread = r_max / r_min
//! ```
//!
//! `r` rapport de vitesse sortie/entrée (sans dimension), `R_menante`,
//! `R_menée` rayons d'enroulement effectifs des poulies menante (entrée) et
//! menée (sortie) (m), `ω_in`, `ω_out` vitesses angulaires d'entrée et de
//! sortie (rad/s), `T_in`, `T_out` couples d'entrée et de sortie (N·m),
//! `η` rendement de transmission (sans dimension, `0 < η ≤ 1`), `r_max`,
//! `r_min` rapports extrêmes atteignables (sans dimension), `spread` plage
//! de variation (sans dimension).
//!
//! **Convention** : SI cohérent. **Limite honnête** : le rapport est fixé par
//! les rayons d'enroulement effectifs des poulies **fournis** par l'appelant
//! (variation continue), et le rendement de transmission est **fourni** — aucune
//! valeur « par défaut » n'est inventée. La puissance se conserve (à `η` près),
//! d'où un couple inversement proportionnel au rapport de vitesse ; le
//! glissement de la courroie est **ignoré**.

/// Rapport de vitesse sortie/entrée `r = R_menante / R_menée`, fixé par les
/// rayons d'enroulement effectifs des poulies (variation continue).
///
/// Panique si un rayon `<= 0`.
pub fn cvt_speed_ratio(driver_pulley_radius: f64, driven_pulley_radius: f64) -> f64 {
    assert!(
        driver_pulley_radius > 0.0 && driven_pulley_radius > 0.0,
        "R_menante > 0 et R_menée > 0 requis"
    );
    driver_pulley_radius / driven_pulley_radius
}

/// Vitesse angulaire de sortie `ω_out = ω_in · r`.
///
/// Panique si `input_speed < 0` ou `speed_ratio <= 0`.
pub fn cvt_output_speed(input_speed: f64, speed_ratio: f64) -> f64 {
    assert!(input_speed >= 0.0, "ω_in ≥ 0 requis");
    assert!(speed_ratio > 0.0, "r > 0 requis");
    input_speed * speed_ratio
}

/// Couple de sortie `T_out = (T_in / r) · η`, inversement proportionnel au
/// rapport de vitesse (conservation de la puissance à `η` près).
///
/// Panique si `input_torque < 0`, `speed_ratio <= 0`, ou `efficiency` ∉ `(0, 1]`.
pub fn cvt_output_torque(input_torque: f64, speed_ratio: f64, efficiency: f64) -> f64 {
    assert!(input_torque >= 0.0, "T_in ≥ 0 requis");
    assert!(speed_ratio > 0.0, "r > 0 requis");
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "0 < η ≤ 1 requis (rendement physique)"
    );
    input_torque / speed_ratio * efficiency
}

/// Plage (spread) de variation `spread = r_max / r_min`.
///
/// Panique si `min_speed_ratio <= 0` ou `max_speed_ratio < min_speed_ratio`.
pub fn cvt_ratio_spread(max_speed_ratio: f64, min_speed_ratio: f64) -> f64 {
    assert!(min_speed_ratio > 0.0, "r_min > 0 requis");
    assert!(
        max_speed_ratio >= min_speed_ratio,
        "r_max ≥ r_min requis (plage ordonnée)"
    );
    max_speed_ratio / min_speed_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn speed_ratio_unity_for_equal_radii() {
        // Rayons égaux → prise directe r = 1.
        assert_relative_eq!(cvt_speed_ratio(0.08, 0.08), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn speed_ratio_realistic_reduction() {
        // R_menante=0.05 m, R_menée=0.10 m → r = 0.5 (réduction par deux).
        assert_relative_eq!(cvt_speed_ratio(0.05, 0.10), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn output_speed_reciprocity() {
        // Appliquer r puis 1/r restitue exactement la vitesse d'entrée.
        let omega_in = 100.0_f64;
        let r = 0.5_f64;
        let omega_out = cvt_output_speed(omega_in, r);
        assert_relative_eq!(
            cvt_output_speed(omega_out, 1.0 / r),
            omega_in,
            epsilon = 1e-12
        );
    }

    #[test]
    fn output_torque_realistic_value() {
        // T_in=20 N·m, r=0.5, η=0.9 → T_out = 20/0.5·0.9 = 40·0.9 = 36 N·m.
        assert_relative_eq!(cvt_output_torque(20.0, 0.5, 0.9), 36.0, epsilon = 1e-12);
    }

    #[test]
    fn power_is_conserved_at_unit_efficiency() {
        // P_out = T_out·ω_out doit égaler P_in = T_in·ω_in quand η = 1.
        let t_in = 15.0_f64;
        let omega_in = 200.0_f64;
        let r = 0.4_f64;
        let t_out = cvt_output_torque(t_in, r, 1.0);
        let omega_out = cvt_output_speed(omega_in, r);
        assert_relative_eq!(t_out * omega_out, t_in * omega_in, epsilon = 1e-9);
    }

    #[test]
    fn ratio_spread_realistic_value() {
        // r_max=2.0, r_min=0.4 → spread = 5.0 (plage typique d'une CVT).
        assert_relative_eq!(cvt_ratio_spread(2.0, 0.4), 5.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < η ≤ 1 requis")]
    fn efficiency_above_one_panics() {
        cvt_output_torque(20.0, 0.5, 1.2);
    }
}
