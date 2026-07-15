//! Effet Doppler acoustique/ultrasonore — décalage de fréquence par mouvement
//! relatif de la source, de l'observateur ou d'une cible réfléchissante.
//!
//! ```text
//! source mobile     f' = f·c/(c - v_s)                (source en rapprochement)
//! observateur mobile f' = f·(c + v_o)/c               (observateur en rapprochement)
//! cible réfléchie   Δf = 2·f·v·cos(θ)/c               (débitmétrie/vélocimétrie)
//! réciproque        v  = Δf·c/(2·f·cos(θ))            (vitesse depuis le décalage)
//! ```
//!
//! `f` fréquence émise à la source (Hz), `c` célérité de l'onde dans le milieu
//! (m·s⁻¹), `v_s` vitesse de rapprochement de la source (m·s⁻¹), `v_o` vitesse de
//! rapprochement de l'observateur (m·s⁻¹), `v` vitesse de la cible réfléchissante
//! (m·s⁻¹), `θ` angle entre le faisceau et la vitesse de la cible (rad), `f'`
//! fréquence perçue (Hz), `Δf` décalage Doppler (Hz).
//!
//! **Convention** : unités SI ; « rapprochement positif » (une vitesse positive
//! augmente la fréquence perçue). L'angle de tir `θ` n'intervient que dans la
//! mesure **réfléchie**.
//! **Limite honnête** : formules **non relativistes**, valables pour des vitesses
//! **faibles devant la célérité** de l'onde (`v ≪ c`) ; la célérité `c` dépend du
//! milieu (air, eau, acier…) et est **fournie par l'appelant** — aucune valeur
//! « par défaut » n'est inventée. La cible réfléchissante suppose émetteur et
//! récepteur colocalisés (facteur 2, aller-retour).

use core::f64::consts::FRAC_PI_2;

/// Fréquence perçue lorsque la **source** se déplace vers l'observateur
/// `f' = f·c/(c - v_s)` (Hz).
///
/// Panique si `source_frequency < 0`, `wave_speed <= 0` ou
/// `source_velocity_toward >= wave_speed` (source non subsonique).
pub fn doppler_observed_moving_source(
    source_frequency: f64,
    wave_speed: f64,
    source_velocity_toward: f64,
) -> f64 {
    assert!(source_frequency >= 0.0, "f ≥ 0 requis");
    assert!(wave_speed > 0.0, "c > 0 requis");
    assert!(
        source_velocity_toward < wave_speed,
        "v_s < c requis (source subsonique)"
    );
    source_frequency * wave_speed / (wave_speed - source_velocity_toward)
}

/// Fréquence perçue lorsque l'**observateur** se déplace vers la source
/// `f' = f·(c + v_o)/c` (Hz).
///
/// Panique si `source_frequency < 0`, `wave_speed <= 0` ou
/// `observer_velocity_toward <= -wave_speed` (fréquence perçue négative).
pub fn doppler_observed_moving_observer(
    source_frequency: f64,
    wave_speed: f64,
    observer_velocity_toward: f64,
) -> f64 {
    assert!(source_frequency >= 0.0, "f ≥ 0 requis");
    assert!(wave_speed > 0.0, "c > 0 requis");
    assert!(
        observer_velocity_toward > -wave_speed,
        "v_o > -c requis (fréquence perçue positive)"
    );
    source_frequency * (wave_speed + observer_velocity_toward) / wave_speed
}

/// Décalage Doppler d'une **cible réfléchissante** (aller-retour, émetteur et
/// récepteur colocalisés) `Δf = 2·f·v·cos(θ)/c` (Hz).
///
/// Panique si `source_frequency < 0`, `wave_speed <= 0` ou
/// `beam_angle_rad ∉ [0, π/2[`.
pub fn doppler_shift_reflected(
    source_frequency: f64,
    wave_speed: f64,
    target_velocity: f64,
    beam_angle_rad: f64,
) -> f64 {
    assert!(source_frequency >= 0.0, "f ≥ 0 requis");
    assert!(wave_speed > 0.0, "c > 0 requis");
    assert!(
        (0.0..FRAC_PI_2).contains(&beam_angle_rad),
        "θ ∈ [0, π/2[ requis"
    );
    2.0 * source_frequency * target_velocity * beam_angle_rad.cos() / wave_speed
}

/// Vitesse de la cible déduite du décalage Doppler mesuré (réciproque)
/// `v = Δf·c/(2·f·cos(θ))` (m·s⁻¹).
///
/// Panique si `source_frequency <= 0`, `wave_speed <= 0` ou
/// `beam_angle_rad ∉ [0, π/2[`.
pub fn doppler_velocity_from_shift(
    frequency_shift: f64,
    source_frequency: f64,
    wave_speed: f64,
    beam_angle_rad: f64,
) -> f64 {
    assert!(source_frequency > 0.0, "f > 0 requis");
    assert!(wave_speed > 0.0, "c > 0 requis");
    assert!(
        (0.0..FRAC_PI_2).contains(&beam_angle_rad),
        "θ ∈ [0, π/2[ requis"
    );
    frequency_shift * wave_speed / (2.0 * source_frequency * beam_angle_rad.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_3;

    #[test]
    fn moving_source_realistic_case() {
        // Source sonore : f=300 Hz, c=340 m·s⁻¹ (air), v_s=40 m·s⁻¹.
        // f' = 300·340/(340-40) = 102000/300 = 340 Hz.
        let f_obs = doppler_observed_moving_source(300.0, 340.0, 40.0);
        assert_relative_eq!(f_obs, 340.0, max_relative = 1e-12);
    }

    #[test]
    fn moving_observer_realistic_case() {
        // Observateur : f=1000 Hz, c=340 m·s⁻¹, v_o=34 m·s⁻¹.
        // f' = 1000·(340+34)/340 = 374000/340 = 1100 Hz.
        let f_obs = doppler_observed_moving_observer(1000.0, 340.0, 34.0);
        assert_relative_eq!(f_obs, 1100.0, max_relative = 1e-12);
    }

    #[test]
    fn stationary_gives_source_frequency() {
        // Cas limite : vitesses nulles → aucun décalage, f' = f.
        let f = 500.0_f64;
        assert_relative_eq!(
            doppler_observed_moving_source(f, 340.0, 0.0),
            f,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            doppler_observed_moving_observer(f, 340.0, 0.0),
            f,
            max_relative = 1e-15
        );
    }

    #[test]
    fn reflected_shift_realistic_case() {
        // Débitmétrie ultrasonore : f=1 MHz, c=1500 m·s⁻¹ (eau), v=3 m·s⁻¹, θ=60°.
        // Δf = 2·1e6·3·cos(π/3)/1500 = 2·1e6·3·0.5/1500 = 3e6/1500 = 2000 Hz.
        let df = doppler_shift_reflected(1.0e6, 1500.0, 3.0, FRAC_PI_3);
        assert_relative_eq!(df, 2000.0, max_relative = 1e-9);
    }

    #[test]
    fn shift_and_velocity_are_reciprocal() {
        // Réciprocité : v → Δf → v redonne la vitesse de départ.
        let (f, c, v, theta) = (1.0e6_f64, 1500.0_f64, 3.0_f64, FRAC_PI_3);
        let df = doppler_shift_reflected(f, c, v, theta);
        let v_back = doppler_velocity_from_shift(df, f, c, theta);
        assert_relative_eq!(v_back, v, max_relative = 1e-9);
    }

    #[test]
    fn reflected_shift_proportional_to_velocity() {
        // Δf ∝ v : doubler la vitesse de la cible double le décalage.
        let df1 = doppler_shift_reflected(1.0e6, 1500.0, 2.0, FRAC_PI_3);
        let df2 = doppler_shift_reflected(1.0e6, 1500.0, 4.0, FRAC_PI_3);
        assert_relative_eq!(df2 / df1, 2.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "v_s < c")]
    fn supersonic_source_panics() {
        doppler_observed_moving_source(300.0, 340.0, 340.0);
    }
}
