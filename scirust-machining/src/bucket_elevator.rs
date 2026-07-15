//! **Élévateur à godets** — débit massique transporté, puissance de levage
//! utile et puissance moteur d'entraînement.
//!
//! ```text
//! nombre de godets/m   n = 1/p                          (godets par mètre)
//! débit massique       q_m = V_g·φ·(n·v)·ρ              (kg/s)
//! puissance de levage  P_l = q_m·g·H                    (W)
//! puissance moteur     P_m = P_l/η                      (W)
//! ```
//!
//! `p` pas des godets sur le brin montant (m), `n` nombre de godets par mètre
//! de brin (1/m), `V_g` volume utile d'un godet (m³), `φ` coefficient de
//! remplissage (sans dimension, 0 < φ ≤ 1), `v` vitesse du brin (m/s), `ρ`
//! masse volumique apparente du produit (kg/m³), `q_m` débit massique (kg/s),
//! `g` accélération de la pesanteur (m/s²), `H` hauteur de levage (m), `P_l`
//! puissance de levage utile (W), `η` rendement mécanique global (sans
//! dimension, 0 < η ≤ 1), `P_m` puissance moteur (W).
//!
//! **Convention** : SI ; unités cohérentes exigées. **Limite honnête** :
//! remplissage **régulier** — le coefficient de remplissage `φ`, la masse
//! volumique apparente `ρ` et le rendement mécanique `η` sont **fournis par
//! l'appelant** (aucune valeur « par défaut » n'est inventée). La valeur de `g`
//! est également fournie. Le modèle néglige les pertes de reprise au pied et de
//! décharge en tête ainsi que l'accélération du produit au chargement. Distinct
//! de la puissance d'élévation d'une bande de [`crate::belt_conveyor`].

/// Nombre de godets par mètre de brin `n = 1/p` à partir du pas `p`.
///
/// Panique si `bucket_pitch <= 0`.
pub fn bucket_spacing_from_pitch(bucket_pitch: f64) -> f64 {
    assert!(bucket_pitch > 0.0, "le pas des godets p > 0 est requis");
    1.0 / bucket_pitch
}

/// Débit massique transporté `q_m = V_g·φ·(n·v)·ρ` (kg/s).
///
/// Panique si `bucket_volume < 0`, si `fill_factor` n'est pas dans `]0, 1]`, si
/// `buckets_per_meter < 0`, si `belt_velocity < 0` ou si `bulk_density < 0`.
pub fn bucket_capacity(
    bucket_volume: f64,
    fill_factor: f64,
    buckets_per_meter: f64,
    belt_velocity: f64,
    bulk_density: f64,
) -> f64 {
    assert!(bucket_volume >= 0.0, "le volume utile V_g ≥ 0 est requis");
    assert!(
        fill_factor > 0.0 && fill_factor <= 1.0,
        "le coefficient de remplissage φ doit vérifier 0 < φ ≤ 1"
    );
    assert!(
        buckets_per_meter >= 0.0,
        "le nombre de godets par mètre n ≥ 0 est requis"
    );
    assert!(belt_velocity >= 0.0, "la vitesse du brin v ≥ 0 est requise");
    assert!(
        bulk_density >= 0.0,
        "la masse volumique apparente ρ ≥ 0 est requise"
    );
    bucket_volume * fill_factor * (buckets_per_meter * belt_velocity) * bulk_density
}

/// Puissance de levage utile `P_l = q_m·g·H` (W), part de puissance dédiée à
/// monter le produit de la hauteur `H`.
///
/// Panique si `mass_flow < 0`, `lift_height < 0` ou `gravity < 0`.
pub fn bucket_lifting_power(mass_flow: f64, lift_height: f64, gravity: f64) -> f64 {
    assert!(
        mass_flow >= 0.0 && lift_height >= 0.0 && gravity >= 0.0,
        "q_m ≥ 0, H ≥ 0 et g ≥ 0 requis"
    );
    mass_flow * gravity * lift_height
}

/// Puissance moteur `P_m = P_l/η` (W), puissance à fournir à l'arbre compte
/// tenu du rendement mécanique global `η`.
///
/// Panique si `lifting_power < 0` ou si `efficiency` n'est pas dans `]0, 1]`.
pub fn bucket_motor_power(lifting_power: f64, efficiency: f64) -> f64 {
    assert!(
        lifting_power >= 0.0,
        "la puissance de levage P_l ≥ 0 est requise"
    );
    assert!(
        efficiency > 0.0 && efficiency <= 1.0,
        "le rendement mécanique η doit vérifier 0 < η ≤ 1"
    );
    lifting_power / efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spacing_is_reciprocal_of_pitch() {
        // n = 1/p : réciprocité pas ↔ densité de godets.
        assert_relative_eq!(bucket_spacing_from_pitch(0.2), 5.0, epsilon = 1e-12);
        // Application deux fois : n puis 1/n redonne le pas de départ.
        let p = 0.25_f64;
        let n = bucket_spacing_from_pitch(p);
        assert_relative_eq!(bucket_spacing_from_pitch(n), p, epsilon = 1e-12);
    }

    #[test]
    fn capacity_is_linear_in_velocity_and_fill() {
        // q_m ∝ v à remplissage fixe : doubler v double le débit.
        let q1 = bucket_capacity(0.002, 0.75, 5.0, 1.5, 800.0);
        let q2 = bucket_capacity(0.002, 0.75, 5.0, 3.0, 800.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-12);
        // q_m ∝ φ : réduire de moitié le remplissage réduit de moitié le débit.
        let q3 = bucket_capacity(0.002, 0.375, 5.0, 1.5, 800.0);
        assert_relative_eq!(q3, 0.5 * q1, epsilon = 1e-12);
    }

    #[test]
    fn capacity_numeric_case() {
        // V_g=0,002 m³, φ=0,75, n=5 /m, v=1,5 m/s, ρ=800 kg/m³.
        // q_m = 0,002·0,75·(5·1,5)·800 = 0,0015·7,5·800 = 9,0 kg/s.
        assert_relative_eq!(
            bucket_capacity(0.002, 0.75, 5.0, 1.5, 800.0),
            9.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn lifting_power_equals_potential_energy_rate() {
        // P_l = q_m·g·H : taux d'énergie potentielle acquise.
        // q_m=9 kg/s, g=9,81, H=20 m → P_l = 9·9,81·20 = 1765,8 W.
        assert_relative_eq!(
            bucket_lifting_power(9.0, 20.0, 9.81),
            1765.8,
            epsilon = 1e-9
        );
        // Hauteur nulle : aucune puissance de levage requise.
        assert_relative_eq!(bucket_lifting_power(9.0, 0.0, 9.81), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn motor_power_reciprocates_efficiency() {
        // P_m = P_l/η : à η=1 le moteur ne fournit que la puissance utile.
        assert_relative_eq!(bucket_motor_power(1765.8, 1.0), 1765.8, epsilon = 1e-9);
        // P_l = P_m·η : cohérence de la relation inverse (η=0,8).
        let pm = bucket_motor_power(1765.8, 0.8);
        assert_relative_eq!(pm * 0.8, 1765.8, epsilon = 1e-9);
        // Cas chiffré : 1765,8/0,8 = 2207,25 W.
        assert_relative_eq!(pm, 2207.25, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "0 < φ ≤ 1")]
    fn out_of_range_fill_factor_panics() {
        bucket_capacity(0.002, 1.2, 5.0, 1.5, 800.0);
    }
}
