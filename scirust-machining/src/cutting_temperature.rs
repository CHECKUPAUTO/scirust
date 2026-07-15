//! **Température de coupe en usinage** — modèle empirique d'élévation de
//! température du copeau, température absolue à l'interface et indice empirique
//! de Cook en fonction des conditions de coupe.
//!
//! ```text
//! échauffement copeau   ΔT = u/(ρ·c)              (élévation adiabatique, u énergie spécifique J/m³)
//! température de coupe   T = T0 + f·ΔT             (T0 ambiante, f fraction de chaleur vers l'outil/pièce)
//! indice de Cook        θ = (Vc^a)·(fz^b)         (forme empirique puissance, exposants fournis)
//! ```
//!
//! `u` énergie spécifique de coupe (J·m⁻³, ≡ effort spécifique Kienzle en Pa),
//! `ρ` masse volumique du matériau usiné (kg·m⁻³), `c` chaleur massique
//! (J·kg⁻¹·K⁻¹), `ΔT` élévation de température (K), `T0` température ambiante
//! (K ou °C, cohérente avec la sortie), `f` fraction de partage de chaleur
//! (sans dimension, 0 ≤ f ≤ 1), `T` température de coupe (même unité que `T0`),
//! `Vc` vitesse de coupe (m·s⁻¹ ou m·min⁻¹), `fz` avance (m ou mm), `a`, `b`
//! exposants empiriques (sans dimension), `θ` indice empirique de Cook (unité
//! dépendante des exposants et des unités d'entrée).
//!
//! **Convention** : SI. **Limite honnête** : modèle **simplifié** de type
//! Loewen-Shaw (échauffement adiabatique du copeau) et Cook (loi puissance
//! empirique). L'énergie spécifique de coupe `u`, les propriétés matériau
//! (`ρ`, `c`), la fraction de partage de chaleur `f` et les exposants empiriques
//! sont **fournis par l'appelant** ; aucune valeur « par défaut » matériau ou
//! procédé n'est inventée. La conduction détaillée, la variation des propriétés
//! avec la température, la géométrie d'outil et le refroidissement ne sont pas
//! modélisés. Voir [`crate::brake_thermal`] (échauffement adiabatique analogue).

/// Élévation de température adiabatique du copeau `ΔT = u/(ρ·c)`.
///
/// `specific_cutting_energy` est l'énergie spécifique de coupe `u` (J·m⁻³),
/// `density` la masse volumique `ρ` (kg·m⁻³), `specific_heat` la chaleur
/// massique `c` (J·kg⁻¹·K⁻¹). Résultat en K.
///
/// Panique si `specific_cutting_energy < 0`, `density <= 0` ou
/// `specific_heat <= 0`.
pub fn cutting_temp_shear_zone_temperature_rise(
    specific_cutting_energy: f64,
    density: f64,
    specific_heat: f64,
) -> f64 {
    assert!(
        specific_cutting_energy >= 0.0,
        "l'énergie spécifique de coupe u doit être positive"
    );
    assert!(
        density > 0.0 && specific_heat > 0.0,
        "ρ > 0 et c > 0 requis (dénominateur non nul)"
    );
    specific_cutting_energy / (density * specific_heat)
}

/// Température de coupe à l'interface `T = T0 + f·ΔT`.
///
/// `ambient` est la température ambiante `T0`, `temp_rise` l'élévation `ΔT` (K),
/// `partition_fraction` la fraction de chaleur `f` transmise (sans dimension,
/// 0 ≤ f ≤ 1). La sortie est dans la même unité que `ambient`.
///
/// Panique si `temp_rise < 0` ou si `partition_fraction` n'est pas dans [0, 1].
pub fn cutting_temp_cutting_temperature(
    ambient: f64,
    temp_rise: f64,
    partition_fraction: f64,
) -> f64 {
    assert!(
        temp_rise >= 0.0,
        "l'élévation de température ΔT doit être positive"
    );
    assert!(
        (0.0..=1.0).contains(&partition_fraction),
        "la fraction de partage f doit être dans [0, 1]"
    );
    ambient + partition_fraction * temp_rise
}

/// Indice empirique de Cook `θ = (Vc^a)·(fz^b)` (loi puissance).
///
/// `cutting_speed` est la vitesse de coupe `Vc`, `feed` l'avance `fz`,
/// `speed_exponent` l'exposant `a` et `feed_exponent` l'exposant `b`. Les unités
/// de `Vc` et `fz` ainsi que les exposants sont **fournis par l'appelant** ;
/// l'unité de `θ` en dépend.
///
/// Panique si `cutting_speed <= 0` ou `feed <= 0` (base de puissance non
/// définie pour un exposant réel quelconque).
pub fn cutting_temp_cook_temperature_index(
    cutting_speed: f64,
    feed: f64,
    speed_exponent: f64,
    feed_exponent: f64,
) -> f64 {
    assert!(
        cutting_speed > 0.0 && feed > 0.0,
        "Vc > 0 et fz > 0 requis (base de puissance strictement positive)"
    );
    cutting_speed.powf(speed_exponent) * feed.powf(feed_exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn temperature_rise_realistic_case() {
        // Acier : u = 2,0 GJ/m³ (≈ 2000 MPa d'effort spécifique), ρ = 7850 kg/m³,
        // c = 470 J/kg/K → ΔT = 2e9/(7850·470) ≈ 542,08 K.
        let dt = cutting_temp_shear_zone_temperature_rise(2.0e9, 7850.0, 470.0);
        assert_relative_eq!(dt, 2.0e9 / (7850.0 * 470.0), epsilon = 1e-6);
        assert_relative_eq!(dt, 542.078_872_475_945_2, epsilon = 1e-3);
    }

    #[test]
    fn temperature_rise_scales_linearly_with_energy() {
        // ΔT ∝ u : doubler l'énergie spécifique double l'échauffement.
        let dt1 = cutting_temp_shear_zone_temperature_rise(1.0e9, 7850.0, 470.0);
        let dt2 = cutting_temp_shear_zone_temperature_rise(2.0e9, 7850.0, 470.0);
        assert_relative_eq!(dt2, 2.0 * dt1, epsilon = 1e-6);
    }

    #[test]
    fn temperature_rise_inverse_of_thermal_capacity() {
        // ΔT ∝ 1/(ρ·c) : doubler ρ·c halve l'échauffement.
        let dt1 = cutting_temp_shear_zone_temperature_rise(2.0e9, 7850.0, 470.0);
        let dt2 = cutting_temp_shear_zone_temperature_rise(2.0e9, 15700.0, 470.0);
        assert_relative_eq!(dt2, 0.5 * dt1, epsilon = 1e-6);
    }

    #[test]
    fn cutting_temperature_partition_bounds() {
        // f = 0 → aucune chaleur transmise, T = T0 ; f = 1 → T = T0 + ΔT.
        let t0 = 20.0;
        let dt = 500.0;
        assert_relative_eq!(
            cutting_temp_cutting_temperature(t0, dt, 0.0),
            t0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            cutting_temp_cutting_temperature(t0, dt, 1.0),
            t0 + dt,
            epsilon = 1e-12
        );
    }

    #[test]
    fn cutting_temperature_realistic_case() {
        // T0 = 20 °C, ΔT ≈ 542,08 K, f = 0,80 → T = 20 + 0,8·ΔT ≈ 453,66 °C.
        let dt = cutting_temp_shear_zone_temperature_rise(2.0e9, 7850.0, 470.0);
        let t = cutting_temp_cutting_temperature(20.0, dt, 0.80);
        assert_relative_eq!(t, 20.0 + 0.80 * dt, epsilon = 1e-9);
        assert_relative_eq!(t, 453.663_097_980_756_2, epsilon = 1e-3);
    }

    #[test]
    fn cook_index_power_law_identity() {
        // θ(Vc, fz) = Vc^a · fz^b. Avec a = 0,5, b = 0,25, Vc = 100, fz = 0,2 :
        // θ = 100^0,5 · 0,2^0,25 = 10 · 0,668740... ≈ 6,687403.
        let theta = cutting_temp_cook_temperature_index(100.0, 0.2, 0.5, 0.25);
        assert_relative_eq!(
            theta,
            100.0_f64.powf(0.5) * 0.2_f64.powf(0.25),
            epsilon = 1e-9
        );
        // Multiplier Vc par 4 (a = 0,5) multiplie θ par 4^0,5 = 2.
        let theta2 = cutting_temp_cook_temperature_index(400.0, 0.2, 0.5, 0.25);
        assert_relative_eq!(theta2, 2.0 * theta, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ρ > 0 et c > 0 requis")]
    fn zero_density_panics() {
        cutting_temp_shear_zone_temperature_rise(2.0e9, 0.0, 470.0);
    }
}
