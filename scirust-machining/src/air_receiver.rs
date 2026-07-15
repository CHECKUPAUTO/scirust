//! **Réservoir d'air comprimé** — dimensionnement d'un ballon tampon par la loi
//! des gaz en détente **isotherme** : volume de réservoir, temps de remplissage
//! et volume d'air utile entre deux seuils de pression.
//!
//! ```text
//! volume réservoir   V = Q·t·p_atm / (p_max − p_min)
//! air utile          V_u = V·(p_max − p_min) / p_atm
//! temps de remplissage t = V·Δp / (p_atm·q)
//! ```
//!
//! `Q` débit d'air demandé (m³·s⁻¹, aux conditions atmosphériques / air libre),
//! `t` durée de cycle ou de remplissage (s), `p_atm` pression atmosphérique
//! **absolue** (Pa), `p_max`/`p_min` pressions **absolues** haute/basse du
//! réservoir (Pa), `Δp` variation de pression absolue du réservoir (Pa),
//! `V` volume géométrique du réservoir (m³), `V_u` volume d'air utile ramené aux
//! conditions atmosphériques (m³), `q` débit du compresseur (m³·s⁻¹, air libre).
//!
//! **Convention** : pressions **absolues**, unités SI ; débits exprimés en air
//! libre (aux conditions atmosphériques).
//!
//! **Limite honnête** : détente supposée **isotherme** et air = **gaz parfait** ;
//! aucune perte de charge, aucun échauffement de compression ni condensation pris
//! en compte. Les pressions de consigne, le débit demandé, la durée de cycle et
//! le débit du compresseur sont des **données de procédé fournies par
//! l'appelant** — aucune valeur « par défaut » n'est inventée ici. Complète
//! [`crate::compressed_air`].

/// Volume géométrique d'un réservoir tampon (détente isotherme)
/// `V = Q·t·p_atm / (p_max − p_min)`.
///
/// `air_demand` débit demandé (m³·s⁻¹, air libre), `cycle_time` durée du cycle
/// (s), pressions **absolues** (Pa) ; renvoie le volume en m³.
///
/// Panique si un paramètre est `<= 0` ou si `pressure_max <= pressure_min`.
pub fn receiver_volume(
    air_demand: f64,
    cycle_time: f64,
    pressure_max: f64,
    pressure_min: f64,
    atmospheric_pressure: f64,
) -> f64 {
    assert!(
        air_demand > 0.0 && cycle_time > 0.0 && atmospheric_pressure > 0.0,
        "débit, durée de cycle et pression atmosphérique strictement positifs requis"
    );
    assert!(
        pressure_max > pressure_min && pressure_min > 0.0,
        "pressions absolues avec pressure_max > pressure_min > 0 requises"
    );
    air_demand * cycle_time * atmospheric_pressure / (pressure_max - pressure_min)
}

/// Volume d'air **utile** stocké entre `p_max` et `p_min`, ramené aux conditions
/// atmosphériques (détente isotherme) `V_u = V·(p_max − p_min) / p_atm`.
///
/// `volume` volume géométrique du réservoir (m³), pressions **absolues** (Pa) ;
/// renvoie le volume d'air libre en m³. Réciproque de [`receiver_volume`] à débit
/// et durée fixés (`V_u = Q·t`).
///
/// Panique si un paramètre est `<= 0` ou si `pressure_max <= pressure_min`.
pub fn receiver_usable_air(
    volume: f64,
    pressure_max: f64,
    pressure_min: f64,
    atmospheric_pressure: f64,
) -> f64 {
    assert!(
        volume > 0.0 && atmospheric_pressure > 0.0,
        "volume et pression atmosphérique strictement positifs requis"
    );
    assert!(
        pressure_max > pressure_min && pressure_min > 0.0,
        "pressions absolues avec pressure_max > pressure_min > 0 requises"
    );
    volume * (pressure_max - pressure_min) / atmospheric_pressure
}

/// Temps de remplissage d'un réservoir par un compresseur (détente isotherme)
/// `t = V·Δp / (p_atm·q)`.
///
/// `volume` volume géométrique du réservoir (m³), `pressure_change` variation de
/// pression **absolue** à obtenir (Pa), `atmospheric_pressure` pression
/// atmosphérique **absolue** (Pa), `compressor_flow` débit du compresseur
/// (m³·s⁻¹, air libre) ; renvoie la durée en s.
///
/// Panique si un paramètre est `<= 0`.
pub fn receiver_pump_up_time(
    volume: f64,
    pressure_change: f64,
    atmospheric_pressure: f64,
    compressor_flow: f64,
) -> f64 {
    assert!(
        volume > 0.0
            && pressure_change > 0.0
            && atmospheric_pressure > 0.0
            && compressor_flow > 0.0,
        "volume, variation de pression, pression atmosphérique et débit strictement positifs requis"
    );
    volume * pressure_change / (atmospheric_pressure * compressor_flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn volume_and_usable_air_are_reciprocal() {
        // V dimensionné pour Q·t doit restituer exactement V_u = Q·t.
        let (q, t, p_max, p_min, p_atm) = (0.02_f64, 30.0, 8e5, 6e5, 1e5);
        let v = receiver_volume(q, t, p_max, p_min, p_atm);
        let usable = receiver_usable_air(v, p_max, p_min, p_atm);
        assert_relative_eq!(usable, q * t, epsilon = 1e-9);
    }

    #[test]
    fn volume_scales_inversely_with_pressure_band() {
        // Doubler la bande (p_max − p_min) réduit de moitié le volume requis.
        let v1 = receiver_volume(0.01, 20.0, 7e5, 6e5, 1e5);
        let v2 = receiver_volume(0.01, 20.0, 8e5, 6e5, 1e5);
        assert_relative_eq!(v1, 2.0 * v2, epsilon = 1e-9);
    }

    #[test]
    fn usable_air_proportional_to_volume() {
        let base = receiver_usable_air(0.5, 8e5, 6e5, 1e5);
        let triple = receiver_usable_air(1.5, 8e5, 6e5, 1e5);
        assert_relative_eq!(triple, 3.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn realistic_receiver_sizing() {
        // Demande 20 L/s pendant 30 s, bande 8→6 bar abs, p_atm 1 bar :
        // V = 0,02·30·1e5 / 2e5 = 0,3 m³ (300 L).
        let v = receiver_volume(0.02, 30.0, 8e5, 6e5, 1e5);
        assert_relative_eq!(v, 0.3, epsilon = 1e-9);
    }

    #[test]
    fn pump_up_time_matches_hand_calc() {
        // V = 0,3 m³, Δp = 2e5 Pa, p_atm = 1e5 Pa, q = 0,02 m³/s :
        // t = 0,3·2e5 / (1e5·0,02) = 30 s.
        let t = receiver_pump_up_time(0.3, 2e5, 1e5, 0.02);
        assert_relative_eq!(t, 30.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "pressure_max > pressure_min")]
    fn volume_rejects_inverted_pressures() {
        let _ = receiver_volume(0.01, 10.0, 6e5, 8e5, 1e5);
    }
}
