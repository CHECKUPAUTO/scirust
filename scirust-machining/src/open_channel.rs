//! Écoulement à surface libre (canal ouvert) — rayon hydraulique et formules de
//! **Manning** et de **Chézy** pour la vitesse et le débit en régime uniforme.
//!
//! ```text
//! rayon hydraulique  Rh = A/P
//! Manning            V = (1/n)·Rh^{2/3}·√S        Q = A·V
//! Chézy              V = C·√(Rh·S)
//! ```
//!
//! `A` section mouillée (m²), `P` périmètre mouillé (m), `Rh` rayon hydraulique
//! (m), `n` coefficient de Manning (rugosité, s·m^{−1/3} ; ~0,013 béton lisse,
//! ~0,03 terre), `S` pente du fond (m/m), `C` coefficient de Chézy (m^{1/2}/s).
//!
//! **Convention** : SI (formule de Manning en unités **métriques**). **Limite
//! honnête** : régime **uniforme et permanent** (pente motrice = pente du fond) ;
//! `n` et `C` sont des données empiriques fournies par l'appelant. Pas de
//! ressaut, de courbe de remous, ni d'écoulement non établi.

/// Rayon hydraulique `Rh = A/P` (m).
///
/// Panique si `wetted_perimeter <= 0`.
pub fn hydraulic_radius(area: f64, wetted_perimeter: f64) -> f64 {
    assert!(
        wetted_perimeter > 0.0,
        "le périmètre mouillé doit être strictement positif"
    );
    area / wetted_perimeter
}

/// Vitesse de Manning `V = (1/n)·Rh^{2/3}·√S` (m/s).
///
/// Panique si `n <= 0` ou `slope < 0`.
pub fn manning_velocity(manning_n: f64, hydraulic_radius: f64, slope: f64) -> f64 {
    assert!(manning_n > 0.0 && slope >= 0.0, "n > 0 et S ≥ 0 requis");
    (1.0 / manning_n) * hydraulic_radius.powf(2.0 / 3.0) * slope.sqrt()
}

/// Débit de Manning `Q = A·V` (m³/s).
pub fn manning_flow(manning_n: f64, area: f64, hydraulic_radius: f64, slope: f64) -> f64 {
    area * manning_velocity(manning_n, hydraulic_radius, slope)
}

/// Vitesse de Chézy `V = C·√(Rh·S)` (m/s).
///
/// Panique si `Rh·S < 0`.
pub fn chezy_velocity(chezy_c: f64, hydraulic_radius: f64, slope: f64) -> f64 {
    assert!(hydraulic_radius * slope >= 0.0, "Rh·S doit être positif");
    chezy_c * (hydraulic_radius * slope).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_radius_of_a_rectangular_channel() {
        // Canal 2 m large, 1 m d'eau : A=2, P=2+2·1=4 → Rh=0,5.
        assert_relative_eq!(hydraulic_radius(2.0, 4.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn manning_velocity_and_flow() {
        // n=0,013, Rh=0,5, S=0,001 → V = (1/0,013)·0,5^(2/3)·√0,001.
        let v = manning_velocity(0.013, 0.5, 0.001);
        assert_relative_eq!(
            v,
            (1.0 / 0.013) * 0.5f64.powf(2.0 / 3.0) * 0.001f64.sqrt(),
            epsilon = 1e-9
        );
        // Q = A·V.
        assert_relative_eq!(
            manning_flow(0.013, 2.0, 0.5, 0.001),
            2.0 * v,
            epsilon = 1e-9
        );
    }

    #[test]
    fn steeper_slope_flows_faster() {
        // V ∝ √S : quadrupler la pente double la vitesse.
        let v1 = manning_velocity(0.013, 0.5, 0.001);
        let v2 = manning_velocity(0.013, 0.5, 0.004);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn chezy_matches_root_rh_s() {
        // C=60, Rh=0,5, S=0,001 → V = 60·√(5e-4).
        assert_relative_eq!(
            chezy_velocity(60.0, 0.5, 0.001),
            60.0 * (0.5f64 * 0.001).sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "périmètre mouillé")]
    fn zero_perimeter_panics() {
        hydraulic_radius(2.0, 0.0);
    }
}
