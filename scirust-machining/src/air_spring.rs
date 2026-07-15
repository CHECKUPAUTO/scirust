//! **Ressort pneumatique** (soufflet / coussin d'air) — raideur autour du point
//! d'équilibre, effort porteur et fréquence propre d'un système masse-ressort à
//! air, sous compression **polytropique**.
//!
//! ```text
//! raideur          k = n·p·A² / V
//! effort porteur   F = p_gauge·A
//! fréquence propre f = (1 / 2π)·√(k / m)
//! ```
//!
//! `p` pression **absolue** dans le soufflet (Pa), `p_gauge` pression
//! **relative** (au-dessus de l'atmosphère, Pa), `A` aire **effective** du
//! soufflet (m²), `V` volume de gaz au point d'équilibre (m³), `n` indice
//! polytropique (sans dimension), `k` raideur (N·m⁻¹), `F` effort porteur (N),
//! `m` masse suspendue (kg), `f` fréquence propre (Hz).
//!
//! **Convention** : unités SI ; `p` en pression **absolue** pour la raideur,
//! `p_gauge` en pression **relative** pour l'effort porteur.
//!
//! **Limite honnête** : linéarisation valable uniquement pour de **petites
//! oscillations** autour de l'équilibre, aire effective supposée **constante**
//! (pas de variation avec la course). L'indice polytropique (≈ 1 isotherme,
//! ≈ 1,4 adiabatique pour l'air), la pression, l'aire effective, le volume et la
//! masse sont des **données de procédé / matériau fournies par l'appelant** —
//! aucune valeur « par défaut » n'est inventée ici. Complète
//! [`crate::air_receiver`] et [`crate::vibration_isolation`].

use core::f64::consts::PI;

/// Raideur d'un ressort pneumatique autour du point d'équilibre (compression
/// polytropique) `k = n·p·A² / V`.
///
/// `absolute_pressure` pression **absolue** du gaz (Pa), `effective_area` aire
/// effective du soufflet (m²), `volume` volume de gaz à l'équilibre (m³),
/// `polytropic_index` indice polytropique (≈ 1 isotherme, ≈ 1,4 adiabatique) ;
/// renvoie la raideur en N·m⁻¹.
///
/// Panique si un paramètre est `<= 0`.
pub fn airspring_rate(
    absolute_pressure: f64,
    effective_area: f64,
    volume: f64,
    polytropic_index: f64,
) -> f64 {
    assert!(
        absolute_pressure > 0.0 && effective_area > 0.0 && volume > 0.0 && polytropic_index > 0.0,
        "pression absolue, aire effective, volume et indice polytropique strictement positifs requis"
    );
    polytropic_index * absolute_pressure * effective_area * effective_area / volume
}

/// Effort porteur statique d'un ressort pneumatique `F = p_gauge·A`.
///
/// `gauge_pressure` pression **relative** (au-dessus de l'atmosphère, Pa),
/// `effective_area` aire effective du soufflet (m²) ; renvoie l'effort en N.
///
/// Panique si un paramètre est `<= 0`.
pub fn airspring_force(gauge_pressure: f64, effective_area: f64) -> f64 {
    assert!(
        gauge_pressure > 0.0 && effective_area > 0.0,
        "pression relative et aire effective strictement positives requises"
    );
    gauge_pressure * effective_area
}

/// Fréquence propre d'un système masse-ressort pneumatique (petites oscillations)
/// `f = (1 / 2π)·√(k / m)`.
///
/// `rate` raideur du ressort (N·m⁻¹, p. ex. issue de [`airspring_rate`]),
/// `mass` masse suspendue (kg) ; renvoie la fréquence propre en Hz.
///
/// Panique si un paramètre est `<= 0`.
pub fn airspring_natural_frequency(rate: f64, mass: f64) -> f64 {
    assert!(
        rate > 0.0 && mass > 0.0,
        "raideur et masse strictement positives requises"
    );
    (1.0 / (2.0 * PI)) * (rate / mass).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_proportional_to_polytropic_index() {
        // k ∝ n : passer d'isotherme (1,0) à adiabatique (1,4) multiplie k par 1,4.
        let iso = airspring_rate(5e5, 0.01, 2e-3, 1.0);
        let adia = airspring_rate(5e5, 0.01, 2e-3, 1.4);
        assert_relative_eq!(adia, 1.4 * iso, epsilon = 1e-6);
    }

    #[test]
    fn rate_scales_with_area_squared() {
        // k ∝ A² : doubler l'aire effective quadruple la raideur.
        let base = airspring_rate(5e5, 0.01, 2e-3, 1.4);
        let doubled = airspring_rate(5e5, 0.02, 2e-3, 1.4);
        assert_relative_eq!(doubled, 4.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn rate_inversely_proportional_to_volume() {
        // k ∝ 1/V : doubler le volume divise la raideur par deux.
        let small = airspring_rate(5e5, 0.01, 1e-3, 1.4);
        let large = airspring_rate(5e5, 0.01, 2e-3, 1.4);
        assert_relative_eq!(large, small / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn realistic_rate_hand_calc() {
        // n = 1,4 ; p = 5 bar abs ; A = 0,01 m² ; V = 2 L :
        // k = 1,4·5e5·(0,01)² / 2e-3 = 1,4·5e5·1e-4 / 2e-3 = 35 000 N/m.
        let k = airspring_rate(5e5, 0.01, 2e-3, 1.4);
        assert_relative_eq!(k, 35_000.0, epsilon = 1e-6);
    }

    #[test]
    fn force_matches_hand_calc_and_is_linear() {
        // p_gauge = 4 bar rel ; A = 0,01 m² : F = 4e5·0,01 = 4000 N.
        let f = airspring_force(4e5, 0.01);
        assert_relative_eq!(f, 4000.0, epsilon = 1e-9);
        // Effort linéaire en aire.
        let f2 = airspring_force(4e5, 0.02);
        assert_relative_eq!(f2, 2.0 * f, epsilon = 1e-9);
    }

    #[test]
    fn natural_frequency_unit_identity_and_case() {
        // Identité : si k = m·(2π)² alors f = 1 Hz exactement.
        let m = 5.0_f64;
        let k = m * (2.0 * PI) * (2.0 * PI);
        assert_relative_eq!(airspring_natural_frequency(k, m), 1.0, epsilon = 1e-12);
        // Cas chiffré : k = 35 000 N/m, m = 350 kg → k/m = 100,
        // f = 10 / (2π) ≈ 1,591549 Hz.
        let f = airspring_natural_frequency(35_000.0, 350.0);
        assert_relative_eq!(f, 10.0 / (2.0 * PI), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "strictement positifs requis")]
    fn rate_rejects_zero_volume() {
        let _ = airspring_rate(5e5, 0.01, 0.0, 1.4);
    }
}
