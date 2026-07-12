//! Conduction **transitoire** — nombres de **Biot** et de **Fourier**, et
//! méthode de la **capacité thermique localisée** (corps thermiquement mince).
//!
//! ```text
//! Biot           Bi = h·Lc/k
//! Fourier        Fo = α·t/Lc²
//! capacité localisée  T(t) = T∞ + (T0 − T∞)·exp(−h·A·t/(ρ·V·c))
//! constante de temps  τ = ρ·V·c/(h·A)
//! ```
//!
//! `h` coefficient de convection (W/(m²·K)), `Lc` longueur caractéristique
//! (`V/A`, m), `k` conductivité (W/(m·K)), `α` diffusivité thermique (m²/s), `t`
//! temps (s), `ρ` masse volumique, `V` volume, `c` chaleur massique, `A` surface
//! d'échange, `T∞` température du fluide, `T0` température initiale.
//!
//! **Convention** : SI cohérent, `T` en K (ou °C, cohérent). **Limite honnête** :
//! la capacité localisée n'est valable que pour un corps **thermiquement mince**
//! (`Bi < 0,1`, gradient interne négligeable) — vérifier le Biot avant de
//! l'appliquer ; au-delà, il faut une solution de conduction avec gradient interne.

/// Nombre de Biot `Bi = h·Lc/k`.
///
/// Panique si `k <= 0`.
pub fn biot_number(h: f64, characteristic_length: f64, k: f64) -> f64 {
    assert!(k > 0.0, "la conductivité doit être strictement positive");
    h * characteristic_length / k
}

/// Nombre de Fourier `Fo = α·t/Lc²`.
///
/// Panique si `characteristic_length <= 0`.
pub fn fourier_number(alpha: f64, time: f64, characteristic_length: f64) -> f64 {
    assert!(
        characteristic_length > 0.0,
        "la longueur caractéristique doit être strictement positive"
    );
    alpha * time / (characteristic_length * characteristic_length)
}

/// Constante de temps `τ = ρ·V·c/(h·A)` (s).
///
/// Panique si `h·A <= 0`.
pub fn time_constant(rho: f64, volume: f64, c: f64, h: f64, area: f64) -> f64 {
    assert!(h * area > 0.0, "h·A doit être strictement positif");
    rho * volume * c / (h * area)
}

/// Température par capacité localisée `T(t) = T∞ + (T0 − T∞)·exp(−t/τ)`.
///
/// Panique si `time_constant <= 0`.
pub fn lumped_temperature(
    fluid_temp: f64,
    initial_temp: f64,
    time: f64,
    time_constant: f64,
) -> f64 {
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive"
    );
    fluid_temp + (initial_temp - fluid_temp) * (-time / time_constant).exp()
}

/// Vrai si l'hypothèse de capacité localisée est valable `Bi < 0,1`.
pub fn lumped_capacitance_valid(biot_number: f64) -> bool {
    biot_number < 0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn biot_gauges_the_lumped_assumption() {
        // h=20, Lc=0,005, k=50 → Bi = 0,002 < 0,1 (corps mince).
        let bi = biot_number(20.0, 0.005, 50.0);
        assert_relative_eq!(bi, 0.002, epsilon = 1e-9);
        assert!(lumped_capacitance_valid(bi));
        assert!(!lumped_capacitance_valid(0.5));
    }

    #[test]
    fn temperature_starts_at_initial_and_relaxes_to_fluid() {
        // t=0 → T0 ; t≫τ → T∞.
        let tau = time_constant(8000.0, 1e-6, 450.0, 20.0, 6e-4);
        assert_relative_eq!(
            lumped_temperature(20.0, 200.0, 0.0, tau),
            200.0,
            epsilon = 1e-9
        );
        let late = lumped_temperature(20.0, 200.0, 20.0 * tau, tau);
        assert!((late - 20.0).abs() < 1.0);
    }

    #[test]
    fn one_time_constant_reaches_63_percent() {
        // À t=τ, l'écart initial a chuté de 1−1/e ≈ 63,2 %.
        let tau = 100.0;
        let t = lumped_temperature(20.0, 120.0, tau, tau);
        // reste (1/e) de l'écart de 100 → T = 20 + 100/e ≈ 56,79.
        assert_relative_eq!(t, 20.0 + 100.0 / core::f64::consts::E, epsilon = 1e-9);
    }

    #[test]
    fn fourier_grows_with_time() {
        assert!(fourier_number(1e-5, 100.0, 0.01) > fourier_number(1e-5, 50.0, 0.01));
    }

    #[test]
    #[should_panic(expected = "conductivité")]
    fn zero_conductivity_biot_panics() {
        biot_number(20.0, 0.005, 0.0);
    }
}
