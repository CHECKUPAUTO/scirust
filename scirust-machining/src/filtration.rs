//! **Filtration sur gâteau** — débit et volume filtré à travers un gâteau
//! incompressible selon la loi de Darcy et l'équation de Ruth.
//!
//! ```text
//! flux de Darcy (vitesse superficielle)  q = ΔP / (μ·(R_c + R_m))
//! résistance du gâteau                    R_c = α·w
//! coefficient quadratique de Ruth         a = μ·α·c / (2·A²·ΔP)
//! équation de Ruth (pression constante)   t = a·V² + b·V
//! volume filtré (racine positive)         V = (-b + √(b² + 4·a·t)) / (2·a)
//! ```
//!
//! `q` flux volumique par unité d'aire, c.-à-d. vitesse superficielle (m/s),
//! `ΔP` perte de charge appliquée (Pa), `μ` viscosité dynamique du filtrat
//! (Pa·s), `R_c` résistance du gâteau (m⁻¹), `R_m` résistance du média filtrant
//! (m⁻¹), `α` résistance spécifique du gâteau (m/kg), `w` masse de solides
//! déposée par unité d'aire (kg/m²), `c` concentration en solides du milieu à
//! filtrer (kg/m³), `A` aire de filtration (m²), `V` volume de filtrat cumulé
//! (m³), `t` temps de filtration (s), `a` coefficient quadratique (s/m⁶),
//! `b` coefficient linéaire du média (s/m³).
//!
//! **Convention** : unités SI. **Limite honnête** : gâteau **incompressible**
//! (résistance spécifique `α` **constante**, indépendante de la pression),
//! écoulement **laminaire** (loi de Darcy), et à **pression constante** le volume
//! suit la forme quadratique `t = a·V² + b·V`. La viscosité `μ`, la résistance
//! spécifique `α`, la concentration solide `c`, ainsi que les résistances de
//! gâteau et de média sont **fournies par l'appelant** : elles dépendent du
//! couple fluide/particules et du procédé et ne sont jamais supposées ici.

/// Flux de Darcy `q = ΔP / (μ·(R_c + R_m))` (m/s) — débit surfacique instantané.
///
/// Vitesse superficielle du filtrat à travers l'ensemble gâteau + média, en
/// régime laminaire ; c'est le débit volumique divisé par l'aire de filtration.
///
/// Panique si `pressure_drop < 0`, `cake_resistance < 0`, `medium_resistance <= 0`
/// ou `dynamic_viscosity <= 0`.
pub fn filtration_darcy_flux(
    pressure_drop: f64,
    cake_resistance: f64,
    medium_resistance: f64,
    dynamic_viscosity: f64,
) -> f64 {
    assert!(pressure_drop >= 0.0, "la perte de charge ΔP doit être ≥ 0");
    assert!(
        cake_resistance >= 0.0,
        "la résistance du gâteau R_c doit être ≥ 0"
    );
    assert!(
        medium_resistance > 0.0,
        "la résistance du média R_m doit être > 0"
    );
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique μ doit être > 0"
    );
    pressure_drop / (dynamic_viscosity * (cake_resistance + medium_resistance))
}

/// Résistance du gâteau `R_c = α·w` (m⁻¹).
///
/// Pour un gâteau incompressible la résistance croît linéairement avec la masse
/// de solides déposée par unité d'aire.
///
/// Panique si `specific_cake_resistance <= 0` ou `solids_per_area < 0`.
pub fn filtration_cake_resistance(specific_cake_resistance: f64, solids_per_area: f64) -> f64 {
    assert!(
        specific_cake_resistance > 0.0,
        "la résistance spécifique α doit être > 0"
    );
    assert!(
        solids_per_area >= 0.0,
        "la masse surfacique de solides w doit être ≥ 0"
    );
    specific_cake_resistance * solids_per_area
}

/// Coefficient quadratique de Ruth `a = μ·α·c / (2·A²·ΔP)` (s/m⁶).
///
/// À pression constante, le temps de filtration suit `t = a·V² + b·V` ; `a` est
/// le terme quadratique porté par l'accumulation de gâteau.
///
/// Panique si `dynamic_viscosity <= 0`, `specific_resistance <= 0`,
/// `solids_concentration <= 0`, `area <= 0` ou `pressure_drop <= 0`.
pub fn filtration_time_constant_pressure(
    dynamic_viscosity: f64,
    specific_resistance: f64,
    solids_concentration: f64,
    area: f64,
    pressure_drop: f64,
) -> f64 {
    assert!(
        dynamic_viscosity > 0.0,
        "la viscosité dynamique μ doit être > 0"
    );
    assert!(
        specific_resistance > 0.0,
        "la résistance spécifique α doit être > 0"
    );
    assert!(
        solids_concentration > 0.0,
        "la concentration en solides c doit être > 0"
    );
    assert!(area > 0.0, "l'aire de filtration A doit être > 0");
    assert!(pressure_drop > 0.0, "la perte de charge ΔP doit être > 0");
    dynamic_viscosity * specific_resistance * solids_concentration
        / (2.0 * area * area * pressure_drop)
}

/// Volume filtré `V = (-b + √(b² + 4·a·t)) / (2·a)` (m³) — racine positive de
/// l'équation de Ruth `a·V² + b·V - t = 0`.
///
/// Inverse [`filtration_time_constant_pressure`] et le terme de média : donne le
/// volume de filtrat cumulé au bout du temps `t` à pression constante.
///
/// Panique si `time_constant <= 0`, `medium_term < 0` ou `time < 0`.
pub fn filtration_volume_from_time(time_constant: f64, medium_term: f64, time: f64) -> f64 {
    assert!(
        time_constant > 0.0,
        "le coefficient quadratique a doit être > 0"
    );
    assert!(
        medium_term >= 0.0,
        "le coefficient de média b doit être ≥ 0"
    );
    assert!(time >= 0.0, "le temps de filtration t doit être ≥ 0");
    let discriminant = medium_term * medium_term + 4.0 * time_constant * time;
    (-medium_term + discriminant.sqrt()) / (2.0 * time_constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn darcy_flux_realistic_case() {
        // ΔP = 1e5 Pa, R_c = 1e11 m⁻¹, R_m = 1e10 m⁻¹, μ = 1e-3 Pa·s.
        // q = 1e5 / (1e-3·(1e11 + 1e10)) = 1e5 / 1.1e8 = 9,0909e-4 m/s.
        let q = filtration_darcy_flux(1.0e5_f64, 1.0e11_f64, 1.0e10_f64, 1.0e-3_f64);
        assert_relative_eq!(q, 9.090_909_090_909e-4, epsilon = 1e-15);
    }

    #[test]
    fn darcy_flux_is_linear_in_pressure_drop() {
        // q ∝ ΔP à résistances et viscosité fixées : doubler ΔP double le flux.
        let q1 = filtration_darcy_flux(5.0e4_f64, 2.0e11_f64, 1.0e10_f64, 2.0e-3_f64);
        let q2 = filtration_darcy_flux(1.0e5_f64, 2.0e11_f64, 1.0e10_f64, 2.0e-3_f64);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-15);
    }

    #[test]
    fn cake_resistance_realistic_and_linear() {
        // R_c = α·w = 1e11 · 0,5 = 5e10 m⁻¹, et R_c ∝ w.
        let r1 = filtration_cake_resistance(1.0e11_f64, 0.5_f64);
        assert_relative_eq!(r1, 5.0e10_f64, epsilon = 1e-3);
        let r2 = filtration_cake_resistance(1.0e11_f64, 1.0_f64);
        assert_relative_eq!(r2, 2.0 * r1, epsilon = 1e-3);
    }

    #[test]
    fn time_constant_realistic_case() {
        // μ = 1e-3, α = 1e11, c = 10, A = 1, ΔP = 1e5.
        // a = 1e-3·1e11·10 / (2·1²·1e5) = 1e9 / 2e5 = 5000 s/m⁶.
        let a =
            filtration_time_constant_pressure(1.0e-3_f64, 1.0e11_f64, 10.0_f64, 1.0_f64, 1.0e5_f64);
        assert_relative_eq!(a, 5000.0_f64, epsilon = 1e-9);
    }

    #[test]
    fn time_constant_is_linear_in_concentration() {
        // a ∝ c à μ, α, A, ΔP fixés : doubler la concentration double a.
        let a1 =
            filtration_time_constant_pressure(1.0e-3_f64, 1.0e11_f64, 5.0_f64, 1.0_f64, 1.0e5_f64);
        let a2 =
            filtration_time_constant_pressure(1.0e-3_f64, 1.0e11_f64, 10.0_f64, 1.0_f64, 1.0e5_f64);
        assert_relative_eq!(a2, 2.0 * a1, epsilon = 1e-9);
    }

    #[test]
    fn volume_and_time_are_reciprocal() {
        // Aller-retour : V → t = a·V² + b·V → V doit redonner le volume initial.
        // a = 5000 s/m⁶, b = 100 s/m³, V = 0,5 m³.
        // t = 5000·0,25 + 100·0,5 = 1250 + 50 = 1300 s.
        // V = (-100 + √(100² + 4·5000·1300)) / (2·5000) = (-100 + 5100)/10000 = 0,5 m³.
        let a = 5000.0_f64;
        let b = 100.0_f64;
        let v = 0.5_f64;
        let t = a * v * v + b * v;
        assert_relative_eq!(t, 1300.0_f64, epsilon = 1e-9);
        let v_back = filtration_volume_from_time(a, b, t);
        assert_relative_eq!(v_back, v, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la viscosité dynamique μ doit être > 0")]
    fn non_positive_viscosity_panics() {
        filtration_darcy_flux(1.0e5_f64, 1.0e11_f64, 1.0e10_f64, 0.0_f64);
    }
}
