//! Résonance (**surge**) d'un ressort hélicoïdal de compression — fréquence du
//! premier mode longitudinal d'un ressort à masse répartie.
//!
//! ```text
//! extrémités fixes-fixes   f = (d/(2·π·n·Dm²))·sqrt(G/(2·ρ))
//! extrémités fixes-libres  f = (d/(4·π·n·Dm²))·sqrt(G/(2·ρ))   (= moitié)
//! ```
//!
//! `d` diamètre du fil (m), `Dm` diamètre moyen d'enroulement (m), `n` nombre de
//! spires actives (sans dimension), `G` module de cisaillement du matériau (Pa),
//! `ρ` masse volumique du matériau (kg·m⁻³), `f` fréquence propre (Hz). Le surge
//! survient lorsqu'une sollicitation cyclique (came, soupape) approche `f` :
//! l'onde longitudinale se propage dans le ressort et amplifie les contraintes.
//!
//! **Convention** : SI cohérent, fréquence en hertz. **Limite honnête** : modèle
//! du ressort comme milieu élastique continu à masse uniformément répartie
//! (masse des spires seule, sans masse rapportée en bout), premier mode
//! longitudinal, amortissement négligé. Les constantes de matériau `G` et `ρ`
//! sont **fournies par l'appelant** : aucune valeur « par défaut » n'est
//! inventée ici.

use core::f64::consts::PI;

/// Fréquence du premier mode de surge, extrémités **fixes-fixes** (Hz) :
/// `f = (d/(2·π·n·Dm²))·sqrt(G/(2·ρ))`.
///
/// `wire_diameter` et `mean_coil_diameter` en m, `active_coils` sans dimension,
/// `shear_modulus` en Pa, `density` en kg·m⁻³.
///
/// Panique si `active_coils <= 0`, `mean_coil_diameter <= 0`, `density <= 0`,
/// `shear_modulus < 0` ou `wire_diameter < 0`.
pub fn spring_surge_frequency_hz(
    wire_diameter: f64,
    mean_coil_diameter: f64,
    active_coils: f64,
    shear_modulus: f64,
    density: f64,
) -> f64 {
    assert!(
        wire_diameter >= 0.0,
        "le diamètre du fil ne peut pas être négatif"
    );
    assert!(
        mean_coil_diameter > 0.0,
        "le diamètre moyen d'enroulement doit être strictement positif"
    );
    assert!(
        active_coils > 0.0,
        "le nombre de spires actives doit être strictement positif"
    );
    assert!(
        shear_modulus >= 0.0,
        "le module de cisaillement ne peut pas être négatif"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    let two = 2.0_f64;
    (wire_diameter / (2.0 * PI * active_coils * mean_coil_diameter.powi(2)))
        * (shear_modulus / (two * density)).sqrt()
}

/// Fréquence du premier mode de surge, extrémités **fixes-libres** (Hz) :
/// `f = (d/(4·π·n·Dm²))·sqrt(G/(2·ρ))`, soit la **moitié** du cas fixes-fixes.
///
/// Mêmes unités que [`spring_surge_frequency_hz`].
///
/// Panique si `active_coils <= 0`, `mean_coil_diameter <= 0`, `density <= 0`,
/// `shear_modulus < 0` ou `wire_diameter < 0`.
pub fn spring_surge_frequency_fixed_free_hz(
    wire_diameter: f64,
    mean_coil_diameter: f64,
    active_coils: f64,
    shear_modulus: f64,
    density: f64,
) -> f64 {
    0.5 * spring_surge_frequency_hz(
        wire_diameter,
        mean_coil_diameter,
        active_coils,
        shear_modulus,
        density,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fixed_free_is_half_of_fixed_fixed() {
        // Identité physique : le mode fixes-libres est exactement la moitié.
        let ff = spring_surge_frequency_hz(0.003, 0.020, 8.0, 79.3e9, 7850.0);
        let fl = spring_surge_frequency_fixed_free_hz(0.003, 0.020, 8.0, 79.3e9, 7850.0);
        assert_relative_eq!(fl, 0.5 * ff, max_relative = 1e-12);
    }

    #[test]
    fn matches_closed_form() {
        // Vérifie l'expression littérale f = (d/(2·π·n·Dm²))·sqrt(G/(2·ρ)).
        let (d, dm, n, g, rho) = (0.003_f64, 0.020_f64, 8.0_f64, 79.3e9_f64, 7850.0_f64);
        let expected = (d / (2.0 * PI * n * dm.powi(2))) * (g / (2.0_f64 * rho)).sqrt();
        assert_relative_eq!(
            spring_surge_frequency_hz(d, dm, n, g, rho),
            expected,
            max_relative = 1e-12
        );
    }

    #[test]
    fn frequency_scales_inversely_with_active_coils() {
        // f ∝ 1/n : doubler le nombre de spires actives divise f par deux.
        let f1 = spring_surge_frequency_hz(0.003, 0.020, 6.0, 79.3e9, 7850.0);
        let f2 = spring_surge_frequency_hz(0.003, 0.020, 12.0, 79.3e9, 7850.0);
        assert_relative_eq!(f1 / f2, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn frequency_scales_inversely_with_coil_diameter_squared() {
        // f ∝ 1/Dm² : doubler Dm divise f par quatre.
        let f1 = spring_surge_frequency_hz(0.003, 0.020, 8.0, 79.3e9, 7850.0);
        let f2 = spring_surge_frequency_hz(0.003, 0.040, 8.0, 79.3e9, 7850.0);
        assert_relative_eq!(f1 / f2, 4.0, max_relative = 1e-12);
    }

    #[test]
    fn frequency_scales_with_sqrt_of_modulus_over_density() {
        // f ∝ sqrt(G/ρ) : quadrupler G/ρ double f.
        let f1 = spring_surge_frequency_hz(0.003, 0.020, 8.0, 79.3e9, 7850.0);
        let f2 = spring_surge_frequency_hz(0.003, 0.020, 8.0, 4.0 * 79.3e9, 7850.0);
        assert_relative_eq!(f2 / f1, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn realistic_valve_spring_case() {
        // Ressort de soupape acier : d = 3 mm, Dm = 20 mm, n = 8,
        // G ≈ 79,3 GPa, ρ = 7850 kg/m³ → surge de l'ordre de quelques centaines de Hz.
        let f = spring_surge_frequency_hz(0.003, 0.020, 8.0, 79.3e9, 7850.0);
        assert!(
            (300.0..900.0).contains(&f),
            "fréquence de surge hors plage attendue : {f} Hz"
        );
    }

    #[test]
    #[should_panic(expected = "spires actives")]
    fn zero_active_coils_panics() {
        spring_surge_frequency_hz(0.003, 0.020, 0.0, 79.3e9, 7850.0);
    }
}
