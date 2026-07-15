//! Acoustique — **affaiblissement acoustique d'une paroi** `TL` (en décibels,
//! dB) par la **loi de masse**, et affaiblissement d'une **paroi composite**
//! dominée par le maillon faible.
//!
//! ```text
//! loi de masse (normale)  TL_n = 20·log10(π·m·f / (ρ·c))              [dB]
//! champ diffus (usuel)    TL_d = TL_n − 5                            [dB]
//! paroi composite         TL_c = 10·log10( Σ S_i / Σ (S_i·τ_i) )     [dB]
//! coefficient depuis TL   τ    = 10^(−TL / 10)              [sans dimension]
//! ```
//!
//! `m` masse surfacique de la paroi [kg·m⁻²], `f` fréquence [Hz], `ρ` masse
//! volumique de l'air (ou du fluide) [kg·m⁻³], `c` célérité du son dans ce
//! fluide [m·s⁻¹], `TL_n` affaiblissement en incidence normale [dB], `TL_d`
//! affaiblissement en champ diffus [dB], `S_i` aire de l'élément `i` [m²],
//! `τ_i` coefficient de transmission de l'élément `i` [sans dimension, 0 < τ ≤ 1],
//! `TL_c` affaiblissement global de la paroi composite [dB], `τ` coefficient de
//! transmission équivalent [sans dimension].
//!
//! **Limite honnête** : les constantes physiques du fluide (`ρ`, `c`), la masse
//! surfacique `m` de la paroi et les coefficients de transmission `τ_i` des
//! éléments sont **fournis par l'appelant** ; aucune valeur « par défaut » n'est
//! inventée. La formulation retenue est la **loi de masse** : elle n'est valable
//! qu'**au-dessus de la fréquence de résonance** de la paroi et **en dessous de
//! la fréquence critique de coïncidence** ; la coïncidence, l'amortissement et
//! la rigidité de flexion **ne sont pas modélisés** (la chute réelle d'isolement
//! au voisinage de la coïncidence peut être bien plus marquée). La correction
//! usuelle de −5 dB pour le champ diffus est un ordre de grandeur empirique,
//! non une identité physique exacte.

use core::f64::consts::PI;

/// Affaiblissement en incidence normale par la loi de masse
/// `TL_n = 20·log10(π·m·f / (ρ·c))` [dB].
///
/// `surface_mass` masse surfacique `m` [kg·m⁻²], `frequency` fréquence `f` [Hz],
/// `air_density` masse volumique `ρ` [kg·m⁻³], `speed_of_sound` célérité `c`
/// [m·s⁻¹] ; le résultat est en dB.
///
/// Panique si l'un des arguments est négatif ou nul.
pub fn stl_mass_law(
    surface_mass: f64,
    frequency: f64,
    air_density: f64,
    speed_of_sound: f64,
) -> f64 {
    assert!(
        surface_mass > 0.0,
        "la masse surfacique doit être strictement positive (kg·m⁻²)"
    );
    assert!(
        frequency > 0.0,
        "la fréquence doit être strictement positive (Hz)"
    );
    assert!(
        air_density > 0.0,
        "la masse volumique du fluide doit être strictement positive (kg·m⁻³)"
    );
    assert!(
        speed_of_sound > 0.0,
        "la célérité du son doit être strictement positive (m·s⁻¹)"
    );
    20.0 * (PI * surface_mass * frequency / (air_density * speed_of_sound)).log10()
}

/// Affaiblissement en champ diffus par la loi de masse `TL_d = TL_n − 5` [dB]
/// (correction empirique usuelle de −5 dB par rapport à l'incidence normale).
///
/// Mêmes arguments et mêmes unités que [`stl_mass_law`] ; le résultat est en dB.
///
/// Panique si l'un des arguments est négatif ou nul.
pub fn stl_mass_law_field(
    surface_mass: f64,
    frequency: f64,
    air_density: f64,
    speed_of_sound: f64,
) -> f64 {
    stl_mass_law(surface_mass, frequency, air_density, speed_of_sound) - 5.0
}

/// Affaiblissement global d'une paroi composite
/// `TL_c = 10·log10( Σ S_i / Σ (S_i·τ_i) )` [dB] : le **maillon faible** (aire
/// mal isolée, τ élevé) domine l'affaiblissement de l'ensemble.
///
/// `areas` aires `S_i` des éléments [m²], `transmission_coefficients` leurs
/// coefficients de transmission `τ_i` [sans dimension, 0 < τ ≤ 1] ; le résultat
/// est en dB.
///
/// Panique si les tranches sont vides ou de longueurs différentes, si une aire
/// est négative ou nulle, ou si un coefficient de transmission n'est pas dans
/// l'intervalle `]0, 1]`.
pub fn stl_composite_transmission(areas: &[f64], transmission_coefficients: &[f64]) -> f64 {
    assert!(
        !areas.is_empty(),
        "la liste des aires ne doit pas être vide (m²)"
    );
    assert!(
        areas.len() == transmission_coefficients.len(),
        "les aires et les coefficients de transmission doivent avoir la même longueur"
    );
    let mut total_area = 0.0_f64;
    let mut transmitted_area = 0.0_f64;
    for (&area, &tau) in areas.iter().zip(transmission_coefficients.iter())
    {
        assert!(
            area > 0.0,
            "chaque aire doit être strictement positive (m²)"
        );
        assert!(
            tau > 0.0 && tau <= 1.0,
            "chaque coefficient de transmission doit être dans ]0, 1]"
        );
        total_area += area;
        transmitted_area += area * tau;
    }
    10.0 * (total_area / transmitted_area).log10()
}

/// Coefficient de transmission depuis l'affaiblissement `τ = 10^(−TL / 10)`
/// [sans dimension] (inverse de `TL = −10·log10(τ)`).
///
/// `transmission_loss_db` affaiblissement `TL` [dB] ; le résultat est sans
/// dimension et appartient à `]0, 1]` pour `TL ≥ 0`.
///
/// Panique si `transmission_loss_db` n'est pas fini.
pub fn stl_transmission_coefficient_from_loss(transmission_loss_db: f64) -> f64 {
    assert!(
        transmission_loss_db.is_finite(),
        "l'affaiblissement doit être fini (dB)"
    );
    10.0_f64.powf(-transmission_loss_db / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Air standard, fourni par l'appelant (aucune valeur imposée par le module).
    const RHO_AIR: f64 = 1.2_f64; // kg·m⁻³
    const C_AIR: f64 = 340.0_f64; // m·s⁻¹

    #[test]
    fn mass_law_known_case() {
        // m = 10 kg·m⁻², f = 1000 Hz, ρ = 1,2, c = 340 :
        // π·m·f/(ρ·c) = π·1e4/408 = 76,99982 ⇒ 20·log10(76,99982) = 37,72979 dB.
        let tl = stl_mass_law(10.0, 1000.0, RHO_AIR, C_AIR);
        assert_relative_eq!(tl, 37.729_794, epsilon = 1e-5);
    }

    #[test]
    fn mass_law_doubling_mass_adds_six_db() {
        // Loi de masse : doubler la masse surfacique ajoute 20·log10(2) ≈ 6,0206 dB.
        let single = stl_mass_law(15.0, 500.0, RHO_AIR, C_AIR);
        let doubled = stl_mass_law(30.0, 500.0, RHO_AIR, C_AIR);
        assert_relative_eq!(doubled - single, 20.0 * 2.0_f64.log10(), epsilon = 1e-12);
    }

    #[test]
    fn mass_law_doubling_frequency_adds_six_db() {
        // Loi de masse : doubler la fréquence ajoute aussi 20·log10(2) ≈ 6,0206 dB.
        let low = stl_mass_law(25.0, 500.0, RHO_AIR, C_AIR);
        let high = stl_mass_law(25.0, 1000.0, RHO_AIR, C_AIR);
        assert_relative_eq!(high - low, 20.0 * 2.0_f64.log10(), epsilon = 1e-12);
    }

    #[test]
    fn field_correction_is_minus_five() {
        // Le champ diffus retire exactement 5 dB à l'incidence normale.
        let normal = stl_mass_law(30.0, 800.0, RHO_AIR, C_AIR);
        let field = stl_mass_law_field(30.0, 800.0, RHO_AIR, C_AIR);
        assert_relative_eq!(normal - field, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn coefficient_and_loss_are_reciprocal() {
        // τ = 10^(−TL/10) puis TL' = −10·log10(τ) redonne TL (aller-retour).
        let tl = 42.0_f64; // dB
        let tau = stl_transmission_coefficient_from_loss(tl);
        assert_relative_eq!(-10.0 * tau.log10(), tl, epsilon = 1e-12);
        // TL = 0 dB ⇒ τ = 1 (transmission totale).
        assert_relative_eq!(
            stl_transmission_coefficient_from_loss(0.0),
            1.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn composite_uniform_equals_single_element() {
        // Paroi uniforme (mêmes τ) : Σ S / Σ (S·τ) = 1/τ, donc TL_c = −10·log10(τ),
        // identique à l'affaiblissement d'un seul élément de ce τ.
        let tau = stl_transmission_coefficient_from_loss(35.0);
        let areas = [4.0, 6.0, 2.5];
        let coeffs = [tau, tau, tau];
        let tl_c = stl_composite_transmission(&areas, &coeffs);
        assert_relative_eq!(tl_c, -10.0 * tau.log10(), epsilon = 1e-12);
        assert_relative_eq!(tl_c, 35.0, epsilon = 1e-12);
    }

    #[test]
    fn composite_weak_link_dominates() {
        // Mur très isolant (TL = 50 dB, τ = 1e-5) percé d'un petit trou ouvert
        // (τ = 1) : l'ensemble s'effondre à ≈ 29,957 dB, très loin des 50 dB du mur.
        let areas = [9.99, 0.01];
        let coeffs = [1e-5, 1.0];
        let tl_c = stl_composite_transmission(&areas, &coeffs);
        assert_relative_eq!(tl_c, 29.956_829, epsilon = 1e-5);
        assert!(
            tl_c < 50.0,
            "le maillon faible doit dominer l'affaiblissement"
        );
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_surface_mass_panics() {
        stl_mass_law(0.0, 1000.0, RHO_AIR, C_AIR);
    }
}
