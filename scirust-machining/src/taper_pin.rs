//! Goupille conique (taper pin) — dimensionnement au cisaillement sur la section
//! moyenne (diamètre milieu du cône).
//!
//! ```text
//! diamètre moyen        d_moy = d_petit + c·L/2
//! aire moyenne          A = π/4·d_moy²
//! cisaillement simple   τ  = F / (π/4·d_moy²)
//! cisaillement double   τ2 = F / (2·π/4·d_moy²)
//! ```
//!
//! `d_petit` diamètre du petit bout (m), `c` conicité (rapport diamétral sans
//! dimension, p. ex. 1/50 pour une goupille normalisée 1:50), `L` longueur du cône
//! (m), `d_moy` diamètre de la section moyenne (m), `A` aire cisaillée (m²), `F`
//! charge transversale (N), `τ` contrainte de cisaillement (Pa). En cisaillement
//! double la même charge se répartit sur deux sections, d'où la moitié de la
//! contrainte.
//!
//! **Convention** : SI cohérent, section circulaire pleine, sollicitation
//! **statique**, cisaillement pur de la section moyenne du cône. **Limite honnête** :
//! la conicité `c` (1:50 pour les goupilles ISO 2339), la charge admissible et les
//! caractéristiques matériau sont **fournies par l'appelant** ; aucune valeur
//! « par défaut » n'est inventée ici.

use core::f64::consts::PI;

/// Diamètre de la section moyenne d'une goupille conique
/// `d_moy = d_petit + c·L/2` (m).
///
/// La conicité `c` est le rapport diamétral (variation de diamètre par unité de
/// longueur) ; le diamètre au milieu du cône vaut le petit diamètre augmenté de la
/// moitié de la variation totale `c·L`.
///
/// Panique si `small_diameter <= 0`, `taper_ratio < 0` ou `length < 0`.
pub fn taperpin_mean_diameter(small_diameter: f64, taper_ratio: f64, length: f64) -> f64 {
    assert!(
        small_diameter > 0.0,
        "le petit diamètre doit être strictement positif"
    );
    assert!(
        taper_ratio >= 0.0,
        "la conicité doit être positive ou nulle"
    );
    assert!(length >= 0.0, "la longueur doit être positive ou nulle");
    small_diameter + taper_ratio * length / 2.0
}

/// Contrainte de cisaillement simple `τ = F / (π/4·d_moy²)` (Pa).
///
/// Panique si `mean_diameter <= 0` ou `load < 0`.
pub fn taperpin_shear_stress(load: f64, mean_diameter: f64) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive ou nulle");
    assert!(
        mean_diameter > 0.0,
        "le diamètre moyen doit être strictement positif"
    );
    load / (PI / 4.0 * mean_diameter.powi(2))
}

/// Contrainte de cisaillement double `τ2 = F / (2·π/4·d_moy²)` (Pa).
///
/// La charge est reprise par deux sections cisaillées : la contrainte vaut la
/// moitié du cas simple.
///
/// Panique si `mean_diameter <= 0` ou `load < 0`.
pub fn taperpin_double_shear_stress(load: f64, mean_diameter: f64) -> f64 {
    assert!(load >= 0.0, "la charge doit être positive ou nulle");
    assert!(
        mean_diameter > 0.0,
        "le diamètre moyen doit être strictement positif"
    );
    load / (2.0 * PI / 4.0 * mean_diameter.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mean_diameter_standard_taper() {
        // d_petit=6 mm, conicité 1:50, L=30 mm → variation totale 0,6 mm,
        // milieu +0,3 mm → d_moy = 6,3 mm.
        let d_moy = taperpin_mean_diameter(0.006, 1.0 / 50.0, 0.030);
        assert_relative_eq!(d_moy, 0.0063, epsilon = 1e-12);
    }

    #[test]
    fn mean_diameter_cylindrical_limit() {
        // Conicité nulle → goupille cylindrique : d_moy = d_petit.
        assert_relative_eq!(
            taperpin_mean_diameter(0.008, 0.0, 0.040),
            0.008,
            epsilon = 1e-15
        );
    }

    #[test]
    fn double_shear_is_half_of_single() {
        // Identité : la même charge sur deux sections → moitié de contrainte.
        let single = taperpin_shear_stress(3500.0, 0.0063);
        let double = taperpin_double_shear_stress(3500.0, 0.0063);
        assert_relative_eq!(double, single / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn shear_stress_proportional_to_load() {
        // τ ∝ F : doubler la charge double la contrainte.
        let base = taperpin_shear_stress(1000.0, 0.010);
        let twice = taperpin_shear_stress(2000.0, 0.010);
        assert_relative_eq!(twice, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn shear_stress_numeric_case() {
        // F=2000 N sur d_moy=6,3 mm : A = π/4·(0,0063)² = 3,117245e-5 m²,
        // τ = 2000 / A ≈ 6,41592e7 Pa (≈ 64,16 MPa).
        let tau = taperpin_shear_stress(2000.0, 0.0063);
        let area = PI / 4.0 * 0.0063_f64.powi(2);
        assert_relative_eq!(tau, 2000.0 / area, epsilon = 1e-6);
        assert_relative_eq!(tau, 6.415_92e7, epsilon = 1e3);
    }

    #[test]
    #[should_panic(expected = "diamètre moyen")]
    fn zero_diameter_shear_panics() {
        taperpin_shear_stress(1000.0, 0.0);
    }
}
