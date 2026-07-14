//! Usinage — **brochage** : nombre de dents simultanément en prise, effort de
//! coupe par dent et effort de brochage maximal.
//!
//! ```text
//! dents en prise         n = floor(L/p) + 1
//! effort par dent        Ft = b·rpt·kc
//! effort de brochage      F = n·Ft
//! ```
//!
//! `L` longueur de coupe (m), `p` pas des dents de la broche (m), `n` nombre de
//! dents simultanément engagées (adimensionnel), `b` largeur de coupe active
//! (m), `rpt` montée (surépaisseur) par dent (m), `kc` effort spécifique de coupe
//! du matériau (Pa = N/m², rapporté à la section de copeau), `Ft` effort de coupe
//! d'une dent (N), `F` effort de brochage résultant (N). Le nombre de dents en
//! prise ajoute un `+1` car sur une longueur `L` couverte par un pas `p` on compte
//! l'entredent parcourue plus la dent d'entrée.
//!
//! **Convention** : SI cohérent (N, m, Pa). **Limite honnête** : effort **maximal**
//! (toutes les dents en prise coupent simultanément à pleine section) ; modèle
//! d'effort spécifique linéaire `Ft = b·rpt·kc` sans correction d'écrouissage,
//! d'usure ni d'angle de coupe. L'effort spécifique de coupe `kc` et la montée par
//! dent `rpt` sont **fournis par l'appelant** — aucune valeur matériau n'est
//! inventée ici. Ne modélise ni l'échauffement, ni la lubrification, ni la flexion
//! de la broche.

/// Nombre de dents simultanément en prise `n = floor(L/p) + 1`.
///
/// Sur une longueur de coupe `L` avec un pas de dents `p`, on compte le nombre
/// entier d'entredents parcourues plus la dent d'attaque.
///
/// Panique si `cut_length < 0` ou si `tooth_pitch <= 0`.
pub fn broaching_teeth_engaged(cut_length: f64, tooth_pitch: f64) -> u32 {
    assert!(cut_length >= 0.0, "la longueur de coupe doit être positive");
    assert!(
        tooth_pitch > 0.0,
        "le pas des dents doit être strictement positif"
    );
    (cut_length / tooth_pitch).floor() as u32 + 1
}

/// Effort de coupe d'une dent `Ft = b·rpt·kc` (N).
///
/// Produit de la section de copeau `b·rpt` (largeur active × montée par dent) par
/// l'effort spécifique de coupe `kc`.
///
/// Panique si `cutting_width < 0`, `rise_per_tooth < 0` ou
/// `specific_cutting_force < 0`.
pub fn broaching_force_per_tooth(
    cutting_width: f64,
    rise_per_tooth: f64,
    specific_cutting_force: f64,
) -> f64 {
    assert!(
        cutting_width >= 0.0,
        "la largeur de coupe doit être positive"
    );
    assert!(
        rise_per_tooth >= 0.0,
        "la montée par dent doit être positive"
    );
    assert!(
        specific_cutting_force >= 0.0,
        "l'effort spécifique de coupe doit être positif"
    );
    cutting_width * rise_per_tooth * specific_cutting_force
}

/// Effort de brochage maximal `F = n·Ft` (N).
///
/// Somme des efforts des `n` dents simultanément en prise, supposées couper
/// toutes à pleine section (effort de crête).
///
/// Panique si `force_per_tooth < 0`.
pub fn broaching_force(teeth_engaged: u32, force_per_tooth: f64) -> f64 {
    assert!(
        force_per_tooth >= 0.0,
        "l'effort par dent doit être positif"
    );
    teeth_engaged as f64 * force_per_tooth
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn teeth_engaged_counts_intervals_plus_one() {
        // L = 30 mm, p = 10 mm → floor(3) + 1 = 4 dents en prise.
        assert_eq!(broaching_teeth_engaged(0.030, 0.010), 4);
        // Longueur nulle → une seule dent en prise (la dent d'attaque).
        assert_eq!(broaching_teeth_engaged(0.0, 0.010), 1);
    }

    #[test]
    fn teeth_engaged_is_stepwise_in_length() {
        // Juste en deçà d'un pas entier : L = 29,9 mm, p = 10 mm → floor(2,99)+1 = 3.
        assert_eq!(broaching_teeth_engaged(0.0299, 0.010), 3);
        // Au franchissement du pas : L = 30,1 mm → floor(3,01)+1 = 4.
        assert_eq!(broaching_teeth_engaged(0.0301, 0.010), 4);
    }

    #[test]
    fn force_per_tooth_is_chip_section_times_kc() {
        // b = 12 mm, rpt = 0,05 mm, kc = 2500 MPa → Ft = 0,012·5e-5·2,5e9 = 1500 N.
        let ft = broaching_force_per_tooth(0.012, 5e-5, 2.5e9);
        assert_relative_eq!(ft, 0.012 * 5e-5 * 2.5e9, epsilon = 1e-9);
        assert_relative_eq!(ft, 1500.0, epsilon = 1e-9);
    }

    #[test]
    fn force_per_tooth_scales_linearly_with_rise() {
        // Ft ∝ rpt : doubler la montée par dent double l'effort par dent.
        let f1 = broaching_force_per_tooth(0.012, 4e-5, 2.5e9);
        let f2 = broaching_force_per_tooth(0.012, 8e-5, 2.5e9);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn total_force_is_sum_over_engaged_teeth() {
        // F = n·Ft : reconstruction d'une chaîne complète depuis les données de coupe.
        // L=30 mm, p=10 mm → n=4 ; Ft=1500 N → F = 6000 N.
        let n = broaching_teeth_engaged(0.030, 0.010);
        let ft = broaching_force_per_tooth(0.012, 5e-5, 2.5e9);
        let f = broaching_force(n, ft);
        assert_eq!(n, 4);
        assert_relative_eq!(f, 4.0 * ft, epsilon = 1e-9);
        assert_relative_eq!(f, 6000.0, epsilon = 1e-9);
    }

    #[test]
    fn total_force_is_proportional_to_engaged_teeth() {
        // F ∝ n à effort par dent constant.
        let ft = 1200.0;
        assert_relative_eq!(
            broaching_force(6, ft) / broaching_force(3, ft),
            2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "pas des dents")]
    fn zero_pitch_panics() {
        broaching_teeth_engaged(0.030, 0.0);
    }
}
