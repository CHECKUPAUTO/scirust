//! Cannelures (accouplements arbre-moyeu par dentures) — couple transmissible par
//! **matage des flancs**.
//!
//! ```text
//! rayon moyen        r_m = (De + Di)/4
//! aire de contact    A = f·Z·h·L            (f fraction de dents en contact)
//! couple admissible  C = p_adm·A·r_m = p_adm·f·Z·h·L·r_m
//! ```
//!
//! `De`/`Di` diamètres extérieur/intérieur des cannelures (m), `r_m` rayon moyen
//! (m), `Z` nombre de cannelures, `h` hauteur radiale active d'un flanc (m), `L`
//! longueur d'engagement (m), `f` fraction effective de dents en contact
//! (~0,25–0,5 selon le désalignement), `p_adm` pression de matage admissible (Pa).
//!
//! **Convention** : SI cohérent. **Limite honnête** : dimensionnement au
//! **matage** (répartition idéalisée par le facteur `f`, hypothèse classique) ;
//! ne traite ni la concentration de contrainte en pied de cannelure, ni le
//! cisaillement des dents. `p_adm` et `f` sont fournis par l'appelant.

/// Rayon moyen des cannelures `r_m = (De + Di)/4` (m).
///
/// Panique si `outer_diameter <= inner_diameter`.
pub fn mean_radius(outer_diameter: f64, inner_diameter: f64) -> f64 {
    assert!(
        outer_diameter > inner_diameter,
        "le diamètre extérieur doit dépasser l'intérieur"
    );
    (outer_diameter + inner_diameter) / 4.0
}

/// Couple transmissible par matage `C = p_adm·f·Z·h·L·r_m` (N·m).
///
/// Panique si `contact_fraction` hors `]0, 1]`.
pub fn torque_capacity(
    allowable_pressure: f64,
    teeth: u32,
    tooth_height: f64,
    length: f64,
    mean_radius: f64,
    contact_fraction: f64,
) -> f64 {
    assert!(
        contact_fraction > 0.0 && contact_fraction <= 1.0,
        "la fraction de contact doit être dans ]0, 1]"
    );
    allowable_pressure * contact_fraction * teeth as f64 * tooth_height * length * mean_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mean_radius_definition() {
        // De=30 mm, Di=26 mm → r_m = 56/4 = 14 mm.
        assert_relative_eq!(mean_radius(0.030, 0.026), 0.014, epsilon = 1e-12);
    }

    #[test]
    fn torque_scales_with_length_and_teeth() {
        // C ∝ Z·L : doubler la longueur double le couple.
        let base = torque_capacity(20e6, 10, 0.002, 0.030, 0.014, 0.5);
        let longer = torque_capacity(20e6, 10, 0.002, 0.060, 0.014, 0.5);
        assert_relative_eq!(longer / base, 2.0, epsilon = 1e-9);
        // valeur explicite.
        assert_relative_eq!(
            base,
            20e6 * 0.5 * 10.0 * 0.002 * 0.030 * 0.014,
            epsilon = 1e-6
        );
    }

    #[test]
    fn poorer_contact_reduces_capacity() {
        // f=0,25 (désalignement) transmet moitié moins que f=0,5.
        let good = torque_capacity(20e6, 10, 0.002, 0.030, 0.014, 0.5);
        let poor = torque_capacity(20e6, 10, 0.002, 0.030, 0.014, 0.25);
        assert_relative_eq!(good / poor, 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "diamètre extérieur")]
    fn inverted_diameters_panic() {
        mean_radius(0.026, 0.030);
    }

    #[test]
    #[should_panic(expected = "fraction de contact")]
    fn invalid_contact_fraction_panics() {
        torque_capacity(20e6, 10, 0.002, 0.030, 0.014, 1.5);
    }
}
