//! Rugosité **géométrique théorique** en tournage : empreinte laissée par un bec
//! d'outil arrondi avançant à une avance donnée.
//!
//! ```text
//! rugosité arithmétique   Ra = f²/(32·r_ε)
//! hauteur max théorique   Rz = f²/(8·r_ε)
//! avance pour un Ra visé  f  = sqrt(32·Ra·r_ε)
//! (par construction Rz = 4·Ra)
//! ```
//!
//! `Ra` écart moyen arithmétique du profil (m), `Rz` hauteur maximale théorique du
//! profil (m), `f` avance par tour (m/tr, ici homogène à une longueur), `r_ε` rayon
//! de bec de l'outil (m).
//!
//! **Convention** : SI ; toutes les longueurs en mètres. **Limite honnête** :
//! rugosité purement **géométrique** (trace idéale du bec arrondi entre deux tours).
//! L'avance et le rayon de bec sont **fournis** par l'appelant ; le modèle ignore
//! l'arête rapportée, les vibrations, l'usure de l'outil et l'écrasement de matière
//! — la rugosité réellement mesurée est donc toujours **supérieure** à ces valeurs.

/// Rugosité arithmétique théorique `Ra = f²/(32·r_ε)` (m).
///
/// Panique si `feed < 0` ou `nose_radius <= 0`.
pub fn turning_theoretical_ra(feed: f64, nose_radius: f64) -> f64 {
    assert!(
        feed >= 0.0 && nose_radius > 0.0,
        "avance f ≥ 0 et rayon de bec r_ε > 0 requis"
    );
    feed * feed / (32.0 * nose_radius)
}

/// Hauteur maximale théorique du profil `Rz = f²/(8·r_ε)` (m).
///
/// Panique si `feed < 0` ou `nose_radius <= 0`.
pub fn turning_theoretical_rz(feed: f64, nose_radius: f64) -> f64 {
    assert!(
        feed >= 0.0 && nose_radius > 0.0,
        "avance f ≥ 0 et rayon de bec r_ε > 0 requis"
    );
    feed * feed / (8.0 * nose_radius)
}

/// Avance donnant un `Ra` visé (réciproque) `f = sqrt(32·Ra·r_ε)` (m/tr).
///
/// Panique si `target_ra < 0` ou `nose_radius <= 0`.
pub fn turning_feed_for_ra(target_ra: f64, nose_radius: f64) -> f64 {
    assert!(
        target_ra >= 0.0 && nose_radius > 0.0,
        "Ra visé ≥ 0 et rayon de bec r_ε > 0 requis"
    );
    (32.0_f64 * target_ra * nose_radius).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ra_realistic_case() {
        // f = 0,1 mm/tr = 1e-4 m ; r_ε = 0,8 mm = 8e-4 m.
        // Ra = (1e-4)²/(32·8e-4) = 1e-8/2,56e-2 = 3,90625e-7 m ≈ 0,39 µm.
        assert_relative_eq!(
            turning_theoretical_ra(1e-4, 8e-4),
            3.906_25e-7,
            max_relative = 1e-12
        );
    }

    #[test]
    fn rz_is_four_times_ra() {
        // Par construction Rz = f²/(8·r_ε) = 4·f²/(32·r_ε) = 4·Ra.
        let (f, r) = (1.2e-4, 4e-4);
        assert_relative_eq!(
            turning_theoretical_rz(f, r),
            4.0 * turning_theoretical_ra(f, r),
            max_relative = 1e-12
        );
    }

    #[test]
    fn feed_for_ra_is_reciprocal() {
        // f → Ra → f doit boucler exactement.
        let (f, r) = (1.5e-4, 6e-4);
        let ra = turning_theoretical_ra(f, r);
        assert_relative_eq!(turning_feed_for_ra(ra, r), f, max_relative = 1e-12);
    }

    #[test]
    fn ra_scales_with_feed_squared() {
        // Ra ∝ f² : doubler l'avance quadruple le Ra.
        let ra1 = turning_theoretical_ra(1e-4, 8e-4);
        let ra2 = turning_theoretical_ra(2e-4, 8e-4);
        assert_relative_eq!(ra2 / ra1, 4.0, max_relative = 1e-12);
    }

    #[test]
    fn ra_inversely_proportional_to_nose_radius() {
        // Ra ∝ 1/r_ε : doubler le rayon de bec divise le Ra par deux.
        let ra1 = turning_theoretical_ra(1e-4, 4e-4);
        let ra2 = turning_theoretical_ra(1e-4, 8e-4);
        assert_relative_eq!(ra1 / ra2, 2.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "rayon de bec r_ε > 0")]
    fn zero_nose_radius_panics() {
        turning_theoretical_ra(1e-4, 0.0);
    }
}
