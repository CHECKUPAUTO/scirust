//! Poutre continue — **théorème des trois moments (Clapeyron)** pour deux travées
//! adjacentes à inertie constante : moment sur l'appui intermédiaire et réaction
//! d'appui déduite des moments d'about.
//!
//! ```text
//! équation générale (3 moments)
//!     M1·L1 + 2·M2·(L1+L2) + M3·L2 = -6·(A1·a1/L1 + A2·b2/L2)
//!
//! deux travées sur appuis simples (M1 = M3 = 0)
//!   charge répartie w1, w2      M2 = -(w1·L1³ + w2·L2³) / (8·(L1+L2))
//!   charge ponctuelle centrée   M2 = -3·(P1·L1² + P2·L2²) / (16·(L1+L2))
//!   deux travées égales (w)     M2 = -w·L² / 8
//!
//! réaction d'appui (gauche d'une travée)
//!     R = w·L/2 + (M_droite - M_gauche) / L
//! ```
//!
//! `L1`, `L2`, `L` portées des travées (m) ; `M1`, `M2`, `M3` moments sur les
//! appuis (N·m, négatif = fibre supérieure tendue sur appui) ; `w`, `w1`, `w2`
//! charges réparties (N/m) ; `P1`, `P2` charges ponctuelles centrées (N) ;
//! `A·a/L` moments statiques des aires du diagramme des moments isostatiques ;
//! `R` réaction d'appui (N). Les termes de charge valent `w·L³/4` (répartie) et
//! `3·P·L²/8` (ponctuelle centrée) par travée.
//!
//! **Convention** : SI cohérent (m, N, N/m, N·m). **Limite honnête** : théorème
//! des trois moments (Clapeyron), poutre à inertie **constante** sur appuis à
//! dénivellations **nulles**, comportement **élastique** linéaire ; seuls les cas
//! de charge répartie uniforme et ponctuelle centrée sont couverts, avec des
//! charges **fournies par l'appelant**. Les cas généraux — dénivellations
//! d'appui, inerties variables, chargements quelconques, plus de deux travées —
//! sont à la charge de l'appelant ; aucune constante matériau ni cas de charge
//! n'est inventé ici.

/// Moment sur l'appui intermédiaire de deux travées sur appuis simples sous
/// charges réparties `M2 = -(w1·L1³ + w2·L2³) / (8·(L1+L2))` (N·m).
///
/// Issu de l'équation des trois moments avec `M1 = M3 = 0` ; le terme de charge
/// répartie vaut `6·A·a/L = w·L³/4` par travée. Réduit à `-w·L²/8` pour deux
/// travées égales.
///
/// Panique si `span1 <= 0`, `span2 <= 0`, `load1 < 0` ou `load2 < 0`.
pub fn contbeam_three_moment_udl(span1: f64, span2: f64, load1: f64, load2: f64) -> f64 {
    assert!(
        span1 > 0.0,
        "la portée de la travée 1 doit être strictement positive"
    );
    assert!(
        span2 > 0.0,
        "la portée de la travée 2 doit être strictement positive"
    );
    assert!(
        load1 >= 0.0,
        "la charge répartie de la travée 1 doit être positive"
    );
    assert!(
        load2 >= 0.0,
        "la charge répartie de la travée 2 doit être positive"
    );
    -(load1 * span1.powi(3) + load2 * span2.powi(3)) / (8.0 * (span1 + span2))
}

/// Moment sur l'appui intermédiaire de deux travées sur appuis simples sous
/// charges ponctuelles centrées `M2 = -3·(P1·L1² + P2·L2²) / (16·(L1+L2))` (N·m).
///
/// Issu de l'équation des trois moments avec `M1 = M3 = 0` ; le terme de charge
/// ponctuelle centrée vaut `6·A·a/L = 3·P·L²/8` par travée. Réduit à `-3·P·L/16`
/// pour deux travées égales.
///
/// Panique si `span1 <= 0`, `span2 <= 0`, `load1 < 0` ou `load2 < 0`.
pub fn contbeam_three_moment_point_center(span1: f64, span2: f64, load1: f64, load2: f64) -> f64 {
    assert!(
        span1 > 0.0,
        "la portée de la travée 1 doit être strictement positive"
    );
    assert!(
        span2 > 0.0,
        "la portée de la travée 2 doit être strictement positive"
    );
    assert!(
        load1 >= 0.0,
        "la charge ponctuelle de la travée 1 doit être positive"
    );
    assert!(
        load2 >= 0.0,
        "la charge ponctuelle de la travée 2 doit être positive"
    );
    -3.0 * (load1 * span1.powi(2) + load2 * span2.powi(2)) / (16.0 * (span1 + span2))
}

/// Moment sur l'appui intermédiaire de deux travées **égales** sous charge
/// répartie `M2 = -w·L²/8` (N·m).
///
/// Cas classique : cas particulier de [`contbeam_three_moment_udl`] pour
/// `span1 = span2 = span` et `load1 = load2 = distributed_load`.
///
/// Panique si `span <= 0` ou `distributed_load < 0`.
pub fn contbeam_support_moment_equal_spans_udl(span: f64, distributed_load: f64) -> f64 {
    assert!(span > 0.0, "la portée doit être strictement positive");
    assert!(
        distributed_load >= 0.0,
        "la charge répartie doit être positive"
    );
    -distributed_load * span * span / 8.0
}

/// Réaction à l'appui gauche d'une travée à partir des moments d'about
/// `R = w·L/2 + (M_droite - M_gauche) / L` (N).
///
/// Superpose la réaction isostatique `w·L/2` de la charge répartie et l'effet
/// des moments aux extrémités de la travée. Avec des moments d'about égaux le
/// terme correctif s'annule et `R = w·L/2`.
///
/// Panique si `span <= 0` ou `distributed_load < 0`.
pub fn contbeam_reaction_from_moments(
    left_moment: f64,
    right_moment: f64,
    span: f64,
    distributed_load: f64,
) -> f64 {
    assert!(span > 0.0, "la portée doit être strictement positive");
    assert!(
        distributed_load >= 0.0,
        "la charge répartie doit être positive"
    );
    distributed_load * span / 2.0 + (right_moment - left_moment) / span
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence : deux travées égales L = 5 m, charge répartie
    // w = 10 000 N/m ; moment d'appui intermédiaire attendu -w·L²/8.
    const L: f64 = 5.0;
    const W: f64 = 10_000.0;

    #[test]
    fn equal_spans_udl_realistic_value() {
        // M2 = -w·L²/8 = -10 000·25/8 = -31 250 N·m.
        let m2 = contbeam_support_moment_equal_spans_udl(L, W);
        assert_relative_eq!(m2, -31_250.0, epsilon = 1e-9);
    }

    #[test]
    fn general_udl_reduces_to_equal_spans() {
        // La formule générale doit redonner le cas égal pour L1=L2, w1=w2.
        let m_general = contbeam_three_moment_udl(L, L, W, W);
        let m_equal = contbeam_support_moment_equal_spans_udl(L, W);
        assert_relative_eq!(m_general, m_equal, epsilon = 1e-9);
        // Valeur chiffrée : -(10 000·125 + 10 000·125)/(8·10) = -31 250 N·m.
        assert_relative_eq!(m_general, -31_250.0, epsilon = 1e-9);
    }

    #[test]
    fn general_udl_unequal_spans_value() {
        // L1=4, L2=6, w1=5 000, w2=8 000 :
        // -(5 000·64 + 8 000·216)/(8·10) = -(320 000 + 1 728 000)/80 = -25 600.
        let m2 = contbeam_three_moment_udl(4.0, 6.0, 5_000.0, 8_000.0);
        assert_relative_eq!(m2, -25_600.0, epsilon = 1e-6);
    }

    #[test]
    fn point_center_equal_spans_matches_classic() {
        // Deux travées égales, charge ponctuelle centrée P = 8 000 N :
        // M2 = -3·P·L/16 = -3·8 000·5/16 = -7 500 N·m.
        let m2 = contbeam_three_moment_point_center(L, L, 8_000.0, 8_000.0);
        assert_relative_eq!(m2, -7_500.0, epsilon = 1e-6);
    }

    #[test]
    fn moment_scales_linearly_with_load() {
        // M2 ∝ w : doubler la charge double le moment.
        let m1 = contbeam_three_moment_udl(4.0, 6.0, 5_000.0, 8_000.0);
        let m2 = contbeam_three_moment_udl(4.0, 6.0, 10_000.0, 16_000.0);
        assert_relative_eq!(m2 / m1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn exterior_reaction_two_equal_spans() {
        // Poutre continue à deux travées égales sous w : appui intermédiaire
        // M2 = -31 250, appuis d'extrémité M = 0. Réaction d'extrémité :
        // R = w·L/2 + (M2 - 0)/L = 25 000 + (-31 250)/5 = 18 750 N,
        // soit le classique 3·w·L/8 = 3·10 000·5/8 = 18 750 N.
        let m2 = contbeam_support_moment_equal_spans_udl(L, W);
        let r = contbeam_reaction_from_moments(0.0, m2, L, W);
        assert_relative_eq!(r, 18_750.0, epsilon = 1e-9);
        assert_relative_eq!(r, 3.0 * W * L / 8.0, epsilon = 1e-9);
    }

    #[test]
    fn reaction_equals_isostatic_when_moments_equal() {
        // Moments d'about égaux : le terme correctif s'annule, R = w·L/2.
        let r = contbeam_reaction_from_moments(-12_000.0, -12_000.0, L, W);
        assert_relative_eq!(r, W * L / 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "portée de la travée 1")]
    fn zero_span_panics() {
        contbeam_three_moment_udl(0.0, L, W, W);
    }
}
