//! Clavette parallèle rectangulaire — **contraintes de la liaison arbre-moyeu**
//! par clavette : effort tangentiel transmis à la surface de l'arbre,
//! cisaillement et matage de la clavette, longueur requise au cisaillement.
//!
//! ```text
//! effort tangentiel     F  = 2·T / d
//! cisaillement clavette τ  = 2·T / (d·w·L)
//! matage (½ hauteur)    σb = 4·T / (d·h·L)
//! longueur requise      L  = 2·T / (d·w·τ_adm)
//! ```
//!
//! `T` couple transmis (N·m), `d` diamètre de l'arbre (m), `w` largeur de la
//! clavette (m), `h` hauteur de la clavette (m), `L` longueur de contact de la
//! clavette (m), `τ_adm` cisaillement admissible du matériau de la clavette (Pa) ;
//! `F` (N), `τ` et `σb` (Pa). L'effort tangentiel est ramené au rayon `d/2` de
//! l'arbre (`F = T / (d/2)`). Le cisaillement s'exerce sur la section
//! longitudinale `w·L` de la clavette ; le matage sur la moitié de la hauteur en
//! appui, soit `(h/2)·L`, d'où le facteur 4.
//!
//! **Convention** : SI cohérent (N, m, Pa, N·m). **Limite honnête** : clavette
//! parallèle rectangulaire, couple transmis par **cisaillement ET matage** avec
//! une répartition d'effort **idéalisée uniforme** sur la longueur ; les
//! contraintes admissibles (matériau/procédé) sont **fournies par l'appelant** —
//! aucune valeur matériau, coefficient de sécurité ni ajustement de portée n'est
//! inventé ici. Pas de prise en compte de la concentration de contrainte en fond
//! de rainure, du serrage ni du frottement au moyeu.

/// Effort tangentiel à la surface de l'arbre `F = 2·T / d` (N).
///
/// Couple ramené au rayon de l'arbre : `F = T / (d/2)`.
///
/// Panique si `torque < 0` ou `shaft_diameter <= 0`.
pub fn keyway_tangential_force(torque: f64, shaft_diameter: f64) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif");
    assert!(
        shaft_diameter > 0.0,
        "le diamètre de l'arbre doit être strictement positif"
    );
    2.0 * torque / shaft_diameter
}

/// Contrainte de cisaillement dans la clavette `τ = 2·T / (d·w·L)` (Pa).
///
/// Effort tangentiel `2·T/d` réparti sur la section longitudinale `w·L`.
///
/// Panique si `torque < 0`, `shaft_diameter <= 0`, `key_width <= 0` ou
/// `key_length <= 0`.
pub fn keyway_shear_stress(
    torque: f64,
    shaft_diameter: f64,
    key_width: f64,
    key_length: f64,
) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif");
    assert!(
        shaft_diameter > 0.0,
        "le diamètre de l'arbre doit être strictement positif"
    );
    assert!(
        key_width > 0.0,
        "la largeur de la clavette doit être strictement positive"
    );
    assert!(
        key_length > 0.0,
        "la longueur de la clavette doit être strictement positive"
    );
    2.0 * torque / (shaft_diameter * key_width * key_length)
}

/// Contrainte de matage sur la clavette `σb = 4·T / (d·h·L)` (Pa).
///
/// Effort tangentiel `2·T/d` réparti sur la moitié de la hauteur en appui,
/// soit la surface `(h/2)·L`, d'où le facteur 4.
///
/// Panique si `torque < 0`, `shaft_diameter <= 0`, `key_height <= 0` ou
/// `key_length <= 0`.
pub fn keyway_bearing_stress(
    torque: f64,
    shaft_diameter: f64,
    key_height: f64,
    key_length: f64,
) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif");
    assert!(
        shaft_diameter > 0.0,
        "le diamètre de l'arbre doit être strictement positif"
    );
    assert!(
        key_height > 0.0,
        "la hauteur de la clavette doit être strictement positive"
    );
    assert!(
        key_length > 0.0,
        "la longueur de la clavette doit être strictement positive"
    );
    4.0 * torque / (shaft_diameter * key_height * key_length)
}

/// Longueur de clavette requise au cisaillement `L = 2·T / (d·w·τ_adm)` (m).
///
/// Réciproque de [`keyway_shear_stress`] : longueur minimale pour ne pas
/// dépasser le cisaillement admissible `τ_adm`.
///
/// Panique si `torque < 0`, `shaft_diameter <= 0`, `key_width <= 0` ou
/// `allowable_shear <= 0`.
pub fn keyway_required_length_shear(
    torque: f64,
    shaft_diameter: f64,
    key_width: f64,
    allowable_shear: f64,
) -> f64 {
    assert!(torque >= 0.0, "le couple doit être positif");
    assert!(
        shaft_diameter > 0.0,
        "le diamètre de l'arbre doit être strictement positif"
    );
    assert!(
        key_width > 0.0,
        "la largeur de la clavette doit être strictement positive"
    );
    assert!(
        allowable_shear > 0.0,
        "le cisaillement admissible doit être strictement positif"
    );
    2.0 * torque / (shaft_diameter * key_width * allowable_shear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence : arbre Ø40 mm, couple 200 N·m,
    // clavette 12×8 mm de longueur 50 mm.
    const T: f64 = 200.0;
    const D: f64 = 0.040;
    const W: f64 = 0.012;
    const H: f64 = 0.008;
    const L: f64 = 0.050;

    #[test]
    fn tangential_force_realistic_value() {
        // F = 2·T/d = 2·200/0,04 = 10 000 N.
        let f = keyway_tangential_force(T, D);
        assert_relative_eq!(f, 10_000.0, epsilon = 1e-9);
    }

    #[test]
    fn shear_stress_is_force_over_shear_area() {
        // τ = F / (w·L) : le cisaillement est bien l'effort tangentiel
        // divisé par la section longitudinale w·L.
        let f = keyway_tangential_force(T, D);
        let tau = keyway_shear_stress(T, D, W, L);
        assert_relative_eq!(tau, f / (W * L), epsilon = 1e-6);
        // Valeur chiffrée : 10 000 / (0,012·0,05) ≈ 16,6667 MPa.
        assert_relative_eq!(tau, 16_666_666.666_666_666, epsilon = 1e-3);
    }

    #[test]
    fn bearing_stress_is_force_over_half_height_area() {
        // σb = F / ((h/2)·L) : matage sur la moitié de la hauteur en appui.
        let f = keyway_tangential_force(T, D);
        let sigma = keyway_bearing_stress(T, D, H, L);
        assert_relative_eq!(sigma, f / (0.5 * H * L), epsilon = 1e-6);
        assert_relative_eq!(sigma, 50_000_000.0, epsilon = 1e-1);
    }

    #[test]
    fn bearing_to_shear_ratio_is_two_w_over_h() {
        // σb/τ = 2·w/h : rapport indépendant du couple et de la longueur.
        let tau = keyway_shear_stress(T, D, W, L);
        let sigma = keyway_bearing_stress(T, D, H, L);
        assert_relative_eq!(sigma / tau, 2.0 * W / H, epsilon = 1e-12);
    }

    #[test]
    fn required_length_is_reciprocal_of_shear_stress() {
        // Réciprocité : dimensionner à τ = contrainte calculée redonne L.
        let tau = keyway_shear_stress(T, D, W, L);
        let l = keyway_required_length_shear(T, D, W, tau);
        assert_relative_eq!(l, L, epsilon = 1e-12);
    }

    #[test]
    fn shear_stress_scales_inversely_with_length() {
        // τ ∝ 1/L : doubler la longueur divise le cisaillement par deux.
        let tau1 = keyway_shear_stress(T, D, W, L);
        let tau2 = keyway_shear_stress(T, D, W, 2.0 * L);
        assert_relative_eq!(tau1 / tau2, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "diamètre de l'arbre")]
    fn zero_diameter_panics() {
        keyway_tangential_force(T, 0.0);
    }
}
