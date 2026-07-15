//! Flexion plastique et rotule plastique : module plastique, facteur de forme,
//! moment d'entrée en plasticité et moment plastique (rotule).
//!
//! ```text
//! module plastique rectangle   Zp = b·h² / 4
//! facteur de forme             k  = Zp / Z
//! moment d'entrée plasticité   My = σy · Z
//! moment plastique (rotule)    Mp = σy · Zp   =   k · My
//! ```
//!
//! `Zp` module plastique de la section (m³), `Z` module de résistance élastique
//! `= I/c` (m³), `b` largeur de la section rectangulaire (m), `h` hauteur dans le
//! plan de flexion (m), `k` facteur de forme (sans dimension, `1,5` pour un
//! rectangle plein), `σy` contrainte d'écoulement du matériau (Pa), `My` moment
//! fléchissant amenant la fibre extrême à la limite d'élasticité (N·m), `Mp`
//! moment plastique, plateau de la rotule plastique (N·m).
//!
//! **Convention** : SI cohérent, flexion pure. **Limite honnête** : matériau
//! élastique-parfaitement plastique (pas d'écrouissage), section symétrique par
//! rapport à l'axe neutre, flexion pure (effort tranchant, voilement et effets de
//! second ordre négligés). La contrainte d'écoulement `σy` et les modules de
//! section sont FOURNIS par l'appelant ; le facteur de forme dépend de la
//! géométrie (rectangle plein `= 1,5`, démontré ici ; autres sections FOURNIES)
//! et aucune valeur matériau ou géométrique n'est supposée par défaut.

/// Module plastique d'une section rectangulaire pleine `Zp = b·h² / 4` (m³),
/// flexion autour de l'axe parallèle à la largeur `b`.
///
/// `width` largeur `b` en m, `height` hauteur `h` (dans le plan de flexion) en m.
///
/// Panique si `width <= 0` ou `height <= 0`.
pub fn plastic_section_modulus_rectangle(width: f64, height: f64) -> f64 {
    assert!(width > 0.0, "la largeur doit être strictement positive");
    assert!(height > 0.0, "la hauteur doit être strictement positive");
    width * height * height / 4.0
}

/// Facteur de forme d'une section `k = Zp / Z` (sans dimension), rapport du
/// module plastique au module de résistance élastique (`1,5` pour un rectangle).
///
/// `plastic_section_modulus` module plastique `Zp` en m³, `elastic_section_modulus`
/// module de résistance élastique `Z` en m³.
///
/// Panique si `plastic_section_modulus <= 0` ou `elastic_section_modulus <= 0`.
pub fn plastic_shape_factor(plastic_section_modulus: f64, elastic_section_modulus: f64) -> f64 {
    assert!(
        plastic_section_modulus > 0.0,
        "le module plastique doit être strictement positif"
    );
    assert!(
        elastic_section_modulus > 0.0,
        "le module de résistance élastique doit être strictement positif"
    );
    plastic_section_modulus / elastic_section_modulus
}

/// Moment plastique (plateau de la rotule plastique) `Mp = σy · Zp` (N·m).
///
/// `yield_stress` contrainte d'écoulement `σy` en Pa, `plastic_section_modulus`
/// module plastique `Zp` en m³.
///
/// Panique si `yield_stress <= 0` ou `plastic_section_modulus <= 0`.
pub fn plastic_moment(yield_stress: f64, plastic_section_modulus: f64) -> f64 {
    assert!(
        yield_stress > 0.0,
        "la contrainte d'écoulement doit être strictement positive"
    );
    assert!(
        plastic_section_modulus > 0.0,
        "le module plastique doit être strictement positif"
    );
    yield_stress * plastic_section_modulus
}

/// Moment d'entrée en plasticité `My = σy · Z` (N·m), moment amenant la fibre
/// extrême à la limite d'élasticité.
///
/// `yield_stress` contrainte d'écoulement `σy` en Pa, `elastic_section_modulus`
/// module de résistance élastique `Z` en m³.
///
/// Panique si `yield_stress <= 0` ou `elastic_section_modulus <= 0`.
pub fn plastic_yield_moment(yield_stress: f64, elastic_section_modulus: f64) -> f64 {
    assert!(
        yield_stress > 0.0,
        "la contrainte d'écoulement doit être strictement positive"
    );
    assert!(
        elastic_section_modulus > 0.0,
        "le module de résistance élastique doit être strictement positif"
    );
    yield_stress * elastic_section_modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rectangle_shape_factor_is_three_halves() {
        // Rectangle plein : Zp = b·h²/4, Z = b·h²/6, donc k = 6/4 = 1,5.
        let (b, h) = (0.03_f64, 0.12_f64);
        let zp = plastic_section_modulus_rectangle(b, h);
        let z = b * h * h / 6.0; // module élastique du rectangle
        assert_relative_eq!(plastic_shape_factor(zp, z), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn plastic_section_modulus_scales_with_height_squared() {
        // Zp ∝ h² à largeur constante : doubler h quadruple Zp.
        let b = 0.04_f64;
        let zp1 = plastic_section_modulus_rectangle(b, 0.1);
        let zp2 = plastic_section_modulus_rectangle(b, 0.2);
        assert_relative_eq!(zp2 / zp1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn moment_ratio_equals_shape_factor() {
        // Mp/My = (σy·Zp)/(σy·Z) = Zp/Z = k : identité entre moments et forme.
        let (sigma_y, zp, z) = (250.0e6_f64, 1.25e-4_f64, 8.3e-5_f64);
        let mp = plastic_moment(sigma_y, zp);
        let my = plastic_yield_moment(sigma_y, z);
        assert_relative_eq!(mp / my, plastic_shape_factor(zp, z), epsilon = 1e-12);
    }

    #[test]
    fn plastic_moment_is_linear_in_yield_stress() {
        // Mp ∝ σy à section constante : tripler σy triple Mp.
        let zp = 1.0e-4_f64;
        let m1 = plastic_moment(200.0e6, zp);
        let m2 = plastic_moment(600.0e6, zp);
        assert_relative_eq!(m2 / m1, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_rectangular_hinge_case() {
        // Section 50×100 mm, σy = 250 MPa (acier doux).
        // Zp = 0,05 · 0,1² / 4 = 1,25e-4 m³ ; Z = 0,05 · 0,1² / 6 = 8,333…e-5 m³.
        // My = 250e6 · 8,333…e-5 = 20 833,33 N·m ; Mp = 250e6 · 1,25e-4 = 31 250 N·m.
        let (b, h, sigma_y) = (0.05_f64, 0.1_f64, 250.0e6_f64);
        let zp = plastic_section_modulus_rectangle(b, h);
        assert_relative_eq!(zp, 1.25e-4, epsilon = 1e-15);
        let z = b * h * h / 6.0;
        let my = plastic_yield_moment(sigma_y, z);
        let mp = plastic_moment(sigma_y, zp);
        assert_relative_eq!(my, 250.0e6 * 0.0005 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(mp, 31250.0, epsilon = 1e-6);
        assert_relative_eq!(mp - my, 31250.0 - 250.0e6 * 0.0005 / 6.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "la contrainte d'écoulement doit être strictement positive")]
    fn plastic_moment_rejects_non_positive_yield_stress() {
        plastic_moment(0.0, 1.0e-4);
    }
}
