//! Métrologie — contrôle par **verre étalon plan** (verre optique) : l'écart de
//! planéité d'une surface se lit dans les **franges d'interférence** du coin d'air
//! entre l'étalon et la pièce, chaque frange valant une **demi-longueur d'onde**.
//!
//! ```text
//! écart de planéité       dev   = N · λ / 2            (N franges comptées)
//! épaisseur du coin d'air gap   = m · λ / 2            (m-ième frange)
//! défaut par courbure     E     = (bow / spacing) · λ / 2
//! nombre de franges       N     = 2 · dev / λ
//! ```
//!
//! `N` (`fringe_count`), `m` (`fringe_order`) nombres de franges (sans dimension) ;
//! `λ` (`wavelength`) longueur d'onde de la lumière monochromatique (m) ; `dev`,
//! `gap`, `E` écarts/épaisseurs (m) ; `bow` (`fringe_bow`) flèche de courbure d'une
//! frange et `spacing` (`fringe_spacing`) pas entre franges, exprimés dans la
//! **même** unité (leur rapport est sans dimension).
//!
//! **Convention** : SI cohérent ; `dev`, `gap`, `E` et `λ` en mètres, `bow` et
//! `spacing` dans une unité commune quelconque.
//! **Limite honnête** : interférométrie en lumière **monochromatique**, incidence
//! quasi normale, franges d'égale épaisseur ; chaque frange vaut exactement `λ/2`
//! d'écart. La longueur d'onde est **fournie par l'appelant** (le HeNe usuel donne
//! ~0,633 µm, mais aucune valeur n'est imposée ici) : ce module n'invente aucune
//! constante physique ni « défaut » de longueur d'onde.

/// Écart de planéité `dev = N · λ / 2` (m) déduit du **nombre de franges** `N`
/// comptées sous le verre étalon, chaque frange valant une demi-longueur d'onde.
///
/// Panique si `fringe_count < 0` ou `wavelength <= 0`.
pub fn opticalflat_surface_deviation(fringe_count: f64, wavelength: f64) -> f64 {
    assert!(
        fringe_count >= 0.0,
        "le nombre de franges ne peut être négatif"
    );
    assert!(
        wavelength > 0.0,
        "la longueur d'onde doit être strictement positive"
    );
    fringe_count * wavelength / 2.0
}

/// Épaisseur du coin d'air `gap = m · λ / 2` (m) à la **m-ième frange**, mesurée
/// depuis la ligne de contact où le coin est nul.
///
/// Panique si `fringe_order < 0` ou `wavelength <= 0`.
pub fn opticalflat_gap_at_fringe(fringe_order: f64, wavelength: f64) -> f64 {
    assert!(
        fringe_order >= 0.0,
        "l'ordre de frange ne peut être négatif"
    );
    assert!(
        wavelength > 0.0,
        "la longueur d'onde doit être strictement positive"
    );
    fringe_order * wavelength / 2.0
}

/// Défaut de planéité `E = (bow / spacing) · λ / 2` (m) estimé par la **courbure
/// des franges** : le rapport flèche/pas donne la fraction de frange, convertie en
/// écart par la demi-longueur d'onde.
///
/// Panique si `fringe_bow < 0`, `fringe_spacing <= 0` ou `wavelength <= 0`.
pub fn opticalflat_flatness_error_from_curvature(
    fringe_bow: f64,
    fringe_spacing: f64,
    wavelength: f64,
) -> f64 {
    assert!(
        fringe_bow >= 0.0,
        "la flèche de courbure ne peut être négative"
    );
    assert!(
        fringe_spacing > 0.0,
        "le pas entre franges doit être strictement positif"
    );
    assert!(
        wavelength > 0.0,
        "la longueur d'onde doit être strictement positive"
    );
    (fringe_bow / fringe_spacing) * (wavelength / 2.0)
}

/// Nombre de franges `N = 2 · dev / λ` (sans dimension) correspondant à un écart de
/// planéité `dev` donné : réciproque exacte de [`opticalflat_surface_deviation`].
///
/// Panique si `deviation < 0` ou `wavelength <= 0`.
pub fn opticalflat_fringes_from_deviation(deviation: f64, wavelength: f64) -> f64 {
    assert!(deviation >= 0.0, "l'écart de planéité ne peut être négatif");
    assert!(
        wavelength > 0.0,
        "la longueur d'onde doit être strictement positive"
    );
    2.0 * deviation / wavelength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Longueur d'onde HeNe usuelle (fournie), en mètres.
    const HENE: f64 = 0.633e-6;

    #[test]
    fn deviation_and_fringes_are_reciprocal() {
        // Réciprocité : compter N franges puis reconvertir l'écart en franges
        // redonne N pour toute longueur d'onde.
        let n = 3.5_f64;
        let dev = opticalflat_surface_deviation(n, HENE);
        assert_relative_eq!(
            opticalflat_fringes_from_deviation(dev, HENE),
            n,
            max_relative = 1e-12
        );
    }

    #[test]
    fn one_fringe_equals_half_wavelength() {
        // Cas limite fondamental : une frange vaut exactement λ/2.
        let dev = opticalflat_surface_deviation(1.0, HENE);
        assert_relative_eq!(dev, HENE / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn deviation_matches_gap_for_same_count() {
        // Écart de planéité et épaisseur du coin d'air suivent la même règle
        // m·λ/2 : à ordre égal, valeurs identiques.
        let m = 4.0_f64;
        assert_relative_eq!(
            opticalflat_surface_deviation(m, HENE),
            opticalflat_gap_at_fringe(m, HENE),
            max_relative = 1e-12
        );
    }

    #[test]
    fn deviation_scales_linearly_with_count() {
        // Homogénéité de degré 1 en N : doubler le nombre de franges double l'écart.
        let base = opticalflat_surface_deviation(2.0, HENE);
        let doubled = opticalflat_surface_deviation(4.0, HENE);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn numeric_case_three_fringes_hene() {
        // Cas chiffré : 3 franges en HeNe (λ = 0,633 µm).
        // dev = 3 · 0,633e-6 / 2 = 1,899e-6 / 2 = 0,9495e-6 m = 949,5 nm.
        let dev = opticalflat_surface_deviation(3.0, HENE);
        assert_relative_eq!(dev, 0.9495e-6, max_relative = 1e-9);
    }

    #[test]
    fn curvature_error_from_fraction_of_fringe() {
        // Cas chiffré : franges courbées d'un cinquième de leur pas (bow/spacing
        // = 0,2) en HeNe → E = 0,2 · 0,633e-6 / 2 = 0,0633e-6 m = 63,3 nm.
        let e = opticalflat_flatness_error_from_curvature(0.2, 1.0, HENE);
        assert_relative_eq!(e, 0.0633e-6, max_relative = 1e-9);
        // Le rapport seul compte : (2/10) donne le même défaut que (0,2/1).
        let e_scaled = opticalflat_flatness_error_from_curvature(2.0, 10.0, HENE);
        assert_relative_eq!(e_scaled, e, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "longueur d'onde doit être strictement positive")]
    fn zero_wavelength_panics() {
        opticalflat_surface_deviation(3.0, 0.0);
    }
}
