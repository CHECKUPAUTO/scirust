//! **Taux de dilution en soudage** — part de métal de base refondu dans le
//! cordon et composition résultante du dépôt par mélange pondéré.
//!
//! ```text
//! taux de dilution      D = A_base / (A_base + A_filler) = A_base / A_total
//! fraction d'apport     f = 1 − D
//! composition diluée    c = D·c_base + (1 − D)·c_filler
//! ```
//!
//! `A_base` aire de métal de base refondu (m²), `A_filler` aire de métal
//! d'apport déposé (m²), `A_total = A_base + A_filler` aire totale fondue du
//! cordon (m²), `D` taux de dilution (sans dimension, ∈ [0, 1]), `f` fraction
//! de métal d'apport (sans dimension), `c_base`/`c_filler` teneur d'un élément
//! dans le métal de base / d'apport (même unité de teneur, p. ex. fraction
//! massique ou % massique), `c` teneur du même élément dans le dépôt (même
//! unité que les entrées).
//!
//! **Convention** : SI pour les aires (m²) ; teneurs dans une unité cohérente
//! quelconque (fraction ou pourcentage), conservée en sortie. **Limite
//! honnête** : modèle de **mélange homogène du bain de fusion** — la teneur du
//! dépôt est une **combinaison linéaire pondérée** des teneurs de base et
//! d'apport. Les aires (ou les fractions) proviennent de la **macrographie**
//! du cordon et sont **fournies par l'appelant** ; les teneurs des matériaux
//! sont elles aussi fournies. Aucune valeur « par défaut » de procédé, de
//! matériau ou de rendement n'est inventée. La ségrégation, la volatilisation
//! d'éléments à l'arc et l'hétérogénéité du bain ne sont pas modélisées. Voir
//! [`crate::welds`] et [`crate::carbon_equivalent`].

/// Taux de dilution `D = A_base / (A_base + A_filler)` (sans dimension).
///
/// `total_weld_area` est l'aire totale fondue `A_total = A_base + A_filler`,
/// mesurée sur la macrographie.
///
/// Panique si `base_metal_melted_area < 0`, si `total_weld_area <= 0` ou si
/// `base_metal_melted_area > total_weld_area`.
pub fn dilution_ratio(base_metal_melted_area: f64, total_weld_area: f64) -> f64 {
    assert!(
        base_metal_melted_area >= 0.0,
        "l'aire de métal de base refondu A_base ≥ 0 requise"
    );
    assert!(
        total_weld_area > 0.0,
        "l'aire totale fondue A_total doit être strictement positive"
    );
    assert!(
        base_metal_melted_area <= total_weld_area,
        "A_base ≤ A_total requis (le métal de base ne peut excéder l'aire fondue)"
    );
    base_metal_melted_area / total_weld_area
}

/// Fraction de métal d'apport `f = 1 − D` (sans dimension).
///
/// Panique si `dilution` ∉ [0, 1].
pub fn dilution_filler_fraction(dilution: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&dilution),
        "le taux de dilution D doit être dans [0, 1]"
    );
    1.0 - dilution
}

/// Teneur d'un élément dans le dépôt `c = D·c_base + (1 − D)·c_filler`.
///
/// Combinaison linéaire pondérée des teneurs de base et d'apport (même unité
/// de teneur en entrée et en sortie).
///
/// Panique si `dilution` ∉ [0, 1].
pub fn weld_diluted_composition(base_content: f64, filler_content: f64, dilution: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&dilution),
        "le taux de dilution D doit être dans [0, 1]"
    );
    dilution * base_content + (1.0 - dilution) * filler_content
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dilution_ratio_from_macrograph_areas() {
        // A_base = 12 mm², A_total = 40 mm² → D = 12/40 = 0,30.
        let d = dilution_ratio(12.0, 40.0);
        assert_relative_eq!(d, 0.30, epsilon = 1e-12);
    }

    #[test]
    fn filler_fraction_complements_dilution() {
        // Réciprocité : D + f = 1 pour tout D ∈ [0, 1].
        let d = 0.30_f64;
        assert_relative_eq!(d + dilution_filler_fraction(d), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_dilution_gives_pure_filler_composition() {
        // D = 0 : dépôt = 100 % métal d'apport (aucune dilution du base).
        assert_relative_eq!(
            weld_diluted_composition(0.18, 0.20, 0.0),
            0.20,
            epsilon = 1e-12
        );
    }

    #[test]
    fn full_dilution_gives_pure_base_composition() {
        // D = 1 : dépôt = teneur du métal de base.
        assert_relative_eq!(
            weld_diluted_composition(0.18, 0.20, 1.0),
            0.18,
            epsilon = 1e-12
        );
    }

    #[test]
    fn realistic_chromium_content_in_deposit() {
        // Chrome (% massique) : base 18 %, apport 20 %, D = 0,30 issu des aires.
        // c = 0,30·18 + 0,70·20 = 5,4 + 14,0 = 19,4 %.
        let d = dilution_ratio(12.0, 40.0);
        let c = weld_diluted_composition(18.0, 20.0, d);
        assert_relative_eq!(c, 19.4, epsilon = 1e-12);
    }

    #[test]
    fn identical_contents_are_invariant_to_dilution() {
        // Si base et apport ont la même teneur, le dépôt la conserve, ∀ D.
        assert_relative_eq!(
            weld_diluted_composition(0.08, 0.08, 0.42),
            0.08,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "A_base ≤ A_total requis")]
    fn base_area_exceeding_total_panics() {
        dilution_ratio(50.0, 40.0);
    }
}
