//! Extraction liquide-liquide — coefficient de partage, facteur d'extraction et
//! fraction de soluté restant dans le raffinat pour un étage unique, une cascade
//! à courants croisés (solvant frais à chaque étage) et une cascade à
//! contre-courant d'étages théoriques.
//!
//! ```text
//! coefficient de partage      K = c_E / c_R                                 [-]
//! facteur d'extraction        E = K · S / F                                 [-]
//! raffinat, 1 étage           φ₁ = 1 / (1 + E)                              [-]
//! raffinat, N à courants     φ_c = (1 / (1 + Eₛ))^N                         [-]
//!   croisés
//! raffinat, N à              φ_cc = (E − 1) / (E^(N+1) − 1)      (E ≠ 1)     [-]
//!   contre-courant
//! ```
//!
//! `K` coefficient de partage (distribution) du soluté extrait/raffinat [sans
//! dimension], `c_E`/`c_R` concentrations du soluté dans l'extrait/le raffinat
//! (mêmes unités, p. ex. mol·m⁻³ ou fraction massique) [cohérentes], `E` facteur
//! d'extraction [sans dimension], `S` débit de solvant [kg·s⁻¹ ou mol·s⁻¹], `F`
//! débit d'alimentation (charge) [même unité que `S`], `Eₛ` facteur d'extraction
//! par étage [sans dimension], `N` nombre d'étages théoriques [étages],
//! `φ₁`/`φ_c`/`φ_cc` fraction de soluté restant dans le raffinat [sans dimension,
//! 0 ≤ φ ≤ 1] ; le rendement d'extraction vaut `1 − φ`.
//!
//! **Limite honnête** : ces relations valent pour une **extraction diluée** à
//! **coefficient de partage `K` CONSTANT FOURNI** par l'appelant (loi de
//! distribution linéaire ; jamais inventé, issu de tables, de la loi de Nernst
//! ou d'essais), des **solvants totalement NON MISCIBLES** (débits de porteur
//! `F` et `S` conservés) et des **étages théoriques à l'équilibre**. Le
//! **rendement d'étage réel** (efficacité de Murphree ou globale) est **FOURNI**
//! par l'appelant pour convertir en étages réels ; de même les **coefficients de
//! partage, volatilités, constantes cinétiques et diffusivités** éventuels ne
//! sont **jamais** supposés par défaut. La formule à **contre-courant** suppose
//! `E ≠ 1` (le cas `E = 1` donne la forme limite `φ_cc = 1/(N+1)`, non traitée
//! ici pour éviter une valeur inventée).

/// Coefficient de partage (distribution) `K = c_E / c_R` (sans dimension).
///
/// `solute_extract_concentration` (c_E) et `solute_raffinate_concentration`
/// (c_R) concentrations du soluté à l'équilibre dans la phase extrait et dans la
/// phase raffinat, exprimées dans les **mêmes unités cohérentes**.
///
/// Panique si `solute_extract_concentration < 0` ou si
/// `solute_raffinate_concentration <= 0`.
pub fn extract_distribution_coefficient(
    solute_extract_concentration: f64,
    solute_raffinate_concentration: f64,
) -> f64 {
    assert!(
        solute_extract_concentration >= 0.0,
        "c_E ≥ 0 requis (concentration de soluté dans l'extrait)"
    );
    assert!(
        solute_raffinate_concentration > 0.0,
        "c_R > 0 requis (concentration de soluté dans le raffinat)"
    );
    solute_extract_concentration / solute_raffinate_concentration
}

/// Facteur d'extraction `E = K · S / F` (sans dimension).
///
/// `distribution_coefficient` (K) sans dimension ; `solvent_flow` (S) débit de
/// solvant et `feed_flow` (F) débit d'alimentation, exprimés dans la **même
/// unité** (kg·s⁻¹ ou mol·s⁻¹).
///
/// Panique si `distribution_coefficient < 0`, `solvent_flow < 0` ou
/// `feed_flow <= 0`.
pub fn extract_factor(distribution_coefficient: f64, solvent_flow: f64, feed_flow: f64) -> f64 {
    assert!(distribution_coefficient >= 0.0, "K ≥ 0 requis");
    assert!(solvent_flow >= 0.0, "S ≥ 0 requis");
    assert!(feed_flow > 0.0, "F > 0 requis (débit d'alimentation)");
    distribution_coefficient * solvent_flow / feed_flow
}

/// Fraction de soluté restant dans le raffinat après **un étage** théorique
/// `φ₁ = 1 / (1 + E)` (sans dimension). Le rendement d'extraction vaut `1 − φ₁`.
///
/// `extraction_factor` (E) sans dimension.
///
/// Panique si `extraction_factor < 0`.
pub fn extract_single_stage_raffinate_fraction(extraction_factor: f64) -> f64 {
    assert!(extraction_factor >= 0.0, "E ≥ 0 requis");
    1.0 / (1.0 + extraction_factor)
}

/// Fraction de soluté restant dans le raffinat après `N` étages à **courants
/// croisés** (solvant frais à chaque étage, même facteur par étage)
/// `φ_c = (1 / (1 + Eₛ))^N` (sans dimension).
///
/// `extraction_factor_per_stage` (Eₛ) sans dimension ; `stages` (N) nombre
/// d'étages théoriques.
///
/// Panique si `extraction_factor_per_stage < 0` ou si `stages == 0`.
pub fn extract_crosscurrent_raffinate_fraction(
    extraction_factor_per_stage: f64,
    stages: u32,
) -> f64 {
    assert!(extraction_factor_per_stage >= 0.0, "Eₛ ≥ 0 requis");
    assert!(stages >= 1, "N ≥ 1 étage requis");
    (1.0 / (1.0 + extraction_factor_per_stage)).powi(stages as i32)
}

/// Fraction de soluté restant dans le raffinat après `N` étages à
/// **contre-courant** `φ_cc = (E − 1) / (E^(N+1) − 1)` (sans dimension),
/// valable pour `E ≠ 1`.
///
/// `extraction_factor` (E) sans dimension ; `stages` (N) nombre d'étages
/// théoriques. Pour `E > 1`, `φ_cc → 0` quand `N → ∞`.
///
/// Panique si `extraction_factor <= 0`, si `extraction_factor ≈ 1` (formule
/// singulière) ou si `stages == 0`.
pub fn extract_countercurrent_raffinate_fraction(extraction_factor: f64, stages: u32) -> f64 {
    assert!(extraction_factor > 0.0, "E > 0 requis");
    assert!(
        (extraction_factor - 1.0).abs() > 1.0e-9,
        "E ≠ 1 requis (formule à contre-courant singulière en E = 1)"
    );
    assert!(stages >= 1, "N ≥ 1 étage requis");
    (extraction_factor - 1.0) / (extraction_factor.powi(stages as i32 + 1) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn distribution_coefficient_definition_and_reciprocity() {
        // c_E = 0.6, c_R = 0.15 ⇒ K = 0.6/0.15 = 4.
        let k = extract_distribution_coefficient(0.6_f64, 0.15_f64);
        assert_relative_eq!(k, 4.0, max_relative = 1e-12);
        // Réciprocité : K·c_R doit redonner c_E.
        assert_relative_eq!(k * 0.15_f64, 0.6, max_relative = 1e-12);
    }

    #[test]
    fn factor_definition() {
        // K = 4, S = 3, F = 6 ⇒ E = 4·3/6 = 2.
        let e = extract_factor(4.0_f64, 3.0_f64, 6.0_f64);
        assert_relative_eq!(e, 2.0, max_relative = 1e-12);
        // Solvant nul ⇒ facteur nul.
        assert_relative_eq!(
            extract_factor(4.0_f64, 0.0_f64, 6.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn single_stage_realistic_and_limits() {
        // E = 2 ⇒ φ₁ = 1/(1+2) = 1/3.
        assert_relative_eq!(
            extract_single_stage_raffinate_fraction(2.0_f64),
            1.0_f64 / 3.0_f64,
            max_relative = 1e-12
        );
        // E = 0 (pas de solvant) ⇒ tout le soluté reste : φ₁ = 1.
        assert_relative_eq!(
            extract_single_stage_raffinate_fraction(0.0_f64),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn crosscurrent_reduces_to_single_stage_at_one_stage() {
        // À N = 1, la cascade à courants croisés se réduit à l'étage unique.
        let e = 2.0_f64;
        assert_relative_eq!(
            extract_crosscurrent_raffinate_fraction(e, 1),
            extract_single_stage_raffinate_fraction(e),
            max_relative = 1e-12
        );
        // E = 2, N = 3 ⇒ (1/3)^3 = 1/27.
        assert_relative_eq!(
            extract_crosscurrent_raffinate_fraction(2.0_f64, 3),
            1.0_f64 / 27.0_f64,
            max_relative = 1e-12
        );
    }

    #[test]
    fn countercurrent_reduces_to_single_stage_and_realistic_case() {
        // À N = 1 : (E−1)/(E²−1) = 1/(E+1), identique à l'étage unique.
        let e = 2.0_f64;
        assert_relative_eq!(
            extract_countercurrent_raffinate_fraction(e, 1),
            extract_single_stage_raffinate_fraction(e),
            max_relative = 1e-12
        );
        // E = 2, N = 3 ⇒ (2−1)/(2^4−1) = 1/15.
        assert_relative_eq!(
            extract_countercurrent_raffinate_fraction(2.0_f64, 3),
            1.0_f64 / 15.0_f64,
            max_relative = 1e-12
        );
    }

    #[test]
    fn countercurrent_tends_to_full_extraction() {
        // Pour E > 1, un grand N extrait la quasi-totalité du soluté (φ_cc → 0).
        let phi = extract_countercurrent_raffinate_fraction(2.0_f64, 60);
        assert!(phi > 0.0, "la fraction restante demeure > 0");
        assert_relative_eq!(phi, 0.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "E ≠ 1 requis")]
    fn countercurrent_panics_at_unit_factor() {
        // E = 1 rend la formule à contre-courant singulière (0/0) ⇒ panique.
        let _ = extract_countercurrent_raffinate_fraction(1.0_f64, 3);
    }
}
