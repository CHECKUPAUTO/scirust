//! **Lubrification** — nombre de **Hersey**, régimes de la courbe de **Stribeck**
//! et rapport de film `λ` (épaisseur/rugosité).
//!
//! ```text
//! nombre de Hersey  H = μ·N/P                         (viscosité·vitesse/charge)
//! rapport de film   λ = h_min/√(σ₁² + σ₂²)
//! régimes (λ)       λ < 1 limite ; 1 ≤ λ < 3 mixte ; λ ≥ 3 hydrodynamique
//! ```
//!
//! `μ` viscosité dynamique (Pa·s), `N` vitesse (tr/s ou rad/s selon convention de
//! l'appelant), `P` charge unitaire (Pa), `H` nombre de Hersey (abscisse de la
//! courbe de Stribeck), `h_min` épaisseur minimale de film (m), `σᵢ` rugosités
//! **RMS** des surfaces (m), `λ` rapport de film.
//!
//! **Convention** : SI ; rugosités en écart quadratique moyen (`Rq`). **Limite
//! honnête** : le nombre de Hersey ordonne les régimes mais **ne donne pas** le
//! frottement (qui dépend de la courbe de Stribeck **mesurée**) ; les seuils de
//! `λ` (1 et 3) sont des **valeurs indicatives** usuelles, ajustables. Voir
//! [`crate::journal_bearings`] pour Petroff/Sommerfeld.

/// Régime de lubrification selon le rapport de film `λ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LubricationRegime {
    /// `λ < 1` : contact des aspérités, frottement et usure élevés.
    Boundary,
    /// `1 ≤ λ < 3` : régime mixte (film partiel).
    Mixed,
    /// `λ ≥ 3` : film complet, séparation totale des surfaces.
    Hydrodynamic,
}

/// Nombre de **Hersey** `H = μ·N/P`.
///
/// Panique si `load <= 0` ou une grandeur `< 0`.
pub fn hersey_number(dynamic_viscosity: f64, speed: f64, unit_load: f64) -> f64 {
    assert!(
        dynamic_viscosity >= 0.0 && speed >= 0.0 && unit_load > 0.0,
        "μ ≥ 0, N ≥ 0 et P > 0 requis"
    );
    dynamic_viscosity * speed / unit_load
}

/// Rapport de film `λ = h_min/√(σ₁² + σ₂²)`.
///
/// Panique si `min_film_thickness < 0` ou les deux rugosités nulles.
pub fn lambda_ratio(min_film_thickness: f64, roughness1: f64, roughness2: f64) -> f64 {
    assert!(
        min_film_thickness >= 0.0,
        "l'épaisseur de film doit être positive"
    );
    let composite = (roughness1 * roughness1 + roughness2 * roughness2).sqrt();
    assert!(
        composite > 0.0,
        "au moins une rugosité doit être strictement positive"
    );
    min_film_thickness / composite
}

/// Classe le régime de lubrification depuis `λ` (seuils 1 et 3).
///
/// Panique si `lambda < 0`.
pub fn regime_from_lambda(lambda: f64) -> LubricationRegime {
    assert!(lambda >= 0.0, "λ doit être positif");
    if lambda < 1.0
    {
        LubricationRegime::Boundary
    }
    else if lambda < 3.0
    {
        LubricationRegime::Mixed
    }
    else
    {
        LubricationRegime::Hydrodynamic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hersey_scales_with_viscosity_and_speed() {
        // H ∝ μ·N/P.
        assert_relative_eq!(
            hersey_number(0.04, 20.0, 1e6),
            0.04 * 20.0 / 1e6,
            epsilon = 1e-15
        );
        assert!(hersey_number(0.08, 20.0, 1e6) > hersey_number(0.04, 20.0, 1e6));
    }

    #[test]
    fn lambda_uses_composite_roughness() {
        // σ₁=σ₂=0,3 µm → composite = 0,3·√2 µm. h=1 µm → λ ≈ 2,36.
        let lam = lambda_ratio(1e-6, 0.3e-6, 0.3e-6);
        assert_relative_eq!(
            lam,
            1e-6 / (2.0f64 * 0.3e-6 * 0.3e-6).sqrt(),
            epsilon = 1e-9
        );
        assert!(lam > 2.3 && lam < 2.4);
    }

    #[test]
    fn regimes_by_lambda_threshold() {
        assert_eq!(regime_from_lambda(0.5), LubricationRegime::Boundary);
        assert_eq!(regime_from_lambda(2.0), LubricationRegime::Mixed);
        assert_eq!(regime_from_lambda(5.0), LubricationRegime::Hydrodynamic);
        // Aux seuils exacts.
        assert_eq!(regime_from_lambda(1.0), LubricationRegime::Mixed);
        assert_eq!(regime_from_lambda(3.0), LubricationRegime::Hydrodynamic);
    }

    #[test]
    fn thicker_film_improves_regime() {
        // Augmenter h_min fait passer du régime limite au film complet.
        let thin = regime_from_lambda(lambda_ratio(0.2e-6, 0.3e-6, 0.3e-6));
        let thick = regime_from_lambda(lambda_ratio(2.0e-6, 0.3e-6, 0.3e-6));
        assert_eq!(thin, LubricationRegime::Boundary);
        assert_eq!(thick, LubricationRegime::Hydrodynamic);
    }

    #[test]
    #[should_panic(expected = "au moins une rugosité")]
    fn zero_roughness_panics() {
        lambda_ratio(1e-6, 0.0, 0.0);
    }
}
