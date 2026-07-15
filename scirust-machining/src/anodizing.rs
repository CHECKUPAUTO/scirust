//! Anodisation de l'aluminium — croissance de la **couche d'oxyde** par la loi
//! **linéaire empirique** (loi de Faraday appliquée) et grandeurs réciproques.
//!
//! ```text
//! épaisseur d'oxyde      e = j·t·k
//! durée d'anodisation    t = e / (j·k)
//! densité de courant     j = e / (t·k)
//! facteur de croissance  k = e / (j·t)
//! couche scellée         e_s = e·f
//! ```
//!
//! `e` épaisseur d'oxyde (µm), `j` densité de courant (A/dm²), `t` durée
//! d'anodisation (min), `k` facteur de croissance (µm/(A·min/dm²)) reliant la
//! charge surfacique déposée à l'épaisseur formée, `f` fraction scellée
//! (adimensionnelle, entre 0 et 1) et `e_s` épaisseur de la couche scellée (µm).
//! La couche croît **linéairement** avec la charge surfacique `j·t` : doubler la
//! densité de courant ou la durée double l'épaisseur.
//!
//! **Convention** : unités cohérentes (µm, A/dm², min). **Limite honnête** : loi
//! **linéaire empirique** valable en régime stationnaire avant que la résistance
//! de la couche ne limite la croissance ; le facteur de croissance `k` dépend du
//! **bain, de la tension, de la température et de l'alliage** et est **fourni par
//! l'appelant** — aucune valeur « par défaut » n'est inventée. Le rendement
//! faradique, la dissolution chimique de l'oxyde et la porosité ne sont pas
//! modélisés.

/// Épaisseur d'oxyde formée `e = j·t·k` (µm).
///
/// `current_density` = `j` (A/dm²), `time` = `t` (min),
/// `growth_factor` = `k` (µm/(A·min/dm²)).
///
/// Panique si `current_density < 0`, `time < 0` ou `growth_factor < 0`.
pub fn oxide_thickness(current_density: f64, time: f64, growth_factor: f64) -> f64 {
    assert!(
        current_density >= 0.0,
        "la densité de courant doit être positive"
    );
    assert!(time >= 0.0, "la durée doit être positive");
    assert!(
        growth_factor >= 0.0,
        "le facteur de croissance doit être positif"
    );
    current_density * time * growth_factor
}

/// Durée d'anodisation pour atteindre une épaisseur `t = e / (j·k)` (min).
///
/// `thickness` = `e` (µm), `current_density` = `j` (A/dm²),
/// `growth_factor` = `k` (µm/(A·min/dm²)).
///
/// Panique si `thickness < 0`, `current_density <= 0` ou `growth_factor <= 0`.
pub fn anodizing_time_for_thickness(
    thickness: f64,
    current_density: f64,
    growth_factor: f64,
) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur doit être positive");
    assert!(
        current_density > 0.0,
        "la densité de courant doit être strictement positive"
    );
    assert!(
        growth_factor > 0.0,
        "le facteur de croissance doit être strictement positif"
    );
    thickness / (current_density * growth_factor)
}

/// Densité de courant requise pour une épaisseur en un temps donné
/// `j = e / (t·k)` (A/dm²).
///
/// `thickness` = `e` (µm), `time` = `t` (min),
/// `growth_factor` = `k` (µm/(A·min/dm²)).
///
/// Panique si `thickness < 0`, `time <= 0` ou `growth_factor <= 0`.
pub fn anodizing_current_density_for_thickness(
    thickness: f64,
    time: f64,
    growth_factor: f64,
) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur doit être positive");
    assert!(time > 0.0, "la durée doit être strictement positive");
    assert!(
        growth_factor > 0.0,
        "le facteur de croissance doit être strictement positif"
    );
    thickness / (time * growth_factor)
}

/// Facteur de croissance déduit d'un essai `k = e / (j·t)` (µm/(A·min/dm²)).
///
/// `thickness` = `e` (µm), `current_density` = `j` (A/dm²), `time` = `t` (min).
///
/// Panique si `thickness < 0`, `current_density <= 0` ou `time <= 0`.
pub fn anodizing_growth_factor(thickness: f64, current_density: f64, time: f64) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur doit être positive");
    assert!(
        current_density > 0.0,
        "la densité de courant doit être strictement positive"
    );
    assert!(time > 0.0, "la durée doit être strictement positive");
    thickness / (current_density * time)
}

/// Épaisseur de la couche scellée `e_s = e·f` (µm).
///
/// `thickness` = `e` (µm), `sealed_fraction` = `f` (fraction scellée entre 0 et 1).
///
/// Panique si `thickness < 0` ou si `sealed_fraction` n'est pas dans `[0, 1]`.
pub fn oxide_sealed_thickness(thickness: f64, sealed_fraction: f64) -> f64 {
    assert!(thickness >= 0.0, "l'épaisseur doit être positive");
    assert!(
        (0.0..=1.0).contains(&sealed_fraction),
        "la fraction scellée doit être comprise entre 0 et 1"
    );
    thickness * sealed_fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn thickness_and_time_are_reciprocal() {
        // t = e/(j·k) doit inverser exactement e = j·t·k.
        let j = 1.5_f64; // A/dm²
        let t = 40.0_f64; // min
        let k = 0.25_f64; // µm/(A·min/dm²)
        let e = oxide_thickness(j, t, k);
        assert_relative_eq!(anodizing_time_for_thickness(e, j, k), t, epsilon = 1e-12);
    }

    #[test]
    fn thickness_and_current_density_are_reciprocal() {
        // j = e/(t·k) doit inverser exactement e = j·t·k.
        let j = 2.0_f64;
        let t = 30.0_f64;
        let k = 0.3_f64;
        let e = oxide_thickness(j, t, k);
        assert_relative_eq!(
            anodizing_current_density_for_thickness(e, t, k),
            j,
            epsilon = 1e-12
        );
    }

    #[test]
    fn growth_factor_recovers_from_test_point() {
        // k = e/(j·t) doit redonner le facteur ayant servi à calculer e.
        let j = 1.2_f64;
        let t = 55.0_f64;
        let k = 0.28_f64;
        let e = oxide_thickness(j, t, k);
        assert_relative_eq!(anodizing_growth_factor(e, j, t), k, epsilon = 1e-12);
    }

    #[test]
    fn thickness_scales_linearly_with_charge() {
        // Croissance linéaire : doubler la charge surfacique j·t double l'épaisseur.
        let k = 0.25_f64;
        let e1 = oxide_thickness(1.0, 20.0, k);
        let e2 = oxide_thickness(2.0, 20.0, k);
        assert_relative_eq!(e2 / e1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_type_ii_thickness() {
        // Anodisation type II : j = 1,5 A/dm², t = 40 min, k = 0,25 µm/(A·min/dm²)
        // → e = 15 µm (couche courante ~15-25 µm).
        let e = oxide_thickness(1.5, 40.0, 0.25);
        assert_relative_eq!(e, 15.0, epsilon = 1e-9);
    }

    #[test]
    fn full_seal_returns_whole_thickness() {
        // Scellage complet (f = 1) : la couche scellée vaut toute l'épaisseur.
        assert_relative_eq!(oxide_sealed_thickness(20.0, 1.0), 20.0, epsilon = 1e-12);
        // Demi-scellage : la moitié.
        assert_relative_eq!(oxide_sealed_thickness(20.0, 0.5), 10.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "fraction scellée")]
    fn sealed_fraction_above_one_panics() {
        oxide_sealed_thickness(20.0, 1.5);
    }
}
