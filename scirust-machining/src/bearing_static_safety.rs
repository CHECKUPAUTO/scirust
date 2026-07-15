//! **Sécurité statique de roulement** — charge statique équivalente (ISO 76),
//! facteur de sécurité statique et charge statique admissible.
//!
//! ```text
//! charge équivalente     P0 = X0·Fr + Y0·Fa           (ISO 76)
//! facteur de sécurité    s0 = C0 / P0                  (marge statique)
//! charge admissible      P0_adm = C0 / s0_req          (charge max tolérée)
//! ```
//!
//! `Fr` charge radiale (N), `Fa` charge axiale (N), `X0` facteur radial statique
//! adimensionnel, `Y0` facteur axial statique adimensionnel, `P0` charge statique
//! équivalente (N), `C0` charge statique de base du catalogue (N), `s0` facteur de
//! sécurité statique adimensionnel, `s0_req` facteur de sécurité mini exigé
//! (adimensionnel), `P0_adm` charge statique admissible (N).
//!
//! **Convention** : SI (efforts en newtons). **Limite honnête** : les facteurs
//! `X0`/`Y0`, la charge statique de base `C0` et le facteur de sécurité mini exigé
//! `s0_req` sont des **données fournies par l'appelant** (catalogue fabricant,
//! norme ISO 76, exigence d'application) ; aucune valeur « par défaut » n'est
//! inventée. La formule `P0 = X0·Fr + Y0·Fa` est celle de la norme ; certains
//! roulements imposent de plus `P0 = max(P0, Fr)`, correction laissée à
//! l'appelant. Voir [`crate::bearings`] et [`crate::bearing_preload`].

/// Charge statique équivalente `P0 = X0·Fr + Y0·Fa` (N), selon ISO 76.
///
/// Panique si l'une des charges (`radial_load`, `axial_load`) ou l'un des facteurs
/// (`radial_factor`, `axial_factor`) est négatif.
pub fn bearing_static_equivalent_load(
    radial_load: f64,
    axial_load: f64,
    radial_factor: f64,
    axial_factor: f64,
) -> f64 {
    assert!(
        radial_load >= 0.0,
        "la charge radiale doit être positive ou nulle"
    );
    assert!(
        axial_load >= 0.0,
        "la charge axiale doit être positive ou nulle"
    );
    assert!(
        radial_factor >= 0.0,
        "le facteur radial X0 doit être positif ou nul"
    );
    assert!(
        axial_factor >= 0.0,
        "le facteur axial Y0 doit être positif ou nul"
    );
    radial_factor * radial_load + axial_factor * axial_load
}

/// Facteur de sécurité statique `s0 = C0 / P0` (adimensionnel).
///
/// Panique si `basic_static_load_rating < 0` ou `equivalent_static_load <= 0`.
pub fn bearing_static_safety_factor(
    basic_static_load_rating: f64,
    equivalent_static_load: f64,
) -> f64 {
    assert!(
        basic_static_load_rating >= 0.0,
        "la charge statique de base C0 doit être positive ou nulle"
    );
    assert!(
        equivalent_static_load > 0.0,
        "la charge statique équivalente P0 doit être strictement positive"
    );
    basic_static_load_rating / equivalent_static_load
}

/// Charge statique admissible `P0_adm = C0 / s0_req` (N).
///
/// Panique si `basic_static_load_rating < 0` ou `required_safety_factor <= 0`.
pub fn bearing_static_permissible_load(
    basic_static_load_rating: f64,
    required_safety_factor: f64,
) -> f64 {
    assert!(
        basic_static_load_rating >= 0.0,
        "la charge statique de base C0 doit être positive ou nulle"
    );
    assert!(
        required_safety_factor > 0.0,
        "le facteur de sécurité exigé s0 doit être strictement positif"
    );
    basic_static_load_rating / required_safety_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn safety_factor_and_permissible_load_are_reciprocal() {
        // P0_adm(C0, s0(C0, P0)) doit rendre P0 : C0 / (C0/P0) = P0.
        let c0 = 13_600.0;
        let p0 = 4500.0;
        let s0 = bearing_static_safety_factor(c0, p0);
        assert_relative_eq!(bearing_static_permissible_load(c0, s0), p0, epsilon = 1e-9);
    }

    #[test]
    fn equivalent_load_is_linear_in_loads() {
        // Doubler Fr et Fa double P0 (facteurs constants).
        let p1 = bearing_static_equivalent_load(5000.0, 3000.0, 0.6, 0.5);
        let p2 = bearing_static_equivalent_load(10_000.0, 6000.0, 0.6, 0.5);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn pure_radial_load_uses_only_radial_factor() {
        // Fa = 0 → P0 = X0·Fr.
        assert_relative_eq!(
            bearing_static_equivalent_load(8000.0, 0.0, 0.6, 0.5),
            0.6 * 8000.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn realistic_deep_groove_ball_bearing() {
        // Roulement à billes à gorge (ISO 76) : X0 = 0,6 ; Y0 = 0,5.
        // Fr = 5000 N, Fa = 3000 N → P0 = 0,6·5000 + 0,5·3000 = 4500 N.
        let p0 = bearing_static_equivalent_load(5000.0, 3000.0, 0.6, 0.5);
        assert_relative_eq!(p0, 4500.0, epsilon = 1e-9);
        // C0 = 13 600 N → s0 = 13600/4500 = 3,0222…
        let s0 = bearing_static_safety_factor(13_600.0, p0);
        assert_relative_eq!(s0, 13_600.0 / 4500.0, epsilon = 1e-12);
        // Exigence s0_req = 2 → charge admissible = 13600/2 = 6800 N.
        assert_relative_eq!(
            bearing_static_permissible_load(13_600.0, 2.0),
            6800.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn safety_factor_is_inversely_proportional_to_equivalent_load() {
        // Doubler P0 (même C0) divise s0 par deux.
        let s1 = bearing_static_safety_factor(20_000.0, 4000.0);
        let s2 = bearing_static_safety_factor(20_000.0, 8000.0);
        assert_relative_eq!(s1 / s2, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la charge statique équivalente P0 doit être strictement positive")]
    fn zero_equivalent_load_panics() {
        bearing_static_safety_factor(13_600.0, 0.0);
    }
}
