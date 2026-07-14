//! Traitement thermochimique — **cémentation** : profondeur de couche cémentée
//! par la **loi en racine du temps** (Harris) et durée de cycle associée.
//!
//! ```text
//! loi de diffusion    x = k·√(D·t)
//! règle pratique      x = K·√t              (K = k·√D, à température fixée)
//! durée de cycle      t = (x/K)²
//! ```
//!
//! `x` profondeur de couche cémentée (m), `D` coefficient de diffusion du carbone
//! (m²/s), `t` durée de maintien à température (s), `k` facteur adimensionnel de
//! profil, `K` constante pratique de cémentation (m/√s) regroupant `k·√D` à une
//! température de traitement donnée. La couche croît en **racine carrée** du temps :
//! doubler la profondeur exige un temps quadruplé.
//!
//! **Convention** : SI cohérent. **Limite honnête** : loi **empirique** de Harris
//! (racine du temps), valable en régime de diffusion à température et potentiel
//! carbone constants sur un massif semi-infini. `D`, `k` et `K` dépendent du
//! matériau, de la température et de l'atmosphère et sont **fournis par
//! l'appelant** — aucune valeur « par défaut » n'est inventée. Le profil de
//! concentration détaillé (solution en fonction erreur `erf`) n'est pas traité.

/// Profondeur de couche cémentée par diffusion `x = k·√(D·t)` (m).
///
/// `factor` = facteur `k` (adimensionnel), `diffusion_coefficient` = `D` (m²/s),
/// `time` = `t` (s).
///
/// Panique si `diffusion_coefficient < 0`, `time < 0` ou `factor < 0`.
pub fn case_depth_from_diffusion(factor: f64, diffusion_coefficient: f64, time: f64) -> f64 {
    assert!(factor >= 0.0, "le facteur de profil doit être positif");
    assert!(
        diffusion_coefficient >= 0.0,
        "le coefficient de diffusion doit être positif"
    );
    assert!(time >= 0.0, "la durée doit être positive");
    factor * (diffusion_coefficient * time).sqrt()
}

/// Profondeur de couche cémentée par la règle pratique `x = K·√t` (m).
///
/// `carburizing_constant` = `K` (m/√s) à la température de traitement,
/// `time` = `t` (s).
///
/// Panique si `carburizing_constant < 0` ou `time < 0`.
pub fn case_depth_rule_of_thumb(carburizing_constant: f64, time: f64) -> f64 {
    assert!(
        carburizing_constant >= 0.0,
        "la constante de cémentation doit être positive"
    );
    assert!(time >= 0.0, "la durée doit être positive");
    carburizing_constant * time.sqrt()
}

/// Durée de maintien nécessaire pour atteindre une profondeur `t = (x/K)²` (s).
///
/// `case_depth` = `x` (m), `carburizing_constant` = `K` (m/√s).
///
/// Panique si `carburizing_constant <= 0` ou `case_depth < 0`.
pub fn carburizing_time_for_depth(case_depth: f64, carburizing_constant: f64) -> f64 {
    assert!(case_depth >= 0.0, "la profondeur doit être positive");
    assert!(
        carburizing_constant > 0.0,
        "la constante de cémentation doit être strictement positive"
    );
    let ratio = case_depth / carburizing_constant;
    ratio * ratio
}

/// Constante pratique de cémentation `K = k·√D` (m/√s) à température fixée.
///
/// `factor` = `k` (adimensionnel), `diffusion_coefficient` = `D` (m²/s).
///
/// Panique si `factor < 0` ou `diffusion_coefficient < 0`.
pub fn carburizing_constant_from_diffusion(factor: f64, diffusion_coefficient: f64) -> f64 {
    assert!(factor >= 0.0, "le facteur de profil doit être positif");
    assert!(
        diffusion_coefficient >= 0.0,
        "le coefficient de diffusion doit être positif"
    );
    factor * diffusion_coefficient.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn depth_and_time_are_reciprocal() {
        // t = (x/K)² doit inverser exactement x = K·√t.
        let k = 5.0e-6_f64; // m/√s
        let t = 3600.0_f64; // 1 h
        let x = case_depth_rule_of_thumb(k, t);
        assert_relative_eq!(carburizing_time_for_depth(x, k), t, epsilon = 1e-9);
    }

    #[test]
    fn depth_scales_with_square_root_of_time() {
        // Loi en racine : quadrupler la durée double la profondeur.
        let k = 4.0e-6_f64;
        let x1 = case_depth_rule_of_thumb(k, 900.0);
        let x2 = case_depth_rule_of_thumb(k, 3600.0);
        assert_relative_eq!(x2 / x1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn rule_of_thumb_matches_diffusion_via_constant() {
        // x = K·√t avec K = k·√D doit égaler x = k·√(D·t).
        let factor = 0.8_f64;
        let d = 1.2e-11_f64; // m²/s
        let t = 7200.0_f64;
        let k = carburizing_constant_from_diffusion(factor, d);
        assert_relative_eq!(
            case_depth_rule_of_thumb(k, t),
            case_depth_from_diffusion(factor, d, t),
            epsilon = 1e-15
        );
    }

    #[test]
    fn time_scales_with_depth_squared() {
        // t = (x/K)² : doubler la profondeur visée quadruple la durée.
        let k = 6.0e-6_f64;
        let t1 = carburizing_time_for_depth(0.5e-3, k);
        let t2 = carburizing_time_for_depth(1.0e-3, k);
        assert_relative_eq!(t2 / t1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_case_depth_after_four_hours() {
        // K = 6,67e-6 m/√s à 925 °C, 4 h → profondeur ~0,8 mm.
        let k = 6.67e-6_f64; // m/√s
        let t = 4.0 * 3600.0_f64; // 4 h en secondes
        let x = case_depth_rule_of_thumb(k, t);
        assert_relative_eq!(x, 8.004e-4, epsilon = 1e-6);
    }

    #[test]
    fn zero_time_gives_no_case() {
        // Sans maintien, aucune couche cémentée.
        assert_relative_eq!(case_depth_rule_of_thumb(5.0e-6, 0.0), 0.0, epsilon = 1e-18);
    }

    #[test]
    #[should_panic(expected = "constante de cémentation")]
    fn zero_constant_time_panics() {
        carburizing_time_for_depth(0.8e-3, 0.0);
    }
}
