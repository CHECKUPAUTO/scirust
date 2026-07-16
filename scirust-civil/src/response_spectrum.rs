//! Spectre de réponse élastique horizontal selon l'**Eurocode 8**
//! (EN 1998-1, §3.2.2.2) : forme spectrale à quatre branches définie par les
//! périodes de coin `TB`, `TC`, `TD`, avec la branche à accélération constante
//! (plateau), la branche à vitesse constante, la branche à déplacement
//! constant, le facteur de correction d'amortissement `η` et le passage du
//! spectre d'accélération au spectre de déplacement.
//!
//! ```text
//! plateau (TB ≤ T ≤ TC)        Se = ag·S·η·β0            (ici amplification = η·β0)
//! vitesse constante (TC ≤ T)   Se = ag·S·amp·TC/T
//! déplacement constant (T≥TD)  Se = ag·S·amp·TC·TD/T²
//! correction d'amortissement   η  = √(10/(5 + 100·ξ)) ≥ 0,55
//! déplacement spectral         Sd = Se·(T/2π)²
//! ```
//!
//! `ag` accélération de calcul du sol sur sol de type A (m/s²), `S` paramètre de
//! sol (–), `amp` amplification spectrale du plateau `η·β0` (–, avec `β0 = 2,5`
//! le facteur d'amplification et `η` la correction d'amortissement), `TB`, `TC`,
//! `TD` périodes de coin (s), `T` période propre du mode considéré (s), `ξ`
//! coefficient d'amortissement visqueux (–, fraction du critique, p. ex. `0,05`
//! pour 5 %), `η` facteur de correction d'amortissement (–), `Se` accélération
//! spectrale élastique (m/s²), `Sd` déplacement spectral élastique (m).
//!
//! **Convention** : SI strict — accélérations en m/s², périodes et temps en
//! secondes, déplacements en mètres, angles en radians (via `2π`). Le
//! coefficient d'amortissement `ξ` est une fraction (0,05 = 5 %), pas un
//! pourcentage. Types `f64`.
//!
//! **Limite honnête** : la forme spectrale suit l'Eurocode 8. L'accélération du
//! sol `ag` (aléa sismique, zonage), le paramètre de sol `S` et les périodes de
//! coin `TB`, `TC`, `TD` (classe de sol, type de spectre 1 ou 2), ainsi que le
//! coefficient d'amortissement `ξ`, sont **fournis par l'appelant** d'après
//! l'Eurocode et son Annexe Nationale — jamais inventés. Le choix de la branche
//! selon la valeur de `T` incombe à l'appelant (chaque branche a sa fonction) ;
//! aucune sélection automatique n'est faite ici. Ce spectre est **élastique** :
//! le spectre de calcul (dimensionnement) s'obtient en divisant par le
//! coefficient de comportement `q`, opération laissée à l'appelant. Ce module ne
//! fournit **aucune** valeur de zonage, de table de sol ni de coefficient `q`.

use core::f64::consts::PI;

/// Branche à accélération constante (plateau), `TB ≤ T ≤ TC` :
/// `Se = ag·S·amp` (m/s²), où `amp = η·β0` est l'amplification spectrale du
/// plateau.
///
/// Panique si `ground_acceleration < 0`, `soil_factor < 0` ou
/// `amplification < 0`.
pub fn respspec_elastic_plateau(
    ground_acceleration: f64,
    soil_factor: f64,
    amplification: f64,
) -> f64 {
    assert!(
        ground_acceleration >= 0.0,
        "l'accélération du sol ag doit être positive ou nulle"
    );
    assert!(
        soil_factor >= 0.0,
        "le paramètre de sol S doit être positif ou nul"
    );
    assert!(
        amplification >= 0.0,
        "l'amplification spectrale amp = η·β0 doit être positive ou nulle"
    );
    ground_acceleration * soil_factor * amplification
}

/// Branche à vitesse constante, `TC ≤ T ≤ TD` :
/// `Se = ag·S·amp·TC/T` (m/s²), décroissance en `1/T`.
///
/// Panique si `ground_acceleration < 0`, `soil_factor < 0`,
/// `amplification < 0`, `corner_period_c ≤ 0` ou `period ≤ 0`.
pub fn respspec_constant_velocity(
    ground_acceleration: f64,
    soil_factor: f64,
    amplification: f64,
    corner_period_c: f64,
    period: f64,
) -> f64 {
    assert!(
        ground_acceleration >= 0.0,
        "l'accélération du sol ag doit être positive ou nulle"
    );
    assert!(
        soil_factor >= 0.0,
        "le paramètre de sol S doit être positif ou nul"
    );
    assert!(
        amplification >= 0.0,
        "l'amplification spectrale amp = η·β0 doit être positive ou nulle"
    );
    assert!(
        corner_period_c > 0.0,
        "la période de coin TC doit être strictement positive"
    );
    assert!(period > 0.0, "la période T doit être strictement positive");
    ground_acceleration * soil_factor * amplification * corner_period_c / period
}

/// Branche à déplacement constant, `T ≥ TD` :
/// `Se = ag·S·amp·TC·TD/T²` (m/s²), décroissance en `1/T²`.
///
/// Panique si `ground_acceleration < 0`, `soil_factor < 0`,
/// `amplification < 0`, `corner_period_c ≤ 0`, `corner_period_d ≤ 0` ou
/// `period ≤ 0`.
pub fn respspec_constant_displacement(
    ground_acceleration: f64,
    soil_factor: f64,
    amplification: f64,
    corner_period_c: f64,
    corner_period_d: f64,
    period: f64,
) -> f64 {
    assert!(
        ground_acceleration >= 0.0,
        "l'accélération du sol ag doit être positive ou nulle"
    );
    assert!(
        soil_factor >= 0.0,
        "le paramètre de sol S doit être positif ou nul"
    );
    assert!(
        amplification >= 0.0,
        "l'amplification spectrale amp = η·β0 doit être positive ou nulle"
    );
    assert!(
        corner_period_c > 0.0,
        "la période de coin TC doit être strictement positive"
    );
    assert!(
        corner_period_d > 0.0,
        "la période de coin TD doit être strictement positive"
    );
    assert!(period > 0.0, "la période T doit être strictement positive");
    ground_acceleration * soil_factor * amplification * corner_period_c * corner_period_d
        / (period * period)
}

/// Facteur de correction d'amortissement
/// `η = √(10/(5 + 100·ξ))`, borné inférieurement à `0,55` (EN 1998-1 §3.2.2.2) ;
/// `ξ` est le coefficient d'amortissement visqueux exprimé en fraction du
/// critique (`0,05` pour 5 %). À `ξ = 0,05`, `η = 1`.
///
/// Panique si `damping_ratio < 0`.
pub fn respspec_damping_correction(damping_ratio: f64) -> f64 {
    assert!(
        damping_ratio >= 0.0,
        "le coefficient d'amortissement ξ doit être positif ou nul"
    );
    let ratio: f64 = 10.0 / (5.0 + 100.0 * damping_ratio);
    ratio.sqrt().max(0.55)
}

/// Déplacement spectral élastique `Sd = Se·(T/2π)²` (m), reliant l'accélération
/// spectrale `Se` (m/s²) au déplacement pour une période `T` (s).
///
/// Panique si `spectral_acceleration < 0` ou `period < 0`.
pub fn respspec_spectral_displacement(spectral_acceleration: f64, period: f64) -> f64 {
    assert!(
        spectral_acceleration >= 0.0,
        "l'accélération spectrale Se doit être positive ou nulle"
    );
    assert!(period >= 0.0, "la période T doit être positive ou nulle");
    spectral_acceleration * (period / (2.0 * PI)).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::SQRT_2;

    #[test]
    fn plateau_is_product_and_proportional() {
        // Se = ag·S·amp : ag = 2,0 m/s², S = 1,2, amp = 2,5 → 2,0·1,2·2,5 = 6,0.
        let se = respspec_elastic_plateau(2.0, 1.2, 2.5);
        assert_relative_eq!(se, 6.0, max_relative = 1e-12);
        // Proportionnalité : doubler ag double l'accélération spectrale.
        assert_relative_eq!(
            respspec_elastic_plateau(4.0, 1.2, 2.5),
            2.0 * se,
            max_relative = 1e-12
        );
    }

    #[test]
    fn velocity_branch_matches_plateau_at_tc() {
        // À T = TC, la branche à vitesse constante vaut le plateau (continuité) :
        // Se = ag·S·amp·TC/TC = ag·S·amp.
        let plateau = respspec_elastic_plateau(2.0, 1.2, 2.5);
        let at_tc = respspec_constant_velocity(2.0, 1.2, 2.5, 0.5, 0.5);
        assert_relative_eq!(at_tc, plateau, max_relative = 1e-12);
        // Décroissance en 1/T : à T = 1,0 s (TC = 0,5 s) → 6,0·0,5/1,0 = 3,0.
        assert_relative_eq!(
            respspec_constant_velocity(2.0, 1.2, 2.5, 0.5, 1.0),
            3.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn displacement_branch_matches_velocity_at_td() {
        // Continuité en T = TD : la branche à déplacement constant y rejoint la
        // branche à vitesse constante. Avec ag=2, S=1,2, amp=2,5, TC=0,5, TD=2,0 :
        // vitesse(TD) = 6,0·0,5/2,0 = 1,5 ; déplacement(TD) = 6,0·0,5·2,0/2,0² = 1,5.
        let velocity_at_td = respspec_constant_velocity(2.0, 1.2, 2.5, 0.5, 2.0);
        let displacement_at_td = respspec_constant_displacement(2.0, 1.2, 2.5, 0.5, 2.0, 2.0);
        assert_relative_eq!(velocity_at_td, 1.5, max_relative = 1e-12);
        assert_relative_eq!(displacement_at_td, velocity_at_td, max_relative = 1e-12);
    }

    #[test]
    fn damping_correction_unit_and_floor() {
        // À ξ = 0,05 (5 %), η = √(10/(5 + 5)) = √1 = 1.
        assert_relative_eq!(respspec_damping_correction(0.05), 1.0, max_relative = 1e-12);
        // À ξ = 0, η = √(10/5) = √2 ≈ 1,414.
        assert_relative_eq!(
            respspec_damping_correction(0.0),
            SQRT_2,
            max_relative = 1e-12
        );
        // Fort amortissement : plancher à 0,55 (√(10/(5+100·0,5)) = √(10/55) ≈ 0,426).
        assert_relative_eq!(respspec_damping_correction(0.5), 0.55, max_relative = 1e-12);
    }

    #[test]
    fn spectral_displacement_scales_with_period_squared() {
        // Sd = Se·(T/2π)². Cas chiffré : Se = 8,0 m/s², T = π s.
        // (T/2π) = π/(2π) = 0,5 ; (0,5)² = 0,25 ; Sd = 8,0·0,25 = 2,0 m.
        let sd = respspec_spectral_displacement(8.0, PI);
        assert_relative_eq!(sd, 2.0, max_relative = 1e-12);
        // Dépendance en T² : à 2T le déplacement est multiplié par 4.
        assert_relative_eq!(
            respspec_spectral_displacement(8.0, 2.0 * PI),
            4.0 * sd,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "la période T doit être strictement positive")]
    fn velocity_branch_rejects_zero_period() {
        // Division par T : une période nulle est physiquement invalide.
        let _ = respspec_constant_velocity(2.0, 1.2, 2.5, 0.5, 0.0);
    }
}
