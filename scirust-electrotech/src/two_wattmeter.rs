//! **Méthode des deux wattmètres** — mesure de la puissance en régime
//! triphasé équilibré sinusoïdal à trois fils à partir des deux lectures
//! wattmétriques `W1` et `W2`.
//!
//! ```text
//! puissance active totale   P  = W1 + W2
//! puissance réactive        Q  = √3 · (W1 − W2)
//! angle du facteur de puis. φ  = atan( √3 · (W1 − W2) / (W1 + W2) )
//! facteur de puissance      pf = cos φ
//! ```
//!
//! `W1` lecture algébrique du premier wattmètre (W), `W2` lecture algébrique
//! du second wattmètre (W), `P` puissance active totale du système triphasé
//! (W), `Q` puissance réactive totale (var), `φ` angle du facteur de puissance
//! (rad), `pf` facteur de puissance `cos φ` (sans dimension). Le triangle des
//! puissances vérifie `S² = P² + Q²` avec `S = √(P² + Q²)` la puissance
//! apparente (VA).
//!
//! **Convention** : SI ; lectures wattmétriques et puissances en W/var/VA,
//! angle `φ` en **radians**. Le facteur `√3` provient du câblage des bobines
//! de tension des wattmètres sur les tensions composées. **Limite honnête** :
//! la méthode suppose un système triphasé **équilibré, sinusoïdal, à trois
//! fils** (sans neutre) ; elle ne s'applique **pas** aux systèmes déséquilibrés
//! à quatre fils. Les **deux lectures** `W1` et `W2` sont **fournies par
//! l'appelant** (relevé des deux wattmètres) ; elles sont **algébriques** —
//! l'une peut être **négative** lorsque le facteur de puissance est inférieur à
//! 0,5, ce que la méthode traduit correctement. Aucune valeur « typique » de
//! puissance n'est inventée ici.

/// Puissance active totale `P = W1 + W2` (W), somme algébrique des deux
/// lectures wattmétriques.
///
/// Panique si `wattmeter_1` ou `wattmeter_2` n'est pas fini.
pub fn wattm_total_active_power(wattmeter_1: f64, wattmeter_2: f64) -> f64 {
    assert!(
        wattmeter_1.is_finite(),
        "la lecture W1 wattmeter_1 doit être finie"
    );
    assert!(
        wattmeter_2.is_finite(),
        "la lecture W2 wattmeter_2 doit être finie"
    );
    wattmeter_1 + wattmeter_2
}

/// Puissance réactive totale `Q = √3 · (W1 − W2)` (var), obtenue à partir de la
/// différence algébrique des deux lectures wattmétriques.
///
/// Panique si `wattmeter_1` ou `wattmeter_2` n'est pas fini.
pub fn wattm_reactive_power(wattmeter_1: f64, wattmeter_2: f64) -> f64 {
    assert!(
        wattmeter_1.is_finite(),
        "la lecture W1 wattmeter_1 doit être finie"
    );
    assert!(
        wattmeter_2.is_finite(),
        "la lecture W2 wattmeter_2 doit être finie"
    );
    3.0_f64.sqrt() * (wattmeter_1 - wattmeter_2)
}

/// Angle du facteur de puissance `φ = atan( √3 · (W1 − W2) / (W1 + W2) )` (rad),
/// déphasage tension–courant déduit des deux lectures wattmétriques.
///
/// Panique si `wattmeter_1` ou `wattmeter_2` n'est pas fini, ou si la puissance
/// active totale `W1 + W2` n'est pas strictement positive (division par zéro ou
/// charge non consommatrice, angle non défini).
pub fn wattm_power_factor_angle(wattmeter_1: f64, wattmeter_2: f64) -> f64 {
    assert!(
        wattmeter_1.is_finite(),
        "la lecture W1 wattmeter_1 doit être finie"
    );
    assert!(
        wattmeter_2.is_finite(),
        "la lecture W2 wattmeter_2 doit être finie"
    );
    let total = wattmeter_1 + wattmeter_2;
    assert!(
        total > 0.0,
        "la puissance active totale W1 + W2 doit être strictement positive"
    );
    (3.0_f64.sqrt() * (wattmeter_1 - wattmeter_2) / total).atan()
}

/// Facteur de puissance `pf = cos φ` (sans dimension), cosinus de l'angle
/// [`wattm_power_factor_angle`] déduit des deux lectures wattmétriques.
///
/// Panique si `wattmeter_1` ou `wattmeter_2` n'est pas fini, ou si la puissance
/// active totale `W1 + W2` n'est pas strictement positive (angle non défini).
pub fn wattm_power_factor(wattmeter_1: f64, wattmeter_2: f64) -> f64 {
    wattm_power_factor_angle(wattmeter_1, wattmeter_2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reactive_matches_active_times_tan_phi() {
        // Identité de cohérence : puisque φ = atan(Q/P), on a tan φ = Q/P,
        // donc Q = P · tan φ. On vérifie que les trois fonctions sont
        // mutuellement cohérentes.
        let w1 = 800.0_f64;
        let w2 = 400.0_f64;
        let p = wattm_total_active_power(w1, w2);
        let q = wattm_reactive_power(w1, w2);
        let phi = wattm_power_factor_angle(w1, w2);
        assert_relative_eq!(q, p * phi.tan(), epsilon = 1e-9);
    }

    #[test]
    fn power_factor_is_active_over_apparent() {
        // Identité du triangle des puissances : pf = cos φ = P / S avec
        // S = √(P² + Q²). C'est la définition géométrique du facteur de
        // puissance retrouvée à partir des deux lectures.
        let w1 = 1500.0_f64;
        let w2 = 500.0_f64;
        let p = wattm_total_active_power(w1, w2);
        let q = wattm_reactive_power(w1, w2);
        let s = (p * p + q * q).sqrt();
        assert_relative_eq!(wattm_power_factor(w1, w2), p / s, epsilon = 1e-12);
    }

    #[test]
    fn equal_readings_give_unity_power_factor() {
        // Cas limite : lectures égales W1 = W2 ⇒ Q = 0, φ = 0 et pf = 1
        // (charge purement résistive, facteur de puissance unité).
        let w1 = 750.0_f64;
        let w2 = 750.0_f64;
        assert_relative_eq!(wattm_reactive_power(w1, w2), 0.0, epsilon = 1e-12);
        assert_relative_eq!(wattm_power_factor_angle(w1, w2), 0.0, epsilon = 1e-12);
        assert_relative_eq!(wattm_power_factor(w1, w2), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn one_wattmeter_zero_gives_half_power_factor() {
        // Cas limite classique : lorsqu'un wattmètre lit zéro (W2 = 0),
        // tan φ = √3 · W1 / W1 = √3, soit φ = π/3 = 60°, donc pf = cos 60° = 0,5.
        let w1 = 1000.0_f64;
        let w2 = 0.0_f64;
        assert_relative_eq!(
            wattm_power_factor_angle(w1, w2),
            core::f64::consts::FRAC_PI_3,
            epsilon = 1e-12
        );
        assert_relative_eq!(wattm_power_factor(w1, w2), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn scaling_both_readings_preserves_power_factor() {
        // Proportionnalité : multiplier les deux lectures par un même facteur
        // multiplie P et Q par ce facteur mais laisse φ et pf inchangés.
        let w1 = 620.0_f64;
        let w2 = 180.0_f64;
        let k = 3.0_f64;
        assert_relative_eq!(
            wattm_total_active_power(k * w1, k * w2),
            k * wattm_total_active_power(w1, w2),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            wattm_power_factor(k * w1, k * w2),
            wattm_power_factor(w1, w2),
            epsilon = 1e-12
        );
    }

    #[test]
    fn worked_case_w1_800_w2_400() {
        // Cas chiffré, W1 = 800 W, W2 = 400 W :
        //   P  = 800 + 400            = 1200 W
        //   Q  = √3 · (800 − 400)     = √3 · 400 ≈ 692,820323 var
        //   φ  = atan(√3·400 / 1200)
        //      = atan(1/√3)           = π/6 = 30°
        //   pf = cos(π/6)             = √3/2 ≈ 0,866025404
        // Recalcul indépendant : W1 − W2 = 400 ; (W1 − W2)/(W1 + W2) = 1/3 ;
        // √3 · (1/3) = 1/√3 = tan 30°, donc φ = 30° et pf = cos 30°.
        let w1 = 800.0_f64;
        let w2 = 400.0_f64;
        assert_relative_eq!(wattm_total_active_power(w1, w2), 1200.0, epsilon = 1e-9);
        assert_relative_eq!(
            wattm_reactive_power(w1, w2),
            400.0 * 3.0_f64.sqrt(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            wattm_power_factor_angle(w1, w2),
            core::f64::consts::FRAC_PI_6,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            wattm_power_factor(w1, w2),
            core::f64::consts::FRAC_PI_6.cos(),
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "strictement positive")]
    fn zero_total_active_power_panics() {
        // W1 + W2 = 0 : la puissance active totale s'annule, l'angle n'est pas
        // défini (division par zéro), la fonction doit paniquer.
        let _ = wattm_power_factor_angle(500.0, -500.0);
    }
}
