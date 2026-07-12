//! Fluage — paramètre de **Larson-Miller** (extrapolation de la durée de vie en
//! rupture) et vitesse de fluage secondaire (loi de **Norton**).
//!
//! ```text
//! Larson-Miller   P = T·(C + log10 t_r)          (T en K, t_r en h)
//! durée en rupture t_r = 10^{P/T − C}
//! Norton (fluage secondaire)  ε̇ = A·σ^n          (isotherme)
//! ```
//!
//! `T` température absolue (K), `t_r` temps à rupture (h), `C` constante du
//! matériau (~20 pour de nombreux aciers), `P` paramètre de Larson-Miller, `σ`
//! contrainte, `A`/`n` constantes de Norton, `ε̇` vitesse de déformation.
//!
//! **Convention** : `T` en kelvin, `t_r` en heures. **Limite honnête** :
//! corrélations **empiriques** d'extrapolation ; la dépendance en température de
//! la loi de Norton est ici supposée absorbée dans `A` (forme isotherme). `C`,
//! `A`, `n` sont des données matériau fournies par l'appelant.

/// Paramètre de Larson-Miller `P = T·(C + log10 t_r)`.
///
/// Panique si `temp_k <= 0` ou `rupture_time_h <= 0`.
pub fn larson_miller_parameter(temp_k: f64, rupture_time_h: f64, c: f64) -> f64 {
    assert!(
        temp_k > 0.0 && rupture_time_h > 0.0,
        "T > 0 et t_r > 0 requis"
    );
    temp_k * (c + rupture_time_h.log10())
}

/// Temps à rupture déduit du paramètre de Larson-Miller `t_r = 10^{P/T − C}` (h).
///
/// Panique si `temp_k <= 0`.
pub fn rupture_time_from_lmp(lmp: f64, temp_k: f64, c: f64) -> f64 {
    assert!(
        temp_k > 0.0,
        "la température (K) doit être strictement positive"
    );
    10.0_f64.powf(lmp / temp_k - c)
}

/// Vitesse de fluage secondaire (loi de Norton, isotherme) `ε̇ = A·σ^n`.
///
/// Panique si `stress < 0`.
pub fn norton_creep_rate(a: f64, stress: f64, exponent: f64) -> f64 {
    assert!(stress >= 0.0, "la contrainte doit être positive");
    a * stress.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn larson_miller_and_inverse_round_trip() {
        // T=800 K, t_r=1000 h, C=20 → P = 800·(20+3) = 18400.
        let p = larson_miller_parameter(800.0, 1000.0, 20.0);
        assert_relative_eq!(p, 800.0 * 23.0, epsilon = 1e-9);
        // L'inverse doit redonner 1000 h.
        assert_relative_eq!(
            rupture_time_from_lmp(p, 800.0, 20.0),
            1000.0,
            max_relative = 1e-9
        );
    }

    #[test]
    fn higher_lmp_means_longer_life_at_fixed_temp() {
        // À T fixe, un paramètre plus grand correspond à une durée plus longue.
        let t1 = rupture_time_from_lmp(18000.0, 800.0, 20.0);
        let t2 = rupture_time_from_lmp(19000.0, 800.0, 20.0);
        assert!(t2 > t1);
    }

    #[test]
    fn norton_rate_is_power_law_in_stress() {
        // n=5 : doubler la contrainte multiplie la vitesse par 2^5 = 32.
        let r1 = norton_creep_rate(1e-20, 100.0, 5.0);
        let r2 = norton_creep_rate(1e-20, 200.0, 5.0);
        assert_relative_eq!(r2 / r1, 32.0, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "t_r > 0")]
    fn zero_rupture_time_panics() {
        larson_miller_parameter(800.0, 0.0, 20.0);
    }
}
