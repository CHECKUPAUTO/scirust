//! Fragmentation / broyage — lois énergétiques empiriques de **Bond**,
//! **Rittinger** et **Kick**, plus le rapport de réduction.
//!
//! ```text
//! Bond          E = 10·Wi·(1/√P80 − 1/√F80)        (kWh/t ; P80, F80 en µm)
//! Rittinger     E = C_R·(1/x_p − 1/x_f)            (∝ surface créée)
//! Kick          E = C_K·ln(x_f/x_p)                (∝ déformation)
//! réduction     R = x_f/x_p
//! ```
//!
//! `E` énergie spécifique de broyage, `Wi` indice de travail de Bond (kWh/t),
//! `P80`/`F80` tailles à 80 % passant du produit/de l'alimentation (µm), `C_R`
//! constante de Rittinger, `C_K` constante de Kick (mêmes unités que `E`), `x_p`/`x_f`
//! tailles caractéristiques du produit/de l'alimentation (unité de longueur cohérente,
//! p. ex. m ou mm), `R` rapport de réduction (sans dimension).
//!
//! **Convention** : Bond en unités mixtes usuelles (µm pour P80/F80, kWh/t pour l'énergie) ;
//! Rittinger/Kick en unités cohérentes choisies par l'appelant. **Limite honnête** :
//! ce sont des lois **empiriques** de fragmentation. L'indice de travail de Bond `Wi`
//! et les constantes de Rittinger/Kick proviennent d'**essais** sur le matériau et sont
//! **fournis** par l'appelant ; les tailles caractéristiques (P80/F80 = 80 % passant)
//! sont également **fournies**. Aucune constante matériau ou procédé n'est inventée ici.

/// Énergie spécifique de **Bond** `E = 10·Wi·(1/√P80 − 1/√F80)` (kWh/t).
///
/// `product_p80` et `feed_f80` en µm ; `work_index` (Wi) en kWh/t.
///
/// Panique si `work_index < 0`, `product_p80 <= 0` ou `feed_f80 <= 0`.
pub fn comminution_bond_work(work_index: f64, product_p80: f64, feed_f80: f64) -> f64 {
    assert!(
        work_index >= 0.0 && product_p80 > 0.0 && feed_f80 > 0.0,
        "Wi ≥ 0, P80 > 0 et F80 > 0 requis"
    );
    10.0 * work_index * (1.0 / product_p80.sqrt() - 1.0 / feed_f80.sqrt())
}

/// Énergie de **Rittinger** `E = C_R·(1/x_p − 1/x_f)` (∝ surface nouvelle créée).
///
/// Panique si `rittinger_constant < 0`, `product_size <= 0` ou `feed_size <= 0`.
pub fn comminution_rittinger_energy(
    rittinger_constant: f64,
    product_size: f64,
    feed_size: f64,
) -> f64 {
    assert!(
        rittinger_constant >= 0.0 && product_size > 0.0 && feed_size > 0.0,
        "C_R ≥ 0, x_p > 0 et x_f > 0 requis"
    );
    rittinger_constant * (1.0 / product_size - 1.0 / feed_size)
}

/// Énergie de **Kick** `E = C_K·ln(x_f/x_p)` (∝ déformation à volume constant).
///
/// Panique si `kick_constant < 0`, `feed_size <= 0` ou `product_size <= 0`.
pub fn comminution_kick_energy(kick_constant: f64, feed_size: f64, product_size: f64) -> f64 {
    assert!(
        kick_constant >= 0.0 && feed_size > 0.0 && product_size > 0.0,
        "C_K ≥ 0, x_f > 0 et x_p > 0 requis"
    );
    kick_constant * (feed_size / product_size).ln()
}

/// Rapport de réduction `R = x_f/x_p` (sans dimension).
///
/// Panique si `feed_size <= 0` ou `product_size <= 0`.
pub fn comminution_reduction_ratio(feed_size: f64, product_size: f64) -> f64 {
    assert!(
        feed_size > 0.0 && product_size > 0.0,
        "x_f > 0 et x_p > 0 requis"
    );
    feed_size / product_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bond_realistic_case() {
        // Wi = 15 kWh/t, F80 = 10 000 µm, P80 = 100 µm :
        // 1/√100 − 1/√10000 = 0.1 − 0.01 = 0.09 ; E = 10·15·0.09 = 13.5 kWh/t.
        assert_relative_eq!(
            comminution_bond_work(15.0_f64, 100.0_f64, 10_000.0_f64),
            13.5,
            max_relative = 1e-12
        );
    }

    #[test]
    fn bond_zero_when_no_size_change() {
        // P80 = F80 ⇒ pas de fragmentation ⇒ énergie nulle.
        assert_relative_eq!(
            comminution_bond_work(15.0_f64, 250.0_f64, 250.0_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn bond_proportional_to_work_index() {
        // E ∝ Wi : doubler Wi double l'énergie.
        let e1 = comminution_bond_work(8.0_f64, 75.0_f64, 6_000.0_f64);
        let e2 = comminution_bond_work(16.0_f64, 75.0_f64, 6_000.0_f64);
        assert_relative_eq!(e2 / e1, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn rittinger_matches_surface_formula() {
        // C_R = 2, x_p = 1e-3, x_f = 1e-2 :
        // 1/1e-3 − 1/1e-2 = 1000 − 100 = 900 ; E = 2·900 = 1800.
        assert_relative_eq!(
            comminution_rittinger_energy(2.0_f64, 1.0e-3_f64, 1.0e-2_f64),
            1800.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn kick_equals_constant_times_ln_reduction() {
        // Identité : E_Kick = C_K·ln(R) avec R = x_f/x_p.
        let (c_k, feed, product) = (5.0_f64, 10.0_f64, 1.0_f64);
        let r = comminution_reduction_ratio(feed, product);
        assert_relative_eq!(
            comminution_kick_energy(c_k, feed, product),
            c_k * r.ln(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn reduction_ratio_is_feed_over_product() {
        // x_f = 100, x_p = 5 ⇒ R = 20.
        assert_relative_eq!(
            comminution_reduction_ratio(100.0_f64, 5.0_f64),
            20.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "P80 > 0")]
    fn bond_zero_product_size_panics() {
        comminution_bond_work(15.0_f64, 0.0_f64, 10_000.0_f64);
    }
}
