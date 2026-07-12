//! Fiabilité des **systèmes** — associations de composants : série, parallèle
//! (redondance active) et redondance `k`-sur-`n`.
//!
//! ```text
//! série       R = Π Ri
//! parallèle   R = 1 − Π(1 − Ri)
//! k-sur-n     R = Σ_{i=k}^{n} C(n,i)·r^i·(1 − r)^{n−i}   (composants identiques r)
//! ```
//!
//! `Ri` fiabilités des composants (dans `[0, 1]`), `r` fiabilité d'un composant
//! identique, `k` nombre minimal de composants sains, `n` nombre total. Un
//! système **série** est moins fiable que son maillon le plus faible ; un système
//! **parallèle** est plus fiable que son meilleur composant.
//!
//! **Convention** : fiabilités sans dimension dans `[0, 1]`. **Limite honnête** :
//! composants **indépendants** ; `k`-sur-`n` suppose des composants **identiques**
//! (mêmes `r`). Ne modélise ni les défaillances de cause commune, ni la
//! redondance passive (stand-by) avec commutation.

/// Fiabilité d'un système **série** `R = Π Ri`.
///
/// Panique si un `Ri` sort de `[0, 1]`.
pub fn series_reliability(component_reliabilities: &[f64]) -> f64 {
    component_reliabilities
        .iter()
        .map(|&r| {
            assert!(
                (0.0..=1.0).contains(&r),
                "chaque fiabilité doit être dans [0, 1]"
            );
            r
        })
        .product()
}

/// Fiabilité d'un système **parallèle** `R = 1 − Π(1 − Ri)`.
///
/// Panique si un `Ri` sort de `[0, 1]`.
pub fn parallel_reliability(component_reliabilities: &[f64]) -> f64 {
    let product_unreliability: f64 = component_reliabilities
        .iter()
        .map(|&r| {
            assert!(
                (0.0..=1.0).contains(&r),
                "chaque fiabilité doit être dans [0, 1]"
            );
            1.0 - r
        })
        .product();
    1.0 - product_unreliability
}

/// Fiabilité d'une redondance `k`-sur-`n` à composants identiques
/// `R = Σ_{i=k}^{n} C(n,i)·r^i·(1 − r)^{n−i}`.
///
/// Panique si `r` sort de `[0, 1]`, `n == 0` ou `k > n`.
pub fn k_out_of_n_reliability(component_reliability: f64, k: u32, n: u32) -> f64 {
    assert!(
        (0.0..=1.0).contains(&component_reliability),
        "r doit être dans [0, 1]"
    );
    assert!(n > 0 && k <= n, "0 < k ≤ n requis");
    let r = component_reliability;
    (k..=n)
        .map(|i| binomial(n, i) as f64 * r.powi(i as i32) * (1.0 - r).powi((n - i) as i32))
        .sum()
}

/// Coefficient binomial `C(n, k)` par produit itératif (sans débordement pour de
/// petits `n`).
fn binomial(n: u32, k: u32) -> u64 {
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k as u64
    {
        result = result * (n as u64 - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn series_is_less_than_weakest() {
        // Série de 0,9 et 0,8 → 0,72 < 0,8.
        let r = series_reliability(&[0.9, 0.8]);
        assert_relative_eq!(r, 0.72, epsilon = 1e-12);
        assert!(r < 0.8);
    }

    #[test]
    fn parallel_beats_best_component() {
        // Parallèle de 0,9 et 0,8 → 1 − 0,1·0,2 = 0,98 > 0,9.
        let r = parallel_reliability(&[0.9, 0.8]);
        assert_relative_eq!(r, 0.98, epsilon = 1e-12);
        assert!(r > 0.9);
    }

    #[test]
    fn k_equals_n_is_series_product() {
        // n-sur-n (tous requis) = série de composants identiques = rⁿ.
        assert_relative_eq!(
            k_out_of_n_reliability(0.9, 3, 3),
            0.9f64.powi(3),
            epsilon = 1e-12
        );
    }

    #[test]
    fn one_out_of_n_is_parallel() {
        // 1-sur-n = parallèle de n composants identiques.
        let koon = k_out_of_n_reliability(0.9, 1, 3);
        let par = parallel_reliability(&[0.9, 0.9, 0.9]);
        assert_relative_eq!(koon, par, epsilon = 1e-9);
    }

    #[test]
    fn two_out_of_three_voting() {
        // 2-sur-3 (redondance majoritaire) : R = 3r²−2r³. Pour r=0,9 → 0,972.
        let r = 0.9;
        assert_relative_eq!(
            k_out_of_n_reliability(r, 2, 3),
            3.0 * r * r - 2.0 * r * r * r,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "0 < k ≤ n")]
    fn k_greater_than_n_panics() {
        k_out_of_n_reliability(0.9, 4, 3);
    }
}
