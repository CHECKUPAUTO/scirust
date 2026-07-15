//! **Association de ressorts** — raideurs équivalentes d'assemblages en série
//! et en parallèle et flèche déduite de la raideur (loi de Hooke).
//!
//! ```text
//! série      1/k_eq = Σ (1/kᵢ)          (plus souple : flèche additive)
//! série (2)  k_eq   = k₁·k₂ / (k₁+k₂)
//! parallèle  k_eq   = Σ kᵢ              (plus raide : effort additif)
//! flèche     δ      = F / k
//! ```
//!
//! `kᵢ`/`k`/`k_eq` raideurs (N·m⁻¹), `F` effort (N), `δ` flèche (m). En **série**
//! l'effort est **commun** à tous les ressorts et les flèches s'ajoutent ; en
//! **parallèle** la flèche est **commune** et les efforts s'ajoutent.
//!
//! **Convention** : unités SI cohérentes (raideurs en N·m⁻¹, efforts en N,
//! flèches en m).
//!
//! **Limite honnête** : modèle de ressorts **linéaires** (loi de Hooke),
//! sollicités selon le **même axe**, en **chargement statique**. Les raideurs
//! individuelles `kᵢ` sont des **données matériau / procédé fournies par
//! l'appelant** — aucune valeur « par défaut » n'est inventée ici. Complète
//! [`crate::helical_springs`].

/// Raideur équivalente d'un assemblage **en série** `1/k_eq = Σ(1/kᵢ)`
/// (N·m⁻¹) ; le résultat est **plus souple** que le ressort le plus souple.
///
/// `rates` liste des raideurs individuelles (N·m⁻¹, chacune `> 0`).
///
/// Panique si `rates` est vide ou si une raideur est `<= 0`.
pub fn spring_series_rate(rates: &[f64]) -> f64 {
    assert!(!rates.is_empty(), "au moins un ressort requis");
    assert!(
        rates.iter().all(|&k| k > 0.0),
        "toutes les raideurs doivent être strictement positives"
    );
    let sum_inverse: f64 = rates.iter().map(|&k| 1.0 / k).sum();
    1.0 / sum_inverse
}

/// Raideur équivalente d'un assemblage **en parallèle** `k_eq = Σ kᵢ`
/// (N·m⁻¹) ; le résultat est **plus raide** que le ressort le plus raide.
///
/// `rates` liste des raideurs individuelles (N·m⁻¹, chacune `> 0`).
///
/// Panique si `rates` est vide ou si une raideur est `<= 0`.
pub fn spring_parallel_rate(rates: &[f64]) -> f64 {
    assert!(!rates.is_empty(), "au moins un ressort requis");
    assert!(
        rates.iter().all(|&k| k > 0.0),
        "toutes les raideurs doivent être strictement positives"
    );
    rates.iter().sum()
}

/// Raideur équivalente de **deux ressorts en série** `k_eq = k₁·k₂/(k₁+k₂)`
/// (N·m⁻¹).
///
/// `rate1`/`rate2` raideurs individuelles (N·m⁻¹, chacune `> 0`).
///
/// Panique si `rate1 <= 0` ou `rate2 <= 0`.
pub fn spring_series_two(rate1: f64, rate2: f64) -> f64 {
    assert!(
        rate1 > 0.0 && rate2 > 0.0,
        "les deux raideurs doivent être strictement positives"
    );
    rate1 * rate2 / (rate1 + rate2)
}

/// Flèche d'un ressort linéaire sous un effort donné `δ = F / k` (m).
///
/// `force` effort appliqué (N), `rate` raideur du ressort (N·m⁻¹, `> 0`) ;
/// renvoie la flèche en mètres.
///
/// Panique si `rate <= 0`.
pub fn spring_deflection_from_rate(force: f64, rate: f64) -> f64 {
    assert!(rate > 0.0, "la raideur doit être strictement positive");
    force / rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `spring_series_two` est le cas particulier à deux ressorts de
    /// `spring_series_rate`.
    #[test]
    fn series_two_matches_series_slice() {
        let k1 = 1_000.0;
        let k2 = 3_000.0;
        assert_relative_eq!(
            spring_series_two(k1, k2),
            spring_series_rate(&[k1, k2]),
            epsilon = 1e-9
        );
    }

    /// `n` ressorts identiques `k` : série → `k/n`, parallèle → `n·k`.
    #[test]
    fn identical_springs_scale_with_count() {
        let k = 2_000.0;
        let n = 4;
        let rates = vec![k; n];
        assert_relative_eq!(spring_series_rate(&rates), k / n as f64, epsilon = 1e-9);
        assert_relative_eq!(spring_parallel_rate(&rates), k * n as f64, epsilon = 1e-9);
    }

    /// La série est plus souple que le plus souple, le parallèle plus raide que
    /// le plus raide.
    #[test]
    fn bounds_series_and_parallel() {
        let rates = [500.0, 1_500.0, 4_000.0];
        let k_min = 500.0;
        let k_max = 4_000.0;
        assert!(spring_series_rate(&rates) < k_min);
        assert!(spring_parallel_rate(&rates) > k_max);
    }

    /// Réciprocité effort-flèche : `k · (F/k) = F`.
    #[test]
    fn deflection_reciprocity() {
        let force = 250.0;
        let rate = 800.0;
        let deflection = spring_deflection_from_rate(force, rate);
        assert_relative_eq!(rate * deflection, force, epsilon = 1e-9);
    }

    /// Cas chiffré : `k₁=1000`, `k₂=3000` en série → `750 N·m⁻¹` ; sous
    /// `F = 750 N` la flèche vaut exactement `1 m`.
    #[test]
    fn worked_example() {
        let k_eq = spring_series_two(1_000.0, 3_000.0);
        assert_relative_eq!(k_eq, 750.0, epsilon = 1e-9);
        let deflection = spring_deflection_from_rate(750.0, k_eq);
        assert_relative_eq!(deflection, 1.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "au moins un ressort requis")]
    fn empty_slice_panics() {
        spring_series_rate(&[]);
    }
}
