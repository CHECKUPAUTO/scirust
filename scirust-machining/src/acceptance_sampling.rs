//! Contrôle de réception par **échantillonnage** (plan simple, loi **binomiale**) :
//! probabilité d'acceptation d'un lot et **qualité moyenne après contrôle** (AOQ).
//!
//! ```text
//! coefficient binomial   C(n,k) = n! / (k!·(n−k)!)
//! prob. d'acceptation    Pa = Σ_{k=0}^{c} C(n,k)·p^k·(1−p)^(n−k)
//! qualité moyenne sortie  AOQ ≈ Pa·p           (lot supposé infini)
//! ```
//!
//! `n` (`sample_size`) taille de l'échantillon, `c` (`acceptance_number`) nombre
//! d'acceptation (nombre maximal de défectueux toléré), `p` (`defect_fraction`)
//! proportion de défectueux du lot (`0 ≤ p ≤ 1`), `Pa` probabilité d'accepter le
//! lot (sans dimension), `AOQ` proportion moyenne de défectueux acceptés
//! (sans dimension). Toutes ces grandeurs sont des fractions ou des comptes,
//! sans unité physique.
//!
//! **Limite honnête** : plan d'échantillonnage **simple** basé sur la loi
//! binomiale (lot supposé infini, tirages indépendants) ; ni double ni multiple
//! échantillonnage, ni correction hypergéométrique pour lot fini. Les paramètres
//! du plan `n` et `c` sont **fournis** par l'appelant (issus d'une table ISO 2859
//! ou d'un cahier des charges) ; ce module ne les dimensionne pas et n'invente
//! aucune valeur « par défaut ».

/// Coefficient binomial `C(n,k) = n!/(k!·(n−k)!)`, calculé de façon exacte et
/// stable par produit itératif (pas de factorielle intermédiaire).
///
/// Panique si `k > n`.
pub fn sampling_binomial_coefficient(n: u32, k: u32) -> f64 {
    assert!(
        k <= n,
        "le rang k doit vérifier k <= n pour le coefficient binomial"
    );
    // On exploite la symétrie C(n,k) = C(n,n−k) pour minimiser le nombre d'itérations.
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k
    {
        result *= f64::from(n - i);
        result /= f64::from(i + 1);
    }
    result
}

/// Probabilité qu'un tirage binomial donne exactement `k` défectueux :
/// `C(n,k)·p^k·(1−p)^(n−k)`.
///
/// Panique si `p` sort de `[0, 1]` ou si `k > n`.
pub fn sampling_binomial_pmf(sample_size: u32, k: u32, defect_fraction: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    assert!(
        k <= sample_size,
        "le nombre de défectueux k doit vérifier k <= n"
    );
    sampling_binomial_coefficient(sample_size, k)
        * defect_fraction.powi(k as i32)
        * (1.0 - defect_fraction).powi((sample_size - k) as i32)
}

/// Probabilité d'acceptation d'un lot pour un plan simple :
/// `Pa = Σ_{k=0}^{c} C(n,k)·p^k·(1−p)^(n−k)`.
///
/// Panique si `p` sort de `[0, 1]` ou si `c > n`.
pub fn probability_of_acceptance_binomial(
    sample_size: u32,
    acceptance_number: u32,
    defect_fraction: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    assert!(
        acceptance_number <= sample_size,
        "le nombre d'acceptation c doit vérifier c <= n"
    );
    (0..=acceptance_number)
        .map(|k| sampling_binomial_pmf(sample_size, k, defect_fraction))
        .sum()
}

/// Qualité moyenne après contrôle (Average Outgoing Quality) `AOQ ≈ Pa·p`,
/// approximation valable pour un lot supposé infini.
///
/// Panique si `paccept` ou `defect_fraction` sort de `[0, 1]`.
pub fn average_outgoing_quality(paccept: f64, defect_fraction: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&paccept),
        "la probabilité d'acceptation doit être dans [0, 1]"
    );
    assert!(
        (0.0..=1.0).contains(&defect_fraction),
        "la proportion de défectueux doit être dans [0, 1]"
    );
    paccept * defect_fraction
}

/// Probabilité de **rejet** (risque de refus) du lot `Pr = 1 − Pa`.
///
/// Panique si `p` sort de `[0, 1]` ou si `c > n`.
pub fn probability_of_rejection_binomial(
    sample_size: u32,
    acceptance_number: u32,
    defect_fraction: f64,
) -> f64 {
    1.0 - probability_of_acceptance_binomial(sample_size, acceptance_number, defect_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn binomial_coefficient_symmetry_and_known_values() {
        // Symétrie C(n,k) = C(n,n−k) et valeurs connues C(5,2)=10, C(6,3)=20.
        assert_relative_eq!(sampling_binomial_coefficient(5, 2), 10.0, epsilon = 1e-9);
        assert_relative_eq!(
            sampling_binomial_coefficient(5, 2),
            sampling_binomial_coefficient(5, 3),
            epsilon = 1e-9
        );
        assert_relative_eq!(sampling_binomial_coefficient(6, 3), 20.0, epsilon = 1e-9);
        assert_relative_eq!(sampling_binomial_coefficient(8, 0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pmf_sums_to_one_over_all_k() {
        // La somme de toutes les probabilités binomiales vaut 1 (loi de probabilité).
        let n = 12;
        let p = 0.17;
        let total: f64 = (0..=n).map(|k| sampling_binomial_pmf(n, k, p)).sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn acceptance_bounds_at_extreme_defect_fractions() {
        // Lot parfait (p=0) → toujours accepté ; c=n → toujours accepté quel que soit p.
        assert_relative_eq!(
            probability_of_acceptance_binomial(20, 2, 0.0),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            probability_of_acceptance_binomial(20, 20, 0.35),
            1.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn acceptance_realistic_case_matches_direct_binomial() {
        // Plan n=10, c=1, p=0.10 : Pa = (0.9)^10 + 10·0.1·(0.9)^9.
        let pa = probability_of_acceptance_binomial(10, 1, 0.10);
        let expected = 0.9_f64.powi(10) + 10.0 * 0.10 * 0.9_f64.powi(9);
        assert_relative_eq!(pa, expected, max_relative = 1e-12);
    }

    #[test]
    fn acceptance_and_rejection_are_complementary() {
        // Pa + Pr = 1 (réciprocité) pour un plan quelconque.
        let pa = probability_of_acceptance_binomial(25, 3, 0.08);
        let pr = probability_of_rejection_binomial(25, 3, 0.08);
        assert_relative_eq!(pa + pr, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn aoq_is_proportional_to_defect_fraction() {
        // AOQ = Pa·p : à Pa fixé, doubler p double l'AOQ.
        let pa = 0.6;
        assert_relative_eq!(
            average_outgoing_quality(pa, 0.04),
            2.0 * average_outgoing_quality(pa, 0.02),
            max_relative = 1e-12
        );
        assert_relative_eq!(average_outgoing_quality(0.5, 0.03), 0.015, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "proportion de défectueux")]
    fn defect_fraction_above_one_panics() {
        probability_of_acceptance_binomial(10, 1, 1.5);
    }
}
