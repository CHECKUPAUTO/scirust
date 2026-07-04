//! Sensitivity / contribution analysis of a tolerance chain.
//!
//! Once an assembly inertia is known, the next engineering question is *which
//! component drives it*. For the statistical combination
//! `I_Y² = Σ αᵢ² Iᵢ²` each term is an additive share of the assembly variance,
//! so component `i` contributes
//!
//! ```text
//! cᵢ = αᵢ² Iᵢ² / I_Y²   ∈ [0, 1] ,   Σ cᵢ = 1 .
//! ```
//!
//! Ranking the `cᵢ` points straight at the few characteristics worth tightening
//! (a large `cᵢ` means re-tolerancing that part moves the assembly most) and the
//! many that are already negligible. [`contributions`] returns them sorted,
//! largest first.
//!
//! With correlated components the shares generalise to the row sums of the
//! quadratic form (`Σⱼ αᵢ αⱼ ρᵢⱼ Iᵢ Iⱼ / I_Y²`), still summing to 1
//! ([`correlated_contributions`]); this reduces to the independent case for the
//! identity correlation and lets a common-tool correlation reveal that two
//! parts share, rather than add, their influence.

use crate::chain::Contributor;
use serde::{Deserialize, Serialize};

/// One component's share of the assembly inertia.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contribution {
    /// Component name.
    pub name: String,
    /// Share of the assembly variance `αᵢ² Iᵢ² / I_Y²`, in `[0, 1]`.
    pub fraction: f64,
    /// Signed inertia contribution `αᵢ Iᵢ` (its term in the worst-case sum).
    pub inertia_contribution: f64,
}

/// Per-component contributions to the **statistical** assembly inertia,
/// `cᵢ = αᵢ² Iᵢ² / Σ αⱼ² Iⱼ²`, sorted largest-share first. The fractions sum to
/// 1 (all zero for a null assembly inertia). A component with `αᵢ = 0` or
/// `Iᵢ = 0` contributes 0.
pub fn contributions(contributors: &[Contributor]) -> Vec<Contribution> {
    let total: f64 = contributors
        .iter()
        .map(|c| c.coeff * c.coeff * c.inertia * c.inertia)
        .sum();
    let mut out: Vec<Contribution> = contributors
        .iter()
        .map(|c| {
            let term = c.coeff * c.coeff * c.inertia * c.inertia;
            Contribution {
                name: c.name.clone(),
                fraction: if total > 0.0 { term / total } else { 0.0 },
                inertia_contribution: c.coeff * c.inertia,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.fraction
            .partial_cmp(&a.fraction)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Per-component contribution fractions to the **correlated** assembly inertia,
/// in the original component order: `cᵢ = (Σⱼ αᵢ αⱼ ρᵢⱼ Iᵢ Iⱼ) / I_Y²`. `corr`
/// is the row-major `n × n` correlation matrix. The fractions sum to 1 (all
/// zero for a null assembly inertia); reduces to [`contributions`]' fractions
/// for the identity correlation. Returns an empty vector on shape mismatch.
pub fn correlated_contributions(coeffs: &[f64], inertias: &[f64], corr: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    if inertias.len() != n || corr.len() != n * n
    {
        return Vec::new();
    }
    let scaled: Vec<f64> = coeffs.iter().zip(inertias).map(|(a, i)| a * i).collect();
    let row_sums: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| scaled[i] * scaled[j] * corr[i * n + j])
                .sum::<f64>()
        })
        .collect();
    let total: f64 = row_sums.iter().sum();
    if total <= 0.0
    {
        return vec![0.0; n];
    }
    row_sums.iter().map(|r| r / total).collect()
}

/// The single largest-contributing component (by variance share), or `None` for
/// an empty chain. Ties resolve to the first such component.
pub fn dominant(contributors: &[Contributor]) -> Option<&Contributor> {
    contributors.iter().max_by(|a, b| {
        let ka = a.coeff * a.coeff * a.inertia * a.inertia;
        let kb = b.coeff * b.coeff * b.inertia * b.inertia;
        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlated::uniform_correlation;
    use approx::assert_relative_eq;

    #[test]
    fn contributions_sum_to_one_and_rank() {
        let cs = [
            Contributor::new("A", 1.0, 0.10),
            Contributor::new("B", 2.0, 0.05),
            Contributor::new("C", 1.0, 0.02),
        ];
        let c = contributions(&cs);
        let sum: f64 = c.iter().map(|x| x.fraction).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        // A: 1·0.01 = 0.01 ; B: 4·0.0025 = 0.01 ; C: 1·0.0004 = 0.0004.
        // A and B tie at the top, C is smallest.
        assert_eq!(c.last().unwrap().name, "C");
        assert_relative_eq!(c.last().unwrap().fraction, 0.0004 / 0.0204, epsilon = 1e-12);
    }

    #[test]
    fn zero_coefficient_contributes_nothing() {
        let cs = [
            Contributor::new("A", 1.0, 0.10),
            Contributor::new("Z", 0.0, 0.99),
        ];
        let c = contributions(&cs);
        let z = c.iter().find(|x| x.name == "Z").unwrap();
        assert_eq!(z.fraction, 0.0);
    }

    #[test]
    fn correlated_reduces_to_independent_for_identity() {
        let coeffs = [1.0, 2.0, 1.0];
        let inertias = [0.10, 0.05, 0.02];
        let corr = uniform_correlation(3, 0.0);
        let frac = correlated_contributions(&coeffs, &inertias, &corr);
        // Compare to the statistical fractions (original order).
        let cs: Vec<Contributor> = coeffs
            .iter()
            .zip(&inertias)
            .map(|(a, i)| Contributor::new("x", *a, *i))
            .collect();
        let total: f64 = cs.iter().map(|c| c.coeff.powi(2) * c.inertia.powi(2)).sum();
        for (k, c) in cs.iter().enumerate()
        {
            let want = c.coeff.powi(2) * c.inertia.powi(2) / total;
            assert_relative_eq!(frac[k], want, epsilon = 1e-12);
        }
        assert_relative_eq!(frac.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dominant_is_the_largest_share() {
        let cs = [
            Contributor::new("A", 1.0, 0.10),
            Contributor::new("B", 3.0, 0.09),
            Contributor::new("C", 1.0, 0.02),
        ];
        assert_eq!(dominant(&cs).unwrap().name, "B");
        assert!(dominant(&[]).is_none());
    }
}
