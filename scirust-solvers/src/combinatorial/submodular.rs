//! Deterministic monotone-submodular maximization under a cardinality budget.
//!
//! This module provides the classical greedy algorithm for
//!
//! ```text
//! maximize   F(S)
//! subject to |S| <= k
//! ```
//!
//! over a finite ground set, when `F` is a **normalized, monotone,
//! submodular** set function and exact marginal gains are available.
//!
//! Under those mathematical assumptions, classical greedy achieves the
//! `(1 - 1/e)` approximation guarantee for a cardinality constraint. The
//! implementation cannot verify submodularity, monotonicity, or normalization
//! from a black-box marginal-gain callback; callers are responsible for that
//! contract. The returned certificate therefore records the theorem whose
//! assumptions the caller opted into rather than claiming that SciRust proved
//! those assumptions.
//!
//! The algorithm is deterministic: candidates are scanned in ascending input
//! index order and equal marginal gains are resolved toward the smallest index.
//! It stops early when every remaining marginal gain is zero, because adding
//! zero-gain elements cannot improve a monotone objective.

use thiserror::Error;

/// Classical approximation statement associated with the greedy algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmodularApproximationGuarantee {
    /// If the supplied objective is normalized (`F(∅)=0`), monotone, and
    /// submodular, greedy under a cardinality constraint achieves at least
    /// `(1 - 1/e)` times the optimum objective value.
    OneMinusInvEIfNormalizedMonotoneSubmodular,
}

/// Execution metadata for a greedy submodular selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmodularSelectionCertificate {
    /// Number of marginal-gain oracle calls performed.
    pub marginal_evaluations: usize,
    /// User-requested cardinality budget before clamping to the ground-set size.
    pub requested_cardinality: usize,
    /// Cardinality limit actually considered (`min(requested, ground_set_size)`).
    pub effective_cardinality: usize,
    /// The classical theorem applicable when the documented oracle contract holds.
    pub guarantee: SubmodularApproximationGuarantee,
}

/// Deterministic greedy solution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmodularSelection {
    /// Selected item indices, in greedy selection order.
    pub selected_indices: Vec<usize>,
    /// Sum of the exact marginal gains returned at selection time.
    ///
    /// For a normalized objective and exact marginals this equals `F(S)`.
    pub total_gain: u64,
    /// Algorithm metadata and conditional approximation statement.
    pub certificate: SubmodularSelectionCertificate,
}

/// Typed failures of [`greedy_monotone_submodular_cardinality`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SubmodularSelectionError {
    /// Accumulating exact marginal gains exceeded `u64`.
    #[error("submodular marginal-gain sum overflowed u64")]
    GainOverflow,
}

/// Greedily maximizes a normalized monotone-submodular objective under a
/// cardinality budget.
///
/// `marginal_gain(selected, candidate)` must return the exact non-negative
/// marginal gain
///
/// ```text
/// F(selected ∪ {candidate}) - F(selected)
/// ```
///
/// for a set function that is normalized, monotone, and submodular. The
/// callback may cache internal state through `FnMut`, but it must behave as the
/// same mathematical set function for the duration of one call to this solver.
///
/// # Determinism
///
/// The ground set is the index range `0..ground_set_size`. At each iteration,
/// every unselected candidate is evaluated in ascending order. The largest
/// marginal gain wins; exact ties keep the smaller index. Selection terminates
/// early when the best remaining marginal gain is zero.
///
/// # Guarantee
///
/// If the callback satisfies the documented normalized/monotone/submodular
/// contract, classical greedy attains the standard `(1 - 1/e)` approximation
/// guarantee under the cardinality constraint. SciRust does **not** infer or
/// verify those semantic properties from callback samples.
pub fn greedy_monotone_submodular_cardinality<F>(
    ground_set_size: usize,
    cardinality: usize,
    mut marginal_gain: F,
) -> Result<SubmodularSelection, SubmodularSelectionError>
where
    F: FnMut(&[usize], usize) -> u64,
{
    let effective_cardinality = cardinality.min(ground_set_size);
    let mut selected = Vec::with_capacity(effective_cardinality);
    let mut chosen = vec![false; ground_set_size];
    let mut total_gain = 0u64;
    let mut marginal_evaluations = 0usize;

    for _ in 0..effective_cardinality
    {
        let mut best_index: Option<usize> = None;
        let mut best_gain = 0u64;

        for candidate in 0..ground_set_size
        {
            if chosen[candidate]
            {
                continue;
            }

            let gain = marginal_gain(&selected, candidate);
            marginal_evaluations = marginal_evaluations.saturating_add(1);

            // Strictly greater preserves the first (smallest) candidate on ties
            // because candidates are scanned in ascending order.
            if best_index.is_none() || gain > best_gain
            {
                best_index = Some(candidate);
                best_gain = gain;
            }
        }

        let Some(index) = best_index
        else
        {
            break;
        };

        if best_gain == 0
        {
            break;
        }

        total_gain = total_gain
            .checked_add(best_gain)
            .ok_or(SubmodularSelectionError::GainOverflow)?;
        chosen[index] = true;
        selected.push(index);
    }

    Ok(SubmodularSelection {
        selected_indices: selected,
        total_gain,
        certificate: SubmodularSelectionCertificate {
            marginal_evaluations,
            requested_cardinality: cardinality,
            effective_cardinality,
            guarantee:
                SubmodularApproximationGuarantee::OneMinusInvEIfNormalizedMonotoneSubmodular,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_objective_selects_complementary_items() {
        // Coverage is normalized, monotone and submodular. Each item covers a
        // subset of a 4-element universe.
        let masks = [0b0011u8, 0b0110u8, 0b1100u8];
        let result =
            greedy_monotone_submodular_cardinality(masks.len(), 2, |selected, candidate| {
                let before = selected
                    .iter()
                    .fold(0u8, |mask, &index| mask | masks[index]);
                let after = before | masks[candidate];
                u64::from(after.count_ones() - before.count_ones())
            })
            .unwrap();

        // All three items initially have gain 2, so deterministic tie-breaking
        // selects index 0. Index 2 then contributes two new covered elements,
        // whereas index 1 contributes only one.
        assert_eq!(result.selected_indices, vec![0, 2]);
        assert_eq!(result.total_gain, 4);
        assert_eq!(
            result.certificate.guarantee,
            SubmodularApproximationGuarantee::OneMinusInvEIfNormalizedMonotoneSubmodular
        );
    }

    #[test]
    fn ties_resolve_to_smallest_index() {
        let result =
            greedy_monotone_submodular_cardinality(4, 2, |_selected, _candidate| 1).unwrap();
        assert_eq!(result.selected_indices, vec![0, 1]);
        assert_eq!(result.total_gain, 2);
    }

    #[test]
    fn zero_gain_stops_early() {
        let result =
            greedy_monotone_submodular_cardinality(5, 5, |_selected, _candidate| 0).unwrap();
        assert!(result.selected_indices.is_empty());
        assert_eq!(result.total_gain, 0);
        assert_eq!(result.certificate.marginal_evaluations, 5);
    }

    #[test]
    fn cardinality_is_clamped_to_ground_set() {
        let result =
            greedy_monotone_submodular_cardinality(2, 9, |_selected, _candidate| 1).unwrap();
        assert_eq!(result.selected_indices, vec![0, 1]);
        assert_eq!(result.certificate.requested_cardinality, 9);
        assert_eq!(result.certificate.effective_cardinality, 2);
    }

    #[test]
    fn empty_ground_set_is_valid() {
        let result =
            greedy_monotone_submodular_cardinality(0, 3, |_selected, _candidate| 1).unwrap();
        assert!(result.selected_indices.is_empty());
        assert_eq!(result.total_gain, 0);
        assert_eq!(result.certificate.marginal_evaluations, 0);
    }
}
