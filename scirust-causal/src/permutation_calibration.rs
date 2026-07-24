//! A deterministic Freedman-Lane-style residual-permutation engine shared by
//! the classical and robust partial-correlation tests.
//!
//! # The scheme, precisely
//!
//! Given two **residual** vectors `x_res`, `y_res` (each variable's values
//! after its linear — classical or robust — dependence on the conditioning
//! set has been removed) and an observed statistic `T_obs` computed from
//! them, this module repeatedly:
//!
//! 1. draws a deterministic permutation `π` of `0..n` from one continuing
//!    [`scirust_stats::SplitMix64`] stream seeded once from the caller's
//!    `seed` (Durstenfeld's Fisher-Yates: for `i` from `n` down to `2`, swap
//!    position `i-1` with a uniformly drawn `j ∈ [0, i)` via
//!    `rng.next_u64() % i` — permutation `k` therefore depends on the exact
//!    sequence of draws already consumed by permutations `0..k`, so the whole
//!    sequence of `B` permutations is completely determined by `seed` and `n`);
//! 2. recomputes the statistic on `(x_res, y_res[π])` via the caller-supplied
//!    closure;
//! 3. counts an *exceedance* when `|T_perm| >= |T_obs|`.
//!
//! The two-sided p-value is `p = (1 + exceedances) / (1 + completed)` — the
//! standard finite-sample-corrected Monte Carlo convention (never exactly `0`
//! even with zero exceedances, since the observed arrangement is itself one
//! of the `1 + B` exchangeable arrangements under the null).
//!
//! # What this calibrates, and under what assumption
//!
//! Permuting a **residual** rather than the raw variable is what makes this
//! valid for a *conditional* test: if `y_res` is `Y`'s residual after
//! removing its best linear predictor from `Z`, permuting `y_res` and
//! recorrelating against `x_res` preserves the null `X ⟂ Y | Z` only under
//! [`crate::CausalAssumption::ResidualExchangeability`] — that, under this
//! null and the declared linear residualization model, `y_res`'s entries are
//! exchangeable. This is the standard Freedman-Lane assumption. It is
//! **not universally valid**: it can fail if `Y`'s true dependence on `Z` is
//! nonlinear or heteroscedastic in a way the residualization does not
//! capture. Permuting raw, un-residualized variables (naive permutation) is
//! explicitly **not** what this module does, and would be invalid whenever
//! `Y` (or `X`) actually depends on `Z`.
//!
//! For an empty conditioning set, residualization is a no-op (centering
//! only), so this reduces to the ordinary two-sample permutation test of
//! marginal correlation — exact under the weaker assumption of exchangeable
//! *raw* observations under the null.

use crate::error::CausalError;
use scirust_stats::SplitMix64;

/// One permutation's worth of book-keeping, rolled into the final outcome.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PermutationCalibrationOutcome {
    pub p_value: f64,
    pub requested_permutations: usize,
    pub completed_permutations: usize,
    pub exceedance_count: usize,
}

/// Durstenfeld's Fisher-Yates shuffle of `0..n`, consuming `n.saturating_sub(1)`
/// draws from `rng`. Deterministic given `rng`'s current state.
fn fisher_yates_permutation(n: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut i = n;
    while i > 1
    {
        i -= 1;
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Runs the deterministic permutation calibration described in the module
/// docs. `sample_count` is the length of the residual vectors being permuted.
/// `recompute_statistic(order)` must apply `order` as the new arrangement of
/// the *second* residual vector and return the recomputed statistic, or
/// `None` if that particular arrangement could not produce one (e.g. a
/// degenerate zero-variance draw) — such draws are excluded from
/// `completed_permutations` and do not count as an exceedance either way.
///
/// # Errors
///
/// [`CausalError::InvalidConfiguration`] if `permutations == 0`.
pub(crate) fn calibrate_by_permutation<F>(
    observed_statistic: f64,
    sample_count: usize,
    permutations: usize,
    seed: u64,
    mut recompute_statistic: F,
) -> Result<PermutationCalibrationOutcome, CausalError>
where
    F: FnMut(&[usize]) -> Option<f64>,
{
    if permutations == 0
    {
        return Err(CausalError::InvalidConfiguration {
            name: "permutations",
            value: 0.0,
        });
    }

    let mut rng = SplitMix64::new(seed);
    let observed_abs = observed_statistic.abs();
    let mut exceedance_count = 0usize;
    let mut completed_permutations = 0usize;

    for _ in 0..permutations
    {
        let order = fisher_yates_permutation(sample_count, &mut rng);
        if let Some(statistic) = recompute_statistic(&order)
        {
            completed_permutations += 1;
            if statistic.abs() >= observed_abs
            {
                exceedance_count += 1;
            }
        }
    }

    let p_value = (1.0 + exceedance_count as f64) / (1.0 + completed_permutations as f64);

    Ok(PermutationCalibrationOutcome {
        p_value,
        requested_permutations: permutations,
        completed_permutations,
        exceedance_count,
    })
}

/// Applies `order` (as produced by [`fisher_yates_permutation`]) to `values`,
/// returning a new vector with `result[i] = values[order[i]]`.
pub(crate) fn apply_permutation(values: &[f64], order: &[usize]) -> Vec<f64> {
    order.iter().map(|&idx| values[idx]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_yates_produces_a_bijection_of_0_n() {
        let mut rng = SplitMix64::new(42);
        let order = fisher_yates_permutation(7, &mut rng);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn same_seed_same_sequence_of_permutations() {
        let mut rng_a = SplitMix64::new(7);
        let mut rng_b = SplitMix64::new(7);
        for _ in 0..5
        {
            let a = fisher_yates_permutation(10, &mut rng_a);
            let b = fisher_yates_permutation(10, &mut rng_b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn different_seed_generally_differs() {
        let mut rng_a = SplitMix64::new(1);
        let mut rng_b = SplitMix64::new(2);
        let a = fisher_yates_permutation(20, &mut rng_a);
        let b = fisher_yates_permutation(20, &mut rng_b);
        assert_ne!(a, b);
    }

    #[test]
    fn rejects_zero_permutations() {
        assert!(matches!(
            calibrate_by_permutation(0.5, 10, 0, 1, |_| Some(0.0)),
            Err(CausalError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn exceedance_counting_and_formula() {
        // Every permutation "recomputes" to exactly the observed statistic,
        // so every one is an exceedance: p = (1+B)/(1+B) = 1.0.
        let outcome = calibrate_by_permutation(0.5, 5, 20, 99, |_| Some(0.5)).unwrap();
        assert_eq!(outcome.completed_permutations, 20);
        assert_eq!(outcome.exceedance_count, 20);
        assert!((outcome.p_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_exceedances_gives_the_finite_sample_floor_not_zero() {
        // Every permuted statistic is far smaller in magnitude than observed.
        let outcome = calibrate_by_permutation(10.0, 5, 20, 99, |_| Some(0.0)).unwrap();
        assert_eq!(outcome.exceedance_count, 0);
        // p = (1+0)/(1+20) = 1/21, never exactly zero.
        assert!((outcome.p_value - (1.0 / 21.0)).abs() < 1e-12);
    }

    #[test]
    fn skipped_permutations_do_not_count_as_completed_or_exceedances() {
        let mut call_count = 0usize;
        let outcome = calibrate_by_permutation(1.0, 5, 10, 1, |_| {
            call_count += 1;
            if call_count.is_multiple_of(2)
            {
                None
            }
            else
            {
                Some(0.0)
            }
        })
        .unwrap();
        assert_eq!(outcome.requested_permutations, 10);
        assert_eq!(outcome.completed_permutations, 5);
    }

    #[test]
    fn apply_permutation_reorders_by_the_given_indices() {
        let values = vec![10.0, 20.0, 30.0];
        let order = vec![2, 0, 1];
        assert_eq!(apply_permutation(&values, &order), vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn determinism_across_repeated_full_runs() {
        let run = || {
            calibrate_by_permutation(0.3, 8, 50, 12345, |order| {
                // A cheap, order-sensitive stand-in statistic.
                Some(order.iter().enumerate().map(|(i, &o)| (i * o) as f64).sum())
            })
            .unwrap()
        };
        assert_eq!(run(), run());
    }
}
