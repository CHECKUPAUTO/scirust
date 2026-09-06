//! Statistical validation utilities for strategy research.
//!
//! The functions in this module are evidence tools, not profitability claims.
//! They make multiple testing, non-normal return moments and temporal dependence
//! explicit so a research result can be reported with the assumptions used.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapInterval {
    pub estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
    pub resamples: usize,
    pub block_len: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatisticalValidationError {
    EmptyInput,
    NonFiniteInput,
    InvalidConfidence,
    InvalidResampleCount,
    InvalidBlockLength,
    InvalidProbability,
    InvalidSharpeInputs,
    InvalidStrategyMatrix,
    InvalidCscvSlices,
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn arithmetic_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Seeded circular moving-block bootstrap confidence interval for the mean.
///
/// Resampling contiguous blocks retains short-range temporal structure that an
/// IID bootstrap would destroy. `block_len` and `seed` are part of the declared
/// experiment and must therefore be recorded by callers.
pub fn moving_block_bootstrap_mean_ci(
    values: &[f64],
    block_len: usize,
    resamples: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapInterval, StatisticalValidationError> {
    if values.is_empty()
    {
        return Err(StatisticalValidationError::EmptyInput);
    }
    if !finite(values)
    {
        return Err(StatisticalValidationError::NonFiniteInput);
    }
    if block_len == 0 || block_len > values.len()
    {
        return Err(StatisticalValidationError::InvalidBlockLength);
    }
    if resamples == 0
    {
        return Err(StatisticalValidationError::InvalidResampleCount);
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0
    {
        return Err(StatisticalValidationError::InvalidConfidence);
    }

    let n = values.len();
    let mut rng = SplitMix64::new(seed);
    let mut boot = Vec::with_capacity(resamples);
    for _ in 0..resamples
    {
        let mut total = 0.0f64;
        let mut produced = 0usize;
        while produced < n
        {
            let start = rng.index(n);
            for offset in 0..block_len
            {
                if produced == n
                {
                    break;
                }
                total += values[(start + offset) % n];
                produced += 1;
            }
        }
        boot.push(total / n as f64);
    }
    boot.sort_by(f64::total_cmp);

    let alpha = 1.0 - confidence;
    let scale = (resamples - 1) as f64;
    let lower_index = ((alpha / 2.0) * scale).floor() as usize;
    let upper_index = ((1.0 - alpha / 2.0) * scale).ceil() as usize;
    Ok(BootstrapInterval {
        estimate: arithmetic_mean(values),
        lower: boot[lower_index.min(resamples - 1)],
        upper: boot[upper_index.min(resamples - 1)],
        confidence,
        resamples,
        block_len,
    })
}

/// Holm step-down family-wise error correction.
///
/// Returns adjusted p-values in the same order as the input.  The adjusted
/// sequence is forced monotone in sorted-p order and clipped to `[0, 1]`.
pub fn holm_adjust(p_values: &[f64]) -> Result<Vec<f64>, StatisticalValidationError> {
    if p_values.is_empty()
    {
        return Err(StatisticalValidationError::EmptyInput);
    }
    if p_values
        .iter()
        .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return Err(StatisticalValidationError::InvalidProbability);
    }
    let m = p_values.len();
    let mut ordered: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut adjusted = vec![0.0; m];
    let mut previous = 0.0f64;
    for (rank, (original_index, p)) in ordered.into_iter().enumerate()
    {
        let raw = ((m - rank) as f64 * p).min(1.0);
        let monotone = raw.max(previous);
        adjusted[original_index] = monotone;
        previous = monotone;
    }
    Ok(adjusted)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReturnMoments {
    pub n: usize,
    pub mean: f64,
    pub sample_std: f64,
    pub sharpe: f64,
    pub skewness: f64,
    /// Raw kurtosis: Gaussian returns have kurtosis 3, not excess kurtosis 0.
    pub raw_kurtosis: f64,
}

pub fn return_moments(values: &[f64]) -> Result<ReturnMoments, StatisticalValidationError> {
    if values.len() < 3
    {
        return Err(StatisticalValidationError::InvalidSharpeInputs);
    }
    if !finite(values)
    {
        return Err(StatisticalValidationError::NonFiniteInput);
    }
    let n = values.len();
    let mean = arithmetic_mean(values);
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for value in values
    {
        let d = *value - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let sample_var = m2 / (n - 1) as f64;
    let sample_std = sample_var.sqrt();
    if sample_std <= f64::EPSILON
    {
        return Err(StatisticalValidationError::InvalidSharpeInputs);
    }
    let population_m2 = m2 / n as f64;
    let skewness = (m3 / n as f64) / population_m2.powf(1.5);
    let raw_kurtosis = (m4 / n as f64) / (population_m2 * population_m2);
    Ok(ReturnMoments {
        n,
        mean,
        sample_std,
        sharpe: mean / sample_std,
        skewness,
        raw_kurtosis,
    })
}

/// Standard normal CDF using a stable Abramowitz-Stegun-style approximation.
fn normal_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs() / std::f64::consts::SQRT_2;
    // Numerical Recipes approximation for erf, max absolute error ~1.2e-7.
    let t = 1.0 / (1.0 + 0.5 * z);
    let tau = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let erf = sign * (1.0 - tau);
    0.5 * (1.0 + erf)
}

/// Acklam inverse-normal approximation.
fn inverse_normal_cdf(p: f64) -> Result<f64, StatisticalValidationError> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0
    {
        return Err(StatisticalValidationError::InvalidProbability);
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW
    {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
    else if p <= P_HIGH
    {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
    else
    {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    Ok(x)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeflatedSharpeReport {
    pub observed_sharpe: f64,
    pub expected_max_sharpe_under_null: f64,
    pub probability: f64,
    pub n_observations: usize,
    pub independent_trials: usize,
    pub skewness: f64,
    pub raw_kurtosis: f64,
}

/// Deflated Sharpe Ratio probability.
///
/// `cross_trial_sharpe_sd` is the standard deviation of Sharpe estimates across
/// the effective independent trials.  Callers must not silently substitute the
/// raw number of highly correlated parameter combinations for
/// `independent_trials`.
pub fn deflated_sharpe_ratio(
    observed_sharpe: f64,
    n_observations: usize,
    skewness: f64,
    raw_kurtosis: f64,
    independent_trials: usize,
    cross_trial_sharpe_sd: f64,
) -> Result<DeflatedSharpeReport, StatisticalValidationError> {
    if !observed_sharpe.is_finite()
        || n_observations < 2
        || !skewness.is_finite()
        || !raw_kurtosis.is_finite()
        || raw_kurtosis < 1.0
        || independent_trials == 0
        || !cross_trial_sharpe_sd.is_finite()
        || cross_trial_sharpe_sd < 0.0
    {
        return Err(StatisticalValidationError::InvalidSharpeInputs);
    }

    let expected_max_sharpe_under_null = if independent_trials == 1 || cross_trial_sharpe_sd == 0.0
    {
        0.0
    }
    else
    {
        const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
        let n = independent_trials as f64;
        let z1 = inverse_normal_cdf(1.0 - 1.0 / n)?;
        let z2 = inverse_normal_cdf(1.0 - 1.0 / (n * std::f64::consts::E))?;
        cross_trial_sharpe_sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    };

    let variance_adjustment = 1.0 - skewness * observed_sharpe
        + ((raw_kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe;
    if !variance_adjustment.is_finite() || variance_adjustment <= 0.0
    {
        return Err(StatisticalValidationError::InvalidSharpeInputs);
    }
    let z = (observed_sharpe - expected_max_sharpe_under_null)
        * ((n_observations - 1) as f64).sqrt()
        / variance_adjustment.sqrt();
    Ok(DeflatedSharpeReport {
        observed_sharpe,
        expected_max_sharpe_under_null,
        probability: normal_cdf(z).clamp(0.0, 1.0),
        n_observations,
        independent_trials,
        skewness,
        raw_kurtosis,
    })
}

fn sharpe_for_indices(returns: &[f64], indices: &[usize]) -> f64 {
    if indices.len() < 2
    {
        return 0.0;
    }
    let mean = indices.iter().map(|&i| returns[i]).sum::<f64>() / indices.len() as f64;
    let mut ss = 0.0f64;
    for &i in indices
    {
        let d = returns[i] - mean;
        ss += d * d;
    }
    let std = (ss / (indices.len() - 1) as f64).sqrt();
    if std <= f64::EPSILON { 0.0 } else { mean / std }
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn recurse(
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k
        {
            output.push(current.clone());
            return;
        }
        let needed = k - current.len();
        for value in start..=n - needed
        {
            current.push(value);
            recurse(value + 1, n, k, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    recurse(0, n, k, &mut Vec::new(), &mut output);
    output
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CscvSplitResult {
    pub in_sample_slices: Vec<usize>,
    pub selected_strategy: usize,
    pub selected_in_sample_sharpe: f64,
    pub selected_out_of_sample_sharpe: f64,
    /// Rank normalized to `(0, 1)`, where values below 0.5 mean the in-sample
    /// winner lands in the lower half out of sample.
    pub normalized_oos_rank: f64,
    pub rank_logit: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PboReport {
    pub probability_backtest_overfitting: f64,
    pub splits: Vec<CscvSplitResult>,
}

/// Combinatorially Symmetric Cross-Validation estimate of PBO.
///
/// `strategy_returns[strategy][observation]` must be a rectangular matrix. The
/// observation axis is divided into an even number of contiguous slices. For
/// each combination of half the slices as in-sample, the best in-sample Sharpe
/// is located and its out-of-sample rank is recorded. PBO is the fraction whose
/// rank logit is below zero.
pub fn cscv_probability_of_backtest_overfitting(
    strategy_returns: &[Vec<f64>],
    slices: usize,
) -> Result<PboReport, StatisticalValidationError> {
    if strategy_returns.len() < 2
    {
        return Err(StatisticalValidationError::InvalidStrategyMatrix);
    }
    let observations = strategy_returns[0].len();
    if observations < 4
        || strategy_returns
            .iter()
            .any(|row| row.len() != observations || !finite(row))
    {
        return Err(StatisticalValidationError::InvalidStrategyMatrix);
    }
    if slices < 2 || !slices.is_multiple_of(2) || slices > observations
    {
        return Err(StatisticalValidationError::InvalidCscvSlices);
    }

    let mut slice_indices = vec![Vec::<usize>::new(); slices];
    for observation in 0..observations
    {
        let slice = (observation * slices / observations).min(slices - 1);
        slice_indices[slice].push(observation);
    }
    if slice_indices.iter().any(|slice| slice.is_empty())
    {
        return Err(StatisticalValidationError::InvalidCscvSlices);
    }

    let assignments = combinations(slices, slices / 2);
    let mut split_results = Vec::with_capacity(assignments.len());
    let mut overfit_count = 0usize;

    for in_sample_slices in assignments
    {
        let mut is_mask = vec![false; slices];
        for &slice in &in_sample_slices
        {
            is_mask[slice] = true;
        }
        let mut in_indices = Vec::new();
        let mut out_indices = Vec::new();
        for slice in 0..slices
        {
            if is_mask[slice]
            {
                in_indices.extend_from_slice(&slice_indices[slice]);
            }
            else
            {
                out_indices.extend_from_slice(&slice_indices[slice]);
            }
        }

        let in_sharpes: Vec<f64> = strategy_returns
            .iter()
            .map(|returns| sharpe_for_indices(returns, &in_indices))
            .collect();
        let selected_strategy = in_sharpes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(index, _)| index)
            .expect("at least two strategies");
        let out_sharpes: Vec<f64> = strategy_returns
            .iter()
            .map(|returns| sharpe_for_indices(returns, &out_indices))
            .collect();
        let selected_oos = out_sharpes[selected_strategy];
        let worse = out_sharpes
            .iter()
            .enumerate()
            .filter(|(index, score)| {
                **score < selected_oos || (**score == selected_oos && *index > selected_strategy)
            })
            .count();
        // Open-interval rank avoids infinite logits at either extreme.
        let normalized_rank = (worse + 1) as f64 / (out_sharpes.len() + 1) as f64;
        let rank_logit = (normalized_rank / (1.0 - normalized_rank)).ln();
        if rank_logit < 0.0
        {
            overfit_count += 1;
        }
        split_results.push(CscvSplitResult {
            in_sample_slices,
            selected_strategy,
            selected_in_sample_sharpe: in_sharpes[selected_strategy],
            selected_out_of_sample_sharpe: selected_oos,
            normalized_oos_rank: normalized_rank,
            rank_logit,
        });
    }

    Ok(PboReport {
        probability_backtest_overfitting: overfit_count as f64 / split_results.len() as f64,
        splits: split_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_bootstrap_is_seed_reproducible() {
        let x = [0.01, -0.02, 0.03, 0.01, 0.0, 0.02, -0.01, 0.04];
        let a = moving_block_bootstrap_mean_ci(&x, 2, 500, 0.95, 42).unwrap();
        let b = moving_block_bootstrap_mean_ci(&x, 2, 500, 0.95, 42).unwrap();
        assert_eq!(a, b);
        assert!(a.lower <= a.estimate && a.estimate <= a.upper);
    }

    #[test]
    fn holm_adjustment_is_monotone_and_bounded() {
        let adjusted = holm_adjust(&[0.01, 0.04, 0.03]).unwrap();
        assert!((adjusted[0] - 0.03).abs() < 1e-12);
        assert!(adjusted.iter().all(|p| (0.0..=1.0).contains(p)));
        assert!(adjusted[1] >= 0.04);
    }

    #[test]
    fn return_moments_use_raw_kurtosis() {
        let values = [-2.0, -1.0, 0.0, 1.0, 3.0, 4.0];
        let moments = return_moments(&values).unwrap();
        assert_eq!(moments.n, values.len());
        assert!(moments.raw_kurtosis > 1.0);
        assert!(moments.sample_std > 0.0);
    }

    #[test]
    fn more_trials_raise_the_deflated_sharpe_benchmark() {
        let one = deflated_sharpe_ratio(0.25, 252, 0.0, 3.0, 1, 0.08).unwrap();
        let many = deflated_sharpe_ratio(0.25, 252, 0.0, 3.0, 50, 0.08).unwrap();
        assert_eq!(one.expected_max_sharpe_under_null, 0.0);
        assert!(many.expected_max_sharpe_under_null > 0.0);
        assert!(many.probability < one.probability);
    }

    #[test]
    fn cscv_detects_noisy_selection_instability() {
        let strategies = vec![
            vec![0.04, 0.03, -0.04, -0.03, 0.04, 0.03, -0.04, -0.03],
            vec![-0.04, -0.03, 0.04, 0.03, -0.04, -0.03, 0.04, 0.03],
            vec![0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        ];
        let report = cscv_probability_of_backtest_overfitting(&strategies, 4).unwrap();
        assert_eq!(report.splits.len(), 6);
        assert!((0.0..=1.0).contains(&report.probability_backtest_overfitting));
        assert!(
            report
                .splits
                .iter()
                .all(|split| split.normalized_oos_rank > 0.0)
        );
    }
}
