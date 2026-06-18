//! Sequential pattern recognition: HMM, CRF, sequence labeling, pattern matching.
//!
//! All algorithms operate on `f64` probabilities. Matrices are represented as
//! flat `Vec<f64>` slices with row-major layout for cache locality.

pub mod crf;
pub mod hmm;
pub mod labeling;
pub mod matching;

pub use crf::LinearChainCRF;
pub use hmm::HMM;
pub use labeling::{bio, edit_distance, needleman_wunsch};
pub use matching::{boyer_moore, dynamic_time_warping, kmp, longest_common_subsequence};

// ─── internal helpers ───────────────────────────────────────────────────────

const NEG_INF: f64 = -1e308;

#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == NEG_INF
    {
        return b;
    }
    if b == NEG_INF
    {
        return a;
    }
    let (max_val, min_val) = if a > b { (a, b) } else { (b, a) };
    let diff = min_val - max_val;
    if diff < -30.0
    {
        max_val
    }
    else
    {
        max_val + diff.exp().ln_1p()
    }
}

#[inline]
fn log_sum_exp_slice(v: &[f64]) -> f64 {
    let max_val = v.iter().cloned().fold(NEG_INF, f64::max);
    if max_val == NEG_INF
    {
        return NEG_INF;
    }
    let sum: f64 = v.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_basic() {
        let a = 1.0f64.ln();
        let b = 2.0f64.ln();
        let result = log_sum_exp(a, b);
        let expected = 3.0f64.ln();
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn log_sum_exp_neg_inf() {
        assert_eq!(log_sum_exp(NEG_INF, 5.0), 5.0);
        assert_eq!(log_sum_exp(3.0, NEG_INF), 3.0);
        assert_eq!(log_sum_exp(NEG_INF, NEG_INF), NEG_INF);
    }

    #[test]
    fn log_sum_exp_slice_basic() {
        let vals = vec![1.0f64.ln(), 2.0f64.ln(), 3.0f64.ln()];
        let result = log_sum_exp_slice(&vals);
        let expected = 6.0f64.ln();
        assert!((result - expected).abs() < 1e-12);
    }
}
