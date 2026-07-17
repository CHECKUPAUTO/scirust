//! Experiment 6 — zero-divisor fibers (spec §Experiment 6).
//!
//! Over any finite domain the octonion algebra is split, so nonzero `a, b` with
//! `a ⊗ b = 0` exist. We find explicit examples, measure the collision-kernel
//! sizes of `L_a`/`R_a`, and probe whether input differences induce detectable
//! round-function bias. Zero divisors may weaken the *round function*; they do
//! NOT break the outer Feistel permutation (which stays structurally invertible).

use crate::algebra::word::Word;
use crate::algebra::{Oct, W2};
use crate::analysis::matrix_lifting::{Side, build_matrix};
use crate::analysis::util::sample_octs;

/// An explicit zero-divisor pair `a ⊗ b = 0` with `a, b ≠ 0`.
#[derive(Clone, Debug)]
pub struct ZeroDivisorPair {
    /// Left factor.
    pub a: [u64; 8],
    /// Right factor.
    pub b: [u64; 8],
    /// Norm of `a` (a zero divisor has non-unit norm).
    pub norm_a: u64,
}

/// Exhaustively find a few explicit zero-divisor pairs over the `W2` domain.
/// Returns up to `limit` pairs with the smallest `a` codes.
pub fn find_zero_divisors_w2(limit: usize) -> Vec<ZeroDivisorPair> {
    let mut out = Vec::new();
    // iterate a from small codes; for each even-norm a, search b for a⊗b==0
    'outer: for acode in 1u64..65536
    {
        let a = Oct::<W2>::from_u64s(std::array::from_fn(|i| (acode >> (2 * i)) & 3));
        if a.norm().to_u64() & 1 == 1
        {
            continue; // odd norm => unit => no zero divisor on the left
        }
        for bcode in 1u64..65536
        {
            let b = Oct::<W2>::from_u64s(std::array::from_fn(|i| (bcode >> (2 * i)) & 3));
            if a.mul(b).c.iter().all(|w| w.to_u64() == 0)
            {
                out.push(ZeroDivisorPair {
                    a: a.to_u64s(),
                    b: b.to_u64s(),
                    norm_a: a.norm().to_u64(),
                });
                if out.len() >= limit
                {
                    break 'outer;
                }
                break; // one b per a is enough for the report
            }
        }
    }
    out
}

/// Collision-kernel sizes for a set of multipliers.
#[derive(Clone, Debug)]
pub struct KernelReport {
    /// Multiplier coefficients.
    pub a: [u64; 8],
    /// Norm of `a`.
    pub norm_a: u64,
    /// Whether `N(a)` is a unit (odd).
    pub norm_is_unit: bool,
    /// `log2` kernel size of `L_a` over `Z/2^k`.
    pub left_kernel_log2: u32,
    /// `log2` kernel size of `R_a` over `Z/2^k`.
    pub right_kernel_log2: u32,
}

/// Kernel sizes of `L_a`, `R_a` for sampled multipliers (spec: "measure kernel
/// sizes of selected `L_a` and `R_a`").
pub fn kernel_sizes<W: Word>(seed: u64, count: usize) -> Vec<KernelReport> {
    sample_octs::<W>(seed, count)
        .into_iter()
        .map(|a| {
            let lk = build_matrix(a, Side::Left).kernel_log2();
            let rk = build_matrix(a, Side::Right).kernel_log2();
            KernelReport {
                a: a.to_u64s(),
                norm_a: a.norm().to_u64(),
                norm_is_unit: a.norm().is_unit(),
                left_kernel_log2: lk,
                right_kernel_log2: rk,
            }
        })
        .collect()
}

/// Differential-bias probe result.
#[derive(Clone, Debug)]
pub struct DifferentialResult {
    /// Input XOR difference tested.
    pub input_delta: [u64; 8],
    /// Most frequent output XOR difference.
    pub best_output_delta: [u64; 8],
    /// Frequency of that output difference, parts-per-million of samples.
    pub best_freq_ppm: u64,
    /// Number of samples.
    pub samples: usize,
}

/// Measure the output-difference distribution of a map `f` under a fixed input
/// XOR difference `delta` (spec: "test whether collision fibers induce
/// detectable Feistel differentials"). Reports the single most frequent output
/// difference and its frequency.
pub fn differential_bias<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W>,
    delta: Oct<W>,
    seed: u64,
    samples: usize,
) -> DifferentialResult {
    use std::collections::HashMap;
    let pts = sample_octs::<W>(seed, samples);
    let mut hist: HashMap<[u64; 8], u64> = HashMap::new();
    for x in &pts
    {
        let d = f(*x).xor(f(x.xor(delta)));
        *hist.entry(d.to_u64s()).or_insert(0) += 1;
    }
    let (best, cnt) = hist
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .unwrap_or(([0u64; 8], 0));
    let n = pts.len().max(1) as u64;
    DifferentialResult {
        input_delta: delta.to_u64s(),
        best_output_delta: best,
        best_freq_ppm: cnt.saturating_mul(1_000_000) / n,
        samples: pts.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W8;
    use crate::fixtures::{Fixture, FixtureId};
    use crate::permutation::round::f_round;

    #[test]
    fn zero_divisors_exist_and_are_valid() {
        let pairs = find_zero_divisors_w2(3);
        assert!(!pairs.is_empty(), "split algebra must have zero divisors");
        for p in &pairs
        {
            let a = Oct::<W2>::from_u64s(p.a);
            let b = Oct::<W2>::from_u64s(p.b);
            assert!(a.mul(b).c.iter().all(|w| w.to_u64() == 0));
            assert_ne!(p.a, [0; 8]);
            assert_ne!(p.b, [0; 8]);
            assert_eq!(p.norm_a & 1, 0, "zero divisor has even (non-unit) norm");
        }
    }

    #[test]
    fn even_norm_multiplier_has_nontrivial_kernel() {
        // a = e0 + e1 has norm 2 -> L_a singular -> kernel_log2 > 0
        let a = Oct::<W8>::from_u64s([1, 1, 0, 0, 0, 0, 0, 0]);
        assert!(build_matrix(a, Side::Left).kernel_log2() > 0);
    }

    #[test]
    fn differential_probe_runs() {
        let m = Fixture::new(FixtureId::PseudoRandom(4)).round_material::<W8>(0);
        let delta = Oct::<W8>::e(1);
        let r = differential_bias::<W8>(|x| f_round(x, &m), delta, 7, 4000);
        assert!(r.best_freq_ppm <= 1_000_000);
        assert_eq!(r.samples, 4000);
    }
}
