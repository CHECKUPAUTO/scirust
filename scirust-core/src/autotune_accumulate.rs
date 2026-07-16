//! **Low-precision accumulation autotuning** — a second CANR autotuner objective
//! (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`, §8/§10)
//! driven by the same generic dev/held-out harness as
//! [`crate::transform_autotune::autotune_by`], but selecting an **accumulation
//! strategy** for `f32` reduction rather than a scalar transform.
//!
//! The choice of summation algorithm is a representation decision in the CANR
//! sense: naive `f32` accumulation stagnates and drifts on long or wide-range
//! sums, while compensated (Kahan–Babuška / Neumaier / Klein), fixed-tree
//! pairwise, and deterministic stochastic accumulation trade accuracy, cost, and
//! bias differently. Which one is best depends on the data — exactly the
//! situation the autotuner is for. This module measures each candidate's actual
//! relative error against the exact sum on a **development** batch, picks the
//! best, and validates it on a disjoint **held-out** batch against the naive
//! baseline (reporting `beats_baseline`, the CANR §13 kill signal).
//!
//! All accumulators here store their running state in `f32` (the "compute wide,
//! store narrow" setting); the exact reference is the correctly-ordered `f64`
//! expansion sum of the `f32` inputs (each `f32 → f64` is exact).

use crate::certified_numerics::sum_expansion;
use crate::stochastic_round::stochastic_sum_f32;
use crate::transform_autotune::{GenericAutotune, autotune_by};

/// An `f32` accumulation strategy (the autotuner's candidate type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccumMethod {
    /// Plain left-to-right `f32` sum — cheapest, stagnates and drifts.
    NaiveF32,
    /// Fixed-tree pairwise `f32` sum — `O(log n)` error growth, deterministic.
    PairwiseF32,
    /// Neumaier compensated `f32` sum — robust to terms larger than the sum.
    NeumaierF32,
    /// Klein second-order compensated `f32` sum.
    KleinF32,
    /// Deterministic stochastic-rounding `f32` sum (unbiased, no stagnation).
    StochasticF32 {
        /// Philox seed.
        seed: u64,
        /// Philox stream.
        stream: u32,
    },
}

impl AccumMethod {
    /// A relative cost proxy (flops/element, order of magnitude).
    pub fn cost(self) -> u32 {
        match self
        {
            AccumMethod::NaiveF32 | AccumMethod::PairwiseF32 => 1,
            AccumMethod::StochasticF32 { .. } => 3,
            AccumMethod::NeumaierF32 => 7,
            AccumMethod::KleinF32 => 14,
        }
    }
}

fn naive_f32(xs: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in xs
    {
        s += x;
    }
    s
}

fn pairwise_f32(xs: &[f32]) -> f32 {
    if xs.len() <= 8
    {
        return naive_f32(xs);
    }
    let mid = xs.len() / 2;
    pairwise_f32(&xs[..mid]) + pairwise_f32(&xs[mid..])
}

fn neumaier_f32(xs: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let mut c = 0.0f32;
    for &x in xs
    {
        let t = s + x;
        if s.abs() >= x.abs()
        {
            c += (s - t) + x;
        }
        else
        {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

fn klein_f32(xs: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let mut cs = 0.0f32;
    let mut ccs = 0.0f32;
    for &x in xs
    {
        let t = s + x;
        let c = if s.abs() >= x.abs()
        {
            (s - t) + x
        }
        else
        {
            (x - t) + s
        };
        s = t;
        let t2 = cs + c;
        let cc = if cs.abs() >= c.abs()
        {
            (cs - t2) + c
        }
        else
        {
            (c - t2) + cs
        };
        cs = t2;
        ccs += cc;
    }
    s + cs + ccs
}

/// Accumulate `xs` with the given strategy (state stored in `f32`).
pub fn accumulate(method: AccumMethod, xs: &[f32]) -> f32 {
    match method
    {
        AccumMethod::NaiveF32 => naive_f32(xs),
        AccumMethod::PairwiseF32 => pairwise_f32(xs),
        AccumMethod::NeumaierF32 => neumaier_f32(xs),
        AccumMethod::KleinF32 => klein_f32(xs),
        AccumMethod::StochasticF32 { seed, stream } => stochastic_sum_f32(xs, seed, stream),
    }
}

/// The exact sum of `f32` inputs, as `f64` (each `f32 → f64` is exact; the
/// expansion sum is order-independent and correctly rounded).
fn exact_sum(xs: &[f32]) -> f64 {
    let widened: Vec<f64> = xs.iter().map(|&x| x as f64).collect();
    sum_expansion(&widened)
}

/// Objective score of a method on `xs`: `−|relative error vs exact|` (higher is
/// better; `0` is exact). `None` if the exact sum is `0` (relative error
/// undefined).
fn accum_score(method: AccumMethod, xs: &[f32]) -> Option<f64> {
    let exact = exact_sum(xs);
    if exact == 0.0
    {
        return None;
    }
    let got = accumulate(method, xs) as f64;
    Some(-((got - exact) / exact).abs())
}

/// Autotune the `f32` accumulation strategy over `candidates`: select on `dev`
/// by measured relative accuracy, validate on the held-out `eval` batch against
/// the naive-`f32` baseline. Returns the generic report ([`GenericAutotune`]);
/// `beats_baseline` is the pre-registered kill signal (fall back to naive when
/// it is `false`).
///
/// Accumulation has no fitted parameter, so the harness's "fit" set is unused —
/// the held-out check still guards against a method that only wins on the dev
/// batch's particular values.
pub fn autotune_accumulator(
    dev: &[f32],
    eval: &[f32],
    candidates: &[AccumMethod],
) -> GenericAutotune<AccumMethod> {
    autotune_by(
        dev,
        eval,
        candidates,
        |m, _fit: &[f32], scr: &[f32]| accum_score(m, scr),
        |_fit: &[f32], scr: &[f32]| {
            accum_score(AccumMethod::NaiveF32, scr).unwrap_or(f64::NEG_INFINITY)
        },
    )
}

/// A default accumulation candidate set: naive, pairwise, Neumaier, Klein, and a
/// deterministic stochastic sum with a fixed seed.
pub fn default_accumulators() -> Vec<AccumMethod> {
    vec![
        AccumMethod::NaiveF32,
        AccumMethod::PairwiseF32,
        AccumMethod::NeumaierF32,
        AccumMethod::KleinF32,
        AccumMethod::StochasticF32 {
            seed: 0x5CA1AB1E,
            stream: 0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::philox::Philox4x32;

    /// A long, wide-range positive batch where naive f32 loses low-order terms:
    /// a large baseline plus many small increments (the stagnation regime).
    fn wide_batch(seed: u64, n: usize) -> Vec<f32> {
        let rng = Philox4x32::new(seed);
        (0..n)
            .map(|i| {
                // ~80% tiny terms near 1e-3, ~20% large terms near 1e3.
                let u = rng.f32_at(0, i as u64);
                if u < 0.8
                {
                    1e-3 * (1.0 + u)
                }
                else
                {
                    1e3 * (1.0 + u)
                }
            })
            .collect()
    }

    #[test]
    fn compensated_accumulation_beats_naive_on_wide_batches() {
        let dev = wide_batch(1, 100_000);
        let eval = wide_batch(2, 100_000);
        let report = autotune_accumulator(&dev, &eval, &default_accumulators());
        let chosen = report.chosen.expect("a method should be chosen");
        // A compensated / low-error-growth method wins over naive and generalizes.
        // (Note: the stochastic sum is *not* chosen here — for a single sum its
        // variance makes it noisier than naive; its value, unbiasedness and
        // no-stagnation, is exercised in the test below.)
        assert!(
            matches!(
                chosen,
                AccumMethod::PairwiseF32 | AccumMethod::NeumaierF32 | AccumMethod::KleinF32
            ),
            "expected a compensated/pairwise winner, got {chosen:?}"
        );
        assert!(
            report.beats_baseline,
            "chosen {:?}: eval score {:.3e} vs naive baseline {:.3e}",
            chosen, report.chosen_eval_score, report.baseline_eval_score
        );
        // The winner's held-out relative error is well below naive's (≈25× here;
        // assert a safe 5× margin).
        let naive_relerr = -report.baseline_eval_score;
        let chosen_relerr = -report.chosen_eval_score;
        assert!(
            chosen_relerr < naive_relerr / 5.0,
            "naive {naive_relerr:.2e} chosen {chosen_relerr:.2e}"
        );
    }

    #[test]
    fn every_candidate_is_scored_and_naive_scores_itself_as_baseline() {
        let dev = wide_batch(3, 20_000);
        let eval = wide_batch(4, 20_000);
        let cands = default_accumulators();
        let report = autotune_accumulator(&dev, &eval, &cands);
        assert_eq!(report.dev_scores.len(), cands.len());
        assert!(report.dev_scores.iter().all(|(_, s)| s.is_some()));
        // Naive scored on eval equals the reported baseline (self-consistency).
        let naive_eval = accum_score(AccumMethod::NaiveF32, &eval).unwrap();
        assert_eq!(report.baseline_eval_score, naive_eval);
    }

    #[test]
    fn stochastic_accumulation_is_unbiased_across_seeds() {
        // Stochastic rounding must not systematically over/under-estimate: the
        // mean of its signed error over several seeds is far smaller than a
        // single naive-f32 error on the same stagnation-prone batch.
        let xs = wide_batch(7, 100_000);
        let exact = exact_sum(&xs);
        let naive_err = (naive_f32(&xs) as f64 - exact) / exact;
        let mut mean_signed = 0.0;
        let k = 16;
        for s in 0..k
        {
            let got = stochastic_sum_f32(&xs, 1000 + s, 0) as f64;
            mean_signed += (got - exact) / exact;
        }
        mean_signed /= k as f64;
        assert!(
            mean_signed.abs() < naive_err.abs(),
            "mean signed SR error {mean_signed:.2e} not below naive bias {naive_err:.2e}"
        );
    }

    #[test]
    fn empty_candidate_set_yields_no_choice() {
        let dev = wide_batch(1, 1000);
        let eval = wide_batch(2, 1000);
        let report = autotune_accumulator(&dev, &eval, &[]);
        assert_eq!(report.chosen, None);
        assert!(!report.beats_baseline);
    }
}
