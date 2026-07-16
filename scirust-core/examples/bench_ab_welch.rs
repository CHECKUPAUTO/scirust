//! Statistically-honest A/B micro-benchmark harness.
//!
//! Wall-clock A/B on a shared/noisy box is treacherous: run-to-run drift can
//! dwarf a real 5–10 % effect, and a single spike poisons a mean. This harness
//! addresses both with SciRust's own statistics:
//!
//!   1. **Interleave** the two variants sample-by-sample (A,B,A,B,…) so slow
//!      drift hits both equally and cancels in the difference.
//!   2. **Reject outliers** per series with Tukey fences (`q1 − 1.5·IQR`,
//!      `q3 + 1.5·IQR`) via `scirust_stats::describe::quantile`.
//!   3. **Test significance** with Welch's two-sample t-test
//!      (`scirust_stats::htest::t_test_two_sample`, unequal variances) — so a
//!      reported speed-up comes with a p-value, not just eyeballed medians.
//!
//! Demonstrated on a real, meaningful pair: the fast architecture-dependent
//! `Tensor::matmul` vs the bit-exact portable reference `Tensor::matmul_portable`.

use scirust_core::autodiff::reverse::Tensor;
use scirust_stats::describe::{median, quantile};
use scirust_stats::htest::{Tail, t_test_two_sample};
use std::time::Instant;

/// Collect `iters` interleaved timing samples (µs) for two closures.
fn interleaved<A: FnMut(), B: FnMut()>(
    iters: usize,
    warmup: usize,
    mut a: A,
    mut b: B,
) -> (Vec<f64>, Vec<f64>) {
    for _ in 0..warmup
    {
        a();
        b();
    }
    let (mut sa, mut sb) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    for _ in 0..iters
    {
        let t = Instant::now();
        a();
        sa.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        b();
        sb.push(t.elapsed().as_secs_f64() * 1e6);
    }
    (sa, sb)
}

/// Drop samples outside the Tukey fences (robust spike rejection).
fn reject_outliers(mut v: Vec<f64>) -> Vec<f64> {
    if v.len() < 4
    {
        return v;
    }
    let q1 = quantile(&v, 0.25);
    let q3 = quantile(&v, 0.75);
    let iqr = q3 - q1;
    let (lo, hi) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
    v.retain(|&x| x >= lo && x <= hi);
    v
}

/// Run the A/B, clean each series, and print the Welch verdict.
fn ab(name_a: &str, name_b: &str, iters: usize, a: impl FnMut(), b: impl FnMut()) {
    let (sa, sb) = interleaved(iters, 5, a, b);
    let (ca, cb) = (reject_outliers(sa.clone()), reject_outliers(sb.clone()));
    let (ma, mb) = (median(&ca), median(&cb));
    let delta = 100.0 * (ma - mb) / mb;
    let dropped = (sa.len() - ca.len()) + (sb.len() - cb.len());

    match t_test_two_sample(&ca, &cb, false, Tail::TwoSided)
    {
        Some(r) =>
        {
            let verdict = if r.p_value < 0.05
            {
                "SIGNIFICANT"
            }
            else
            {
                "not significant (within noise)"
            };
            println!(
                "{name_a} {ma:.2} µs  vs  {name_b} {mb:.2} µs  → {delta:+.1}%\n\
                 Welch t={:.2}  df={:.0}  p={:.4}  [{verdict}]  ({dropped} outliers dropped, n={})",
                r.statistic,
                r.df,
                r.p_value,
                ca.len().min(cb.len()),
            );
        },
        None => println!("{name_a} vs {name_b}: not enough clean samples"),
    }
}

fn main() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a = Tensor::from_vec((0..m * k).map(|i| (i as f32 * 0.001).sin()).collect(), m, k);
    let b = Tensor::from_vec((0..k * n).map(|i| (i as f32 * 0.002).cos()).collect(), k, n);

    println!("A/B: fast Tensor::matmul vs bit-exact Tensor::matmul_portable ({m}×{k}×{n})");
    ab(
        "matmul       ",
        "matmul_portable",
        200,
        || {
            std::hint::black_box(a.matmul(std::hint::black_box(&b)));
        },
        || {
            std::hint::black_box(a.matmul_portable(std::hint::black_box(&b)));
        },
    );
}
