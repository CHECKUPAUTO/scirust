//! Micro-timing harness for the non-local-means **kernel** of
//! `scirust_signal::denoise` — a measurement tool, not a pass/fail gate.
//!
//! Non-local means is the most flop-hungry denoiser in the module
//! (`O(n · search · patch)` — a few hundred flops per sample with the default
//! parameters), and its patch-distance inner loop is deliberately laid out for
//! LLVM auto-vectorization: one mirrored-extended copy of the signal makes every
//! patch a contiguous slice, and the sum of squared differences runs over
//! `chunks_exact(4)` with four independent accumulators (see `nlm1d`). This
//! harness puts a number on that kernel so layout regressions are visible.
//!
//! Protocol, chosen for stability rather than statistical ceremony:
//!
//! * **Deterministic input** — a fixed-seed LCG (copied below; the module's
//!   `testutil` helpers are `#[cfg(test)]`-only and invisible to examples)
//!   lays noise over a four-cycle sine, so every run times the same record and
//!   the printed checksum is byte-identical run to run.
//! * **Warmup 1, median of 5** — one untimed call warms caches and the branch
//!   predictor, then the median of five timed calls shrugs off scheduler noise
//!   without needing dozens of repetitions.
//! * **`black_box` on input and output** — the optimizer can neither hoist the
//!   call out of the timing loop nor delete the computation whose result is
//!   otherwise unused.
//!
//! The 2-D non-local means (`scirust_vision::denoise::nlm2d`) shares the same
//! layout optimization — a replicate-padded image makes every patch row a
//! contiguous slice fed to the same unrolled-accumulator kernel — but it is not
//! timed here: an example of `scirust-signal` cannot depend on `scirust-vision`
//! (which itself depends on this crate) without a dependency cycle.
//!
//! Timings are wall-clock and machine-dependent; only the table *format* is
//! stable. Run with:
//!
//! ```text
//! cargo run --release -p scirust-signal --example denoise_kernel_timing
//! ```

use core::f64::consts::PI;
use scirust_signal::denoise::{nlm1d, nlm1d_auto};
use std::hint::black_box;
use std::time::Instant;

/// Samples in the timed record.
const N: usize = 4096;
/// Timed repetitions per case; the reported figure is their median.
const RUNS: usize = 5;
/// Untimed warmup repetitions per case.
const WARMUP: usize = 1;

// ---------------------------------------------------------------------------
// Deterministic helpers, copied from `denoise::testutil` — that module is
// `#[cfg(test)]`-only, so an example cannot reuse it directly.
// ---------------------------------------------------------------------------

/// Deterministic 64-bit LCG (Knuth's MMIX multiplier) so the harness times the
/// exact same record on every run without a `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// The timed record: a four-cycle sine plus white Gaussian noise (σ = 0.4) —
/// the module test-suite's own self-similar fixture, at benchmark length.
fn fixture(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin() + 0.4 * rng.gauss())
        .collect()
}

/// Median of the collected per-run timings (ns). `RUNS` is a small constant,
/// so clone-and-sort is fine; ties/even lengths average the middle pair.
fn median_ns(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1
    {
        v[n / 2]
    }
    else
    {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Time one denoiser: `WARMUP` untimed calls, then the median of `RUNS` timed
/// calls. Returns `(median ns/call, checksum)` — the checksum (sum of the
/// output) both pins determinism across runs and anchors the result against
/// dead-code elimination.
fn time_case(run: impl Fn(&[f64]) -> Vec<f64>, signal: &[f64]) -> (f64, f64) {
    for _ in 0..WARMUP
    {
        black_box(run(black_box(signal)));
    }
    let mut times = Vec::with_capacity(RUNS);
    let mut checksum = 0.0;
    for _ in 0..RUNS
    {
        let start = Instant::now();
        let out = run(black_box(signal));
        times.push(start.elapsed().as_nanos() as f64);
        checksum = black_box(out).iter().sum();
    }
    (median_ns(&times), checksum)
}

fn main() {
    let signal = fixture(N, 0xD1CE_0001);

    // Each case: display name and the call under test. `nlm1d_auto` is the
    // headline number (the defaults the rest of the crate reaches for); the
    // second row scales both radii up ~50 % as a coarse cost-model probe —
    // ns/sample should grow roughly with `search · patch`.
    type Case = (&'static str, Box<dyn Fn(&[f64]) -> Vec<f64>>);
    let cases: [Case; 2] = [
        ("nlm1d_auto (patch 4, search 24)", Box::new(nlm1d_auto)),
        (
            "nlm1d (patch 6, search 36, auto h)",
            Box::new(|x: &[f64]| nlm1d(x, 6, 36, 0.0)),
        ),
    ];

    println!("# NLM kernel timing (n = {N}, warmup {WARMUP}, median of {RUNS} runs)");
    println!();
    println!(
        "| {:<34} | {:>12} | {:>10} | {:>14} |",
        "kernel", "ms/call", "ns/sample", "checksum"
    );
    println!("|{:-<36}|{:->14}|{:->12}|{:->16}|", "", "", "", "");
    for (name, run) in cases.iter()
    {
        let (ns, checksum) = time_case(run, &signal);
        println!(
            "| {:<34} | {:>12.3} | {:>10.1} | {:>14.6} |",
            name,
            ns / 1.0e6,
            ns / N as f64,
            checksum
        );
    }
    println!();
    println!(
        "Note: the 2-D kernel (scirust_vision::denoise::nlm2d) shares the same layout\n\
         optimization (replicate-padded rows + unrolled-accumulator distance) but cannot\n\
         be timed from a scirust-signal example without a dependency cycle."
    );
}
