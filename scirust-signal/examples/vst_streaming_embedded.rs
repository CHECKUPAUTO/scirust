//! Streaming variance-stabilizing denoising for the edge: a photon-counting-style
//! Poisson stream cleaned one sample at a time, with bounded memory and no
//! look-ahead — the embedded counterpart of the batch VST pipeline of
//! `scirust_signal::denoise::vst`.
//!
//! Run with `cargo run --release -p scirust-signal --example vst_streaming_embedded`.
//! Deterministic (fixed-seed LCG), so two runs print identical numbers.
//!
//! The scenario: an edge sensor delivers Poisson counts of a slowly varying
//! intensity (photon flux, particle rate, a dim optical signal). The noise variance
//! equals the local level, so a plain smoother over-trusts the loud samples. The
//! calibration step — done once, offline, on a captured record — identifies the law
//! with [`detect_noise_model`]; the device then streams through
//! [`StreamingVst`] with that fixed transform, holding only the inner filter's
//! window plus a short residual buffer in memory.

use scirust_signal::denoise::streaming::{StreamingMovingAverage, StreamingVst};
use scirust_signal::denoise::{detect_noise_model, moving_average, vst_denoise};

/// Deterministic 64-bit LCG (same constants as the crate test RNG) so the example
/// is reproducible without a `rand` dependency.
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
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Knuth's product-of-uniforms Poisson sampler (adequate for the λ ≤ 12 here).
    fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0
        {
            return 0.0;
        }
        let l = (-lambda).exp();
        let mut k: u64 = 0;
        let mut p = 1.0;
        loop
        {
            k += 1;
            p *= self.uniform();
            if p <= l
            {
                break;
            }
        }
        (k - 1) as f64
    }
}

fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
    let sig: f64 = clean.iter().map(|&x| x * x).sum();
    let err: f64 = clean
        .iter()
        .zip(est)
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (sig / err.max(1.0e-30)).log10()
}

fn main() {
    let n = 4096;
    // Slowly varying intensity λᵢ ∈ [1, ~12] — the VST's target regime.
    let clean: Vec<f64> = (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64;
            6.0 + 5.0 * phase.sin()
        })
        .collect();
    let mut rng = Lcg::new(7);
    let counts: Vec<f64> = clean.iter().map(|&l| rng.poisson(l)).collect();

    // ── Calibration (offline, once): identify the noise law on a captured record ──
    let kind = detect_noise_model(&counts);
    println!("## Calibration");
    println!("detected noise model: {kind:?}   (Anscombe ⇒ Poisson-like σ² ∝ level)");

    // ── Edge deployment: stream sample-by-sample through the fixed transform ──────
    let window = 9;
    let mut filter = StreamingVst::new(kind, StreamingMovingAverage::new(window));
    let delay = filter.delay();
    let streamed: Vec<f64> = counts.iter().map(|&x| filter.push(x)).collect();

    // A plain streaming smoother on the raw counts, for reference.
    let mut plain = StreamingMovingAverage::new(window);
    let plain_out: Vec<f64> = counts.iter().map(|&x| plain.push(x)).collect();

    // Delay-align (out[i] estimates x[i − delay]) before scoring.
    let reference = &clean[..n - delay];
    let s_raw = snr_db(reference, &counts[..n - delay]);
    let s_plain = snr_db(reference, &plain_out[delay..]);
    let s_vst = snr_db(reference, &streamed[delay..]);

    println!("\n## Streaming denoise (delay {delay} samples, window {window})");
    println!("raw Poisson counts        : {s_raw:6.2} dB");
    println!("streaming moving average  : {s_plain:6.2} dB");
    println!("streaming Anscombe ∘ MA   : {s_vst:6.2} dB");

    // ── Batch-equivalence: the pointwise VST stream reproduces the batch pipeline ──
    // For Anscombe (a pointwise-inverse kind) the streaming output equals the batch
    // vst_denoise around the batch moving average, delayed by `delay`, on the interior.
    let batch = vst_denoise(&counts, kind, |x| moving_average(x, window));
    let mut max_abs = 0.0_f64;
    let mut checked = 0usize;
    for i in (2 * delay)..n
    {
        max_abs = max_abs.max((streamed[i] - batch[i - delay]).abs());
        checked += 1;
    }
    println!("\n## Batch equivalence (pointwise kind)");
    println!(
        "max |stream[i] − batch[i−delay]| over {checked} interior samples: {max_abs:.3e}  (bit-exact ⇒ 0)"
    );

    // ── Memory footprint: bounded, independent of stream length ───────────────────
    println!("\n## Embedded footprint");
    println!(
        "state held: inner window {window} + transformed buffer {} + residual window (smearing kinds only)",
        delay + 1
    );
    println!("→ O(window + residual_window), constant in the number of samples processed.");
}
